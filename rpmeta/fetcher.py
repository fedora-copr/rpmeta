import gzip
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import koji
import requests
import tqdm
from fedora_distro_aliases import get_distro_aliases

from rpmeta.constants import KOJI_HUB_URL
from rpmeta.dataset import HwInfo, Record

logger = logging.getLogger(__name__)


def _get_distro_aliases_retry(retries=5, delay=20) -> dict:
    """
    Retries a few times if the Bodhi API is unavailable before failing.
    """
    for attempt in range(1, retries + 1):
        try:
            return get_distro_aliases()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
                continue

    logging.error("Failed to fetch Fedora Rawhide number after multiple attempts.")
    raise


class Fetcher(ABC):
    def __init__(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date

    @abstractmethod
    def fetch_data(self) -> list[Record]:
        """
        Fetches data from the source and returns a list of records.

        Returns:
            A list of records.
        """
        ...


class KojiFetcher(Fetcher):
    def __init__(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10000,
    ) -> None:
        super().__init__(start_date, end_date)

        self._limit = limit

        logger.info(f"Initializing KojiFetcher instance: {KOJI_HUB_URL}")
        self._koji_session = koji.ClientSession(KOJI_HUB_URL)

        self._host_hw_info_map: dict[int, HwInfo] = {}
        self._current_page = 0
        # keep it here so it fails right away if bodhi API is not available at the moment
        self._fedora_rawhide_number = max(
            int(alias.version_number) for alias in _get_distro_aliases_retry()["fedora-all"]
        )

    def _fetch_hw_info_from_koji(self, task_info: dict) -> Optional[HwInfo]:
        task_id = task_info["id"]
        logger.info(f"Fetching hw_info for task: {task_id}")
        if task_info.get("host_id") and task_info["host_id"] in self._host_hw_info_map:
            return self._host_hw_info_map[task_info["host_id"]]

        try:
            lscpu_log = self._koji_session.downloadTaskOutput(task_id, "hw_info.log").decode(
                "utf-8",
            )
            logger.debug(f"lscpu log for task: {task_id} - {lscpu_log}")

            hw_info = HwInfo.parse_from_lscpu(lscpu_log)
            self._host_hw_info_map[task_info["host_id"]] = hw_info
            return hw_info
        except koji.GenericError:
            if task_info.get("host_id") and task_info["host_id"] in self._host_hw_info_map:
                return self._host_hw_info_map[task_info["host_id"]]

            logger.error(
                f"Failed to fetch hw_info for task: {task_id}, no hw_info.log found in map",
            )
            return None
        except Exception as e:
            logger.error(f"Failed to fetch hw_info for task: {task_id} - {e!s}")
            return None

    def _fetch_dataset_record(self, build: dict, task_info: dict) -> Optional[Record]:
        mock_chroot_name = None
        regex = re.search(r"\.fc(\d{2})", build["release"])
        if regex:
            fedora_version = regex.group(1)
            if int(fedora_version) == self._fedora_rawhide_number:
                fedora_version = "rawhide"

            mock_chroot_name = f"fedora-{fedora_version}-{task_info['arch']}"
            logger.info(f"Mock chroot name: {mock_chroot_name} for build: {build['nvr']}")
        else:
            logger.error(f"Failed to parse Fedora version from release: {build['release']}")

        hw_info = self._fetch_hw_info_from_koji(task_info)
        if not hw_info:
            return None

        return Record(
            package_name=build["package_name"],
            version=build["version"],
            release=build["release"],
            epoch=build["epoch"] or 0,
            mock_chroot_name=mock_chroot_name,
            build_duration=task_info["completion_ts"] - task_info["start_ts"],
            hw_info=hw_info,
        )

    def _append_batch_of_successful_builds(
        self,
        successful_builds: list[Record],
        builds: list[dict],
    ) -> None:
        for build in builds:
            logger.info(f"Fetching build: {build['nvr']}")
            try:
                task_descendents = self._koji_session.getTaskDescendents(
                    build["task_id"],
                )[str(build["task_id"])]
                for task_info in task_descendents:
                    # this is the task that produces the RPM, thus it has the hw_info.log needed
                    # for HwInfo dataclass
                    if task_info["method"] == "buildArch":
                        logger.info(f"Fetching task descendant: {task_info['id']}")
                        dataset_record = self._fetch_dataset_record(build, task_info)
                        if dataset_record:
                            successful_builds.append(dataset_record)
            except koji.GenericError as e:
                logger.error(f"Failed to fetch build: {e!s}")
                continue

    def fetch_data(self) -> list[Record]:
        successful_builds: list[Record] = []
        # TODO: tqdm for progress bar, but how to get the total number of pages?
        while True:
            try:
                time_params = {}
                if self.start_date:
                    time_params["createdAfter"] = self.start_date.timestamp()

                if self.end_date:
                    time_params["createdBefore"] = self.end_date.timestamp()

                logger.info(f"Fetching page {self._current_page} of builds...")
                builds = self._koji_session.listBuilds(
                    state=koji.BUILD_STATES["COMPLETE"],
                    queryOpts={
                        "limit": self._limit,
                        "offset": self._current_page * self._limit,
                        "order": "-completion_ts",
                    },
                    **time_params,
                )
                if not builds:
                    # last page
                    break

                self._append_batch_of_successful_builds(successful_builds, builds)
                self._current_page += 1
            except Exception as e:
                print(f"Failed to fetch builds: {e!s}")
                # Sometimes koji throws an generic error, unexpected exceptions or something
                # like that. In that case, just skip to next page instead... this is dealing with
                # really old data so the koji python API freaks out sometimes.
                self._current_page += 1
                continue

        return successful_builds


class CoprFetcher(Fetcher):
    def __init__(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        is_copr_instance: bool = False,
    ) -> None:
        super().__init__(start_date, end_date)
        self.is_copr_instance = is_copr_instance

    def _fetch_copr_data_from_instance(self) -> list[Record]:
        from copr_common.enums import StatusEnum
        from coprs import models
        from coprs.logic.builds_logic import BuildChrootsLogic

        # TODO: something will move from this scope once the other fetcher is implemented
        def _fetch_hw_info_from_copr_instance(path_to_hw_info: str) -> Optional[HwInfo]:
            try:
                if not path_to_hw_info or not os.path.exists(path_to_hw_info):
                    return None

                with gzip.open(path_to_hw_info, "r") as f:
                    file_content = f.read()
                    return HwInfo.parse_from_lscpu(file_content.decode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to fetch hw_info from Copr instance: {e!s}")
                return None

        def _parse_build_chroot(build_chroot: models.BuildChroot) -> Optional[Record]:
            try:
                pkg_version = build_chroot.build.pkg_version
                epoch = pkg_version.split(":")[0] if ":" in pkg_version else 0
                version = pkg_version.split(":")[-1].split("-")[0]
                release = pkg_version.split("-")[-1]

                # TODO: function for this, or make it public...
                path_to_hw_info = build_chroot._compressed_log_variant("hw_info.log", [])
                if path_to_hw_info:
                    path_to_hw_info = path_to_hw_info.replace("http://backend_httpd:5002", "")
                    hw_info = _fetch_hw_info_from_copr_instance(path_to_hw_info)
                else:
                    return None

                if hw_info is None:
                    logger.error(f"Failed to fetch hw_info for build_chroot: {build_chroot.id}")
                    return None

                return Record(
                    package_name=build_chroot.build.package.name,
                    epoch=epoch,
                    version=version,
                    release=release,
                    mock_chroot_name=build_chroot.mock_chroot.name,
                    build_duration=build_chroot.ended_on - build_chroot.started_on,
                    hw_info=hw_info,
                )
            except Exception as e:
                logger.error(f"Failed to parse Copr build_chroot: {e!s}")
                return None

        build_chroots = (
            BuildChrootsLogic.get_multiply()
            .filter(
                models.BuildChroot.status == StatusEnum("succeeded"),
            )
            .filter(
                models.Build.pkg_version.is_not(None),
            )
            .filter(
                models.BuildChroot.started_on.is_not(None),
            )
            .filter(
                models.BuildChroot.ended_on.is_not(None),
            )
            .filter(
                models.Copr.deleted.is_(False),
            )
            .filter(
                models.MockChroot.is_active.is_(True),
            )
            .all()
        )

        result = []
        for build_chroot in tqdm.tqdm(build_chroots):
            record = _parse_build_chroot(build_chroot)
            if record:
                logger.info(f"Succesfully retrieved record for {record.nevra}")
                result.append(record)
            else:
                logger.warning(f"Parsing for build chroot {build_chroot.id}")

        return result

    def fetch_data(self) -> list[Record]:
        if self.is_copr_instance:
            return self._fetch_copr_data_from_instance()
        raise NotImplementedError("CoprFetcher is not implemented yet")
