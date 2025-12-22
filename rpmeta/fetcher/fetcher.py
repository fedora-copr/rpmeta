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
from copr.v3 import Client
from copr.v3.pagination import next_page
from fedora_distro_aliases import get_distro_aliases

from rpmeta.config import Config
from rpmeta.dataset import HwInfo, Record

logger = logging.getLogger(__name__)


def _get_distro_aliases_retry(retries=5, delay=20) -> dict:
    """
    Retries a few times if the Bodhi API is unavailable before failing.
    """
    for attempt in range(1, retries + 1):
        try:
            return get_distro_aliases(cache=True)
        except requests.exceptions.RequestException as e:
            logging.warning("Attempt %d/%d failed: %s", attempt, retries, e)
            if attempt < retries:
                time.sleep(delay)
                continue

    logging.error("Failed to fetch Fedora Rawhide number after multiple attempts.")
    raise


class Fetcher(ABC):
    def __init__(
        self,
        config: Config,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10000,
    ) -> None:
        self.config = config
        self.start_date = None
        if start_date:
            self.start_date = int(start_date.timestamp())

        if end_date:
            self.end_date = int(end_date.timestamp())
        else:
            self.end_date = int(datetime.now().timestamp())

        self.limit = limit

        # keep it here so it fails right away if bodhi API is not available at the moment
        self._fedora_rawhide_number = max(
            int(alias.version_number) for alias in _get_distro_aliases_retry()["fedora-all"]
        )

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
        config: Config,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10000,
    ) -> None:
        super().__init__(config, start_date, end_date, limit)

        logger.info("Initializing KojiFetcher instance: %s", self.config.koji.hub_url)
        self._koji_session = koji.ClientSession(self.config.koji.hub_url)

        self._host_hw_info_map: dict[int, HwInfo] = {}
        self._current_page = 0

    def _fetch_hw_info_from_koji(self, task_info: dict) -> Optional[HwInfo]:
        task_id = task_info["id"]
        logger.info("Fetching hw_info for task: %d", task_id)
        if task_info.get("host_id") and task_info["host_id"] in self._host_hw_info_map:
            return self._host_hw_info_map[task_info["host_id"]]

        try:
            lscpu_log = self._koji_session.downloadTaskOutput(task_id, "hw_info.log").decode(
                "utf-8",
            )
            logger.debug("lscpu log for task: %d - %s", task_id, lscpu_log)

            hw_info = HwInfo.parse_from_lscpu(lscpu_log)
            self._host_hw_info_map[task_info["host_id"]] = hw_info
            return hw_info
        except koji.GenericError:
            if task_info.get("host_id") and task_info["host_id"] in self._host_hw_info_map:
                return self._host_hw_info_map[task_info["host_id"]]

            logger.error(
                "Failed to fetch hw_info for task: %d, no hw_info.log found in map",
                task_id,
            )
            return None
        except Exception as e:
            logger.error("Failed to fetch hw_info for task: %d - %s", task_id, e)
            return None

    def _get_chroot_from_release(self, release: str, arch: str) -> Optional[str]:
        match = None
        # searches for strings like "X.fcXX"
        regex_fc = re.search(r"\.fc(\d{2})", release)
        if regex_fc:
            fedora_version = regex_fc.group(1)
            match = f"fedora-{fedora_version}"

        # searches for strings like "X.elnX"
        regex_eln = re.search(r"\.eln\d+", release)
        if regex_eln:
            match = "fedora-eln"

        # searches for strings like "X.elX(_X)"
        regex_epel = re.search(r"\.el(\d+)(?:_\d+)?", release)
        if regex_epel:
            match = f"epel-{regex_epel.group(1)}"

        if match is not None:
            return f"{match}-{arch}"

        logger.error("Failed to parse chroot from release: %s", release)
        return None

    def _fetch_dataset_record(self, build: dict, task_info: dict) -> Optional[Record]:
        hw_info = self._fetch_hw_info_from_koji(task_info)
        if not hw_info:
            return None

        mock_chroot = self._get_chroot_from_release(build["release"], hw_info.cpu_arch)

        return Record(
            package_name=build["package_name"],
            version=build["version"],
            epoch=build["epoch"] or 0,
            mock_chroot=mock_chroot,
            build_duration=int(task_info["completion_ts"] - task_info["start_ts"]),
            hw_info=hw_info,
        )

    def _append_batch_of_successful_builds(
        self,
        successful_builds: list[Record],
        builds: list[dict],
    ) -> None:
        for build in builds:
            logger.info("Fetching build: %s", build["nvr"])
            try:
                task_descendents = self._koji_session.getTaskDescendents(
                    build["task_id"],
                )[str(build["task_id"])]
                for task_info in task_descendents:
                    # this is the task that produces the RPM, thus it has the hw_info.log needed
                    # for HwInfo dataclass
                    if task_info["method"] == "buildArch":
                        logger.info("Fetching task descendant: %d", task_info["id"])
                        dataset_record = self._fetch_dataset_record(build, task_info)
                        if dataset_record:
                            successful_builds.append(dataset_record)
            except koji.GenericError as e:
                logger.error("Failed to fetch build: %s", e)
                continue

    def fetch_data(self) -> list[Record]:
        successful_builds: list[Record] = []
        # TODO: tqdm for progress bar, but how to get the total number of pages?
        while True:
            try:
                time_params = {}
                if self.start_date:
                    time_params["createdAfter"] = self.start_date

                if self.end_date:
                    time_params["createdBefore"] = self.end_date

                logger.info("Fetching page %d of builds...", self._current_page)
                builds = self._koji_session.listBuilds(
                    state=koji.BUILD_STATES["COMPLETE"],
                    queryOpts={
                        "limit": self.limit,
                        "offset": self._current_page * self.limit,
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
                logger.error("Failed to fetch builds: %s", e)
                # Sometimes koji throws an generic error, unexpected exceptions or something
                # like that. In that case, just skip to next page instead... this is dealing with
                # really old data so the koji python API freaks out sometimes.
                self._current_page += 1
                continue

        return successful_builds


class CoprFetcher(Fetcher):
    def __init__(
        self,
        config: Config,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        is_copr_instance: bool = False,
        limit: int = 10000,
    ) -> None:
        super().__init__(config, start_date, end_date, limit)

        self.is_copr_instance = is_copr_instance
        self.client = Client({"copr_url": self.config.copr.api_url})

    def _fetch_copr_data_from_instance(self) -> list[Record]:
        from copr_common.enums import StatusEnum
        from coprs import models
        from coprs.logic.builds_logic import BuildChrootsLogic

        build_chroots_query = (
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
                models.BuildChroot.ended_on <= self.end_date,
            )
            .filter(
                models.Copr.deleted.is_(False),
            )
            .filter(
                models.MockChroot.is_active.is_(True),
            )
        )

        if self.start_date:
            build_chroots_query = build_chroots_query.filter(
                models.BuildChroot.ended_on >= self.start_date,
            )

        build_chroots = build_chroots_query.all()
        result = []
        for build_chroot in tqdm.tqdm(build_chroots):
            if not build_chroot.result_dir_url:
                logger.error(
                    "Failed to fetch path_to_hw_info for build_chroot: %d",
                    build_chroot.id,
                )
                continue

            parsed_mock_chroot = build_chroot.mock_chroot.name
            if parsed_mock_chroot and parsed_mock_chroot.startswith("fedora-rawhide"):
                parsed_mock_chroot = parsed_mock_chroot.replace(
                    "rawhide",
                    str(self._fedora_rawhide_number),
                )

            record = CoprFetcher._parse_build_chroot(
                pkg_name=build_chroot.build.package.name,
                pkg_version=build_chroot.build.pkg_version,
                mock_chroot=parsed_mock_chroot,
                result_dir_url=build_chroot.result_dir_url,
                build_duration=int(build_chroot.ended_on - build_chroot.started_on),
            )
            if record:
                logger.info("Succesfully retrieved record for %s", record.neva)
                result.append(record)
            else:
                logger.warning("Parsing for build chroot %d failed", build_chroot.id)

        return result

    @staticmethod
    def _parse_build_chroot(
        pkg_name: str,
        pkg_version: str,
        mock_chroot: str,
        result_dir_url: str,
        build_duration: int,
    ) -> Optional[Record]:
        try:
            url_to_hw_info = CoprFetcher._get_url_to_hw_info_log(result_dir_url)
            logger.debug("URL to hw_info: %s", url_to_hw_info)
            if url_to_hw_info.startswith("http://backend_httpd:5002"):
                # Copr instance is running in a container, replace the URL then to real copr
                logger.debug(
                    "Replacing URL to hw_info: %s for Copr instance",
                    url_to_hw_info,
                )
                url_to_hw_info = url_to_hw_info.replace(
                    "http://backend_httpd:5002",
                    "https://download.copr.fedorainfracloud.org",
                )
                logger.debug("Replaced URL to hw_info: %s", url_to_hw_info)

            logger.debug("Fetching hw_info from: %s", url_to_hw_info)
            hw_info = CoprFetcher._fetch_hw_info_from_copr(url_to_hw_info)

            if hw_info is None:
                logger.error("Failed to fetch hw_info")
                return None

            epoch, version = CoprFetcher._epoch_and_version_from_pkg_version(pkg_version)
            return Record(
                package_name=pkg_name,
                epoch=epoch,
                version=version,
                mock_chroot=mock_chroot,
                build_duration=build_duration,
                hw_info=hw_info,
            )
        except Exception as e:
            logger.error("Failed to parse Copr build_chroot: %s", e)
            return None

    @staticmethod
    def _epoch_and_version_from_pkg_version(pkg_version: str) -> tuple[int, str]:
        epoch = int(pkg_version.split(":")[0]) if ":" in pkg_version else 0
        version = pkg_version.split(":")[-1].split("-")[0]
        return epoch, version

    @staticmethod
    def _fetch_hw_info_from_copr(url_to_hw_info: str) -> Optional[HwInfo]:
        try:
            response = requests.get(url_to_hw_info)
            response.raise_for_status()
            return HwInfo.parse_from_lscpu(response.content.decode("utf-8"))
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch hw_info from Copr instance: %s", e)
        except OSError as e:
            logger.error("Failed to read hw_info.log: %s", e)

        return None

    @staticmethod
    def _get_url_to_hw_info_log(result_dir_url: str) -> str:
        return os.path.join(result_dir_url, "hw_info.log.gz")

    def _append_records_from_build(self, build: dict, records: list[Record]) -> None:
        build_chroots = self.client.build_chroot_proxy.get_list(build_id=build["id"])
        if not build_chroots:
            logger.info("No build_chroots for build: %d found", build["id"])
            return

        for build_chroot in build_chroots:
            if build_chroot["state"] != "succeeded":
                continue

            record = self._parse_build_chroot(
                pkg_name=build["source_package"]["name"],
                pkg_version=build["source_package"]["version"],
                mock_chroot=build_chroot["name"],
                result_dir_url=build_chroot["result_url"],
                build_duration=int(build_chroot["ended_on"] - build_chroot["started_on"]),
            )
            if record:
                logger.info("Succesfully retrieved record for %s", record.neva)
                records.append(record)
            else:
                logger.warning(
                    "Parsing for build chroot %s for build id: %s failed",
                    build_chroot["name"],
                    build["id"],
                )

    def _get_records_from_project(self, project: dict) -> Optional[list[Record]]:
        builds = self.client.build_proxy.get_list(
            ownername=project["ownername"],
            projectname=project["name"],
        )
        if not builds:
            logger.info("No builds for project: %s found", project["full_name"])
            return []

        records: list[Record] = []
        for build in builds:
            if build["ended_on"] is None or build["ended_on"] > self.end_date:
                logger.info("Skipping build: %s as it's newer than end_date", build["id"])
                continue

            if self.start_date and build["ended_on"] < self.start_date:
                logger.info("Skipping build: %s as it's older than start_date", build["id"])
                return None

            self._append_records_from_build(build, records)

        return records

    def _fetch_copr_data_from_api(self) -> list[Record]:
        pagination = {"limit": self.limit, "order": "id", "order_type": "DESC"}
        projects_page = self.client.project_proxy.get_list(pagination=pagination)
        result: list[Record] = []

        while projects_page:
            for project in projects_page:
                if project.get("name") is None or project.get("ownername") is None:
                    logger.error("Skipping project with missing name or ownername: %s", project)
                    continue

                logger.info("Fetching builds for project: %s", project["full_name"])
                records = self._get_records_from_project(project)
                if records is None:
                    # end of date range reached
                    return result

                result.extend(records)

            projects_page = next_page(projects_page)

        # last page
        return result

    def fetch_data(self) -> list[Record]:
        if self.is_copr_instance:
            from coprs import app

            with app.app_context():
                return self._fetch_copr_data_from_instance()

        return self._fetch_copr_data_from_api()
