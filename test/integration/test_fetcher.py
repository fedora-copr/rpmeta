from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from rpmeta.fetcher import CoprFetcher, KojiFetcher


@patch("rpmeta.fetcher.koji.ClientSession")
@patch("rpmeta.fetcher._get_distro_aliases_retry")
def test_koji_fetcher_fetch_data(
    mock_get_distro_aliases,
    mock_client_session,
    dataset_record,
    koji_build,
    koji_task_descendant,
):
    mock_get_distro_aliases.return_value = {
        "fedora-all": [SimpleNamespace(version_number="36", name="fedora")],
    }

    mock_session = mock_client_session.return_value
    mock_session.listBuilds.side_effect = [
        koji_build,
        [],
    ]
    mock_session.getTaskDescendents.side_effect = [koji_task_descendant]
    mock_session.downloadTaskOutput.return_value = b"mock lscpu log"

    with patch("rpmeta.dataset.HwInfo.parse_from_lscpu", return_value=dataset_record.hw_info):
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        fetcher = KojiFetcher(start_date=start_date, end_date=end_date)
        result = fetcher.fetch_data()

    mock_session.listBuilds.assert_called_with(
        state=1,
        queryOpts={"limit": 10000, "offset": 10000, "order": "-completion_ts"},
        createdAfter=int(start_date.timestamp()),
        createdBefore=int(end_date.timestamp()),
    )
    mock_session.getTaskDescendents.assert_called_with(koji_build[0]["task_id"])
    mock_session.downloadTaskOutput.assert_called_with(131647469, "hw_info.log")

    assert len(result) == 1
    assert result[0] == dataset_record


@patch("rpmeta.fetcher.Client")
@patch("rpmeta.fetcher.next_page", return_value=None)
@patch("rpmeta.fetcher.requests.get")
def test_copr_fetcher_fetch_data(mock_requests_get, mock_next_page, mock_client, dataset_record):
    # Mock requests.get
    mock_response = mock_requests_get.return_value
    mock_response.status_code = 200
    mock_response.content.decode.return_value = "mock lscpu log"

    # Mock Copr client
    mock_copr_client = mock_client.return_value
    mock_copr_client.build_chroot_proxy.get_list.return_value = [
        {
            "id": 1,
            "name": dataset_record.mock_chroot,
            "ended_on": 894,
            "started_on": 1,
            "result_url": "xyz",
            "state": "succeeded",
        },
    ]
    mock_copr_client.build_proxy.get_list.return_value = [
        {
            "id": 1,
            "ended_on": 2,
            "source_package": {
                "name": dataset_record.package_name,
                "version": f"{dataset_record.version}-1.fc43",
            },
        },
    ]
    mock_copr_client.project_proxy.get_list.return_value = [
        {
            "name": "test-project",
            "ownername": "test-owner",
            "full_name": "test-project/test-package",
        },
    ]

    # Mock HwInfo parsing
    with patch("rpmeta.dataset.HwInfo.parse_from_lscpu", return_value=dataset_record.hw_info):
        fetcher = CoprFetcher()
        result = fetcher.fetch_data()

    # Assert requests.get was called with the correct URL
    mock_requests_get.assert_called_with("xyz/hw_info.log.gz")

    assert len(result) == 1
    assert result[0] == dataset_record
