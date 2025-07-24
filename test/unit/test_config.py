from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from rpmeta.cli.ctx import Context
from rpmeta.cli.main import entry_point
from rpmeta.config import Api, Config, ConfigManager, Model
from rpmeta.constants import KOJI_HUB_URL, RESULT_DIR_LOCATIONS


def test_default_config_creation():
    # just a few examples to check it works
    config = Config()

    assert config.result_dir == RESULT_DIR_LOCATIONS[0]

    assert config.api.host == "localhost"
    assert config.api.port == 44882
    assert config.api.debug is False

    assert config.koji.hub_url == KOJI_HUB_URL

    assert config.copr.api_url == "https://copr.fedorainfracloud.org/api_3"

    assert config.model.random_state == 42
    assert config.model.n_jobs == -1
    assert config.model.test_size == 0.2
    assert config.model.verbose is False

    assert config.logging.format == "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    assert config.logging.datefmt == "%Y-%m-%d %H:%M:%S"


def test_config_validation():
    # just a few examples to check it works
    with pytest.raises(ValidationError):
        Api(port=0)


def test_ensure_result_dir_exists(tmp_path):
    non_existent_dir = tmp_path / "non_existent"
    assert not non_existent_dir.exists()

    config = Config(result_dir=non_existent_dir)
    assert non_existent_dir.exists()
    assert config.result_dir == non_existent_dir


def test_get_result_dir(monkeypatch, tmp_path):
    mock_result_dir = tmp_path / "result_dir"
    mock_result_dir.mkdir()

    monkeypatch.setattr("rpmeta.config.RESULT_DIR_LOCATIONS", [mock_result_dir])

    result_dir = ConfigManager._get_result_dir()
    assert result_dir == mock_result_dir

    new_dir = tmp_path / "new_dir"
    monkeypatch.setattr("rpmeta.config.RESULT_DIR_LOCATIONS", [new_dir])

    result_dir = ConfigManager._get_result_dir()
    assert result_dir == new_dir
    assert new_dir.exists()


def test_config_merging():
    merged_config = Config(
        api=Api(host="overridehost", port=1111),
        model=Model(verbose=True),
    )

    assert merged_config.api.host == "overridehost"
    assert merged_config.api.port == 1111
    assert merged_config.model.verbose is True


def test_cli_context_config():
    mock_config = MagicMock()
    ctx = Context(mock_config)
    assert ctx.config == mock_config


@patch("rpmeta.cli.main.ConfigManager.get_config")
def test_cli_with_custom_config_file(mock_get_config, example_config, config_file):
    mock_get_config.return_value = example_config

    runner = CliRunner()

    @entry_point.command("dummy-command")
    def dummy_command():
        pass

    result = runner.invoke(
        entry_point,
        ["--config", str(config_file), "dummy-command"],
    )

    assert result.exit_code == 0, result.output
    mock_get_config.assert_called_once_with(config_file=config_file)
