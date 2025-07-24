from rpmeta.config import Config
from scripts.generate_config import generate_toml_from_pydantic


def test_generated_config_up_to_date():
    current_content = generate_toml_from_pydantic(Config)

    with open("files/config.toml.example", encoding="utf-8") as f:
        expected_content = f.read()

    # remove all the newlines, compare only the content
    current_content = current_content.replace("\n", "")
    expected_content = expected_content.replace("\n", "")

    assert current_content == expected_content, (
        "The generated config file is not up to date. Run `just generate-config` to update it."
    )
