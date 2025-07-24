#!/usr/bin/env python3

import inspect
import sys
from pathlib import Path

import tomlkit
from pydantic import BaseModel
from tomlkit.items import Comment, Table, Trivia
from tomlkit.toml_document import TOMLDocument

# Add the parent directory to the path so we can import rpmeta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rpmeta.config import Config


def generate_toml_from_pydantic(model_class) -> str:
    """
    Inspects a Pydantic model and generates a commented TOML configuration file.

    Args:
        model_class: The root Pydantic model class (e.g., Config).

    Returns:
        A string containing the formatted and commented TOML configuration.
    """
    doc = tomlkit.document()

    header_lines = [
        "RPMeta Example Configuration File",
        "Configuration is taken from ~/.config/rpmeta/config.toml or /etc/rpmeta/config.toml",
        "or specify the path when running rpmeta with --config /path/to/config.toml",
    ]
    for line in header_lines:
        doc.add(Comment(Trivia(comment=f"# {line}")))

    doc.add(tomlkit.nl())
    doc.add(tomlkit.nl())

    class_doc = inspect.getdoc(model_class)
    if class_doc:
        doc.add(Comment(Trivia(comment=f"# {class_doc}")))
        doc.add(tomlkit.nl())

    for field_name, field_info in model_class.model_fields.items():
        process_field(doc, field_name, field_info)
        doc.add(tomlkit.nl())

    return tomlkit.dumps(doc).rstrip() + "\n"


def process_field(parent_table: TOMLDocument | Table, field_name: str, field_info):
    description = field_info.description or ""
    field_type = field_info.annotation

    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        parent_table.add(tomlkit.nl())
        table = tomlkit.table()

        class_doc = inspect.getdoc(field_type)
        if class_doc:
            parent_table.add(Comment(Trivia(comment=f"# {class_doc}", trail="")))

        parent_table.add(field_name, table)

        # recursively process the fields of the nested model
        for sub_field_name, sub_field_info in field_type.model_fields.items():  # type: ignore
            process_field(table, sub_field_name, sub_field_info)
    else:
        # this is a simple key-value pair
        value = None

        # prioritize example value over default
        if field_info.examples and isinstance(field_info.examples, list):
            value = field_info.examples[0]
        elif field_info.default is not None:
            value = field_info.default
        elif field_info.default_factory is not None:
            # handled by the recursive call above
            return
        elif field_info.default is None:
            # skip
            return

        if isinstance(value, Path):
            value = str(value)

        item = tomlkit.item(value)

        if description:
            item.comment(description)

        parent_table.add(field_name, item)


if __name__ == "__main__":
    print("Generating documentation TOML configuration from Pydantic models...")
    toml_content = generate_toml_from_pydantic(Config)

    with open("files/config.toml.example", "w", encoding="utf-8") as f:
        f.write(toml_content)

    print("\nSuccessfully generated 'files/config.toml.example'")

    print("\n--- File Content ---")
    print(toml_content)
