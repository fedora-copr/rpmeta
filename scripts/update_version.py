#!/usr/bin/env python3
"""
Script to update version numbers across the project.
This script ensures that all version references are kept in sync.

Usage:
    python scripts/update_version.py [new_version]

If new_version is not provided, the script will just print the current version.
"""

import re
import sys
from pathlib import Path


def get_current_version():
    init_path = Path(__file__).parent.parent / "rpmeta" / "__init__.py"
    with open(init_path) as f:
        content = f.read()

    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find version in __init__.py")

    return match.group(1)


def update_version(new_version):
    # __init__.py
    init_path = Path(__file__).parent.parent / "rpmeta" / "__init__.py"
    with open(init_path) as f:
        content = f.read()

    updated = re.sub(
        r'(__version__\s*=\s*["\'])([^"\']+)(["\'])',
        f"\\1{new_version}\\3",
        content,
    )

    with open(init_path, "w") as f:
        f.write(updated)

    print(f"Updated version in {init_path} to {new_version}")

    # pyproject.toml
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path) as f:
        content = f.read()

    updated = re.sub(
        r'(version\s*=\s*["\'])([^"\']+)(["\'])',
        f"\\1{new_version}\\3",
        content,
        count=1,  # replace the first occurrence
    )

    with open(pyproject_path, "w") as f:
        f.write(updated)

    print(f"Updated version in {pyproject_path} to {new_version}")

    # spec file
    spec_path = Path(__file__).parent.parent / "files" / "rpmeta.spec"
    if spec_path.exists():
        with open(spec_path) as f:
            content = f.read()

        updated = re.sub(
            r"(Version:\s*)([0-9.]+)",
            f"\\1{new_version}",
            content,
        )

        with open(spec_path, "w") as f:
            f.write(updated)

        print(f"Updated version in {spec_path} to {new_version}")


def main():
    current_version = get_current_version()
    print(f"Current version: {current_version}")

    if len(sys.argv) > 1:
        new_version = sys.argv[1]
        if new_version != current_version:
            update_version(new_version)
            print(f"Version updated to {new_version}")
        else:
            print(f"No update needed, version is already {current_version}")


if __name__ == "__main__":
    main()
