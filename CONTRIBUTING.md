# Contributing to RPMeta

Thank you for your interest in contributing to RPMeta! This document provides guidelines and information for contributors to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Testing](#testing)
- [Code Style and Linting](#code-style-and-linting)
- [Version Management](#version-management)
- [Submitting Changes](#submitting-changes)
- [Creating a Release](#creating-a-release)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Docker or Podman (for containerized testing)
- Just command runner (for automation)

### Setting Up Your Development Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/fedora-copr/rpmeta.git
   cd rpmeta
   ```

2. Install pre-commit hooks (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Testing

RPMeta uses pytest for testing.

### Running Tests Locally

To run tests locally in your development environment:

```bash
# Run all tests
pytest

# Run specific test categories
pytest test/unit
pytest test/integration
pytest test/e2e
```

### Running Tests in Containers

The project uses the `just` command runner to automate testing in containers, which ensures consistent testing environments. Run:

```bash
just --list
```

to see every recipe available, including its documentation. You can run tests, manage versioning and build testing or experimenting
containers with it.

## Code Style and Linting

The project uses pre-commit hooks for code style enforcement, linting, and formatting. Configuration is defined in `.pre-commit-config.yaml`.

Rather than running linting tools directly, you should use pre-commit:

```bash
# Run pre-commit on all files
pre-commit run --all-files

# Run pre-commit on staged files only
pre-commit run
```

Pre-commit will automatically run these checks in isolated environment before each commit if you've installed the hooks as described in the setup section.

If you need to bypass the hooks for a particular commit (not recommended for normal development, you'll still need to fix it in PR):

```bash
git commit -m "Your message" --no-verify
```

## Version Management

The canonical version is defined in `rpmeta/__init__.py` as the `__version__` variable. All other files (like `pyproject.toml` and the spec file) reference this version.

To check the current version:

```bash
python scripts/update_version.py

# or using the just target
just version-get
```

To update the version across all files:

```bash
python scripts/update_version.py NEW_VERSION

# or using the just target
just version-set NEW_VERSION
```

For example:

```bash
just version-set 0.2.0
```

## Creating a Release

1. Update the version number as described in [Version Management](#version-management), commit and push the new version.
2. Create a new release on GitHub with release notes.
3. The CI/CD pipeline will build and publish the package to PyPI and Copr.
