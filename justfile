# ubuntu has archaic version of just, thus nice documenting decorators
# like doc or groups are not available :(

ci := "false"

test_container_name := "rpmeta_test:latest"
run_container_name := "rpmeta:latest"

test_target := "test/unit test/integration"
test_e2e_target := "test/e2e"
test_all_targets := test_target + " " + test_e2e_target

bind_path := "/app/bind"
minimal_python_version := "3.11"

uv_cmd := "uv --color always"
uv_sync := uv_cmd + " sync --all-extras --all-groups"
pytest_cmd := "pytest -vvv --log-level DEBUG --color=yes --cov-report term"

container_engine := if ci == "true" {
    "podman"
} else {
    `podman --version > /dev/null 2>&1 && echo "podman" || echo "docker"`
}

container_run_opts := if ci == "true" {
    " --rm -v "
} else {
    " --rm -ti -v "
}

container_run_base := container_engine + " run " + container_run_opts + "$(pwd)" + ":" + \
    bind_path + ":Z --security-opt label=disable "

_build_cmd := container_engine + " build -f test/Containerfile -t"
_rebuild_cmd := container_engine + " build --no-cache -f test/Containerfile -t"
_rm_image_cmd := container_engine + " image rm"


default: help
test: test-all


help:
    @just --list


# public targets

# Builds the container image - test (default) or run
build target="test":
    @if [ "{{target}}" = "test" ]; then \
        {{_build_cmd}} {{test_container_name}}; \
    else \
        {{_build_cmd}} {{run_container_name}}; \
    fi

# Rebuilds the container image - test (default) or run
rebuild target="test":
    @if [ "{{target}}" = "test" ]; then \
        {{_rebuild_cmd}} {{test_container_name}}; \
    else \
        {{_rebuild_cmd}} {{run_container_name}}; \
    fi


# Removes the container image - test (default) or run
rm-image target="test":
    @if [ "{{target}}" = "test" ]; then \
        {{_rm_image_cmd}} {{test_container_name}}; \
    else \
        {{_rm_image_cmd}} {{run_container_name}}; \
    fi

# Spawns bash shell in the container - test (default) or run
shell target="test":
    @if [ "{{target}}" = "test" ]; then \
        {{container_run_base}} {{test_container_name}} /bin/bash; \
    else \
        {{container_run_base}} {{run_container_name}} /bin/bash; \
    fi


# Runs the unit tests in the container
test-unit: build
    @echo "Running unit and integration tests in container"
    {{container_run_base + test_container_name}} /bin/bash -c \
        "cd {{bind_path}} && \
        {{uv_sync}} && \
        {{uv_cmd}} run -- {{pytest_cmd}} {{test_target}}"


# Runs the e2e tests in the container, testing both the native python version and the oldest supported
test-e2e: build
    @echo "Running e2e tests in container with fedora native python version"
    {{container_run_base + test_container_name}} /bin/bash -c \
        "cd {{bind_path}} && \
        {{uv_sync}} --reinstall && \
        {{uv_cmd}} run -- {{pytest_cmd}} {{test_e2e_target}}"

    @echo "Running e2e tests in container with minimal python version supported: {{minimal_python_version}}"
    {{container_run_base + test_container_name}} /bin/bash -c \
        "cd {{bind_path}} && \
        {{uv_cmd}} python install {{minimal_python_version}} && \
        {{uv_sync}} --reinstall --python {{minimal_python_version}} && \
        {{uv_cmd}} run --python {{minimal_python_version}} -- {{pytest_cmd}} {{test_e2e_target}}"

# for fast re-running of the e2e tests
# this should be used for development only, not for CI

# Runs the native python e2e tests inside the container, but without a fresh install
test-e2e-fast:
    @echo "Running e2e tests with only newer Python and no fresh install... this should be used for development only"
    {{container_run_base + test_container_name}} /bin/bash -c \
        "cd {{bind_path}} && \
        {{uv_sync}} && \
        {{uv_cmd}} run -- {{pytest_cmd}} {{test_e2e_target}}"

# Runs all tests in the container
test-all: test-unit test-e2e


# Shows the current version of the project
version-get:
    @python scripts/update_version.py

# Updates the version across all project files
version-set new_version:
    @python scripts/update_version.py {{new_version}}

# Generates the config.toml file from the Pydantic models
generate-config:
    @echo "Generating config.toml from Pydantic models..."
    @python scripts/generate_config.py
