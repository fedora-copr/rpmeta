# ubuntu has archaic version of just, thus nice documenting decorators
# like doc or groups are not available :(

ci := "false"

test_container_name := "rpmeta_test:latest"
run_container_name := "rpmeta:latest"

test_target := "test/unit test/integration"
test_e2e_target := "test/e2e"

bind_path := "/app/bind"
minimal_python_version := "3.10"

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
test_container_run := container_run_base + test_container_name
run_container_run := container_run_base + run_container_name


[private]
build image:
    {{container_engine}} build -t {{image}} -f test/Containerfile .

[private]
rebuild image:
    {{container_engine}} build --no-cache -t {{image}} -f test/Containerfile .

[private]
shell run:
    {{run}} /bin/bash

[private]
rm-image image:
    {{container_engine}} image rm {{image}}

# Builds the testing container image
build-test-container: (build test_container_name)

# Builds the testing container image without cache, rebuilding all layers
rebuild-test-container: (rebuild test_container_name)

# Removes the testing container image
rm-image-test-container: (rm-image test_container_name)

# Spawns bash shell in the testing container
run-test-container: (shell test_container_run)


# Builds the rpmeta containerized image
build-rpmeta-container: (build run_container_name)

# Builds the rpmeta containerized image without cache, rebuilding all layers
rebuild-rpmeta-container: (rebuild run_container_name)

# Removes the rpmeta containerized image
rm-image-rpmeta-container: (rm-image run_container_name)

# Spawns bash shell in the rpmeta container
run-rpmeta-container: (shell run_container_run)


# Runs the unit and integration tests in the container
test-in-container: build-test-container
    @echo "Running test targets in container"
    {{test_container_run}} /bin/bash -c \
        "cd {{bind_path}} && \
        {{uv_sync}} && \
        {{uv_cmd}} run -- {{pytest_cmd}} {{test_target}}"

# Runs the e2e tests in the container, testing both the native python version and the oldest supported
test-e2e-in-container: build-test-container
    @echo "Running e2e tests in container with fedora native python version"
    {{test_container_run}} /bin/bash -c \
        "cd {{bind_path}} && \
        {{uv_sync}} --reinstall && \
        {{uv_cmd}} run -- {{pytest_cmd}} {{test_e2e_target}}"

    @echo "Running e2e tests in container with minimal python version supported: " \
        "{{minimal_python_version}}"
    {{test_container_run}} /bin/bash -c \
        "cd {{bind_path}} && \
        {{uv_cmd}} python install {{minimal_python_version}} && \
        {{uv_sync}} --reinstall --python {{minimal_python_version}} && \
        {{uv_cmd}} run --python {{minimal_python_version}} -- {{pytest_cmd}} {{test_e2e_target}}"

# for fast re-running of the e2e tests
# this should be used for development only, not for CI

# Runs the native python e2e tests inside the container, but without a fresh install
test-e2e-fast-in-container:
    @echo "Running e2e tests with only newer Python and no fresh install... \
     this should be used for development only"
    {{test_container_run}} /bin/bash -c \
        "cd {{bind_path}} && \
        {{uv_sync}} && \
        {{uv_cmd}} run -- {{pytest_cmd}} {{test_e2e_target}}"

# Runs all tests in the container
test-everything-in-container: test-in-container test-e2e-in-container


# Builds and installs the rpmeta container
rpmeta-container-install: build-rpmeta-container
    @echo "Installing the rpmeta container"
    {{run_container_run}} /bin/bash -c "cd {{bind_path}} && {{uv_cmd}} pip install -e ."

# Runs the rpmeta tool inside the container as it would be run outside; useful for testing dev branch
rpmeta-containerized +args:
    {{run_container_run}} /bin/bash -c \
        "cd {{bind_path}} && rpmeta {{args}}"
