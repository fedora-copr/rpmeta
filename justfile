ci := "false"

container_name := "rpmeta_test:latest"
working_dir := "$(pwd)"
bind_path := "/app/bind"
test_target := "test/unit test/integration"
test_e2e_target := "test/e2e"
minimal_python_version := "3.9"

uv_cmd := "uv --color always"
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
container_run := container_engine + " run " + container_run_opts + working_dir + ":" + \
    bind_path + ":Z --security-opt label=disable " + container_name

build:
    {{container_engine}} build -t {{container_name}} -f test/Containerfile .

rebuild:
    {{container_engine}} build --no-cache -t {{container_name}} -f test/Containerfile .

shell:
    {{container_run}} /bin/bash

rm-image:
    {{container_engine}} image rm {{container_name}}

test-e2e-in-container: build
    @echo "Running e2e tests in container with fedora native python version"
    {{container_run}} /bin/bash -c \
        "cd {{bind_path}} && \
        {{uv_cmd}} sync --all-extras --all-groups --reinstall && \
        {{uv_cmd}} run -- {{pytest_cmd}} {{test_e2e_target}}"

    @echo "Running e2e tests in container with minimal python version supported: " \
        "{{minimal_python_version}}"
    {{container_run}} /bin/bash -c \
        "cd {{bind_path}} && \
        {{uv_cmd}} python install {{minimal_python_version}} && \
        {{uv_cmd}} sync --all-extras --all-groups --reinstall \
            --python {{minimal_python_version}} && \
        {{uv_cmd}} run --python {{minimal_python_version}} -- {{pytest_cmd}} {{test_e2e_target}}"

test-in-container: test-e2e-in-container
