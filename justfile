container_name := "rpmeta_test:latest"
working_dir := "$(pwd)"
bind_path := "/app/bind"
minimal_python_version := "3.9"

build:
    podman build -t {{container_name}} -f test/Containerfile .

rebuild:
    podman build --no-cache -t {{container_name}} -f test/Containerfile .

shell:
    podman run --rm -it -v {{working_dir}}:{{bind_path}}:Z {{container_name}} /bin/bash

rm-image:
    podman image rm {{container_name}}

test-e2e-in-container: rebuild
    @echo "Running e2e tests in container with fedora native python version"
    podman run --rm -v {{working_dir}}:{{bind_path}}:Z {{container_name}} bash -c \
        "cd {{bind_path}} && \
        uv sync --all-extras --all-groups --reinstall && \
        uv run -- \
            pytest -vvv --log-level DEBUG --cov-report term test/e2e"

    @echo "Running e2e tests in container with minimal python version supported: " \
        "$(minimal_python_version)"
    just rm-image
    just rebuild
    podman run --rm -v {{working_dir}}:{{bind_path}}:Z {{container_name}} bash -c \
        "cd {{bind_path}} && \
        uv python install {{minimal_python_version}} && \
        uv sync --all-extras --all-groups --reinstall --python {{minimal_python_version}} && \
        uv run --python {{minimal_python_version}} -- \
            pytest -vvv --log-level DEBUG --cov-report term test/e2e"
