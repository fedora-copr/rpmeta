container_name := "rpmeta_test"
working_dir := "$(pwd)"
bind_path := "/app/bind"

build:
    podman build -t {{container_name}} -f test/Containerfile .

rebuild:
    podman build --no-cache -t {{container_name}} -f test/Containerfile .

shell:
    podman run --rm -it -v {{working_dir}}:{{bind_path}}:Z {{container_name}} /bin/bash

rm-image:
    podman image rm {{container_name}}
