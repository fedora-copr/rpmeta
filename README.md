## rpmeta: RPM Estimated Time of (build) Arrival

**RPMETA** is a command-line tool designed to **predict RPM build durations** and manage related
data. It provides a set of commands for training a predictive model, making predictions,
fetching data, and serving a REST API endpoint.

---

### Table of Contents

- [Installation](#installation)
- [Usage](#usage)

---

### Installation

#### Fedora:

```bash
dnf copr enable @copr/rpmeta
dnf install rpmeta
```

to install the also subpackages, add `+SUBPACKAGE` to the `rpmeta`.

Fedora is missing a few dependencies, so you need to install them manually:

Dependencies for model training:

- xgboost
- lightgbm

Dependencies for fancy graphs output from Optuna:

- kaleido

#### Other distributions:

```bash
pipx install rpmeta
```

to install subpackages, use `rpmeta[SUBPACKAGE1, SUBPACKAGE2, ...]` syntax

Or from the source:

```bash
pipx install "rpmeta[SUBCOMMANDS] @ git+https://github.com/fedora-copr/rpmeta.git"
```

##### Dependencies

In order for `rpmeta` (and all of its subpackages) installation from PyPI to work directly,
you need to install these dependencies:

- gcc
- krb5-config
- python3-devel

##### Man pages

Since pip cannot distribute UNIX manpages, if you want them available, you need
to install them manually via:

```bash
click-man rpmeta --target <path-to-mandir>/man1
```

---

#### Usage

To see available commands and options, run:

```bash
rpmeta --help
```

For detailed information about a specific (sub)command, run:

```bash
rpmeta <command> --help
```

To see the whole documentation at once, use manpages:

```bash
man 1 rpmeta(-SUBCOMMANDS)?
```
