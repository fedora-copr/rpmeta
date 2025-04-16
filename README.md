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

On Fedora:

```bash
dnf copr enable @copr/rpmeta
dnf install rpmeta
```

Or from the source:

```bash
pipx install git+https://github.com/fedora-copr/rpmeta.git
```

However, since pip cannot distribute UNIX manpages, if you want them available, you need
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

For detailed information about a specific command, run:

```bash
rpmeta <command> --help
```

To see the whole documentation at once, use manpages:

```bash
man 1 rpmeta(-SUBCOMMANDS)?
```
