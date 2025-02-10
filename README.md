## rpmeta: RPM Estimated Time of (build) Arrival

**RPMeta** is a command-line tool designed to **predict RPM build durations** and manage related
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
# dnf copr enable @copr/rpmeta
# dnf install rpmeta
```

Or from the source:

```bash
pipx install git+https://github.com/fedora-copr/rpmeta.git
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
