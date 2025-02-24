%global desc %{expand:
RPMeta is a command-line tool designed to predict RPM build durations and manage related data.
It provides a set of commands for training a predictive model, making predictions, fetching data,
and serving a REST API endpoint.
}

Name:           rpmeta
Version:        0.1.0
Release:        %autorelease
Summary:        Estimate duration of RPM package build

License:        GPL-3.0-or-later
URL:            https://github.com/fedora-copr/%{name}
Source0:        %{url}/archive/refs/tags/%{name}-%{version}.tar.gz

BuildArch:      noarch

BuildRequires:  python3-devel
BuildRequires:  python3-rpm-macros

# test dependencies
BuildRequires:  python3-pytest
BuildRequires:  python3-pytest-cov

Requires:       python3-click
Requires:       python3-joblib
Requires:       python3-pandas

# server
Recommends:     python3-starlette
Recommends:     python3-uvicorn


%description
%{desc}


%package trainer
Summary:        RPMeta trainer module for predictive model training

# test dependencies
BuildRequires: python3-scikit-learn

Requires:       %{name} = %{version}-%{release}
Requires:       python3-fedora-distro-aliases
Requires:       python3-koji
Requires:       python3-scikit-learn
# copr fetcher
Requires:       python3-tqdm
# xgboost not in fedora :/


%description trainer
This package provides the training module of RPMeta, including data processing, data fetchin from
Copr and Koji build systems, and model training.


%package server
Summary:        RPMeta server module for serving REST API endpoint

# test dependencies
BuildRequires: python3-starlette
BuildRequires: python3-uvicorn

Requires:       %{name} = %{version}-%{release}
Requires:       python3-starlette
Requires:       python3-uvicorn


%description server
This package provides the server module of RPMeta, including a REST API endpoint for making
predictions.


%prep
%autosetup


%generate_buildrequires
%pyproject_buildrequires -r


%build
%pyproject_wheel


%install
%pyproject_install
%pyproject_save_files %{name}


%check
# TODO: xgboost is not in fedora, this test/dep needs to be resolved before shipping to fedora
# --enable-net needed for tests for now
pip install xgboost
%pytest


%files -f %{pyproject_files}
%license LICENSE
%doc README.md
%{_bindir}/%{name}


%changelog
%autochangelog
