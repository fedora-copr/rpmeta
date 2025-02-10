%global desc %{expand:
TODO long description
}

Name:           rpmeta
Version:        0.1
Release:        %autorelease
Summary:        TODO short description

License:        GPL-3.0-or-later
URL:            https://github.com/fedora-copr/%{name}
Source0:        %{url}/archive/refs/tags/%{name}-%{version}.tar.gz

BuildArch:      noarch

BuildRequires:  python3-devel
BuildRequires:  python3-rpm-macros

Requires:       python3-click
Requires:       python3-joblib
Requires:       python3-pandas

# TODO: separate into subpackages for server, trainer and fetcher?
# trainer
Recommends:     python3-fedora-distro-aliases
Recommends:     python3-koji
Recommends:     python3-scikit-learn
# copr fetcher
Recommends:     python3-tqdm
# xgboost not in fedora :/

# server
Recommends:     python3-starlette
Recommends:     python3-uvicorn


%description
%{desc}


%prep
%autosetup


%generate_buildrequires
%pyproject_buildrequires -r


%build
%pyproject_wheel


%install
%pyproject_install
%pyproject_save_files %{name}


%files -f %{pyproject_files}
%license LICENSE
%doc README.md
%{_bindir}/%{name}


%changelog
%autochangelog
