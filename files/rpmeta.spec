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
BuildRequires:  python3dist(click-man)


%description
%{desc}


%package -n     server
Summary:        RPMeta server module for serving REST API endpoint
Requires:       %{name} = %{version}-%{release}

%description -n server
This package provides the server module of RPMeta, including a REST API endpoint for making
predictions.

%pyproject_extras_subpkg -n %{name} server


# xgboost nor any other boosting algorithm is packaged to fedora
%package -n     trainer
Summary:        RPMeta trainer module for predictive model training
Requires:       %{name} = %{version}-%{release}

%description -n trainer
This package provides the training module of RPMeta, including data processing, data fetchin from
Copr and Koji build systems, and model training.

%pyproject_extras_subpkg -n %{name} trainer


%package -n     fetcher
Summary:        RPMeta fetcher module for data fetching
Requires:       %{name} = %{version}-%{release}

%description -n fetcher
This package provides the fetcher module of RPMeta, including data fetching from Copr and Koji.

%pyproject_extras_subpkg -n %{name} fetcher


%prep
%autosetup
# boosting models like xgboost and ligthgbm are not packaged in fedora
# the same goes for the kaleido, tool optuna uses for generating fancy graphs
# if user want's to use this, they have to install it via other package manager (e.g. pipx)
sed -i "/xgboost>=2.0.0/d" pyproject.toml
sed -i '/lightgbm>=4.0.0/d' pyproject.toml
sed -i '/kaleido==0.2.1/d' pyproject.toml


%generate_buildrequires
%pyproject_buildrequires -r -x server -x fetcher -x trainer


%build
%pyproject_wheel


%install
%pyproject_install
%pyproject_save_files %{name}

# generate man 1 pages
PYTHONPATH="%{buildroot}%{python3_sitelib}" click-man %{name} --target %{buildroot}%{_mandir}/man1


%files -f %{pyproject_files}
%license LICENSE
%doc README.md
%{_mandir}/man1/%{name}*.1*
%{_bindir}/%{name}


%changelog
%autochangelog
