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
Source1:        95-rpmeta.preset

BuildArch:      noarch

BuildRequires:  python3-devel
BuildRequires:  python3-rpm-macros
BuildRequires:  python3dist(click-man)
BuildRequires:  systemd-rpm-macros


%description
%{desc}


%package -n     server
Summary:        RPMeta server module for serving REST API endpoint
%systemd_requires

%description -n server
This package provides the server module of RPMeta, including a REST API endpoint for making
predictions. It includes a systemd service for running the RPMeta server as a background service.

%pyproject_extras_subpkg -n %{name} server


# xgboost nor any other boosting algorithm is packaged to fedora
%package -n     trainer
Summary:        RPMeta trainer module for predictive model training

%description -n trainer
This package provides the training module of RPMeta, including data processing, data fetchin from
Copr and Koji build systems, and model training.

%pyproject_extras_subpkg -n %{name} trainer


%package -n     fetcher
Summary:        RPMeta fetcher module for data fetching

%description -n fetcher
This package provides the fetcher module of RPMeta, including data fetching from Copr and Koji.

%pyproject_extras_subpkg -n %{name} fetcher


%prep
%autosetup
# boosting models like xgboost and ligthgbm are not packaged in fedora
# the same goes for the kaleido, tool optuna uses for generating fancy graphs
# if user want's to use this, they have to install it via other package manager (e.g. pipx)
sed -i '/xgboost>=/d' pyproject.toml
sed -i '/lightgbm>=/d' pyproject.toml
sed -i '/kaleido==/d' pyproject.toml


%generate_buildrequires
%pyproject_buildrequires -r -x server -x fetcher -x trainer


%build
%pyproject_wheel


%install
%pyproject_install
%pyproject_save_files %{name} "" server fetcher trainer

install -D -m 644 -p files/config.toml.example %{buildroot}%{_sysconfdir}/%{name}/config.toml.example
install -D -m 644 -p files/rpmeta.service %{buildroot}%{_unitdir}/rpmeta.service

install -D -m 644 %{S:1} %{buildroot}%{_presetdir}/95-%{name}.preset

# Create sysusers.d config file inline
install -d %{buildroot}%{_sysusersdir}
cat > %{buildroot}%{_sysusersdir}/%{name}.conf <<EOF
# Type  User    ID  GECOS                     Home dir
u       rpmeta  -   "RPMeta Service Account"  /var/lib/rpmeta
EOF

# Create tmpfiles.d config file inline
install -d %{buildroot}%{_tmpfilesdir}
cat > %{buildroot}%{_tmpfilesdir}/%{name}.conf <<EOF
# Type  Path                      Mode  User    Group   Age  Argument
d       /var/lib/rpmeta           0755  rpmeta  rpmeta  -    -
d       /var/lib/rpmeta/models    0755  rpmeta  rpmeta  -    -
Z       /var/log/rpmeta.log       0644  rpmeta  rpmeta  -    -
EOF

# generate man 1 pages
PYTHONPATH="%{buildroot}%{python3_sitelib}" click-man %{name} --target %{buildroot}%{_mandir}/man1


%files -f %{pyproject_files}
%license LICENSE
%doc README.md
%{_mandir}/man1/%{name}*.1*
%{_bindir}/%{name}
%config(noreplace) %{_sysconfdir}/%{name}/config.toml.example


%files -n server -f %{pyproject_files_server}
%{_unitdir}/rpmeta.service
%{_sysusersdir}/%{name}.conf
%{_tmpfilesdir}/%{name}.conf
%{_presetdir}/95-%{name}.preset


%files -n trainer -f %{pyproject_files_trainer}


%files -n fetcher -f %{pyproject_files_fetcher}


%changelog
%autochangelog
