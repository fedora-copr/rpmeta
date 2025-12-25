%global desc %{expand:
RPMeta is a command-line tool designed to predict RPM build durations and manage related data.
It provides a set of commands for training a predictive model, making predictions, fetching data,
and serving a REST API endpoint.
}

%global srcname rpmeta

%if 0%{?git_build}
%global pkg_name %{srcname}-git
%else
%global pkg_name %{srcname}
%endif

Name:           %pkg_name
Version:        0.1.0
Release:        %autorelease
Summary:        Estimate duration of RPM package build

License:        GPL-3.0-or-later
URL:            https://github.com/fedora-copr/rpmeta
Source0:        %{url}/archive/refs/tags/%{srcname}-%{version}.tar.gz
Source1:        95-%{srcname}.preset

BuildArch:      noarch

BuildRequires:  python3-devel
BuildRequires:  python3-rpm-macros
BuildRequires:  python3dist(click-man)
BuildRequires:  systemd-rpm-macros

# prevent having both
%if 0%{?git_build}
Conflicts:      %{srcname}
%else
Conflicts:      %{srcname}-git
%endif


%description
%{desc}
%if 0%{?git_build}

This is a development build from the main branch.
%endif


# automatic creation of subpackages
# xgboost nor any other boosting algorithm is packaged to fedora
%pyproject_extras_subpkg -n %{name} trainer
%pyproject_extras_subpkg -n %{name} fetcher


%package -n     %{name}+server
Summary:        RPMeta server module for serving REST API endpoint
Requires:       python3dist(fastapi)
Requires:       python3dist(uvicorn)
Requires:       %{name} = %{version}-%{release}
%{?systemd_requires}

%description -n %{name}+server
This package provides the server module of RPMeta, including a REST API endpoint for making
predictions. It includes a systemd service for running the RPMeta server as a background service.


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
%pyproject_save_files %{srcname}

install -D -m 644 -p files/config.toml.example %{buildroot}%{_sysconfdir}/%{srcname}/config.toml.example
install -D -m 644 -p files/rpmeta.service %{buildroot}%{_unitdir}/rpmeta.service
install -D -m 644 -p files/rpmeta.env %{buildroot}%{_sysconfdir}/sysconfig/rpmeta

install -D -m 644 %{S:1} %{buildroot}%{_presetdir}/95-%{srcname}.preset

# Create sysusers.d config file inline
install -d %{buildroot}%{_sysusersdir}
cat > %{buildroot}%{_sysusersdir}/%{srcname}.conf <<EOF
# Type  User    ID  GECOS                     Home dir
u       rpmeta  -   "RPMeta Service Account"  /var/lib/rpmeta
EOF

# Create tmpfiles.d config file inline
install -d %{buildroot}%{_tmpfilesdir}
cat > %{buildroot}%{_tmpfilesdir}/%{srcname}.conf <<EOF
# Type  Path                      Mode  User    Group   Age  Argument
d       /var/lib/rpmeta           2775  rpmeta  rpmeta  -    -
d       /var/lib/rpmeta/models    2775  rpmeta  rpmeta  -    -
d       /var/lib/rpmeta/pylibs    2775  rpmeta  rpmeta  -    -
f       /var/log/rpmeta.log       0644  rpmeta  rpmeta  -    -
EOF

# generate man 1 pages
PYTHONPATH="%{buildroot}%{python3_sitelib}" click-man %{srcname} --target %{buildroot}%{_mandir}/man1


%files -f %{pyproject_files}
%license LICENSE
%doc README.md
%{_mandir}/man1/%{srcname}*.1*
%{_bindir}/%{srcname}
%config(noreplace) %{_sysconfdir}/%{srcname}/config.toml.example
# This is not part of the main package, but of the subpackages
%exclude %{python3_sitelib}/rpmeta/server/


%files -n %{name}+server
%{python3_sitelib}/rpmeta/server/
%{_unitdir}/rpmeta.service
%{_sysusersdir}/%{srcname}.conf
%{_tmpfilesdir}/%{srcname}.conf
%{_presetdir}/95-%{srcname}.preset
%config(noreplace) %{_sysconfdir}/sysconfig/rpmeta


%post -n %{name}+server
%systemd_post rpmeta.service

%preun -n %{name}+server
%systemd_preun rpmeta.service

%postun -n %{name}+server
%systemd_postun_with_restart rpmeta.service


%changelog
%autochangelog
