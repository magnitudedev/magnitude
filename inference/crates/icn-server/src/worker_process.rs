use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use clap::Args;

use crate::installation::Installation;

#[derive(Clone, Debug)]
pub(crate) enum NativeRuntimeAuthority {
    Installation(Installation),
    Development,
}

impl NativeRuntimeAuthority {
    pub(crate) fn installed(installation: Installation) -> Self {
        Self::Installation(installation)
    }

    pub(crate) fn development() -> Self {
        Self::Development
    }

    pub(crate) fn installation(&self) -> Option<&Installation> {
        match self {
            Self::Installation(installation) => Some(installation),
            Self::Development => None,
        }
    }
}

#[derive(Debug, Args)]
pub(crate) struct NativeWorkerArgs {
    #[arg(long)]
    installation: Option<PathBuf>,
    #[arg(long)]
    development_runtime: bool,
}

impl NativeWorkerArgs {
    pub(crate) fn authority(self) -> anyhow::Result<NativeRuntimeAuthority> {
        anyhow::ensure!(
            self.installation.is_some() != self.development_runtime,
            "native worker requires exactly one installation or development runtime authority"
        );
        match self.installation {
            Some(path) => Installation::load(&path).map(NativeRuntimeAuthority::installed),
            None => Ok(NativeRuntimeAuthority::development()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum NativeWorkerRole {
    BackendPreparation,
    Planner,
    Template,
    Inference,
}

impl NativeWorkerRole {
    fn subcommand(self) -> &'static str {
        match self {
            Self::BackendPreparation => "prepare-backend-worker",
            Self::Planner => "plan-worker",
            Self::Template => "template-worker",
            Self::Inference => "inference-worker",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NativeWorkerLauncher {
    authority: NativeRuntimeAuthority,
}

impl NativeWorkerLauncher {
    pub(crate) fn new(authority: NativeRuntimeAuthority) -> Self {
        Self { authority }
    }

    #[cfg(test)]
    pub(crate) fn development() -> Self {
        Self::new(NativeRuntimeAuthority::development())
    }

    pub(crate) fn command(&self, role: NativeWorkerRole) -> anyhow::Result<Command> {
        let executable =
            std::env::current_exe().context("failed to locate ICN worker executable")?;
        let mut command = Command::new(executable);
        command
            .arg(role.subcommand())
            .env("MAGNITUDE_OTEL", "0")
            .env("RUST_LOG", "error")
            .env_remove("MAGNITUDE_OTEL_ENDPOINT")
            .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
            .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            .env_remove("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT");
        match &self.authority {
            NativeRuntimeAuthority::Installation(installation) => {
                command
                    .arg("--installation")
                    .arg(installation.declaration_path());
            }
            NativeRuntimeAuthority::Development => {
                command.arg("--development-runtime");
            }
        }
        Ok(command)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use icn_contracts::bootstrap_protocol::{IcnInstallationBackend, IcnInstallationDeclaration};

    use super::{NativeWorkerArgs, NativeWorkerLauncher, NativeWorkerRole};

    #[test]
    fn worker_runtime_authority_is_explicit() {
        assert!(
            NativeWorkerArgs {
                installation: None,
                development_runtime: false,
            }
            .authority()
            .is_err()
        );
        assert!(
            NativeWorkerArgs {
                installation: None,
                development_runtime: true,
            }
            .authority()
            .is_ok()
        );
        assert!(
            NativeWorkerArgs {
                installation: Some("installation.json".into()),
                development_runtime: true,
            }
            .authority()
            .is_err()
        );
    }

    #[test]
    fn every_worker_role_uses_the_same_runtime_command_boundary() {
        let launcher = NativeWorkerLauncher::development();
        for (role, expected_subcommand) in [
            (
                NativeWorkerRole::BackendPreparation,
                "prepare-backend-worker",
            ),
            (NativeWorkerRole::Planner, "plan-worker"),
            (NativeWorkerRole::Template, "template-worker"),
            (NativeWorkerRole::Inference, "inference-worker"),
        ] {
            let command = launcher.command(role).expect("worker command");
            assert_eq!(
                command
                    .get_args()
                    .map(|argument| argument.to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
                vec![expected_subcommand, "--development-runtime"]
            );
        }
    }

    #[test]
    fn installed_worker_command_uses_the_verified_declaration() {
        let root = tempfile::tempdir().expect("installation root");
        for directory in ["bin", "catalog", "runtime", "backends"] {
            fs::create_dir(root.path().join(directory)).expect("installation directory");
        }
        for (path, contents) in [
            (
                root.path().join("bin").join(if cfg!(windows) {
                    "magnitude-icn.exe"
                } else {
                    "magnitude-icn"
                }),
                b"executable".as_slice(),
            ),
            (
                root.path().join("catalog/model-planner-inputs.bundle"),
                b"planner".as_slice(),
            ),
            (
                root.path().join("backends/backend-cpu"),
                b"backend".as_slice(),
            ),
        ] {
            fs::write(path, contents).expect("installation file");
        }
        let declaration_path = root.path().join("installation.json");
        fs::write(
            &declaration_path,
            serde_json::to_vec(&IcnInstallationDeclaration {
                schema_version: 1,
                backend: IcnInstallationBackend::Cpu,
                native_build: "native-build".to_owned(),
                backend_module_abi: "backend-abi".to_owned(),
            })
            .expect("serialize declaration"),
        )
        .expect("installation declaration");

        let authority = NativeWorkerArgs {
            installation: Some(declaration_path),
            development_runtime: false,
        }
        .authority()
        .expect("installed authority");
        let installation_path = authority
            .installation()
            .expect("installed runtime")
            .declaration_path();
        let command = NativeWorkerLauncher::new(authority)
            .command(NativeWorkerRole::Inference)
            .expect("worker command");

        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            vec![
                std::ffi::OsStr::new("inference-worker"),
                std::ffi::OsStr::new("--installation"),
                installation_path.as_os_str(),
            ]
        );
    }
}
