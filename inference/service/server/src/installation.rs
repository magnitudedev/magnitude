use std::fs;
use std::path::{Path, PathBuf};

use magnitude_service_contracts::bootstrap_protocol::IcnInstallationDeclaration;

const MAX_DECLARATION_BYTES: u64 = 64 * 1024;

pub(crate) fn executable_name() -> String {
    format!("{}{}", env!("CARGO_BIN_NAME"), std::env::consts::EXE_SUFFIX)
}

/// A verified installation: `installation.json`, `bin/`, the model planner inputs and, where the
/// host's engine loads NVRTC, `runtime/` (integration spec §11.1).
#[derive(Clone, Debug)]
pub(crate) struct Installation {
    root: PathBuf,
    declaration: IcnInstallationDeclaration,
}

impl Installation {
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        if path.file_name().and_then(|name| name.to_str()) != Some("installation.json") {
            anyhow::bail!("ICN installation declaration must be named installation.json");
        }
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.len() > MAX_DECLARATION_BYTES
        {
            anyhow::bail!("ICN installation declaration is not a bounded regular file");
        }
        let declaration: IcnInstallationDeclaration = serde_json::from_slice(&fs::read(path)?)?;
        if declaration.schema_version != 1 || declaration.native_build.trim().is_empty() {
            anyhow::bail!("ICN installation declaration is incomplete");
        }
        let root = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("ICN installation has no root"))?
            .canonicalize()?;
        let installation = Self { root, declaration };
        installation.validate_layout()?;
        Ok(installation)
    }

    pub(crate) fn native_build(&self) -> &str {
        &self.declaration.native_build
    }

    pub(crate) fn executable(&self) -> PathBuf {
        self.root.join("bin").join(executable_name())
    }

    pub(crate) fn planner_bundle(&self) -> PathBuf {
        self.root.join("catalog/model-planner-inputs.bundle")
    }

    fn validate_layout(&self) -> anyhow::Result<()> {
        for (label, path) in [
            ("executable", self.executable()),
            ("model planner inputs", self.planner_bundle()),
        ] {
            let metadata = fs::symlink_metadata(path)?;
            if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() == 0 {
                anyhow::bail!("ICN installation {label} is not a non-empty regular file");
            }
        }
        if cfg!(any(target_os = "linux", windows)) {
            let metadata = fs::symlink_metadata(self.root.join("runtime"))?;
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                anyhow::bail!("ICN installation runtime directory is invalid");
            }
        }
        Ok(())
    }
}
