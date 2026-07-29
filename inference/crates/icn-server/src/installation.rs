use std::fs;
use std::path::{Path, PathBuf};

use icn_contracts::bootstrap_protocol::{IcnInstallationBackend, IcnInstallationDeclaration};

const MAX_DECLARATION_BYTES: u64 = 64 * 1024;

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
        if declaration.schema_version != 1
            || declaration.native_build.trim().is_empty()
            || declaration.backend_module_abi.trim().is_empty()
        {
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

    pub(crate) fn backend(&self) -> IcnInstallationBackend {
        self.declaration.backend
    }

    pub(crate) fn native_build(&self) -> &str {
        &self.declaration.native_build
    }

    pub(crate) fn backend_module_abi(&self) -> &str {
        &self.declaration.backend_module_abi
    }

    pub(crate) fn executable(&self) -> PathBuf {
        self.root.join("bin").join(if cfg!(windows) {
            "magnitude-icn.exe"
        } else {
            "magnitude-icn"
        })
    }

    pub(crate) fn catalog_lock(&self) -> PathBuf {
        self.root.join("catalog/release-catalog.lock.json")
    }

    pub(crate) fn planner_bundle(&self) -> PathBuf {
        self.root.join("catalog/model-planner-inputs.bundle")
    }

    pub(crate) fn backend_directory(&self) -> PathBuf {
        self.root.join("backends")
    }

    fn validate_layout(&self) -> anyhow::Result<()> {
        for (label, path) in [
            ("executable", self.executable()),
            ("catalog lock", self.catalog_lock()),
            ("model planner inputs", self.planner_bundle()),
        ] {
            let metadata = fs::symlink_metadata(path)?;
            if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() == 0 {
                anyhow::bail!("ICN installation {label} is not a non-empty regular file");
            }
        }
        for directory in [self.root.join("runtime"), self.backend_directory()] {
            let metadata = fs::symlink_metadata(directory)?;
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                anyhow::bail!("ICN installation contains an invalid directory");
            }
        }
        let names = fs::read_dir(self.backend_directory())?
            .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().to_lowercase()))
            .collect::<Result<Vec<_>, _>>()?;
        if !names.iter().any(|name| name.contains("cpu")) {
            anyhow::bail!("ICN installation has no CPU backend module");
        }
        for backend in ["metal", "cuda", "vulkan"] {
            let present = names.iter().any(|name| name.contains(backend));
            if present != (self.backend().name() == backend) {
                anyhow::bail!(
                    "ICN installation backend directory does not enforce selected {} policy",
                    self.backend().name()
                );
            }
        }
        Ok(())
    }
}
