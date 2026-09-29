//! Installation startup checks.

use anyhow::Context;
use magnitude_service_models::{ReleaseCatalog, load_release_catalog};

use magnitude_service_server::build_identity;

use crate::installation::Installation;

/// Verify that the running executable is the installation's and was built as the engine build the
/// installation declares, then load its release catalog.
pub(crate) fn open_installation_catalog(
    installation: &Installation,
) -> anyhow::Result<ReleaseCatalog> {
    anyhow::ensure!(
        installation.native_build() == build_identity::native_build(),
        "ICN installation native build does not match its executable"
    );
    let declared = installation
        .executable()
        .canonicalize()
        .context("failed to resolve the declared ICN executable")?;
    let running = std::env::current_exe()?
        .canonicalize()
        .context("failed to resolve the running ICN executable")?;
    anyhow::ensure!(
        declared == running,
        "running executable is not part of the declared ICN installation"
    );
    load_release_catalog(&installation.planner_bundle())
        .context("failed to load the release planner bundle")
}
