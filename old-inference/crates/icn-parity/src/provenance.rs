use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail};
use serde_json::{Map, Value, json};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::digest::{sha256_file, sha256_json};
use crate::model::{BuildInfo, CommandSpec, Component, HostInfo, InvocationKind, Provenance};

pub fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_owned())
}

pub fn new_run_id() -> String {
    let now = OffsetDateTime::now_utc();
    let date = now.date();
    let time = now.time();
    let entropy = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z-{}-{:08x}",
        date.year(),
        u8::from(date.month()),
        date.day(),
        time.hour(),
        time.minute(),
        time.second(),
        std::process::id(),
        (entropy as u64) ^ (entropy >> 64) as u64,
    )
}

pub fn validate_run_id(run_id: &str) -> anyhow::Result<()> {
    if run_id.is_empty()
        || !run_id.as_bytes()[0].is_ascii_alphanumeric()
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        bail!("invalid run id {run_id:?}");
    }
    Ok(())
}

pub fn resolve_program(program: &Path) -> anyhow::Result<PathBuf> {
    if program.is_absolute() || program.components().count() > 1 {
        if !program.is_file() {
            bail!("executable does not exist: {}", program.display());
        }
        return std::fs::canonicalize(program)
            .with_context(|| format!("failed to resolve executable {}", program.display()));
    }
    let path = std::env::var_os("PATH").context("PATH is not set")?;
    for directory in std::env::split_paths(&path) {
        let candidate = directory.join(program);
        if candidate.is_file() {
            return std::fs::canonicalize(&candidate)
                .with_context(|| format!("failed to resolve executable {}", candidate.display()));
        }
    }
    bail!("executable {:?} was not found on PATH", program)
}

pub async fn command_provenance(
    invocation: InvocationKind,
    spec: &CommandSpec,
    resolved_program: &Path,
    effective_configuration: Map<String, Value>,
    extra_components: Vec<Component>,
) -> anyhow::Result<Provenance> {
    let (binary_sha256, _) = sha256_file(resolved_program).await?;
    let mut components = vec![Component {
        kind: match invocation {
            InvocationKind::UpstreamTest | InvocationKind::UpstreamTool => "upstream-binary",
            InvocationKind::NativeOracle => "oracle-binary",
            InvocationKind::IcnProbe => "probe-binary",
        }
        .to_owned(),
        name: resolved_program
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_owned(),
        revision: None,
        tree_sha256: None,
        binary_sha256: Some(binary_sha256),
        dirty: None,
    }];
    components.extend(extra_components);

    if let Ok(current_exe) = std::env::current_exe()
        && current_exe.is_file()
        && let Ok((digest, _)) = sha256_file(&current_exe).await
    {
        components.push(Component {
            kind: "runner".to_owned(),
            name: "icn-parity".to_owned(),
            revision: Some(env!("CARGO_PKG_VERSION").to_owned()),
            tree_sha256: None,
            binary_sha256: Some(digest),
            dirty: None,
        });
    }

    let environment = BTreeMap::from([
        ("clear_env", json!(spec.clear_env)),
        ("cwd", json!(spec.cwd)),
        ("declared_env", json!(spec.env)),
    ]);
    Ok(Provenance {
        components,
        build: BuildInfo {
            build_type: "unknown".to_owned(),
            compiler: "unknown".to_owned(),
            compiler_version: "unknown".to_owned(),
            flags: Vec::new(),
            assertions: None,
            sanitizers: None,
        },
        host: HostInfo {
            os: std::env::consts::OS.to_owned(),
            os_version: None,
            arch: std::env::consts::ARCH.to_owned(),
            cpu: "unreported".to_owned(),
            logical_cpus: std::thread::available_parallelism()
                .ok()
                .map(|count| count.get() as u64),
            memory_bytes: None,
        },
        devices: Vec::new(),
        artifacts: Vec::new(),
        effective_configuration,
        environment_sha256: Some(sha256_json(&environment)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_run_id_is_valid() {
        validate_run_id(&new_run_id()).unwrap();
    }

    #[test]
    fn rejects_path_like_run_ids() {
        assert!(validate_run_id("../escape").is_err());
    }
}
