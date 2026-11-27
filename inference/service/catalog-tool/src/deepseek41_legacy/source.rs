//! Authenticate exactly the files the importer reads, not filenames selected by the manifest.

use std::collections::BTreeMap;
use std::fs::{self, File, Metadata};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result, ensure};
use magnitude_artifacts::FileSource;
use sha2::{Digest, Sha256};

pub(super) const BACKBONE_SHARDS: usize = 4;

pub(super) struct TrustedSources(Vec<Input>);

impl TrustedSources {
    pub(super) fn read(first: &Path, prefix: &str, engram: &Path, manifest: &Path) -> Result<Self> {
        let paths = (1..=BACKBONE_SHARDS)
            .map(|i| first.with_file_name(format!("{prefix}-{i:05}-of-{BACKBONE_SHARDS:05}.gguf")))
            .chain([engram.to_owned()]);
        // Check every journal before opening any GGUF, including journals next
        // to the targets of links (e.g. Hugging Face cache snapshots).
        let inputs = paths.map(Input::inspect).collect::<Result<Vec<_>>>()?;
        let mut contents = String::new();
        let limit = 64 * 1024;
        File::open(manifest)
            .and_then(|file| file.take(limit + 1).read_to_string(&mut contents))
            .with_context(|| format!("read source SHA-256 manifest {}", manifest.display()))?;
        ensure!(
            contents.len() as u64 <= limit,
            "{}: source SHA-256 manifest is too large",
            manifest.display()
        );
        Self::parse(inputs, &contents)
            .with_context(|| format!("source SHA-256 manifest {}", manifest.display()))
    }

    fn parse(mut inputs: Vec<Input>, contents: &str) -> Result<Self> {
        let mut by_name = BTreeMap::new();
        for (index, input) in inputs.iter().enumerate() {
            let basename = input
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .with_context(|| format!("{}: expected a UTF-8 basename", input.path.display()))?;
            ensure!(
                !basename.contains(['\\', '\n', '\r'])
                    && by_name.insert(basename.to_owned(), index).is_none(),
                "ambiguous input basename {basename:?}"
            );
        }
        for (index, line) in contents.lines().enumerate() {
            let (expected, name) = line
                .split_once(' ')
                .and_then(|(sha, rest)| rest.strip_prefix([' ', '*']).map(|name| (sha, name)))
                .filter(|(sha, _)| sha.len() == 64 && sha.bytes().all(|ch| ch.is_ascii_hexdigit()))
                .with_context(|| {
                    format!(
                        "line {}: expected a SHA-256 and basename in sha256sum format",
                        index + 1
                    )
                })?;
            let position = by_name.remove(name).with_context(|| {
                format!(
                    "line {}: {name:?} is not a remaining input basename",
                    index + 1
                )
            })?;
            inputs[position].sha256 = expected.to_ascii_lowercase();
        }
        ensure!(
            by_name.is_empty(),
            "missing SHA-256 for {}",
            by_name.keys().cloned().collect::<Vec<_>>().join(", ")
        );
        Ok(Self(inputs))
    }

    pub(super) fn backbone(&self) -> Vec<PathBuf> {
        self.0[..BACKBONE_SHARDS]
            .iter()
            .map(|input| input.path.clone())
            .collect()
    }

    pub(super) fn check(&self, opened: &[&FileSource]) -> Result<()> {
        ensure!(
            opened.len() == self.0.len(),
            "expected four backbone shards and one Engram file"
        );
        for (input, source) in self.0.iter().zip(opened) {
            input.check(source)?;
        }
        Ok(())
    }

    pub(super) fn authenticate(&self, opened: &[&FileSource]) -> Result<()> {
        self.check(opened)?;
        for (input, source) in self.0.iter().zip(opened) {
            println!("Authenticating {}", input.path.display());
            let actual = format!("{:x}", sha256(source)?);
            ensure!(
                actual == input.sha256,
                "{}: source SHA-256 mismatch (expected {}, got {actual}); get the file from its trusted revision",
                input.path.display(),
                input.sha256
            );
        }
        self.check(opened)
    }
}

struct Input {
    path: PathBuf,
    resolved: PathBuf,
    stamp: (u64, SystemTime),
    sha256: String,
}

impl Input {
    fn inspect(path: PathBuf) -> Result<Self> {
        check_journal(&path)?;
        let resolved =
            fs::canonicalize(&path).with_context(|| format!("resolve input {}", path.display()))?;
        if resolved != path {
            check_journal(&resolved)?;
        }
        let metadata =
            fs::metadata(&resolved).with_context(|| format!("inspect input {}", path.display()))?;
        ensure!(
            metadata.is_file(),
            "{} is not a regular input file",
            path.display()
        );
        let stamp = Self::stamp(&metadata)
            .with_context(|| format!("read input timestamp {}", path.display()))?;
        Ok(Self {
            path,
            resolved,
            stamp,
            sha256: String::new(),
        })
    }

    fn stamp(metadata: &Metadata) -> io::Result<(u64, SystemTime)> {
        Ok((metadata.len(), metadata.modified()?))
    }

    /// Ordinary concurrent edits and link changes must fail the import. This
    /// is not a filesystem lock: inputs must remain immutable during import.
    fn check(&self, source: &FileSource) -> Result<()> {
        let current = Self::inspect(self.path.clone())?;
        let opened = source
            .metadata()
            .with_context(|| format!("inspect opened input {}", source.path().display()))?;
        ensure!(
            current.resolved == self.resolved
                && current.stamp == self.stamp
                && Self::stamp(&opened)? == self.stamp,
            "{} changed during import; keep the original files unchanged and retry",
            self.path.display()
        );
        Ok(())
    }
}

fn check_journal(path: &Path) -> Result<()> {
    let name = path
        .file_name()
        .with_context(|| format!("{} has no file name", path.display()))?;
    let mut journal = std::ffi::OsString::from(".");
    journal.push(name);
    journal.push(".chat-template-recovery");
    let journal = path.with_file_name(journal);
    match fs::symlink_metadata(&journal) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("check recovery journal {}", journal.display()))
        }
        Ok(_) => anyhow::bail!(
            "unfinished chat-template update: {} exists; finish the original update or download a clean copy",
            journal.display()
        ),
    }
}

pub(super) fn sha256(source: &FileSource) -> Result<sha2::digest::Output<Sha256>> {
    let mut digest = Sha256::new();
    source
        .stream(
            0,
            source.size(),
            seismic_io::Want::now(),
            |_, bytes| -> Result<()> {
                digest.update(bytes);
                Ok(())
            },
        )
        .with_context(|| format!("hash {}", source.path().display()))?;
    Ok(digest.finalize())
}
