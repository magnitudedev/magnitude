//! A GGUF read from a local path or from a pinned Hugging Face revision: every shard's header,
//! and tensor bytes by range, through the model store's Hub client when it is remote.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use clap::Subcommand;
use magnitude_artifacts::gguf::{
    Directory, TensorDescriptor, inspect_header, read_header, split_paths,
};
use magnitude_service_contracts::inventory::{
    HuggingFaceRepositoryRequest, HuggingFaceRepositorySnapshot,
};
use magnitude_service_models::{ManagedModelStore, pinned_hugging_face_repository};

use crate::StoreArgs;

/// Where a GGUF file is. Only a Hub file is read through a model store.
#[derive(Debug, Subcommand)]
pub enum GgufSource {
    /// A local file (the first shard of a split GGUF).
    Local { path: PathBuf },
    /// A file of a Hugging Face repository at a commit, read by range requests.
    Hub {
        #[arg(long)]
        repository: String,
        /// An immutable commit.
        #[arg(long)]
        revision: String,
        /// The file's path in the repository (the first shard of a split GGUF).
        #[arg(long)]
        path: PathBuf,
        #[command(flatten)]
        store: StoreArgs,
    },
}

/// The pinned repository snapshot of `repository` at the commit `revision`.
pub async fn pinned_snapshot(
    store: &Arc<ManagedModelStore>,
    repository: &str,
    revision: &str,
) -> anyhow::Result<HuggingFaceRepositorySnapshot> {
    let snapshot = pinned_hugging_face_repository(
        store,
        HuggingFaceRepositoryRequest {
            repository: repository.to_owned(),
            revision: revision.to_owned(),
        },
    )
    .await
    .with_context(|| format!("failed to read {repository}@{revision}"))?;
    anyhow::ensure!(
        snapshot.commit == revision,
        "{repository}@{revision} is not an immutable commit (it is {})",
        snapshot.commit
    );
    Ok(snapshot)
}

/// Every byte of the Hub file `path` of `snapshot` through its aligned tensor-data offset.
pub async fn hub_header(
    store: &ManagedModelStore,
    snapshot: &HuggingFaceRepositorySnapshot,
    path: &Path,
) -> anyhow::Result<Vec<u8>> {
    let file = snapshot
        .gguf_files
        .iter()
        .find(|file| file.path == path)
        .with_context(|| format!("{} has no {}", snapshot.repository, path.display()))?;
    Ok(store
        .hugging_face_gguf_header(&snapshot.repository, &snapshot.commit, file, &[])
        .await
        .with_context(|| format!("failed to read the header of {}", path.display()))?
        .bytes)
}

enum Location {
    Local,
    Hub {
        store: Arc<ManagedModelStore>,
        snapshot: HuggingFaceRepositorySnapshot,
    },
}

/// One file of a [`Gguf`]: its path (local, or inside the repository), its header bytes through
/// the aligned tensor-data offset, and the directory they hold.
pub struct Shard {
    pub path: PathBuf,
    pub header: Vec<u8>,
    pub directory: Directory,
}

/// A GGUF file, or every shard of a split GGUF named by its first shard: the first shard's
/// metadata and every shard's tensors.
pub struct Gguf {
    location: Location,
    pub shards: Vec<Shard>,
    tensors: HashMap<String, usize>,
}

impl Gguf {
    pub async fn open(source: &GgufSource) -> anyhow::Result<Self> {
        let (location, first) = match source {
            GgufSource::Local { path } => (Location::Local, path.clone()),
            GgufSource::Hub {
                repository,
                revision,
                path,
                store,
            } => {
                let store = store.open().await?;
                let snapshot = pinned_snapshot(&store, repository, revision).await?;
                (Location::Hub { store, snapshot }, path.clone())
            }
        };
        let mut gguf = Self {
            location,
            shards: Vec::new(),
            tensors: HashMap::new(),
        };
        let shard = gguf.read_shard(first.clone()).await?;
        let paths = split_paths(&first, &shard.directory)?;
        gguf.add(shard)?;
        for path in paths.into_iter().flatten().skip(1) {
            let shard = gguf.read_shard(path).await?;
            gguf.add(shard)?;
        }
        Ok(gguf)
    }

    async fn read_shard(&self, path: PathBuf) -> anyhow::Result<Shard> {
        let (header, directory) = match &self.location {
            Location::Local => {
                let directory = inspect_header(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                let mut header = Vec::new();
                std::fs::File::open(&path)?
                    .take(directory.data_offset)
                    .read_to_end(&mut header)?;
                (header, directory)
            }
            Location::Hub { store, snapshot } => {
                let header = hub_header(store, snapshot, &path).await?;
                let directory = read_header(&header)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                (header, directory)
            }
        };
        Ok(Shard {
            path,
            header,
            directory,
        })
    }

    fn add(&mut self, shard: Shard) -> anyhow::Result<()> {
        let index = self.shards.len();
        for tensor in &shard.directory.tensors {
            anyhow::ensure!(
                self.tensors.insert(tensor.name.clone(), index).is_none(),
                "tensor {} appears in more than one shard",
                tensor.name
            );
        }
        self.shards.push(shard);
        Ok(())
    }

    /// The first shard's directory, whose metadata is the GGUF's.
    pub fn first(&self) -> &Directory {
        &self.shards[0].directory
    }

    /// Every tensor, shard by shard in stored order.
    pub fn tensors(&self) -> impl Iterator<Item = &TensorDescriptor> {
        self.shards
            .iter()
            .flat_map(|shard| &shard.directory.tensors)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        let shard = &self.shards[*self.tensors.get(name)?];
        shard.directory.tensor(name)
    }

    /// Each shard as provenance names it: `REPO@REVISION/PATH`, or its local path.
    pub fn describe(&self) -> Vec<String> {
        self.shards
            .iter()
            .map(|shard| match &self.location {
                Location::Local => shard.path.display().to_string(),
                Location::Hub { snapshot, .. } => format!(
                    "{}@{}/{}",
                    snapshot.repository,
                    snapshot.commit,
                    shard.path.display()
                ),
            })
            .collect()
    }

    /// The stored bytes of the tensor `name`.
    pub async fn tensor_bytes(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        let shard = &self.shards[*self
            .tensors
            .get(name)
            .with_context(|| format!("no tensor {name}"))?];
        let tensor = shard
            .directory
            .tensor(name)
            .expect("an indexed tensor is in its shard");
        let start = shard.directory.data_offset + tensor.offset;
        let end = start + tensor.nbytes;
        match &self.location {
            Location::Local => {
                let mut file = std::fs::File::open(&shard.path)?;
                file.seek(SeekFrom::Start(start))?;
                let mut bytes = vec![0; usize::try_from(tensor.nbytes)?];
                file.read_exact(&mut bytes)
                    .with_context(|| format!("{} ends inside {name}", shard.path.display()))?;
                Ok(bytes)
            }
            Location::Hub { store, snapshot } => Ok(store
                .hub_file_range(snapshot, &shard.path, start..end)
                .await
                .with_context(|| format!("failed to read {name}"))?),
        }
    }
}
