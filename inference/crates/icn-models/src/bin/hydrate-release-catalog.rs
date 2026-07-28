use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use clap::Parser;
use futures_util::{StreamExt, TryStreamExt, stream};
use icn_models::{
    catalog_source_digest_from, encode_release_planner_bundle_with_progress,
};
use reqwest::StatusCode;
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_RANGE, RANGE, USER_AGENT};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const LOCK_NAME: &str = "release-catalog.lock.json";
const BUNDLE_NAME: &str = "model-planner-inputs.bundle";
const MAX_LOCK_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SOURCE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    lock: PathBuf,
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    cache: Option<PathBuf>,
    #[arg(long)]
    offline: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Lock {
    source_digest: String,
    planner_bundle_digest: String,
    planner_artifacts: Vec<Artifact>,
}

#[derive(Deserialize)]
struct Artifact {
    source: Source,
    components: Vec<Component>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Source {
    HuggingFace {
        repository: String,
        commit: String,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Component {
    component: ComponentPath,
    header_digest: String,
    header_size_bytes: u64,
}

#[derive(Deserialize)]
struct ComponentPath {
    path: PathBuf,
}

#[derive(Clone)]
struct HeaderSource {
    repository: String,
    commit: String,
    path: PathBuf,
    bytes: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let arguments = Arguments::parse();
    let lock_bytes = read_bounded(&arguments.lock, MAX_LOCK_BYTES).await?;
    let source_bytes = read_bounded(&arguments.source, MAX_SOURCE_BYTES).await?;
    let lock: Lock = serde_json::from_slice(&lock_bytes)?;
    if catalog_source_digest_from(&source_bytes).map_err(anyhow::Error::msg)?
        != lock.source_digest
    {
        anyhow::bail!(
            "models.json differs from release-catalog.lock.json; run `bun icn:catalog:update`"
        );
    }

    if valid_output(&arguments.output, &lock_bytes, &lock.planner_bundle_digest).await {
        eprintln!("Model catalog data is already hydrated.");
        return Ok(());
    }
    let cached = arguments
        .cache
        .as_ref()
        .map(|root| root.join(&lock.planner_bundle_digest));
    let bundle = if let Some(cache) = &cached
        && let Some(bytes) = read_valid_bundle(cache, &lock.planner_bundle_digest).await
    {
        eprintln!("Restoring model catalog data from the local cache...");
        bytes
    } else {
        if arguments.offline {
            anyhow::bail!(
                "catalog bundle {} is unavailable offline; run `bun icn:catalog:hydrate` while online",
                lock.planner_bundle_digest
            );
        }
        let sources = header_sources(&lock)?;
        eprintln!(
            "Downloading {} model catalog file{} from Hugging Face...",
            sources.len(),
            if sources.len() == 1 { "" } else { "s" }
        );
        let headers = download_headers(sources).await?;
        eprintln!("Encoding model catalog bundle...");
        let bytes = encode_release_planner_bundle_with_progress(&headers, |completed, total| {
            report_progress("Encoded model catalog files", completed, total);
        })
        .map_err(anyhow::Error::msg)?;
        require_digest(&bytes, &lock.planner_bundle_digest)?;
        if let Some(cache) = &cached {
            publish_pair(cache, &lock_bytes, &bytes).await?;
        }
        bytes
    };
    publish_pair(&arguments.output, &lock_bytes, &bundle).await?;
    if !valid_output(&arguments.output, &lock_bytes, &lock.planner_bundle_digest).await {
        anyhow::bail!("catalog hydration did not publish valid output");
    }
    println!("{}", arguments.output.display());
    Ok(())
}

fn header_sources(lock: &Lock) -> anyhow::Result<BTreeMap<String, HeaderSource>> {
    let mut sources = BTreeMap::new();
    for artifact in &lock.planner_artifacts {
        let Source::HuggingFace {
            repository,
            commit,
        } = &artifact.source;
        for component in &artifact.components {
            if component.header_size_bytes == 0
                || component.header_size_bytes > MAX_HEADER_BYTES
                || component.header_digest.len() != 64
                || !component
                    .header_digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
            {
                anyhow::bail!("catalog contains an invalid planner header declaration");
            }
            let source = HeaderSource {
                repository: repository.clone(),
                commit: commit.clone(),
                path: component.component.path.clone(),
                bytes: component.header_size_bytes,
            };
            if let Some(previous) = sources.insert(component.header_digest.clone(), source.clone())
                && (previous.repository != source.repository
                    || previous.commit != source.commit
                    || previous.path != source.path
                    || previous.bytes != source.bytes)
            {
                anyhow::bail!("catalog reuses a planner digest for different sources");
            }
        }
    }
    Ok(sources)
}

async fn download_headers(
    sources: BTreeMap<String, HeaderSource>,
) -> anyhow::Result<BTreeMap<String, Vec<u8>>> {
    let total = sources.len();
    let completed = Arc::new(AtomicUsize::new(0));
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .build()?;
    let token = std::env::var("HF_TOKEN")
        .ok()
        .filter(|value| !value.is_empty());
    let values = stream::iter(sources)
        .map(|(digest, source)| {
            let client = client.clone();
            let token = token.clone();
            let completed = Arc::clone(&completed);
            async move {
                let bytes = download_with_retry(&client, token.as_deref(), &source).await?;
                require_digest(&bytes, &digest)?;
                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                report_progress("Downloaded model catalog files", count, total);
                Ok::<_, anyhow::Error>((digest, bytes))
            }
        })
        .buffer_unordered(8)
        .try_collect::<Vec<_>>()
        .await?;
    Ok(values.into_iter().collect())
}

fn report_progress(stage: &str, completed: usize, total: usize) {
    if completed == 1 || completed.is_multiple_of(5) || completed == total {
        eprintln!("{stage}: {completed}/{total}");
    }
}

async fn download_with_retry(
    client: &reqwest::Client,
    token: Option<&str>,
    source: &HeaderSource,
) -> anyhow::Result<Vec<u8>> {
    let mut last = None;
    for attempt in 0..3 {
        match download_once(client, token, source).await {
            Ok(bytes) => return Ok(bytes),
            Err(error) => last = Some(error),
        }
        if attempt < 2 {
            tokio::time::sleep(Duration::from_millis(500 * (1 << attempt))).await;
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("catalog download failed")))
}

async fn download_once(
    client: &reqwest::Client,
    token: Option<&str>,
    source: &HeaderSource,
) -> anyhow::Result<Vec<u8>> {
    let mut request = client
        .get(source_url(source)?)
        .header(USER_AGENT, "magnitude-catalog-hydrator")
        .header(RANGE, format!("bytes=0-{}", source.bytes - 1));
    if let Some(token) = token {
        request = request.header(AUTHORIZATION, format!("Bearer {token}"));
    }
    let response = request.send().await?;
    if response.status() != StatusCode::PARTIAL_CONTENT {
        anyhow::bail!(
            "Hugging Face returned HTTP {} for {}",
            response.status(),
            source.path.display()
        );
    }
    let expected_range = format!("bytes 0-{}/", source.bytes - 1);
    if !response
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with(&expected_range))
        || response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            != Some(source.bytes)
    {
        anyhow::bail!("Hugging Face returned an invalid byte range");
    }
    let expected = usize::try_from(source.bytes)?;
    let mut bytes = Vec::with_capacity(expected);
    let mut body = response.bytes_stream();
    while let Some(chunk) = body.next().await {
        let chunk = chunk?;
        if bytes.len().saturating_add(chunk.len()) > expected {
            anyhow::bail!("Hugging Face returned an oversized planner header");
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.len() != expected {
        anyhow::bail!("Hugging Face returned a short planner header");
    }
    Ok(bytes)
}

fn source_url(source: &HeaderSource) -> anyhow::Result<reqwest::Url> {
    let mut url = reqwest::Url::parse("https://huggingface.co")?;
    let mut segments = url
        .path_segments_mut()
        .map_err(|()| anyhow::anyhow!("invalid Hugging Face base URL"))?;
    segments.extend(source.repository.split('/'));
    segments.push("resolve");
    segments.push(&source.commit);
    for component in source.path.components() {
        segments.push(
            component
                .as_os_str()
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("catalog path is not UTF-8"))?,
        );
    }
    drop(segments);
    Ok(url)
}

async fn read_bounded(path: &Path, maximum: u64) -> anyhow::Result<Vec<u8>> {
    let metadata = tokio::fs::symlink_metadata(path).await?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > maximum {
        anyhow::bail!("{} is not a bounded regular file", path.display());
    }
    Ok(tokio::fs::read(path).await?)
}

async fn read_valid_bundle(directory: &Path, digest: &str) -> Option<Vec<u8>> {
    let lock = read_bounded(&directory.join(LOCK_NAME), MAX_LOCK_BYTES)
        .await
        .ok()?;
    serde_json::from_slice::<Lock>(&lock).ok()?;
    let bundle = read_bounded(&directory.join(BUNDLE_NAME), 2 * 1024 * 1024 * 1024)
        .await
        .ok()?;
    require_digest(&bundle, digest).ok()?;
    Some(bundle)
}

async fn valid_output(directory: &Path, lock: &[u8], digest: &str) -> bool {
    let Some(bundle) = read_valid_bundle(directory, digest).await else {
        return false;
    };
    let actual_lock = tokio::fs::read(directory.join(LOCK_NAME)).await.ok();
    actual_lock.as_deref() == Some(lock) && require_digest(&bundle, digest).is_ok()
}

async fn publish_pair(directory: &Path, lock: &[u8], bundle: &[u8]) -> anyhow::Result<()> {
    let parent = directory
        .parent()
        .ok_or_else(|| anyhow::anyhow!("catalog output has no parent"))?;
    tokio::fs::create_dir_all(parent).await?;
    let suffix = format!("{}-{}", std::process::id(), monotonic_suffix());
    let staging = parent.join(format!(".catalog-{suffix}"));
    tokio::fs::create_dir(&staging).await?;
    tokio::fs::write(staging.join(LOCK_NAME), lock).await?;
    tokio::fs::write(staging.join(BUNDLE_NAME), bundle).await?;
    if directory.exists() {
        let stale = parent.join(format!(".catalog-stale-{suffix}"));
        tokio::fs::rename(directory, &stale).await?;
        if let Err(error) = tokio::fs::rename(&staging, directory).await {
            let _ = tokio::fs::rename(&stale, directory).await;
            return Err(error.into());
        }
        let _ = tokio::fs::remove_dir_all(stale).await;
    } else {
        tokio::fs::rename(staging, directory).await?;
    }
    Ok(())
}

fn require_digest(bytes: &[u8], expected: &str) -> anyhow::Result<()> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        anyhow::bail!("catalog bundle digest {actual} differs from {expected}");
    }
    Ok(())
}

fn monotonic_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos())
}
