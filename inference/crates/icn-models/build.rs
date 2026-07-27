use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

use reqwest::StatusCode;
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, RANGE, USER_AGENT};
use serde::Deserialize;

#[path = "../../catalog/planner_bundle.rs"]
#[allow(dead_code)]
mod planner_bundle;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
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
    HuggingFace { repository: String, commit: String },
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
    size: u64,
}

fn main() {
    let crate_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let manifest_path = crate_dir.join("../../catalog/generated/recommendable-models.json");
    let bundle_source = crate_dir.join("../../catalog/planner_bundle.rs");
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    println!("cargo:rerun-if-changed={}", bundle_source.display());

    let manifest: Manifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", manifest_path.display())),
    )
    .unwrap_or_else(|error| {
        panic!(
            "invalid release catalog {}: {error}",
            manifest_path.display()
        )
    });
    let output =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("planner-headers.bin");
    if fs::read(&output)
        .ok()
        .is_some_and(|bytes| planner_bundle::sha256(&bytes) == manifest.planner_bundle_digest)
    {
        return;
    }

    let sources = header_sources(&manifest);
    println!(
        "cargo:warning=hydrating {} pinned local-model planner headers from Hugging Face",
        sources.len()
    );
    let client = Client::builder()
        .build()
        .expect("failed to construct Hugging Face client");
    let token = env::var("HF_TOKEN").ok().filter(|value| !value.is_empty());
    let mut headers = BTreeMap::new();
    for (digest, source) in sources {
        let bytes = download_header(&client, token.as_deref(), &source)
            .unwrap_or_else(|error| panic!("failed to hydrate planner header {digest}: {error}"));
        if planner_bundle::sha256(&bytes) != digest {
            panic!("planner header {digest} failed integrity validation");
        }
        headers.insert(digest, bytes);
    }
    let encoded = planner_bundle::encode(&headers).expect("failed to encode planner bundle");
    let actual_digest = planner_bundle::sha256(&encoded);
    assert_eq!(
        actual_digest, manifest.planner_bundle_digest,
        "hydrated planner bundle does not match the committed catalog"
    );
    let temporary = output.with_extension("bin.tmp");
    fs::write(&temporary, &encoded)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", temporary.display()));
    if output.exists() {
        fs::remove_file(&output)
            .unwrap_or_else(|error| panic!("failed to replace {}: {error}", output.display()));
    }
    fs::rename(&temporary, &output)
        .unwrap_or_else(|error| panic!("failed to publish {}: {error}", output.display()));
}

fn header_sources(manifest: &Manifest) -> BTreeMap<String, HeaderSource> {
    let mut sources = BTreeMap::new();
    for artifact in &manifest.planner_artifacts {
        let Source::HuggingFace { repository, commit } = &artifact.source;
        for component in &artifact.components {
            assert!(
                component.header_size_bytes > 0,
                "planner header size must be positive"
            );
            let source = HeaderSource {
                repository: repository.clone(),
                commit: commit.clone(),
                path: component.component.path.clone(),
                size: component.header_size_bytes,
            };
            if let Some(previous) = sources.insert(component.header_digest.clone(), source.clone())
            {
                assert_eq!(previous.size, source.size, "planner header size mismatch");
            }
        }
    }
    sources
}

fn download_header(
    client: &Client,
    token: Option<&str>,
    source: &HeaderSource,
) -> Result<Vec<u8>, String> {
    let mut url =
        reqwest::Url::parse("https://huggingface.co").map_err(|error| error.to_string())?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|()| "Hugging Face URL cannot accept path segments".to_owned())?;
        segments.extend(source.repository.split('/'));
        segments.push("resolve");
        segments.push(&source.commit);
        for component in source.path.components() {
            let value = component
                .as_os_str()
                .to_str()
                .ok_or_else(|| format!("planner path {} is not UTF-8", source.path.display()))?;
            segments.push(value);
        }
    }
    let mut request = client
        .get(url)
        .header(USER_AGENT, "magnitude-release-catalog-build")
        .header(RANGE, format!("bytes=0-{}", source.size - 1));
    if let Some(token) = token {
        request = request.header(AUTHORIZATION, format!("Bearer {token}"));
    }
    let response = request.send().map_err(|error| error.to_string())?;
    if response.status() != StatusCode::PARTIAL_CONTENT {
        return Err(format!(
            "Hugging Face returned {} instead of a ranged response for {}",
            response.status(),
            source.path.display()
        ));
    }
    let capacity = usize::try_from(source.size).map_err(|_| "planner header is too large")?;
    let mut bytes = Vec::with_capacity(capacity);
    response
        .take(source.size + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() != capacity {
        return Err(format!(
            "Hugging Face returned {} bytes for {}, expected {}",
            bytes.len(),
            source.path.display(),
            source.size
        ));
    }
    Ok(bytes)
}
