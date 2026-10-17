//! Candidate inspection, drafter enumeration, candidate assessment and bundles against a local
//! stub of the Hugging Face Hub serving synthetic GGUF headers, assessed on the CPU backend.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use magnitude_artifacts::gguf::Encoding;
use magnitude_service_contracts::models::{
    CandidateDrafter, CandidateFile, CandidateInspectionRequest, CandidateSource,
    CatalogInstallationAdmission, CatalogInstallationRemoval, CatalogModels,
    CatalogModelsResponse, CatalogSupport, CustomModel, CustomModelOrigin, CustomModelState,
    DrafterCandidatesRequest, DrafterCategory, DrafterLineage, DrafterReference, DrafterVerdict,
    HuggingFaceRepositoryId, LocalModelRemoval, LocalModels, LocalModelsConfiguration,
    LocalModelsResponse, ModelAssessment, ModelCandidate, ModelCandidates,
    ModelDomainInvalidation, ModelId, ModelParameterization, ModelReleaseDate,
    ModelServingConfiguration, RecommendableModel, RecommendableModelCatalog,
    ServableModelBundle, ServingProfile, SpeculativeDraftSource, SpeculativeMethod,
    UnsupportedReason,
};
use magnitude_service_contracts::{
    HuggingFaceRepositoryRequest, InventoryError, ModelPreviewSource,
};
use magnitude_service_models::lineage::CatalogLineage;
use magnitude_service_models::{
    InventoryConfig, ManagedModelStore, ModelDomainResolver, ReleaseCatalog,
    refresh_hugging_face_repository,
};
use sha2::{Digest, Sha256};

use super::ManagedModelCandidates;
use crate::assessment::assessor::ModelAssessor;
use crate::assessment::environment::AssessmentEnvironment;

// Synthetic GGUF headers ---------------------------------------------------------------------

enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strings(Vec<String>),
    I32s(Vec<i32>),
    U32s(Vec<u32>),
}

#[derive(Default)]
struct Gguf {
    metadata: Vec<(String, Meta)>,
    tensors: Vec<(String, Vec<u64>, Encoding)>,
}

impl Gguf {
    fn set(&mut self, key: &str, value: Meta) {
        self.metadata.retain(|(name, _)| name != key);
        self.metadata.push((key.into(), value));
    }

    fn tensor(&mut self, name: &str, shape: &[u64], encoding: Encoding) {
        self.tensors.retain(|(existing, ..)| existing != name);
        self.tensors.push((name.into(), shape.to_vec(), encoding));
    }

    fn encode_all(&mut self, from: Encoding, to: Encoding) {
        for tensor in &mut self.tensors {
            if tensor.2 == from {
                tensor.2 = to;
            }
        }
    }

    /// The header and the logical size of the file it heads.
    fn bytes(&self) -> (Vec<u8>, u64) {
        fn string(out: &mut Vec<u8>, value: &str) {
            out.extend((value.len() as u64).to_le_bytes());
            out.extend(value.as_bytes());
        }
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((self.tensors.len() as u64).to_le_bytes());
        out.extend((self.metadata.len() as u64).to_le_bytes());
        for (key, value) in &self.metadata {
            string(&mut out, key);
            match value {
                Meta::U32(value) => {
                    out.extend(4u32.to_le_bytes());
                    out.extend(value.to_le_bytes());
                }
                Meta::F32(value) => {
                    out.extend(6u32.to_le_bytes());
                    out.extend(value.to_le_bytes());
                }
                Meta::Bool(value) => {
                    out.extend(7u32.to_le_bytes());
                    out.push(u8::from(*value));
                }
                Meta::Str(value) => {
                    out.extend(8u32.to_le_bytes());
                    string(&mut out, value);
                }
                Meta::Strings(values) => {
                    out.extend(9u32.to_le_bytes());
                    out.extend(8u32.to_le_bytes());
                    out.extend((values.len() as u64).to_le_bytes());
                    for value in values {
                        string(&mut out, value);
                    }
                }
                Meta::I32s(values) => {
                    out.extend(9u32.to_le_bytes());
                    out.extend(5u32.to_le_bytes());
                    out.extend((values.len() as u64).to_le_bytes());
                    for value in values {
                        out.extend(value.to_le_bytes());
                    }
                }
                Meta::U32s(values) => {
                    out.extend(9u32.to_le_bytes());
                    out.extend(4u32.to_le_bytes());
                    out.extend((values.len() as u64).to_le_bytes());
                    for value in values {
                        out.extend(value.to_le_bytes());
                    }
                }
            }
        }
        let mut offset = 0u64;
        for (name, shape, encoding) in &self.tensors {
            string(&mut out, name);
            out.extend((shape.len() as u32).to_le_bytes());
            for dimension in shape.iter().rev() {
                out.extend(dimension.to_le_bytes());
            }
            out.extend((*encoding as u32).to_le_bytes());
            out.extend(offset.to_le_bytes());
            let bytes =
                shape.iter().product::<u64>() / encoding.block_elements() * encoding.block_bytes();
            offset += bytes.div_ceil(32) * 32;
        }
        out.resize(out.len().div_ceil(32) * 32, 0);
        let logical = out.len() as u64 + offset;
        (out, logical)
    }
}

const HIDDEN: u64 = 2048;
const HEADS: u64 = 16;
const KV_HEADS: u64 = 8;
const WIDTH: u64 = 128;
const FEED_FORWARD: u64 = 5632;
const VOCABULARY: u64 = 32000;
const BLOCKS: u64 = 2;
const CHATML: &str = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

fn tokens() -> Vec<String> {
    let mut tokens = (0..VOCABULARY)
        .map(|id| format!("t{id}"))
        .collect::<Vec<_>>();
    tokens[0] = "<unk>".into();
    tokens[1] = "<s>".into();
    tokens[2] = "</s>".into();
    tokens[3] = "<|im_start|>".into();
    tokens[4] = "<|im_end|>".into();
    tokens
}

/// A two-layer `llama` target with `general.file_type` `file_type`.
fn llama(file_type: u32, weights: Encoding) -> Gguf {
    let mut gguf = Gguf::default();
    gguf.set("general.architecture", Meta::Str("llama".into()));
    gguf.set("general.file_type", Meta::U32(file_type));
    for (key, value) in [
        ("block_count", BLOCKS),
        ("context_length", 8192),
        ("embedding_length", HIDDEN),
        ("feed_forward_length", FEED_FORWARD),
        ("attention.head_count", HEADS),
        ("attention.head_count_kv", KV_HEADS),
        ("attention.key_length", WIDTH),
        ("attention.value_length", WIDTH),
        ("rope.dimension_count", WIDTH),
        ("vocab_size", VOCABULARY),
    ] {
        gguf.set(&format!("llama.{key}"), Meta::U32(value as u32));
    }
    gguf.set("llama.attention.layer_norm_rms_epsilon", Meta::F32(1e-5));
    gguf.set("llama.rope.freq_base", Meta::F32(500_000.0));
    gguf.set("tokenizer.ggml.model", Meta::Str("gpt2".into()));
    gguf.set("tokenizer.ggml.pre", Meta::Str("llama-bpe".into()));
    gguf.set("tokenizer.ggml.tokens", Meta::Strings(tokens()));
    let mut kinds = vec![1; VOCABULARY as usize];
    kinds[..5].fill(3);
    gguf.set("tokenizer.ggml.token_type", Meta::I32s(kinds));
    gguf.set("tokenizer.ggml.merges", Meta::Strings(Vec::new()));
    gguf.set("tokenizer.ggml.bos_token_id", Meta::U32(1));
    gguf.set("tokenizer.ggml.eos_token_id", Meta::U32(4));
    gguf.set("tokenizer.chat_template", Meta::Str(CHATML.into()));
    let query = HEADS * WIDTH;
    let key = KV_HEADS * WIDTH;
    gguf.tensor("token_embd.weight", &[VOCABULARY, HIDDEN], Encoding::Q8_0);
    gguf.tensor("output_norm.weight", &[HIDDEN], Encoding::F32);
    gguf.tensor("output.weight", &[VOCABULARY, HIDDEN], Encoding::Q8_0);
    for block in 0..BLOCKS {
        let name = |role: &str| format!("blk.{block}.{role}.weight");
        gguf.tensor(&name("attn_norm"), &[HIDDEN], Encoding::F32);
        gguf.tensor(&name("attn_q"), &[query, HIDDEN], weights);
        gguf.tensor(&name("attn_k"), &[key, HIDDEN], weights);
        gguf.tensor(&name("attn_v"), &[key, HIDDEN], weights);
        gguf.tensor(&name("attn_output"), &[HIDDEN, query], weights);
        gguf.tensor(&name("ffn_norm"), &[HIDDEN], Encoding::F32);
        gguf.tensor(&name("ffn_gate"), &[FEED_FORWARD, HIDDEN], weights);
        gguf.tensor(&name("ffn_up"), &[FEED_FORWARD, HIDDEN], weights);
        gguf.tensor(&name("ffn_down"), &[HIDDEN, FEED_FORWARD], weights);
    }
    gguf
}

/// A one-layer DFlash drafter of width `hidden` tapping the target's first layer.
fn dflash(hidden: u64) -> Gguf {
    let mut gguf = Gguf::default();
    gguf.set("general.architecture", Meta::Str("dflash".into()));
    for (key, value) in [
        ("block_count", 1),
        ("context_length", 4096),
        ("embedding_length", hidden),
        ("feed_forward_length", FEED_FORWARD),
        ("attention.head_count", HEADS),
        ("attention.head_count_kv", KV_HEADS),
        ("attention.key_length", WIDTH),
        ("attention.value_length", WIDTH),
        ("block_size", 16),
    ] {
        gguf.set(&format!("dflash.{key}"), Meta::U32(value as u32));
    }
    gguf.set("dflash.attention.layer_norm_rms_epsilon", Meta::F32(1e-6));
    gguf.set("dflash.rope.freq_base", Meta::F32(1_000_000.0));
    gguf.set("dflash.target_layers", Meta::U32s(vec![1]));
    gguf.set("dflash.attention.causal", Meta::Bool(false));
    gguf.set("tokenizer.ggml.mask_token_id", Meta::U32(5));
    let query = HEADS * WIDTH;
    let key = KV_HEADS * WIDTH;
    gguf.tensor("fc.weight", &[hidden, hidden], Encoding::Q8_0);
    gguf.tensor("enc.output_norm.weight", &[hidden], Encoding::F32);
    gguf.tensor("output_norm.weight", &[hidden], Encoding::F32);
    let name = |role: &str| format!("blk.0.{role}.weight");
    gguf.tensor(&name("attn_norm"), &[hidden], Encoding::F32);
    gguf.tensor(&name("attn_q"), &[query, hidden], Encoding::Q8_0);
    gguf.tensor(&name("attn_q_norm"), &[WIDTH], Encoding::F32);
    gguf.tensor(&name("attn_k"), &[key, hidden], Encoding::Q8_0);
    gguf.tensor(&name("attn_k_norm"), &[WIDTH], Encoding::F32);
    gguf.tensor(&name("attn_v"), &[key, hidden], Encoding::Q8_0);
    gguf.tensor(&name("attn_output"), &[hidden, query], Encoding::Q8_0);
    gguf.tensor(&name("ffn_norm"), &[hidden], Encoding::F32);
    gguf.tensor(&name("ffn_gate"), &[FEED_FORWARD, hidden], Encoding::Q8_0);
    gguf.tensor(&name("ffn_up"), &[FEED_FORWARD, hidden], Encoding::Q8_0);
    gguf.tensor(&name("ffn_down"), &[hidden, FEED_FORWARD], Encoding::Q8_0);
    gguf
}

/// A DSpark drafter: a DFlash drafter with a Markov head and a confidence head.
fn dspark() -> Gguf {
    const RANK: u64 = 32;
    let mut gguf = dflash(HIDDEN);
    gguf.tensor("markov_w1.weight", &[VOCABULARY, RANK], Encoding::F32);
    gguf.tensor("markov_w2.weight", &[VOCABULARY, RANK], Encoding::F32);
    gguf.tensor("conf_proj.weight", &[HIDDEN + RANK], Encoding::F32);
    gguf.tensor("conf_proj.bias", &[1], Encoding::F32);
    gguf
}

/// A DFlash2 drafter: a DFlash drafter with convolutions around its layer and a candidate
/// selector.
fn dflash2() -> Gguf {
    const KERNEL: u64 = 4;
    const GROUP: u64 = 128;
    const RANK: u64 = 32;
    let mut gguf = dflash(HIDDEN);
    gguf.set("dflash.conv_kernel_size", Meta::U32(KERNEL as u32));
    gguf.set("dflash.conv_group_size", Meta::U32(GROUP as u32));
    gguf.set("dflash.selector_rank", Meta::U32(RANK as u32));
    gguf.set("dflash.selector_top_k", Meta::U32(4));
    for sublayer in ["attn", "ffn"] {
        gguf.tensor(&format!("blk.0.{sublayer}_conv_base"), &[2, KERNEL, HIDDEN], Encoding::F32);
        gguf.tensor(
            &format!("blk.0.{sublayer}_conv_proj.weight"),
            &[2 * KERNEL * (HIDDEN / GROUP), HIDDEN],
            Encoding::F32,
        );
    }
    gguf.tensor("selector_hidden.weight", &[RANK, HIDDEN], Encoding::F32);
    gguf.tensor("selector_predecessor.weight", &[VOCABULARY, RANK], Encoding::F32);
    gguf.tensor("selector_successor.weight", &[VOCABULARY, RANK], Encoding::F32);
    gguf
}

fn projector() -> Gguf {
    let mut gguf = Gguf::default();
    gguf.set("general.architecture", Meta::Str("clip".into()));
    gguf.set("clip.has_vision_encoder", Meta::Bool(true));
    gguf.tensor("v.patch_embd.weight", &[64, 64], Encoding::F32);
    gguf
}

// The stub Hub -------------------------------------------------------------------------------

#[derive(Clone)]
struct StubFile {
    path: String,
    header: Vec<u8>,
    size_bytes: u64,
}

impl StubFile {
    fn new(path: &str, gguf: &Gguf) -> Self {
        let (header, size_bytes) = gguf.bytes();
        Self {
            path: path.to_owned(),
            header,
            size_bytes,
        }
    }

    /// The same file with `tensor` stored under a tensor type id the engine doesn't know, as
    /// files of since-removed ggml types (`Q4_0_4_4` is 31) are.
    fn with_tensor_type(mut self, tensor: &str, type_id: u32) -> Self {
        let mut name = (tensor.len() as u64).to_le_bytes().to_vec();
        name.extend(tensor.as_bytes());
        let start = self
            .header
            .windows(name.len())
            .position(|window| window == name)
            .expect("tensor in header")
            + name.len();
        let rank = u32::from_le_bytes(self.header[start..start + 4].try_into().unwrap()) as usize;
        let at = start + 4 + rank * 8;
        self.header[at..at + 4].copy_from_slice(&type_id.to_le_bytes());
        self
    }

    fn sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(&self.header);
        digest.update(self.size_bytes.to_le_bytes());
        format!("{:x}", digest.finalize())
    }
}

#[derive(Clone)]
struct StubRepository {
    commit: String,
    files: Vec<StubFile>,
    card_base_models: Vec<String>,
}

#[derive(Default)]
struct StubHub {
    repositories: BTreeMap<String, StubRepository>,
    search: Vec<serde_json::Value>,
    drafter_searches: AtomicUsize,
    header_requests: AtomicUsize,
}

fn commit(seed: char) -> String {
    seed.to_string().repeat(40)
}

impl StubHub {
    fn serve(self) -> (String, Arc<Self>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("hub listener");
        let endpoint = format!("http://{}", listener.local_addr().expect("hub address"));
        let hub = Arc::new(self);
        let served = Arc::clone(&hub);
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(connection) = connection else {
                    return;
                };
                let hub = Arc::clone(&served);
                std::thread::spawn(move || hub.respond(connection));
            }
        });
        (endpoint, hub)
    }

    fn respond(&self, mut connection: std::net::TcpStream) {
        let mut reader = BufReader::new(connection.try_clone().expect("stream"));
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() {
            return;
        }
        let mut range = None;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("range")
            {
                range = Some(value.trim().to_owned());
            }
        }
        let target = request_line.split_whitespace().nth(1).unwrap_or("/").to_owned();
        let (path, query) = target.split_once('?').unwrap_or((&target, ""));
        let segments = path
            .trim_start_matches('/')
            .split('/')
            .map(|segment| segment.replace("%2F", "/"))
            .collect::<Vec<_>>();
        let (status, headers, body) = self.route(&segments, query, range.as_deref());
        let _ = write!(
            connection,
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{headers}Connection: close\r\n\r\n",
            body.len()
        );
        let _ = connection.write_all(&body);
    }

    fn route(&self, segments: &[String], query: &str, range: Option<&str>) -> (String, String, Vec<u8>) {
        let json = |value: serde_json::Value| {
            (
                "200 OK".to_owned(),
                "Content-Type: application/json\r\n".to_owned(),
                serde_json::to_vec(&value).unwrap(),
            )
        };
        let missing = || ("404 Not Found".to_owned(), String::new(), Vec::new());
        match segments {
            [api, models] if api == "api" && models == "models" => {
                if query.contains("base_model") {
                    self.drafter_searches.fetch_add(1, Ordering::SeqCst);
                    json(serde_json::Value::Array(Vec::new()))
                } else {
                    json(serde_json::Value::Array(self.search.clone()))
                }
            }
            [api, models, owner, name, revision_segment, revision]
                if api == "api" && models == "models" && revision_segment == "revision" =>
            {
                let Some(repository) = self.repositories.get(&format!("{owner}/{name}")) else {
                    return missing();
                };
                if revision != "main" && revision != &repository.commit {
                    return missing();
                }
                json(serde_json::json!({
                    "id": format!("{owner}/{name}"),
                    "sha": repository.commit,
                    "tags": ["gguf"],
                    "cardData": { "base_model": repository.card_base_models },
                    "siblings": repository.files.iter().map(|file| serde_json::json!({
                        "rfilename": file.path,
                        "size": file.size_bytes,
                        "lfs": { "sha256": file.sha256(), "size": file.size_bytes },
                    })).collect::<Vec<_>>(),
                }))
            }
            [api, models, owner, name] if api == "api" && models == "models" => {
                let Some(repository) = self.repositories.get(&format!("{owner}/{name}")) else {
                    return missing();
                };
                json(serde_json::json!({
                    "id": format!("{owner}/{name}"),
                    "baseModels": {
                        "relation": "finetune",
                        "models": repository.card_base_models.iter()
                            .map(|id| serde_json::json!({ "id": id }))
                            .collect::<Vec<_>>(),
                    },
                }))
            }
            [owner, name, resolve, revision, rest @ ..] if resolve == "resolve" => {
                let Some(repository) = self.repositories.get(&format!("{owner}/{name}")) else {
                    return missing();
                };
                let path = rest.join("/");
                let Some(file) = repository
                    .files
                    .iter()
                    .find(|file| file.path == path && revision == &repository.commit)
                else {
                    return missing();
                };
                self.header_requests.fetch_add(1, Ordering::SeqCst);
                let (start, end) = range
                    .and_then(|range| range.strip_prefix("bytes="))
                    .and_then(|range| range.split_once('-'))
                    .map(|(start, end)| (start.parse::<u64>().unwrap(), end.parse::<u64>().unwrap()))
                    .unwrap_or((0, file.size_bytes - 1));
                let body = (start..=end)
                    .map(|offset| file.header.get(offset as usize).copied().unwrap_or(0))
                    .collect::<Vec<_>>();
                (
                    "206 Partial Content".to_owned(),
                    format!("Content-Range: bytes {start}-{end}/{}\r\n", file.size_bytes),
                    body,
                )
            }
            _ => missing(),
        }
    }
}

// Domain stubs -------------------------------------------------------------------------------

struct StubLocal(Vec<CustomModel>);

impl LocalModels for StubLocal {
    fn configuration(&self) -> BoxFuture<'_, Result<LocalModelsConfiguration, InventoryError>> {
        Box::pin(async { Ok(LocalModelsConfiguration::default()) })
    }

    fn configure(
        &self,
        configuration: LocalModelsConfiguration,
    ) -> BoxFuture<'_, Result<LocalModelsConfiguration, InventoryError>> {
        Box::pin(async move { Ok(configuration) })
    }

    fn list_local(&self) -> BoxFuture<'_, Result<LocalModelsResponse, InventoryError>> {
        Box::pin(async {
            Ok(LocalModelsResponse {
                revision: 1,
                reconciliation_complete: true,
                models: self.0.clone(),
                hidden: Vec::new(),
                unsupported: Vec::new(),
            })
        })
    }

    fn refresh_local(&self) -> BoxFuture<'_, Result<LocalModelsResponse, InventoryError>> {
        self.list_local()
    }

    fn remove_local(
        &self,
        model_id: &ModelId,
    ) -> BoxFuture<'_, Result<LocalModelRemoval, InventoryError>> {
        let model_id = model_id.clone();
        Box::pin(async move { Err(InventoryError::NotFound(model_id.as_str().to_owned())) })
    }

    fn prune_drafters(
        &self,
    ) -> BoxFuture<'_, Result<magnitude_service_contracts::models::DrafterPruning, InventoryError>> {
        Box::pin(async { Err(InventoryError::Unsupported("no drafters".to_owned())) })
    }

    fn watch_local(&self) -> BoxStream<'static, ModelDomainInvalidation> {
        Box::pin(futures_util::stream::empty())
    }
}

struct StubCatalog;

impl CatalogModels for StubCatalog {
    fn list_catalog(&self) -> BoxFuture<'_, Result<CatalogModelsResponse, InventoryError>> {
        Box::pin(async {
            Ok(CatalogModelsResponse {
                revision: 1,
                reconciliation_complete: true,
                models: Vec::new(),
            })
        })
    }

    fn install_catalog_model(
        &self,
        model_id: &ModelId,
    ) -> BoxFuture<'_, Result<CatalogInstallationAdmission, InventoryError>> {
        let model_id = model_id.clone();
        Box::pin(async move { Err(InventoryError::NotFound(model_id.as_str().to_owned())) })
    }

    fn remove_catalog_model_installation(
        &self,
        model_id: &ModelId,
    ) -> BoxFuture<'_, Result<CatalogInstallationRemoval, InventoryError>> {
        let model_id = model_id.clone();
        Box::pin(async move { Err(InventoryError::NotFound(model_id.as_str().to_owned())) })
    }

    fn watch_catalog(&self) -> BoxStream<'static, ModelDomainInvalidation> {
        Box::pin(futures_util::stream::empty())
    }
}

// Fixture ------------------------------------------------------------------------------------

const CATALOG_TARGET: &str = "catalog/Llama-GGUF";
const CATALOG_DRAFT: &str = "catalog/Llama-DFlash-GGUF";
const COMMUNITY: &str = "community/Llama-Finetune-GGUF";
const THIRD_PARTY: &str = "thirdparty/Llama-DFlash-GGUF";
const SPECULATIVE: &str = "community/Llama-Speculative-GGUF";
const REJECTED: &str = "community/Llama-Rejected-GGUF";
const PLAIN: &str = "community/Llama-Plain-GGUF";
const LEGACY: &str = "community/Llama-Legacy-GGUF";

fn repository(seed: char, files: Vec<StubFile>, base_models: &[&str]) -> StubRepository {
    StubRepository {
        commit: commit(seed),
        files,
        card_base_models: base_models.iter().map(|base| (*base).to_owned()).collect(),
    }
}

fn hub() -> StubHub {
    let q8 = llama(7, Encoding::Q8_0);
    let q4 = llama(15, Encoding::Q4K);
    let mut q2 = llama(10, Encoding::Q8_0);
    q2.encode_all(Encoding::Q8_0, Encoding::Q2K);
    let mut deepseek = llama(7, Encoding::Q8_0);
    deepseek.set("general.architecture", Meta::Str("deepseek4".into()));
    let mut repositories = BTreeMap::new();
    repositories.insert(
        CATALOG_TARGET.to_owned(),
        repository('a', vec![StubFile::new("llama-Q8_0.gguf", &q8)], &["base/Llama"]),
    );
    repositories.insert(
        CATALOG_DRAFT.to_owned(),
        repository('b', vec![StubFile::new("dflash.gguf", &dflash(HIDDEN))], &["base/Llama"]),
    );
    repositories.insert(
        COMMUNITY.to_owned(),
        repository(
            'c',
            vec![
                StubFile::new("llama-Q8_0.gguf", &q8),
                StubFile::new("llama-Q4_K_M.gguf", &q4),
                StubFile::new("llama-Q2_K.gguf", &q2),
                StubFile::new("llama-MTP-Q8_0.gguf", &q8),
                StubFile::new("dflash-draft.gguf", &dflash(HIDDEN)),
                StubFile::new("dflash-narrow.gguf", &dflash(HIDDEN / 2)),
            ],
            &["community/Llama-Finetune"],
        ),
    );
    repositories.insert(
        "community/Llama-Finetune".to_owned(),
        repository('d', vec![StubFile::new("unused.gguf", &q8)], &["base/Llama"]),
    );
    repositories.insert(
        "vision/Llama-Vision-GGUF".to_owned(),
        repository(
            'e',
            vec![
                StubFile::new("llama-Q8_0.gguf", &q8),
                StubFile::new("mmproj-F16.gguf", &projector()),
                StubFile::new("mmproj-F32.gguf", &{
                    let mut larger = projector();
                    larger.tensor("v.patch_embd.weight", &[128, 128], Encoding::F32);
                    larger
                }),
            ],
            &[],
        ),
    );
    repositories.insert(
        "unsloth/DeepSeek-V4-GGUF".to_owned(),
        repository('f', vec![StubFile::new("deepseek-Q8_0.gguf", &deepseek)], &[]),
    );
    repositories.insert(
        SPECULATIVE.to_owned(),
        repository(
            '1',
            vec![
                StubFile::new("llama-Q8_0.gguf", &q8),
                StubFile::new("drafters/dflash.gguf", &dflash(HIDDEN)),
                StubFile::new("dspark.gguf", &dspark()),
                StubFile::new("dflash2.gguf", &dflash2()),
            ],
            &["community/Llama-Finetune"],
        ),
    );
    repositories.insert(
        REJECTED.to_owned(),
        repository(
            '2',
            vec![
                StubFile::new("llama-Q8_0.gguf", &q8),
                StubFile::new("dflash-narrow.gguf", &dflash(HIDDEN / 2)),
            ],
            &["community/Llama-Finetune"],
        ),
    );
    repositories.insert(
        PLAIN.to_owned(),
        repository('3', vec![StubFile::new("llama-Q8_0.gguf", &q8)], &["community/Llama-Finetune"]),
    );
    repositories.insert(
        LEGACY.to_owned(),
        repository(
            '4',
            vec![
                StubFile::new("llama-Q4_0_4_4.gguf", &q8).with_tensor_type("blk.0.ffn_up.weight", 31),
                StubFile::new("llama-Q8_0.gguf", &q8),
            ],
            &[],
        ),
    );
    repositories.insert(
        THIRD_PARTY.to_owned(),
        repository('9', vec![StubFile::new("draft.gguf", &dflash(HIDDEN))], &["base/Llama"]),
    );
    let search_result = |id: &str, seed: char, architecture: &str, base_models: &[&str]| {
        serde_json::json!({
            "id": id,
            "sha": commit(seed),
            "downloads": 10,
            "tags": ["gguf"],
            "gguf": { "architecture": architecture },
            "baseModels": {
                "relation": "quantized",
                "models": base_models.iter().map(|id| serde_json::json!({ "id": id })).collect::<Vec<_>>(),
            },
        })
    };
    StubHub {
        repositories,
        search: vec![
            search_result(CATALOG_TARGET, 'a', "llama", &["base/Llama"]),
            search_result("unsloth/DeepSeek-V4-GGUF", 'f', "deepseek4", &[]),
            search_result("vision/Llama-Vision-GGUF", 'e', "clip", &[]),
            search_result(COMMUNITY, 'c', "llama", &["community/Llama-Finetune"]),
        ],
        drafter_searches: AtomicUsize::new(0),
        header_requests: AtomicUsize::new(0),
    }
}

struct Fixture {
    _directory: tempfile::TempDir,
    hub: Arc<StubHub>,
    models: Arc<ManagedModelStore>,
    lineage: Arc<CatalogLineage>,
    candidates: Arc<ManagedModelCandidates>,
    catalog_id: ModelId,
}

async fn hugging_face_package(
    models: &Arc<ManagedModelStore>,
    repository: &str,
    path: &str,
) -> magnitude_service_contracts::models::ModelPackage {
    let snapshot = refresh_hugging_face_repository(
        models,
        HuggingFaceRepositoryRequest {
            repository: repository.to_owned(),
            revision: "main".to_owned(),
        },
    )
    .await
    .expect("stub snapshot");
    models
        .hugging_face_package(
            &ModelPreviewSource {
                repository: repository.to_owned(),
                revision: snapshot.commit.clone(),
                primary_gguf: PathBuf::from(path),
                additional_components: Vec::new(),
            },
            &snapshot,
        )
        .await
        .expect("stub package")
        .0
}

fn cpu_environment() -> Arc<AssessmentEnvironment> {
    let catalog = seismic::DeviceCatalog::discover().expect("device catalog");
    let setup = magnitude_engine::assessment::AssessmentSetup::discover(
        &catalog,
        magnitude_executor::platform::DeviceRequest::Backend(seismic::BackendName::Cpu),
        magnitude_executor::platform::MemoryReserves::standard(),
        crate::assessment::environment::serving_policy(),
        magnitude_engine::options::standard_service_limits(),
    )
    .expect("the CPU backend is selectable");
    Arc::new(AssessmentEnvironment {
        id: magnitude_service_contracts::models::AssessmentEnvironmentId("cpu-test".to_owned()),
        setup,
    })
}

async fn fixture(custom: Vec<CustomModel>) -> Fixture {
    fixture_scanning(custom, LocalModelsConfiguration::default()).await
}

/// A fixture whose store has scanned the locations `local_models` names.
async fn fixture_scanning(custom: Vec<CustomModel>, local_models: LocalModelsConfiguration) -> Fixture {
    let (endpoint, hub) = hub().serve();
    let directory = tempfile::tempdir().expect("temporary directory");
    let mut config = InventoryConfig::with_roots(
        directory.path().join("models"),
        directory.path().join("cache"),
    )
    .expect("inventory config");
    config.hub_endpoint = Some(endpoint);
    config.local_models = local_models;
    let models = Arc::new(ManagedModelStore::open(config).await.expect("model store"));
    models.ensure_model_inventory().await.expect("location scan");
    let target = hugging_face_package(&models, CATALOG_TARGET, "llama-Q8_0.gguf").await;
    let draft = hugging_face_package(&models, CATALOG_DRAFT, "dflash.gguf").await;
    let base = magnitude_service_contracts::models::CatalogBaseId::new("llama-test").unwrap();
    let variant = magnitude_service_contracts::models::CatalogVariantId::new("gguf:q8").unwrap();
    let catalog = RecommendableModelCatalog {
        models: vec![RecommendableModel {
            model_id: base.clone(),
            variant_id: variant.clone(),
            configuration: ModelServingConfiguration {
                bundle: ServableModelBundle::SpeculativeDecoding {
                    target,
                    draft_source: SpeculativeDraftSource::Separate { draft },
                    method: SpeculativeMethod::DFlash,
                },
                profile: ServingProfile {
                    context_length: 8192,
                },
            },
            display_name: "Llama Test".to_owned(),
            variant_label: "Q8".to_owned(),
            description: "A test model".to_owned(),
            release_date: ModelReleaseDate::new("2026-01-01").unwrap(),
            license: "apache-2.0".to_owned(),
            parameterization: ModelParameterization::Dense {
                total_parameters: 1,
            },
            intelligence: 1,
            support: CatalogSupport::Supported,
            fidelity_rank: 1,
            quantization_aware: false,
        }],
        diagnostics: Vec::new(),
    };
    let lineage = CatalogLineage::new(Arc::clone(&models), catalog.clone());
    let resolver = ModelDomainResolver::new(Arc::clone(&models), catalog.clone());
    let assessor = Arc::new(ModelAssessor::new(
        Arc::clone(&models),
        resolver,
        Arc::new(ReleaseCatalog::without_material(catalog).expect("release catalog")),
        cpu_environment(),
    ));
    let candidates = ManagedModelCandidates::new(
        assessor,
        Arc::clone(&lineage),
        super::engine_architectures(),
        Arc::new(StubCatalog),
        Arc::new(StubLocal(custom)),
    );
    Fixture {
        _directory: directory,
        hub,
        models,
        lineage,
        candidates,
        catalog_id: ModelId::catalog(&base, &variant),
    }
}

fn hugging_face_candidate(repository: &str, seed: char, target: &str) -> ModelCandidate {
    ModelCandidate {
        source: CandidateSource::HuggingFace {
            repository: HuggingFaceRepositoryId::new(repository.to_owned()).unwrap(),
            commit: commit(seed),
        },
        target: vec![target.to_owned()],
        projector: None,
        drafter: None,
    }
}

fn version<'a>(
    inspection: &'a magnitude_service_contracts::models::CandidateInspection,
    file: &str,
) -> &'a magnitude_service_contracts::models::CandidateVersion {
    inspection
        .versions
        .iter()
        .find(|version| version.files.iter().any(|candidate| candidate.path == file))
        .unwrap_or_else(|| panic!("no version for {file}"))
}

// Tests --------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn inspection_groups_versions_and_reports_each_engine_verdict() {
    let on_disk = ModelId::hugging_face(
        &HuggingFaceRepositoryId::new(COMMUNITY.to_owned()).unwrap(),
        &magnitude_service_contracts::models::HuggingFaceArtifactSelector::new(
            "llama-Q8_0.gguf".to_owned(),
        )
        .unwrap(),
    );
    let q8_size = llama(7, Encoding::Q8_0).bytes().1;
    // Found with a projector beside it, as an Ollama `hf.co/…` pull is: the version is still the
    // same bytes, so it is already on this computer.
    let mut found = custom_model(on_disk.clone(), q8_size + 1_000);
    found.files[0].size_bytes = q8_size;
    found.projector = Some(magnitude_service_contracts::models::CustomModelFile {
        path: PathBuf::from("/found/mmproj.gguf"),
        size_bytes: 1_000,
    });
    let fixture = fixture(vec![found]).await;
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new(COMMUNITY.to_owned()).unwrap(),
            revision: "main".to_owned(),
        })
        .await
        .expect("inspection");

    assert_eq!(inspection.display_name, "Llama Finetune");
    assert_eq!(inspection.variant_of, Some(fixture.catalog_id.clone()));
    assert!(inspection.unsupported_architecture.is_none());
    assert_eq!(
        inspection.versions.len(),
        4,
        "drafter companions are excluded and the MTP-named target is kept"
    );
    assert!(matches!(
        version(&inspection, "llama-Q8_0.gguf").assessment,
        Some(ModelAssessment::Fits { .. })
    ));
    assert!(matches!(
        version(&inspection, "llama-MTP-Q8_0.gguf").assessment,
        Some(ModelAssessment::Fits { .. })
    ));
    let Some(ModelAssessment::Unsupported { reasons, .. }) =
        &version(&inspection, "llama-Q2_K.gguf").assessment
    else {
        panic!("Q2_K weights are unsupported");
    };
    assert!(matches!(
        reasons.as_slice(),
        [UnsupportedReason::WeightTypes { types }] if types[0].ggml_type == "Q2_K"
    ));
    let recommended = inspection
        .versions
        .iter()
        .filter(|version| version.recommended)
        .map(|version| version.files[0].path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(recommended, ["llama-Q4_K_M.gguf"]);
    assert_eq!(version(&inspection, "llama-Q4_K_M.gguf").label, "Q4_K_M");
    assert_eq!(
        version(&inspection, "llama-Q8_0.gguf").on_this_computer,
        Some(on_disk.clone())
    );
    assert_eq!(version(&inspection, "llama-Q4_K_M.gguf").on_this_computer, None);
    assert_eq!(version(&inspection, "llama-Q8_0.gguf").model_id, on_disk);
    assert_eq!(
        version(&inspection, "llama-Q4_K_M.gguf").model_id.as_str(),
        format!("hf:{COMMUNITY}/llama-Q4_K_M.gguf")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_local_file_is_identified_by_its_header_digest_unless_a_listed_model_has_it() {
    let folder = tempfile::tempdir().expect("model folder");
    let (header, _) = llama(7, Encoding::Q8_0).bytes();
    let path = folder.path().join("model.gguf");
    std::fs::write(&path, header).unwrap();
    let digest = magnitude_service_models::candidate_files::local_header_digest(&path).unwrap();
    let request = || CandidateInspectionRequest::Local { path: path.clone() };

    let unlisted = fixture(Vec::new()).await;
    let inspection = unlisted.candidates.inspect(request()).await.expect("inspection");
    let version = &inspection.versions[0];
    assert_eq!(version.model_id.as_str(), format!("local:{digest}"));
    assert_eq!(version.on_this_computer, None);

    let found = ModelId::hugging_face(
        &HuggingFaceRepositoryId::new(COMMUNITY.to_owned()).unwrap(),
        &magnitude_service_contracts::models::HuggingFaceArtifactSelector::new(
            "model.gguf".to_owned(),
        )
        .unwrap(),
    );
    let mut listed = custom_model(found.clone(), 0);
    listed.files = vec![magnitude_service_contracts::models::CustomModelFile {
        path: path.clone(),
        size_bytes: 0,
    }];
    let listed = fixture(vec![listed]).await;
    let inspection = listed.candidates.inspect(request()).await.expect("inspection");
    let version = &inspection.versions[0];
    assert_eq!(version.model_id, found);
    assert_eq!(version.on_this_computer, Some(found));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_copy_of_a_listed_model_elsewhere_is_already_on_this_computer_as_that_model() {
    let folder = tempfile::tempdir().expect("model folders");
    let (bytes, _) = llama(7, Encoding::Q8_0).bytes();
    let found_path = folder.path().join("lmstudio/community/Llama-GGUF/llama-Q8_0.gguf");
    let copy = folder.path().join("Downloads/llama-copy.gguf");
    for path in [&found_path, &copy] {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, &bytes).unwrap();
    }
    let found = ModelId::hugging_face(
        &HuggingFaceRepositoryId::new("community/Llama-GGUF".to_owned()).unwrap(),
        &magnitude_service_contracts::models::HuggingFaceArtifactSelector::new(
            "llama-Q8_0.gguf".to_owned(),
        )
        .unwrap(),
    );
    let mut listed = custom_model(found.clone(), bytes.len() as u64);
    listed.files[0].path = found_path;
    let fixture = fixture(vec![listed]).await;

    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::Local { path: copy.clone() })
        .await
        .expect("inspection");

    assert_eq!(
        inspection.versions[0].on_this_computer,
        Some(found),
        "the same bytes at another path are the model already listed"
    );
    let digest = magnitude_service_models::candidate_files::local_header_digest(&copy).unwrap();
    assert_eq!(
        inspection.versions[0].model_id.as_str(),
        format!("local:{digest}"),
        "the copy itself is identified by its header, so it can be recorded as one more location"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drafter_already_in_another_apps_folder_is_on_this_computer() {
    let fixture = fixture(Vec::new()).await;
    let candidates = candidate_drafters(&fixture, PLAIN, '3').await;
    assert!(!candidates[0].on_this_computer, "{candidates:#?}");

    let lm_studio = tempfile::tempdir().expect("LM Studio folder");
    let copy = lm_studio.path().join("someone/Mirror-GGUF/renamed-dflash.gguf");
    std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
    let (header, length) = dflash(HIDDEN).bytes();
    std::fs::write(&copy, header).unwrap();
    std::fs::OpenOptions::new().write(true).open(&copy).unwrap().set_len(length).unwrap();
    let fixture = fixture_scanning(
        Vec::new(),
        LocalModelsConfiguration {
            roots: vec![magnitude_service_contracts::models::ModelRoot {
                kind: magnitude_service_contracts::models::ModelRootKind::LmStudio,
                path: lm_studio.path().to_path_buf(),
                enabled: true,
            }],
            ..LocalModelsConfiguration::default()
        },
    )
    .await;

    let catalog = candidate_drafters(&fixture, PLAIN, '3').await;
    assert_eq!(catalog[0].category, DrafterCategory::CatalogDrafter);
    assert!(
        catalog[0].on_this_computer,
        "the catalog drafter's bytes in LM Studio are used where they are"
    );
    let own = candidate_drafters(&fixture, COMMUNITY, 'c').await;
    assert_eq!(hugging_face_path(&own[0]), "dflash-draft.gguf");
    assert!(own[0].on_this_computer, "a repository drafter's bytes elsewhere count too");
    assert!(!own[1].on_this_computer, "a different drafter is not");

    use magnitude_service_models::CustomModelEnrichment as _;
    let header_digest =
        magnitude_service_models::candidate_files::local_header_digest(&copy).unwrap();
    let known = fixture
        .candidates
        .enrichment
        .drafter_content(&DrafterReference::Catalog {
            model_id: fixture.catalog_id.clone(),
        })
        .expect("the catalog drafter's bytes are known to the projection");
    assert_eq!(known.header_digest, Some(header_digest));
    assert_eq!(known.size_bytes, std::fs::metadata(&copy).unwrap().len());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_file_with_an_unknown_tensor_type_is_an_unreadable_version_not_a_failed_inspection() {
    let fixture = fixture(Vec::new()).await;
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new(LEGACY.to_owned()).unwrap(),
            revision: "main".to_owned(),
        })
        .await
        .expect("one unreadable file never fails the repository");

    assert_eq!(inspection.versions.len(), 2);
    assert_eq!(inspection.unsupported_architecture, None);
    let legacy = version(&inspection, "llama-Q4_0_4_4.gguf");
    let Some(ModelAssessment::Unsupported {
        reasons, detail, ..
    }) = &legacy.assessment
    else {
        panic!("the legacy file is unsupported: {:?}", legacy.assessment);
    };
    assert_eq!(reasons, &[UnsupportedReason::Unreadable]);
    assert!(detail.contains("unknown GGUF tensor type 31"), "{detail}");
    assert_eq!(
        legacy.model_id.as_str(),
        format!("hf:{LEGACY}/llama-Q4_0_4_4.gguf")
    );
    assert!(matches!(
        version(&inspection, "llama-Q8_0.gguf").assessment,
        Some(ModelAssessment::Fits { .. })
    ));
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn a_symlinked_local_file_is_inspected_through_its_link_under_its_own_name() {
    let folder = tempfile::tempdir().expect("model folder");
    let (header, _) = llama(7, Encoding::Q8_0).bytes();
    let blob = folder.path().join("blobs/0123abcd");
    std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
    std::fs::write(&blob, header).unwrap();
    let snapshot = folder.path().join("snapshots/commit");
    std::fs::create_dir_all(&snapshot).unwrap();
    let link = snapshot.join("model-Q8_0.gguf");
    std::os::unix::fs::symlink("../../blobs/0123abcd", &link).unwrap();
    let digest = magnitude_service_models::candidate_files::local_header_digest(&blob).unwrap();
    let fixture = fixture(Vec::new()).await;

    for request in [link.clone(), snapshot.clone()] {
        let inspection = fixture
            .candidates
            .inspect(CandidateInspectionRequest::Local { path: request })
            .await
            .expect("a symlinked file inspects like the file it links to");
        assert_eq!(inspection.versions.len(), 1);
        let version = &inspection.versions[0];
        assert_eq!(version.files[0].path, "model-Q8_0.gguf");
        assert_eq!(version.model_id.as_str(), format!("local:{digest}"));
        assert!(version.assessment.is_some());
    }
}

fn custom_model(model_id: ModelId, size_bytes: u64) -> CustomModel {
    CustomModel {
        model_id,
        origin: CustomModelOrigin::Found {
            source: magnitude_service_contracts::models::FoundSource::HuggingFaceCache,
        },
        display_name: "Found".to_owned(),
        variant_label: None,
        variant_of: None,
        files: vec![magnitude_service_contracts::models::CustomModelFile {
            path: PathBuf::from("/found/model.gguf"),
            size_bytes,
        }],
        size_bytes,
        projector: None,
        drafter: None,
        recommended_drafter: None,
        license: None,
        profile: ServingProfile {
            context_length: 8192,
        },
        state: CustomModelState::Unavailable {
            installation: magnitude_service_contracts::models::ResolvedModelInstallation::Resolved {
                installed_bytes: size_bytes,
                primary_path: PathBuf::from("/found/model.gguf"),
                ownership: magnitude_service_contracts::models::ModelInstallationOwnership::External,
            },
            failure: magnitude_service_contracts::models::ModelFailure {
                code: "test".to_owned(),
                message: "test".to_owned(),
                retryable: false,
            },
        },
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn inspection_selects_the_smallest_projector_and_explains_unknown_architectures() {
    let fixture = fixture(Vec::new()).await;
    let vision = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new("vision/Llama-Vision-GGUF".to_owned())
                .unwrap(),
            revision: "main".to_owned(),
        })
        .await
        .expect("vision inspection");
    assert_eq!(vision.versions.len(), 1);
    assert_eq!(
        vision.projector.as_ref().map(|projector| projector.path.as_str()),
        Some("mmproj-F16.gguf")
    );

    let deepseek = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new("unsloth/DeepSeek-V4-GGUF".to_owned())
                .unwrap(),
            revision: "main".to_owned(),
        })
        .await
        .expect("unsupported inspection");
    assert_eq!(
        deepseek.unsupported_architecture,
        Some(vec![UnsupportedReason::Architecture {
            architecture: "deepseek4".to_owned()
        }])
    );
    assert!(deepseek.versions.iter().all(|version| version.assessment.is_none()));
}

/// The drafters offered for a Hugging Face candidate, with no explicit reference.
async fn candidate_drafters(
    fixture: &Fixture,
    repository: &str,
    seed: char,
) -> Vec<magnitude_service_contracts::models::DrafterCandidate> {
    fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: hugging_face_candidate(repository, seed, "llama-Q8_0.gguf"),
            other: None,
        })
        .await
        .expect("drafter candidates")
        .candidates
}

fn preselected(
    candidates: &[magnitude_service_contracts::models::DrafterCandidate],
) -> &magnitude_service_contracts::models::DrafterCandidate {
    candidates
        .iter()
        .find(|candidate| matches!(candidate.verdict, DrafterVerdict::Bound { .. }))
        .expect("a bound drafter")
}

fn hugging_face_path(candidate: &magnitude_service_contracts::models::DrafterCandidate) -> &str {
    match &candidate.reference {
        DrafterReference::HuggingFace { path, .. } => path,
        other => panic!("not a Hugging Face drafter: {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drafter_in_the_model_repository_precedes_the_catalog_drafter() {
    let fixture = fixture(Vec::new()).await;
    let response = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: hugging_face_candidate(COMMUNITY, 'c', "llama-Q8_0.gguf"),
            other: Some(format!("{THIRD_PARTY}:draft.gguf")),
        })
        .await
        .expect("drafter candidates");
    let candidates = response.candidates;
    let categories = candidates
        .iter()
        .map(|candidate| candidate.category)
        .collect::<Vec<_>>();
    assert_eq!(
        categories,
        [
            DrafterCategory::NextToModel,
            DrafterCategory::NextToModel,
            DrafterCategory::CatalogDrafter,
            DrafterCategory::Other,
        ],
        "{candidates:#?}"
    );
    let own = &candidates[0];
    assert_eq!(hugging_face_path(own), "dflash-draft.gguf");
    assert_eq!(own.label, "In this repository");
    assert!(matches!(
        own.verdict,
        DrafterVerdict::Bound {
            method: SpeculativeMethod::DFlash,
            ..
        }
    ));
    assert_eq!(preselected(&candidates), own);
    let narrow = &candidates[1];
    assert_eq!(
        hugging_face_path(narrow),
        "dflash-narrow.gguf",
        "a rejected drafter follows the bound ones, though it is the smaller download"
    );
    assert!(matches!(
        &narrow.verdict,
        DrafterVerdict::Rejected { reasons, .. }
            if matches!(reasons.as_slice(), [UnsupportedReason::TargetMismatch { .. }])
    ));
    let catalog = &candidates[2];
    assert_eq!(
        catalog.reference,
        DrafterReference::Catalog {
            model_id: fixture.catalog_id.clone()
        }
    );
    assert_eq!(catalog.lineage, DrafterLineage::Lineage);
    assert!(matches!(
        catalog.verdict,
        DrafterVerdict::Bound {
            method: SpeculativeMethod::DFlash,
            ..
        }
    ));
    assert_eq!(catalog.name, "Magnitude DFlash for Llama Test");
    assert_eq!(catalog.label, "Trained for Llama Test");
    assert_eq!(candidates[3].label, "Entered by you");
    assert!(matches!(candidates[3].verdict, DrafterVerdict::Bound { .. }));
    assert_eq!(
        fixture.hub.drafter_searches.load(Ordering::SeqCst),
        0,
        "Hugging Face is never searched for drafters"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn repository_drafters_follow_the_method_order_at_any_path() {
    let fixture = fixture(Vec::new()).await;
    let candidates = candidate_drafters(&fixture, SPECULATIVE, '1').await;
    let listed = candidates
        .iter()
        .map(|candidate| (candidate.category, candidate.method.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        listed,
        [
            (DrafterCategory::NextToModel, Some(SpeculativeMethod::DFlash2)),
            (DrafterCategory::NextToModel, Some(SpeculativeMethod::DSpark)),
            (DrafterCategory::NextToModel, Some(SpeculativeMethod::DFlash)),
            (DrafterCategory::CatalogDrafter, Some(SpeculativeMethod::DFlash)),
        ],
        "{candidates:#?}"
    );
    assert_eq!(hugging_face_path(&candidates[2]), "drafters/dflash.gguf");
    assert_eq!(hugging_face_path(preselected(&candidates)), "dflash2.gguf");
    assert_eq!(fixture.hub.drafter_searches.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_repository_drafter_the_catalog_drafter_is_preselected() {
    let fixture = fixture(Vec::new()).await;
    let candidates = candidate_drafters(&fixture, PLAIN, '3').await;
    assert_eq!(candidates.len(), 1, "{candidates:#?}");
    assert_eq!(preselected(&candidates).category, DrafterCategory::CatalogDrafter);

    let rejected = candidate_drafters(&fixture, REJECTED, '2').await;
    assert_eq!(
        rejected
            .iter()
            .map(|candidate| candidate.category)
            .collect::<Vec<_>>(),
        [DrafterCategory::NextToModel, DrafterCategory::CatalogDrafter]
    );
    assert!(matches!(rejected[0].verdict, DrafterVerdict::Rejected { .. }));
    assert_eq!(
        preselected(&rejected).category,
        DrafterCategory::CatalogDrafter,
        "a rejected repository drafter is listed but never preselected"
    );
    assert_eq!(fixture.hub.drafter_searches.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn drafters_in_a_local_model_folder_precede_the_fingerprint_matched_catalog_drafter() {
    let fixture = fixture(Vec::new()).await;
    let folder = tempfile::tempdir().expect("model folder");
    let write = |name: &str, gguf: &Gguf| std::fs::write(folder.path().join(name), gguf.bytes().0).unwrap();
    write("model.gguf", &llama(7, Encoding::Q8_0));
    write("dflash.gguf", &dflash(HIDDEN));
    write("dflash-narrow.gguf", &dflash(HIDDEN / 2));
    let candidates = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: ModelCandidate {
                source: CandidateSource::Local {
                    path: folder.path().to_path_buf(),
                },
                target: vec!["model.gguf".to_owned()],
                projector: None,
                drafter: None,
            },
            other: None,
        })
        .await
        .expect("drafter candidates")
        .candidates;
    let listed = candidates
        .iter()
        .map(|candidate| (candidate.category, candidate.label.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(
        listed,
        [
            (DrafterCategory::NextToModel, "In this folder"),
            (DrafterCategory::NextToModel, "In this folder"),
            (DrafterCategory::CatalogDrafter, "Trained for Llama Test"),
        ],
        "{candidates:#?}"
    );
    assert_eq!(
        preselected(&candidates).reference,
        DrafterReference::Local {
            path: folder.path().join("dflash.gguf")
        }
    );
    assert!(candidates[0].on_this_computer);
    assert_eq!(candidates[2].lineage, DrafterLineage::Fingerprint);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_explicit_drafter_is_listed_with_its_verdict_and_lineage() {
    let fixture = fixture(Vec::new()).await;
    let candidates = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: hugging_face_candidate("vision/Llama-Vision-GGUF", 'e', "llama-Q8_0.gguf"),
            other: Some(format!("{COMMUNITY}:dflash-narrow.gguf")),
        })
        .await
        .expect("drafter candidates")
        .candidates;
    let other = candidates
        .iter()
        .find(|candidate| candidate.category == DrafterCategory::Other)
        .expect("the explicit drafter");
    assert!(matches!(other.verdict, DrafterVerdict::Rejected { .. }));
    assert_eq!(other.lineage, DrafterLineage::Lineage);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_local_file_without_lineage_matches_its_catalog_parent_by_fingerprint() {
    let fixture = fixture(Vec::new()).await;
    let folder = tempfile::tempdir().expect("model folder");
    let (header, _) = llama(7, Encoding::Q8_0).bytes();
    std::fs::write(folder.path().join("model.gguf"), header).unwrap();
    let candidates = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: ModelCandidate {
                source: CandidateSource::Local {
                    path: folder.path().to_path_buf(),
                },
                target: vec!["model.gguf".to_owned()],
                projector: None,
                drafter: None,
            },
            other: None,
        })
        .await
        .expect("drafter candidates")
        .candidates;
    let catalog = candidates
        .iter()
        .find(|candidate| candidate.category == DrafterCategory::CatalogDrafter)
        .expect("the catalog drafter");
    assert_eq!(catalog.lineage, DrafterLineage::Fingerprint);
    assert!(matches!(catalog.verdict, DrafterVerdict::Bound { .. }));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_exact_catalog_target_and_a_catalog_model_use_only_the_catalog_drafter() {
    let fixture = fixture(Vec::new()).await;
    let exact = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: hugging_face_candidate(CATALOG_TARGET, 'a', "llama-Q8_0.gguf"),
            other: None,
        })
        .await
        .expect("drafter candidates")
        .candidates;
    assert_eq!(exact[0].lineage, DrafterLineage::Exact);
    let catalog = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Model {
            model_id: fixture.catalog_id.clone(),
            other: None,
        })
        .await
        .expect("catalog drafter")
        .candidates;
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0].category, DrafterCategory::CatalogDrafter);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_repeated_inspection_reads_each_header_once_and_reports_download_sizes() {
    let fixture = fixture(Vec::new()).await;
    let request = CandidateInspectionRequest::HuggingFace {
        repository: HuggingFaceRepositoryId::new("vision/Llama-Vision-GGUF".to_owned()).unwrap(),
        revision: commit('e'),
    };
    let first = fixture
        .candidates
        .inspect(request.clone())
        .await
        .expect("inspection");
    let requests = fixture.hub.header_requests.load(Ordering::SeqCst);
    let version = version(&first, "llama-Q8_0.gguf");
    let projector = first.projector.as_ref().expect("projector");
    assert_eq!(version.download_bytes, version.size_bytes + projector.size_bytes);

    let second = fixture.candidates.inspect(request).await.expect("cached inspection");
    assert_eq!(first, second);
    assert_eq!(
        fixture.hub.header_requests.load(Ordering::SeqCst),
        requests,
        "a repeated inspection reads no header again"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn candidate_assessment_is_cached_by_its_material() {
    let fixture = fixture(Vec::new()).await;
    let candidate = ModelCandidate {
        drafter: Some(CandidateDrafter {
            reference: DrafterReference::HuggingFace {
                repository: HuggingFaceRepositoryId::new(COMMUNITY.to_owned()).unwrap(),
                commit: commit('c'),
                path: "dflash-draft.gguf".to_owned(),
            },
            method: SpeculativeMethod::DFlash,
        }),
        ..hugging_face_candidate(COMMUNITY, 'c', "llama-Q8_0.gguf")
    };
    let first = fixture
        .candidates
        .assess_candidate(candidate.clone())
        .await
        .expect("assessment");
    assert!(matches!(
        first,
        ModelAssessment::Fits {
            drafter: DrafterVerdict::Bound { .. },
            ..
        }
    ));
    let requests = fixture.hub.header_requests.load(Ordering::SeqCst);
    let second = fixture
        .candidates
        .assess_candidate(candidate)
        .await
        .expect("cached assessment");
    assert_eq!(first, second);
    assert_eq!(
        fixture.hub.header_requests.load(Ordering::SeqCst),
        requests,
        "a repeated assessment reads no header again"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn bundles_carry_the_exact_packages_a_rescan_produces() {
    let fixture = fixture(Vec::new()).await;
    let candidate = ModelCandidate {
        drafter: Some(CandidateDrafter {
            reference: DrafterReference::Catalog {
                model_id: fixture.catalog_id.clone(),
            },
            method: SpeculativeMethod::DFlash,
        }),
        ..hugging_face_candidate(COMMUNITY, 'c', "llama-Q8_0.gguf")
    };
    let bundle = fixture
        .candidates
        .bundle(candidate.clone())
        .await
        .expect("bundle");
    let expected_target = hugging_face_package(&fixture.models, COMMUNITY, "llama-Q8_0.gguf").await;
    let ServableModelBundle::SpeculativeDecoding {
        target,
        draft_source: SpeculativeDraftSource::Separate { draft },
        method,
    } = bundle
    else {
        panic!("a separate-drafter bundle");
    };
    assert_eq!(target.id, expected_target.id);
    assert_eq!(method, SpeculativeMethod::DFlash);
    assert_eq!(
        draft.id,
        hugging_face_package(&fixture.models, CATALOG_DRAFT, "dflash.gguf").await.id
    );

    let local = fixture
        .candidates
        .bundle(ModelCandidate {
            source: CandidateSource::Local {
                path: PathBuf::from("/models"),
            },
            target: vec!["model.gguf".to_owned()],
            projector: Some(CandidateFile {
                path: "mmproj.gguf".to_owned(),
                size_bytes: 1,
            }),
            ..candidate
        })
        .await
        .expect("drafter-only bundle");
    assert!(matches!(
        local,
        ServableModelBundle::Standalone { package } if package.id == draft.id
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_recommended_drafter_of_a_catalog_model_is_its_catalog_drafter() {
    let fixture = fixture(Vec::new()).await;
    let recommended = fixture
        .candidates
        .recommended_drafter(&fixture.catalog_id)
        .await
        .expect("recommendation")
        .expect("a bound drafter");
    assert_eq!(recommended.category, DrafterCategory::CatalogDrafter);
}

#[test]
fn enrichment_answers_only_for_the_exact_target_it_was_computed_for() {
    use magnitude_service_models::{CustomModelEnrichment, CustomModelEnrichmentResult, CustomModelFacts};
    let enrichment = super::CandidateEnrichment::default();
    let model_id = ModelId::local(
        &magnitude_service_contracts::models::LocalModelDigest::new("a".repeat(64)).unwrap(),
    );
    let result = CustomModelEnrichmentResult {
        variant_of: Some("llama-test:gguf:q8".parse().unwrap()),
        recommended_drafter: None,
        license: Some("apache-2.0".to_owned()),
    };
    assert!(enrichment.record((model_id.clone(), "package_a".to_owned()), result.clone()));
    assert!(!enrichment.record((model_id.clone(), "package_a".to_owned()), result.clone()));
    let origin = CustomModelOrigin::Imported {
        path: PathBuf::from("/models/model.gguf"),
    };
    let package = |id: &str| magnitude_service_contracts::models::ModelPackage {
        id: magnitude_service_contracts::models::ModelPackageId(id.to_owned()),
        source: magnitude_service_contracts::models::ModelPackageSource::Local {
            path: PathBuf::from("/models"),
        },
        files: Vec::new(),
        relationships: Vec::new(),
        properties: magnitude_service_contracts::models::ModelPackageProperties {
            format: "gguf".to_owned(),
            quantization: "Q8_0".to_owned(),
            quantization_name: "8-bit".to_owned(),
            architecture: "llama".to_owned(),
            maximum_context_length: None,
            intrinsic_model_id: None,
            intrinsic_quality_id: None,
        },
    };
    let facts = |target| CustomModelFacts {
        model_id: &model_id,
        origin: &origin,
        display_name: "model",
        target,
        hub: None,
        ollama_name: None,
        name: None,
        architecture: Some("llama"),
        base_models: &[],
        embedded_mtp: false,
    };
    let (package_a, package_b) = (package("package_a"), package("package_b"));
    assert_eq!(enrichment.enrich(&facts(&package_a)), result);
    assert_eq!(
        enrichment.enrich(&facts(&package_b)),
        CustomModelEnrichmentResult::default()
    );
    assert!(enrichment.record(
        (model_id.clone(), "package_b".to_owned()),
        CustomModelEnrichmentResult::default()
    ));
}
