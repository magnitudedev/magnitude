//! Candidate inspection, drafter enumeration, candidate assessment and bundles against a local
//! stub of the Hugging Face Hub serving synthetic GGUF headers, assessed on the CPU backend.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use magnitude_artifacts::gguf::Encoding;
use magnitude_service_contracts::models::{
    CandidateDrafter, CandidateFile, CandidateInspectionPhase, CandidateInspectionProgress,
    CandidateInspectionProgressSink, CandidateInspectionRequest, CandidateSource,
    CatalogInstallationAdmission, CatalogInstallationRemoval, CatalogModels, CatalogModelsResponse,
    CatalogSupport, CustomModel, CustomModelOrigin, CustomModelState, DrafterCandidatesRequest,
    DrafterCategory, DrafterLineage, DrafterReference, DrafterVerdict, HuggingFaceRepositoryId,
    KnownUnsupportedReason, LocalModelRemoval, LocalModels, LocalModelsConfiguration,
    LocalModelsResponse, ModelAssessment, ModelCandidate, ModelCandidates, ModelDomainInvalidation,
    ModelId, ModelParameterization, ModelReleaseDate, ModelServingConfiguration, ProjectorVerdict,
    RecommendableModel, RecommendableModelCatalog, ServableModelBundle, ServingProfile,
    SpeculativeDraftSource, SpeculativeMethod, UnsupportedReason,
};
use magnitude_service_contracts::{
    HuggingFaceRepositoryRequest, InventoryError, ModelPreviewSource,
};
use magnitude_service_models::lineage::CatalogLineage;
use magnitude_service_models::{
    CatalogDrafterFit, InventoryConfig, ManagedModelStore, ModelDomainResolver, ReleaseCatalog,
    WeightSample, refresh_hugging_face_repository, servable_model_bundle_key_for_bundle,
};
use sha2::{Digest, Sha256};

use super::ManagedModelCandidates;
use crate::assessment::assessor::ModelAssessor;
use crate::assessment::environment::AssessmentEnvironment;

// Synthetic GGUF headers ---------------------------------------------------------------------

#[derive(Clone)]
enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strings(Vec<String>),
    I32s(Vec<i32>),
    U32s(Vec<u32>),
    F32s(Vec<f32>),
}

#[derive(Default)]
struct Gguf {
    metadata: Vec<(String, Meta)>,
    tensors: Vec<(String, Vec<u64>, Encoding)>,
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend((value.len() as u64).to_le_bytes());
    out.extend(value.as_bytes());
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

    /// The encoded bytes of one metadata entry: its key, value type and value.
    fn entry(key: &str, value: &Meta) -> Vec<u8> {
        fn array<T>(
            out: &mut Vec<u8>,
            element: u32,
            values: &[T],
            encode: impl Fn(&mut Vec<u8>, &T),
        ) {
            out.extend(9u32.to_le_bytes());
            out.extend(element.to_le_bytes());
            out.extend((values.len() as u64).to_le_bytes());
            for value in values {
                encode(out, value);
            }
        }
        let mut out = Vec::new();
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
            Meta::Strings(values) => array(&mut out, 8, values, |out, value| string(out, value)),
            Meta::I32s(values) => array(&mut out, 5, values, |out, value| {
                out.extend(value.to_le_bytes())
            }),
            Meta::U32s(values) => array(&mut out, 4, values, |out, value| {
                out.extend(value.to_le_bytes())
            }),
            Meta::F32s(values) => array(&mut out, 6, values, |out, value| {
                out.extend(value.to_le_bytes())
            }),
        }
        out
    }

    /// The header and the logical size of the file it heads.
    fn bytes(&self) -> (Vec<u8>, u64) {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((self.tensors.len() as u64).to_le_bytes());
        out.extend((self.metadata.len() as u64).to_le_bytes());
        for (key, value) in &self.metadata {
            out.extend(Self::entry(key, value));
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
    // Byte-level BPE spells every byte (GPT-2's byte-to-character map), so the engine resolves a
    // stub target for loading as well as for assessment.
    let mut shifted = 0;
    for byte in 0..=255_u32 {
        let printable = (u32::from(b'!')..=u32::from(b'~')).contains(&byte)
            || (0xA1..=0xAC).contains(&byte)
            || (0xAE..=0xFF).contains(&byte);
        let character = if printable {
            byte
        } else {
            shifted += 1;
            255 + shifted
        };
        tokens[5 + byte as usize] = char::from_u32(character).unwrap().to_string();
    }
    tokens
}

/// A two-layer `llama` target with `general.file_type` `file_type`.
/// A local GGUF holding every byte of tensor data its header declares (zeros after `contents`),
/// as a model on disk does; a shorter file holds no weights and is never a version.
fn write_complete_bytes(path: &std::path::Path, contents: &[u8], length: u64) {
    std::fs::write(path, contents).unwrap();
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(length.max(contents.len() as u64))
        .unwrap();
}

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
    dflash_with_feed_forward(hidden, FEED_FORWARD)
}

/// A DFlash drafter whose layer has `feed_forward` units.
fn dflash_with_feed_forward(hidden: u64, feed_forward: u64) -> Gguf {
    let mut gguf = Gguf::default();
    gguf.set("general.architecture", Meta::Str("dflash".into()));
    for (key, value) in [
        ("block_count", 1),
        ("context_length", 4096),
        ("embedding_length", hidden),
        ("feed_forward_length", feed_forward),
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
    gguf.tensor(&name("ffn_gate"), &[feed_forward, hidden], Encoding::Q8_0);
    gguf.tensor(&name("ffn_up"), &[feed_forward, hidden], Encoding::Q8_0);
    gguf.tensor(&name("ffn_down"), &[hidden, feed_forward], Encoding::Q8_0);
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
        gguf.tensor(
            &format!("blk.0.{sublayer}_conv_base"),
            &[2, KERNEL, HIDDEN],
            Encoding::F32,
        );
        gguf.tensor(
            &format!("blk.0.{sublayer}_conv_proj.weight"),
            &[2 * KERNEL * (HIDDEN / GROUP), HIDDEN],
            Encoding::F32,
        );
    }
    gguf.tensor("selector_hidden.weight", &[RANK, HIDDEN], Encoding::F32);
    gguf.tensor(
        "selector_predecessor.weight",
        &[VOCABULARY, RANK],
        Encoding::F32,
    );
    gguf.tensor(
        "selector_successor.weight",
        &[VOCABULARY, RANK],
        Encoding::F32,
    );
    gguf
}

fn projector() -> Gguf {
    let mut gguf = Gguf::default();
    gguf.set("general.architecture", Meta::Str("clip".into()));
    gguf.set("clip.has_vision_encoder", Meta::Bool(true));
    gguf.tensor("v.patch_embd.weight", &[64, 64], Encoding::F32);
    gguf
}

// A repository laid out as published ones are ------------------------------------------------

/// The tokenizer arrays every file of `LAYOUTS` shares, sized like a real vocabulary's relative to
/// the rest of a header: tokens, scores, token types and merges.
fn layout_tokenizer() -> Vec<(String, Meta)> {
    let tokens = (0..VOCABULARY)
        .map(|id| match id {
            0 => "<unk>".to_owned(),
            1 => "<s>".to_owned(),
            2 => "</s>".to_owned(),
            3 => "<|im_start|>".to_owned(),
            4 => "<|im_end|>".to_owned(),
            id => format!("t{id}-{:08x}", id.wrapping_mul(2_654_435_761) as u32),
        })
        .collect::<Vec<_>>();
    let mut kinds = vec![1; VOCABULARY as usize];
    kinds[..5].fill(3);
    let merges = (0..120_000_u64)
        .map(|index| format!("t{} t{}", 5 + index % 31_000, 5 + (index * 7919) % 31_000))
        .collect::<Vec<_>>();
    vec![
        ("tokenizer.ggml.tokens".to_owned(), Meta::Strings(tokens)),
        (
            "tokenizer.ggml.scores".to_owned(),
            Meta::F32s((0..VOCABULARY).map(|id| -(id as f32)).collect()),
        ),
        ("tokenizer.ggml.token_type".to_owned(), Meta::I32s(kinds)),
        ("tokenizer.ggml.merges".to_owned(), Meta::Strings(merges)),
    ]
}

/// The encoded length of the shared tokenizer arrays.
fn layout_shared_bytes() -> usize {
    layout_tokenizer()
        .iter()
        .map(|(key, value)| Gguf::entry(key, value).len())
        .sum()
}

/// `gguf` with the shared tokenizer in place of its token list, its metadata ordered as `order`
/// moves keys to the end.
fn with_layout_tokenizer(mut gguf: Gguf, padding: u32, to_end: &[&str]) -> Gguf {
    let at = gguf
        .metadata
        .iter()
        .position(|(key, _)| key == "tokenizer.ggml.tokens")
        .unwrap_or(gguf.metadata.len());
    gguf.metadata.retain(|(key, _)| {
        !matches!(
            key.as_str(),
            "tokenizer.ggml.tokens" | "tokenizer.ggml.token_type" | "tokenizer.ggml.merges"
        )
    });
    let at = at.min(gguf.metadata.len());
    gguf.metadata.splice(at..at, layout_tokenizer());
    gguf.set("tokenizer.ggml.padding_token_id", Meta::U32(padding));
    for key in to_end {
        if let Some(index) = gguf.metadata.iter().position(|(name, _)| name == key) {
            let entry = gguf.metadata.remove(index);
            gguf.metadata.push(entry);
        }
    }
    gguf
}

/// The files of `LAYOUTS`: quantizations whose metadata order, names, padding token and
/// imatrix keys differ around one shared tokenizer, one preceded by a large metadata array of its
/// own, a three-way split version, a projector, an MTP drafter whose end-of-sequence token and
/// first and last token types differ, an imatrix, and a file whose header the engine can't read.
fn layout_files() -> Vec<StubFile> {
    let mut bf16 = llama(32, Encoding::BF16);
    bf16.metadata.rotate_right(1);
    let bf16 = with_layout_tokenizer(bf16, 0, &[]);
    let mut q8 = with_layout_tokenizer(
        llama(7, Encoding::Q8_0),
        0,
        &["general.file_type", "tokenizer.chat_template"],
    );
    q8.set(
        "quantize.imatrix.file",
        Meta::Str("imatrix/imatrix.gguf".into()),
    );
    q8.set("quantize.imatrix.entries_count", Meta::U32(14));
    let mut q4 = with_layout_tokenizer(llama(15, Encoding::Q4K), 2, &["general.file_type"]);
    q4.set(
        "general.name",
        Meta::Str("Llama Layouts Abliterated Q4_K_M".into()),
    );
    let mut q2 = llama(10, Encoding::Q8_0);
    q2.encode_all(Encoding::Q8_0, Encoding::Q2K);
    q2.metadata.insert(
        2,
        (
            "prism.hadamard.sign_values".to_owned(),
            Meta::I32s((0..40_000).collect()),
        ),
    );
    let q2 = with_layout_tokenizer(q2, 0, &[]);
    let iq1 = with_layout_tokenizer(llama(29, Encoding::Q8_0), 0, &[]);
    let mut mtp = dflash(HIDDEN);
    mtp.metadata.extend(layout_tokenizer());
    if let Some((_, Meta::I32s(kinds))) = mtp
        .metadata
        .iter_mut()
        .find(|(key, _)| key == "tokenizer.ggml.token_type")
    {
        kinds[3] = 1;
        kinds[VOCABULARY as usize - 10] = 4;
    }
    mtp.set("tokenizer.ggml.eos_token_id", Meta::U32(2));
    let mut imatrix = Gguf::default();
    imatrix.set("general.type", Meta::Str("imatrix".into()));
    for block in 0..BLOCKS {
        imatrix.tensor(
            &format!("blk.{block}.ffn_up.weight.in_sum2"),
            &[HIDDEN],
            Encoding::F32,
        );
    }
    let mut files = vec![
        StubFile::new("Llama-Layouts-BF16.gguf", &bf16),
        StubFile::new("Llama-Layouts-Q8_0.gguf", &q8),
        StubFile::new("Llama-Layouts-Q4_K_M.gguf", &q4),
        StubFile::new("Llama-Layouts-Q2_K.gguf", &q2),
        StubFile::new("Llama-Layouts-IQ1_M.gguf", &iq1)
            .with_tensor_type("blk.0.ffn_up.weight", 142),
        StubFile::new("MTP/mtp-Llama-Layouts.gguf", &mtp),
        StubFile::new("mmproj-F16.gguf", &projector()),
        StubFile::new("imatrix/imatrix.gguf", &imatrix),
    ];
    let q6 = with_layout_tokenizer(llama(18, Encoding::Q8_0), 0, &[]);
    let shard_tensors = q6.tensors.len().div_ceil(3);
    for (index, tensors) in q6.tensors.chunks(shard_tensors).enumerate() {
        let mut shard = Gguf::default();
        if index == 0 {
            shard.metadata = q6
                .metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
        }
        shard.set("split.no", Meta::U32(index as u32));
        shard.set("split.count", Meta::U32(3));
        shard.set("split.tensors.count", Meta::U32(q6.tensors.len() as u32));
        shard.tensors = tensors.to_vec();
        files.push(StubFile::new(
            &format!("Q6_K/Llama-Layouts-Q6_K-{:05}-of-00003.gguf", index + 1),
            &shard,
        ));
    }
    files
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
    header_bytes: AtomicUsize,
    unreachable: AtomicBool,
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
        // An unreachable Hub never answers, as when its name doesn't resolve and every request
        // waits out its timeout.
        if self.unreachable.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_secs(3600));
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
        let target = request_line
            .split_whitespace()
            .nth(1)
            .unwrap_or("/")
            .to_owned();
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

    fn route(
        &self,
        segments: &[String],
        query: &str,
        range: Option<&str>,
    ) -> (String, String, Vec<u8>) {
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
                    .map(|(start, end)| {
                        (start.parse::<u64>().unwrap(), end.parse::<u64>().unwrap())
                    })
                    .unwrap_or((0, file.size_bytes - 1));
                let body = (start..=end)
                    .map(|offset| file.header.get(offset as usize).copied().unwrap_or(0))
                    .collect::<Vec<_>>();
                self.header_bytes.fetch_add(body.len(), Ordering::SeqCst);
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
    ) -> BoxFuture<'_, Result<magnitude_service_contracts::models::DrafterPruning, InventoryError>>
    {
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

    fn install_catalog_component(
        &self,
        model_id: &ModelId,
        _component: magnitude_service_contracts::models::CatalogComponent,
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
const LAYOUTS: &str = "layouts/Llama-Layouts-GGUF";
const REJECTED: &str = "community/Llama-Rejected-GGUF";
const PLAIN: &str = "community/Llama-Plain-GGUF";
const LEGACY: &str = "community/Llama-Legacy-GGUF";
const OTHER_SHAPE: &str = "other/Llama-Retokenized-GGUF";

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
        repository(
            'a',
            vec![StubFile::new("llama-Q8_0.gguf", &q8)],
            &["base/Llama"],
        ),
    );
    repositories.insert(
        CATALOG_DRAFT.to_owned(),
        repository(
            'b',
            vec![StubFile::new("dflash.gguf", &dflash(HIDDEN))],
            &["base/Llama"],
        ),
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
        repository(
            'd',
            vec![StubFile::new("unused.gguf", &q8)],
            &["base/Llama"],
        ),
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
        repository(
            'f',
            vec![StubFile::new("deepseek-Q8_0.gguf", &deepseek)],
            &[],
        ),
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
        repository(
            '3',
            vec![StubFile::new("llama-Q8_0.gguf", &q8)],
            &["community/Llama-Finetune"],
        ),
    );
    repositories.insert(
        LEGACY.to_owned(),
        repository(
            '4',
            vec![
                // A newer quantization: a tensor type id and a file type the engine doesn't know.
                StubFile::new("llama-PTQ1_0.gguf", &llama(143, Encoding::Q8_0))
                    .with_tensor_type("blk.0.ffn_up.weight", 200),
                StubFile::new("llama-Q8_0.gguf", &q8),
            ],
            &[],
        ),
    );
    repositories.insert(LAYOUTS.to_owned(), repository('5', layout_files(), &[]));
    // The same architecture with another vocabulary: another shape fingerprint.
    let mut retokenized = llama(7, Encoding::Q8_0);
    let mut other_tokens = tokens();
    if let Some(last) = other_tokens.last_mut() {
        last.push_str("-other");
    }
    retokenized.set("tokenizer.ggml.tokens", Meta::Strings(other_tokens));
    repositories.insert(
        OTHER_SHAPE.to_owned(),
        repository(
            '6',
            vec![
                StubFile::new("llama-Q8_0.gguf", &retokenized),
                StubFile::new("dflash-narrow.gguf", &dflash(HIDDEN / 2)),
            ],
            &["base/Llama"],
        ),
    );
    repositories.insert(
        THIRD_PARTY.to_owned(),
        repository(
            '9',
            vec![StubFile::new("draft.gguf", &dflash(HIDDEN))],
            &["base/Llama"],
        ),
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
        header_bytes: AtomicUsize::new(0),
        unreachable: AtomicBool::new(false),
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
    let catalog = seismic::Devices::discover(&[]).expect("device catalog");
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
async fn fixture_scanning(
    custom: Vec<CustomModel>,
    local_models: LocalModelsConfiguration,
) -> Fixture {
    fixture_with(custom, local_models, false, BTreeMap::new()).await
}

/// A fixture whose catalog lists, before Llama Test, "Other Llama": a retokenized Llama (another
/// shape fingerprint) whose narrow DFlash drafter the engine rejects for Llama.
async fn fixture_with_other_catalog_model() -> Fixture {
    fixture_with(
        Vec::new(),
        LocalModelsConfiguration::default(),
        true,
        BTreeMap::new(),
    )
    .await
}

/// The catalog the fixtures serve: Llama Test, whose drafter is the DFlash file in its own
/// repository.
async fn llama_test_catalog(
    models: &Arc<ManagedModelStore>,
) -> (RecommendableModelCatalog, ModelId) {
    let target = hugging_face_package(models, CATALOG_TARGET, "llama-Q8_0.gguf").await;
    let draft = hugging_face_package(models, CATALOG_DRAFT, "dflash.gguf").await;
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
    (catalog, ModelId::catalog(&base, &variant))
}

async fn fixture_with(
    custom: Vec<CustomModel>,
    local_models: LocalModelsConfiguration,
    other_catalog_model: bool,
    extra_repositories: BTreeMap<String, StubRepository>,
) -> Fixture {
    fixture_from(
        custom,
        local_models,
        other_catalog_model,
        extra_repositories,
        false,
        cpu_environment(),
    )
    .await
}

/// A fixture over every option: with `qwen`, the hub also serves the real Qwen repositories and
/// the catalog also lists Qwen3.5 4B, Qwen3.6 35B-A3B and Qwen3.8 27B with their drafters.
async fn fixture_from(
    custom: Vec<CustomModel>,
    local_models: LocalModelsConfiguration,
    other_catalog_model: bool,
    extra_repositories: BTreeMap<String, StubRepository>,
    qwen: bool,
    environment: Arc<AssessmentEnvironment>,
) -> Fixture {
    let mut stub = hub();
    stub.repositories.extend(extra_repositories);
    if qwen {
        stub.repositories.extend(qwen_repositories());
        stub.repositories.extend(qwen_generation_repositories());
    }
    let (endpoint, hub) = stub.serve();
    let directory = tempfile::tempdir().expect("temporary directory");
    let mut config = InventoryConfig::with_roots(
        directory.path().join("models"),
        directory.path().join("cache"),
    )
    .expect("inventory config");
    config.hub_endpoint = Some(endpoint);
    config.local_models = local_models;
    config.architectures = super::engine_architectures();
    let models = Arc::new(ManagedModelStore::open(config, std::sync::Arc::new(|| Ok::<u64, std::convert::Infallible>(u64::MAX))).await.expect("model store"));
    models
        .ensure_model_inventory()
        .await
        .expect("location scan");
    let (catalog, catalog_id) = llama_test_catalog(&models).await;
    let mut catalog = catalog;
    if other_catalog_model {
        let mut other = catalog.models[0].clone();
        other.model_id =
            magnitude_service_contracts::models::CatalogBaseId::new("other-llama").unwrap();
        other.display_name = "Other Llama".to_owned();
        other.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
            target: hugging_face_package(&models, OTHER_SHAPE, "llama-Q8_0.gguf").await,
            draft_source: SpeculativeDraftSource::Separate {
                draft: hugging_face_package(&models, OTHER_SHAPE, "dflash-narrow.gguf").await,
            },
            method: SpeculativeMethod::DFlash,
        };
        catalog.models.insert(0, other);
    }
    if qwen {
        catalog.models.push(qwen_catalog_model(&models).await);
        catalog.models.push(qwen36_catalog_model(&models).await);
        catalog.models.push(qwen38_catalog_model(&models).await);
    }
    let lineage = CatalogLineage::new(Arc::clone(&models), catalog.clone());
    let resolver = ModelDomainResolver::new(Arc::clone(&models), catalog.clone());
    let weight_samples = qwen_generation_weight_samples(&catalog);
    let assessor = Arc::new(ModelAssessor::new(
        Arc::clone(&models),
        resolver,
        Arc::new(
            ReleaseCatalog::without_material(catalog)
                .expect("release catalog")
                .with_target_weight_samples(weight_samples),
        ),
        environment,
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
        catalog_id,
    }
}

/// A real file's weight sample, from the assessment fixtures.
fn weight_sample(name: &str) -> WeightSample {
    let path = fixtures_root().join(format!("weight-samples/{name}.json"));
    WeightSample::decode(&std::fs::read(path).expect("weight sample fixture"))
        .expect("weight sample")
}

/// The weight samples the release bundle carries for the real Qwen catalog targets listed.
fn qwen_generation_weight_samples(
    catalog: &magnitude_service_contracts::models::RecommendableModelCatalog,
) -> BTreeMap<magnitude_service_models::ServableModelBundleKey, WeightSample> {
    catalog
        .models
        .iter()
        .filter_map(|model| {
            let name = match model.model_id.as_str() {
                "qwen3.6-35b-a3b" => "Qwen3.6-35B-A3B-UD-Q4_K_XL",
                "qwen3.8-27b" => "Qwen3.8-27B-UD-Q4_K_XL",
                _ => return None,
            };
            Some((
                servable_model_bundle_key_for_bundle(&model.configuration.bundle),
                weight_sample(name),
            ))
        })
        .collect()
}

/// A model store with nothing in it, for what reads local files through its caches.
pub(super) async fn bare_store() -> (tempfile::TempDir, Arc<ManagedModelStore>) {
    let directory = tempfile::tempdir().expect("temporary directory");
    let config = InventoryConfig::with_roots(
        directory.path().join("models"),
        directory.path().join("cache"),
    )
    .expect("inventory config");
    let store = Arc::new(ManagedModelStore::open(config, std::sync::Arc::new(|| Ok::<u64, std::convert::Infallible>(u64::MAX))).await.expect("model store"));
    (directory, store)
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
        [UnsupportedReason::Known(KnownUnsupportedReason::WeightTypes { types })] if types[0].ggml_type == "Q2_K"
    ));
    let recommended = inspection
        .versions
        .iter()
        .filter(|version| version.recommended)
        .map(|version| version.files[0].path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(recommended, ["llama-Q4_K_M.gguf"]);
    let drafter = |file: &str| match &version(&inspection, file).assessment {
        Some(ModelAssessment::Fits { drafter, .. }) => drafter.clone(),
        other => panic!("{file} fits: {other:?}"),
    };
    assert!(
        matches!(drafter("llama-Q4_K_M.gguf"), DrafterVerdict::Bound { .. }),
        "the recommended version is composed with its preselected drafter, as it installs"
    );
    assert_eq!(
        drafter("llama-Q8_0.gguf"),
        DrafterVerdict::None,
        "another version keeps its plain verdict; its drafters come with its own drafter list"
    );
    assert_eq!(version(&inspection, "llama-Q4_K_M.gguf").label, "Q4_K_M");
    assert_eq!(
        version(&inspection, "llama-Q8_0.gguf").on_this_computer,
        Some(on_disk.clone())
    );
    assert_eq!(
        version(&inspection, "llama-Q4_K_M.gguf").on_this_computer,
        None
    );
    assert_eq!(version(&inspection, "llama-Q8_0.gguf").model_id, on_disk);
    assert_eq!(
        version(&inspection, "llama-Q4_K_M.gguf").model_id.as_str(),
        format!("hf:{COMMUNITY}/llama-Q4_K_M.gguf")
    );
}

/// An inspection reports, in order: every header read, counted from none to all; each version
/// check with the rows it returns; then the recommended version's drafter checks. Each version's
/// verdict appears as soon as it is checked, and the rows end as the inspection's versions.
#[tokio::test(flavor = "multi_thread")]
async fn inspection_reports_its_progress_in_order_with_counts_and_each_verdict() {
    let fixture = fixture(Vec::new()).await;
    let reported = Arc::new(std::sync::Mutex::new(Vec::<CandidateInspectionProgress>::new()));
    let sink = {
        let reported = Arc::clone(&reported);
        CandidateInspectionProgressSink::new(move |progress| {
            reported.lock().unwrap().push(progress);
        })
    };
    let inspection = fixture
        .candidates
        .inspect_with_progress(
            CandidateInspectionRequest::HuggingFace {
                repository: HuggingFaceRepositoryId::new(COMMUNITY.to_owned()).unwrap(),
                revision: "main".to_owned(),
            },
            sink,
        )
        .await
        .expect("inspection");
    let reported = reported.lock().unwrap().clone();
    let phase_rank = |progress: &CandidateInspectionProgress| match progress.phase {
        CandidateInspectionPhase::ReadingHeaders { .. } => 0,
        CandidateInspectionPhase::CheckingVersions { .. } => 1,
        CandidateInspectionPhase::CheckingDrafters { .. } => 2,
    };
    assert!(
        reported.windows(2).all(|pair| phase_rank(&pair[0]) <= phase_rank(&pair[1])),
        "phases never go back: {reported:#?}"
    );

    let headers = reported
        .iter()
        .filter_map(|progress| match progress.phase {
            CandidateInspectionPhase::ReadingHeaders { completed, total } => Some((completed, total)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let total = headers[0].1;
    assert!(total >= 4, "every GGUF of the repository has a header: {headers:?}");
    assert_eq!(
        headers,
        (0..=total).map(|completed| (completed, total)).collect::<Vec<_>>(),
        "each header read counts once"
    );
    assert!(
        reported[0].versions.iter().all(|row| row.assessment.is_none()),
        "files are named before any is checked"
    );

    let versions = reported
        .iter()
        .filter(|progress| matches!(progress.phase, CandidateInspectionPhase::CheckingVersions { .. }))
        .collect::<Vec<_>>();
    assert_eq!(versions.len(), inspection.versions.len() + 1);
    for (step, progress) in versions.iter().enumerate() {
        assert_eq!(
            progress.phase,
            CandidateInspectionPhase::CheckingVersions {
                completed: u32::try_from(step).unwrap(),
                total: u32::try_from(inspection.versions.len()).unwrap(),
            }
        );
        assert_eq!(
            progress.versions.iter().filter(|row| row.assessment.is_some()).count(),
            step,
            "one more verdict with each check"
        );
        assert_eq!(
            progress
                .versions
                .iter()
                .map(|row| (row.label.as_str(), row.size_bytes))
                .collect::<Vec<_>>(),
            inspection
                .versions
                .iter()
                .map(|version| (version.label.as_str(), version.size_bytes))
                .collect::<Vec<_>>(),
            "the rows are the versions the inspection returns"
        );
    }

    let drafters = reported
        .iter()
        .filter_map(|progress| match &progress.phase {
            CandidateInspectionPhase::CheckingDrafters { version, completed, total } => {
                Some((version.as_str(), *completed, *total))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(drafters.iter().all(|(version, completed, total)| *version == "Q4_K_M" && completed <= total));
    let (_, completed, total) = drafters.last().copied().expect("drafter checks are reported");
    assert!(total >= 1);
    assert_eq!(completed, total, "every planned drafter check finishes");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_local_file_is_identified_by_its_header_digest_unless_a_listed_model_has_it() {
    let folder = tempfile::tempdir().expect("model folder");
    let (header, length) = llama(7, Encoding::Q8_0).bytes();
    let path = folder.path().join("model.gguf");
    write_complete_bytes(&path, &header, length);
    let digest =
        magnitude_service_models::candidate_files::local_model_digest(&[path.clone()]).unwrap();
    let request = || CandidateInspectionRequest::Local { path: path.clone() };

    let unlisted = fixture(Vec::new()).await;
    let inspection = unlisted
        .candidates
        .inspect(request())
        .await
        .expect("inspection");
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
    let inspection = listed
        .candidates
        .inspect(request())
        .await
        .expect("inspection");
    let version = &inspection.versions[0];
    assert_eq!(version.model_id, found);
    assert_eq!(version.on_this_computer, Some(found));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_copy_of_a_listed_model_elsewhere_is_already_on_this_computer_as_that_model() {
    let folder = tempfile::tempdir().expect("model folders");
    let (bytes, length) = llama(7, Encoding::Q8_0).bytes();
    let found_path = folder
        .path()
        .join("lmstudio/community/Llama-GGUF/llama-Q8_0.gguf");
    let copy = folder.path().join("Downloads/llama-copy.gguf");
    for path in [&found_path, &copy] {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_complete_bytes(path, &bytes, length);
    }
    let found = ModelId::hugging_face(
        &HuggingFaceRepositoryId::new("community/Llama-GGUF".to_owned()).unwrap(),
        &magnitude_service_contracts::models::HuggingFaceArtifactSelector::new(
            "llama-Q8_0.gguf".to_owned(),
        )
        .unwrap(),
    );
    let mut listed = custom_model(found.clone(), length);
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
    let digest =
        magnitude_service_models::candidate_files::local_model_digest(&[copy.clone()]).unwrap();
    assert_eq!(
        inspection.versions[0].model_id.as_str(),
        format!("local:{digest}"),
        "the copy itself is identified by its header, so it can be recorded as one more location"
    );
}

/// Two files with the same header but different weights are two models: a file listed as one is
/// already on this computer as that model, never as its same-header twin listed before it, and
/// an unlisted file with a listed twin's header is not on this computer at all.
#[tokio::test(flavor = "multi_thread")]
async fn a_same_header_twin_with_other_weights_is_not_already_on_this_computer() {
    let folder = tempfile::tempdir().expect("model folders");
    let (mut bytes, length) = llama(7, Encoding::Q8_0).bytes();
    bytes.resize(bytes.len() + 8192, 0);
    let mut altered_bytes = bytes.clone();
    *altered_bytes.last_mut().unwrap() ^= 0xff;
    let twin = folder.path().join("B2/moved.gguf");
    let altered = folder.path().join("B/renamed-altered.gguf");
    for (path, contents) in [(&twin, &bytes), (&altered, &altered_bytes)] {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_complete_bytes(path, contents, length);
    }
    let digests = |path: &PathBuf| {
        (
            magnitude_service_models::candidate_files::local_header_digest(path).unwrap(),
            magnitude_service_models::candidate_files::local_model_digest(&[path.clone()])
                .unwrap(),
        )
    };
    let (twin_header, twin_digest) = digests(&twin);
    let (altered_header, altered_digest) = digests(&altered);
    assert_eq!(twin_header, altered_header);
    assert_ne!(twin_digest, altered_digest);
    let listed = |digest: &str, path: &PathBuf| {
        let mut model = custom_model(
            ModelId::local(
                &magnitude_service_contracts::models::LocalModelDigest::new(digest.to_owned())
                    .unwrap(),
            ),
            bytes.len() as u64,
        );
        model.files[0].path = path.clone();
        model
    };
    let request = || CandidateInspectionRequest::Local {
        path: altered.clone(),
    };

    let both = fixture(vec![
        listed(&twin_digest, &twin),
        listed(&altered_digest, &altered),
    ])
    .await;
    let inspection = both.candidates.inspect(request()).await.expect("inspection");
    assert_eq!(
        inspection.versions[0].on_this_computer.as_ref().map(ModelId::as_str),
        Some(format!("local:{altered_digest}").as_str())
    );

    let twin_only = fixture(vec![listed(&twin_digest, &twin)]).await;
    let inspection = twin_only
        .candidates
        .inspect(request())
        .await
        .expect("inspection");
    assert_eq!(inspection.versions[0].on_this_computer, None);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drafter_already_in_another_apps_folder_is_on_this_computer() {
    let fixture = fixture(Vec::new()).await;
    let candidates = candidate_drafters(&fixture, PLAIN, '3').await;
    assert!(!candidates[0].on_this_computer, "{candidates:#?}");

    let lm_studio = tempfile::tempdir().expect("LM Studio folder");
    let copy = lm_studio
        .path()
        .join("someone/Mirror-GGUF/renamed-dflash.gguf");
    std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
    let (header, length) = dflash(HIDDEN).bytes();
    std::fs::write(&copy, header).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&copy)
        .unwrap()
        .set_len(length)
        .unwrap();
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
    assert!(
        own[0].on_this_computer,
        "a repository drafter's bytes elsewhere count too"
    );
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

/// A tensor type id the engine doesn't know (a quantization newer than it, as PTQ1_0 is) is an
/// unsupported weight type, named by its id, never a damaged file; the header's metadata still gives
/// the shape, so the candidate is still a variant of its catalog model.
#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_tensor_type_is_an_unsupported_weight_type_and_keeps_the_variant() {
    let fixture = fixture(Vec::new()).await;
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new(LEGACY.to_owned()).unwrap(),
            revision: "main".to_owned(),
        })
        .await
        .expect("a file of an unknown weight type never fails the repository");

    assert_eq!(inspection.versions.len(), 2);
    assert_eq!(inspection.unsupported_architecture, None);
    assert_eq!(inspection.variant_of.as_ref(), Some(&fixture.catalog_id));
    let newer = version(&inspection, "llama-PTQ1_0.gguf");
    let Some(ModelAssessment::Unsupported { reasons, detail, .. }) = &newer.assessment else {
        panic!("the file is unsupported: {:?}", newer.assessment);
    };
    assert_eq!(
        reasons,
        &[UnsupportedReason::Known(KnownUnsupportedReason::WeightTypes {
            types: vec![magnitude_service_contracts::models::WeightTypeCount {
                ggml_type: "type 200".to_owned(),
                tensor_count: 1,
            }]
        })]
    );
    assert!(detail.contains("unknown GGUF tensor types 200"), "{detail}");
    assert_eq!(
        newer.model_id.as_str(),
        format!("hf:{LEGACY}/llama-PTQ1_0.gguf")
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
    let (header, length) = llama(7, Encoding::Q8_0).bytes();
    let blob = folder.path().join("blobs/0123abcd");
    std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
    write_complete_bytes(&blob, &header, length);
    let snapshot = folder.path().join("snapshots/commit");
    std::fs::create_dir_all(&snapshot).unwrap();
    let link = snapshot.join("model-Q8_0.gguf");
    std::os::unix::fs::symlink("../../blobs/0123abcd", &link).unwrap();
    let digest =
        magnitude_service_models::candidate_files::local_model_digest(&[blob.clone()]).unwrap();
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
        drafter_pending: false,
        recommended_drafter: None,
        license: None,
        profile: ServingProfile {
            context_length: 8192,
        },
        update_commit: None,
        state: CustomModelState::Unavailable {
            installation:
                magnitude_service_contracts::models::ResolvedModelInstallation::Resolved {
                    installed_bytes: size_bytes,
                    primary_path: PathBuf::from("/found/model.gguf"),
                    ownership:
                        magnitude_service_contracts::models::ModelInstallationOwnership::External,
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
        vision
            .projector
            .as_ref()
            .map(|projector| projector.path.as_str()),
        Some("mmproj-F16.gguf")
    );
    // QA (Linux CUDA, round 4): a projector the target's family can't bind (llama is text-only,
    // as Nemotron-H is for Nemotron 3.5 VL repositories) leaves the text model assessed without
    // it, the projector rejected with a mismatch, never an unreadable or unsupported version.
    match &vision.versions[0].assessment {
        Some(
            ModelAssessment::Fits { projector, .. } | ModelAssessment::DoesNotFit { projector, .. },
        ) => assert!(
            matches!(
                projector,
                ProjectorVerdict::Rejected { reasons, .. }
                    if matches!(
                        reasons.as_slice(),
                        [UnsupportedReason::Known(KnownUnsupportedReason::TargetMismatch { detail })]
                            if detail.contains("no vision projector")
                    )
            ),
            "{projector:?}"
        ),
        other => panic!("the text model is assessed without its projector: {other:?}"),
    }

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
        Some(vec![UnsupportedReason::Known(
            KnownUnsupportedReason::Architecture {
                architecture: "deepseek4".to_owned()
            }
        )])
    );
    assert!(
        deepseek.versions.iter().all(|version| matches!(
            &version.assessment,
            Some(ModelAssessment::Unsupported { reasons, .. })
                if reasons.contains(&UnsupportedReason::Known(KnownUnsupportedReason::Architecture {
                    architecture: "deepseek4".to_owned()
                }))
        )),
        "each version carries the engine's own reasons: {:?}",
        deepseek.versions
    );
}

/// A file whose header names no architecture (a bare tensor container, as diffusion GGUFs are)
/// is unreadable, never an architecture without a name, so the client can decode it.
#[tokio::test(flavor = "multi_thread")]
async fn a_file_naming_no_architecture_is_unreadable_never_an_empty_architecture() {
    let image = Gguf {
        metadata: Vec::new(),
        tensors: vec![("img_in.weight".to_owned(), vec![64, 64], Encoding::Q8_0)],
    };
    let fixture = fixture_with(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::from([(
            "someone/Image-GGUF".to_owned(),
            repository('8', vec![StubFile::new("image-Q8_0.gguf", &image)], &[]),
        )]),
    )
    .await;
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new("someone/Image-GGUF".to_owned()).unwrap(),
            revision: "main".to_owned(),
        })
        .await
        .expect("inspection");
    let unreadable = UnsupportedReason::Known(KnownUnsupportedReason::Unreadable);
    assert_eq!(inspection.unsupported_architecture, Some(vec![unreadable.clone()]));
    assert!(
        inspection.versions.iter().all(|version| matches!(
            &version.assessment,
            Some(ModelAssessment::Unsupported { reasons, .. })
                if reasons.contains(&unreadable)
                    && !reasons.iter().any(|reason| matches!(
                        reason,
                        UnsupportedReason::Known(KnownUnsupportedReason::Architecture { .. })
                    ))
        )),
        "{:?}",
        inspection.versions
    );
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
        .find(|candidate| candidate.automatic)
        .expect("an automatic drafter")
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
            if matches!(reasons.as_slice(), [UnsupportedReason::Known(KnownUnsupportedReason::TargetMismatch { .. })])
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
    assert!(matches!(
        candidates[3].verdict,
        DrafterVerdict::Bound { .. }
    ));
    assert_eq!(
        fixture.hub.drafter_searches.load(Ordering::SeqCst),
        0,
        "Hugging Face is never searched for drafters"
    );
}

/// A drafter the engine binds but with which a model that fits without one no longer fits is
/// listed, never chosen automatically: the next candidate (here the catalog drafter) is. A version's
/// verdict before adding it is the verdict of exactly what adding it installs: its target with
/// the preselected drafter, assessed the same way as that composition is afterwards.
#[tokio::test(flavor = "multi_thread")]
async fn a_drafter_that_stops_the_model_fitting_is_never_preselected_and_versions_carry_the_preselected_verdict()
 {
    const HEAVY: &str = "community/Llama-Heavy-Drafter-GGUF";
    let fixture = fixture_with(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::from([(
            HEAVY.to_owned(),
            repository(
                '7',
                vec![
                    StubFile::new("llama-Q8_0.gguf", &llama(7, Encoding::Q8_0)),
                    StubFile::new(
                        "dflash-heavy.gguf",
                        &dflash_with_feed_forward(HIDDEN, 1 << 27),
                    ),
                ],
                &["base/Llama"],
            ),
        )]),
    )
    .await;
    let candidates = candidate_drafters(&fixture, HEAVY, '7').await;
    let heavy = candidates
        .iter()
        .find(|candidate| matches!(&candidate.reference, DrafterReference::HuggingFace { path, .. } if path == "dflash-heavy.gguf"))
        .unwrap_or_else(|| panic!("the heavy drafter is listed: {candidates:#?}"));
    assert!(
        matches!(heavy.verdict, DrafterVerdict::Bound { .. }),
        "the engine binds it: {:?}",
        heavy.verdict
    );
    assert!(!heavy.automatic, "it would stop the model fitting");
    let chosen = preselected(&candidates);
    assert_eq!(
        chosen.reference,
        DrafterReference::Catalog {
            model_id: fixture.catalog_id.clone()
        }
    );

    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new(HEAVY.to_owned()).unwrap(),
            revision: commit('7'),
        })
        .await
        .expect("inspection");
    let before = version(&inspection, "llama-Q8_0.gguf")
        .assessment
        .clone()
        .expect("an assessment");
    let after = fixture
        .candidates
        .assess_candidate(ModelCandidate {
            drafter: Some(CandidateDrafter {
                reference: chosen.reference.clone(),
                method: chosen.method.clone().expect("a bound method"),
            }),
            ..hugging_face_candidate(HEAVY, '7', "llama-Q8_0.gguf")
        })
        .await
        .expect("the installed composition's assessment");
    assert!(
        matches!(
            &before,
            ModelAssessment::Fits { drafter: DrafterVerdict::Bound { .. }, .. }
        ),
        "{before:?}"
    );
    assert_eq!(before, after, "the verdict before adding is the verdict after");
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
            (
                DrafterCategory::NextToModel,
                Some(SpeculativeMethod::DFlash2)
            ),
            (
                DrafterCategory::NextToModel,
                Some(SpeculativeMethod::DSpark)
            ),
            (
                DrafterCategory::NextToModel,
                Some(SpeculativeMethod::DFlash)
            ),
            (
                DrafterCategory::CatalogDrafter,
                Some(SpeculativeMethod::DFlash)
            ),
        ],
        "{candidates:#?}"
    );
    assert_eq!(hugging_face_path(&candidates[2]), "drafters/dflash.gguf");
    assert_eq!(hugging_face_path(preselected(&candidates)), "dflash2.gguf");
    assert_eq!(fixture.hub.drafter_searches.load(Ordering::SeqCst), 0);
}

/// Same-named drafters in a repository's subfolders are listed under their whole paths, so their
/// rows are told apart.
#[tokio::test(flavor = "multi_thread")]
async fn same_named_drafters_in_repository_subfolders_are_named_by_their_paths() {
    const SUBFOLDERS: &str = "community/Llama-Subfolders-GGUF";
    let fixture = fixture_with(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::from([(
            SUBFOLDERS.to_owned(),
            repository(
                '8',
                vec![
                    StubFile::new("llama-Q8_0.gguf", &llama(7, Encoding::Q8_0)),
                    StubFile::new("Q4/dflash.gguf", &dflash(HIDDEN)),
                    StubFile::new("Q8/dflash.gguf", &dflash(HIDDEN)),
                ],
                &["community/Llama-Finetune"],
            ),
        )]),
    )
    .await;
    let candidates = candidate_drafters(&fixture, SUBFOLDERS, '8').await;
    let names = candidates
        .iter()
        .filter(|candidate| candidate.category == DrafterCategory::NextToModel)
        .map(|candidate| candidate.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names.iter().copied().collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from([
            "community/Llama-Subfolders-GGUF/Q4/dflash.gguf",
            "community/Llama-Subfolders-GGUF/Q8/dflash.gguf",
        ]),
        "{candidates:#?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_repository_drafter_the_catalog_drafter_is_preselected() {
    let fixture = fixture(Vec::new()).await;
    let candidates = candidate_drafters(&fixture, PLAIN, '3').await;
    assert_eq!(candidates.len(), 1, "{candidates:#?}");
    assert_eq!(
        preselected(&candidates).category,
        DrafterCategory::CatalogDrafter
    );

    let rejected = candidate_drafters(&fixture, REJECTED, '2').await;
    assert_eq!(
        rejected
            .iter()
            .map(|candidate| candidate.category)
            .collect::<Vec<_>>(),
        [
            DrafterCategory::NextToModel,
            DrafterCategory::CatalogDrafter
        ]
    );
    assert!(matches!(
        rejected[0].verdict,
        DrafterVerdict::Rejected { .. }
    ));
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
    let write = |name: &str, gguf: &Gguf| {
        let (header, length) = gguf.bytes();
        write_complete_bytes(&folder.path().join(name), &header, length);
    };
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
            (DrafterCategory::CatalogDrafter, "Untested · Trained for Llama Test"),
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
    assert!(!catalog.automatic, "a shape match alone is untested");
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
    assert_eq!(
        exact.len(),
        1,
        "an exact catalog target uses only its catalog drafter: {exact:#?}"
    );
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
async fn inspecting_a_catalog_repository_names_the_catalog_model_a_version_is() {
    let fixture = fixture(Vec::new()).await;
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new(CATALOG_TARGET.to_owned()).unwrap(),
            revision: commit('a'),
        })
        .await
        .expect("inspection");
    let version = version(&inspection, "llama-Q8_0.gguf");
    assert_eq!(version.catalog_model.as_ref(), Some(&fixture.catalog_id));
    assert_eq!(
        version.on_this_computer, None,
        "a catalog model not installed is not on this computer"
    );
}

/// QA (mac-m4, round 3c): pulling `openbmb/MiniCPM-V-4.7-1B-gguf:Q4_K_M` at the repository's
/// newer main commit declared an added model, but its download held exactly the catalog target's
/// bytes, so it installed as the catalog model and the added entry stayed "Assessing" forever. A
/// version with the catalog target's bytes, at any commit of its repository, is that catalog
/// model; other bytes or another repository are not.
#[tokio::test(flavor = "multi_thread")]
async fn a_version_with_the_catalog_targets_bytes_at_another_commit_is_the_catalog_model() {
    let fixture = fixture(Vec::new()).await;
    let ServableModelBundle::SpeculativeDecoding { target, .. } =
        &fixture.lineage.catalog().models[0].configuration.bundle
    else {
        panic!("the catalog model has a drafter");
    };
    let weights = target
        .files
        .iter()
        .filter(|file| file.role == magnitude_service_contracts::models::ModelFileRole::Weights)
        .map(|file| (file.path.clone(), file.sha256.clone()))
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        fixture
            .lineage
            .target_with_weights(&CATALOG_TARGET.to_uppercase(), &weights)
            .map(|matched| matched.model_id),
        Some(fixture.catalog_id.clone())
    );
    assert_eq!(
        fixture
            .lineage
            .target_with_weights(COMMUNITY, &weights)
            .map(|matched| matched.model_id),
        None,
        "the same bytes in another repository are found by content elsewhere, not here"
    );
    let other_bytes = weights
        .iter()
        .map(|(path, _)| (path.clone(), "f".repeat(64)))
        .collect();
    assert_eq!(
        fixture
            .lineage
            .target_with_weights(CATALOG_TARGET, &other_bytes)
            .map(|matched| matched.model_id),
        None
    );
    assert_eq!(
        fixture
            .lineage
            .target_with_weights(CATALOG_TARGET, &Default::default())
            .map(|matched| matched.model_id),
        None
    );
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
    assert_eq!(
        version.download_bytes,
        version.size_bytes + projector.size_bytes
    );

    let second = fixture
        .candidates
        .inspect(request)
        .await
        .expect("cached inspection");
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
        hugging_face_package(&fixture.models, CATALOG_DRAFT, "dflash.gguf")
            .await
            .id
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
fn enrichment_for_a_changed_target_keeps_the_models_last_answer_until_it_is_recomputed() {
    use super::Reach::Hub;
    use magnitude_service_models::{
        CustomModelEnrichment, CustomModelEnrichmentResult, CustomModelFacts,
    };
    let enrichment = super::CandidateEnrichment::default();
    let model_id = ModelId::local(
        &magnitude_service_contracts::models::LocalModelDigest::new("a".repeat(64)).unwrap(),
    );
    let result = CustomModelEnrichmentResult {
        variant_of: Some("llama-test:gguf:q8".parse().unwrap()),
        drafters: Vec::new(),
        catalog_drafters: Vec::new(),
        license: Some("apache-2.0".to_owned()),
    };
    let key = (model_id.clone(), "package_a".to_owned());
    assert!(enrichment.record(key.clone(), 1, Hub, result.clone()));
    assert!(enrichment.is_current(&key, 1, Hub));
    assert!(
        !enrichment.is_current(&key, 2, Hub),
        "a result is stale once the inputs it was computed from change"
    );
    assert!(!enrichment.record(key.clone(), 2, Hub, result.clone()));
    assert!(enrichment.is_current(&key, 2, Hub));
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
    };
    let (package_a, package_b) = (package("package_a"), package("package_b"));
    assert_eq!(enrichment.enrich(&facts(&package_a)), result);
    assert!(enrichment.knows(&facts(&package_a)));
    // QA (Debian, round 3b): a projector appearing beside a model changed its target package, and
    // the model showed "Drafter None" while its heads drafted, until the recompute finished.
    assert_eq!(
        enrichment.enrich(&facts(&package_b)),
        result,
        "the model's last answer stands while its new target is computed"
    );
    assert!(
        !enrichment.knows(&facts(&package_b)),
        "the new target is still pending"
    );
    assert!(enrichment.record(
        (model_id.clone(), "package_b".to_owned()),
        2,
        Hub,
        CustomModelEnrichmentResult::default()
    ));
    assert_eq!(
        enrichment.enrich(&facts(&package_b)),
        CustomModelEnrichmentResult::default()
    );
}

/// QA (macOS, round 4): what the files here say is published before the Hub answers, and stays
/// pending until it does; the Hub's answer then replaces it and is never replaced by it.
#[test]
fn a_local_enrichment_answers_until_the_hub_reaching_one_and_never_replaces_it() {
    use super::Reach::{Hub, Local};
    use magnitude_service_models::{
        CustomModelEnrichment, CustomModelEnrichmentResult, CustomModelFacts,
    };
    let enrichment = super::CandidateEnrichment::default();
    let model_id = ModelId::local(
        &magnitude_service_contracts::models::LocalModelDigest::new("b".repeat(64)).unwrap(),
    );
    let local = CustomModelEnrichmentResult {
        variant_of: Some("llama-test:gguf:q8".parse().unwrap()),
        ..CustomModelEnrichmentResult::default()
    };
    let hub = CustomModelEnrichmentResult {
        license: Some("apache-2.0".to_owned()),
        ..local.clone()
    };
    let key = (model_id.clone(), "package".to_owned());
    let origin = CustomModelOrigin::Imported {
        path: PathBuf::from("/models/model.gguf"),
    };
    let package = magnitude_service_contracts::models::ModelPackage {
        id: magnitude_service_contracts::models::ModelPackageId("package".to_owned()),
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
    let facts = CustomModelFacts {
        model_id: &model_id,
        origin: &origin,
        display_name: "model",
        target: &package,
        hub: None,
        ollama_name: None,
        name: None,
        architecture: Some("llama"),
        base_models: &[],
    };
    assert!(enrichment.record(key.clone(), 1, Local, local.clone()));
    assert_eq!(enrichment.enrich(&facts), local);
    assert!(
        !enrichment.knows(&facts),
        "a local answer leaves the automatic drafter pending"
    );
    assert!(enrichment.is_current(&key, 1, Local));
    assert!(!enrichment.is_current(&key, 1, Hub));
    assert!(enrichment.record(key.clone(), 1, Hub, hub.clone()));
    assert_eq!(enrichment.enrich(&facts), hub);
    assert!(enrichment.knows(&facts));
    assert!(
        enrichment.is_current(&key, 2, Local),
        "once the Hub has answered, the local pass leaves the model to it"
    );
    assert!(!enrichment.record(key.clone(), 2, Local, local));
    assert_eq!(
        enrichment.enrich(&facts),
        hub,
        "the Hub's answer is never replaced locally"
    );
}

/// QA saw a cold inspection of multi-version repositories pull 450 MB to 1.4 GB, far beyond
/// their headers. A cold inspection, with the drafter list a CLI inspection asks for, reads each
/// file's own header bytes and the shared tokenizer once, whatever order the metadata is in, and a
/// split version's other shards only as far as their tensor directories.
#[tokio::test(flavor = "multi_thread")]
async fn a_cold_inspection_reads_about_the_distinct_header_bytes_of_every_layout() {
    let fixture = fixture(Vec::new()).await;
    let files = layout_files();
    let carrying = files
        .iter()
        .filter(|file| {
            file.header
                .windows(b"tokenizer.ggml.merges".len())
                .any(|window| window == b"tokenizer.ggml.merges")
        })
        .count();
    // The MTP drafter's token types are its own.
    let own_token_types = layout_tokenizer()
        .iter()
        .find(|(key, _)| key == "tokenizer.ggml.token_type")
        .map(|(key, value)| Gguf::entry(key, value).len())
        .unwrap();
    let needed = files.iter().map(|file| file.header.len()).sum::<usize>()
        - (carrying - 1) * layout_shared_bytes()
        + own_token_types;
    let before = fixture.hub.header_bytes.load(Ordering::SeqCst);
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new(LAYOUTS.to_owned()).unwrap(),
            revision: "main".to_owned(),
        })
        .await
        .expect("inspection");
    fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: hugging_face_candidate(LAYOUTS, '5', "Llama-Layouts-Q8_0.gguf"),
            other: None,
        })
        .await
        .expect("drafter candidates");
    let served = fixture.hub.header_bytes.load(Ordering::SeqCst) - before;
    eprintln!(
        "cold inspection of {} files: {served} bytes served for {needed} distinct header bytes ({:.2}x); every header whole is {} bytes",
        files.len(),
        served as f64 / needed as f64,
        files.iter().map(|file| file.header.len()).sum::<usize>(),
    );
    assert!(carrying >= 7, "{carrying} files carry the shared tokenizer");
    for file in [
        "Llama-Layouts-BF16.gguf",
        "Llama-Layouts-Q8_0.gguf",
        "Llama-Layouts-Q4_K_M.gguf",
        "Llama-Layouts-Q2_K.gguf",
        "Q6_K/Llama-Layouts-Q6_K-00001-of-00003.gguf",
    ] {
        let assessment = &version(&inspection, file).assessment;
        assert!(
            assessment.as_ref().is_some_and(|assessment| !matches!(
                assessment,
                ModelAssessment::Unsupported { reasons, .. } if reasons.contains(&UnsupportedReason::Known(KnownUnsupportedReason::Unreadable))
            )),
            "{file} is assessed from its exact header: {assessment:?}"
        );
    }
    assert!(matches!(
        &version(&inspection, "Llama-Layouts-IQ1_M.gguf").assessment,
        Some(ModelAssessment::Unsupported { reasons, .. }) if reasons == &[UnsupportedReason::Known(KnownUnsupportedReason::WeightTypes {
            types: vec![magnitude_service_contracts::models::WeightTypeCount {
                ggml_type: "type 142".to_owned(),
                tensor_count: 1,
            }]
        })]
    ), "a tensor type id the engine doesn't know is an unsupported weight type");
    assert!(
        served <= 2 * needed,
        "{served} bytes served for {needed} distinct header bytes"
    );
    let snapshot = refresh_hugging_face_repository(
        &fixture.models,
        HuggingFaceRepositoryRequest {
            repository: LAYOUTS.to_owned(),
            revision: commit('5'),
        },
    )
    .await
    .expect("snapshot");
    for file in &files {
        let listed = snapshot
            .gguf_files
            .iter()
            .find(|listed| listed.path == PathBuf::from(&file.path))
            .expect("listed");
        let expected = format!("{:x}", Sha256::digest(&file.header));
        let digest = fixture
            .models
            .cached_hugging_face_header_digest(LAYOUTS, &snapshot.commit, listed)
            .or_else(|| {
                fixture
                    .models
                    .hugging_face_reference_header(LAYOUTS, &snapshot.commit, listed)
                    .map(|bytes| format!("{:x}", Sha256::digest(&bytes)))
            });
        assert_eq!(digest, Some(expected), "{} is read exactly", file.path);
    }
}

/// A local inspection reads each file once (its header and the two ends of its tensor data) and
/// never again for the engine, its drafters, its lineage or its identity; an unchanged file is not
/// read at all by a later inspection or drafter listing.
#[tokio::test(flavor = "multi_thread")]
async fn a_local_inspection_reads_each_file_at_most_once() {
    use magnitude_service_models::gguf::{CONTENT_SAMPLE_BYTES, LocalReads};
    let fixture = fixture(Vec::new()).await;
    let folder = tempfile::tempdir().expect("model folder");
    let model = folder.path().join("model.gguf");
    let drafter = folder.path().join("dflash.gguf");
    let target = llama(7, Encoding::Q8_0);
    write_complete(&model, &target);
    write_complete(&drafter, &dflash(HIDDEN));
    let model_reads = LocalReads::watch(&model);
    let drafter_reads = LocalReads::watch(&drafter);

    let request = || CandidateInspectionRequest::Local {
        path: model.clone(),
    };
    let first = fixture.candidates.inspect(request()).await.expect("inspection");
    assert!(
        matches!(
            first.versions[0].assessment,
            Some(ModelAssessment::Fits { .. } | ModelAssessment::DoesNotFit { .. })
        ),
        "{first:#?}"
    );
    let (header, length) = target.bytes();
    let (reads, bytes) = model_reads.reads();
    assert_eq!(reads, 1, "the model is read once");
    assert!(
        bytes <= length && bytes <= header.len() as u64 + 3 * CONTENT_SAMPLE_BYTES,
        "only the header and the two ends of the tensor data are read: {bytes} of {length}"
    );
    assert_eq!(drafter_reads.reads().0, 1, "the drafter beside it is read once");

    let second = fixture.candidates.inspect(request()).await.expect("inspection");
    assert_eq!(first, second);
    fixture
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
        .expect("drafter candidates");
    assert_eq!(model_reads.reads().0, 1, "an unchanged model is not read again");
    assert_eq!(drafter_reads.reads().0, 1, "an unchanged drafter is not read again");
}

fn write_complete(path: &std::path::Path, gguf: &Gguf) {
    let (header, length) = gguf.bytes();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, header).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(length)
        .unwrap();
}

/// llama.cpp keeps every repository's files flat in one root as `<owner>_<repository>_<file>`. A
/// model whose repository's snapshot can't be read is offered only its own repository's flat
/// drafters, never another repository's beside it; with the snapshot read, the repository's own
/// drafter list replaces the folder entirely.
#[tokio::test(flavor = "multi_thread")]
async fn a_flat_llama_cpp_root_offers_only_the_models_own_repository_drafters() {
    use magnitude_service_contracts::models::FoundSource;
    let fixture = fixture(Vec::new()).await;
    let root = tempfile::tempdir().expect("llama.cpp root");
    let model = root.path().join("owner_Llama-GGUF_llama-Q8_0.gguf");
    write_complete(&model, &llama(7, Encoding::Q8_0));
    write_complete(
        &root.path().join("owner_Llama-GGUF_dflash.gguf"),
        &dflash(HIDDEN),
    );
    write_complete(
        &root.path().join("other_Qwen-GGUF_dflash.gguf"),
        &dflash(HIDDEN),
    );
    let found = CustomModelOrigin::Found {
        source: FoundSource::LlamaCpp,
    };

    assert!(
        super::drafter_folder(&found, Some("owner/Llama-GGUF"), true, Some(root.path())).is_none(),
        "with its snapshot read, a known repository lists its own drafters, not the folder"
    );
    let added = CustomModelOrigin::Added {
        repository: HuggingFaceRepositoryId::new("owner/Llama-GGUF".to_owned()).unwrap(),
        commit: commit('a'),
    };
    assert!(
        super::drafter_folder(&added, Some("owner/Llama-GGUF"), false, Some(root.path())).is_none()
    );

    let target = super::DrafterTarget {
        material: fixture
            .candidates
            .local_target(model.clone(), None)
            .expect("target material"),
        repository: Some(("owner/Llama-GGUF".to_owned(), String::new())),
        primary: None,
        folders: super::drafter_folder(&found, Some("owner/Llama-GGUF"), false, Some(root.path()))
            .into_iter()
            .collect(),
        snapshot: None,
        lineage: Default::default(),
        catalog: None,
        reach: super::Reach::Hub,
        weights: None,
    };
    let offered = fixture
        .candidates
        .next_to_model(&target, None, &crate::assessment::inspection_progress::InspectionProgress::silent())
        .await;
    assert_eq!(
        offered
            .iter()
            .map(|candidate| candidate.reference.clone())
            .collect::<Vec<_>>(),
        [DrafterReference::Local {
            path: root.path().join("owner_Llama-GGUF_dflash.gguf")
        }],
        "{offered:#?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drafter_in_a_sibling_subfolder_is_not_in_this_folder() {
    let fixture = fixture(Vec::new()).await;
    let folder = tempfile::tempdir().expect("model folder");
    write_complete(&folder.path().join("model.gguf"), &llama(7, Encoding::Q8_0));
    write_complete(&folder.path().join("dflash.gguf"), &dflash(HIDDEN));
    write_complete(&folder.path().join("other/dflash.gguf"), &dflash(HIDDEN));
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
    let local = candidates
        .iter()
        .filter(|candidate| candidate.category == DrafterCategory::NextToModel)
        .map(|candidate| candidate.reference.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        local,
        [DrafterReference::Local {
            path: folder.path().join("dflash.gguf")
        }]
    );
}

/// A local Llama file whose header names `base` as its base model.
/// The Llama Q8_0 file declaring `base` as its base model.
fn llama_with_base(base: &str) -> Gguf {
    let mut gguf = llama(7, Encoding::Q8_0);
    gguf.set("general.base_model.count", Meta::U32(1));
    gguf.set(
        "general.base_model.0.repo_url",
        Meta::Str(format!("https://huggingface.co/{base}")),
    );
    gguf
}

fn local_llama_with_base(base: &str) -> (tempfile::TempDir, ModelCandidate) {
    let folder = tempfile::tempdir().expect("model folder");
    write_complete(&folder.path().join("model.gguf"), &llama_with_base(base));
    let candidate = ModelCandidate {
        source: CandidateSource::Local {
            path: folder.path().to_path_buf(),
        },
        target: vec!["model.gguf".to_owned()],
        projector: None,
        drafter: None,
    };
    (folder, candidate)
}

/// "Variant of" is the shape match alone, as in search: a file naming a base model the catalog
/// doesn't know is still a variant of the catalog model whose shape it has, and so is a file
/// declaring no lineage. Its fingerprint also offers that catalog model's drafter.
#[tokio::test(flavor = "multi_thread")]
async fn variant_of_is_the_shape_match_whatever_lineage_a_file_declares() {
    let fixture = fixture(Vec::new()).await;
    let (unrelated, candidate) = local_llama_with_base("someone/Unrelated-Base");
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::Local {
            path: unrelated.path().join("model.gguf"),
        })
        .await
        .expect("inspection");
    assert_eq!(inspection.variant_of.as_ref(), Some(&fixture.catalog_id));
    let drafters = drafters_of(&fixture, candidate).await;
    assert!(
        drafters.iter().any(|candidate| candidate.category == DrafterCategory::CatalogDrafter
            && candidate.lineage == DrafterLineage::Fingerprint),
        "{drafters:#?}"
    );

    let plain = tempfile::tempdir().expect("model folder");
    write_complete(&plain.path().join("model.gguf"), &llama(7, Encoding::Q8_0));
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::Local {
            path: plain.path().join("model.gguf"),
        })
        .await
        .expect("inspection");
    assert_eq!(inspection.variant_of.as_ref(), Some(&fixture.catalog_id));
}

async fn drafters_of(
    fixture: &Fixture,
    candidate: ModelCandidate,
) -> Vec<magnitude_service_contracts::models::DrafterCandidate> {
    fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate,
            other: None,
        })
        .await
        .expect("drafter candidates")
        .candidates
}

/// `general.base_model` drives the catalog match, and when two catalog models share the lineage
/// the one whose shape fingerprint equals the target's wins over catalog order.
#[tokio::test(flavor = "multi_thread")]
async fn of_two_lineage_matches_the_one_with_the_targets_fingerprint_supplies_the_drafter() {
    let fixture = fixture_with_other_catalog_model().await;
    let (_folder, candidate) = local_llama_with_base("base/Llama");
    let candidates = drafters_of(&fixture, candidate).await;
    let catalog = candidates
        .iter()
        .filter(|candidate| candidate.category == DrafterCategory::CatalogDrafter)
        .map(|candidate| (candidate.label.as_str(), candidate.lineage))
        .collect::<Vec<_>>();
    assert_eq!(
        catalog,
        [("Trained for Llama Test", DrafterLineage::Lineage)],
        "{candidates:#?}"
    );
}

/// When the lineage-matched catalog model's drafter is rejected, the fingerprint-matched catalog
/// model's drafter is offered too, untested and never preselected.
#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_lineage_drafter_falls_back_to_the_fingerprint_matched_catalog_drafter() {
    let fixture = fixture_with_other_catalog_model().await;
    let (_folder, candidate) = local_llama_with_base(OTHER_SHAPE);
    let candidates = drafters_of(&fixture, candidate).await;
    let catalog = candidates
        .iter()
        .filter(|candidate| candidate.category == DrafterCategory::CatalogDrafter)
        .map(|candidate| {
            (
                candidate.label.as_str(),
                candidate.lineage,
                is_bound(&candidate.verdict),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        catalog,
        [
            ("Trained for Other Llama", DrafterLineage::Lineage, false),
            (
                "Untested · Trained for Llama Test",
                DrafterLineage::Fingerprint,
                true
            ),
        ],
        "{candidates:#?}"
    );
    assert!(
        candidates.iter().all(|candidate| !candidate.automatic),
        "{candidates:#?}"
    );
}

fn is_bound(verdict: &DrafterVerdict) -> bool {
    matches!(verdict, DrafterVerdict::Bound { .. })
}

/// A listed model added at one commit is not the same version at another commit of its
/// repository, even with the same Hub ID and file sizes; with no commit known, it still is.
#[tokio::test]
async fn a_version_at_another_commit_is_not_the_added_model_at_its_own() {
    use magnitude_service_models::candidate_files::{CandidateGguf, GgufGroup};
    let model_id = ModelId::hugging_face(
        &HuggingFaceRepositoryId::new(COMMUNITY.to_owned()).unwrap(),
        &magnitude_service_contracts::models::HuggingFaceArtifactSelector::new(
            "model.gguf".to_owned(),
        )
        .unwrap(),
    );
    let group = GgufGroup {
        files: vec![CandidateGguf {
            path: PathBuf::from("model.gguf"),
            size_bytes: 100,
        }],
    };
    let mut added = custom_model(model_id.clone(), 100);
    added.origin = CustomModelOrigin::Added {
        repository: HuggingFaceRepositoryId::new(COMMUNITY.to_owned()).unwrap(),
        commit: commit('a'),
    };
    let (_directory, store) = bare_store().await;
    let at = |commit: &str, model: &CustomModel| {
        super::on_this_computer(
            &store,
            std::slice::from_ref(model),
            Some((&model_id, commit)),
            &group,
            &"0".repeat(64),
            None,
            None,
        )
    };
    assert_eq!(at(&commit('a'), &added), Some(model_id.clone()));
    assert_eq!(at(&commit('b'), &added), None);
    let found = custom_model(model_id.clone(), 100);
    assert_eq!(at(&commit('b'), &found), Some(model_id.clone()));
}

// Shared real-header fixtures ---------------------------------------------------------------

/// One `inference/engine/fixtures/assessment` component.
#[derive(serde::Deserialize, Clone)]
struct FixtureComponent {
    header: String,
    repository: String,
    revision: String,
    path: String,
}

#[derive(serde::Deserialize)]
struct AssessmentFixture {
    name: String,
    target: Vec<FixtureComponent>,
    draft: Option<FixtureComponent>,
    expected: serde_json::Value,
}

fn fixtures_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engine/fixtures/assessment")
}

/// Search reads a shape from a header's first bytes: every fixture target whose fingerprint the
/// engine computes from its header alone has that fingerprint's shape, and the shape of its first
/// bytes is the same whenever they hold one.
#[test]
fn search_shapes_agree_with_engine_fingerprints() {
    use magnitude_service_models::{ModelShape, SHAPE_PREFIX_BYTES};
    use std::io::Read;
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixtures_root().join("manifest.json")).expect("manifest"),
    )
    .expect("manifest json");
    let fixtures: Vec<AssessmentFixture> =
        serde_json::from_value(manifest["fixtures"].clone()).expect("fixtures");
    let (mut compared, mut from_prefix) = (0, 0);
    for fixture in fixtures {
        let mut header = Vec::new();
        flate2::read::GzDecoder::new(
            std::fs::File::open(fixtures_root().join(&fixture.target[0].header)).unwrap(),
        )
        .read_to_end(&mut header)
        .unwrap();
        let Ok(fingerprint) = magnitude_engine::fingerprint::ModelFingerprint::from_directory(
            &magnitude_artifacts::gguf::read_header(&header).unwrap(),
        ) else {
            continue;
        };
        let fingerprint = crate::assessment::assessor::model_fingerprint(fingerprint);
        let shape = ModelShape::from_header_prefix(&header)
            .unwrap_or_else(|| panic!("{} has a shape", fixture.name));
        assert!(shape.matches(&fingerprint), "{}: {shape:?} {fingerprint:?}", fixture.name);
        compared += 1;
        if let Some(prefix) =
            ModelShape::from_header_prefix(&header[..header.len().min(SHAPE_PREFIX_BYTES)])
        {
            assert_eq!(prefix, shape, "{}", fixture.name);
            from_prefix += 1;
        }
    }
    assert!(compared > 10, "{compared} fixtures compared");
    assert!(from_prefix > 10, "{from_prefix} of {compared} shapes read from a prefix");
}

/// A shared fixture's header (`headers/<name>.gz`) served as the repository file `path`, sized as
/// its tensor directory declares.
fn fixture_file(header: &str, path: &str) -> StubFile {
    use std::io::Read;
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(
        std::fs::File::open(fixtures_root().join(header)).expect("fixture header"),
    )
    .read_to_end(&mut bytes)
    .expect("gzipped header");
    let directory = magnitude_artifacts::gguf::read_header(&bytes).expect("GGUF header");
    let size_bytes = directory.data_offset
        + directory
            .tensors
            .iter()
            .map(|tensor| tensor.offset + tensor.nbytes)
            .max()
            .unwrap_or(0);
    StubFile {
        path: path.to_owned(),
        header: bytes,
        size_bytes,
    }
}

/// The contract form of a reason in the engine's serde form: `"Unreadable"`, or
/// `{ "WeightTypes": { "types": [{ "ggml_type": .. }] } }` with snake-case fields.
fn contract_reason(engine: &serde_json::Value) -> serde_json::Value {
    fn camel(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(fields) => fields
                .iter()
                .map(|(key, value)| {
                    let mut parts = key.split('_');
                    let first = parts.next().unwrap_or_default().to_owned();
                    let key = parts.fold(first, |mut camel, part| {
                        let mut chars = part.chars();
                        if let Some(head) = chars.next() {
                            camel.extend(head.to_uppercase());
                            camel.push_str(chars.as_str());
                        }
                        camel
                    });
                    (key, camel(value))
                })
                .collect::<serde_json::Map<_, _>>()
                .into(),
            serde_json::Value::Array(values) => values.iter().map(camel).collect(),
            other => other.clone(),
        }
    }
    match engine {
        serde_json::Value::String(tag) => serde_json::json!({ "_tag": tag }),
        serde_json::Value::Object(variant) => {
            let (tag, fields) = variant.iter().next().expect("one variant");
            let mut reason = camel(fields).as_object().cloned().unwrap_or_default();
            if tag == "KernelDomain"
                && let Some(serde_json::Value::Array(statics)) = reason.get_mut("statics")
            {
                for entry in statics.iter_mut() {
                    if let serde_json::Value::Array(pair) = entry.clone() {
                        *entry = serde_json::json!({ "name": pair[0], "value": pair[1] });
                    }
                }
            }
            reason.insert("_tag".to_owned(), serde_json::json!(tag));
            reason.into()
        }
        other => panic!("unexpected reason form {other}"),
    }
}

/// Whether actual reasons are the expected ones: equal, except that a `TargetMismatch` detail
/// only has to occur in the actual one.
fn reasons_match(actual: &[UnsupportedReason], expected: &[serde_json::Value]) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(actual, expected)| {
            let actual = serde_json::to_value(actual).unwrap();
            let expected = contract_reason(expected);
            match (actual.get("_tag"), expected.get("detail")) {
                (Some(tag), Some(detail)) if tag == "TargetMismatch" => {
                    expected.get("_tag") == Some(tag)
                        && actual["detail"]
                            .as_str()
                            .is_some_and(|actual| actual.contains(detail.as_str().unwrap()))
                }
                _ => actual == expected,
            }
        })
}

fn method_name(method: &SpeculativeMethod) -> String {
    format!("{method:?}").to_lowercase()
}

/// The shared real-header fixtures, served from a stub Hugging Face hub and inspected through
/// the service: every target's verdict and unsupported reasons, and each listed drafter's verdict
/// (a separate draft's, or the built-in heads'), are the manifest's. Fingerprints are not compared:
/// the fixtures' token strings are blanked.
#[tokio::test(flavor = "multi_thread")]
async fn the_shared_real_header_fixtures_get_their_manifest_verdicts_through_the_service() {
    let root = fixtures_root();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).expect("manifest"))
            .expect("manifest JSON");
    let fixtures: Vec<AssessmentFixture> =
        serde_json::from_value(manifest["fixtures"].clone()).expect("fixtures");

    let mut repositories = BTreeMap::<String, StubRepository>::new();
    for component in fixtures
        .iter()
        .flat_map(|fixture| fixture.target.iter().chain(fixture.draft.iter()))
    {
        let StubFile {
            header, size_bytes, ..
        } = fixture_file(&component.header, &component.path);
        let repository = repositories
            .entry(component.repository.clone())
            .or_insert_with(|| StubRepository {
                commit: component.revision.clone(),
                files: Vec::new(),
                card_base_models: Vec::new(),
            });
        assert_eq!(
            repository.commit, component.revision,
            "one commit per repository"
        );
        if !repository
            .files
            .iter()
            .any(|file| file.path == component.path)
        {
            repository.files.push(StubFile {
                path: component.path.clone(),
                header,
                size_bytes,
            });
        }
    }
    let fixture = fixture_with(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        repositories,
    )
    .await;

    for case in &fixtures {
        let primary = &case.target[0];
        let inspection = fixture
            .candidates
            .inspect(CandidateInspectionRequest::HuggingFace {
                repository: HuggingFaceRepositoryId::new(primary.repository.clone()).unwrap(),
                revision: primary.revision.clone(),
            })
            .await
            .unwrap_or_else(|error| panic!("{}: inspection failed: {error}", case.name));
        let version = version(&inspection, &primary.path);
        let assessment = version
            .assessment
            .as_ref()
            .unwrap_or_else(|| panic!("{}: no assessment", case.name));
        match &case.expected["target"] {
            serde_json::Value::String(supported) if supported == "supported" => assert!(
                matches!(
                    assessment,
                    ModelAssessment::Fits { .. } | ModelAssessment::DoesNotFit { .. }
                ),
                "{}: {assessment:?}",
                case.name
            ),
            expected => {
                let expected = expected["unsupported"]
                    .as_array()
                    .expect("unsupported reasons");
                let ModelAssessment::Unsupported { reasons, .. } = assessment else {
                    panic!("{}: expected unsupported, got {assessment:?}", case.name);
                };
                assert!(
                    reasons_match(reasons, expected),
                    "{}: {reasons:?} != {expected:?}",
                    case.name
                );
                continue;
            }
        }

        let candidates = fixture
            .candidates
            .drafters(DrafterCandidatesRequest::Candidate {
                candidate: ModelCandidate {
                    source: CandidateSource::HuggingFace {
                        repository: HuggingFaceRepositoryId::new(primary.repository.clone())
                            .unwrap(),
                        commit: primary.revision.clone(),
                    },
                    target: case
                        .target
                        .iter()
                        .map(|component| component.path.clone())
                        .collect(),
                    projector: None,
                    drafter: None,
                },
                other: case
                    .draft
                    .as_ref()
                    .map(|draft| format!("{}@{}:{}", draft.repository, draft.revision, draft.path)),
            })
            .await
            .unwrap_or_else(|error| panic!("{}: drafters failed: {error}", case.name))
            .candidates;
        let listed = match &case.draft {
            Some(draft) => candidates.iter().find(|candidate| {
                matches!(
                    &candidate.reference,
                    DrafterReference::HuggingFace { repository, path, .. }
                        if repository.as_str() == draft.repository && *path == draft.path
                )
            }),
            None => candidates
                .iter()
                .find(|candidate| candidate.category == DrafterCategory::BuiltIn),
        };
        match (&case.expected["drafter"], listed) {
            (serde_json::Value::Null, None) => {}
            (serde_json::Value::String(none), None) if none == "none" => {}
            (expected, Some(candidate)) if expected.get("bound").is_some() => {
                let DrafterVerdict::Bound { method, .. } = &candidate.verdict else {
                    panic!("{}: expected bound, got {:?}", case.name, candidate.verdict);
                };
                assert_eq!(
                    method_name(method),
                    expected["bound"].as_str().unwrap(),
                    "{}",
                    case.name
                );
            }
            (expected, Some(candidate)) if expected.get("rejected").is_some() => {
                let DrafterVerdict::Rejected { reasons, .. } = &candidate.verdict else {
                    panic!(
                        "{}: expected rejected, got {:?}",
                        case.name, candidate.verdict
                    );
                };
                assert!(
                    reasons_match(reasons, expected["rejected"].as_array().unwrap()),
                    "{}: {reasons:?}",
                    case.name
                );
            }
            (expected, listed) => panic!("{}: expected {expected}, listed {listed:?}", case.name),
        }
    }
}

/// HauhauCS's Qwen3.8 27B MTP repository ships a FastMTP file: a `qwen35` prediction head alone,
/// declaring the target's metadata and `general.file_type` (Q3_K_M). Its header makes it a drafter,
/// so it is never a version; the engine doesn't bind a separate `qwen35` head, so the drafter list
/// shows it Rejected with the engine's reasons. Version labels come from each header's file type.
#[tokio::test(flavor = "multi_thread")]
async fn a_fastmtp_file_is_a_rejected_drafter_never_a_version() {
    const REPOSITORY: &str = "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF";
    const REVISION: &str = "993a5971fda8f30dd1b7eb2654792ba4415c7460";
    const TARGET: &str = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ3_XS.gguf";
    const FASTMTP: &str = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf";
    let served = |path: &str| fixture_file(&format!("headers/{path}.gz"), path);
    let fixture = fixture_with(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::from([(
            REPOSITORY.to_owned(),
            StubRepository {
                commit: REVISION.to_owned(),
                files: vec![served(TARGET), served(FASTMTP)],
                card_base_models: Vec::new(),
            },
        )]),
    )
    .await;
    let repository = HuggingFaceRepositoryId::new(REPOSITORY.to_owned()).unwrap();
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: repository.clone(),
            revision: REVISION.to_owned(),
        })
        .await
        .expect("inspection");
    assert_eq!(
        inspection
            .versions
            .iter()
            .map(|version| version.label.as_str())
            .collect::<Vec<_>>(),
        ["IQ3_XS"],
        "{inspection:#?}"
    );

    let candidates = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: ModelCandidate {
                source: CandidateSource::HuggingFace {
                    repository,
                    commit: REVISION.to_owned(),
                },
                target: vec![TARGET.to_owned()],
                projector: None,
                drafter: None,
            },
            other: None,
        })
        .await
        .expect("drafter candidates")
        .candidates;
    let fastmtp = candidates
        .iter()
        .find(|candidate| {
            matches!(&candidate.reference, DrafterReference::HuggingFace { path, .. } if path == FASTMTP)
        })
        .unwrap_or_else(|| panic!("the FastMTP drafter is listed: {candidates:#?}"));
    assert_eq!(fastmtp.category, DrafterCategory::NextToModel);
    let DrafterVerdict::Rejected { reasons, .. } = &fastmtp.verdict else {
        panic!("expected rejected, got {:?}", fastmtp.verdict);
    };
    assert!(
        reasons_match(
            reasons,
            &[
                serde_json::json!({ "WeightTypes": { "types": [{ "ggml_type": "I32", "tensor_count": 1 }] } }),
                serde_json::json!({ "DraftArchitecture": { "architecture": "qwen35" } }),
            ]
        ),
        "{reasons:?}"
    );
}

/// FadedRedStar's Qwen3.5 9B heretic repository ships MTP draft files that declare `qwen35` with a
/// prediction layer but hold no block, only the embedding and head. Their headers make them
/// drafters, so they are never versions (nor damaged versions), in the repository or in a folder;
/// the drafter list shows them Rejected with the engine's reason.
#[tokio::test(flavor = "multi_thread")]
async fn a_head_file_without_blocks_is_a_rejected_drafter_never_a_version() {
    const REPOSITORY: &str = "FadedRedStar/Qwen3.5-9B-heretic-GGUF";
    const REVISION: &str = "df0a5cba190b86ef81cc804bb24f7d6ccee4e83e";
    const TARGET: &str = "Qwen3.5-9B-heretic-Q4_K_M.gguf";
    const HEADS: [&str; 2] = [
        "mtp-Qwen3.5-9B-heretic-Q8_0.gguf",
        "mtp-Qwen3.5-9B-heretic-BF16.gguf",
    ];
    let served = |path: &str| fixture_file(&format!("headers/{path}.gz"), path);
    let fixture = fixture_with(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::from([(
            REPOSITORY.to_owned(),
            StubRepository {
                commit: REVISION.to_owned(),
                files: vec![served(TARGET), served(HEADS[0]), served(HEADS[1])],
                card_base_models: Vec::new(),
            },
        )]),
    )
    .await;
    let rejected_qwen35 = [serde_json::json!({ "DraftArchitecture": { "architecture": "qwen35" } })];
    let assert_heads_rejected =
        |candidates: &[magnitude_service_contracts::models::DrafterCandidate],
         is_head: &dyn Fn(&DrafterReference, &str) -> bool| {
            for head in HEADS {
                let listed = candidates
                    .iter()
                    .find(|candidate| is_head(&candidate.reference, head))
                    .unwrap_or_else(|| panic!("{head} is listed: {candidates:#?}"));
                assert_eq!(listed.category, DrafterCategory::NextToModel, "{head}");
                let DrafterVerdict::Rejected { reasons, .. } = &listed.verdict else {
                    panic!("{head}: expected rejected, got {:?}", listed.verdict);
                };
                assert!(reasons_match(reasons, &rejected_qwen35), "{head}: {reasons:?}");
            }
        };

    let repository = HuggingFaceRepositoryId::new(REPOSITORY.to_owned()).unwrap();
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: repository.clone(),
            revision: REVISION.to_owned(),
        })
        .await
        .expect("inspection");
    assert_eq!(
        inspection
            .versions
            .iter()
            .map(|version| version.label.as_str())
            .collect::<Vec<_>>(),
        ["Q4_K_M"],
        "{inspection:#?}"
    );
    let candidates = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: ModelCandidate {
                source: CandidateSource::HuggingFace {
                    repository,
                    commit: REVISION.to_owned(),
                },
                target: vec![TARGET.to_owned()],
                projector: None,
                drafter: None,
            },
            other: None,
        })
        .await
        .expect("drafter candidates")
        .candidates;
    assert_heads_rejected(&candidates, &|reference, head| {
        matches!(reference, DrafterReference::HuggingFace { path, .. } if path == head)
    });

    let folder = tempfile::tempdir().expect("model folder");
    for name in [TARGET, HEADS[0], HEADS[1]] {
        let StubFile {
            header, size_bytes, ..
        } = served(name);
        let path = folder.path().join(name);
        std::fs::write(&path, &header).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(size_bytes)
            .unwrap();
    }
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::Local {
            path: folder.path().to_path_buf(),
        })
        .await
        .expect("folder inspection");
    assert_eq!(
        inspection
            .versions
            .iter()
            .flat_map(|version| version.files.iter().map(|file| file.path.as_str()))
            .collect::<Vec<_>>(),
        [TARGET],
        "{inspection:#?}"
    );
    let candidates = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: ModelCandidate {
                source: CandidateSource::Local {
                    path: folder.path().to_path_buf(),
                },
                target: vec![TARGET.to_owned()],
                projector: None,
                drafter: None,
            },
            other: None,
        })
        .await
        .expect("folder drafter candidates")
        .candidates;
    assert_heads_rejected(&candidates, &|reference, head| {
        matches!(reference, DrafterReference::Local { path } if path.file_name().is_some_and(|name| name == head))
    });
}

/// A repository whose base model is a finetune of the catalog model's base reaches the catalog
/// model through two Hub lineage hops, and its catalog drafter is a lineage match.
#[tokio::test(flavor = "multi_thread")]
async fn two_hop_hub_lineage_matches_the_catalog_drafter() {
    let fixture = fixture(Vec::new()).await;
    let candidates = candidate_drafters(&fixture, PLAIN, '3').await;
    let catalog = candidates
        .iter()
        .find(|candidate| candidate.category == DrafterCategory::CatalogDrafter)
        .expect("the catalog drafter");
    assert_eq!(catalog.lineage, DrafterLineage::Lineage, "{candidates:#?}");
}

/// A drafter the user enters that shares no lineage with the model is labeled "Untested"; the
/// engine's verdict alone decides whether it can be chosen.
#[tokio::test(flavor = "multi_thread")]
async fn a_user_entered_drafter_without_lineage_is_untested() {
    let fixture = fixture(Vec::new()).await;
    let drafters = tempfile::tempdir().expect("drafter folder");
    let path = drafters.path().join("someone-dflash.gguf");
    write_complete(&path, &dflash(HIDDEN));
    let candidates = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: hugging_face_candidate("vision/Llama-Vision-GGUF", 'e', "llama-Q8_0.gguf"),
            other: Some(path.to_string_lossy().into_owned()),
        })
        .await
        .expect("drafter candidates")
        .candidates;
    let entered = candidates
        .iter()
        .find(|candidate| candidate.category == DrafterCategory::Other)
        .expect("the entered drafter");
    assert_eq!(entered.lineage, DrafterLineage::None);
    assert_eq!(entered.label, "Untested");
    assert!(is_bound(&entered.verdict), "{:?}", entered.verdict);
}

struct NoCatalogEffects;

impl magnitude_service_contracts::models::CatalogModelOptimizer for NoCatalogEffects {
    fn optimize_catalog_model(
        &self,
        _model_id: ModelId,
        _progress: Box<
            dyn Fn(magnitude_service_contracts::models::CatalogOptimizationProgress) + Send + Sync,
        >,
    ) -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
}

impl magnitude_service_contracts::models::CatalogPackageRemover for NoCatalogEffects {
    fn remove_catalog_packages(
        &self,
        _package_ids: Vec<magnitude_service_contracts::models::ModelPackageId>,
    ) -> BoxFuture<'_, Result<u64, InventoryError>> {
        Box::pin(async { Err(InventoryError::Unsupported("no removal in tests".into())) })
    }
}

/// The service as it runs: the store scanning `local_models`, the local-models and catalog domains
/// over it, and the candidates enriching custom models in the background.
struct LiveFixture {
    _directory: tempfile::TempDir,
    _hub: Arc<StubHub>,
    resolver: Arc<ModelDomainResolver>,
    configurations: crate::configurations::ResolvedConfigurations,
    local: Arc<magnitude_service_models::ManagedLocalModels>,
    _candidates: Arc<ManagedModelCandidates>,
}

async fn live_fixture(local_models: LocalModelsConfiguration) -> LiveFixture {
    live_fixture_from(local_models, false, cpu_environment()).await
}

/// A live fixture whose catalog, with `qwen`, also lists Qwen3.5 4B with its DFlash drafter.
async fn live_fixture_from(
    local_models: LocalModelsConfiguration,
    qwen: bool,
    environment: Arc<AssessmentEnvironment>,
) -> LiveFixture {
    live_fixture_reaching(
        local_models,
        qwen,
        environment,
        LiveHub {
            reachable: true,
            ..LiveHub::default()
        },
    )
    .await
}

/// What a live fixture's Hub serves beyond the standard repositories, what this computer has
/// already read from it (repository snapshots, each at its revision), and whether it still
/// answers once that is read, before custom models are enriched.
#[derive(Default)]
struct LiveHub {
    repositories: BTreeMap<String, StubRepository>,
    snapshots_read: Vec<(&'static str, String)>,
    reachable: bool,
}

async fn live_fixture_reaching(
    local_models: LocalModelsConfiguration,
    qwen: bool,
    environment: Arc<AssessmentEnvironment>,
    live_hub: LiveHub,
) -> LiveFixture {
    let mut stub = hub();
    stub.repositories.extend(live_hub.repositories);
    if qwen {
        stub.repositories.extend(qwen_repositories());
    }
    let (endpoint, hub) = stub.serve();
    let directory = tempfile::tempdir().expect("temporary directory");
    let mut config = InventoryConfig::with_roots(
        directory.path().join("models"),
        directory.path().join("cache"),
    )
    .expect("inventory config");
    config.hub_endpoint = Some(endpoint);
    config.local_models = local_models;
    config.architectures = super::engine_architectures();
    let models = Arc::new(ManagedModelStore::open(config, std::sync::Arc::new(|| Ok::<u64, std::convert::Infallible>(u64::MAX))).await.expect("model store"));
    models
        .ensure_model_inventory()
        .await
        .expect("location scan");
    let (mut catalog, _) = llama_test_catalog(&models).await;
    if qwen {
        catalog.models.push(qwen_catalog_model(&models).await);
    }
    for (repository, revision) in live_hub.snapshots_read {
        magnitude_service_models::pinned_hugging_face_repository(
            &models,
            HuggingFaceRepositoryRequest {
                repository: repository.to_owned(),
                revision,
            },
        )
        .await
        .expect("the snapshot is read");
    }
    let lineage = CatalogLineage::new(Arc::clone(&models), catalog.clone());
    let resolver = ModelDomainResolver::new(Arc::clone(&models), catalog.clone());
    let downloads = Arc::new(
        magnitude_service_models::ManagedModelDownloads::open(Arc::clone(&models))
            .await
            .expect("downloads"),
    );
    let services = magnitude_service_models::managed_model_services(
        resolver.clone(),
        downloads,
        Arc::new(NoCatalogEffects),
        Arc::new(NoCatalogEffects),
    )
    .expect("model services");
    let assessor = Arc::new(ModelAssessor::new(
        Arc::clone(&models),
        resolver.clone(),
        Arc::new(ReleaseCatalog::without_material(catalog).expect("release catalog")),
        environment,
    ));
    let candidates = ManagedModelCandidates::new(
        assessor,
        lineage,
        super::engine_architectures(),
        services.catalog.clone(),
        services.local.clone(),
    );
    hub.unreachable.store(!live_hub.reachable, Ordering::SeqCst);
    candidates.start_enrichment(resolver.clone());
    let configurations = crate::configurations::ResolvedConfigurations::new(
        Arc::clone(&models),
        directory.path().join("kernels"),
        magnitude_executor::platform::MemoryReserves::standard(),
    );
    LiveFixture {
        _directory: directory,
        _hub: hub,
        resolver,
        configurations,
        local: services.local,
        _candidates: candidates,
    }
}

impl LiveFixture {
    /// Resolves the model for loading and opens its material as a worker does, returning the
    /// target and draft files it reads.
    async fn load_files(&self, model_id: &ModelId) -> (PathBuf, Option<PathBuf>) {
        let configuration = self
            .resolver
            .serving_configuration(model_id)
            .expect("serving configuration");
        let resolved = self
            .configurations
            .resolve(&configuration)
            .await
            .expect("the model resolves for loading");
        magnitude_artifacts::Package::open_manifest(&resolved.manifest.package)
            .expect("the worker opens the files it is given");
        (
            resolved.manifest.package.target.path().to_path_buf(),
            resolved
                .manifest
                .package
                .draft
                .as_ref()
                .map(|draft| draft.path().to_path_buf()),
        )
    }

    /// Rescans every location, as a watch event does, then waits until the attached drafter of the
    /// custom model whose file is `file` is `expected` (its reference, and whether it serves).
    async fn rescan_until(
        &self,
        file: &std::path::Path,
        expected: Option<(&DrafterReference, bool)>,
        what: &str,
    ) -> CustomModel {
        self.local.refresh_local().await.expect("rescan");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
        loop {
            let listed = self.local.list_local().await.expect("local models");
            let model = listed
                .models
                .iter()
                .find(|model| model.files.first().is_some_and(|first| first.path == file))
                .unwrap_or_else(|| panic!("{what}: no model at {}", file.display()));
            let serves = matches!(
                self.resolver
                    .serving_configuration(&model.model_id)
                    .map(|configuration| configuration.bundle),
                Ok(ServableModelBundle::SpeculativeDecoding { .. })
            );
            let attached = model.drafter.as_ref().map(|drafter| {
                (
                    &drafter.reference,
                    drafter.state
                        == magnitude_service_contracts::models::AttachedDrafterState::Ready
                        && serves,
                )
            });
            if attached == expected {
                return model.clone();
            }
            assert!(
                std::time::Instant::now() < deadline,
                "{what}: attached {attached:?}, recommended {:?}",
                model
                    .recommended_drafter
                    .as_ref()
                    .map(|candidate| &candidate.reference)
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
}

fn write_sparse(path: &std::path::Path, gguf: &Gguf) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let (header, length) = gguf.bytes();
    std::fs::write(path, header).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(length)
        .unwrap();
}

/// QA (Debian, round 3): a drafter added to a found model's folder was listed as recommended but
/// never attached until a restart, and once moved into a subfolder the model kept naming the
/// missing file and served without the catalog drafter that was on disk. The automatic drafter
/// follows every change to the files, live.
#[tokio::test(flavor = "multi_thread")]
async fn a_found_models_automatic_drafter_follows_its_folder_and_the_catalog_drafter_live() {
    let locations = tempfile::tempdir().expect("locations");
    let folder = locations.path().join("A");
    let drafters = locations.path().join("drafters");
    let model = folder.join("model-IQ3_XXS.gguf");
    write_sparse(&model, &llama_with_base("base/Llama"));
    let catalog_copy = drafters.join("catalog-dflash.gguf");
    write_sparse(&catalog_copy, &dflash(HIDDEN));
    let root = |path: &std::path::Path| magnitude_service_contracts::models::ModelRoot {
        kind: magnitude_service_contracts::models::ModelRootKind::Folder,
        path: path.to_path_buf(),
        enabled: true,
    };
    let fixture = live_fixture(LocalModelsConfiguration {
        roots: vec![root(&folder), root(&drafters)],
        ..LocalModelsConfiguration::default()
    })
    .await;
    let catalog = DrafterReference::Catalog {
        model_id: "llama-test:gguf:q8".parse().unwrap(),
    };
    fixture
        .rescan_until(
            &model,
            Some((&catalog, true)),
            "the catalog drafter on disk",
        )
        .await;

    let beside = folder.join("dflash2.gguf");
    write_sparse(&beside, &dflash2());
    let in_folder = DrafterReference::Local {
        path: beside.clone(),
    };
    fixture
        .rescan_until(
            &model,
            Some((&in_folder, true)),
            "a drafter added to the folder takes over without a restart",
        )
        .await;

    let moved = folder.join("sub/dflash2.gguf");
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    std::fs::rename(&beside, &moved).unwrap();
    fixture
        .rescan_until(
            &model,
            Some((&catalog, true)),
            "moved into a subfolder, the drafter gives way to the catalog drafter on disk",
        )
        .await;

    std::fs::remove_file(&catalog_copy).unwrap();
    fixture
        .rescan_until(
            &model,
            Some((&catalog, false)),
            "without the catalog drafter on disk the model serves plain and offers it",
        )
        .await;

    std::fs::rename(&moved, &beside).unwrap();
    fixture
        .rescan_until(
            &model,
            Some((&in_folder, true)),
            "moved back, it attaches again",
        )
        .await;
}

/// QA (Arch, round 3): a copy of a model with its last 4 KiB changed (the same header, other
/// weights) is a second model, and it was offered and attached the "In this folder" drafter beside
/// the first model, in another folder. Each model's folder drafters come from its own folder.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_sharing_a_header_with_another_never_takes_the_drafter_in_its_folder() {
    let locations = tempfile::tempdir().expect("locations");
    let pair = locations.path().join("gemma-pair");
    let picked = locations.path().join("picked");
    let drafters = locations.path().join("drafters");
    let original = pair.join("model-Q8_0.gguf");
    write_sparse(&original, &llama_with_base("base/Llama"));
    let beside = pair.join("mtp-model-Q8_0.gguf");
    write_sparse(&beside, &dflash2());
    write_sparse(&drafters.join("catalog-dflash.gguf"), &dflash(HIDDEN));
    let copy = picked.join("model-Q8_0.gguf");
    std::fs::create_dir_all(&picked).unwrap();
    std::fs::copy(&original, &copy).unwrap();
    {
        use std::io::{Seek, SeekFrom};
        let mut file = std::fs::OpenOptions::new().write(true).open(&copy).unwrap();
        let length = file.metadata().unwrap().len();
        file.seek(SeekFrom::Start(length - 4096)).unwrap();
        file.write_all(&[0xff; 4096]).unwrap();
    }
    let root = |path: &std::path::Path| magnitude_service_contracts::models::ModelRoot {
        kind: magnitude_service_contracts::models::ModelRootKind::Folder,
        path: path.to_path_buf(),
        enabled: true,
    };
    let fixture = live_fixture(LocalModelsConfiguration {
        roots: vec![root(&pair), root(&picked), root(&drafters)],
        ..LocalModelsConfiguration::default()
    })
    .await;
    let in_pair = DrafterReference::Local {
        path: beside.clone(),
    };
    let catalog = DrafterReference::Catalog {
        model_id: "llama-test:gguf:q8".parse().unwrap(),
    };

    let first = fixture
        .rescan_until(
            &original,
            Some((&in_pair, true)),
            "the pair keeps its drafter",
        )
        .await;
    let second = fixture
        .rescan_until(
            &copy,
            Some((&catalog, true)),
            "the copy with other weights takes the catalog drafter, never the pair's",
        )
        .await;
    assert_ne!(first.model_id, second.model_id, "two models");
    assert_eq!(
        second
            .recommended_drafter
            .as_ref()
            .map(|candidate| &candidate.reference),
        Some(&catalog)
    );
    let offered = fixture
        ._candidates
        .drafters(DrafterCandidatesRequest::Model {
            model_id: second.model_id.clone(),
            other: None,
        })
        .await
        .expect("drafter candidates")
        .candidates;
    assert!(
        offered
            .iter()
            .all(|candidate| candidate.category != DrafterCategory::NextToModel),
        "no drafter from another folder is offered: {offered:#?}"
    );
    let served = |model_id: &ModelId| match fixture
        .resolver
        .serving_configuration(model_id)
        .unwrap()
        .bundle
    {
        ServableModelBundle::SpeculativeDecoding { target, .. }
        | ServableModelBundle::Standalone { package: target } => target.id,
    };
    assert_ne!(
        served(&first.model_id),
        served(&second.model_id),
        "each model serves its own weights"
    );
}

/// QA (Fedora and arm64, round 3): with the drafter's bytes in two places, moving the attached copy
/// into a subfolder switched the model to the catalog drafter (the same package, the other copy),
/// but every load kept the resolution of the moved file and failed until a restart. Moving the
/// target itself did the same. A resolution follows where its packages are now.
#[tokio::test(flavor = "multi_thread")]
async fn a_load_after_a_file_moves_reads_it_where_it_is_now_without_a_restart() {
    let locations = tempfile::tempdir().expect("locations");
    let folder = locations.path().join("lfm-own");
    let elsewhere = locations.path().join("lfm-imp");
    let other = locations.path().join("other");
    let model = folder.join("model-Q4_K_M.gguf");
    write_sparse(&model, &llama_with_base("base/Llama"));
    let beside = folder.join("folder-dflash.gguf");
    write_sparse(&beside, &dflash(HIDDEN));
    let copy = elsewhere.join("dflash-copy.gguf");
    write_sparse(&copy, &dflash(HIDDEN));
    std::fs::create_dir_all(&other).unwrap();
    let root = |path: &std::path::Path| magnitude_service_contracts::models::ModelRoot {
        kind: magnitude_service_contracts::models::ModelRootKind::Folder,
        path: path.to_path_buf(),
        enabled: true,
    };
    let fixture = live_fixture(LocalModelsConfiguration {
        roots: vec![root(&folder), root(&elsewhere), root(&other)],
        ..LocalModelsConfiguration::default()
    })
    .await;
    let in_folder = DrafterReference::Local {
        path: beside.clone(),
    };
    let catalog = DrafterReference::Catalog {
        model_id: "llama-test:gguf:q8".parse().unwrap(),
    };

    let listed = fixture
        .rescan_until(&model, Some((&in_folder, true)), "the folder drafter")
        .await;
    let id = listed.model_id.clone();
    // The drafter package is one content in two places; the load reads one of them.
    let (target, read) = fixture.load_files(&id).await;
    assert_eq!(target, model);
    let read = read.expect("a draft");
    assert!(read == beside || read == copy, "{}", read.display());

    let moved = read
        .parent()
        .unwrap()
        .join("sub")
        .join(read.file_name().unwrap());
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    std::fs::rename(&read, &moved).unwrap();
    let attached = if read == beside { &catalog } else { &in_folder };
    fixture
        .rescan_until(&model, Some((attached, true)), "after the move")
        .await;
    let (target, draft) = fixture.load_files(&id).await;
    assert_eq!(target, model);
    let draft = draft.expect("a draft");
    assert_ne!(
        draft, read,
        "the load never reads the moved file's old path"
    );
    assert!(draft.exists(), "{}", draft.display());

    let relocated = other.join("model-Q4_K_M.gguf");
    std::fs::rename(&model, &relocated).unwrap();
    let listed = fixture
        .rescan_until(&relocated, Some((attached, true)), "the moved target")
        .await;
    assert_eq!(listed.model_id, id, "a moved file keeps its ID");
    let (target, draft) = fixture.load_files(&id).await;
    assert_eq!(target, relocated, "the target loads from its new path");
    assert!(draft.is_some_and(|draft| draft.exists()));
}

/// QA (Debian, round 3b): once a projector appeared beside a model, its target package changed
/// and the model showed "Drafter None" for half a minute while its drafter served. However the
/// target package changes, an attached drafter stays shown while the enrichment catches up.
#[tokio::test(flavor = "multi_thread")]
async fn a_projector_appearing_beside_a_model_never_shows_its_drafter_as_none() {
    let locations = tempfile::tempdir().expect("locations");
    let folder = locations.path().join("A");
    let model = folder.join("model-Q4_K_M.gguf");
    write_sparse(&model, &llama(7, Encoding::Q8_0));
    let beside = folder.join("dflash2.gguf");
    write_sparse(&beside, &dflash2());
    let fixture = live_fixture(LocalModelsConfiguration {
        roots: vec![magnitude_service_contracts::models::ModelRoot {
            kind: magnitude_service_contracts::models::ModelRootKind::Folder,
            path: folder.clone(),
            enabled: true,
        }],
        ..LocalModelsConfiguration::default()
    })
    .await;
    let in_folder = DrafterReference::Local {
        path: beside.clone(),
    };
    let before = fixture
        .rescan_until(&model, Some((&in_folder, true)), "the folder drafter")
        .await;

    write_sparse(&folder.join("mmproj-F16.gguf"), &projector());
    fixture.local.refresh_local().await.expect("rescan");
    // Until the enrichment has answered for the model's new target (the stub projector leaves the
    // engine nothing to bind, so its eventual answer offers no drafter), the drafter it had stays.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut pending_seen = 0;
    loop {
        let listed = fixture.local.list_local().await.expect("local models");
        let now = listed
            .models
            .iter()
            .find(|listed| listed.model_id == before.model_id)
            .expect("the model");
        assert!(now.projector.is_some(), "the projector is part of the model");
        let target = match fixture
            .resolver
            .serving_configuration(&now.model_id)
            .expect("serving configuration")
            .bundle
        {
            ServableModelBundle::SpeculativeDecoding { target, .. }
            | ServableModelBundle::Standalone { package: target } => target.id.0,
        };
        if fixture
            ._candidates
            .enrichment
            .knows_key(&(now.model_id.clone(), target))
        {
            break;
        }
        pending_seen += 1;
        assert_eq!(
            now.drafter.as_ref().map(|drafter| &drafter.reference),
            Some(&in_folder),
            "while the new target is pending, the attached drafter is never shown as none"
        );
        assert!(std::time::Instant::now() < deadline, "the enrichment answers");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    eprintln!("pending samples: {pending_seen}");
}

/// Arki05's shard-set repositories ship `skeleton.gguf`: a whole Qwen3.5 4B header (the real one)
/// whose directory declares 8.41 GB of tensor data, in a 10,968,000-byte file. It holds no weights,
/// so it is never a version, in the repository or in a folder; the model beside it is.
#[tokio::test(flavor = "multi_thread")]
async fn a_shard_set_skeleton_holding_no_weights_is_never_a_version() {
    const REPOSITORY: &str = "Arki05/Qwen3.5-4B-GGUF-shards";
    const REVISION: &str = "a3b39a557f0d8cf364f196911e0ba220ae0e48f2";
    const SKELETON: &str = "skeleton.gguf";
    const SKELETON_BYTES: u64 = 10_968_000;
    const TARGET: &str = "Qwen3.5-4B-Q3_K_M.gguf";
    let served = |path: &str| fixture_file(&format!("headers/{path}.gz"), path);
    let skeleton = StubFile {
        size_bytes: SKELETON_BYTES,
        ..served(SKELETON)
    };
    let target = served(TARGET);
    let fixture = fixture_with(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::from([(
            REPOSITORY.to_owned(),
            StubRepository {
                commit: REVISION.to_owned(),
                files: vec![skeleton.clone(), target.clone()],
                card_base_models: Vec::new(),
            },
        )]),
    )
    .await;
    let labels = |inspection: &magnitude_service_contracts::models::CandidateInspection| {
        inspection
            .versions
            .iter()
            .map(|version| version.label.clone())
            .collect::<Vec<_>>()
    };
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new(REPOSITORY.to_owned()).unwrap(),
            revision: REVISION.to_owned(),
        })
        .await
        .expect("inspection");
    assert_eq!(labels(&inspection), ["Q3_K_M"], "{inspection:#?}");

    let folder = tempfile::tempdir().expect("model folder");
    for file in [&skeleton, &target] {
        let path = folder.path().join(&file.path);
        std::fs::write(&path, &file.header).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(file.size_bytes)
            .unwrap();
    }
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::Local {
            path: folder.path().to_path_buf(),
        })
        .await
        .expect("folder inspection");
    assert_eq!(labels(&inspection), ["Q3_K_M"], "{inspection:#?}");
}

// Fit with built-in heads ------------------------------------------------------------------------

const QWEN_TARGET: &str = "unsloth/Qwen3.5-4B-MTP-GGUF";
const QWEN_DRAFT: &str = "magnitudedev/Qwen3.5-4B-DFlash-GGUF";
const QWEN_COMMUNITY: &str = "community/Qwen3.5-4B-GGUF";
const QWEN_FILE: &str = "Qwen3.5-4B-UD-Q4_K_XL.gguf";
const QWEN_Q3_FILE: &str = "Qwen3.5-4B-Q3_K_M.gguf";
const QWEN_DRAFT_FILE: &str = "Qwen3.5-4B-DFlash-Q8_0.gguf";

/// The real Qwen3.5 4B headers, whose targets carry built-in MTP heads: the catalog target and
/// its DFlash drafter, and a community repository with the same UD-Q4_K_XL file and a Q3_K_M.
fn qwen_repositories() -> BTreeMap<String, StubRepository> {
    let served = |path: &str| fixture_file(&format!("headers/{path}.gz"), path);
    BTreeMap::from([
        (
            QWEN_TARGET.to_owned(),
            repository('1', vec![served(QWEN_FILE)], &["Qwen/Qwen3.5-4B"]),
        ),
        (
            QWEN_DRAFT.to_owned(),
            repository('2', vec![served(QWEN_DRAFT_FILE)], &["Qwen/Qwen3.5-4B"]),
        ),
        (
            QWEN_COMMUNITY.to_owned(),
            repository(
                '3',
                vec![served(QWEN_FILE), served(QWEN_Q3_FILE)],
                &["Qwen/Qwen3.5-4B"],
            ),
        ),
    ])
}

/// The catalog's Qwen3.5 4B, served with its DFlash drafter.
async fn qwen_catalog_model(models: &Arc<ManagedModelStore>) -> RecommendableModel {
    let (mut catalog, _) = llama_test_catalog(models).await;
    let mut model = catalog.models.remove(0);
    model.model_id = magnitude_service_contracts::models::CatalogBaseId::new("qwen3.5-4b").unwrap();
    model.variant_id = magnitude_service_contracts::models::CatalogVariantId::new("gguf:q4").unwrap();
    model.display_name = "Qwen3.5 4B".to_owned();
    model.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
        target: hugging_face_package(models, QWEN_TARGET, QWEN_FILE).await,
        draft_source: SpeculativeDraftSource::Separate {
            draft: hugging_face_package(models, QWEN_DRAFT, QWEN_DRAFT_FILE).await,
        },
        method: SpeculativeMethod::DFlash,
    };
    model
}

fn qwen_catalog_id() -> ModelId {
    "qwen3.5-4b:gguf:q4".parse().unwrap()
}

// Catalog drafters matched by shape ---------------------------------------------------------------

const QWEN36_TARGET: &str = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF";
const QWEN36_DRAFT: &str = "magnitudedev/Qwen3.6-35B-A3B-DFlash-GGUF";
const QWEN36_FILE: &str = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";
const QWEN36_DRAFT_FILE: &str = "Qwen3.6-35B-A3B-DFlash-Q8_0.gguf";
const QWEN38_TARGET: &str = "unsloth/Qwen3.8-27B-GGUF";
const QWEN38_DRAFT: &str = "magnitudedev/Qwen3.8-27B-DFlash2-GGUF";
const QWEN38_FILE: &str = "Qwen3.8-27B-UD-Q4_K_XL.gguf";
const QWEN38_DRAFT_FILE: &str = "Qwen3.8-27B-DFlash2-Q8_0.gguf";
/// Ollama's `qwen3.6:35b-a3b` model blob.
const OLLAMA_QWEN36_FILE: &str = "ollama-qwen3.6-35b-a3b.gguf";
/// lmstudio-community's Qwen3.6 27B: Qwen3.8 27B's shape and token table, another generation.
const QWEN36_27B_FILE: &str = "Qwen3.6-27B-Q4_K_M.gguf";

/// The real catalog targets of Qwen3.6 35B-A3B (with built-in MTP heads) and Qwen3.8 27B, and
/// their catalog drafters.
fn qwen_generation_repositories() -> BTreeMap<String, StubRepository> {
    let served = |path: &str| fixture_file(&format!("headers/{path}.gz"), path);
    BTreeMap::from([
        (
            QWEN36_TARGET.to_owned(),
            repository('7', vec![served(QWEN36_FILE)], &["Qwen/Qwen3.6-35B-A3B"]),
        ),
        (
            QWEN36_DRAFT.to_owned(),
            repository('8', vec![served(QWEN36_DRAFT_FILE)], &["Qwen/Qwen3.6-35B-A3B"]),
        ),
        (
            QWEN38_TARGET.to_owned(),
            repository('a', vec![served(QWEN38_FILE)], &["Qwen/Qwen3.8-27B"]),
        ),
        (
            QWEN38_DRAFT.to_owned(),
            repository('b', vec![served(QWEN38_DRAFT_FILE)], &["Qwen/Qwen3.8-27B"]),
        ),
    ])
}

/// A catalog model of the Qwen fixtures, served with its separate drafter.
async fn qwen_generation_catalog_model(
    models: &Arc<ManagedModelStore>,
    (id, display_name): (&str, &str),
    (target, target_file): (&str, &str),
    (draft, draft_file): (&str, &str),
    method: SpeculativeMethod,
) -> RecommendableModel {
    let mut model = qwen_catalog_model(models).await;
    model.model_id = magnitude_service_contracts::models::CatalogBaseId::new(id).unwrap();
    model.display_name = display_name.to_owned();
    model.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
        target: hugging_face_package(models, target, target_file).await,
        draft_source: SpeculativeDraftSource::Separate {
            draft: hugging_face_package(models, draft, draft_file).await,
        },
        method,
    };
    model
}

async fn qwen36_catalog_model(models: &Arc<ManagedModelStore>) -> RecommendableModel {
    qwen_generation_catalog_model(
        models,
        ("qwen3.6-35b-a3b", "Qwen3.6 35B-A3B"),
        (QWEN36_TARGET, QWEN36_FILE),
        (QWEN36_DRAFT, QWEN36_DRAFT_FILE),
        SpeculativeMethod::DFlash,
    )
    .await
}

async fn qwen38_catalog_model(models: &Arc<ManagedModelStore>) -> RecommendableModel {
    qwen_generation_catalog_model(
        models,
        ("qwen3.8-27b", "Qwen3.8 27B"),
        (QWEN38_TARGET, QWEN38_FILE),
        (QWEN38_DRAFT, QWEN38_DRAFT_FILE),
        SpeculativeMethod::DFlash2,
    )
    .await
}

/// A fixture serving the Qwen repositories and listing their catalog models, and a folder holding
/// only the shared fixture header `name`.
async fn found_qwen(name: &str) -> (Fixture, tempfile::TempDir) {
    found_qwen_with_weights(name, None).await
}

/// `found_qwen`, with the file's tensor data holding `weights`, a real file's weight sample, where
/// its header locates the sampled tensors.
async fn found_qwen_with_weights(
    name: &str,
    weights: Option<&str>,
) -> (Fixture, tempfile::TempDir) {
    let fixture = fixture_from(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::new(),
        true,
        cpu_environment(),
    )
    .await;
    let folder = tempfile::tempdir().expect("model folder");
    let StubFile {
        header, size_bytes, ..
    } = fixture_file(&format!("headers/{name}.gz"), name);
    let contents = match weights {
        Some(weights) => with_weights(&header, weights),
        None => header,
    };
    write_complete_bytes(&folder.path().join(name), &contents, size_bytes);
    (fixture, folder)
}

/// A file's leading bytes: `header`, then zeros but where it locates the tensors a weight sample
/// reads, which hold `weights`, a real file's weight sample.
fn with_weights(header: &[u8], weights: &str) -> Vec<u8> {
    let directory = magnitude_artifacts::gguf::read_header(header).expect("GGUF header");
    let reads = WeightSample::reads(&directory);
    let mut bytes = header.to_vec();
    for (tensor, values) in weight_sample(weights).tensors() {
        let read = reads
            .iter()
            .find(|read| read.tensor == tensor)
            .expect("a sampled F32 tensor");
        let start = usize::try_from(read.start).unwrap();
        let end = start + values.len() * 4;
        if bytes.len() < end {
            bytes.resize(end, 0);
        }
        for (index, value) in values.iter().enumerate() {
            bytes[start + index * 4..start + index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// The variant a found file is shown as, and its drafter candidates.
async fn found_variant_and_drafters(
    fixture: &Fixture,
    folder: &std::path::Path,
    name: &str,
) -> (
    Option<ModelId>,
    Vec<magnitude_service_contracts::models::DrafterCandidate>,
) {
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::Local {
            path: folder.join(name),
        })
        .await
        .expect("inspection");
    let drafters = drafters_of(
        fixture,
        ModelCandidate {
            source: CandidateSource::Local {
                path: folder.to_path_buf(),
            },
            target: vec![name.to_owned()],
            projector: None,
            drafter: None,
        },
    )
    .await;
    (inspection.variant_of, drafters)
}

/// Ollama's `qwen3.6:35b-a3b` declares no lineage, writes a zero dense `feed_forward_length`, and
/// writes its tokenizer's audio and padding entries as `[PAD<n>]`, so its token table is not the
/// catalog target's though its shape is. As a variant of the catalog Qwen3.6 35B-A3B, it is
/// offered that model's DFlash drafter, which binds, ahead of its built-in heads; with nothing
/// but its shape to go on, the drafter is untested and never chosen automatically.
#[tokio::test(flavor = "multi_thread")]
async fn a_variant_without_lineage_is_offered_its_catalog_drafter_as_untested() {
    let (fixture, blobs) = found_qwen(OLLAMA_QWEN36_FILE).await;
    let (variant_of, candidates) =
        found_variant_and_drafters(&fixture, blobs.path(), OLLAMA_QWEN36_FILE).await;
    assert_eq!(variant_of, Some("qwen3.6-35b-a3b:gguf:q4".parse().unwrap()));
    let listed = candidates
        .iter()
        .map(|candidate| {
            (
                candidate.category,
                candidate.reference.clone(),
                candidate.lineage,
                candidate.method.clone(),
                is_bound(&candidate.verdict),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        listed,
        [
            (
                DrafterCategory::CatalogDrafter,
                DrafterReference::Catalog {
                    model_id: "qwen3.6-35b-a3b:gguf:q4".parse().unwrap()
                },
                DrafterLineage::Fingerprint,
                Some(SpeculativeMethod::DFlash),
                true,
            ),
            (
                DrafterCategory::BuiltIn,
                DrafterReference::Embedded,
                DrafterLineage::Exact,
                Some(SpeculativeMethod::Mtp),
                true,
            ),
        ],
        "{candidates:#?}"
    );
    assert_eq!(candidates[0].label, "Untested · Trained for Qwen3.6 35B-A3B");
    assert!(!candidates[0].automatic, "{candidates:#?}");
}

/// The same Ollama blob holding its real weights: its early normalization weights are the catalog
/// Qwen3.6 35B-A3B target's bit for bit, which is lineage the header can't state, so the catalog
/// DFlash drafter is known to work and chosen automatically, ahead of the built-in heads.
#[tokio::test(flavor = "multi_thread")]
async fn a_variant_carrying_its_catalog_models_weights_gets_its_catalog_drafter_automatically() {
    let (fixture, blobs) =
        found_qwen_with_weights(OLLAMA_QWEN36_FILE, Some("ollama-qwen3.6-35b-a3b")).await;
    let (variant_of, candidates) =
        found_variant_and_drafters(&fixture, blobs.path(), OLLAMA_QWEN36_FILE).await;
    assert_eq!(variant_of, Some("qwen3.6-35b-a3b:gguf:q4".parse().unwrap()));
    let catalog = &candidates[0];
    assert_eq!(
        (
            catalog.category,
            catalog.lineage,
            catalog.label.as_str(),
            is_bound(&catalog.verdict)
        ),
        (
            DrafterCategory::CatalogDrafter,
            DrafterLineage::Lineage,
            "Trained for Qwen3.6 35B-A3B",
            true
        ),
        "{candidates:#?}"
    );
    assert_eq!(
        preselected(&candidates).reference,
        DrafterReference::Catalog {
            model_id: "qwen3.6-35b-a3b:gguf:q4".parse().unwrap()
        },
        "{candidates:#?}"
    );
}

/// A Hub file without declared lineage is sampled by range requests: the Ollama blob's weights
/// re-uploaded get the catalog Qwen3.6 35B-A3B DFlash automatically as lineage, and Qwen3.6 27B,
/// whose weights are not Qwen3.8 27B's, is only offered its DFlash2 as untested.
#[tokio::test(flavor = "multi_thread")]
async fn a_hub_files_weights_are_sampled_for_its_catalog_drafter() {
    let weighted = |name: &str, weights: &str| {
        let file = fixture_file(&format!("headers/{name}.gz"), name);
        StubFile {
            header: with_weights(&file.header, weights),
            ..file
        }
    };
    let fixture = fixture_from(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::from([
            (
                "someone/Qwen3.6-35B-A3B-GGUF".to_owned(),
                repository(
                    'c',
                    vec![weighted(OLLAMA_QWEN36_FILE, "ollama-qwen3.6-35b-a3b")],
                    &[],
                ),
            ),
            (
                "someone/Qwen3.6-27B-GGUF".to_owned(),
                repository(
                    'd',
                    vec![weighted(QWEN36_27B_FILE, "Qwen3.6-27B-Q4_K_M")],
                    &[],
                ),
            ),
        ]),
        true,
        cpu_environment(),
    )
    .await;
    let catalog_drafter = |candidates: &[magnitude_service_contracts::models::DrafterCandidate]| {
        candidates
            .iter()
            .find(|candidate| candidate.category == DrafterCategory::CatalogDrafter)
            .map(|candidate| (candidate.lineage, candidate.automatic, candidate.label.clone()))
    };
    let same_weights = drafters_of(
        &fixture,
        hugging_face_candidate("someone/Qwen3.6-35B-A3B-GGUF", 'c', OLLAMA_QWEN36_FILE),
    )
    .await;
    assert_eq!(
        catalog_drafter(&same_weights),
        Some((
            DrafterLineage::Lineage,
            true,
            "Trained for Qwen3.6 35B-A3B".to_owned()
        )),
        "{same_weights:#?}"
    );
    let other_generation = drafters_of(
        &fixture,
        hugging_face_candidate("someone/Qwen3.6-27B-GGUF", 'd', QWEN36_27B_FILE),
    )
    .await;
    assert_eq!(
        catalog_drafter(&other_generation),
        Some((
            DrafterLineage::Fingerprint,
            false,
            "Untested · Trained for Qwen3.8 27B".to_owned()
        )),
        "{other_generation:#?}"
    );
}

/// Qwen3.6 27B has every fingerprinted dimension and the token table of Qwen3.8 27B, so it is shown
/// as its variant, and Qwen3.8 27B's DFlash2 binds to it; but that drafter was trained on another
/// generation's hidden states and slows it down, so a shape match never makes it automatic. Its
/// real weights, read from the file, are not Qwen3.8 27B's either.
#[tokio::test(flavor = "multi_thread")]
async fn a_catalog_drafter_matched_by_shape_alone_is_never_automatic() {
    let (fixture, folder) =
        found_qwen_with_weights(QWEN36_27B_FILE, Some("Qwen3.6-27B-Q4_K_M")).await;
    let (variant_of, candidates) =
        found_variant_and_drafters(&fixture, folder.path(), QWEN36_27B_FILE).await;
    assert_eq!(variant_of, Some("qwen3.8-27b:gguf:q4".parse().unwrap()));
    let listed = candidates
        .iter()
        .map(|candidate| {
            (
                candidate.category,
                candidate.lineage,
                candidate.method.clone(),
                is_bound(&candidate.verdict),
                candidate.automatic,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        listed,
        [(
            DrafterCategory::CatalogDrafter,
            DrafterLineage::Fingerprint,
            Some(SpeculativeMethod::DFlash2),
            true,
            false,
        )],
        "{candidates:#?}"
    );
}

/// What a file needs on the standard CPU environment served plain, with its built-in heads, and
/// with the catalog DFlash: the host's capacity and each composition's required bytes.
struct Requirements {
    capacity: u64,
    plain: u64,
    heads: u64,
    dflash: u64,
}

impl Requirements {
    async fn of(file: &str) -> Self {
        let standard = fixture_from(
            Vec::new(),
            LocalModelsConfiguration::default(),
            false,
            BTreeMap::new(),
            true,
            cpu_environment(),
        )
        .await;
        let assess = |drafter: Option<CandidateDrafter>| {
            let candidate = ModelCandidate {
                drafter,
                ..hugging_face_candidate(QWEN_COMMUNITY, '3', file)
            };
            let candidates = Arc::clone(&standard.candidates);
            async move {
                let assessment = candidates
                    .assess_candidate(candidate)
                    .await
                    .expect("assessment");
                let ModelAssessment::Fits { memory, .. } = &assessment else {
                    panic!("everything fits the standard host: {assessment:?}");
                };
                let host = memory.first().expect("a memory domain");
                (host.capacity_bytes, host.required_bytes)
            }
        };
        let (capacity, plain) = assess(None).await;
        let (_, heads) = assess(Some(CandidateDrafter {
            reference: DrafterReference::Embedded,
            method: SpeculativeMethod::Mtp,
        }))
        .await;
        let (_, dflash) = assess(Some(CandidateDrafter {
            reference: DrafterReference::Catalog {
                model_id: qwen_catalog_id(),
            },
            method: SpeculativeMethod::DFlash,
        }))
        .await;
        assert!(
            plain < heads.min(dflash),
            "the heads and the drafter each need more than the model served plain"
        );
        Self {
            capacity,
            plain,
            heads,
            dflash,
        }
    }

    /// A CPU environment whose host may use halfway between serving plain and serving with the
    /// cheaper drafter: the model fits plain and with neither its heads nor the DFlash.
    fn between(&self) -> Arc<AssessmentEnvironment> {
        self.with_budget((self.plain + self.heads.min(self.dflash)) / 2)
    }

    /// A CPU environment whose host may use a little less than serving plain needs: the model
    /// fits neither plain nor with its heads or the DFlash.
    fn short(&self) -> Arc<AssessmentEnvironment> {
        self.with_budget(self.plain - (self.heads.min(self.dflash) - self.plain) / 2)
    }

    fn with_budget(&self, budget: u64) -> Arc<AssessmentEnvironment> {
        let reserve = |floor: u64| serde_json::json!({ "fraction_divisor": u64::MAX, "floor_bytes": floor });
        let reserves: magnitude_executor::platform::MemoryReserves =
            serde_json::from_value(serde_json::json!({
                "host": { "planning": reserve(self.capacity - budget), "emergency": reserve(0) },
                "device": { "planning": reserve(0), "emergency": reserve(0) },
                "volume_bytes": 0,
            }))
            .expect("memory reserves");
        let catalog = seismic::Devices::discover(&[]).expect("device catalog");
        let setup = magnitude_engine::assessment::AssessmentSetup::discover(
            &catalog,
            magnitude_executor::platform::DeviceRequest::Backend(seismic::BackendName::Cpu),
            reserves,
            crate::assessment::environment::serving_policy(),
            magnitude_engine::options::standard_service_limits(),
        )
        .expect("the CPU backend is selectable");
        Arc::new(AssessmentEnvironment {
            id: magnitude_service_contracts::models::AssessmentEnvironmentId(format!(
                "cpu-test-{budget}"
            )),
            setup,
        })
    }
}

/// QA (Windows CUDA 8 GB, round 3b): a found Qwen3.5 4B with built-in MTP heads fit served plain,
/// yet its catalog DFlash was preselected and attached and the load was refused: whether it fits
/// "without a drafter" was judged with its heads. A drafter that would push a model that fits
/// plain out of fitting is never automatic, for a file from any repository, the exact catalog
/// target and the catalog model alike; a version's verdict is then the model served plain.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_that_fits_only_served_plain_gets_no_automatic_drafter_even_with_built_in_heads() {
    let requirements = Requirements::of(QWEN_FILE).await;
    let fixture = fixture_from(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::new(),
        true,
        requirements.between(),
    )
    .await;
    let drafters = |repository: &'static str, seed: char| {
        let candidates = Arc::clone(&fixture.candidates);
        async move {
            candidates
                .drafters(DrafterCandidatesRequest::Candidate {
                    candidate: hugging_face_candidate(repository, seed, QWEN_FILE),
                    other: None,
                })
                .await
                .expect("drafter candidates")
                .candidates
        }
    };

    let community = drafters(QWEN_COMMUNITY, '3').await;
    for category in [DrafterCategory::CatalogDrafter, DrafterCategory::BuiltIn] {
        let listed = community
            .iter()
            .find(|candidate| candidate.category == category)
            .unwrap_or_else(|| panic!("{category:?} is listed: {community:#?}"));
        assert!(is_bound(&listed.verdict), "{category:?} binds: {listed:?}");
        assert!(!listed.automatic, "{category:?} would stop the model fitting");
    }
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new(QWEN_COMMUNITY.to_owned()).unwrap(),
            revision: commit('3'),
        })
        .await
        .expect("inspection");
    let verdict = version(&inspection, QWEN_FILE).assessment.clone();
    assert!(
        matches!(verdict, Some(ModelAssessment::Fits { .. })),
        "adding it installs the model served plain, which fits: {verdict:?}"
    );

    let exact = drafters(QWEN_TARGET, '1').await;
    assert_eq!(
        exact
            .iter()
            .map(|candidate| candidate.category)
            .collect::<Vec<_>>(),
        [DrafterCategory::CatalogDrafter, DrafterCategory::BuiltIn],
        "{exact:#?}"
    );
    for candidate in &exact {
        assert!(is_bound(&candidate.verdict));
        assert!(
            !candidate.automatic,
            "the exact catalog target serves plain"
        );
    }
    assert_eq!(
        fixture
            .candidates
            .recommended_drafter(&qwen_catalog_id())
            .await
            .expect("recommendation"),
        None,
        "the catalog model serves plain"
    );
    assert_eq!(
        fixture
            .candidates
            .catalog_drafter_fit(&qwen_catalog_id())
            .await
            .expect("catalog drafter fit"),
        CatalogDrafterFit {
            withheld: true,
            built_in_heads: false,
        },
        "served plain, never with heads that would stop it fitting"
    );
}

/// QA (V710 27 GB, round 4): Qwen3.6 35B-A3B MXFP4 needed 83.5 MB more served plain, yet Auto
/// attached its catalog DFlash and the refusal asked for 1.4 GB more. When the model doesn't fit
/// served plain, no drafter is automatic, for a file from any repository, the exact catalog target
/// and the catalog model alike; each stays listed with its verdict, and the shortfall reported is
/// the model's own.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_that_does_not_fit_served_plain_gets_no_automatic_drafter() {
    let requirements = Requirements::of(QWEN_FILE).await;
    let fixture = fixture_from(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::new(),
        true,
        requirements.short(),
    )
    .await;
    let drafters = |repository: &'static str, seed: char| {
        let candidates = Arc::clone(&fixture.candidates);
        async move {
            candidates
                .drafters(DrafterCandidatesRequest::Candidate {
                    candidate: hugging_face_candidate(repository, seed, QWEN_FILE),
                    other: None,
                })
                .await
                .expect("drafter candidates")
                .candidates
        }
    };

    for (repository, seed) in [(QWEN_COMMUNITY, '3'), (QWEN_TARGET, '1')] {
        let listed = drafters(repository, seed).await;
        for category in [DrafterCategory::CatalogDrafter, DrafterCategory::BuiltIn] {
            let candidate = listed
                .iter()
                .find(|candidate| candidate.category == category)
                .unwrap_or_else(|| panic!("{category:?} is listed: {listed:#?}"));
            assert!(is_bound(&candidate.verdict), "{category:?} binds: {candidate:?}");
            assert!(
                !candidate.automatic,
                "{repository}: {category:?} is not automatic for a model that doesn't fit plain"
            );
        }
    }

    let plain = fixture
        .candidates
        .assess_candidate(hugging_face_candidate(QWEN_COMMUNITY, '3', QWEN_FILE))
        .await
        .expect("plain assessment");
    let ModelAssessment::DoesNotFit {
        deficit_bytes: plain_deficit,
        ..
    } = plain
    else {
        panic!("the model doesn't fit served plain: {plain:?}");
    };
    let inspection = fixture
        .candidates
        .inspect(CandidateInspectionRequest::HuggingFace {
            repository: HuggingFaceRepositoryId::new(QWEN_COMMUNITY.to_owned()).unwrap(),
            revision: commit('3'),
        })
        .await
        .expect("inspection");
    let verdict = version(&inspection, QWEN_FILE).assessment.clone();
    assert!(
        matches!(
            verdict,
            Some(ModelAssessment::DoesNotFit { deficit_bytes, .. }) if deficit_bytes == plain_deficit
        ),
        "the shortfall reported is the model's own, {plain_deficit} bytes: {verdict:?}"
    );

    assert_eq!(
        fixture
            .candidates
            .recommended_drafter(&qwen_catalog_id())
            .await
            .expect("recommendation"),
        None,
        "the catalog model is recommended no drafter"
    );
    assert_eq!(
        fixture
            .candidates
            .catalog_drafter_fit(&qwen_catalog_id())
            .await
            .expect("catalog drafter fit"),
        CatalogDrafterFit {
            withheld: true,
            built_in_heads: false,
        },
        "the catalog model is served plain"
    );
}

/// QA (mac-m4, round 3c): the catalog's Qwen3.5 4B, its exact target found in a Hugging Face
/// cache and its DFlash drafter not yet downloaded, served plain although its target has built-in
/// MTP heads, while a found non-catalog Qwen3.5 4B in the same state drafted with its heads. A
/// catalog model lists its built-in heads after its catalog drafter, and they serve while that
/// drafter is not on this computer.
#[tokio::test(flavor = "multi_thread")]
async fn a_catalog_model_with_built_in_heads_serves_them_while_its_catalog_drafter_is_missing() {
    let fixture = fixture_from(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::new(),
        true,
        cpu_environment(),
    )
    .await;
    let catalog = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Model {
            model_id: qwen_catalog_id(),
            other: None,
        })
        .await
        .expect("drafter candidates")
        .candidates;
    assert_eq!(
        catalog
            .iter()
            .map(|candidate| (candidate.category, candidate.automatic))
            .collect::<Vec<_>>(),
        [
            (DrafterCategory::CatalogDrafter, true),
            (DrafterCategory::BuiltIn, true)
        ],
        "{catalog:#?}"
    );
    assert_eq!(
        fixture
            .candidates
            .recommended_drafter(&qwen_catalog_id())
            .await
            .expect("recommendation")
            .map(|candidate| candidate.category),
        Some(DrafterCategory::CatalogDrafter),
        "the catalog drafter stays the one recommended"
    );
    assert_eq!(
        fixture
            .candidates
            .catalog_drafter_fit(&qwen_catalog_id())
            .await
            .expect("catalog drafter fit"),
        CatalogDrafterFit {
            withheld: false,
            built_in_heads: true,
        }
    );
}

fn folder_root(path: &std::path::Path) -> magnitude_service_contracts::models::ModelRoot {
    magnitude_service_contracts::models::ModelRoot {
        kind: magnitude_service_contracts::models::ModelRootKind::Folder,
        path: path.to_path_buf(),
        enabled: true,
    }
}

/// Writes a shared fixture header at `path`, sized as its tensor directory declares.
fn write_fixture_sparse(path: &std::path::Path, name: &str) {
    let StubFile {
        header, size_bytes, ..
    } = fixture_file(&format!("headers/{name}.gz"), name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, &header).unwrap();
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(size_bytes)
        .unwrap();
}

/// The QA case end to end: a found Qwen3.5 4B with heads that fits only served plain, with the
/// catalog DFlash on this computer, attaches no drafter and is recommended none, so it loads.
#[tokio::test(flavor = "multi_thread")]
async fn a_found_model_that_fits_only_served_plain_never_attaches_its_catalog_drafter_on_disk() {
    let requirements = Requirements::of(QWEN_Q3_FILE).await;
    let locations = tempfile::tempdir().expect("locations");
    let model = locations.path().join("found").join(QWEN_Q3_FILE);
    write_fixture_sparse(&model, QWEN_Q3_FILE);
    write_fixture_sparse(
        &locations.path().join("drafters").join(QWEN_DRAFT_FILE),
        QWEN_DRAFT_FILE,
    );
    let fixture = live_fixture_from(
        LocalModelsConfiguration {
            roots: vec![
                folder_root(&locations.path().join("found")),
                folder_root(&locations.path().join("drafters")),
            ],
            ..LocalModelsConfiguration::default()
        },
        true,
        requirements.between(),
    )
    .await;
    // Let the enrichment answer before checking that nothing is attached.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
    loop {
        fixture.local.refresh_local().await.expect("rescan");
        let listed = fixture.local.list_local().await.expect("local models");
        let found = listed
            .models
            .iter()
            .find(|candidate| candidate.files.first().is_some_and(|first| first.path == model))
            .expect("the found model");
        if found.variant_of.is_some() {
            assert_eq!(found.recommended_drafter, None, "{found:#?}");
            assert_eq!(found.drafter, None, "{found:#?}");
            assert!(matches!(
                fixture
                    .resolver
                    .serving_configuration(&found.model_id)
                    .expect("serving configuration")
                    .bundle,
                ServableModelBundle::Standalone { .. }
            ));
            break;
        }
        assert!(std::time::Instant::now() < deadline, "the enrichment answers");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// QA (Linux CUDA, round 3b): with the catalog drafter's bytes gone, a found model with working
/// built-in heads showed "Drafter None · Add" and served plain, because only the first automatic
/// drafter was known. Attachment follows the whole automatic order over what is on this
/// computer: the heads serve while the better catalog drafter stays recommended, to be added.
#[tokio::test(flavor = "multi_thread")]
async fn a_found_model_falls_back_to_its_built_in_heads_while_the_catalog_drafter_is_missing() {
    let locations = tempfile::tempdir().expect("locations");
    let model = locations.path().join("found").join(QWEN_Q3_FILE);
    write_fixture_sparse(&model, QWEN_Q3_FILE);
    let catalog_copy = locations.path().join("drafters").join(QWEN_DRAFT_FILE);
    write_fixture_sparse(&catalog_copy, QWEN_DRAFT_FILE);
    let fixture = live_fixture_from(
        LocalModelsConfiguration {
            roots: vec![
                folder_root(&locations.path().join("found")),
                folder_root(&locations.path().join("drafters")),
            ],
            ..LocalModelsConfiguration::default()
        },
        true,
        cpu_environment(),
    )
    .await;
    let catalog = DrafterReference::Catalog {
        model_id: qwen_catalog_id(),
    };
    fixture
        .rescan_until(&model, Some((&catalog, true)), "the catalog drafter on disk")
        .await;

    std::fs::remove_file(&catalog_copy).unwrap();
    let listed = fixture
        .rescan_until(
            &model,
            Some((&DrafterReference::Embedded, true)),
            "without the catalog drafter the built-in heads serve",
        )
        .await;
    assert_eq!(
        listed
            .recommended_drafter
            .as_ref()
            .map(|candidate| &candidate.reference),
        Some(&catalog),
        "the catalog drafter stays recommended, offered to add"
    );

    write_fixture_sparse(&catalog_copy, QWEN_DRAFT_FILE);
    fixture
        .rescan_until(
            &model,
            Some((&catalog, true)),
            "back on disk, the catalog drafter takes over again",
        )
        .await;
}

/// QA (macOS, round 4): a folder added while Hugging Face was unreachable listed its found models
/// with no "Variant of" and no drafter for minutes, and a model with built-in MTP heads loaded in
/// that window served plain: every enrichment waited for the Hub (catalog lineage, the catalog
/// drafter's header) before publishing anything. What the files here say is published first: the
/// heads attach and a load uses them while the Hub never answers, and stay recommended: the catalog
/// drafter it matches by shape alone is never automatic.
#[tokio::test(flavor = "multi_thread")]
async fn a_found_models_built_in_heads_serve_while_the_hub_is_unreachable() {
    let locations = tempfile::tempdir().expect("locations");
    let found = locations.path().join("found");
    let model = found.join(QWEN_Q3_FILE);
    write_fixture_sparse(&model, QWEN_Q3_FILE);
    let fixture = live_fixture_reaching(
        LocalModelsConfiguration {
            roots: vec![folder_root(&found)],
            ..LocalModelsConfiguration::default()
        },
        true,
        cpu_environment(),
        LiveHub::default(),
    )
    .await;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
    let listed = loop {
        let listed = fixture.local.list_local().await.expect("local models");
        let found = listed
            .models
            .into_iter()
            .find(|listed| {
                listed
                    .files
                    .first()
                    .is_some_and(|first| first.path == model)
            })
            .expect("the found model is listed");
        if found
            .drafter
            .as_ref()
            .is_some_and(|drafter| drafter.reference == DrafterReference::Embedded)
        {
            break found;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the built-in heads attach without the Hub: attached {:?}",
            found.drafter
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    };
    assert_eq!(
        listed.variant_of.as_ref(),
        Some(&qwen_catalog_id()),
        "Variant of comes from the file's shape, never the Hub"
    );
    assert_eq!(
        listed
            .recommended_drafter
            .as_ref()
            .map(|candidate| &candidate.reference),
        Some(&DrafterReference::Embedded),
        "without the Hub's lineage the catalog drafter matches by shape alone, offered but never \
         automatic, so the built-in heads are recommended"
    );
    assert!(
        matches!(
            fixture
                .resolver
                .serving_configuration(&listed.model_id)
                .map(|configuration| configuration.bundle),
            Ok(ServableModelBundle::SpeculativeDecoding {
                draft_source: SpeculativeDraftSource::Embedded,
                method: SpeculativeMethod::Mtp,
                ..
            })
        ),
        "a load in this window serves with the heads"
    );
}

/// A catalog model that fits only without its catalog drafter is assessed, and would be served,
/// without it, so it loads rather than being refused; with room for the drafter it keeps it.
#[tokio::test(flavor = "multi_thread")]
async fn a_catalog_model_that_fits_only_without_its_drafter_is_served_plain() {
    use magnitude_service_contracts::models::{CatalogModelSelection, ModelAssessmentSubject};
    let requirements = Requirements::of(QWEN_FILE).await;
    let desired = |fixture: &LiveFixture| {
        fixture
            .resolver
            .assessment_configuration(&ModelAssessmentSubject::Catalog {
                model_id: qwen_catalog_id(),
                selection: CatalogModelSelection::Desired,
            })
            .expect("configuration")
            .bundle
    };
    let constrained = live_fixture_from(
        LocalModelsConfiguration::default(),
        true,
        requirements.between(),
    )
    .await;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
    while !matches!(desired(&constrained), ServableModelBundle::Standalone { .. }) {
        assert!(
            std::time::Instant::now() < deadline,
            "the catalog model is served plain"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let standard =
        live_fixture_from(LocalModelsConfiguration::default(), true, cpu_environment()).await;
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    assert!(
        matches!(
            desired(&standard),
            ServableModelBundle::SpeculativeDecoding { .. }
        ),
        "with room for it, the catalog drafter stays"
    );
}

/// A different drafter that is the catalog drafter's own file (its package's repository, commit
/// and path) is that listed catalog candidate: no Other row is added, and it is what the entered
/// drafter resolves to. Any other file is its own Other row.
#[tokio::test(flavor = "multi_thread")]
async fn a_different_drafter_that_is_the_catalog_drafters_file_resolves_to_the_catalog_candidate() {
    let fixture = fixture(Vec::new()).await;
    let drafters = |other: String| {
        fixture
            .candidates
            .drafters(DrafterCandidatesRequest::Candidate {
                candidate: hugging_face_candidate(COMMUNITY, 'c', "llama-Q8_0.gguf"),
                other: Some(other),
            })
    };
    let listed = candidate_drafters(&fixture, COMMUNITY, 'c').await;
    let catalog = listed
        .iter()
        .find(|candidate| candidate.category == DrafterCategory::CatalogDrafter)
        .expect("the catalog drafter is listed")
        .reference
        .clone();

    let same = drafters(format!("{CATALOG_DRAFT}:dflash.gguf"))
        .await
        .expect("drafter candidates");
    assert_eq!(same.candidates, listed, "no second row for the same file");
    assert_eq!(same.entered, Some(catalog));

    let third_party = drafters(format!("{THIRD_PARTY}:draft.gguf"))
        .await
        .expect("drafter candidates");
    let other = third_party
        .candidates
        .last()
        .expect("the different drafter is listed");
    assert_eq!(other.category, DrafterCategory::Other);
    assert_eq!(third_party.entered, Some(other.reference.clone()));
}

struct NoCatalogFingerprints;

impl magnitude_service_models::CatalogFingerprints for NoCatalogFingerprints {
    fn catalog_fingerprints(
        &self,
    ) -> BoxFuture<'_, Vec<(ModelId, magnitude_service_contracts::models::ModelFingerprint)>> {
        Box::pin(async { Vec::new() })
    }
}

/// Search applies the engine's interpretation to the model file header it reads, so a repository
/// whose header the engine refuses is shown disabled with the engine's reasons: a tokenizer the
/// leading metadata declares, read from the first bytes alone, and a tensor no admitted role binds,
/// seen when the whole header is read (here because the writer put the block count after the token
/// list). Weight types never disable a row, since another version may have supported ones.
#[tokio::test(flavor = "multi_thread")]
async fn search_disables_a_result_whose_header_the_engine_refuses() {
    // The tokenizer scheme where writers put it, before the token list.
    let mut sentencepiece = llama(7, Encoding::Q8_0);
    for (key, value) in &mut sentencepiece.metadata {
        if key == "tokenizer.ggml.model" {
            *value = Meta::Str("llama".into());
        }
    }
    let mut biased = llama(7, Encoding::Q8_0);
    let block_count = biased
        .metadata
        .iter()
        .position(|(key, _)| key == "llama.block_count")
        .expect("a block count");
    let block_count = biased.metadata.remove(block_count);
    biased.metadata.push(block_count);
    biased.tensor("blk.0.attn_q.bias", &[HEADS * WIDTH], Encoding::F32);
    let mut q2 = llama(10, Encoding::Q8_0);
    q2.encode_all(Encoding::Q8_0, Encoding::Q2K);
    let mut stub = hub();
    let mut search = Vec::new();
    for (id, seed, gguf) in [
        ("someone/Sentencepiece-GGUF", '7', &sentencepiece),
        ("someone/Biased-GGUF", '8', &biased),
        ("someone/Q2-GGUF", '0', &q2),
    ] {
        stub.repositories.insert(
            id.to_owned(),
            repository(seed, vec![StubFile::new("model-Q8_0.gguf", gguf)], &[]),
        );
        search.push(serde_json::json!({
            "id": id,
            "sha": commit(seed),
            "downloads": 10,
            "tags": ["gguf"],
            "gguf": { "architecture": "llama" },
        }));
    }
    stub.search = search;
    let (endpoint, _hub) = stub.serve();
    let directory = tempfile::tempdir().expect("temporary directory");
    let mut config = InventoryConfig::with_roots(
        directory.path().join("models"),
        directory.path().join("cache"),
    )
    .expect("inventory config");
    config.hub_endpoint = Some(endpoint);
    config.architectures = super::engine_architectures();
    let models = Arc::new(ManagedModelStore::open(config, std::sync::Arc::new(|| Ok::<u64, std::convert::Infallible>(u64::MAX))).await.expect("model store"));
    let (catalog, _) = llama_test_catalog(&models).await;
    let discovery = magnitude_service_models::HuggingFaceDiscovery::new(
        Arc::clone(&models),
        super::engine_architectures(),
        Arc::new(crate::assessment::assessor::EngineHeaderVerdicts),
        CatalogLineage::new(Arc::clone(&models), catalog),
        Arc::new(NoCatalogFingerprints),
    );
    let results = magnitude_service_contracts::HuggingFaceModelCatalog::search(
        &discovery,
        magnitude_service_contracts::HuggingFaceModelSearchRequest {
            query: "llama".to_owned(),
            limit: 10,
        },
    )
    .await
    .expect("search");
    let unsupported = |id: &str| {
        results
            .results
            .iter()
            .find(|result| result.repository == id)
            .unwrap_or_else(|| panic!("{id} is a result"))
            .unsupported
            .clone()
    };
    assert_eq!(
        unsupported("someone/Sentencepiece-GGUF"),
        [UnsupportedReason::Known(KnownUnsupportedReason::Tokenizer {
            model: "llama".to_owned(),
            pre: Some("llama-bpe".to_owned()),
        })]
    );
    assert_eq!(
        unsupported("someone/Biased-GGUF"),
        [UnsupportedReason::Known(
            KnownUnsupportedReason::UnrecognizedTensor {
                name: "blk.0.attn_q.bias".to_owned()
            }
        )]
    );
    assert_eq!(unsupported("someone/Q2-GGUF"), []);
    assert_eq!(
        results.results.last().map(|result| result.repository.as_str()),
        Some("someone/Biased-GGUF"),
        "refused results are listed last"
    );
}

// Declared Hub lineage -------------------------------------------------------------------------

const LM_STUDIO_LLAMA: &str = "lmstudio-community/Llama-GGUF";
const QUANTIZED_LLAMA: &str = "community/Llama-Quantized-GGUF";
const LM_STUDIO_QWEN36_27B: &str = "lmstudio-community/Qwen3.6-27B-GGUF";
const QUANTIZED_FILE: &str = "Llama-Q4_K_M.gguf";

fn llama_test_id() -> ModelId {
    "llama-test:gguf:q8".parse().unwrap()
}

fn quantized_llama() -> StubFile {
    StubFile::new(QUANTIZED_FILE, &llama(15, Encoding::Q4K))
}

fn location(
    kind: magnitude_service_contracts::models::ModelRootKind,
    path: &std::path::Path,
) -> LocalModelsConfiguration {
    LocalModelsConfiguration {
        roots: vec![magnitude_service_contracts::models::ModelRoot {
            kind,
            path: path.to_path_buf(),
            enabled: true,
        }],
        ..LocalModelsConfiguration::default()
    }
}

impl LiveFixture {
    /// Waits until the custom model whose file is `file` is recommended `expected`.
    async fn recommended_until(
        &self,
        file: &std::path::Path,
        expected: Option<&DrafterReference>,
        what: &str,
    ) -> CustomModel {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
        loop {
            let listed = self.local.list_local().await.expect("local models");
            let model = listed
                .models
                .iter()
                .find(|model| model.files.first().is_some_and(|first| first.path == file))
                .unwrap_or_else(|| panic!("{what}: no model at {}: {listed:#?}", file.display()));
            let recommended = model
                .recommended_drafter
                .as_ref()
                .map(|candidate| &candidate.reference);
            if recommended == expected {
                return model.clone();
            }
            assert!(
                std::time::Instant::now() < deadline,
                "{what}: recommended {recommended:?}"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
}

/// QA (Linux, round 4): LM Studio's lmstudio-community/gemma-4-E2B-it-GGUF declares
/// google/gemma-4-E2B-it as its base model, and the catalog Gemma 4 E2B's target is a
/// quantization of google/gemma-4-E2B-it-qat-q4_0-unquantized, a finetune of it, so the found
/// model got no catalog drafter. The catalog's upstream follows the same two hops a model's
/// lineage does, so the repository LM Studio's folders name gives the found model lineage, and its
/// catalog drafter is chosen automatically.
#[tokio::test(flavor = "multi_thread")]
async fn a_found_lm_studio_model_whose_repository_names_the_catalog_targets_parent_takes_its_catalog_drafter()
 {
    let lm_studio = tempfile::tempdir().expect("LM Studio folder");
    let model = lm_studio.path().join(LM_STUDIO_LLAMA).join(QUANTIZED_FILE);
    write_sparse(&model, &llama(15, Encoding::Q4K));
    let fixture = live_fixture_reaching(
        location(
            magnitude_service_contracts::models::ModelRootKind::LmStudio,
            lm_studio.path(),
        ),
        false,
        cpu_environment(),
        LiveHub {
            repositories: BTreeMap::from([
                (
                    "base/Llama".to_owned(),
                    repository('0', Vec::new(), &["base/Llama-Instruct"]),
                ),
                (
                    LM_STUDIO_LLAMA.to_owned(),
                    repository('8', vec![quantized_llama()], &["base/Llama-Instruct"]),
                ),
            ]),
            reachable: true,
            ..LiveHub::default()
        },
    )
    .await;
    let listed = fixture
        .recommended_until(
            &model,
            Some(&DrafterReference::Catalog {
                model_id: llama_test_id(),
            }),
            "the LM Studio model's repository lineage",
        )
        .await;
    let recommended = listed.recommended_drafter.expect("a recommended drafter");
    assert_eq!(recommended.lineage, DrafterLineage::Lineage);
    assert_eq!(recommended.label, "Trained for Llama Test");
}

/// QA (Linux, round 4): an unsloth quantization added from Hugging Face, whose repository declares
/// the catalog target's base model, showed no drafter: the answer from what is on this computer
/// used only the lineage its header declares, and the Hub's answer had not replaced it. Its
/// repository's lineage and the catalog's upstream, once read, are on this computer, so with the
/// Hub unreachable that answer already chooses the catalog drafter.
#[tokio::test(flavor = "multi_thread")]
async fn a_models_repository_lineage_already_read_chooses_the_catalog_drafter_while_the_hub_is_unreachable()
 {
    let cache = tempfile::tempdir().expect("Hugging Face cache");
    let file = quantized_llama();
    let repository_root = cache.path().join("models--community--Llama-Quantized-GGUF");
    let blob = repository_root.join("blobs").join(file.sha256());
    write_sparse(&blob, &llama(15, Encoding::Q4K));
    let model = repository_root
        .join("snapshots")
        .join(commit('7'))
        .join(QUANTIZED_FILE);
    std::fs::create_dir_all(model.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(&blob, &model).unwrap();
    std::fs::create_dir_all(repository_root.join("refs")).unwrap();
    std::fs::write(repository_root.join("refs/main"), commit('7')).unwrap();
    let fixture = live_fixture_reaching(
        location(
            magnitude_service_contracts::models::ModelRootKind::HuggingFaceCache,
            cache.path(),
        ),
        false,
        cpu_environment(),
        LiveHub {
            repositories: BTreeMap::from([(
                QUANTIZED_LLAMA.to_owned(),
                repository('7', vec![file], &["base/Llama"]),
            )]),
            snapshots_read: vec![
                (QUANTIZED_LLAMA, commit('7')),
                (QUANTIZED_LLAMA, "main".to_owned()),
                (CATALOG_TARGET, commit('a')),
            ],
            reachable: false,
        },
    )
    .await;
    let listed = fixture
        .recommended_until(
            &model,
            Some(&DrafterReference::Catalog {
                model_id: llama_test_id(),
            }),
            "the repository lineage on this computer",
        )
        .await;
    assert_eq!(
        listed
            .recommended_drafter
            .map(|candidate| candidate.lineage),
        Some(DrafterLineage::Lineage)
    );
}

/// mac-r4-16: lmstudio-community/Qwen3.6-27B-GGUF declares Qwen/Qwen3.6-27B, and the catalog
/// Qwen3.8 27B's upstream is Qwen/Qwen3.8-27B. Declared lineage naming another generation is no
/// lineage, so the catalog drafter it shares every dimension with stays untested and never
/// automatic.
#[tokio::test(flavor = "multi_thread")]
async fn a_repository_declaring_another_generation_never_makes_the_shape_matched_drafter_automatic()
{
    let fixture = fixture_from(
        Vec::new(),
        LocalModelsConfiguration::default(),
        false,
        BTreeMap::from([(
            LM_STUDIO_QWEN36_27B.to_owned(),
            repository(
                '9',
                vec![fixture_file(
                    &format!("headers/{QWEN36_27B_FILE}.gz"),
                    QWEN36_27B_FILE,
                )],
                &["Qwen/Qwen3.6-27B"],
            ),
        )]),
        true,
        cpu_environment(),
    )
    .await;
    let candidates = fixture
        .candidates
        .drafters(DrafterCandidatesRequest::Candidate {
            candidate: hugging_face_candidate(LM_STUDIO_QWEN36_27B, '9', QWEN36_27B_FILE),
            other: None,
        })
        .await
        .expect("drafter candidates")
        .candidates;
    let listed = candidates
        .iter()
        .map(|candidate| {
            (
                candidate.category,
                candidate.lineage,
                candidate.method.clone(),
                is_bound(&candidate.verdict),
                candidate.automatic,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        listed,
        [(
            DrafterCategory::CatalogDrafter,
            DrafterLineage::Fingerprint,
            Some(SpeculativeMethod::DFlash2),
            true,
            false,
        )],
        "{candidates:#?}"
    );
}
