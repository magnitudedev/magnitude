//! The engine boundary over synthetic GGUF headers: every unsupported
//! reason a header check raises, all of them at once, unreadable files, the
//! drafter verdict and the shape fingerprint.
//!
//! Header checks are exercised through [`kernel_inventory`], which admits and
//! plans a package without a device. Drafter verdicts and memory need a
//! selected device; they run on the CPU backend, which every build has.

use magnitude_artifacts::gguf::Encoding;
use magnitude_engine::{
    assessment::{
        assess_model, kernel_inventory, prepare_model_assessment, AssessmentSetup,
        DrafterVerdict, KernelInventory, ModelAssessment, ModelPackagePaths,
        PreparedModelAssessment, SpeculativeMethod,
    },
    error::{UnsupportedModel, UnsupportedReason, WeightTypeCount},
    fingerprint::ModelFingerprint,
    options::{standard_service_limits, ModelMethod, ModelPolicy},
};
use magnitude_executor::{
    assessment::ExecutionAssessment,
    platform::{DeviceRequest, MemoryReserves},
};
use seismic::BackendName;
use std::path::{Path, PathBuf};

enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strings(Vec<String>),
    I32s(Vec<i32>),
    U32s(Vec<u32>),
}

/// A GGUF header written without tensor payloads: assessment reads headers
/// only.
#[derive(Default)]
struct Gguf {
    metadata: Vec<(String, Meta)>,
    tensors: Vec<(String, Vec<u64>, u32)>,
}

impl Gguf {
    fn set(&mut self, key: &str, value: Meta) {
        self.metadata.retain(|(name, _)| name != key);
        self.metadata.push((key.into(), value));
    }

    fn remove(&mut self, key: &str) {
        self.metadata.retain(|(name, _)| name != key);
    }

    /// `shape` is outermost first, as the engine reads it.
    fn tensor(&mut self, name: &str, shape: &[u64], encoding: Encoding) {
        self.tensor_type(name, shape, encoding as u32);
    }

    fn tensor_type(&mut self, name: &str, shape: &[u64], type_id: u32) {
        self.tensors.retain(|(existing, ..)| existing != name);
        self.tensors.push((name.into(), shape.to_vec(), type_id));
    }

    fn encode(&mut self, name: &str, encoding: Encoding) {
        let tensor = self
            .tensors
            .iter_mut()
            .find(|(existing, ..)| existing == name)
            .unwrap_or_else(|| panic!("no tensor {name}"));
        tensor.2 = encoding as u32;
    }

    fn remove_tensor(&mut self, name: &str) {
        self.tensors.retain(|(existing, ..)| existing != name);
    }

    fn bytes(&self) -> Vec<u8> {
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
        for (name, shape, type_id) in &self.tensors {
            string(&mut out, name);
            out.extend((shape.len() as u32).to_le_bytes());
            for dimension in shape.iter().rev() {
                out.extend(dimension.to_le_bytes());
            }
            out.extend(type_id.to_le_bytes());
            out.extend(offset.to_le_bytes());
            let bytes = Encoding::ALL
                .into_iter()
                .find(|encoding| *encoding as u32 == *type_id)
                .map_or(0, |encoding| {
                    shape.iter().product::<u64>() / encoding.block_elements()
                        * encoding.block_bytes()
                });
            offset += bytes.div_ceil(32) * 32;
        }
        out.resize(out.len().div_ceil(32) * 32, 0);
        out
    }

    fn write(&self, path: &Path) -> PathBuf {
        std::fs::write(path, self.bytes()).unwrap();
        path.to_path_buf()
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

/// A two-layer target of the admitted `llama` feature set with a supported
/// tokenizer and a ChatML template.
fn llama() -> Gguf {
    let mut gguf = Gguf::default();
    gguf.set("general.architecture", Meta::Str("llama".into()));
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
        gguf.tensor(&name("attn_q"), &[query, HIDDEN], Encoding::Q8_0);
        gguf.tensor(&name("attn_k"), &[key, HIDDEN], Encoding::Q8_0);
        gguf.tensor(&name("attn_v"), &[key, HIDDEN], Encoding::Q8_0);
        gguf.tensor(&name("attn_output"), &[HIDDEN, query], Encoding::Q8_0);
        gguf.tensor(&name("ffn_norm"), &[HIDDEN], Encoding::F32);
        gguf.tensor(&name("ffn_gate"), &[FEED_FORWARD, HIDDEN], Encoding::Q8_0);
        gguf.tensor(&name("ffn_up"), &[FEED_FORWARD, HIDDEN], Encoding::Q8_0);
        gguf.tensor(&name("ffn_down"), &[HIDDEN, FEED_FORWARD], Encoding::Q8_0);
    }
    gguf
}

/// A one-layer DFlash drafter of width `hidden` tapping the target's first
/// layer.
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

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "magnitude-assessment-boundary-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn package(target: PathBuf) -> ModelPackagePaths {
    ModelPackagePaths {
        target,
        projector: None,
        draft: None,
        method: ModelMethod::Auto,
    }
}

/// The header-only inventory a CPU load of `package` would prepare, or why
/// the engine refuses it.
fn inventory(package: &ModelPackagePaths) -> KernelInventory {
    kernel_inventory(
        package,
        &ModelPolicy::default(),
        &standard_service_limits(),
        BackendName::Cpu,
    )
    .unwrap_or_else(|error| panic!("operational failure: {error}"))
}

fn refusal(gguf: &Gguf, name: &str) -> UnsupportedModel {
    let scratch = Scratch::new(name);
    let target = gguf.write(&scratch.file("target.gguf"));
    match inventory(&package(target)) {
        KernelInventory::Unsupported(unsupported) => unsupported,
        KernelInventory::Requests(_) => panic!("{name}: the package is admitted"),
    }
}

#[test]
fn the_synthetic_target_is_admitted() {
    let scratch = Scratch::new("admitted");
    let target = llama().write(&scratch.file("target.gguf"));
    let KernelInventory::Requests(requests) = inventory(&package(target)) else {
        panic!("the synthetic llama target is supported");
    };
    assert!(!requests.is_empty());
}

#[test]
fn an_unknown_architecture_is_named() {
    for architecture in ["qwen3", "deepseek4"] {
        let mut gguf = llama();
        gguf.set("general.architecture", Meta::Str(architecture.into()));
        assert_eq!(
            refusal(&gguf, architecture).reasons,
            [UnsupportedReason::Architecture {
                architecture: architecture.into()
            }]
        );
    }
}

#[test]
fn every_unsupported_weight_type_is_counted() {
    let mut gguf = llama();
    gguf.encode("blk.0.ffn_up.weight", Encoding::Q2K);
    gguf.encode("blk.1.ffn_up.weight", Encoding::Q2K);
    gguf.encode("blk.1.ffn_gate.weight", Encoding::Iq1M);
    assert_eq!(
        refusal(&gguf, "weights").reasons,
        [UnsupportedReason::WeightTypes {
            types: vec![
                WeightTypeCount {
                    ggml_type: "IQ1_M".into(),
                    tensor_count: 1
                },
                WeightTypeCount {
                    ggml_type: "Q2_K".into(),
                    tensor_count: 2
                },
            ]
        }]
    );
}

#[test]
fn every_unknown_metadata_key_is_named() {
    let mut gguf = llama();
    gguf.set("llama.attention.sliding_window", Meta::U32(4096));
    gguf.set("llama.expert_count", Meta::U32(8));
    assert_eq!(
        refusal(&gguf, "metadata").reasons,
        [
            UnsupportedReason::UnrecognizedMetadata {
                key: "llama.attention.sliding_window".into()
            },
            UnsupportedReason::UnrecognizedMetadata {
                key: "llama.expert_count".into()
            },
        ]
    );
}

#[test]
fn every_unbound_tensor_is_named() {
    let mut gguf = llama();
    gguf.tensor("blk.0.attn_q.bias", &[HEADS * WIDTH], Encoding::F32);
    gguf.tensor("blk.1.attn_k_norm.weight", &[WIDTH], Encoding::F32);
    assert_eq!(
        refusal(&gguf, "tensors").reasons,
        [
            UnsupportedReason::UnrecognizedTensor {
                name: "blk.0.attn_q.bias".into()
            },
            UnsupportedReason::UnrecognizedTensor {
                name: "blk.1.attn_k_norm.weight".into()
            },
        ]
    );
}

/// Llama 3.1's GGUF carries its rope scaling as `rope_freqs.weight`.
#[test]
fn llama_rope_frequencies_are_an_unrecognized_tensor() {
    let mut gguf = llama();
    gguf.tensor("rope_freqs.weight", &[WIDTH / 2], Encoding::F32);
    assert_eq!(
        refusal(&gguf, "rope-freqs").reasons,
        [UnsupportedReason::UnrecognizedTensor {
            name: "rope_freqs.weight".into()
        }]
    );
}

#[test]
fn a_tied_llama_head_is_named() {
    let mut gguf = llama();
    gguf.remove_tensor("output.weight");
    assert_eq!(
        refusal(&gguf, "tied").reasons,
        [UnsupportedReason::TiedOutputHead]
    );
}

#[test]
fn llama_rope_scaling_is_named() {
    let mut gguf = llama();
    gguf.set("llama.rope.scaling.type", Meta::Str("linear".into()));
    gguf.set("llama.rope.scaling.factor", Meta::F32(4.0));
    assert_eq!(
        refusal(&gguf, "llama-rope-scaling").reasons,
        [UnsupportedReason::RopeScaling {
            kind: "linear".into()
        }]
    );
}

#[test]
fn an_unsupported_tokenizer_names_its_profile_and_every_unknown_key() {
    let mut gguf = llama();
    gguf.set("tokenizer.ggml.pre", Meta::Str("deepseek-v3".into()));
    gguf.set("tokenizer.ggml.precompiled_charsmap", Meta::Str("x".into()));
    assert_eq!(
        refusal(&gguf, "tokenizer").reasons,
        [
            UnsupportedReason::UnrecognizedMetadata {
                key: "tokenizer.ggml.precompiled_charsmap".into()
            },
            UnsupportedReason::Tokenizer {
                model: "gpt2".into(),
                pre: Some("deepseek-v3".into())
            },
        ]
    );
}

#[test]
fn an_unreadable_chat_template_is_named() {
    let mut gguf = llama();
    gguf.set(
        "tokenizer.chat_template",
        Meta::Str("{% for message in messages %}{{ message.content }".into()),
    );
    assert_eq!(
        refusal(&gguf, "template").reasons,
        [UnsupportedReason::ChatTemplate]
    );
}

/// A weight-type refusal does not stop the family, tokenizer and template
/// checks: each independent check's reasons are reported together.
#[test]
fn independent_checks_report_every_reason_together() {
    let mut gguf = llama();
    gguf.encode("blk.0.ffn_down.weight", Encoding::Q2K);
    gguf.tensor("rope_freqs.weight", &[WIDTH / 2], Encoding::F32);
    gguf.set("tokenizer.ggml.pre", Meta::Str("deepseek-v3".into()));
    gguf.remove("tokenizer.chat_template");
    gguf.set("tokenizer.chat_template", Meta::Str("{{".into()));
    let refusal = refusal(&gguf, "union");
    assert_eq!(
        refusal.reasons,
        [
            UnsupportedReason::WeightTypes {
                types: vec![WeightTypeCount {
                    ggml_type: "Q2_K".into(),
                    tensor_count: 1
                }]
            },
            UnsupportedReason::UnrecognizedTensor {
                name: "rope_freqs.weight".into()
            },
            UnsupportedReason::Tokenizer {
                model: "gpt2".into(),
                pre: Some("deepseek-v3".into())
            },
            UnsupportedReason::ChatTemplate,
        ]
    );
    assert!(refusal.detail.contains("Q2_K"), "{}", refusal.detail);
}

#[test]
fn truncated_and_invalid_files_are_unreadable_and_missing_files_are_operational() {
    let scratch = Scratch::new("unreadable");
    let bytes = llama().bytes();
    let truncated = scratch.file("truncated.gguf");
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
    let mut unknown_type = llama();
    unknown_type.tensor_type("blk.0.ffn_up.weight", &[FEED_FORWARD, HIDDEN], 99);
    let unknown_type = unknown_type.write(&scratch.file("unknown-type.gguf"));
    let not_gguf = scratch.file("not.gguf");
    std::fs::write(&not_gguf, b"not a model at all").unwrap();
    for path in [truncated, unknown_type, not_gguf] {
        let KernelInventory::Unsupported(unsupported) = inventory(&package(path.clone())) else {
            panic!("{} is admitted", path.display());
        };
        assert_eq!(
            unsupported.reasons,
            [UnsupportedReason::Unreadable],
            "{path:?}"
        );
    }
    assert!(kernel_inventory(
        &package(scratch.file("absent.gguf")),
        &ModelPolicy::default(),
        &standard_service_limits(),
        BackendName::Cpu,
    )
    .is_err());
}

#[test]
fn the_fingerprint_reads_the_header_shape_and_the_token_table() {
    let scratch = Scratch::new("fingerprint");
    let gguf = llama();
    let path = gguf.write(&scratch.file("target.gguf"));
    let fingerprint = ModelFingerprint::from_path(&path).unwrap();
    assert_eq!(
        fingerprint,
        ModelFingerprint {
            architecture: "llama".into(),
            block_count: BLOCKS as u32,
            embedding_length: HIDDEN as u32,
            head_count: HEADS as u32,
            head_count_kv: KV_HEADS as u32,
            key_length: WIDTH as u32,
            vocab_size: VOCABULARY as u32,
            expert_count: None,
            tokenizer_digest: fingerprint.tokenizer_digest.clone(),
        }
    );

    // A requantization keeps the fingerprint; a changed token table does not.
    let mut requantized = llama();
    requantized.encode("blk.0.ffn_up.weight", Encoding::Q4K);
    requantized.remove("tokenizer.chat_template");
    let requantized = requantized.write(&scratch.file("requantized.gguf"));
    assert_eq!(
        ModelFingerprint::from_path(&requantized).unwrap(),
        fingerprint
    );
    let mut retokenized = llama();
    let mut table = tokens();
    table[10] = "t10x".into();
    retokenized.set("tokenizer.ggml.tokens", Meta::Strings(table));
    let retokenized = retokenized.write(&scratch.file("retokenized.gguf"));
    assert_ne!(
        ModelFingerprint::from_path(&retokenized)
            .unwrap()
            .tokenizer_digest,
        fingerprint.tokenizer_digest
    );

    // MTP heads counted as extra blocks don't change the fingerprint: Ollama's qwen3.5:4b
    // (32 blocks, heads dropped) matches the catalog's unsloth MTP target (33 blocks, one
    // prediction layer).
    let mut with_heads = llama();
    with_heads.set("llama.block_count", Meta::U32(BLOCKS as u32 + 1));
    with_heads.set("llama.nextn_predict_layers", Meta::U32(1));
    let with_heads = with_heads.write(&scratch.file("with-heads.gguf"));
    assert_eq!(ModelFingerprint::from_path(&with_heads).unwrap(), fingerprint);

    // The digest is pinned: sha256 over the length-prefixed version tag and
    // tokens.
    let mut small = llama();
    small.set(
        "tokenizer.ggml.tokens",
        Meta::Strings(vec!["a".into(), "bc".into()]),
    );
    let small = small.write(&scratch.file("small.gguf"));
    assert_eq!(
        ModelFingerprint::from_path(&small)
            .unwrap()
            .tokenizer_digest,
        "eebfe72455238a43be77d699ab93425a2fc9ac6debeeabd32a4ac7ab96b7ceb7"
    );

    // Absent key-value heads and key length are derived; per-layer counts
    // take the largest layer's, and a nonzero expert count is kept.
    let mut derived = llama();
    derived.remove("llama.attention.key_length");
    derived.set("llama.attention.head_count_kv", Meta::U32s(vec![0, 4, 2]));
    derived.set("llama.expert_count", Meta::U32(64));
    let derived =
        ModelFingerprint::from_path(&derived.write(&scratch.file("derived.gguf"))).unwrap();
    assert_eq!(
        (
            derived.head_count_kv,
            derived.key_length,
            derived.expert_count
        ),
        (4, (HIDDEN / HEADS) as u32, Some(64))
    );
}

fn cpu_setup() -> AssessmentSetup {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    AssessmentSetup::discover(
        &catalog,
        DeviceRequest::Backend(BackendName::Cpu),
        MemoryReserves::standard(),
        ModelPolicy::default(),
        standard_service_limits(),
    )
    .unwrap_or_else(|error| panic!("the CPU backend is selectable: {error}"))
}

fn prepared(package: &ModelPackagePaths) -> PreparedModelAssessment {
    prepare_model_assessment(package, &cpu_setup())
        .unwrap_or_else(|error| panic!("operational failure: {error}"))
}

#[test]
fn an_unsupported_target_reports_its_fingerprint() {
    let scratch = Scratch::new("unsupported-fingerprint");
    let mut gguf = llama();
    gguf.encode("blk.0.ffn_up.weight", Encoding::Q2K);
    let target = gguf.write(&scratch.file("target.gguf"));
    let PreparedModelAssessment::Unsupported {
        unsupported,
        fingerprint,
    } = prepared(&package(target.clone()))
    else {
        panic!("Q2_K is not imported");
    };
    assert!(matches!(
        unsupported.reasons.as_slice(),
        [UnsupportedReason::WeightTypes { .. }]
    ));
    assert_eq!(fingerprint, ModelFingerprint::from_path(&target).ok());
}

#[test]
fn a_target_without_a_drafter_drafts_with_nothing() {
    let scratch = Scratch::new("no-drafter");
    let target = llama().write(&scratch.file("target.gguf"));
    let PreparedModelAssessment::Planned { facts, .. } = prepared(&package(target.clone())) else {
        panic!("the target plans");
    };
    assert_eq!(facts.drafter, DrafterVerdict::None);
    assert_eq!(facts.fingerprint, ModelFingerprint::from_path(&target).ok());
}

#[test]
fn a_drafter_of_another_width_is_rejected_and_the_target_still_plans() {
    let scratch = Scratch::new("rejected-drafter");
    let target = llama().write(&scratch.file("target.gguf"));
    let draft = dflash(HIDDEN / 2).write(&scratch.file("draft.gguf"));
    let PreparedModelAssessment::Planned { facts, .. } = prepared(&ModelPackagePaths {
        draft: Some(draft),
        method: ModelMethod::DFlash,
        ..package(target)
    }) else {
        panic!("the target plans without the drafter");
    };
    let DrafterVerdict::Rejected(rejection) = facts.drafter else {
        panic!(
            "a drafter of another width does not bind: {:?}",
            facts.drafter
        );
    };
    assert!(
        matches!(
            rejection.reasons.as_slice(),
            [UnsupportedReason::TargetMismatch { detail }] if detail.contains("width")
        ),
        "{rejection:?}"
    );
}

#[test]
fn a_drafter_with_unsupported_weights_names_them() {
    let scratch = Scratch::new("drafter-weights");
    let target = llama().write(&scratch.file("target.gguf"));
    let mut draft = dflash(HIDDEN);
    draft.encode("blk.0.ffn_up.weight", Encoding::Iq2S);
    let draft = draft.write(&scratch.file("draft.gguf"));
    let PreparedModelAssessment::Planned { facts, .. } = prepared(&ModelPackagePaths {
        draft: Some(draft),
        method: ModelMethod::DFlash,
        ..package(target)
    }) else {
        panic!("the target plans without the drafter");
    };
    let DrafterVerdict::Rejected(rejection) = facts.drafter else {
        panic!("IQ2_S is not imported: {:?}", facts.drafter);
    };
    assert_eq!(
        rejection.reasons,
        [UnsupportedReason::WeightTypes {
            types: vec![WeightTypeCount {
                ggml_type: "IQ2_S".into(),
                tensor_count: 1
            }]
        }]
    );
}

#[test]
fn a_drafter_that_binds_is_bound_with_the_memory_it_adds() {
    let scratch = Scratch::new("bound-drafter");
    let target = llama().write(&scratch.file("target.gguf"));
    let draft = dflash(HIDDEN).write(&scratch.file("draft.gguf"));
    let PreparedModelAssessment::Planned { facts, .. } = prepared(&ModelPackagePaths {
        draft: Some(draft),
        method: ModelMethod::Auto,
        ..package(target)
    }) else {
        panic!("the bundle plans");
    };
    let DrafterVerdict::Bound {
        method,
        memory_bytes,
    } = facts.drafter
    else {
        panic!("the drafter binds: {:?}", facts.drafter);
    };
    assert_eq!(method, SpeculativeMethod::DFlash);
    assert!(memory_bytes > 0);
}

#[test]
fn an_unreadable_drafter_is_rejected_without_the_target() {
    let scratch = Scratch::new("unreadable-drafter");
    let target = llama().write(&scratch.file("target.gguf"));
    let draft = scratch.file("draft.gguf");
    std::fs::write(&draft, &dflash(HIDDEN).bytes()[..64]).unwrap();
    let PreparedModelAssessment::Planned { facts, .. } = prepared(&ModelPackagePaths {
        draft: Some(draft),
        method: ModelMethod::DFlash,
        ..package(target)
    }) else {
        panic!("the target plans without the drafter");
    };
    let DrafterVerdict::Rejected(rejection) = facts.drafter else {
        panic!("a truncated drafter is rejected: {:?}", facts.drafter);
    };
    assert_eq!(rejection.reasons, [UnsupportedReason::Unreadable]);
}


/// The synthetic target with attention heads of `width`.
fn llama_of_width(width: u64) -> Gguf {
    let mut gguf = llama();
    for key in ["key_length", "value_length"] {
        gguf.set(&format!("llama.attention.{key}"), Meta::U32(width as u32));
    }
    gguf.set("llama.rope.dimension_count", Meta::U32(width as u32));
    for block in 0..BLOCKS {
        let name = |role: &str| format!("blk.{block}.{role}.weight");
        gguf.tensor(&name("attn_q"), &[HEADS * width, HIDDEN], Encoding::Q8_0);
        gguf.tensor(&name("attn_k"), &[KV_HEADS * width, HIDDEN], Encoding::Q8_0);
        gguf.tensor(&name("attn_v"), &[KV_HEADS * width, HIDDEN], Encoding::Q8_0);
        gguf.tensor(&name("attn_output"), &[HIDDEN, HEADS * width], Encoding::Q8_0);
    }
    gguf
}

/// The refusal of `package` on the CPU backend under `policy`, from both
/// the device-free inventory and the assessment of the selected device:
/// they classify alike.
fn cpu_refusal(package: &ModelPackagePaths, policy: ModelPolicy) -> UnsupportedModel {
    let KernelInventory::Unsupported(inventoried) =
        kernel_inventory(package, &policy, &standard_service_limits(), BackendName::Cpu)
            .unwrap_or_else(|error| panic!("operational failure: {error}"))
    else {
        panic!("the package is admitted");
    };
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let setup = AssessmentSetup::discover(
        &catalog,
        DeviceRequest::Backend(BackendName::Cpu),
        MemoryReserves::standard(),
        policy,
        standard_service_limits(),
    )
    .unwrap();
    let PreparedModelAssessment::Unsupported {
        unsupported: assessed,
        ..
    } = prepare_model_assessment(package, &setup)
        .unwrap_or_else(|error| panic!("operational failure: {error}"))
    else {
        panic!("the package plans");
    };
    assert_eq!(inventoried.reasons, assessed.reasons);
    assessed
}

#[test]
fn a_form_the_backend_does_not_execute_names_the_backend() {
    let scratch = Scratch::new("backend");
    let target = llama().write(&scratch.file("target.gguf"));
    let rotated = ModelPolicy {
        kv_codec: magnitude_state::KvCodec::RotatedK4V4,
        ..ModelPolicy::default()
    };
    assert_eq!(
        cpu_refusal(&package(target), rotated).reasons,
        [UnsupportedReason::Backend {
            backend: "cpu".into()
        }]
    );

    // A planner refusal of a validated definition names the backend too: no
    // codec group divides a head width of 72.
    let target = llama_of_width(72).write(&scratch.file("width-72.gguf"));
    let refusal = cpu_refusal(&package(target), ModelPolicy::default());
    assert_eq!(
        refusal.reasons,
        [UnsupportedReason::Backend {
            backend: "cpu".into()
        }],
        "{}",
        refusal.detail
    );
}

#[test]
fn a_kernel_call_outside_its_domain_names_the_kernel_and_its_statics() {
    let scratch = Scratch::new("kernel-domain");
    let target = llama_of_width(96).write(&scratch.file("target.gguf"));
    let refusal = cpu_refusal(&package(target), ModelPolicy::default());
    let [UnsupportedReason::KernelDomain {
        backend,
        entry,
        statics,
    }] = refusal.reasons.as_slice()
    else {
        panic!("a head width of 96 is outside a kernel domain: {refusal:?}");
    };
    assert_eq!(backend, "cpu");
    assert!(!entry.is_empty());
    assert!(!statics.is_empty());
}

/// Memory is charged as stored after import: Q3_K weights are held widened
/// as q6k, so a Q3_K file is charged as its Q6_K twin though it is smaller.
#[test]
fn widened_weights_are_charged_as_resident() {
    let scratch = Scratch::new("widened");
    let charged = |encoding: Encoding, name: &str| {
        let mut gguf = llama();
        let mut stored = 0;
        for block in 0..BLOCKS {
            for role in ["ffn_gate", "ffn_up", "ffn_down"] {
                let tensor = format!("blk.{block}.{role}.weight");
                gguf.encode(&tensor, encoding);
                stored += FEED_FORWARD * HIDDEN / encoding.block_elements()
                    * encoding.block_bytes();
            }
        }
        let target = gguf.write(&scratch.file(name));
        let ModelAssessment::Assessed {
            execution: ExecutionAssessment::Fits { domains, .. },
            ..
        } = assess_model(&package(target), &cpu_setup(), &[])
            .unwrap_or_else(|error| panic!("operational failure: {error}"))
        else {
            panic!("{name} fits");
        };
        let required = domains
            .iter()
            .map(|domain| domain.required_bytes)
            .collect::<Vec<_>>();
        (stored, required)
    };
    let (q3k_stored, q3k_required) = charged(Encoding::Q3K, "q3k.gguf");
    let (q6k_stored, q6k_required) = charged(Encoding::Q6K, "q6k.gguf");
    assert!(q3k_stored < q6k_stored);
    assert_eq!(q3k_required, q6k_required);
    assert!(q3k_required.iter().all(|required| *required > q6k_stored));
}
