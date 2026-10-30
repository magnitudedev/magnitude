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
        assess_model, header_refusal, kernel_inventory, prepare_model_assessment,
        AssessmentSetup, AssessmentStructure, DrafterVerdict, HeaderExtent, KernelInventory,
        Measurements, ModelAssessment, ModelExecution, ModelFacts, ModelPackagePaths,
        PreparedModelAssessment, ProjectorVerdict, SpeculativeMethod,
    },
    error::{UnsupportedModel, UnsupportedReason, WeightTypeCount},
    fingerprint::ModelFingerprint,
    options::{standard_service_limits, ModelMethod, ModelPolicy, PROFILE_SEQUENCES},
};
use magnitude_executor::{
    assessment::ExecutionFit,
    platform::{DeviceRequest, MemoryReserves},
};
use seismic::BackendName;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    gguf.encode("blk.0.ffn_up.weight", Encoding::Q8K);
    gguf.encode("blk.1.ffn_up.weight", Encoding::Q8K);
    gguf.encode("blk.1.ffn_gate.weight", Encoding::Q8_1);
    assert_eq!(
        refusal(&gguf, "weights").reasons,
        [UnsupportedReason::WeightTypes {
            types: vec![
                WeightTypeCount {
                    ggml_type: "Q8_1".into(),
                    tensor_count: 1
                },
                WeightTypeCount {
                    ggml_type: "Q8_K".into(),
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
    gguf.encode("blk.0.ffn_down.weight", Encoding::Q8K);
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
                    ggml_type: "Q8_K".into(),
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
    assert!(refusal.detail.contains("Q8_K"), "{}", refusal.detail);
}

/// A header's own refusal is admission's without the weight types: with the
/// whole header, the family, tokenizer and template reasons; with leading
/// metadata only, what those entries settle, and nothing the entries a prefix
/// doesn't reach would decide.
#[test]
fn a_header_refuses_what_admission_does_without_its_weight_types() {
    let directory =
        |gguf: &Gguf| magnitude_artifacts::gguf::read_header(&gguf.bytes()).expect("header");
    let reasons = |gguf: &Gguf, extent| {
        header_refusal(&directory(gguf), extent).map(|refusal| refusal.reasons)
    };
    let mut quantized = llama();
    quantized.encode("blk.0.ffn_down.weight", Encoding::Q8K);
    assert_eq!(reasons(&quantized, HeaderExtent::Whole), None);
    let mut refused = quantized;
    refused.tensor("blk.0.attn_q.bias", &[HEADS * WIDTH], Encoding::F32);
    refused.set("tokenizer.ggml.pre", Meta::Str("deepseek-v3".into()));
    assert_eq!(
        reasons(&refused, HeaderExtent::Whole),
        Some(vec![
            UnsupportedReason::UnrecognizedTensor {
                name: "blk.0.attn_q.bias".into()
            },
            UnsupportedReason::Tokenizer {
                model: "gpt2".into(),
                pre: Some("deepseek-v3".into())
            },
        ])
    );
    // A prefix: the metadata before the token list, no tensor directory.
    let leading = |model: &str| {
        let mut gguf = Gguf::default();
        for (key, value) in llama().metadata {
            if key == "tokenizer.ggml.tokens" {
                break;
            }
            gguf.set(&key, value);
        }
        gguf.set("tokenizer.ggml.model", Meta::Str(model.into()));
        gguf
    };
    assert_eq!(reasons(&leading("gpt2"), HeaderExtent::Leading), None);
    assert_eq!(
        reasons(&leading("llama"), HeaderExtent::Leading),
        Some(vec![UnsupportedReason::Tokenizer {
            model: "llama".into(),
            pre: Some("llama-bpe".into())
        }])
    );
    let mut unknown = leading("gpt2");
    unknown.set("general.architecture", Meta::Str("deepseek4".into()));
    assert_eq!(
        reasons(&unknown, HeaderExtent::Leading),
        Some(vec![UnsupportedReason::Architecture {
            architecture: "deepseek4".into()
        }])
    );
}

#[test]
fn truncated_and_invalid_files_are_unreadable_and_missing_files_are_operational() {
    let scratch = Scratch::new("unreadable");
    let bytes = llama().bytes();
    let truncated = scratch.file("truncated.gguf");
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
    let not_gguf = scratch.file("not.gguf");
    std::fs::write(&not_gguf, b"not a model at all").unwrap();
    for path in [truncated, not_gguf] {
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

fn cpu_structure() -> AssessmentStructure {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    AssessmentStructure::discover(
        &catalog,
        DeviceRequest::Backend(BackendName::Cpu),
        MemoryReserves::standard(),
        ModelPolicy::default(),
        standard_service_limits(),
    )
    .unwrap_or_else(|error| panic!("the CPU backend is selectable: {error}"))
}

/// The CPU backend before any measurement: fit is assessed, speed is pending.
fn cpu_setup() -> AssessmentSetup {
    AssessmentSetup {
        structure: Arc::new(cpu_structure()),
        measurements: Measurements::Pending,
    }
}

fn prepared(package: &ModelPackagePaths) -> PreparedModelAssessment {
    prepare_model_assessment(package, &cpu_structure())
        .unwrap_or_else(|error| panic!("operational failure: {error}"))
}

#[test]
fn an_unsupported_target_reports_its_fingerprint() {
    let scratch = Scratch::new("unsupported-fingerprint");
    let mut gguf = llama();
    gguf.encode("blk.0.ffn_up.weight", Encoding::Q8K);
    let target = gguf.write(&scratch.file("target.gguf"));
    let PreparedModelAssessment::Unsupported {
        unsupported,
        fingerprint,
    } = prepared(&package(target.clone()))
    else {
        panic!("Q8_K is not imported");
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
    draft.encode("blk.0.ffn_up.weight", Encoding::Q8K);
    let draft = draft.write(&scratch.file("draft.gguf"));
    let PreparedModelAssessment::Planned { facts, .. } = prepared(&ModelPackagePaths {
        draft: Some(draft),
        method: ModelMethod::DFlash,
        ..package(target)
    }) else {
        panic!("the target plans without the drafter");
    };
    let DrafterVerdict::Rejected(rejection) = facts.drafter else {
        panic!("Q8_K is not imported: {:?}", facts.drafter);
    };
    assert_eq!(
        rejection.reasons,
        [UnsupportedReason::WeightTypes {
            types: vec![WeightTypeCount {
                ggml_type: "Q8_K".into(),
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
    let catalog = seismic::Devices::discover(&[]).unwrap();
    let structure = AssessmentStructure::discover(
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
    } = prepare_model_assessment(package, &structure)
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

/// Memory is charged as stored after import: TQ1_0 weights are held widened
/// as q2g256, TQ2_0's form, so a TQ1_0 file is charged as its TQ2_0 twin
/// though it is smaller.
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
            execution:
                ModelExecution::Fits {
                    fit: ExecutionFit { domains, .. },
                    ..
                },
            ..
        } = assess_model(&package(target), &cpu_setup(), &[], PROFILE_SEQUENCES)
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
    let (tq1_stored, tq1_required) = charged(Encoding::Tq1_0, "tq1_0.gguf");
    let (tq2_stored, tq2_required) = charged(Encoding::Tq2_0, "tq2_0.gguf");
    assert!(tq1_stored < tq2_stored);
    assert_eq!(tq1_required, tq2_required);
    assert!(tq1_required.iter().all(|required| *required > tq2_stored));
}

/// A shared real-header fixture (`fixtures/assessment/headers`),
/// decompressed into `scratch` under its repository file name.
fn real_header(scratch: &Scratch, name: &str) -> PathBuf {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/assessment/headers")
        .join(format!("{name}.gz"));
    let mut decoder = flate2::read::GzDecoder::new(
        std::fs::File::open(&source).unwrap_or_else(|error| panic!("{}: {error}", source.display())),
    );
    let path = scratch.file(name);
    std::io::copy(&mut decoder, &mut std::fs::File::create(&path).unwrap()).unwrap();
    path
}

/// A projector never decides the drafter: unsloth's Qwen3.5 0.8B MTP target
/// binds its built-in heads alone and beside its projector, where it also
/// gains vision. Generation drafts its text requests and generates requests
/// with media rows (which the heads cannot enter) without drafting.
#[test]
fn embedded_mtp_heads_stay_bound_beside_a_projector() {
    let scratch = Scratch::new("mtp-projector");
    let target = real_header(&scratch, "Qwen3.5-0.8B-Q3_K_M.gguf");
    let projector = real_header(&scratch, "mmproj-F16.gguf");
    let assessed = |projector: Option<PathBuf>| {
        let package = ModelPackagePaths {
            projector,
            ..package(target.clone())
        };
        let PreparedModelAssessment::Planned { facts, .. } = prepared(&package) else {
            panic!("the target plans");
        };
        facts
    };
    for (projector, vision) in [(None, false), (Some(projector), true)] {
        let facts = assessed(projector);
        assert_eq!(facts.capabilities.vision, vision);
        assert!(
            matches!(
                facts.drafter,
                DrafterVerdict::Bound {
                    method: SpeculativeMethod::Mtp,
                    ..
                }
            ),
            "vision {vision}: {:?}",
            facts.drafter
        );
    }
}

/// A projector binds by its width, never by names: DavidAU's Qwen3.5 9B finetune names another
/// base model than its repository's projector does, and plans with vision beside it; a projector
/// of another width (Qwen3.5 0.8B's) is a mismatch the target plans without, taking text input
/// only, never an unsupported or unreadable target.
#[test]
fn a_projector_binds_by_its_width_not_the_base_model_names() {
    let scratch = Scratch::new("projector-width");
    let target = real_header(
        &scratch,
        "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q4_K_M.gguf",
    );
    let own = real_header(&scratch, "mmproj-BF16.gguf");
    let narrow = real_header(&scratch, "mmproj-F16.gguf");
    let with = |projector: PathBuf| ModelPackagePaths {
        projector: Some(projector),
        ..package(target.clone())
    };
    let PreparedModelAssessment::Planned { facts, .. } = prepared(&with(own)) else {
        panic!("the finetune plans beside its base model's projector");
    };
    assert!(facts.capabilities.vision);
    assert_eq!(facts.projector, ProjectorVerdict::Bound);
    let PreparedModelAssessment::Planned { facts, .. } = prepared(&with(narrow)) else {
        panic!("a projector of another width leaves the target planning text only");
    };
    assert!(!facts.capabilities.vision);
    let ProjectorVerdict::Rejected(rejection) = &facts.projector else {
        panic!("a projector of another width can't feed this decoder: {:?}", facts.projector);
    };
    assert!(
        matches!(
            rejection.reasons.as_slice(),
            [UnsupportedReason::TargetMismatch { .. }]
        ),
        "{:?}",
        rejection.reasons
    );
}

/// A text-only family given a repository's projector (Nemotron 3.5 VL repositories ship an
/// mmproj the nemotron_h family has no vision for) gets the same verdict as without it, the
/// projector `Rejected` with a mismatch: never an unreadable target.
#[test]
fn a_text_only_family_is_assessed_without_a_projector_it_cannot_bind() {
    let scratch = Scratch::new("text-only-projector");
    let target = real_header(&scratch, "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf");
    let projector = real_header(&scratch, "mmproj-F16.gguf");
    let PreparedModelAssessment::Planned { facts: alone, .. } = prepared(&package(target.clone()))
    else {
        panic!("the Nemotron target plans");
    };
    assert_eq!(alone.projector, ProjectorVerdict::None);
    let PreparedModelAssessment::Planned { facts, .. } = prepared(&ModelPackagePaths {
        projector: Some(projector),
        ..package(target)
    }) else {
        panic!("a projector the family cannot bind never refuses the target");
    };
    assert!(!facts.capabilities.vision);
    let ProjectorVerdict::Rejected(rejection) = &facts.projector else {
        panic!("Nemotron-H has no vision projector: {:?}", facts.projector);
    };
    assert_eq!(
        rejection.reasons,
        vec![UnsupportedReason::TargetMismatch {
            detail: "Nemotron-H has no vision projector".to_owned()
        }]
    );
    assert_eq!(
        ModelFacts {
            projector: ProjectorVerdict::None,
            ..facts
        },
        alone
    );
}

/// A tensor type id the engine doesn't know (a quantization newer than it) is an unsupported
/// weight type named by its id, never an unreadable file.
#[test]
fn an_unknown_tensor_type_is_an_unsupported_weight_type() {
    let scratch = Scratch::new("unknown-type");
    let target = llama().write(&scratch.file("target.gguf"));
    let mut bytes = std::fs::read(&target).unwrap();
    let name = b"blk.0.ffn_up.weight";
    let mut key = (name.len() as u64).to_le_bytes().to_vec();
    key.extend_from_slice(name);
    let at = bytes.windows(key.len()).position(|window| window == key).unwrap() + key.len();
    let rank = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let encoding = at + 4 + rank * 8;
    bytes[encoding..encoding + 4].copy_from_slice(&200_u32.to_le_bytes());
    std::fs::write(&target, &bytes).unwrap();
    let PreparedModelAssessment::Unsupported { unsupported, .. } = prepared(&package(target)) else {
        panic!("an unknown weight type doesn't run");
    };
    assert_eq!(
        unsupported.reasons,
        [UnsupportedReason::WeightTypes {
            types: vec![magnitude_artifacts::WeightTypeCount {
                ggml_type: "type 200".into(),
                tensor_count: 1,
            }]
        }]
    );
}

/// A load interprets the same way: a Nemotron package opened with a projector the family has no
/// vision for defines its text model, with no image input, and keeps the projector's refusal.
#[test]
fn a_load_serves_text_beside_a_projector_the_family_cannot_bind() {
    let scratch = Scratch::new("text-only-projector-load");
    let target = real_header(&scratch, "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf");
    let projector = real_header(&scratch, "mmproj-F16.gguf");
    // Header fixtures stand in for whole files: their payloads read as sparse zeros.
    for (path, bytes) in [(&target, 24_u64 << 30), (&projector, 1 << 30)] {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(bytes)
            .unwrap();
    }
    let package = magnitude_artifacts::Package::open_with_projector(target, projector).unwrap();
    let (_, admitted) = magnitude_engine::host::package_definition(&package)
        .unwrap_or_else(|error| panic!("the text model is defined: {error}"));
    assert!(package.projector().is_some());
    assert!(admitted.definition.vision.is_none());
    assert_eq!(
        admitted.projector_unavailable.map(|refusal| refusal.reasons),
        Some(vec![UnsupportedReason::TargetMismatch {
            detail: "Nemotron-H has no vision projector".to_owned()
        }])
    );
}
