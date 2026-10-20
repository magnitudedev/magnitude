//! Synthetic GGUF files for tests: real headers with small F32 tensors.

use std::path::Path;

pub(crate) enum Meta<'a> {
    Str(&'a str),
    U32(u32),
    Bool(bool),
    Strs(&'a [&'a str]),
}

const ALIGNMENT: usize = 32;
const TENSOR_ELEMENTS: u64 = 32;

fn push_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

pub(crate) fn gguf_bytes(metadata: &[(&str, Meta<'_>)], tensors: &[&str]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        push_string(&mut bytes, key);
        match value {
            Meta::Str(value) => {
                bytes.extend_from_slice(&8_u32.to_le_bytes());
                push_string(&mut bytes, value);
            }
            Meta::U32(value) => {
                bytes.extend_from_slice(&4_u32.to_le_bytes());
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Meta::Bool(value) => {
                bytes.extend_from_slice(&7_u32.to_le_bytes());
                bytes.push(u8::from(*value));
            }
            Meta::Strs(values) => {
                bytes.extend_from_slice(&9_u32.to_le_bytes());
                bytes.extend_from_slice(&8_u32.to_le_bytes());
                bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in *values {
                    push_string(&mut bytes, value);
                }
            }
        }
    }
    let tensor_bytes = TENSOR_ELEMENTS * 4;
    for (index, name) in tensors.iter().enumerate() {
        push_string(&mut bytes, name);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&TENSOR_ELEMENTS.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&(index as u64 * tensor_bytes).to_le_bytes());
    }
    bytes.resize(bytes.len().next_multiple_of(ALIGNMENT), 0);
    bytes.resize(bytes.len() + tensors.len() * tensor_bytes as usize, 0);
    bytes
}

pub(crate) fn write_gguf(path: &Path, metadata: &[(&str, Meta<'_>)], tensors: &[&str]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, gguf_bytes(metadata, tensors)).unwrap();
}

/// A model whose header is distinguished by `name`.
pub(crate) fn write_model(path: &Path, name: &str) {
    write_gguf(
        path,
        &[
            ("general.architecture", Meta::Str("llama")),
            ("general.name", Meta::Str(name)),
        ],
        &["token_embd.weight", "blk.0.attn_q.weight"],
    );
}

/// A model with built-in prediction heads, whatever its name says.
pub(crate) fn write_mtp_model(path: &Path, name: &str) {
    write_gguf(
        path,
        &[
            ("general.architecture", Meta::Str("qwen35")),
            ("general.name", Meta::Str(name)),
            ("qwen35.nextn_predict_layers", Meta::U32(1)),
        ],
        &[
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.1.nextn.eh_proj.weight",
        ],
    );
}

pub(crate) fn write_dflash_drafter(path: &Path, name: &str) {
    write_gguf(
        path,
        &[
            ("general.architecture", Meta::Str("dflash")),
            ("general.name", Meta::Str(name)),
        ],
        &["blk.0.attn_q.weight"],
    );
}

pub(crate) fn write_projector(path: &Path, name: &str) {
    write_gguf(
        path,
        &[
            ("general.architecture", Meta::Str("clip")),
            ("general.name", Meta::Str(name)),
        ],
        &["v.patch_embd.weight"],
    );
}

/// Architecture sets shaped like the engine's: the service never holds its own, so tests supply
/// these the way the server supplies the engine's.
pub(crate) fn engine_architectures() -> crate::candidate_files::RecognizedArchitectures {
    crate::candidate_files::RecognizedArchitectures::new(
        ["llama", "qwen35", "qwen35moe", "qwen4exp", "nemotron_h_moe", "gemma4"],
        ["dflash", "gemma4-assistant", "qwen4exp"],
    )
}

/// A real GGUF header from the engine's assessment fixtures, decompressed.
pub(crate) fn fixture_header(name: &str) -> Vec<u8> {
    use std::io::Read;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../engine/fixtures/assessment/headers")
        .join(format!("{name}.gguf.gz"));
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(&path).expect("fixture"))
        .read_to_end(&mut bytes)
        .expect("gzipped fixture");
    bytes
}
