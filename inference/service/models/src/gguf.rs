//! Bounded GGUF metadata inspection without reading tensor contents.
//!
//! The container is read by the engine's header reader
//! ([`magnitude_artifacts::gguf::inspect_header`]), which validates header bounds, tensor
//! geometry and offsets; this module only interprets catalog and inventory metadata.

use magnitude_artifacts::gguf::{Directory, Scalar, Value};
use magnitude_service_contracts::models::SpeculativeMethod;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const MAGIC: [u8; 4] = *b"GGUF";
const SUPPORTED_VERSIONS: std::ops::RangeInclusive<u32> = 2..=3;

/// Header facts of one GGUF file. `header_digest` is the lowercase hex SHA-256 of the first
/// `header_bytes` bytes, which identifies the file's content without reading tensor data.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GgufInspection {
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub quantization: Option<String>,
    pub quantization_name: Option<String>,
    pub parameter_count: Option<u64>,
    pub active_parameter_count: Option<u64>,
    pub training_context_length: Option<u32>,
    pub nextn_predict_layers: Option<u32>,
    pub tokenizer: Option<String>,
    pub base_models: Vec<GgufBaseModel>,
    pub modalities: Vec<String>,
    pub tensor_storage_bytes: u64,
    /// Bytes from the start of the file through the aligned tensor-data offset.
    pub header_bytes: u64,
    pub header_digest: String,
    pub fingerprint_material: Vec<u8>,
    pub role: GgufRole,
}

/// What a GGUF file is, read from its architecture and tensors rather than its name. A `Drafter`
/// is packaged separately: DFlash-family drafts, Gemma assistants, prediction heads sharing their
/// target's tables, and EAGLE-3 drafts (which no method serves yet).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GgufRole {
    Model {
        embedded_mtp: bool,
    },
    Projector,
    Drafter {
        method: Option<SpeculativeMethod>,
    },
}

/// One `general.base_model.N.*` lineage entry.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GgufBaseModel {
    pub name: Option<String>,
    pub organization: Option<String>,
    pub repo_url: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("failed to read GGUF: {0}")]
    Io(#[from] io::Error),
    #[error("file is not GGUF")]
    InvalidMagic,
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("GGUF header is invalid: {0}")]
    Header(magnitude_artifacts::Error),
    #[error("unknown GGUF model file type {0}")]
    UnknownFileType(u64),
}

/// Header inspection of a file that must be complete on disk: a local file shorter than the tensor
/// data its header declares is truncated, so it is refused as unreadable. Header-only previews use
/// [`inspect`].
pub fn inspect_complete(path: &Path) -> Result<GgufInspection, GgufError> {
    let directory = read_directory(path)?;
    let declared = directory
        .tensors
        .iter()
        .map(|tensor| tensor.offset.saturating_add(tensor.nbytes))
        .max()
        .unwrap_or(0)
        .saturating_add(directory.data_offset);
    let actual = std::fs::metadata(path)?.len();
    if actual < declared {
        return Err(GgufError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "GGUF file is truncated: it has {actual} bytes of the {declared} its header declares"
            ),
        )));
    }
    let header_digest = header_digest(path, directory.data_offset)?;
    interpret(&directory, header_digest)
}

pub fn inspect(path: &Path) -> Result<GgufInspection, GgufError> {
    let directory = read_directory(path)?;
    let header_digest = header_digest(path, directory.data_offset)?;
    interpret(&directory, header_digest)
}

fn read_directory(path: &Path) -> Result<Directory, GgufError> {
    let mut preamble = [0_u8; 8];
    File::open(path)?.read_exact(&mut preamble)?;
    if preamble[..4] != MAGIC {
        return Err(GgufError::InvalidMagic);
    }
    let version = u32::from_le_bytes([preamble[4], preamble[5], preamble[6], preamble[7]]);
    if !SUPPORTED_VERSIONS.contains(&version) && !SUPPORTED_VERSIONS.contains(&version.swap_bytes())
    {
        return Err(GgufError::UnsupportedVersion(version));
    }
    magnitude_artifacts::gguf::inspect_header(path).map_err(GgufError::Header)
}

fn header_digest(path: &Path, header_bytes: u64) -> Result<String, GgufError> {
    let mut header = File::open(path)?.take(header_bytes);
    let mut digest = Sha256::new();
    let copied = io::copy(&mut header, &mut digest)?;
    if copied != header_bytes {
        return Err(GgufError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "GGUF header is shorter than its tensor-data offset",
        )));
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn role(directory: &Directory, architecture: Option<&str>) -> GgufRole {
    let nextn_tensors = directory
        .tensors
        .iter()
        .filter(|tensor| is_nextn_tensor(&tensor.name))
        .count();
    let shares_target_tables = architecture.is_some_and(|architecture| {
        matches!(
            directory.value(&format!("{architecture}.nextn_shared_target_tensors")),
            Some(Value::Scalar(Scalar::Bool(true)))
        )
    });
    match architecture {
        Some("clip") => GgufRole::Projector,
        Some("dflash") => GgufRole::Drafter {
            method: Some(if directory.tensor("markov_w1.weight").is_some() {
                SpeculativeMethod::DSpark
            } else if directory.tensor("selector_hidden.weight").is_some() {
                SpeculativeMethod::DFlash2
            } else {
                SpeculativeMethod::DFlash
            }),
        },
        Some("eagle3") => GgufRole::Drafter { method: None },
        Some("gemma4-assistant") => GgufRole::Drafter {
            method: Some(SpeculativeMethod::Mtp),
        },
        _ if shares_target_tables
            || (nextn_tensors > 0 && nextn_tensors == directory.tensors.len()) =>
        {
            GgufRole::Drafter {
                method: Some(SpeculativeMethod::Mtp),
            }
        }
        _ => GgufRole::Model {
            embedded_mtp: nextn_tensors > 0,
        },
    }
}

fn is_nextn_tensor(name: &str) -> bool {
    name.strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .is_some_and(|(index, rest)| {
            index.bytes().all(|byte| byte.is_ascii_digit()) && rest.starts_with("nextn.")
        })
}

pub(crate) fn interpret(directory: &Directory, header_digest: String) -> Result<GgufInspection, GgufError> {
    let architecture = string_value(directory, "general.architecture");
    let role = role(directory, architecture.as_deref());
    let training_context_length = architecture
        .as_ref()
        .and_then(|architecture| u32_value(directory, &format!("{architecture}.context_length")))
        .or_else(|| u32_value(directory, "llama.context_length"));
    let nextn_predict_layers = architecture.as_ref().and_then(|architecture| {
        u32_value(directory, &format!("{architecture}.nextn_predict_layers"))
    });
    // The expert activation ratio cannot be applied to the whole model: embeddings, attention,
    // shared experts, and output tensors remain active. Publish no active count unless GGUF grows
    // an authoritative aggregate field.
    let active_parameter_count = u64_value(directory, "general.active_parameter_count");
    let base_model_count = u32_value(directory, "general.base_model.count").unwrap_or(0);
    let base_models = (0..base_model_count)
        .map(|index| {
            let field = |field: &str| {
                string_value(directory, &format!("general.base_model.{index}.{field}"))
            };
            GgufBaseModel {
                name: field("name"),
                organization: field("organization"),
                repo_url: field("repo_url"),
            }
        })
        .filter(|base| base.name.is_some() || base.repo_url.is_some())
        .collect();
    let modalities = if directory.metadata.iter().any(|entry| {
        entry.name.contains("vision")
            || entry.name.contains("clip")
            || entry.name.contains("projector")
    }) {
        vec!["text".to_owned(), "image".to_owned()]
    } else {
        vec!["text".to_owned()]
    };
    let file_type = directory
        .value("general.file_type")
        .and_then(Value::unsigned)
        .map(FileType::from_raw)
        .transpose()?;
    let tensor_storage_bytes = directory.tensors.iter().try_fold(0_u64, |total, tensor| {
        total.checked_add(tensor.nbytes).ok_or_else(|| {
            GgufError::Header(magnitude_artifacts::Error::Invalid(
                "tensor storage bytes overflow".to_owned(),
            ))
        })
    })?;
    let derived_parameter_count = directory.tensors.iter().try_fold(0_u64, |total, tensor| {
        tensor
            .shape
            .iter()
            .try_fold(1_u64, |elements, dimension| {
                elements.checked_mul(*dimension)
            })
            .and_then(|elements| total.checked_add(elements))
            .ok_or_else(|| {
                GgufError::Header(magnitude_artifacts::Error::Invalid(
                    "model parameter count overflows".to_owned(),
                ))
            })
    })?;
    let parameter_count = u64_value(directory, "general.parameter_count")
        .or((derived_parameter_count > 0).then_some(derived_parameter_count));

    let metadata_count = u64::try_from(directory.metadata.len()).expect("metadata count fits u64");
    let tensor_count = u64::try_from(directory.tensors.len()).expect("tensor count fits u64");
    let mut fingerprint_material = Vec::new();
    fingerprint_material.extend_from_slice(&directory.version.to_le_bytes());
    fingerprint_material.extend_from_slice(&tensor_count.to_le_bytes());
    fingerprint_material.extend_from_slice(&metadata_count.to_le_bytes());
    if let Some(value) = architecture.as_ref() {
        fingerprint_material.extend_from_slice(value.as_bytes());
    }
    if let Some(value) = string_value(directory, "tokenizer.chat_template") {
        fingerprint_material.extend_from_slice(value.as_bytes());
    }
    if let Some(value) = nextn_predict_layers {
        fingerprint_material.extend_from_slice(b"nextn_predict_layers");
        fingerprint_material.extend_from_slice(&value.to_le_bytes());
    }

    Ok(GgufInspection {
        architecture,
        name: string_value(directory, "general.name"),
        quantization: file_type.map(|file_type| file_type.name()),
        quantization_name: file_type.map(|file_type| file_type.bit_width().to_owned()),
        parameter_count,
        active_parameter_count,
        training_context_length,
        nextn_predict_layers,
        tokenizer: string_value(directory, "tokenizer.ggml.model"),
        base_models,
        modalities,
        tensor_storage_bytes,
        header_bytes: directory.data_offset,
        header_digest,
        fingerprint_material,
        role,
    })
}

/// A model file type recorded in GGUF `general.file_type`. The numbering and names are the
/// GGUF writers' (`llama_ftype`) convention; the value describes the file for inventory and
/// catalog presentation only and plays no part in execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileType {
    kind: FileTypeKind,
    guessed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FileTypeKind {
    AllF32,
    MostlyF16,
    MostlyQ4_0,
    MostlyQ4_1,
    MostlyQ8_0,
    MostlyQ5_0,
    MostlyQ5_1,
    MostlyQ2K,
    MostlyQ3KSmall,
    MostlyQ3KMedium,
    MostlyQ3KLarge,
    MostlyQ4KSmall,
    MostlyQ4KMedium,
    MostlyQ5KSmall,
    MostlyQ5KMedium,
    MostlyQ6K,
    MostlyIq2Xxs,
    MostlyIq2Xs,
    MostlyQ2KSmall,
    MostlyIq3Xs,
    MostlyIq3Xxs,
    MostlyIq1Small,
    MostlyIq4Nl,
    MostlyIq3Small,
    MostlyIq3Medium,
    MostlyIq2Small,
    MostlyIq2Medium,
    MostlyIq4Xs,
    MostlyIq1Medium,
    MostlyBf16,
    MostlyTq1_0,
    MostlyTq2_0,
    MostlyMxfp4Moe,
    MostlyNvfp4,
    MostlyQ1_0,
    MostlyQ2_0,
}

/// The flag writers set when the file type was inferred rather than declared.
const GUESSED_FILE_TYPE: u64 = 1024;

impl FileType {
    fn from_raw(raw: u64) -> Result<Self, GgufError> {
        // Only the bare flag denotes a guessed file type; the flag combined with a declared
        // type is not a value writers produce.
        let (kind, guessed) = if raw == GUESSED_FILE_TYPE {
            (FileTypeKind::AllF32, true)
        } else {
            (
                FileTypeKind::from_raw(raw).ok_or(GgufError::UnknownFileType(raw))?,
                false,
            )
        };
        Ok(Self { kind, guessed })
    }

    fn name(self) -> String {
        let name = self.kind.name();
        if self.guessed {
            format!("(guessed) {name}")
        } else {
            name.to_owned()
        }
    }

    fn bit_width(self) -> &'static str {
        if self.guessed {
            return "unknown";
        }
        self.kind.bit_width()
    }
}

impl FileTypeKind {
    fn from_raw(raw: u64) -> Option<Self> {
        Some(match raw {
            0 => Self::AllF32,
            1 => Self::MostlyF16,
            2 => Self::MostlyQ4_0,
            3 => Self::MostlyQ4_1,
            7 => Self::MostlyQ8_0,
            8 => Self::MostlyQ5_0,
            9 => Self::MostlyQ5_1,
            10 => Self::MostlyQ2K,
            11 => Self::MostlyQ3KSmall,
            12 => Self::MostlyQ3KMedium,
            13 => Self::MostlyQ3KLarge,
            14 => Self::MostlyQ4KSmall,
            15 => Self::MostlyQ4KMedium,
            16 => Self::MostlyQ5KSmall,
            17 => Self::MostlyQ5KMedium,
            18 => Self::MostlyQ6K,
            19 => Self::MostlyIq2Xxs,
            20 => Self::MostlyIq2Xs,
            21 => Self::MostlyQ2KSmall,
            22 => Self::MostlyIq3Xs,
            23 => Self::MostlyIq3Xxs,
            24 => Self::MostlyIq1Small,
            25 => Self::MostlyIq4Nl,
            26 => Self::MostlyIq3Small,
            27 => Self::MostlyIq3Medium,
            28 => Self::MostlyIq2Small,
            29 => Self::MostlyIq2Medium,
            30 => Self::MostlyIq4Xs,
            31 => Self::MostlyIq1Medium,
            32 => Self::MostlyBf16,
            36 => Self::MostlyTq1_0,
            37 => Self::MostlyTq2_0,
            38 => Self::MostlyMxfp4Moe,
            39 => Self::MostlyNvfp4,
            40 => Self::MostlyQ1_0,
            41 => Self::MostlyQ2_0,
            _ => return None,
        })
    }

    const fn name(self) -> &'static str {
        match self {
            Self::AllF32 => "all F32",
            Self::MostlyF16 => "F16",
            Self::MostlyBf16 => "BF16",
            Self::MostlyQ1_0 => "Q1_0",
            Self::MostlyQ2_0 => "Q2_0",
            Self::MostlyQ4_0 => "Q4_0",
            Self::MostlyQ4_1 => "Q4_1",
            Self::MostlyQ5_0 => "Q5_0",
            Self::MostlyQ5_1 => "Q5_1",
            Self::MostlyQ8_0 => "Q8_0",
            Self::MostlyMxfp4Moe => "MXFP4 MoE",
            Self::MostlyNvfp4 => "NVFP4",
            Self::MostlyQ2K => "Q2_K - Medium",
            Self::MostlyQ2KSmall => "Q2_K - Small",
            Self::MostlyQ3KSmall => "Q3_K - Small",
            Self::MostlyQ3KMedium => "Q3_K - Medium",
            Self::MostlyQ3KLarge => "Q3_K - Large",
            Self::MostlyQ4KSmall => "Q4_K - Small",
            Self::MostlyQ4KMedium => "Q4_K - Medium",
            Self::MostlyQ5KSmall => "Q5_K - Small",
            Self::MostlyQ5KMedium => "Q5_K - Medium",
            Self::MostlyQ6K => "Q6_K",
            Self::MostlyTq1_0 => "TQ1_0 - 1.69 bpw ternary",
            Self::MostlyTq2_0 => "TQ2_0 - 2.06 bpw ternary",
            Self::MostlyIq2Xxs => "IQ2_XXS - 2.0625 bpw",
            Self::MostlyIq2Xs => "IQ2_XS - 2.3125 bpw",
            Self::MostlyIq2Small => "IQ2_S - 2.5 bpw",
            Self::MostlyIq2Medium => "IQ2_M - 2.7 bpw",
            Self::MostlyIq3Xs => "IQ3_XS - 3.3 bpw",
            Self::MostlyIq3Xxs => "IQ3_XXS - 3.0625 bpw",
            Self::MostlyIq1Small => "IQ1_S - 1.5625 bpw",
            Self::MostlyIq1Medium => "IQ1_M - 1.75 bpw",
            Self::MostlyIq4Nl => "IQ4_NL - 4.5 bpw",
            Self::MostlyIq4Xs => "IQ4_XS - 4.25 bpw",
            Self::MostlyIq3Small => "IQ3_S - 3.4375 bpw",
            Self::MostlyIq3Medium => "IQ3_S mix - 3.66 bpw",
        }
    }

    const fn bit_width(self) -> &'static str {
        match self {
            Self::AllF32 => "32-bit",
            Self::MostlyF16 | Self::MostlyBf16 => "16-bit",
            Self::MostlyQ8_0 => "8-bit",
            Self::MostlyQ6K => "6-bit",
            Self::MostlyQ5_0 | Self::MostlyQ5_1 | Self::MostlyQ5KSmall | Self::MostlyQ5KMedium => {
                "5-bit"
            }
            Self::MostlyQ4_0
            | Self::MostlyQ4_1
            | Self::MostlyQ4KSmall
            | Self::MostlyQ4KMedium
            | Self::MostlyIq4Nl
            | Self::MostlyIq4Xs
            | Self::MostlyMxfp4Moe
            | Self::MostlyNvfp4 => "4-bit",
            Self::MostlyQ3KSmall
            | Self::MostlyQ3KMedium
            | Self::MostlyQ3KLarge
            | Self::MostlyIq3Xs
            | Self::MostlyIq3Xxs
            | Self::MostlyIq3Small
            | Self::MostlyIq3Medium => "3-bit",
            Self::MostlyQ2K
            | Self::MostlyIq2Xxs
            | Self::MostlyIq2Xs
            | Self::MostlyQ2KSmall
            | Self::MostlyIq2Small
            | Self::MostlyIq2Medium
            | Self::MostlyTq2_0
            | Self::MostlyQ2_0 => "2-bit",
            Self::MostlyIq1Small | Self::MostlyIq1Medium | Self::MostlyTq1_0 | Self::MostlyQ1_0 => {
                "1-bit"
            }
        }
    }
}

fn string_value(directory: &Directory, key: &str) -> Option<String> {
    directory
        .value(key)
        .and_then(Value::string)
        .map(str::to_owned)
}

fn u32_value(directory: &Directory, key: &str) -> Option<u32> {
    u64_value(directory, key).and_then(|value| u32::try_from(value).ok())
}

fn u64_value(directory: &Directory, key: &str) -> Option<u64> {
    match directory.value(key)? {
        Value::Scalar(Scalar::Unsigned(value)) => Some(*value),
        Value::Scalar(Scalar::Signed(value)) => u64::try_from(*value).ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_u32_entry(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_string(bytes, key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string_entry(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_string(bytes, key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(bytes, value);
    }

    /// A header-only GGUF with one Q8_0 tensor of shape [64, 2] (GGML order).
    fn header(metadata: impl FnOnce(&mut Vec<u8>) -> u64) -> Vec<u8> {
        let mut entries = Vec::new();
        let metadata_count = metadata(&mut entries);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&metadata_count.to_le_bytes());
        bytes.extend_from_slice(&entries);
        push_string(&mut bytes, "blk.0.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&64_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes
    }

    fn inspect_bytes(bytes: &[u8]) -> Result<GgufInspection, GgufError> {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("model.gguf");
        std::fs::write(&path, bytes).unwrap();
        inspect(&path)
    }

    #[test]
    fn rejects_non_gguf_without_panicking() {
        assert!(matches!(
            inspect_bytes(b"not a gguf"),
            Err(GgufError::InvalidMagic)
        ));
    }

    #[test]
    fn distinguishes_unsupported_versions() {
        let mut bytes = header(|_| 0);
        bytes[4..8].copy_from_slice(&7_u32.to_le_bytes());
        assert!(matches!(
            inspect_bytes(&bytes),
            Err(GgufError::UnsupportedVersion(7))
        ));
    }

    #[test]
    fn interprets_header_metadata_and_tensor_storage_without_payload() {
        let bytes = header(|entries| {
            push_string_entry(entries, "general.architecture", "qwen35");
            push_string_entry(entries, "general.name", "Test");
            push_u32_entry(entries, "general.file_type", 15);
            push_u32_entry(entries, "qwen35.context_length", 262_144);
            push_u32_entry(entries, "qwen35.nextn_predict_layers", 1);
            5
        });
        let inspection = inspect_bytes(&bytes).unwrap();
        assert_eq!(inspection.architecture.as_deref(), Some("qwen35"));
        assert_eq!(inspection.quantization.as_deref(), Some("Q4_K - Medium"));
        assert_eq!(inspection.quantization_name.as_deref(), Some("4-bit"));
        assert_eq!(inspection.training_context_length, Some(262_144));
        assert_eq!(inspection.nextn_predict_layers, Some(1));
        // Two rows of 64 Q8_0 elements: two 32-element blocks of 34 bytes each per row.
        assert_eq!(inspection.tensor_storage_bytes, 4 * 34);
        assert_eq!(inspection.parameter_count, Some(128));
        assert_eq!(inspection.header_bytes, u64::try_from(bytes.len()).unwrap());
        assert!(
            inspection
                .fingerprint_material
                .windows(b"nextn_predict_layers".len())
                .any(|window| window == b"nextn_predict_layers")
        );
    }

    fn role_of(metadata: &[(&str, crate::test_gguf::Meta<'_>)], tensors: &[&str]) -> GgufRole {
        inspect_bytes(&crate::test_gguf::gguf_bytes(metadata, tensors))
            .unwrap()
            .role
    }

    #[test]
    fn roles_come_from_architecture_and_tensors() {
        use crate::test_gguf::Meta;
        let arch = |name| ("general.architecture", Meta::Str(name));
        assert_eq!(
            role_of(&[arch("clip")], &["v.patch_embd.weight"]),
            GgufRole::Projector
        );
        assert_eq!(
            role_of(&[arch("dflash")], &["blk.0.attn_q.weight"]),
            GgufRole::Drafter {
                method: Some(SpeculativeMethod::DFlash)
            }
        );
        assert_eq!(
            role_of(
                &[arch("dflash")],
                &["blk.0.attn_q.weight", "markov_w1.weight"]
            ),
            GgufRole::Drafter {
                method: Some(SpeculativeMethod::DSpark)
            }
        );
        assert_eq!(
            role_of(
                &[arch("dflash")],
                &["blk.0.attn_q.weight", "selector_hidden.weight"]
            ),
            GgufRole::Drafter {
                method: Some(SpeculativeMethod::DFlash2)
            }
        );
        assert_eq!(
            role_of(&[arch("eagle3")], &["blk.0.attn_q.weight"]),
            GgufRole::Drafter { method: None }
        );
        assert_eq!(
            role_of(&[arch("gemma4-assistant")], &["blk.0.attn_q.weight"]),
            GgufRole::Drafter {
                method: Some(SpeculativeMethod::Mtp)
            }
        );
        assert_eq!(
            role_of(
                &[
                    arch("qwen4exp"),
                    ("qwen4exp.nextn_shared_target_tensors", Meta::Bool(true)),
                ],
                &["blk.4.nextn.enorm.weight", "blk.4.attn_q.weight"],
            ),
            GgufRole::Drafter {
                method: Some(SpeculativeMethod::Mtp)
            },
            "a prediction head sharing its target's tables is a drafter"
        );
        assert_eq!(
            role_of(&[arch("qwen35")], &["blk.3.nextn.eh_proj.weight"]),
            GgufRole::Drafter {
                method: Some(SpeculativeMethod::Mtp)
            },
            "a file of prediction-head tensors only is a drafter"
        );
        assert_eq!(
            role_of(
                &[arch("qwen35")],
                &[
                    "token_embd.weight",
                    "blk.0.attn_q.weight",
                    "blk.1.nextn.eh_proj.weight"
                ],
            ),
            GgufRole::Model { embedded_mtp: true }
        );
        assert_eq!(
            role_of(
                &[arch("llama")],
                &["token_embd.weight", "blk.0.nextnormal.weight"]
            ),
            GgufRole::Model {
                embedded_mtp: false
            }
        );
    }

    #[test]
    fn base_models_keep_every_lineage_field_and_the_header_digest_covers_the_header() {
        use crate::test_gguf::Meta;
        let bytes = crate::test_gguf::gguf_bytes(
            &[
                ("general.architecture", Meta::Str("llama")),
                ("general.base_model.count", Meta::U32(2)),
                ("general.base_model.0.name", Meta::Str("Qwen3.6 35B A3B")),
                ("general.base_model.0.organization", Meta::Str("Qwen")),
                (
                    "general.base_model.0.repo_url",
                    Meta::Str("https://huggingface.co/Qwen/Qwen3.6-35B-A3B"),
                ),
                ("general.base_model.1.organization", Meta::Str("Nobody")),
            ],
            &["token_embd.weight"],
        );
        let inspection = inspect_bytes(&bytes).unwrap();
        assert_eq!(
            inspection.base_models,
            vec![GgufBaseModel {
                name: Some("Qwen3.6 35B A3B".to_owned()),
                organization: Some("Qwen".to_owned()),
                repo_url: Some("https://huggingface.co/Qwen/Qwen3.6-35B-A3B".to_owned()),
            }]
        );
        let header = usize::try_from(inspection.header_bytes).unwrap();
        assert!(header < bytes.len(), "tensor data follows the header");
        assert_eq!(
            inspection.header_digest,
            format!("{:x}", Sha256::digest(&bytes[..header]))
        );
        let mut changed_data = bytes.clone();
        *changed_data.last_mut().unwrap() = 1;
        assert_eq!(
            inspect_bytes(&changed_data).unwrap().header_digest,
            inspection.header_digest,
            "tensor data is not part of the digest"
        );
    }

    #[test]
    fn file_types_follow_the_writer_convention() {
        assert!(matches!(
            FileType::from_raw(4),
            Err(GgufError::UnknownFileType(4))
        ));
        let guessed = FileType::from_raw(GUESSED_FILE_TYPE).unwrap();
        assert_eq!(guessed.name(), "(guessed) all F32");
        assert_eq!(guessed.bit_width(), "unknown");
        assert!(FileType::from_raw(GUESSED_FILE_TYPE | 15).is_err());
        let mxfp4 = FileType::from_raw(38).unwrap();
        assert_eq!(
            (mxfp4.name().as_str(), mxfp4.bit_width()),
            ("MXFP4 MoE", "4-bit")
        );
    }
}
