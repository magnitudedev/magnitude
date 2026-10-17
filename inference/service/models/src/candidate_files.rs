//! The files of a candidate (a Hugging Face repository at a commit, or a local folder) grouped
//! into versions and companions. A file's role comes from its header, never its name; names
//! only group shards and label versions for presentation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use magnitude_artifacts::gguf::{Directory, PROJECTOR_ARCHITECTURE, Value};
use magnitude_service_contracts::InventoryError;
use serde::{Deserialize, Serialize};

/// The `general.architecture` values the engine recognizes as targets and as separate drafters.
/// The service holds no list of its own; the server supplies the engine's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecognizedArchitectures {
    pub targets: BTreeSet<String>,
    pub drafters: BTreeSet<String>,
}

impl RecognizedArchitectures {
    /// A stable key for these sets, for caches of results that depend on them.
    #[must_use]
    pub fn key(&self) -> String {
        format!(
            "{}|{}",
            self.targets.iter().cloned().collect::<Vec<_>>().join(","),
            self.drafters.iter().cloned().collect::<Vec<_>>().join(",")
        )
    }

    #[must_use]
    pub fn new<'a>(
        targets: impl IntoIterator<Item = &'a str>,
        drafters: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        Self {
            targets: targets.into_iter().map(ToOwned::to_owned).collect(),
            drafters: drafters.into_iter().map(ToOwned::to_owned).collect(),
        }
    }

    /// Whether search keeps a repository whose Hub-level architecture is `architecture`. The Hub
    /// describes one file it picks, sometimes a projector, so `clip` and an absent architecture
    /// stay for the per-file check.
    #[must_use]
    pub fn admits_repository(&self, architecture: Option<&str>) -> bool {
        match architecture {
            None | Some(PROJECTOR_ARCHITECTURE) => true,
            Some(architecture) => self.targets.contains(architecture),
        }
    }

    /// Whether a GGUF declaring `architecture` is a separate drafter by its architecture alone:
    /// one the engine binds only as a drafter. An architecture the engine also runs as a target (a
    /// prediction head) is told apart by its tensors.
    #[must_use]
    pub fn is_separate_drafter(&self, architecture: &str) -> bool {
        self.drafters.contains(architecture) && !self.targets.contains(architecture)
    }

    /// Whether a Hub repository holds a drafter rather than a model: it declares an architecture
    /// the engine binds only as a drafter, or the Hub tags it for speculative decoding. Names
    /// never decide. Drafters are offered in drafter lists, never as models.
    #[must_use]
    pub fn is_drafter_repository(&self, architecture: Option<&str>, tags: &[String]) -> bool {
        let tagged = tags.iter().any(|tag| {
            matches!(
                tag.to_ascii_lowercase().as_str(),
                "speculative-decoding"
                    | "speculative-decoding-draft"
                    | "draft-model"
                    | "speculator"
            )
        });
        tagged || architecture.is_some_and(|architecture| self.is_separate_drafter(architecture))
    }
}

/// What a GGUF is for, read from its header.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateFileRole {
    Target {
        architecture: Option<String>,
    },
    Projector,
    Drafter {
        architecture: String,
    },
    /// Quantization data that is not a model, such as an importance matrix.
    Auxiliary,
}

impl CandidateFileRole {
    #[must_use]
    pub fn of(directory: &Directory, architectures: &RecognizedArchitectures) -> Self {
        if directory.value("general.type").and_then(Value::string) == Some("imatrix") {
            return Self::Auxiliary;
        }
        let architecture = directory
            .value("general.architecture")
            .and_then(Value::string)
            .map(ToOwned::to_owned);
        let tensors = crate::gguf::GgufTensors::of(directory, architecture.as_deref());
        match crate::gguf::role(architecture.as_deref(), tensors, architectures) {
            crate::gguf::GgufRole::Projector => Self::Projector,
            crate::gguf::GgufRole::Drafter => match architecture {
                Some(architecture) => Self::Drafter { architecture },
                None => Self::Target { architecture: None },
            },
            crate::gguf::GgufRole::Model { .. } => Self::Target { architecture },
        }
    }
}

/// Inspect one GGUF header held in memory (every byte through the aligned tensor-data offset).
pub fn header_directory(header: &[u8]) -> Result<Directory, InventoryError> {
    let file =
        tempfile::NamedTempFile::new().map_err(|error| InventoryError::Io(error.to_string()))?;
    std::fs::write(file.path(), header).map_err(|error| InventoryError::Io(error.to_string()))?;
    magnitude_artifacts::gguf::inspect_header(file.path())
        .map_err(|error| InventoryError::Integrity(error.to_string()))
}

/// Inspect one GGUF header held in memory for both its directory and its catalog-facing metadata
/// (the `general.file_type` quantization among them). The header is parsed once.
pub fn header_inspection(
    header: &[u8],
) -> Result<(Directory, crate::gguf::GgufInspection), InventoryError> {
    use sha2::Digest;
    let directory = header_directory(header)?;
    let digest = format!("{:x}", sha2::Sha256::digest(header));
    let header_bytes = crate::gguf::header_extent(&directory, header.len() as u64);
    let inspection = crate::gguf::interpret(&directory, digest, header_bytes)
        .map_err(|error| InventoryError::Integrity(error.to_string()))?;
    Ok((directory, inspection))
}

/// What one GGUF header says about its file, which a candidate needs to group, label and match
/// it. It is a function of the header bytes and the recognized architectures, so it is cached by
/// both.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeaderClassification {
    pub role: CandidateFileRole,
    pub base_repositories: BTreeSet<String>,
    pub quantization: Option<String>,
}

/// Classify one GGUF header held in memory.
pub fn classify_header(
    header: &[u8],
    architectures: &RecognizedArchitectures,
) -> Result<HeaderClassification, InventoryError> {
    let (directory, inspection) = header_inspection(header)?;
    Ok(HeaderClassification {
        role: CandidateFileRole::of(&directory, architectures),
        base_repositories: header_base_repositories(&directory),
        quantization: inspection.quantization,
    })
}

/// The header directory of a GGUF on disk (every shard's, for a split file's first shard).
pub fn local_directory(path: &Path) -> Result<Directory, InventoryError> {
    magnitude_artifacts::gguf::inspect_header(path)
        .map_err(|error| InventoryError::Integrity(format!("{}: {error}", path.display())))
}

/// The lowercase hex sha256 of a GGUF's header bytes on disk, through the aligned tensor-data
/// offset: what assessments and drafter bytes are matched by.
pub fn local_header_digest(path: &Path) -> Result<String, InventoryError> {
    crate::gguf::file_header_digest(path)
        .map_err(|error| InventoryError::Integrity(format!("{}: {error}", path.display())))
}

/// The `local:` digest of a model on disk, from its shards first shard first (see
/// [`crate::gguf::local_model_digest`]).
pub fn local_model_digest(shards: &[PathBuf]) -> Result<String, InventoryError> {
    crate::gguf::local_model_digest_of(shards)
        .map_err(|error| InventoryError::Integrity(error.to_string()))
}

/// The Hugging Face repositories a GGUF names as its base models: `general.base_model.N.repo_url`,
/// or else the repository its organization and name spell (`Qwen` and `Qwen3.5 4B` give
/// `qwen/qwen3.5-4b`), lowercased for lineage comparison.
#[must_use]
pub fn header_base_repositories(directory: &Directory) -> BTreeSet<String> {
    let count = directory
        .value("general.base_model.count")
        .and_then(Value::unsigned)
        .unwrap_or(0)
        .min(64);
    let field = |index: u64, name: &str| {
        directory
            .value(&format!("general.base_model.{index}.{name}"))
            .and_then(Value::string)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    (0..count)
        .filter_map(|index| {
            field(index, "repo_url")
                .and_then(hugging_face_repository_from_url)
                .or_else(|| {
                    let organization = field(index, "organization")?;
                    let name = field(index, "name")?;
                    let repository = format!("{organization}/{}", name.replace(' ', "-"));
                    (!organization.contains(['/', ' ']) && !name.contains('/'))
                        .then(|| repository.to_lowercase())
                })
        })
        .collect()
}

/// `owner/repository` from a Hugging Face model URL, lowercased for lineage comparison.
#[must_use]
pub fn hugging_face_repository_from_url(url: &str) -> Option<String> {
    let rest = url
        .trim()
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.trim().strip_prefix("http://huggingface.co/"))
        .or_else(|| url.trim().strip_prefix("huggingface.co/"))
        .or_else(|| url.trim().strip_prefix("hf.co/"))?;
    let mut parts = rest.split(['/', '?', '#']).filter(|part| !part.is_empty());
    let owner = parts.next()?;
    let repository = parts.next()?;
    Some(format!("{owner}/{repository}").to_lowercase())
}

/// One GGUF of a candidate: its path relative to the candidate root and its size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateGguf {
    pub path: PathBuf,
    pub size_bytes: u64,
}

/// One complete GGUF: a standalone file, or every shard of a split file in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufGroup {
    pub files: Vec<CandidateGguf>,
}

impl GgufGroup {
    #[must_use]
    pub fn primary(&self) -> &CandidateGguf {
        &self.files[0]
    }

    #[must_use]
    pub fn size_bytes(&self) -> u64 {
        self.files.iter().map(|file| file.size_bytes).sum()
    }
}

/// Group GGUFs into complete files: split shards (`name-00001-of-00003.gguf`) together in shard
/// order, every other file on its own. Incomplete shard sets are dropped.
#[must_use]
pub fn group_ggufs(files: &[CandidateGguf]) -> Vec<GgufGroup> {
    let mut groups = Vec::new();
    let mut shard_sets = BTreeMap::<(PathBuf, String, u32), BTreeMap<u32, CandidateGguf>>::new();
    for file in files {
        match split_shard(&file.path) {
            Some((prefix, index, total)) => {
                let parent = file
                    .path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_default();
                shard_sets
                    .entry((parent, prefix, total))
                    .or_default()
                    .insert(index, file.clone());
            }
            None => groups.push(GgufGroup {
                files: vec![file.clone()],
            }),
        }
    }
    for ((_, _, total), shards) in shard_sets {
        if (1..=total).all(|index| shards.contains_key(&index)) {
            groups.push(GgufGroup {
                files: shards.into_values().collect(),
            });
        }
    }
    groups.sort_by(|left, right| left.primary().path.cmp(&right.primary().path));
    groups
}

/// `(prefix, index, total)` of a split shard name.
#[must_use]
pub fn split_shard(path: &Path) -> Option<(String, u32, u32)> {
    let name = path.file_name()?.to_str()?;
    let stem = name
        .strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))?;
    let (left, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = left.rsplit_once('-')?;
    if index.len() != 5 || total.len() != 5 {
        return None;
    }
    let index = index.parse::<u32>().ok()?;
    let total = total.parse::<u32>().ok()?;
    (index >= 1 && index <= total && total <= 256).then(|| (prefix.to_owned(), index, total))
}

/// A short presentation label for a version: the quantization written in its name the way GGUF
/// writers spell it (`Q4_K_M`, `IQ2_M`, `UD-Q4_K_XL`, `BF16`), else its file stem. Labels never
/// decide compatibility.
#[must_use]
/// A version's label: the quantization its file or folder name spells (keeping a qualifier such as
/// `UD-`), else its header's quantization, else its file name.
pub fn version_label(path: &Path, header_quantization: Option<&str>) -> String {
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            let name = name
                .strip_suffix(".gguf")
                .or_else(|| name.strip_suffix(".GGUF"))
                .unwrap_or(name);
            match name.rsplit_once("-of-") {
                Some((left, _)) if split_shard(path).is_some() => left
                    .rsplit_once('-')
                    .map_or(left, |(prefix, _)| prefix)
                    .to_owned(),
                _ => name.to_owned(),
            }
        })
        .unwrap_or_default();
    let directory = path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    quantization_label(&stem)
        .or_else(|| quantization_label(directory))
        .or_else(|| header_label(header_quantization))
        .unwrap_or(stem)
}

fn header_label(header_quantization: Option<&str>) -> Option<String> {
    header_quantization
        .and_then(crate::gguf::quantization_label)
        .map(str::to_owned)
}

fn quantization_label(text: &str) -> Option<String> {
    let tokens = text.split(['-', '.']).collect::<Vec<_>>();
    tokens.iter().enumerate().rev().find_map(|(index, token)| {
        let upper = token.to_ascii_uppercase();
        let quantization = is_quantization_token(&upper);
        quantization.then(|| {
            match index
                .checked_sub(1)
                .and_then(|previous| tokens.get(previous))
            {
                Some(previous) if previous.eq_ignore_ascii_case("UD") => format!("UD-{upper}"),
                _ => upper,
            }
        })
    })
}

fn is_quantization_token(token: &str) -> bool {
    const FLOATS: &[&str] = &["F32", "F16", "BF16", "FP16", "FP32", "MXFP4", "NVFP4"];
    if FLOATS.contains(&token) || token.starts_with("MXFP4_") || token.starts_with("NVFP4_") {
        return true;
    }
    let rest = token
        .strip_prefix("IQ")
        .or_else(|| token.strip_prefix("TQ"))
        .or_else(|| token.strip_prefix('Q'));
    let Some(rest) = rest else {
        return false;
    };
    // One digit of bits: `Q4_0`, `IQ2_XXS`; `q40` is no quantization name.
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    digits == 1
        && rest[digits..]
            .split('_')
            .all(|part| part.bytes().all(|byte| byte.is_ascii_alphanumeric()))
        && (rest.len() == digits || rest.as_bytes()[digits] == b'_')
}

/// A display name from a repository or folder name: separators become spaces and a trailing
/// `GGUF` is dropped.
#[must_use]
pub fn display_name(name: &str) -> String {
    let name = name.rsplit('/').next().unwrap_or(name);
    let trimmed = ["-GGUF", "-gguf", "_GGUF", "_gguf", "-Gguf"]
        .iter()
        .find_map(|suffix| name.strip_suffix(suffix))
        .unwrap_or(name);
    trimmed.replace(['-', '_'], " ")
}

/// How a model on this computer is named: its display name and the quantization label of the
/// variant, the one its file names (`Q4_K_M`, `UD-Q4_K_XL`) or else its header's. The name is the repository's when
/// the file has a Hub identity (`Qwen3.5 4B` for `lmstudio-community/Qwen3.5-4B-GGUF`), else the
/// GGUF's `general.name`, else the file name, each without its quantization and with separators as
/// spaces. Names are presentation only.
#[must_use]
pub fn model_presentation(
    primary: &Path,
    repository: Option<&str>,
    header_name: Option<&str>,
    header_quantization: Option<&str>,
) -> (String, Option<String>) {
    let stem = primary
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            name.strip_suffix(".gguf")
                .or_else(|| name.strip_suffix(".GGUF"))
                .unwrap_or(name)
        })
        .unwrap_or_default();
    let label = quantization_label(stem).or_else(|| header_label(header_quantization));
    let without_label = |text: &str| {
        let cleaned = match quantization_label(text) {
            Some(label) => {
                let lower = text.to_ascii_lowercase();
                match lower.rfind(&label.to_ascii_lowercase()) {
                    Some(index) => text[..index].to_owned(),
                    None => text.to_owned(),
                }
            }
            None => text.to_owned(),
        };
        display_name(cleaned.trim_end_matches(['-', '_', '.', ' ']))
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let name = repository
        .map(display_name)
        .or_else(|| header_name.map(without_label))
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| without_label(stem));
    let name = if name.trim().is_empty() {
        stem.to_owned()
    } else {
        name
    };
    (name, label)
}

/// Whether a version's quantization (the GGUF's `general.file_type` name) is in the Q4_K class,
/// which the repository step recommends among fitting versions.
#[must_use]
pub fn is_q4_k_class(quantization: &str) -> bool {
    quantization.to_ascii_uppercase().starts_with("Q4_K")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gguf(path: &str, size_bytes: u64) -> CandidateGguf {
        CandidateGguf {
            path: PathBuf::from(path),
            size_bytes,
        }
    }

    #[test]
    fn shards_group_in_order_and_incomplete_sets_drop() {
        let groups = group_ggufs(&[
            gguf("Q8_0/model-Q8_0-00002-of-00002.gguf", 2),
            gguf("Q8_0/model-Q8_0-00001-of-00002.gguf", 1),
            gguf("model-Q4_K_M.gguf", 4),
            gguf("broken-00001-of-00003.gguf", 1),
            gguf("broken-00003-of-00003.gguf", 1),
        ]);
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups[0]
                .files
                .iter()
                .map(|file| file.path.to_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "Q8_0/model-Q8_0-00001-of-00002.gguf",
                "Q8_0/model-Q8_0-00002-of-00002.gguf"
            ]
        );
        assert_eq!(groups[0].size_bytes(), 3);
        assert_eq!(groups[1].primary().path, Path::new("model-Q4_K_M.gguf"));
    }

    #[test]
    fn labels_follow_writer_conventions_and_fall_back_to_the_stem() {
        for (path, label) in [
            (
                "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
                "Q4_K_P",
            ),
            ("model-IQ2_M.gguf", "IQ2_M"),
            ("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
            ("Q8_0/model-Q8_0-00001-of-00002.gguf", "Q8_0"),
            ("BF16/model-00001-of-00002.gguf", "BF16"),
            ("huihui-model.Q4_K.gguf", "Q4_K"),
            ("model-bf16.gguf", "BF16"),
            ("weights.gguf", "weights"),
        ] {
            assert_eq!(version_label(Path::new(path), None), label, "{path}");
        }
        assert_eq!(
            version_label(Path::new("new-q40.gguf"), Some("Q4_0")),
            "Q4_0",
            "a name that isn't a quantization label gives way to the header's"
        );
        assert_eq!(
            version_label(Path::new("weights.gguf"), Some("Q4_K - Medium")),
            "Q4_K_M"
        );
        assert_eq!(
            version_label(Path::new("model-UD-Q4_K_XL.gguf"), Some("Q4_K - Medium")),
            "UD-Q4_K_XL"
        );
    }

    #[test]
    fn repository_urls_and_display_names_normalize() {
        assert_eq!(
            hugging_face_repository_from_url("https://huggingface.co/Qwen/Qwen3.6-35B-A3B"),
            Some("qwen/qwen3.6-35b-a3b".to_owned())
        );
        assert_eq!(
            hugging_face_repository_from_url("https://example.com/a/b"),
            None
        );
        assert_eq!(
            display_name("HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-GGUF"),
            "Qwen3.6 35B A3B Uncensored HauhauCS Aggressive"
        );
    }

    #[test]
    fn drafter_repositories_are_recognized_by_the_engines_drafter_architectures_or_hub_tags() {
        let architectures = RecognizedArchitectures::new(
            ["qwen35", "qwen4exp"],
            ["dflash", "gemma4-assistant", "qwen4exp"],
        );
        let tags = |tags: &[&str]| tags.iter().map(|tag| (*tag).to_owned()).collect::<Vec<_>>();
        for architecture in ["dflash", "gemma4-assistant"] {
            assert!(
                architectures.is_drafter_repository(Some(architecture), &[]),
                "{architecture}"
            );
        }
        assert!(
            architectures
                .is_drafter_repository(Some("qwen3"), &tags(&["gguf", "speculative-decoding"]))
        );
        for architecture in [
            "eagle3",
            "qwen35_mtp",
            "llama-assistant",
            "qwen3-dflash-like",
        ] {
            assert!(
                !architectures.is_drafter_repository(Some(architecture), &[]),
                "names never decide: {architecture}"
            );
        }
        assert!(!architectures.is_drafter_repository(Some("qwen35"), &tags(&["gguf"])));
        assert!(!architectures.is_drafter_repository(Some("qwen4exp"), &[]));
        assert!(!architectures.is_drafter_repository(Some("deepseek4"), &[]));
        assert!(!architectures.is_drafter_repository(None, &[]));
    }

    #[test]
    fn a_qwen4exp_full_target_declaring_prediction_layers_stays_a_target_and_its_head_file_is_a_drafter()
     {
        use crate::test_gguf::{Meta, gguf_bytes};
        let architectures = RecognizedArchitectures::new(["qwen4exp"], ["dflash", "qwen4exp"]);
        let full = [
            ("general.architecture", Meta::Str("qwen4exp")),
            ("qwen4exp.block_count", Meta::U32(3)),
            ("qwen4exp.nextn_predict_layers", Meta::U32(1)),
        ];
        let full_tensors = [
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
            "blk.0.attn_q.weight",
            "blk.1.attn_q.weight",
            "blk.2.attn_q.weight",
            "blk.2.nextn.eh_proj.weight",
            "blk.2.nextn.enorm.weight",
        ];
        assert_eq!(
            classify_header(&gguf_bytes(&full, &full_tensors), &architectures)
                .unwrap()
                .role,
            CandidateFileRole::Target {
                architecture: Some("qwen4exp".to_owned())
            }
        );
        let head = [
            ("general.architecture", Meta::Str("qwen4exp")),
            ("qwen4exp.block_count", Meta::U32(3)),
            ("qwen4exp.nextn_predict_layers", Meta::U32(1)),
            ("qwen4exp.nextn_shared_target_tensors", Meta::Bool(true)),
        ];
        let head_tensors = [
            "blk.2.attn_q.weight",
            "blk.2.ffn_up_exps.weight",
            "blk.2.nextn.eh_proj.weight",
            "blk.2.nextn.enorm.weight",
            "blk.2.nextn.hnorm.weight",
        ];
        assert_eq!(
            classify_header(&gguf_bytes(&head, &head_tensors), &architectures)
                .unwrap()
                .role,
            CandidateFileRole::Drafter {
                architecture: "qwen4exp".to_owned()
            }
        );
    }

    #[test]
    fn repository_prefilter_keeps_projector_and_unknown_architectures() {
        let architectures = RecognizedArchitectures::new(["qwen35moe"], ["dflash"]);
        assert!(architectures.admits_repository(Some("qwen35moe")));
        assert!(architectures.admits_repository(Some("clip")));
        assert!(architectures.admits_repository(None));
        assert!(!architectures.admits_repository(Some("deepseek4")));
        assert!(!architectures.admits_repository(Some("dflash")));
    }

    #[test]
    fn an_importance_matrix_is_auxiliary_not_a_version() {
        use magnitude_artifacts::gguf::{ByteOrder, Metadata, Scalar};
        let directory = |entries: &[(&str, &str)]| Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata: entries
                .iter()
                .map(|(name, value)| Metadata {
                    name: (*name).to_owned(),
                    value: Value::Scalar(Scalar::String((*value).to_owned())),
                })
                .collect(),
            tensors: Vec::new(),
        };
        let architectures = RecognizedArchitectures::new(["qwen35"], ["dflash"]);
        assert_eq!(
            CandidateFileRole::of(&directory(&[("general.type", "imatrix")]), &architectures),
            CandidateFileRole::Auxiliary
        );
        assert_eq!(
            CandidateFileRole::of(
                &directory(&[
                    ("general.type", "model"),
                    ("general.architecture", "qwen35")
                ]),
                &architectures
            ),
            CandidateFileRole::Target {
                architecture: Some("qwen35".to_owned())
            }
        );
    }

    /// bartowski's Nemotron-H `mtp-*.gguf` keeps the model's architecture and metadata (its
    /// `nextn_predict_layers` too) but only the trailing prediction block (`blk.52`, a full layer
    /// beside its `nextn.*` combine and head norm), the embedding and the head. The full model
    /// declares the same metadata and holds every block.
    #[test]
    fn a_nemotron_h_mtp_file_is_a_drafter_and_its_model_a_target() {
        use crate::test_gguf::{Meta, gguf_bytes};
        let architectures = RecognizedArchitectures::new(["nemotron_h_moe"], ["dflash"]);
        let metadata = [
            ("general.architecture", Meta::Str("nemotron_h_moe")),
            ("general.type", Meta::Str("model")),
            ("nemotron_h_moe.block_count", Meta::U32(53)),
            ("nemotron_h_moe.nextn_predict_layers", Meta::U32(1)),
        ];
        let head_block = [
            "blk.52.attn_k.weight",
            "blk.52.attn_norm.weight",
            "blk.52.attn_output.weight",
            "blk.52.attn_q.weight",
            "blk.52.attn_v.weight",
            "blk.52.exp_probs_b.bias",
            "blk.52.ffn_down_exps.weight",
            "blk.52.ffn_down_shexp.weight",
            "blk.52.ffn_gate_inp.weight",
            "blk.52.ffn_up_exps.weight",
            "blk.52.ffn_up_shexp.weight",
            "blk.52.nextn.eh_proj.weight",
            "blk.52.nextn.enorm.weight",
            "blk.52.nextn.hnorm.weight",
            "blk.52.nextn.shared_head_norm.weight",
            "blk.52.post_attention_norm.weight",
        ];
        let mtp = ["output.weight", "output_norm.weight", "token_embd.weight"]
            .into_iter()
            .chain(head_block)
            .collect::<Vec<_>>();
        assert_eq!(
            classify_header(&gguf_bytes(&metadata, &mtp), &architectures)
                .unwrap()
                .role,
            CandidateFileRole::Drafter {
                architecture: "nemotron_h_moe".to_owned()
            }
        );
        let model = [
            "output.weight",
            "output_norm.weight",
            "token_embd.weight",
            "blk.0.ssm_in.weight",
            "blk.1.ffn_up_exps.weight",
            "blk.5.attn_q.weight",
        ]
        .into_iter()
        .chain(head_block)
        .collect::<Vec<_>>();
        assert_eq!(
            classify_header(&gguf_bytes(&metadata, &model), &architectures)
                .unwrap()
                .role,
            CandidateFileRole::Target {
                architecture: Some("nemotron_h_moe".to_owned())
            }
        );
    }

    #[test]
    fn base_repositories_come_from_repo_urls_or_else_organization_and_name() {
        use magnitude_artifacts::gguf::{ByteOrder, Metadata, Scalar};
        let metadata = |name: &str, value: Scalar| Metadata {
            name: name.to_owned(),
            value: Value::Scalar(value),
        };
        let directory = Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata: vec![
                metadata("general.base_model.count", Scalar::Unsigned(2)),
                metadata(
                    "general.base_model.0.repo_url",
                    Scalar::String("https://huggingface.co/Qwen/Qwen3.6-35B-A3B".to_owned()),
                ),
                metadata(
                    "general.base_model.1.organization",
                    Scalar::String("Qwen".to_owned()),
                ),
                metadata(
                    "general.base_model.1.name",
                    Scalar::String("Qwen3.5 4B".to_owned()),
                ),
            ],
            tensors: Vec::new(),
        };
        assert_eq!(
            header_base_repositories(&directory),
            BTreeSet::from([
                "qwen/qwen3.6-35b-a3b".to_owned(),
                "qwen/qwen3.5-4b".to_owned()
            ])
        );
    }

    #[test]
    fn models_are_named_by_repository_then_header_then_file_without_their_quantization() {
        assert_eq!(
            model_presentation(
                Path::new("MiniCPM5-1B-Q3_K_M.gguf"),
                Some("NANI-Nithin/MiniCPM5-1B-GGUF"),
                Some("MiniCPM5-1B"),
                Some("Q3_K - Medium")
            ),
            ("MiniCPM5 1B".to_owned(), Some("Q3_K_M".to_owned()))
        );
        assert_eq!(
            model_presentation(
                Path::new("blobs/sha256-abc"),
                None,
                Some("Qwen3.5-4B-Q4_K_M"),
                None
            ),
            ("Qwen3.5 4B".to_owned(), None)
        );
        assert_eq!(
            model_presentation(
                Path::new("blobs/sha256-abc"),
                None,
                Some("Qwen3.5-4B-Q4_K_M"),
                Some("Q4_K - Medium")
            ),
            ("Qwen3.5 4B".to_owned(), Some("Q4_K_M".to_owned()))
        );
        assert_eq!(
            model_presentation(Path::new("models/new-q40.gguf"), None, None, Some("Q4_0")),
            ("new q40".to_owned(), Some("Q4_0".to_owned()))
        );
        assert_eq!(
            model_presentation(
                Path::new("models/MiniCPM5-1B-q4_k_s.gguf"),
                None,
                None,
                None
            ),
            ("MiniCPM5 1B".to_owned(), Some("Q4_K_S".to_owned()))
        );
        assert_eq!(
            model_presentation(
                Path::new("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
                None,
                None,
                None
            ),
            ("Qwen3.6 35B A3B".to_owned(), Some("UD-Q4_K_XL".to_owned()))
        );
    }
}
