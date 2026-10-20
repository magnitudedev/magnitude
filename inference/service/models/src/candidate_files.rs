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

    /// Whether a GGUF header declares a drafter: an architecture the engine binds only as a
    /// drafter, or an architecture the engine doesn't run as a target whose `general.tags` mark
    /// it for speculative decoding (a drafter architecture the engine doesn't know yet, such as
    /// a community DFlash conversion). An architecture the engine runs as a target is told apart
    /// by its tensors.
    #[must_use]
    pub fn declares_drafter(&self, architecture: &str, tags: &[String]) -> bool {
        self.is_separate_drafter(architecture)
            || (!self.targets.contains(architecture) && speculative_decoding_tags(tags))
    }

    /// Whether a Hub repository holds a drafter rather than a model: it declares an architecture
    /// the engine binds only as a drafter, or the Hub tags it for speculative decoding. Names
    /// never decide. Drafters are offered in drafter lists, never as models.
    #[must_use]
    pub fn is_drafter_repository(&self, architecture: Option<&str>, tags: &[String]) -> bool {
        speculative_decoding_tags(tags)
            || architecture.is_some_and(|architecture| self.is_separate_drafter(architecture))
    }
}

/// Whether tags (a Hub repository's, or a GGUF header's `general.tags`) mark a drafter for
/// speculative decoding.
#[must_use]
pub fn speculative_decoding_tags(tags: &[String]) -> bool {
    tags.iter().any(|tag| {
        matches!(
            tag.to_ascii_lowercase().as_str(),
            "speculative-decoding" | "speculative-decoding-draft" | "draft-model" | "speculator"
        )
    })
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
    /// A file that is not a model: quantization data such as an importance matrix, or a GGUF that
    /// holds no weights.
    Auxiliary,
}

impl CandidateFileRole {
    #[must_use]
    pub fn of(directory: &Directory, architectures: &RecognizedArchitectures) -> Self {
        if directory.value("general.type").and_then(Value::string) == Some("imatrix") {
            return Self::Auxiliary;
        }
        // A split's first shard may hold only metadata; any other file without tensors holds no
        // weights.
        let split = directory
            .value("split.count")
            .and_then(Value::unsigned)
            .is_some_and(|count| count > 1);
        if directory.tensors.is_empty() && !split {
            return Self::Auxiliary;
        }
        let architecture = directory
            .value("general.architecture")
            .and_then(Value::string)
            .map(ToOwned::to_owned);
        let tensors = crate::gguf::GgufTensors::of(directory, architecture.as_deref());
        let tags = crate::gguf::header_tags(directory);
        match crate::gguf::role(architecture.as_deref(), &tags, tensors, architectures) {
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
    crate::gguf::known_tensors(magnitude_artifacts::gguf::read_header(header))
        .and_then(|directory| directory.require_execution_byte_order().map(|()| directory))
        .map_err(|error| InventoryError::Integrity(error.to_string()))
}

/// Inspect one GGUF header held in memory for both its directory and its catalog-facing metadata
/// (the `general.file_type` quantization among them). The header is parsed once.
pub fn header_inspection(
    header: &[u8],
) -> Result<(Directory, crate::gguf::GgufInspection), InventoryError> {
    use sha2::Digest;
    let (directory, unknown_types) =
        crate::gguf::known_tensors_and_unknown_types(magnitude_artifacts::gguf::read_header(header))
            .and_then(|(directory, unknown)| {
                directory.require_execution_byte_order().map(|()| (directory, unknown))
            })
            .map_err(|error| InventoryError::Integrity(error.to_string()))?;
    let digest = format!("{:x}", sha2::Sha256::digest(header));
    let header_bytes = crate::gguf::header_extent(&directory, header.len() as u64);
    let inspection = crate::gguf::interpret(&directory, &unknown_types, digest, header_bytes)
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
    pub declared_bytes: u64,
    pub embedded_mtp: bool,
}

impl HeaderClassification {
    /// This header's classification for a file of `size_bytes`. A file shorter than the tensor
    /// data its header declares, such as a shard set's skeleton whose directory describes weights
    /// stored in other files, holds no weights: it is auxiliary, never a version, drafter or
    /// projector.
    #[must_use]
    pub fn for_file(mut self, size_bytes: u64) -> Self {
        if size_bytes < self.declared_bytes {
            self.role = CandidateFileRole::Auxiliary;
        }
        self
    }
}

/// Classify one GGUF header held in memory.
pub fn classify_header(
    header: &[u8],
    architectures: &RecognizedArchitectures,
) -> Result<HeaderClassification, InventoryError> {
    let (directory, inspection) = header_inspection(header)?;
    let architecture = directory
        .value("general.architecture")
        .and_then(Value::string);
    Ok(HeaderClassification {
        embedded_mtp: matches!(
            crate::gguf::GgufTensors::of(&directory, architecture),
            crate::gguf::GgufTensors::Model { embedded_mtp: true }
        ),
        role: CandidateFileRole::of(&directory, architectures),
        base_repositories: header_base_repositories(&directory),
        quantization: inspection.quantization,
        declared_bytes: crate::gguf::declared_length(&directory),
    })
}

/// The header directory of a GGUF on disk (every shard's, for a split file's first shard).
pub fn local_directory(path: &Path) -> Result<Directory, InventoryError> {
    crate::gguf::known_tensors(magnitude_artifacts::gguf::inspect_header(path))
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
            .map(ToOwned::to_owned)
    };
    base_model_repositories(
        &(0..count)
            .map(|index| crate::gguf::GgufBaseModel {
                name: field(index, "name"),
                organization: field(index, "organization"),
                repo_url: field(index, "repo_url"),
            })
            .collect::<Vec<_>>(),
    )
}

/// The repositories `general.base_model.N.*` entries name, lowercased: each one's repository URL,
/// else its organization and name.
#[must_use]
pub fn base_model_repositories(base_models: &[crate::gguf::GgufBaseModel]) -> BTreeSet<String> {
    let field = |value: &Option<String>| {
        value
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    };
    base_models
        .iter()
        .filter_map(|base| {
            field(&base.repo_url)
                .as_deref()
                .and_then(hugging_face_repository_from_url)
                .or_else(|| {
                    let organization = field(&base.organization)?;
                    let name = field(&base.name)?;
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
    let url = url.trim();
    let url = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let url = url.strip_prefix("www.").unwrap_or(url);
    let rest = url
        .strip_prefix("huggingface.co/")
        .or_else(|| url.strip_prefix("hf.co/"))?;
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

/// A version's presentation label: its header's quantization, spelled by the file or folder name
/// when that names the same weight type more precisely the way GGUF writers do (`UD-Q4_K_XL`,
/// `Q4_K_P`); with nothing in the header, the quantization the name spells, else its file stem.
/// Labels never decide compatibility.
#[must_use]
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
    let named = quantization_label(&stem).or_else(|| quantization_label(directory));
    match (header_label(header_quantization), named) {
        // A name that spells the header's weight type more precisely (`UD-Q4_K_XL`, `Q4_K_P` for a
        // Q4_K_M header) keeps its writer's label; any other name is ignored, since files are
        // renamed freely.
        (Some(header), Some(named)) if weight_type(&named) == weight_type(&header) => named,
        (Some(header), _) => header,
        (None, Some(named)) => named,
        (None, None) => stem,
    }
}

/// The weight type a quantization label names, without its size or mix suffix: `Q4_K` for
/// `UD-Q4_K_XL`, `Q8_0` for `Q8_0`, `IQ2` for `IQ2_M`, `MXFP4` for `MXFP4_MOE`.
fn weight_type(label: &str) -> &str {
    let label = label.strip_prefix("UD-").unwrap_or(label);
    if label.starts_with('Q') {
        if let Some(index) = label.find("_K") {
            return &label[..index + 2];
        }
        return label;
    }
    label.split('_').next().unwrap_or(label)
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
/// GGUF's `general.name`, else the file name, else the one base model the header declares, the
/// first that identifies anything (`identifies`), each without its quantization and with
/// separators as spaces. Names are presentation only.
#[must_use]
pub fn model_presentation(
    primary: &Path,
    repository: Option<&str>,
    header_name: Option<&str>,
    header_base_models: &[crate::gguf::GgufBaseModel],
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
    // Header first, as for a candidate version (`version_label`): files are renamed freely.
    let label = match header_quantization {
        Some(quantization) if header_label(Some(quantization)).is_some() => {
            Some(version_label(primary, Some(quantization)))
        }
        _ => quantization_label(stem),
    };
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
    // A merge declares several bases, and none of them alone names it.
    let base_model = match header_base_models {
        [base] => base.name.as_deref(),
        _ => None,
    };
    let name = repository
        .map(display_name)
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| {
            [header_name, Some(stem), base_model]
                .into_iter()
                .flatten()
                .map(without_label)
                .find(|name| identifies(name))
                .unwrap_or_else(|| without_label(stem))
        });
    let name = if name.trim().is_empty() {
        stem.to_owned()
    } else {
        name
    };
    (name, label)
}

/// Whether a name tells a model apart, judged by its form alone and never against a list of names:
/// it has at least three letters or digits, which a converter's placeholder (`Hf`, from a source
/// directory named `hf`) lacks, and is not only a digest (a source commit, a content address).
fn identifies(name: &str) -> bool {
    name.chars().filter(|c| c.is_alphanumeric()).count() >= 3 && !is_digest(name)
}

/// Whether a name is only a hexadecimal digest: a commit hash (`2639448d…`), or a content digest
/// after its algorithm's name (`sha256 4cd5…` for an Ollama blob, once separators are spaces).
fn is_digest(name: &str) -> bool {
    let is_hex = |text: &str, lengths: std::ops::RangeInclusive<usize>| {
        lengths.contains(&text.len())
            && text.bytes().all(|byte| byte.is_ascii_hexdigit())
            && text.bytes().any(|byte| byte.is_ascii_digit())
    };
    let name = name.trim();
    is_hex(name, 7..=64)
        || name
            .split_once([' ', '-', ':'])
            .is_some_and(|(algorithm, digest)| {
                algorithm.bytes().all(|byte| byte.is_ascii_alphanumeric())
                    && is_hex(digest, 32..=128)
            })
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
        assert_eq!(
            version_label(Path::new("Model-Q4_K_P.gguf"), Some("Q4_K - Medium")),
            "Q4_K_P"
        );
        assert_eq!(
            version_label(Path::new("renamed-q8.gguf"), Some("Q4_K - Medium")),
            "Q4_K_M",
            "a renamed file is labelled by its header, not its name"
        );
        assert_eq!(
            version_label(Path::new("Q8_0/renamed.gguf"), Some("IQ4_XS - 4.25 bpw")),
            "IQ4_XS",
            "a folder name that contradicts the header gives way to it"
        );
        assert_eq!(
            version_label(Path::new("my-model-final.gguf"), Some("Q6_K")),
            "Q6_K",
            "a name with no quantization takes the header's, never the whole name"
        );
        assert_eq!(
            version_label(Path::new("qwen35-9b-instruct-mxfp4.gguf"), Some("MXFP4")),
            "MXFP4",
            "weights an undeclared file type names label it as the header's do"
        );
        assert_eq!(
            model_presentation(
                Path::new("qwen35-9b-instruct-mxfp4.gguf"),
                Some("FreedomAISVR/Qwen3.5-9B-Instruct-MXFP4-GGUF"),
                None,
                Some("MXFP4"),
            )
            .1,
            model_presentation(
                Path::new("qwen35-9b-instruct-mxfp4.gguf"),
                Some("FreedomAISVR/Qwen3.5-9B-Instruct-MXFP4-GGUF"),
                None,
                None,
            )
            .1,
            "an installed model is labelled as its file arriving is"
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
        let model = crate::test_gguf::gguf_bytes(
            &[
                ("general.type", crate::test_gguf::Meta::Str("model")),
                (
                    "general.architecture",
                    crate::test_gguf::Meta::Str("qwen35"),
                ),
            ],
            &["token_embd.weight"],
        );
        assert_eq!(
            classify_header(&model, &architectures).unwrap().role,
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

    fn fixture_header(name: &str) -> Vec<u8> {
        use std::io::Read;
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../engine/fixtures/assessment/headers")
            .join(format!("{name}.gz"));
        let mut header = Vec::new();
        flate2::read::GzDecoder::new(std::fs::File::open(path).expect("fixture header"))
            .read_to_end(&mut header)
            .expect("gzipped header");
        header
    }

    /// Arki05's `skeleton.gguf` (the real header) is a whole Qwen3.5 4B header: 426 tensors whose
    /// directory declares 8.41 GB of data after its 3 MB header, in a file of 10,968,000 bytes. It
    /// holds none of the weights it describes, so at its real size it is auxiliary, never a
    /// version; the same header on a file holding its data is a target.
    #[test]
    fn a_file_shorter_than_its_declared_tensor_data_holds_no_weights() {
        let architectures = RecognizedArchitectures::new(["qwen35"], ["dflash"]);
        let classification =
            classify_header(&fixture_header("skeleton.gguf"), &architectures).unwrap();
        assert_eq!(classification.declared_bytes, 3_017_952 + 8_413_425_664);
        let target = CandidateFileRole::Target {
            architecture: Some("qwen35".to_owned()),
        };
        assert_eq!(classification.role, target);
        assert_eq!(
            classification.clone().for_file(10_968_000).role,
            CandidateFileRole::Auxiliary
        );
        assert_eq!(
            classification.for_file(3_017_952 + 8_413_425_664).role,
            target
        );
    }

    #[test]
    fn a_file_without_tensors_holds_no_weights_unless_it_is_a_splits_first_shard() {
        use crate::test_gguf::{Meta, gguf_bytes};
        let architectures = RecognizedArchitectures::new(["qwen35"], ["dflash"]);
        let classify = |metadata: &[(&str, Meta<'_>)]| {
            classify_header(&gguf_bytes(metadata, &[]), &architectures)
                .unwrap()
                .role
        };
        assert_eq!(
            classify(&[("general.architecture", Meta::Str("qwen35"))]),
            CandidateFileRole::Auxiliary
        );
        assert_eq!(
            classify(&[
                ("general.architecture", Meta::Str("qwen35")),
                ("split.count", Meta::U32(3)),
            ]),
            CandidateFileRole::Target {
                architecture: Some("qwen35".to_owned())
            }
        );
    }

    /// HauhauCS's FastMTP file (the real header) declares the target's `qwen35` architecture,
    /// metadata and `general.file_type`, but holds only the trailing prediction block, a
    /// reduced-vocabulary output and an I32 `d2t` map. Its sibling holds every block.
    #[test]
    fn a_fastmtp_head_file_is_a_drafter_and_its_sibling_a_target() {
        let architectures = RecognizedArchitectures::new(
            ["qwen35", "qwen4exp"],
            ["dflash", "gemma4-assistant", "qwen4exp"],
        );
        let classify = |name: &str| classify_header(&fixture_header(name), &architectures).unwrap();
        assert_eq!(
            classify("Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf").role,
            CandidateFileRole::Drafter {
                architecture: "qwen35".to_owned()
            }
        );
        assert_eq!(
            classify("Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ3_XS.gguf").role,
            CandidateFileRole::Target {
                architecture: Some("qwen35".to_owned())
            }
        );
    }

    /// FadedRedStar's Qwen3.5 9B MTP draft files (the real headers) declare `qwen35` with 33
    /// blocks and one prediction layer, but hold only the token embedding, output norm and output
    /// head: no block at all. They are head files whose blocks are missing, so they are drafters
    /// the engine judges, never models. Their sibling holds the 32 model blocks.
    #[test]
    fn a_head_file_without_its_prediction_blocks_is_a_drafter() {
        let architectures = RecognizedArchitectures::new(
            ["qwen35", "qwen4exp"],
            ["dflash", "gemma4-assistant", "qwen4exp"],
        );
        let classify = |name: &str| classify_header(&fixture_header(name), &architectures).unwrap();
        for head in [
            "mtp-Qwen3.5-9B-heretic-Q8_0.gguf",
            "mtp-Qwen3.5-9B-heretic-BF16.gguf",
        ] {
            assert_eq!(
                classify(head).role,
                CandidateFileRole::Drafter {
                    architecture: "qwen35".to_owned()
                },
                "{head}"
            );
        }
        assert_eq!(
            classify("Qwen3.5-9B-heretic-Q4_K_M.gguf").role,
            CandidateFileRole::Target {
                architecture: Some("qwen35".to_owned())
            }
        );
        assert_eq!(
            classify("Qwen3.5-4B-Q3_K_M.gguf").role,
            CandidateFileRole::Target {
                architecture: Some("qwen35".to_owned())
            },
            "a model with built-in prediction blocks holds every block"
        );
    }

    /// The Arch round 3b files (the real headers): the catalog's UD-Q6_K_XL and the Q6_K beside
    /// it declare `nextn_predict_layers` 1 of 33 blocks, and their MTP block is the last of the
    /// 33 they hold. Each is a full model with built-in heads, never a head file.
    #[test]
    fn a_full_model_declaring_prediction_layers_is_a_target_with_built_in_heads() {
        let architectures = RecognizedArchitectures::new(
            ["qwen35", "qwen4exp"],
            ["dflash", "gemma4-assistant", "qwen4exp"],
        );
        for name in ["Qwen3.5-4B-UD-Q6_K_XL.gguf", "Qwen3.5-4B-Q6_K.gguf"] {
            let header = fixture_header(name);
            assert_eq!(
                classify_header(&header, &architectures).unwrap().role,
                CandidateFileRole::Target {
                    architecture: Some("qwen35".to_owned())
                },
                "{name}"
            );
            let directory = tempfile::tempdir().unwrap();
            let file = directory.path().join(name);
            std::fs::write(&file, &header).unwrap();
            let inspection = crate::gguf::inspect(&file).unwrap();
            assert_eq!(inspection.nextn_predict_layers, Some(1), "{name}");
            assert_eq!(
                inspection.role(&architectures),
                crate::gguf::GgufRole::Model { embedded_mtp: true },
                "{name}"
            );
        }
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
                &[],
                Some("Q3_K - Medium")
            ),
            ("MiniCPM5 1B".to_owned(), Some("Q3_K_M".to_owned()))
        );
        assert_eq!(
            model_presentation(
                Path::new("blobs/sha256-abc"),
                None,
                Some("Qwen3.5-4B-Q4_K_M"),
                &[],
                None
            ),
            ("Qwen3.5 4B".to_owned(), None)
        );
        assert_eq!(
            model_presentation(
                Path::new("blobs/sha256-abc"),
                None,
                Some("Qwen3.5-4B-Q4_K_M"),
                &[],
                Some("Q4_K - Medium")
            ),
            ("Qwen3.5 4B".to_owned(), Some("Q4_K_M".to_owned()))
        );
        assert_eq!(
            model_presentation(
                Path::new("models/new-q40.gguf"),
                None,
                None,
                &[],
                Some("Q4_0")
            ),
            ("new q40".to_owned(), Some("Q4_0".to_owned()))
        );
        assert_eq!(
            model_presentation(
                Path::new("models/MiniCPM5-1B-q4_k_s.gguf"),
                None,
                None,
                &[],
                None
            ),
            ("MiniCPM5 1B".to_owned(), Some("Q4_K_S".to_owned()))
        );
        assert_eq!(
            model_presentation(
                Path::new("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
                None,
                None,
                &[],
                None
            ),
            ("Qwen3.6 35B A3B".to_owned(), Some("UD-Q4_K_XL".to_owned()))
        );
    }

    /// QA (linux-arch and mac-m1, round 3c): a found Q4_K_M or Q5_K_M file renamed
    /// `renamed-q8.gguf` was labelled "Q8" from its name, while inspecting it said its header's
    /// type. A found model's label is its header's, as a candidate version's is; a name keeps its
    /// writer's spelling only of that same weight type.
    #[test]
    fn a_found_models_label_comes_from_its_header_before_its_file_name() {
        assert_eq!(
            model_presentation(
                Path::new("models/R/renamed-q8.gguf"),
                None,
                Some("Qwen3.5-0.8B"),
                &[],
                Some("Q4_K - Medium")
            ),
            ("Qwen3.5 0.8B".to_owned(), Some("Q4_K_M".to_owned()))
        );
        assert_eq!(
            model_presentation(
                Path::new("renamed-q8.gguf"),
                None,
                Some("4cd563d5"),
                &[],
                Some("Q5_K - Medium")
            )
            .1,
            Some("Q5_K_M".to_owned())
        );
        assert_eq!(
            model_presentation(
                Path::new("Qwen3.5-4B-UD-Q4_K_XL.gguf"),
                Some("unsloth/Qwen3.5-4B-GGUF"),
                None,
                &[],
                Some("Q4_K - Medium")
            )
            .1,
            Some("UD-Q4_K_XL".to_owned()),
            "a name spelling the header's type more precisely keeps it"
        );
        assert_eq!(
            model_presentation(Path::new("renamed-q8.gguf"), None, None, &[], None).1,
            Some("Q8".to_owned()),
            "without a header type, the name's"
        );
    }

    /// QA (windows, round 4): `LFM2.5-230M-Q8_0.gguf` from LiquidAI/LFM2.5-230M-GGUF names a
    /// source commit as its `general.name`, and was shown as that hash. A digest is no name, so the
    /// file names the model.
    #[test]
    fn a_digest_header_name_gives_way_to_the_file_name() {
        assert_eq!(
            model_presentation(
                Path::new("models/LFM2.5-230M-Q8_0.gguf"),
                None,
                Some("2639448dbae4a4cc657be3a5a8fcce8e531ee6c4"),
                &[],
                Some("Q8_0")
            ),
            ("LFM2.5 230M".to_owned(), Some("Q8_0".to_owned()))
        );
        assert_eq!(
            model_presentation(Path::new("weights.gguf"), None, Some("4cd563d"), &[], None).0,
            "weights"
        );
        assert_eq!(
            model_presentation(
                Path::new("weights.gguf"),
                None,
                Some("Qwen3-8B-abcdef"),
                &[],
                None
            )
            .0,
            "Qwen3 8B abcdef"
        );
    }

    /// QA (linux-cpu-10): Google's `gemma-4-E4B_q4_0-it.gguf` names its converter's source folder
    /// as its `general.name` (`Hf`), and was shown as "Hf". A placeholder identifies nothing, so the
    /// file names the model, and the header's one base model when the file name is no better.
    #[test]
    fn a_placeholder_header_name_gives_way_to_the_file_then_the_base_model() {
        let gemma = [crate::gguf::GgufBaseModel {
            name: Some("Gemma 4 E4B It".to_owned()),
            organization: Some("Google".to_owned()),
            repo_url: None,
        }];
        assert_eq!(
            model_presentation(
                Path::new("models/gemma-4-E4B_q4_0-it.gguf"),
                None,
                Some("Hf"),
                &gemma,
                Some("Q4_0")
            ),
            ("gemma 4 E4B q4 0 it".to_owned(), Some("Q4_0".to_owned()))
        );
        assert_eq!(
            model_presentation(
                Path::new(
                    "blobs/sha256-4cd563d5f1e2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c"
                ),
                None,
                Some("Hf"),
                &gemma,
                None
            )
            .0,
            "Gemma 4 E4B It"
        );
        assert_eq!(
            model_presentation(Path::new("m.gguf"), None, Some("2639448d"), &gemma, None).0,
            "Gemma 4 E4B It"
        );
        let merge = [gemma[0].clone(), gemma[0].clone()];
        assert_eq!(
            model_presentation(Path::new("m.gguf"), None, Some("Hf"), &merge, None).0,
            "m",
            "a merge's bases do not name it, and nothing better than the file name is left"
        );
        for name in [
            "Phi 4",
            "Qwen3.5 4B",
            "Gemma 4 E2B It",
            "Yi 6B",
            "deadbeefcafe",
            "LFM2.5 230M",
        ] {
            assert_eq!(
                model_presentation(Path::new("other.gguf"), None, Some(name), &gemma, None).0,
                name,
                "a real name, however short, names the model"
            );
        }
        for name in [
            "Hf",
            "",
            "4cd563d",
            "sha256 4cd563d5f1e2a3b4c5d6e7f8091a2b3c4d5e",
        ] {
            assert!(!identifies(name), "{name}");
        }
    }
}
