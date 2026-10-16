//! The files of a candidate (a Hugging Face repository at a commit, or a local folder) grouped
//! into versions and companions. A file's role comes from its header, never its name; names
//! only group shards and label versions for presentation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use magnitude_artifacts::gguf::{Directory, Value};
use magnitude_service_contracts::InventoryError;

/// The `general.architecture` values the engine recognizes as targets and as separate drafters.
/// The service holds no list of its own; the server supplies the engine's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecognizedArchitectures {
    pub targets: BTreeSet<String>,
    pub drafters: BTreeSet<String>,
}

impl RecognizedArchitectures {
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
}

const PROJECTOR_ARCHITECTURE: &str = "clip";

/// What a GGUF is for, read from its header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateFileRole {
    Target { architecture: Option<String> },
    Projector,
    Drafter { architecture: String },
}

impl CandidateFileRole {
    #[must_use]
    pub fn of(directory: &Directory, architectures: &RecognizedArchitectures) -> Self {
        let architecture = directory
            .value("general.architecture")
            .and_then(Value::string)
            .map(ToOwned::to_owned);
        let Some(architecture) = architecture else {
            return Self::Target { architecture: None };
        };
        if architecture == PROJECTOR_ARCHITECTURE
            || directory
                .metadata
                .iter()
                .any(|entry| entry.name.starts_with("clip.has_vision_encoder"))
        {
            return Self::Projector;
        }
        // A separate drafter declares a drafter architecture. One the engine also runs as a
        // target (a prediction head) is a drafter only when it declares prediction layers.
        let prediction_only = directory
            .value(&format!("{architecture}.nextn_predict_layers"))
            .is_some();
        if architectures.drafters.contains(&architecture)
            && (!architectures.targets.contains(&architecture) || prediction_only)
        {
            return Self::Drafter { architecture };
        }
        Self::Target {
            architecture: Some(architecture),
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
/// (the `general.file_type` quantization among them).
pub fn header_inspection(
    header: &[u8],
) -> Result<(Directory, crate::gguf::GgufInspection), InventoryError> {
    let file =
        tempfile::NamedTempFile::new().map_err(|error| InventoryError::Io(error.to_string()))?;
    std::fs::write(file.path(), header).map_err(|error| InventoryError::Io(error.to_string()))?;
    let directory = magnitude_artifacts::gguf::inspect_header(file.path())
        .map_err(|error| InventoryError::Integrity(error.to_string()))?;
    let inspection = crate::gguf::inspect(file.path())
        .map_err(|error| InventoryError::Integrity(error.to_string()))?;
    Ok((directory, inspection))
}

/// The header directory of a GGUF on disk (every shard's, for a split file's first shard).
pub fn local_directory(path: &Path) -> Result<Directory, InventoryError> {
    magnitude_artifacts::gguf::inspect_header(path)
        .map_err(|error| InventoryError::Integrity(format!("{}: {error}", path.display())))
}

/// The lowercase hex sha256 of a GGUF's header bytes on disk, through the aligned tensor-data
/// offset: the content identity of a file with no Hugging Face identity.
pub fn local_header_digest(path: &Path) -> Result<String, InventoryError> {
    use sha2::Digest;
    use std::io::Read;
    let directory = local_directory(path)?;
    let length = usize::try_from(directory.data_offset)
        .map_err(|_| InventoryError::Integrity("GGUF header is too large".to_owned()))?;
    let mut header = vec![0_u8; length];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut header))
        .map_err(|error| InventoryError::Io(format!("{}: {error}", path.display())))?;
    Ok(format!("{:x}", sha2::Sha256::digest(&header)))
}

/// The Hugging Face repositories a GGUF names as its base models (`general.base_model.N.repo_url`).
#[must_use]
pub fn header_base_repositories(directory: &Directory) -> BTreeSet<String> {
    let count = directory
        .value("general.base_model.count")
        .and_then(Value::unsigned)
        .unwrap_or(0)
        .min(64);
    (0..count)
        .filter_map(|index| {
            directory
                .value(&format!("general.base_model.{index}.repo_url"))
                .and_then(Value::string)
                .and_then(hugging_face_repository_from_url)
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
                let parent = file.path.parent().map(Path::to_path_buf).unwrap_or_default();
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
pub fn version_label(path: &Path) -> String {
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
        .unwrap_or(stem)
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
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    digits > 0
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
            ("Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf", "Q4_K_P"),
            ("model-IQ2_M.gguf", "IQ2_M"),
            ("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
            ("Q8_0/model-Q8_0-00001-of-00002.gguf", "Q8_0"),
            ("BF16/model-00001-of-00002.gguf", "BF16"),
            ("huihui-model.Q4_K.gguf", "Q4_K"),
            ("model-bf16.gguf", "BF16"),
            ("weights.gguf", "weights"),
        ] {
            assert_eq!(version_label(Path::new(path)), label, "{path}");
        }
    }

    #[test]
    fn repository_urls_and_display_names_normalize() {
        assert_eq!(
            hugging_face_repository_from_url("https://huggingface.co/Qwen/Qwen3.6-35B-A3B"),
            Some("qwen/qwen3.6-35b-a3b".to_owned())
        );
        assert_eq!(hugging_face_repository_from_url("https://example.com/a/b"), None);
        assert_eq!(
            display_name("HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-GGUF"),
            "Qwen3.6 35B A3B Uncensored HauhauCS Aggressive"
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
}
