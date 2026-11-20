//! Which projectors a family binds to a target: the one rule every vision
//! family applies before interpreting a projector header.

use crate::{HeaderError, Metadata};
use magnitude_artifacts::{
    UnsupportedReason,
    gguf::{Directory, PROJECTOR_ARCHITECTURE, Scalar, Value},
};
use magnitude_family_contracts::FamilyError;
use std::{error, fmt};

/// Why a header given as a projector is not one the family binds to the
/// target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectorIdentityError {
    /// The header is not a CLIP mmproj with a vision encoder: nothing the
    /// engine can build as a projector.
    NotProjector,
    /// A vision projector that declares no type: nothing names it, so
    /// nothing can be read from it.
    Untyped,
    /// A vision projector of a type (`found`) the family does not read:
    /// another model's, which disagrees with the target. `read` are the
    /// types the family reads.
    Foreign {
        found: String,
        read: Vec<&'static str>,
    },
    /// A projector whose features (`projector` wide) are not the target's
    /// embedding width: it cannot feed this decoder. Names (a base model's)
    /// never decide; a fine-tune keeps its base's width.
    Width { projector: u64, target: u64 },
}

impl ProjectorIdentityError {
    /// `Unreadable` for a header that is not a projector or declares no
    /// type; `TargetMismatch` for a projector that disagrees with the target.
    pub fn reasons(&self) -> Vec<UnsupportedReason> {
        match self {
            Self::NotProjector | Self::Untyped => vec![UnsupportedReason::Unreadable],
            Self::Foreign { .. } | Self::Width { .. } => vec![UnsupportedReason::TargetMismatch {
                detail: self.to_string(),
            }],
        }
    }
}

impl fmt::Display for ProjectorIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotProjector => {
                formatter.write_str("the projector is not a CLIP mmproj with a vision encoder")
            }
            Self::Untyped => formatter.write_str("the projector declares no type"),
            Self::Foreign { found, read } => write!(
                formatter,
                "a {found} projector is not one this model reads ({})",
                read.join(", ")
            ),
            Self::Width { projector, target } => write!(
                formatter,
                "projector width {projector} differs from the target's {target}"
            ),
        }
    }
}

impl error::Error for ProjectorIdentityError {}

impl From<ProjectorIdentityError> for FamilyError {
    fn from(error: ProjectorIdentityError) -> Self {
        FamilyError::unsupported(error.reasons(), error.to_string())
    }
}

/// The keys a projector declares its type under: which one depends on the
/// model, and no projector declares both.
const TYPE_KEYS: [&str; 2] = ["clip.projector_type", "clip.vision.projector_type"];

/// The type of a projector the family binds to a target of `target_width`:
/// `directory` is a CLIP mmproj with a vision encoder whose type is one
/// `types` names, and whose declared feature width
/// (`clip.vision.projection_dim`) is the target's. `types` is the family's
/// one table of the types it reads, each with the reader it selects. A
/// projector that declares no width is the family's reader to refuse.
pub fn projector_type<K: Copy>(
    directory: &Directory,
    types: &[(&'static str, K)],
    target_width: u64,
) -> Result<K, ProjectorIdentityError> {
    if directory.value("general.architecture").and_then(Value::string)
        != Some(PROJECTOR_ARCHITECTURE)
        || directory.value("general.type").and_then(Value::string) != Some("mmproj")
        || !matches!(
            directory.value("clip.has_vision_encoder"),
            Some(Value::Scalar(Scalar::Bool(true)))
        )
    {
        return Err(ProjectorIdentityError::NotProjector);
    }
    let Some(found) = TYPE_KEYS
        .iter()
        .find_map(|key| directory.value(key).and_then(Value::string))
    else {
        return Err(ProjectorIdentityError::Untyped);
    };
    let Some(kind) = types
        .iter()
        .find(|(name, _)| *name == found)
        .map(|(_, kind)| *kind)
    else {
        return Err(ProjectorIdentityError::Foreign {
            found: found.to_owned(),
            read: types.iter().map(|(name, _)| *name).collect(),
        });
    };
    match directory
        .value("clip.vision.projection_dim")
        .and_then(Value::unsigned)
    {
        Some(projector) if projector != target_width => Err(ProjectorIdentityError::Width {
            projector,
            target: target_width,
        }),
        _ => Ok(kind),
    }
}

/// Strict tracked reader for a projector's `clip.*` metadata.
///
/// Wraps [`Metadata`] over the `clip.` namespace so callers address keys by
/// their full `clip.*` names. Every read marks its key as accounted for;
/// [`Self::finish`] rejects any `clip.*` key that was neither read nor
/// explicitly ignored. Container and provenance namespaces (`general.*`,
/// `split.*`, `quantize.*`, `imatrix.*`, …) are left untouched.
pub struct ProjectorMetadata<'a> {
    inner: Metadata<'a>,
}

impl<'a> ProjectorMetadata<'a> {
    /// Begins reading `clip.*` keys from `directory`.
    pub fn new(directory: &'a Directory) -> Self {
        Self {
            inner: Metadata::new(directory, PROJECTOR_ARCHITECTURE),
        }
    }

    /// Begins reading `clip.*` keys from a validated vision projector
    /// `directory`, automatically accounting for the projector-identity keys
    /// checked by [`projector_type`] (`clip.has_vision_encoder`,
    /// `clip.projector_type`, `clip.vision.projector_type`) and a benign
    /// `clip.has_audio_encoder = false` flag.
    pub fn open(directory: &'a Directory) -> Self {
        let meta = Self::new(directory);
        meta.ignore("clip.has_vision_encoder");
        meta.ignore("clip.projector_type");
        meta.ignore("clip.vision.projector_type");
        if matches!(
            directory.value("clip.has_audio_encoder"),
            Some(Value::Scalar(Scalar::Bool(false)))
        ) {
            meta.ignore("clip.has_audio_encoder");
        }
        meta
    }

    fn strip(key: &str) -> &str {
        key.strip_prefix("clip.").unwrap_or(key)
    }

    /// Explicitly accounts for a known non-semantic or already-validated `clip.*` key.
    pub fn ignore(&self, key: &str) {
        self.inner.ignore(Self::strip(key));
    }

    /// Explicitly accounts for every `clip.*` key starting with `prefix` (e.g. `"clip.audio."`).
    pub fn ignore_prefix(&self, prefix: &str) {
        self.inner.ignore_prefix(Self::strip(prefix));
    }

    /// Verifies that every `clip.*` key in the directory was either read or explicitly ignored.
    pub fn finish(self) -> Result<(), HeaderError> {
        self.inner.finish()
    }

    pub fn value(&self, key: &str) -> Option<&'a Value> {
        self.inner.value(Self::strip(key))
    }

    /// An optional nonnegative integer (`>= 0`).
    pub fn optional_count(&self, key: &str) -> Result<Option<u64>, HeaderError> {
        self.inner.optional_count(Self::strip(key))
    }

    /// A required nonnegative integer (`>= 0`).
    pub fn count(&self, key: &str) -> Result<u64, HeaderError> {
        self.inner.count(Self::strip(key))
    }

    /// An optional nonnegative integer that defaults to `default` when absent.
    pub fn default_count(&self, key: &str, default: u64) -> Result<u64, HeaderError> {
        self.inner.default_count(Self::strip(key), default)
    }

    /// An optional nonnegative integer that must equal `expected` when present.
    pub fn invariant_count(
        &self,
        key: &str,
        expected: u64,
        reason: &'static str,
    ) -> Result<(), HeaderError> {
        self.inner.invariant_count(Self::strip(key), expected, reason)
    }

    /// An optional positive integer (`> 0`).
    pub fn optional_integer(&self, key: &str) -> Result<Option<u64>, HeaderError> {
        self.inner.optional_integer(Self::strip(key))
    }

    /// A required positive integer (`> 0`).
    pub fn integer(&self, key: &str) -> Result<u64, HeaderError> {
        self.inner.integer(Self::strip(key))
    }

    /// A required positive block count bounded by the directory's tensor count
    /// (`1 <= count <= directory.tensors.len() as u64`), guarding per-block
    /// allocations against untrusted headers.
    pub fn bounded_block_count(&self, key: &str) -> Result<u64, HeaderError> {
        self.inner.bounded_block_count(Self::strip(key))
    }

    /// An optional finite number, stored as a float or an integer.
    pub fn optional_number(&self, key: &str) -> Result<Option<f64>, HeaderError> {
        self.inner.optional_number(Self::strip(key))
    }

    /// A required finite number, stored as a float or an integer.
    pub fn number(&self, key: &str) -> Result<f64, HeaderError> {
        self.inner.number(Self::strip(key))
    }

    /// An optional finite positive number (`> 0.0`).
    pub fn optional_positive_number(&self, key: &str) -> Result<Option<f64>, HeaderError> {
        self.inner.optional_positive_number(Self::strip(key))
    }

    /// A required finite positive number (`> 0.0`).
    pub fn positive_number(&self, key: &str) -> Result<f64, HeaderError> {
        self.inner.positive_number(Self::strip(key))
    }

    /// An optional boolean.
    pub fn optional_bool(&self, key: &str) -> Result<Option<bool>, HeaderError> {
        self.inner.optional_flag(Self::strip(key))
    }

    /// A required boolean.
    pub fn bool(&self, key: &str) -> Result<bool, HeaderError> {
        self.inner.flag(Self::strip(key))
    }

    /// An optional boolean that defaults to `default` when absent.
    pub fn default_flag(&self, key: &str, default: bool) -> Result<bool, HeaderError> {
        self.inner.default_flag(Self::strip(key), default)
    }

    /// An optional boolean that must equal `expected` when present.
    pub fn invariant_flag(
        &self,
        key: &str,
        expected: bool,
        reason: &'static str,
    ) -> Result<(), HeaderError> {
        self.inner.invariant_flag(Self::strip(key), expected, reason)
    }

    /// An optional string.
    pub fn optional_string(&self, key: &str) -> Result<Option<&'a str>, HeaderError> {
        self.inner.optional_string(Self::strip(key))
    }

    /// A required string.
    pub fn string(&self, key: &str) -> Result<&'a str, HeaderError> {
        self.inner.string(Self::strip(key))
    }

    /// A required fixed-length array of finite numbers.
    pub fn f64_array<const N: usize>(&self, key: &str) -> Result<[f64; N], HeaderError> {
        self.inner.f64_array(Self::strip(key))
    }

    /// An optional boolean array of exact `length`.
    pub fn optional_bool_array(
        &self,
        key: &str,
        length: u64,
    ) -> Result<Option<Vec<bool>>, HeaderError> {
        self.inner.optional_flags(Self::strip(key), length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_artifacts::gguf::{ByteOrder, Encoding, Metadata as Entry, TensorDescriptor};

    fn directory(entries: &[(&str, Value)], tensor_count: usize) -> Directory {
        Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata: entries
                .iter()
                .map(|(name, value)| Entry {
                    name: (*name).into(),
                    value: value.clone(),
                })
                .collect(),
            tensors: (0..tensor_count)
                .map(|index| TensorDescriptor {
                    name: format!("v.blk.{index}.ln1.weight"),
                    shape: vec![4],
                    encoding: Encoding::F32,
                    offset: 0,
                    nbytes: 16,
                })
                .collect(),
        }
    }

    #[test]
    fn admits_container_namespaces_while_rejecting_unknown_clip_keys() {
        let with_container_keys = directory(
            &[
                ("general.architecture", Value::Scalar(Scalar::String("clip".into()))),
                ("quantize.imatrix.file", Value::Scalar(Scalar::String("imatrix.dat".into()))),
                ("split.count", Value::Scalar(Scalar::Unsigned(2))),
                ("clip.has_vision_encoder", Value::Scalar(Scalar::Bool(true))),
                ("clip.vision.block_count", Value::Scalar(Scalar::Unsigned(1))),
            ],
            1,
        );
        let meta = ProjectorMetadata::new(&with_container_keys);
        assert_eq!(meta.bool("clip.has_vision_encoder"), Ok(true));
        assert_eq!(meta.integer("clip.vision.block_count"), Ok(1));
        assert_eq!(meta.finish(), Ok(()));

        let with_unknown_clip = directory(
            &[
                ("quantize.imatrix.file", Value::Scalar(Scalar::String("imatrix.dat".into()))),
                ("split.count", Value::Scalar(Scalar::Unsigned(2))),
                ("clip.has_vision_encoder", Value::Scalar(Scalar::Bool(true))),
                ("clip.vision.unknown_flag", Value::Scalar(Scalar::Bool(true))),
            ],
            1,
        );
        let meta = ProjectorMetadata::new(&with_unknown_clip);
        assert_eq!(meta.bool("clip.has_vision_encoder"), Ok(true));
        assert_eq!(
            meta.finish(),
            Err(HeaderError::UnknownMetadata(vec![
                "clip.vision.unknown_flag".into()
            ]))
        );
    }

    #[test]
    fn bounded_block_count_rejects_counts_exceeding_tensor_count() {
        let valid = directory(
            &[("clip.vision.block_count", Value::Scalar(Scalar::Unsigned(2)))],
            2,
        );
        let meta = ProjectorMetadata::new(&valid);
        assert_eq!(meta.bounded_block_count("clip.vision.block_count"), Ok(2));
        assert_eq!(meta.finish(), Ok(()));

        let oversized = directory(
            &[("clip.vision.block_count", Value::Scalar(Scalar::Unsigned(3)))],
            2,
        );
        let meta = ProjectorMetadata::new(&oversized);
        assert!(matches!(
            meta.bounded_block_count("clip.vision.block_count"),
            Err(HeaderError::MetadataType { key, .. }) if key == "clip.vision.block_count"
        ));

        let zero = directory(
            &[("clip.vision.block_count", Value::Scalar(Scalar::Unsigned(0)))],
            2,
        );
        let meta = ProjectorMetadata::new(&zero);
        assert!(matches!(
            meta.bounded_block_count("clip.vision.block_count"),
            Err(HeaderError::MetadataType { key, .. }) if key == "clip.vision.block_count"
        ));
    }

    #[test]
    fn open_accounts_for_projector_identity_and_false_audio_only() {
        let benign = directory(
            &[
                ("clip.has_vision_encoder", Value::Scalar(Scalar::Bool(true))),
                ("clip.projector_type", Value::Scalar(Scalar::String("demo".into()))),
                ("clip.vision.projector_type", Value::Scalar(Scalar::String("demo".into()))),
                ("clip.has_audio_encoder", Value::Scalar(Scalar::Bool(false))),
            ],
            1,
        );
        assert_eq!(ProjectorMetadata::open(&benign).finish(), Ok(()));

        let with_audio = directory(
            &[
                ("clip.has_vision_encoder", Value::Scalar(Scalar::Bool(true))),
                ("clip.projector_type", Value::Scalar(Scalar::String("demo".into()))),
                ("clip.has_audio_encoder", Value::Scalar(Scalar::Bool(true))),
            ],
            1,
        );
        assert_eq!(
            ProjectorMetadata::open(&with_audio).finish(),
            Err(HeaderError::UnknownMetadata(vec![
                "clip.has_audio_encoder".into()
            ]))
        );
    }
}
