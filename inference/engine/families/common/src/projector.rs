//! Which projectors a family binds to a target: the one rule every vision
//! family applies before interpreting a projector header.

use crate::{HeaderError, architecture_keys, array_elements};
use magnitude_artifacts::{
    UnsupportedReason,
    gguf::{Directory, Element, PROJECTOR_ARCHITECTURE, Scalar, Value},
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

/// Strict reader for a projector's `clip.*` metadata.
///
/// Only keys in the `clip.*` namespace are checked against the projector's
/// admitted key set (and any ignored prefixes, such as `clip.audio.` on a
/// multimodal projector that shares a file with an audio encoder). Container
/// and provenance namespaces (`general.*`, `split.*`, `quantize.*`,
/// `imatrix.*`, …) are left untouched.
pub struct ProjectorMetadata<'a> {
    directory: &'a Directory,
}

impl<'a> ProjectorMetadata<'a> {
    /// Validates that every `clip.*` metadata key in `directory` is listed in
    /// `admitted_keys`, leaving non-`clip.*` namespaces untouched.
    pub fn new(directory: &'a Directory, admitted_keys: &[&str]) -> Result<Self, HeaderError> {
        Self::with_ignored_prefixes(directory, admitted_keys, &[])
    }

    /// Validates that every `clip.*` metadata key in `directory` is either
    /// listed in `admitted_keys` or starts with one of `ignored_prefixes`,
    /// leaving non-`clip.*` namespaces untouched.
    pub fn with_ignored_prefixes(
        directory: &'a Directory,
        admitted_keys: &[&str],
        ignored_prefixes: &[&str],
    ) -> Result<Self, HeaderError> {
        let unknown = architecture_keys(directory, PROJECTOR_ARCHITECTURE)
            .filter(|(_, entry)| {
                !admitted_keys.contains(&entry.name.as_str())
                    && !ignored_prefixes
                        .iter()
                        .any(|prefix| entry.name.starts_with(prefix))
            })
            .map(|(_, entry)| entry.name.clone())
            .collect::<Vec<_>>();
        if !unknown.is_empty() {
            return Err(HeaderError::UnknownMetadata(unknown));
        }
        Ok(Self { directory })
    }

    pub fn value(&self, key: &str) -> Option<&'a Value> {
        self.directory.value(key)
    }

    fn required<T>(&self, key: &str, value: Option<T>) -> Result<T, HeaderError> {
        value.ok_or_else(|| HeaderError::MissingMetadata(key.to_owned()))
    }

    fn malformed(&self, key: &str, expected: &'static str) -> HeaderError {
        HeaderError::MetadataType {
            key: key.to_owned(),
            expected,
        }
    }

    fn typed<T>(
        &self,
        key: &str,
        expected: &'static str,
        read: impl FnOnce(&'a Value) -> Option<T>,
    ) -> Result<Option<T>, HeaderError> {
        self.value(key)
            .map(|value| read(value).ok_or_else(|| self.malformed(key, expected)))
            .transpose()
    }

    /// An optional nonnegative integer (`>= 0`).
    pub fn optional_count(&self, key: &str) -> Result<Option<u64>, HeaderError> {
        self.typed(key, "a nonnegative integer", Value::unsigned)
    }

    /// A required nonnegative integer (`>= 0`).
    pub fn count(&self, key: &str) -> Result<u64, HeaderError> {
        let value = self.optional_count(key)?;
        self.required(key, value)
    }

    /// An optional positive integer (`> 0`).
    pub fn optional_integer(&self, key: &str) -> Result<Option<u64>, HeaderError> {
        self.typed(key, "a positive integer", |value| {
            value.unsigned().filter(|value| *value > 0)
        })
    }

    /// A required positive integer (`> 0`).
    pub fn integer(&self, key: &str) -> Result<u64, HeaderError> {
        let value = self.optional_integer(key)?;
        self.required(key, value)
    }

    /// A required positive block count bounded by the directory's tensor count
    /// (`1 <= count <= directory.tensors.len() as u64`), guarding per-block
    /// allocations against untrusted headers.
    pub fn bounded_block_count(&self, key: &str) -> Result<u64, HeaderError> {
        let count = self.integer(key)?;
        if count > self.directory.tensors.len() as u64 {
            return Err(self.malformed(
                key,
                "a block count not exceeding the stored tensor count",
            ));
        }
        Ok(count)
    }

    /// An optional finite number, stored as a float or an integer.
    pub fn optional_number(&self, key: &str) -> Result<Option<f64>, HeaderError> {
        self.typed(key, "a finite number", scalar_number)
    }

    /// A required finite number, stored as a float or an integer.
    pub fn number(&self, key: &str) -> Result<f64, HeaderError> {
        let value = self.optional_number(key)?;
        self.required(key, value)
    }

    /// An optional finite positive number (`> 0.0`).
    pub fn optional_positive_number(&self, key: &str) -> Result<Option<f64>, HeaderError> {
        self.typed(key, "a finite positive number", |value| {
            scalar_number(value).filter(|value| *value > 0.0)
        })
    }

    /// A required finite positive number (`> 0.0`).
    pub fn positive_number(&self, key: &str) -> Result<f64, HeaderError> {
        let value = self.optional_positive_number(key)?;
        self.required(key, value)
    }

    /// An optional boolean.
    pub fn optional_bool(&self, key: &str) -> Result<Option<bool>, HeaderError> {
        self.typed(key, "a boolean", |value| match value {
            Value::Scalar(Scalar::Bool(value)) => Some(*value),
            _ => None,
        })
    }

    /// A required boolean.
    pub fn bool(&self, key: &str) -> Result<bool, HeaderError> {
        let value = self.optional_bool(key)?;
        self.required(key, value)
    }

    /// An optional string.
    pub fn optional_string(&self, key: &str) -> Result<Option<&'a str>, HeaderError> {
        self.typed(key, "a string", Value::string)
    }

    /// A required string.
    pub fn string(&self, key: &str) -> Result<&'a str, HeaderError> {
        let value = self.optional_string(key)?;
        self.required(key, value)
    }

    /// A required fixed-length array of finite numbers.
    pub fn f64_array<const N: usize>(&self, key: &str) -> Result<[f64; N], HeaderError> {
        let Some(value) = self.value(key) else {
            return Err(HeaderError::MissingMetadata(key.to_owned()));
        };
        let Value::Array(values) = value else {
            return Err(self.malformed(key, "a fixed-length array of finite numbers"));
        };
        if values.len() != N {
            return Err(self.malformed(key, "a fixed-length array of finite numbers"));
        }
        let parsed = array_elements(key, values)?
            .iter()
            .map(element_number)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| self.malformed(key, "a fixed-length array of finite numbers"))?;
        <[f64; N]>::try_from(parsed)
            .map_err(|_| self.malformed(key, "a fixed-length array of finite numbers"))
    }

    /// An optional boolean array of exact `length`.
    pub fn optional_bool_array(
        &self,
        key: &str,
        length: u64,
    ) -> Result<Option<Vec<bool>>, HeaderError> {
        let Some(value) = self.value(key) else {
            return Ok(None);
        };
        let Value::Array(values) = value else {
            return Err(self.malformed(key, "a boolean array"));
        };
        if values.len() as u64 != length {
            return Err(self.malformed(key, "a boolean array"));
        }
        array_elements(key, values)?
            .iter()
            .map(|element| match element {
                Element::Bool(value) => Some(value),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(Some)
            .ok_or_else(|| self.malformed(key, "a boolean array"))
    }
}

fn scalar_number(value: &Value) -> Option<f64> {
    match value {
        Value::Scalar(scalar) => element_number(scalar.element()),
        Value::Array(_) => None,
    }
}

fn element_number(element: Element) -> Option<f64> {
    match element {
        Element::Float(value) => Some(value),
        Element::Unsigned(value) => Some(value as f64),
        Element::Signed(value) => Some(value as f64),
        Element::String(_) | Element::Bool(_) => None,
    }
    .filter(|value| value.is_finite())
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
        let admitted = &["clip.has_vision_encoder", "clip.vision.block_count"];
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
        assert!(ProjectorMetadata::new(&with_container_keys, admitted).is_ok());

        let with_unknown_clip = directory(
            &[
                ("quantize.imatrix.file", Value::Scalar(Scalar::String("imatrix.dat".into()))),
                ("split.count", Value::Scalar(Scalar::Unsigned(2))),
                ("clip.has_vision_encoder", Value::Scalar(Scalar::Bool(true))),
                ("clip.vision.unknown_flag", Value::Scalar(Scalar::Bool(true))),
            ],
            1,
        );
        assert_eq!(
            ProjectorMetadata::new(&with_unknown_clip, admitted).err(),
            Some(HeaderError::UnknownMetadata(vec![
                "clip.vision.unknown_flag".into()
            ]))
        );
    }

    #[test]
    fn bounded_block_count_rejects_counts_exceeding_tensor_count() {
        let admitted = &["clip.vision.block_count"];
        let valid = directory(
            &[("clip.vision.block_count", Value::Scalar(Scalar::Unsigned(2)))],
            2,
        );
        let meta = ProjectorMetadata::new(&valid, admitted).unwrap();
        assert_eq!(meta.bounded_block_count("clip.vision.block_count"), Ok(2));

        let oversized = directory(
            &[("clip.vision.block_count", Value::Scalar(Scalar::Unsigned(3)))],
            2,
        );
        let meta = ProjectorMetadata::new(&oversized, admitted).unwrap();
        assert!(matches!(
            meta.bounded_block_count("clip.vision.block_count"),
            Err(HeaderError::MetadataType { key, .. }) if key == "clip.vision.block_count"
        ));

        let zero = directory(
            &[("clip.vision.block_count", Value::Scalar(Scalar::Unsigned(0)))],
            2,
        );
        let meta = ProjectorMetadata::new(&zero, admitted).unwrap();
        assert!(matches!(
            meta.bounded_block_count("clip.vision.block_count"),
            Err(HeaderError::MetadataType { key, .. }) if key == "clip.vision.block_count"
        ));
    }
}
