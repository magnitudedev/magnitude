//! The architecture's metadata keys, read strictly: every `<architecture>.*`
//! key must be one the family interprets, and every value must have the type
//! its reader admits.

use crate::HeaderError;
use magnitude_artifacts::gguf::{self, Directory, Element, Scalar, Value};
use std::cell::RefCell;
use std::collections::HashSet;

/// Every metadata entry in `architecture`'s own namespace, with its name
/// within that namespace.
///
/// Only that namespace configures a model, so it is the only one a family
/// reads strictly (the chat layer reads `tokenizer.*` strictly). Every other
/// namespace configures nothing: `general.*` describes the model, and
/// container and uploader namespaces (`split.*`, `quantize.*`, `imatrix.*`,
/// `mradermacher.*`, `unsloth.*`, ...) record how the file was made. A family
/// may read a named key there, but never refuses one it does not know.
pub fn architecture_keys<'a>(
    directory: &'a Directory,
    architecture: &str,
) -> impl Iterator<Item = (&'a str, &'a gguf::Metadata)> + 'a {
    let prefix = format!("{architecture}.");
    directory
        .metadata
        .iter()
        .filter_map(move |entry| entry.name.strip_prefix(&prefix).map(|key| (key, entry)))
}

/// The elements of the metadata array named `key`: the one reading of an
/// array's payload a family makes, a payload that doesn't read or decode
/// (text that isn't UTF-8, a payload shorter than its elements) being
/// [`HeaderError::Payload`].
pub fn array_elements<'a>(
    key: &str,
    values: &'a gguf::Array,
) -> Result<&'a gguf::Elements, HeaderError> {
    values.elements().map_err(|error| HeaderError::Payload {
        key: key.to_owned(),
        detail: error.to_string(),
    })
}

pub struct Metadata<'a> {
    directory: &'a Directory,
    prefix: String,
    accounted: RefCell<HashSet<String>>,
    ignored_prefixes: RefCell<Vec<String>>,
}

impl<'a> Metadata<'a> {
    /// Begins reading `architecture.*` keys. Every read marks its key as
    /// accounted for; [`Self::finish`] rejects any `<architecture>.*` key
    /// that was neither read nor explicitly ignored.
    pub fn new(directory: &'a Directory, architecture: &str) -> Self {
        Self {
            directory,
            prefix: format!("{architecture}."),
            accounted: RefCell::new(HashSet::new()),
            ignored_prefixes: RefCell::new(Vec::new()),
        }
    }

    /// Explicitly accounts for a known non-semantic or provenance key in
    /// `architecture`'s namespace so [`Self::finish`] admits it.
    pub fn ignore(&self, key: &str) {
        self.accounted.borrow_mut().insert(key.to_owned());
    }

    /// Explicitly accounts for every key in `architecture`'s namespace whose
    /// suffix starts with `prefix` (e.g. an uninterpreted sibling component).
    pub fn ignore_prefix(&self, prefix: &str) {
        self.ignored_prefixes.borrow_mut().push(prefix.to_owned());
    }

    /// Verifies that every `<architecture>.*` key in the directory was either
    /// read or explicitly ignored.
    pub fn finish(self) -> Result<(), HeaderError> {
        let accounted = self.accounted.into_inner();
        let ignored_prefixes = self.ignored_prefixes.into_inner();
        let unknown = self
            .directory
            .metadata
            .iter()
            .filter_map(|entry| {
                let key = entry.name.strip_prefix(&self.prefix)?;
                let covered = accounted.contains(key)
                    || ignored_prefixes
                        .iter()
                        .any(|prefix| key.starts_with(prefix));
                (!covered).then(|| entry.name.clone())
            })
            .collect::<Vec<_>>();
        if unknown.is_empty() {
            Ok(())
        } else {
            Err(HeaderError::UnknownMetadata(unknown))
        }
    }

    /// The full metadata name of `key`.
    pub fn key(&self, key: &str) -> String {
        format!("{}{key}", self.prefix)
    }

    pub fn value(&self, key: &str) -> Option<&'a Value> {
        self.accounted.borrow_mut().insert(key.to_owned());
        self.directory.value(&self.key(key))
    }

    fn required<T>(&self, key: &str, value: Option<T>) -> Result<T, HeaderError> {
        value.ok_or_else(|| HeaderError::MissingMetadata(self.key(key)))
    }

    fn malformed(&self, key: &str, expected: &'static str) -> HeaderError {
        HeaderError::MetadataType {
            key: self.key(key),
            expected,
        }
    }

    /// `key`'s value read by `read`, when present.
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

    /// A nonnegative integer.
    pub fn optional_count(&self, key: &str) -> Result<Option<u64>, HeaderError> {
        self.typed(key, "a nonnegative integer", Value::unsigned)
    }

    pub fn count(&self, key: &str) -> Result<u64, HeaderError> {
        let value = self.optional_count(key)?;
        self.required(key, value)
    }

    /// An optional nonnegative integer that defaults to `default` when absent.
    pub fn default_count(&self, key: &str, default: u64) -> Result<u64, HeaderError> {
        Ok(self.optional_count(key)?.unwrap_or(default))
    }

    /// An optional nonnegative integer that must equal `expected` when present.
    pub fn invariant_count(
        &self,
        key: &str,
        expected: u64,
        reason: &'static str,
    ) -> Result<(), HeaderError> {
        match self.optional_count(key)? {
            None => Ok(()),
            Some(value) if value == expected => Ok(()),
            Some(_) => Err(self.malformed(key, reason)),
        }
    }

    /// A positive integer.
    pub fn optional_integer(&self, key: &str) -> Result<Option<u64>, HeaderError> {
        self.typed(key, "a positive integer", |value| {
            value.unsigned().filter(|value| *value > 0)
        })
    }

    pub fn integer(&self, key: &str) -> Result<u64, HeaderError> {
        let value = self.optional_integer(key)?;
        self.required(key, value)
    }

    /// A finite number, stored as a float or an integer.
    pub fn optional_number(&self, key: &str) -> Result<Option<f64>, HeaderError> {
        self.typed(key, "a finite number", number)
    }

    pub fn number(&self, key: &str) -> Result<f64, HeaderError> {
        let value = self.optional_number(key)?;
        self.required(key, value)
    }

    pub fn optional_positive_number(&self, key: &str) -> Result<Option<f64>, HeaderError> {
        self.typed(key, "a finite positive number", |value| {
            number(value).filter(|value| *value > 0.0)
        })
    }

    pub fn positive_number(&self, key: &str) -> Result<f64, HeaderError> {
        let value = self.optional_positive_number(key)?;
        self.required(key, value)
    }

    pub fn optional_flag(&self, key: &str) -> Result<Option<bool>, HeaderError> {
        self.typed(key, "a boolean", |value| match value {
            Value::Scalar(Scalar::Bool(value)) => Some(*value),
            _ => None,
        })
    }

    pub fn flag(&self, key: &str) -> Result<bool, HeaderError> {
        let value = self.optional_flag(key)?;
        self.required(key, value)
    }

    /// An optional boolean that defaults to `default` when absent.
    pub fn default_flag(&self, key: &str, default: bool) -> Result<bool, HeaderError> {
        Ok(self.optional_flag(key)?.unwrap_or(default))
    }

    /// An optional boolean that must equal `expected` when present.
    pub fn invariant_flag(
        &self,
        key: &str,
        expected: bool,
        reason: &'static str,
    ) -> Result<(), HeaderError> {
        match self.optional_flag(key)? {
            None => Ok(()),
            Some(value) if value == expected => Ok(()),
            Some(_) => Err(self.malformed(key, reason)),
        }
    }

    /// The model's vocabulary size, inferred from the embedding table's rows
    /// and verified against `<arch>.vocab_size` when that key is present.
    pub fn vocab_size(&self, embedding_rows: u64) -> Result<u64, HeaderError> {
        if let Some(declared) = self.optional_integer("vocab_size")? {
            if declared != embedding_rows {
                return Err(self.malformed(
                    "vocab_size",
                    "equal to the embedding table's row count",
                ));
            }
        }
        Ok(embedding_rows)
    }

    /// Admits an absent `rope.scaling.type` or `"none"`, and rejects any
    /// other scaling type with [`HeaderError::RopeScaling`].
    pub fn reject_rope_scaling(&self) -> Result<(), HeaderError> {
        match self.value("rope.scaling.type") {
            None => Ok(()),
            Some(value) if value.string() == Some("none") => Ok(()),
            Some(value) => Err(HeaderError::RopeScaling(
                value.string().unwrap_or("unspecified").to_owned(),
            )),
        }
    }

    pub fn optional_string(&self, key: &str) -> Result<Option<&'a str>, HeaderError> {
        self.typed(key, "a string", Value::string)
    }

    pub fn string(&self, key: &str) -> Result<&'a str, HeaderError> {
        let value = self.optional_string(key)?;
        self.required(key, value)
    }

    /// A positive integer per layer, stated once for all layers or as an
    /// array with one entry per layer.
    pub fn per_layer(&self, key: &str, layers: u64) -> Result<Vec<u64>, HeaderError> {
        let value = self.uniform_or_array(
            key,
            "a positive integer or one per layer",
            layers,
            |element| element.unsigned().filter(|value| *value > 0),
        )?;
        self.required(key, value)
    }

    /// `key`'s value per layer, when present: stated once for all layers
    /// or as an array with one entry per layer, each read by `read`.
    fn uniform_or_array<T: Clone>(
        &self,
        key: &str,
        expected: &'static str,
        layers: u64,
        read: impl Fn(Element<'a>) -> Option<T>,
    ) -> Result<Option<Vec<T>>, HeaderError> {
        if layers > self.directory.tensors.len() as u64 {
            return Err(self.malformed(
                key,
                "a layer count not exceeding the stored tensor count",
            ));
        }
        match self.value(key) {
            Some(Value::Array(_)) => self.array(key, expected, Some(layers), read),
            _ => self.typed(key, expected, |value| match value {
                Value::Scalar(scalar) => {
                    read(scalar.element()).map(|value| vec![value; layers as usize])
                }
                Value::Array(_) => unreachable!("an array value is read by `array`"),
            }),
        }
    }

    /// `key`'s array, of `length` elements when given, each read by `read`,
    /// when present: the one reader of array values.
    fn array<T>(
        &self,
        key: &str,
        expected: &'static str,
        length: Option<u64>,
        read: impl Fn(Element<'a>) -> Option<T>,
    ) -> Result<Option<Vec<T>>, HeaderError> {
        match self.value(key) {
            None => Ok(None),
            Some(Value::Array(values))
                if length.is_none_or(|length| values.len() as u64 == length) =>
            {
                array_elements(&self.key(key), values)?
                    .iter()
                    .map(read)
                    .collect::<Option<Vec<T>>>()
                    .map(Some)
                    .ok_or_else(|| self.malformed(key, expected))
            }
            Some(_) => Err(self.malformed(key, expected)),
        }
    }

    /// An array of nonnegative integers.
    pub fn counts(&self, key: &str) -> Result<Vec<u64>, HeaderError> {
        let value = self.array(
            key,
            "an array of nonnegative integers",
            None,
            Element::unsigned,
        )?;
        self.required(key, value)
    }

    /// One nonnegative integer per layer, stated as an array.
    pub fn layer_counts(&self, key: &str, layers: u64) -> Result<Vec<u64>, HeaderError> {
        let value = self.array(
            key,
            "a per-layer array of nonnegative integers with one entry per layer",
            Some(layers),
            Element::unsigned,
        )?;
        self.required(key, value)
    }

    /// One finite number per layer, stated as an array.
    pub fn layer_numbers(&self, key: &str, layers: u64) -> Result<Vec<f64>, HeaderError> {
        let value = self.array(
            key,
            "a per-layer array of finite numbers with one entry per layer",
            Some(layers),
            element_number,
        )?;
        self.required(key, value)
    }

    /// A nonnegative integer per layer, stated once for all layers or as an
    /// array with one entry per layer.
    pub fn uniform_or_layer_counts(&self, key: &str, layers: u64) -> Result<Vec<u64>, HeaderError> {
        let value = self.uniform_or_array(
            key,
            "a nonnegative integer or one per layer",
            layers,
            Element::unsigned,
        )?;
        self.required(key, value)
    }

    /// A finite number per layer, stated once for all layers or as an array
    /// with one entry per layer.
    pub fn uniform_or_layer_numbers(&self, key: &str, layers: u64) -> Result<Vec<f64>, HeaderError> {
        let value = self.uniform_or_array(
            key,
            "a finite number or one per layer",
            layers,
            element_number,
        )?;
        self.required(key, value)
    }

    /// One flag per layer.
    pub fn optional_flags(&self, key: &str, layers: u64) -> Result<Option<Vec<bool>>, HeaderError> {
        self.array(key, "one flag per layer", Some(layers), |element| match element {
            Element::Bool(value) => Some(value),
            _ => None,
        })
    }

    pub fn flags(&self, key: &str, layers: u64) -> Result<Vec<bool>, HeaderError> {
        let value = self.optional_flags(key, layers)?;
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

    /// A required fixed-length array of finite numbers.
    pub fn f64_array<const N: usize>(&self, key: &str) -> Result<[f64; N], HeaderError> {
        let expected = "a fixed-length array of finite numbers";
        let value = self.array(key, expected, Some(N as u64), element_number)?;
        let parsed = self.required(key, value)?;
        <[f64; N]>::try_from(parsed).map_err(|_| self.malformed(key, expected))
    }
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Scalar(scalar) => element_number(scalar.element()),
        Value::Array(_) => None,
    }
}

/// A finite number, stored as a float or an integer.
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

    fn directory(entries: &[(&str, Value)]) -> Directory {
        directory_with_tensors(entries, 0)
    }

    fn directory_with_tensors(entries: &[(&str, Value)], tensor_count: usize) -> Directory {
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
                    name: format!("blk.{index}.attn_norm.weight"),
                    shape: vec![4],
                    encoding: Encoding::F32,
                    offset: 0,
                    nbytes: 16,
                })
                .collect(),
        }
    }

    #[test]
    fn finish_rejects_unaccounted_keys_and_admits_read_or_ignored() {
        let dir = directory(&[
            ("arch.block_count", Value::Scalar(Scalar::Unsigned(4))),
            ("arch.provenance", Value::Scalar(Scalar::Unsigned(1))),
            ("arch.sibling.a", Value::Scalar(Scalar::Unsigned(2))),
            ("arch.sibling.b", Value::Scalar(Scalar::Bool(true))),
            ("arch.unknown_switch", Value::Scalar(Scalar::Bool(true))),
            ("general.name", Value::Scalar(Scalar::String("demo".into()))),
        ]);

        let m = Metadata::new(&dir, "arch");
        assert_eq!(m.integer("block_count"), Ok(4));
        m.ignore("provenance");
        m.ignore_prefix("sibling.");
        assert_eq!(
            m.finish(),
            Err(HeaderError::UnknownMetadata(vec![
                "arch.unknown_switch".into()
            ]))
        );

        let m = Metadata::new(&dir, "arch");
        assert_eq!(m.integer("block_count"), Ok(4));
        m.ignore("provenance");
        m.ignore_prefix("sibling.");
        assert_eq!(m.optional_flag("unknown_switch"), Ok(Some(true)));
        assert_eq!(m.finish(), Ok(()));
    }

    #[test]
    fn bounded_block_count_and_uniform_or_array_reject_inflated_layers_without_allocating() {
        let valid = directory_with_tensors(
            &[
                ("arch.block_count", Value::Scalar(Scalar::Unsigned(2))),
                ("arch.feed_forward_length", Value::Scalar(Scalar::Unsigned(8))),
                ("arch.expert_used_count", Value::Scalar(Scalar::Unsigned(0))),
            ],
            2,
        );
        let m = Metadata::new(&valid, "arch");
        assert_eq!(m.bounded_block_count("block_count"), Ok(2));
        assert_eq!(m.per_layer("feed_forward_length", 2), Ok(vec![8, 8]));
        assert_eq!(
            m.uniform_or_layer_counts("expert_used_count", 2),
            Ok(vec![0, 0])
        );
        assert_eq!(m.finish(), Ok(()));

        let inflated = directory_with_tensors(
            &[
                ("arch.block_count", Value::Scalar(Scalar::Unsigned(u64::MAX))),
                ("arch.feed_forward_length", Value::Scalar(Scalar::Unsigned(8))),
                ("arch.expert_used_count", Value::Scalar(Scalar::Unsigned(1))),
            ],
            2,
        );
        let m = Metadata::new(&inflated, "arch");
        assert!(matches!(
            m.bounded_block_count("block_count"),
            Err(HeaderError::MetadataType { key, .. }) if key == "arch.block_count"
        ));
        assert!(matches!(
            m.per_layer("feed_forward_length", u64::MAX),
            Err(HeaderError::MetadataType { key, .. }) if key == "arch.feed_forward_length"
        ));
        assert!(matches!(
            m.uniform_or_layer_counts("expert_used_count", 3),
            Err(HeaderError::MetadataType { key, .. }) if key == "arch.expert_used_count"
        ));
    }

    #[test]
    fn policy_primitives_default_validate_and_reject_invariants() {
        let absent = directory(&[]);
        let m = Metadata::new(&absent, "arch");
        assert_eq!(m.default_count("shared", 0), Ok(0));
        assert_eq!(m.default_flag("norm", true), Ok(true));
        assert_eq!(m.invariant_count("gating", 2, "2 (sigmoid)"), Ok(()));
        assert_eq!(m.invariant_flag("tail", true, "true"), Ok(()));
        assert_eq!(m.vocab_size(32_000), Ok(32_000));
        assert_eq!(m.reject_rope_scaling(), Ok(()));
        assert_eq!(m.finish(), Ok(()));

        let matching = directory(&[
            ("arch.shared", Value::Scalar(Scalar::Unsigned(2))),
            ("arch.norm", Value::Scalar(Scalar::Bool(false))),
            ("arch.gating", Value::Scalar(Scalar::Unsigned(2))),
            ("arch.tail", Value::Scalar(Scalar::Bool(true))),
            ("arch.vocab_size", Value::Scalar(Scalar::Unsigned(32_000))),
            ("arch.rope.scaling.type", Value::Scalar(Scalar::String("none".into()))),
        ]);
        let m = Metadata::new(&matching, "arch");
        assert_eq!(m.default_count("shared", 0), Ok(2));
        assert_eq!(m.default_flag("norm", true), Ok(false));
        assert_eq!(m.invariant_count("gating", 2, "2 (sigmoid)"), Ok(()));
        assert_eq!(m.invariant_flag("tail", true, "true"), Ok(()));
        assert_eq!(m.vocab_size(32_000), Ok(32_000));
        assert_eq!(m.reject_rope_scaling(), Ok(()));
        assert_eq!(m.finish(), Ok(()));

        let mismatched = directory(&[
            ("arch.gating", Value::Scalar(Scalar::Unsigned(1))),
            ("arch.tail", Value::Scalar(Scalar::Bool(false))),
            ("arch.vocab_size", Value::Scalar(Scalar::Unsigned(16_000))),
            ("arch.rope.scaling.type", Value::Scalar(Scalar::String("yarn".into()))),
        ]);
        let m = Metadata::new(&mismatched, "arch");
        assert_eq!(
            m.invariant_count("gating", 2, "2 (sigmoid)"),
            Err(HeaderError::MetadataType {
                key: "arch.gating".into(),
                expected: "2 (sigmoid)",
            })
        );
        assert_eq!(
            m.invariant_flag("tail", true, "true"),
            Err(HeaderError::MetadataType {
                key: "arch.tail".into(),
                expected: "true",
            })
        );
        assert!(matches!(
            m.vocab_size(32_000),
            Err(HeaderError::MetadataType { key, .. }) if key == "arch.vocab_size"
        ));
        assert_eq!(
            m.reject_rope_scaling(),
            Err(HeaderError::RopeScaling("yarn".into()))
        );
    }
}
