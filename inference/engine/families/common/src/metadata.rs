//! The architecture's metadata keys, read strictly: every `<architecture>.*`
//! key must be one the family interprets, and every value must have the type
//! its reader admits.

use crate::HeaderError;
use magnitude_artifacts::gguf::{Directory, Scalar, Value};

pub struct Metadata<'a> {
    directory: &'a Directory,
    prefix: String,
}

impl<'a> Metadata<'a> {
    /// Reads `architecture.*` keys, rejecting any outside `keys`.
    pub fn new(
        directory: &'a Directory,
        architecture: &str,
        keys: &[&str],
    ) -> Result<Self, HeaderError> {
        let prefix = format!("{architecture}.");
        if let Some(unknown) = directory.metadata.iter().find(|entry| {
            entry
                .name
                .strip_prefix(&prefix)
                .is_some_and(|key| !keys.contains(&key))
        }) {
            return Err(HeaderError::UnknownMetadata(unknown.name.clone()));
        }
        Ok(Self { directory, prefix })
    }

    /// The full metadata name of `key`.
    pub fn key(&self, key: &str) -> String {
        format!("{}{key}", self.prefix)
    }

    pub fn value(&self, key: &str) -> Option<&'a Value> {
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
        let value = self.typed(key, "a positive integer or one per layer", |value| {
            match value {
                Value::Array(values) if values.len() as u64 == layers => values
                    .iter()
                    .map(|value| nonnegative(value).filter(|value| *value > 0))
                    .collect(),
                Value::Array(_) => None,
                value => value
                    .unsigned()
                    .filter(|value| *value > 0)
                    .map(|value| vec![value; layers as usize]),
            }
        })?;
        self.required(key, value)
    }

    /// An array of nonnegative integers.
    pub fn counts(&self, key: &str) -> Result<Vec<u64>, HeaderError> {
        let value = self.typed(key, "an array of nonnegative integers", |value| {
            match value {
                Value::Array(values) => values.iter().map(nonnegative).collect(),
                Value::Scalar(_) => None,
            }
        })?;
        self.required(key, value)
    }

    /// One nonnegative integer per layer, stated as an array.
    pub fn layer_counts(&self, key: &str, layers: u64) -> Result<Vec<u64>, HeaderError> {
        let value = self.typed(
            key,
            "a per-layer array of nonnegative integers with one entry per layer",
            |value| match value {
                Value::Array(values) if values.len() as u64 == layers => {
                    values.iter().map(nonnegative).collect()
                }
                _ => None,
            },
        )?;
        self.required(key, value)
    }

    /// One flag per layer.
    pub fn optional_flags(&self, key: &str, layers: u64) -> Result<Option<Vec<bool>>, HeaderError> {
        self.typed(key, "one flag per layer", |value| match value {
            Value::Array(values) if values.len() as u64 == layers => values
                .iter()
                .map(|value| match value {
                    Scalar::Bool(value) => Some(*value),
                    _ => None,
                })
                .collect(),
            _ => None,
        })
    }

    pub fn flags(&self, key: &str, layers: u64) -> Result<Vec<bool>, HeaderError> {
        let value = self.optional_flags(key, layers)?;
        self.required(key, value)
    }
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Scalar(Scalar::Float(value)) => Some(*value),
        Value::Scalar(Scalar::Unsigned(value)) => Some(*value as f64),
        Value::Scalar(Scalar::Signed(value)) => Some(*value as f64),
        _ => None,
    }
    .filter(|value| value.is_finite())
}

/// An array entry; converters store integer arrays signed or unsigned.
fn nonnegative(value: &Scalar) -> Option<u64> {
    match value {
        Scalar::Unsigned(value) => Some(*value),
        Scalar::Signed(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}
