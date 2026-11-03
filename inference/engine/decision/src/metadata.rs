//! Typed reads of a model's decision keys. Every key a description reads
//! is required where it is read, and a key nothing reads is refused
//! ([`Keys::finish`]), each naming the key (see `KEYS.md`).

use magnitude_artifacts::{DecisionMetadata, DecisionValue};
use std::cell::RefCell;
use std::collections::BTreeSet;

pub(crate) struct Keys<'a> {
    metadata: &'a DecisionMetadata,
    read: RefCell<BTreeSet<&'a str>>,
}

/// `<key>` as a refusal names it.
pub(crate) fn wrong(key: &str, expected: &str) -> String {
    format!("decision metadata `{key}` must be {expected}")
}

impl<'a> Keys<'a> {
    pub fn new(metadata: &'a DecisionMetadata) -> Self {
        Self {
            metadata,
            read: RefCell::new(BTreeSet::new()),
        }
    }

    /// The value of `key`, when the model writes it; read either way.
    pub fn optional(&self, key: &str) -> Option<&'a DecisionValue> {
        let (key, value) = self.metadata.values().get_key_value(key)?;
        self.read.borrow_mut().insert(key.as_str());
        Some(value)
    }

    fn value(&self, key: &str) -> Result<&'a DecisionValue, String> {
        self.optional(key)
            .ok_or_else(|| format!("decision metadata `{key}` is missing"))
    }

    /// The keys under `prefix`, in order; none read.
    pub fn under(&self, prefix: &str) -> Vec<&'a str> {
        self.metadata
            .values()
            .keys()
            .map(String::as_str)
            .filter(|key| key.starts_with(prefix))
            .collect()
    }

    pub fn string(&self, key: &str) -> Result<&'a str, String> {
        match self.value(key)? {
            DecisionValue::String(value) => Ok(value),
            _ => Err(wrong(key, "a string")),
        }
    }

    pub fn strings(&self, key: &str) -> Result<&'a [String], String> {
        match self.value(key)? {
            DecisionValue::Strings(values) => Ok(values),
            _ => Err(wrong(key, "an array of strings")),
        }
    }

    /// One of `names`, by its index there.
    pub fn choose(&self, key: &str, names: &[&str]) -> Result<usize, String> {
        let value = self.string(key)?;
        names.iter().position(|name| *name == value).ok_or_else(|| {
            format!(
                "decision metadata `{key}` names `{value}`, which the engine does not implement (it implements {})",
                names
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }

    pub fn flag(&self, key: &str) -> Result<bool, String> {
        match self.value(key)? {
            DecisionValue::Bool(value) => Ok(*value),
            _ => Err(wrong(key, "a boolean")),
        }
    }

    /// A nonnegative integer.
    pub fn integer(&self, key: &str) -> Result<usize, String> {
        match self.value(key)? {
            DecisionValue::Integer(value) => {
                usize::try_from(*value).map_err(|_| wrong(key, "a nonnegative integer"))
            }
            _ => Err(wrong(key, "an integer")),
        }
    }

    pub fn count(&self, key: &str) -> Result<usize, String> {
        match self.integer(key)? {
            0 => Err(wrong(key, "a positive integer")),
            count => Ok(count),
        }
    }

    pub fn floats(&self, key: &str) -> Result<&'a [f64], String> {
        match self.value(key)? {
            DecisionValue::Floats(values) if values.iter().all(|value| value.is_finite()) => {
                Ok(values)
            }
            _ => Err(wrong(key, "an array of finite floats")),
        }
    }

    /// A finite, positive float (a temperature).
    pub fn positive(&self, key: &str) -> Result<f64, String> {
        match self.value(key)? {
            DecisionValue::Float(value) if value.is_finite() && *value > 0.0 => Ok(*value),
            _ => Err(wrong(key, "a finite positive float")),
        }
    }

    /// Refuse every key nothing read, so a conversion that writes a key
    /// the engine would ignore is caught.
    pub fn finish(self) -> Result<(), String> {
        let read = self.read.into_inner();
        match self
            .metadata
            .values()
            .keys()
            .find(|key| !read.contains(key.as_str()))
        {
            Some(key) => Err(format!(
                "decision metadata `{key}` is not read by the engine's decision description"
            )),
            None => Ok(()),
        }
    }
}
