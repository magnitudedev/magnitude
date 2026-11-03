//! A model's decision description as its GGUF carries it: the decision
//! template `tokenizer.chat_template.decision`, the keys under
//! [`DECISION_NAMESPACE`] with that prefix removed, and llama.cpp's
//! attention causality flag `<arch>.attention.causal`. The `decision` crate
//! owns their meaning; the artifact only carries them.
//!
//! What a model serves follows from them: a model that carries a decision
//! template serves decisions only, and every other model serves chat.
//!
//! The description is the same for every architecture, so its keys are a
//! namespace of their own beside `general.*` and `tokenizer.*`, which no
//! family reads. A described header keeps its causality flag and its
//! readout head ([`READOUT_HEAD`]) beside its family's: package inspection
//! reads them, once, for every family, and a family's strict reading leaves
//! them ([`DecisionMetadata::owns_key`], [`DecisionMetadata::owns_tensor`]). The
//! decision template is not a chat template variant
//! ([`crate::TemplatePayload`] leaves it).

use crate::gguf::{Array, Directory, Scalar, Value};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// The namespace of a decision description's keys.
pub const DECISION_NAMESPACE: &str = "decision.";
/// The decision template: the prompt a decision model renders its requests
/// with. A header that carries it is a decision model.
pub const DECISION_TEMPLATE: &str = "tokenizer.chat_template.decision";
/// llama.cpp's `<arch>.attention.causal`: whether the model's attention, as
/// trained, is causal; true when absent.
const CAUSAL: &str = "attention.causal";
/// The stored name of a decision model's readout head: a `[outputs, hidden]`
/// matrix over the final normalized hidden state.
pub const READOUT_HEAD: &str = "decision.readout.weight";
/// The key naming a header's architecture, under whose namespace its
/// causality flag is.
const ARCHITECTURE: &str = "general.architecture";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DecisionValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Bools(Vec<bool>),
    Integers(Vec<i64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionMetadata {
    /// [`DECISION_TEMPLATE`]: the Jinja source the model's requests render
    /// with.
    template: String,
    /// `<arch>.attention.causal`: whether the model's global attention, as
    /// trained, is causal. The decision type's attention layout must agree.
    causal: bool,
    /// Every key under [`DECISION_NAMESPACE`], without that prefix.
    values: BTreeMap<String, DecisionValue>,
}

/// Why a header's decision metadata cannot be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionError {
    /// A required key is absent.
    Missing { key: String },
    /// A key holds a value outside what it admits.
    Malformed { key: String, expected: &'static str },
}

impl fmt::Display for DecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { key } => write!(formatter, "missing metadata {key}"),
            Self::Malformed { key, expected } => write!(formatter, "{key} must be {expected}"),
        }
    }
}

impl std::error::Error for DecisionError {}

impl DecisionMetadata {
    pub fn new(
        template: String,
        causal: bool,
        values: BTreeMap<String, DecisionValue>,
    ) -> Self {
        Self {
            template,
            causal,
            values,
        }
    }

    /// Whether `directory` carries a decision template, readable or not.
    pub fn described(directory: &Directory) -> bool {
        directory.value(DECISION_TEMPLATE).is_some()
    }

    /// Whether `key`, a key within the header's architecture namespace,
    /// belongs to its decision description rather than its family: the
    /// header is described and the key is llama.cpp's causality flag.
    pub fn owns_key(directory: &Directory, key: &str) -> bool {
        key == CAUSAL && Self::described(directory)
    }

    /// Whether the stored tensor `name` is the readout head of a described
    /// header.
    pub fn owns_tensor(directory: &Directory, name: &str) -> bool {
        name == READOUT_HEAD && Self::described(directory)
    }

    /// The decision description of a target header, when it carries a
    /// decision template.
    ///
    /// Decision keys without a template are refused, and so is a non-causal
    /// model without one: the engine runs non-causal attention only in a
    /// decision layout.
    pub fn of(directory: &Directory) -> Result<Option<Self>, DecisionError> {
        let architecture = architecture(directory).ok_or_else(|| DecisionError::Missing {
            key: ARCHITECTURE.to_owned(),
        })?;
        Self::read(directory, architecture)
    }

    fn read(directory: &Directory, architecture: &str) -> Result<Option<Self>, DecisionError> {
        let key = |name: &str| format!("{architecture}.{name}");
        let causal = match directory.value(&key(CAUSAL)) {
            None => true,
            Some(Value::Scalar(Scalar::Bool(causal))) => *causal,
            Some(_) => {
                return Err(DecisionError::Malformed {
                    key: key(CAUSAL),
                    expected: "a boolean",
                });
            }
        };
        let mut values = BTreeMap::new();
        for entry in &directory.metadata {
            let Some(name) = entry.name.strip_prefix(DECISION_NAMESPACE) else {
                continue;
            };
            let value =
                decision_value(&entry.value).ok_or_else(|| DecisionError::Malformed {
                    key: entry.name.clone(),
                    expected: "integers within the range of i64",
                })?;
            values.insert(name.to_owned(), value);
        }
        let template = directory
            .value(DECISION_TEMPLATE)
            .map(|template| {
                template
                    .string()
                    .filter(|template| !template.is_empty())
                    .ok_or_else(|| DecisionError::Malformed {
                        key: DECISION_TEMPLATE.to_owned(),
                        expected: "a nonempty string",
                    })
            })
            .transpose()?;
        match template {
            Some(template) => Ok(Some(Self {
                template: template.to_owned(),
                causal,
                values,
            })),
            None if !values.is_empty() => Err(DecisionError::Missing {
                key: DECISION_TEMPLATE.to_owned(),
            }),
            None if !causal => Err(DecisionError::Malformed {
                key: key(CAUSAL),
                expected: "true unless the model carries a decision template",
            }),
            None => Ok(None),
        }
    }

    pub fn template(&self) -> &str {
        &self.template
    }

    pub fn causal(&self) -> bool {
        self.causal
    }

    pub fn values(&self) -> &BTreeMap<String, DecisionValue> {
        &self.values
    }
}

fn architecture(directory: &Directory) -> Option<&str> {
    directory.value(ARCHITECTURE).and_then(Value::string)
}

/// `value` as a decision value. Only an unsigned integer beyond `i64` has
/// none.
fn decision_value(value: &Value) -> Option<DecisionValue> {
    Some(match value {
        Value::Scalar(Scalar::Bool(value)) => DecisionValue::Bool(*value),
        Value::Scalar(Scalar::Unsigned(value)) => DecisionValue::Integer(i64::try_from(*value).ok()?),
        Value::Scalar(Scalar::Signed(value)) => DecisionValue::Integer(*value),
        Value::Scalar(Scalar::Float(value)) => DecisionValue::Float(*value),
        Value::Scalar(Scalar::String(value)) => DecisionValue::String(value.clone()),
        Value::Array(Array::Bool(values)) => DecisionValue::Bools(values.clone()),
        Value::Array(Array::Unsigned(values)) => DecisionValue::Integers(
            values
                .iter()
                .map(|value| i64::try_from(*value).ok())
                .collect::<Option<_>>()?,
        ),
        Value::Array(Array::Signed(values)) => DecisionValue::Integers(values.clone()),
        Value::Array(Array::Float(values)) => DecisionValue::Floats(values.clone()),
        Value::Array(Array::String(values)) => {
            DecisionValue::Strings(values.iter().map(str::to_owned).collect())
        }
    })
}

/// What a model serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Serves {
    /// Generated text: every model without a decision template.
    Chat,
    /// Decisions only: a model with a decision template.
    Decisions,
}

impl Serves {
    /// What a model with this read decision description serves.
    pub fn of_description(decision: Option<&DecisionMetadata>) -> Self {
        match decision {
            Some(_) => Self::Decisions,
            None => Self::Chat,
        }
    }

    /// What a header serves when its description may not read: decisions
    /// when it carries a decision template, and chat otherwise.
    pub fn of(directory: &Directory) -> Self {
        if DecisionMetadata::described(directory) {
            Self::Decisions
        } else {
            Self::Chat
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{ByteOrder, Metadata};

    /// A description key: `key` under [`DECISION_NAMESPACE`].
    fn described(key: &str) -> String {
        format!("{DECISION_NAMESPACE}{key}")
    }

    fn header(entries: Vec<(&str, Value)>) -> Directory {
        Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata: entries
                .into_iter()
                .map(|(name, value)| Metadata {
                    name: name.into(),
                    value,
                })
                .collect(),
            tensors: Vec::new(),
        }
    }

    fn string(value: &str) -> Value {
        Value::Scalar(Scalar::String(value.into()))
    }

    fn flag(value: bool) -> Value {
        Value::Scalar(Scalar::Bool(value))
    }

    /// A described header's decision keys read with their prefix removed,
    /// beside its template.
    #[test]
    fn a_described_header_reads_with_its_prefix_removed() {
        let directory = header(vec![
            ("general.architecture", string("qwen35")),
            ("qwen35.block_count", Value::Scalar(Scalar::Unsigned(24))),
            ("qwen35.attention.causal", flag(false)),
            (DECISION_TEMPLATE, string("{{ state }}")),
            (
                described("temperature.choice").as_str(),
                Value::Scalar(Scalar::Float(1.0)),
            ),
            (
                described("labels").as_str(),
                Value::Array(Array::of([Scalar::String("A".into())]).unwrap()),
            ),
        ]);
        let decision = DecisionMetadata::of(&directory).unwrap().unwrap();
        assert_eq!(decision.template(), "{{ state }}");
        assert!(!decision.causal());
        assert_eq!(
            decision.values().keys().collect::<Vec<_>>(),
            ["labels", "temperature.choice"]
        );
        assert_eq!(
            decision.values().get("labels"),
            Some(&DecisionValue::Strings(vec!["A".into()]))
        );
        assert_eq!(Serves::of(&directory), Serves::Decisions);
        assert!(DecisionMetadata::described(&directory));
        assert!(DecisionMetadata::owns_key(&directory, "attention.causal"));
        assert!(!DecisionMetadata::owns_key(&directory, "block_count"));
        assert!(DecisionMetadata::owns_tensor(&directory, READOUT_HEAD));
        assert!(!DecisionMetadata::owns_tensor(&directory, "output.weight"));
    }

    /// A header without a decision template keeps its causality flag and
    /// readout head with its family, which refuses them.
    #[test]
    fn an_undescribed_header_owns_no_causality_flag_or_head() {
        let directory = header(vec![
            ("general.architecture", string("qwen35")),
            ("qwen35.attention.causal", flag(false)),
        ]);
        assert!(!DecisionMetadata::described(&directory));
        assert!(!DecisionMetadata::owns_key(&directory, "attention.causal"));
        assert!(!DecisionMetadata::owns_tensor(&directory, READOUT_HEAD));
    }

    #[test]
    fn a_header_without_decision_keys_serves_chat() {
        for directory in [
            header(vec![("lfm3.block_count", Value::Scalar(Scalar::Unsigned(24)))]),
            header(vec![("lfm3.attention.causal", flag(true))]),
        ] {
            assert_eq!(DecisionMetadata::read(&directory, "lfm3"), Ok(None));
            assert_eq!(Serves::of(&directory), Serves::Chat);
        }
    }

    /// Decision keys without a template are refused; so is an undescribed
    /// non-causal model and an empty template.
    #[test]
    fn decision_keys_without_a_template_and_undescribed_non_causal_models_are_refused() {
        let undescribed = header(vec![
            (described("segments").as_str(), string("whole")),
            (
                described("temperature.choice").as_str(),
                Value::Scalar(Scalar::Float(1.337)),
            ),
        ]);
        assert_eq!(
            DecisionMetadata::read(&undescribed, "lfm3"),
            Err(DecisionError::Missing {
                key: DECISION_TEMPLATE.into()
            })
        );
        let non_causal = header(vec![("qwen35.attention.causal", flag(false))]);
        assert!(matches!(
            DecisionMetadata::read(&non_causal, "qwen35"),
            Err(DecisionError::Malformed { key, .. }) if key == "qwen35.attention.causal"
        ));
        let empty = header(vec![(DECISION_TEMPLATE, string(""))]);
        assert!(DecisionMetadata::read(&empty, "qwen35").is_err());
        let wide = header(vec![
            (DECISION_TEMPLATE, string("t")),
            (described("n").as_str(), Value::Scalar(Scalar::Unsigned(u64::MAX))),
        ]);
        assert!(DecisionMetadata::read(&wide, "qwen35").is_err());
    }
}
