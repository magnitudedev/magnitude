//! Literal tokenizer and chat-template payloads carried by a container.
//!
//! These types deliberately preserve GGUF metadata rather than selecting a
//! tokenizer implementation or template/parser family.

use crate::{
    gguf::{Array, Directory, Metadata, Value},
    Error, DECISION_TEMPLATE,
};

#[derive(Clone, Debug, PartialEq)]
pub struct TokenizerPayload {
    pub metadata: Vec<Metadata>,
}

impl TokenizerPayload {
    pub fn from_directory(directory: &Directory) -> Self {
        Self {
            metadata: directory
                .metadata
                .iter()
                .filter(|entry| {
                    entry.name.starts_with("tokenizer.")
                        && entry.name != "tokenizer.chat_template"
                        && entry.name != TEMPLATE_INDEX
                        && !entry.name.starts_with("tokenizer.chat_template.")
                })
                .cloned()
                .collect(),
        }
    }

    pub fn value(&self, name: &str) -> Option<&Value> {
        self.metadata
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| &entry.value)
    }
}

/// llama.cpp's index of a GGUF's named chat templates: the names of its
/// `tokenizer.chat_template.<name>` variants. The decision template is the
/// decision description's, not a chat variant, and is neither indexed nor
/// carried here.
const TEMPLATE_INDEX: &str = "tokenizer.chat_templates";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateSource {
    /// `default` for `tokenizer.chat_template`, otherwise the metadata suffix.
    pub name: String,
    pub source: String,
    pub provenance: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TemplatePayload {
    pub sources: Vec<TemplateSource>,
}

impl TemplatePayload {
    pub fn from_directory(directory: &Directory, provenance: &str) -> Result<Self, Error> {
        let has_unsuffixed_default = directory.value("tokenizer.chat_template").is_some();
        let mut sources = Vec::new();
        for entry in directory
            .metadata
            .iter()
            .filter(|entry| entry.name != DECISION_TEMPLATE)
        {
            let name = if entry.name == "tokenizer.chat_template" {
                "default"
            } else if let Some(name) = entry.name.strip_prefix("tokenizer.chat_template.") {
                name
            } else {
                continue;
            };
            if name.is_empty() {
                return Err(Error::Invalid(
                    "GGUF chat-template variant has an empty name".into(),
                ));
            }
            if name == "default"
                && entry.name != "tokenizer.chat_template"
                && has_unsuffixed_default
            {
                continue;
            }
            let source = entry
                .value
                .string()
                .ok_or_else(|| Error::Invalid("GGUF chat template must be text".into()))?;
            sources.push(TemplateSource {
                name: name.to_owned(),
                source: source.to_owned(),
                provenance: format!("{provenance}#{}", entry.name),
            });
        }
        sources.sort_by(|left, right| left.name.cmp(&right.name));
        if let Some(index) = directory.value(TEMPLATE_INDEX) {
            let Value::Array(Array::String(names)) = index else {
                return Err(Error::Invalid(format!("{TEMPLATE_INDEX} must list names")));
            };
            let mut listed = names.iter().collect::<Vec<&str>>();
            listed.sort_unstable();
            let named = sources
                .iter()
                .map(|source| source.name.as_str())
                .filter(|name| *name != "default")
                .collect::<Vec<_>>();
            if listed != named {
                return Err(Error::Invalid(format!(
                    "{TEMPLATE_INDEX} must list exactly the named chat templates"
                )));
            }
        }
        Ok(Self { sources })
    }
}
