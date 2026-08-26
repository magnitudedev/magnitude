use serde::de::Error as _;
use serde_json::{Map, Value};

use crate::ImageInput;

use super::{InferenceRequestError, NonEmptyText};

macro_rules! non_empty_string_newtype {
    ($name:ident, $field:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn try_new(value: impl Into<String>) -> Result<Self, InferenceRequestError> {
                let value = value.into();
                if value.is_empty() {
                    return Err(InferenceRequestError::Empty { field: $field });
                }
                Ok(Self(value))
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(&self.0)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::try_new(value).map_err(D::Error::custom)
            }
        }
    };
}

non_empty_string_newtype!(ToolCallId, "tool call ID");
non_empty_string_newtype!(ToolName, "tool name");

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct JsonObject(Map<String, Value>);

impl JsonObject {
    #[must_use]
    pub fn new(value: Map<String, Value>) -> Self {
        Self(value)
    }

    #[must_use]
    pub fn as_map(&self) -> &Map<String, Value> {
        &self.0
    }

    #[must_use]
    pub fn into_map(self) -> Map<String, Value> {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    id: ToolCallId,
    name: ToolName,
    input: JsonObject,
}

impl ToolCall {
    #[must_use]
    pub fn new(id: ToolCallId, name: ToolName, input: JsonObject) -> Self {
        Self { id, name, input }
    }

    #[must_use]
    pub fn id(&self) -> &ToolCallId {
        &self.id
    }

    #[must_use]
    pub fn name(&self) -> &ToolName {
        &self.name
    }

    #[must_use]
    pub fn input(&self) -> &JsonObject {
        &self.input
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutcome {
    Success,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: NonEmptyText },
    Image { image: ImageInput },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolResult {
    outcome: ToolOutcome,
    content: Vec<ToolResultContent>,
}

impl ToolResult {
    #[must_use]
    pub fn new(outcome: ToolOutcome, content: Vec<ToolResultContent>) -> Self {
        Self { outcome, content }
    }

    #[must_use]
    pub fn outcome(&self) -> ToolOutcome {
        self.outcome
    }

    #[must_use]
    pub fn content(&self) -> &[ToolResultContent] {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolExchange {
    call: ToolCall,
    result: ToolResult,
}

impl ToolExchange {
    #[must_use]
    pub fn new(call: ToolCall, result: ToolResult) -> Self {
        Self { call, result }
    }

    #[must_use]
    pub fn call(&self) -> &ToolCall {
        &self.call
    }

    #[must_use]
    pub fn result(&self) -> &ToolResult {
        &self.result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_may_have_no_content() {
        let result = ToolResult::new(ToolOutcome::Success, Vec::new());
        assert!(result.content().is_empty());
    }
}
