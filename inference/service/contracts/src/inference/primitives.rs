use serde::de::Error as _;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InferenceRequestError {
    #[error("{field} must not be empty")]
    Empty { field: &'static str },
    #[error("{field} must be between {minimum} and {maximum}, inclusive")]
    OutOfRange {
        field: &'static str,
        minimum: f32,
        maximum: f32,
    },
    #[error("duplicate tool name: {name}")]
    DuplicateToolName { name: String },
    #[error("tool choice references unknown tool: {name}")]
    UnknownToolName { name: String },
    #[error("allowed tool choice must contain at least one tool")]
    EmptyAllowedTools,
    #[error("duplicate allowed tool name: {name}")]
    DuplicateAllowedToolName { name: String },
    #[error("required tool choice needs at least one tool definition")]
    RequiredToolsWithoutDefinitions,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonEmptyText(String);

impl NonEmptyText {
    pub fn try_new(
        value: impl Into<String>,
        field: &'static str,
    ) -> Result<Self, InferenceRequestError> {
        let value = value.into();
        if value.is_empty() {
            return Err(InferenceRequestError::Empty { field });
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl serde::Serialize for NonEmptyText {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> serde::Deserialize<'de> for NonEmptyText {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::try_new(value, "text").map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonEmptyVec<T>(Vec<T>);

impl<T> NonEmptyVec<T> {
    pub fn try_new(values: Vec<T>, field: &'static str) -> Result<Self, InferenceRequestError> {
        if values.is_empty() {
            return Err(InferenceRequestError::Empty { field });
        }
        Ok(Self(values))
    }

    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.0
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T: serde::Serialize> serde::Serialize for NonEmptyVec<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for NonEmptyVec<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let values = Vec::<T>::deserialize(deserializer)?;
        Self::try_new(values, "collection").map_err(D::Error::custom)
    }
}
