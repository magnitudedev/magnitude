use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use serde::de::Error as _;

use crate::{AutomaticReasoningBudget, NativeReasoningControls, NormalizedReasoningEffort};

use super::{InferenceContext, InferenceRequestError, JsonObject, ToolName};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferenceInvocation {
    model: InferenceModelSelector,
    request: InferenceRequest<ReasoningIntent>,
}

impl InferenceInvocation {
    #[must_use]
    pub fn new(model: InferenceModelSelector, request: InferenceRequest<ReasoningIntent>) -> Self {
        Self { model, request }
    }

    #[must_use]
    pub fn model(&self) -> &InferenceModelSelector {
        &self.model
    }

    #[must_use]
    pub fn request(&self) -> &InferenceRequest<ReasoningIntent> {
        &self.request
    }

    #[must_use]
    pub fn into_parts(self) -> (InferenceModelSelector, InferenceRequest<ReasoningIntent>) {
        (self.model, self.request)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(transparent)]
pub struct InferenceModelSelector(String);

impl InferenceModelSelector {
    pub fn try_new(value: impl Into<String>) -> Result<Self, InferenceRequestError> {
        let value = value.into();
        if value.is_empty() {
            return Err(InferenceRequestError::Empty {
                field: "inference model selector",
            });
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

impl<'de> serde::Deserialize<'de> for InferenceModelSelector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::try_new(String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferenceRequest<R> {
    context: InferenceContext,
    tools: ToolConfiguration,
    reasoning: R,
    output: OutputConstraint,
    generation: GenerationParameters,
    prompt_reuse: PromptReusePolicy,
}

pub type ResolvedInferenceRequest = InferenceRequest<ResolvedReasoning>;

impl<R> InferenceRequest<R> {
    #[must_use]
    pub fn new(
        context: InferenceContext,
        tools: ToolConfiguration,
        reasoning: R,
        output: OutputConstraint,
        generation: GenerationParameters,
        prompt_reuse: PromptReusePolicy,
    ) -> Self {
        Self {
            context,
            tools,
            reasoning,
            output,
            generation,
            prompt_reuse,
        }
    }

    #[must_use]
    pub fn context(&self) -> &InferenceContext {
        &self.context
    }

    #[must_use]
    pub fn tools(&self) -> &ToolConfiguration {
        &self.tools
    }

    #[must_use]
    pub fn reasoning(&self) -> &R {
        &self.reasoning
    }

    #[must_use]
    pub fn output(&self) -> &OutputConstraint {
        &self.output
    }

    #[must_use]
    pub fn generation(&self) -> &GenerationParameters {
        &self.generation
    }

    #[must_use]
    pub fn prompt_reuse(&self) -> PromptReusePolicy {
        self.prompt_reuse
    }

    #[must_use]
    pub fn map_reasoning<T>(self, map: impl FnOnce(R) -> T) -> InferenceRequest<T> {
        InferenceRequest {
            context: self.context,
            tools: self.tools,
            reasoning: map(self.reasoning),
            output: self.output,
            generation: self.generation,
            prompt_reuse: self.prompt_reuse,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningIntent {
    ModelDefault {
        template_args: BTreeMap<String, serde_json::Value>,
        budget: Option<NonZeroU32>,
    },
    Disabled {
        template_args: BTreeMap<String, serde_json::Value>,
    },
    Enabled {
        template_args: BTreeMap<String, serde_json::Value>,
        budget: Option<NonZeroU32>,
    },
    Effort {
        effort: NormalizedReasoningEffort,
        template_args: BTreeMap<String, serde_json::Value>,
        budget: Option<NonZeroU32>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedReasoning {
    effort: NormalizedReasoningEffort,
    controls: NativeReasoningControls,
    automatic_budget: AutomaticReasoningBudget,
    explicit_budget: Option<NonZeroU32>,
    template_fingerprint: String,
}

impl ResolvedReasoning {
    #[must_use]
    pub fn new(
        effort: NormalizedReasoningEffort,
        controls: NativeReasoningControls,
        automatic_budget: AutomaticReasoningBudget,
        explicit_budget: Option<NonZeroU32>,
        template_fingerprint: String,
    ) -> Self {
        Self {
            effort,
            controls,
            automatic_budget,
            explicit_budget,
            template_fingerprint,
        }
    }

    #[must_use]
    pub fn effort(&self) -> &NormalizedReasoningEffort {
        &self.effort
    }

    #[must_use]
    pub fn controls(&self) -> &NativeReasoningControls {
        &self.controls
    }

    #[must_use]
    pub fn automatic_budget(&self) -> &AutomaticReasoningBudget {
        &self.automatic_budget
    }

    #[must_use]
    pub fn explicit_budget(&self) -> Option<NonZeroU32> {
        self.explicit_budget
    }

    #[must_use]
    pub fn template_fingerprint(&self) -> &str {
        &self.template_fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    name: ToolName,
    description: Option<String>,
    input_schema: JsonObject,
}

impl ToolDefinition {
    #[must_use]
    pub fn new(name: ToolName, description: Option<String>, input_schema: JsonObject) -> Self {
        Self {
            name,
            description,
            input_schema,
        }
    }

    #[must_use]
    pub fn name(&self) -> &ToolName {
        &self.name
    }

    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    #[must_use]
    pub fn input_schema(&self) -> &JsonObject {
        &self.input_schema
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Disabled,
    Auto,
    Required,
    Specific {
        name: ToolName,
    },
    Allowed {
        names: Vec<ToolName>,
        required: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolParallelism {
    Sequential,
    Parallel,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ToolConfiguration {
    definitions: Vec<ToolDefinition>,
    choice: ToolChoice,
    parallelism: ToolParallelism,
}

impl ToolConfiguration {
    pub fn try_new(
        definitions: Vec<ToolDefinition>,
        choice: ToolChoice,
        parallelism: ToolParallelism,
    ) -> Result<Self, InferenceRequestError> {
        let mut names = BTreeSet::new();
        for definition in &definitions {
            if !names.insert(definition.name().as_str()) {
                return Err(InferenceRequestError::DuplicateToolName {
                    name: definition.name().as_str().to_owned(),
                });
            }
        }
        match &choice {
            ToolChoice::Required if definitions.is_empty() => {
                return Err(InferenceRequestError::RequiredToolsWithoutDefinitions);
            }
            ToolChoice::Specific { name } if !names.contains(name.as_str()) => {
                return Err(InferenceRequestError::UnknownToolName {
                    name: name.as_str().to_owned(),
                });
            }
            ToolChoice::Allowed { names: allowed, .. } => {
                if allowed.is_empty() {
                    return Err(InferenceRequestError::EmptyAllowedTools);
                }
                let mut seen = BTreeSet::new();
                for name in allowed {
                    if !seen.insert(name.as_str()) {
                        return Err(InferenceRequestError::DuplicateAllowedToolName {
                            name: name.as_str().to_owned(),
                        });
                    }
                    if !names.contains(name.as_str()) {
                        return Err(InferenceRequestError::UnknownToolName {
                            name: name.as_str().to_owned(),
                        });
                    }
                }
            }
            ToolChoice::Disabled
            | ToolChoice::Auto
            | ToolChoice::Required
            | ToolChoice::Specific { .. } => {}
        }
        Ok(Self {
            definitions,
            choice,
            parallelism,
        })
    }

    #[must_use]
    pub fn none() -> Self {
        Self {
            definitions: Vec::new(),
            choice: ToolChoice::Disabled,
            parallelism: ToolParallelism::Sequential,
        }
    }

    #[must_use]
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    #[must_use]
    pub fn choice(&self) -> &ToolChoice {
        &self.choice
    }

    #[must_use]
    pub fn parallelism(&self) -> ToolParallelism {
        self.parallelism
    }
}

impl<'de> serde::Deserialize<'de> for ToolConfiguration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Wire {
            definitions: Vec<ToolDefinition>,
            choice: ToolChoice,
            parallelism: ToolParallelism,
        }

        let wire = Wire::deserialize(deserializer)?;
        Self::try_new(wire.definitions, wire.choice, wire.parallelism).map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputConstraint {
    Text,
    JsonObject,
    JsonSchema { constraint: JsonSchemaConstraint },
    Grammar { constraint: GrammarConstraint },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JsonSchemaConstraint {
    name: String,
    schema: JsonObject,
    strict: bool,
}

impl JsonSchemaConstraint {
    #[must_use]
    pub fn new(name: String, schema: JsonObject, strict: bool) -> Self {
        Self {
            name,
            schema,
            strict,
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn schema(&self) -> &JsonObject {
        &self.schema
    }

    #[must_use]
    pub fn strict(&self) -> bool {
        self.strict
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct GrammarConstraint(String);

impl GrammarConstraint {
    pub fn try_new(grammar: String) -> Result<Self, InferenceRequestError> {
        if grammar.is_empty() {
            return Err(InferenceRequestError::Empty { field: "grammar" });
        }
        Ok(Self(grammar))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> serde::Deserialize<'de> for GrammarConstraint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::try_new(String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GenerationParameters {
    max_output_tokens: NonZeroU32,
    sampling: SamplingParameters,
    stop_sequences: Vec<StopSequence>,
    end_of_generation: EndOfGenerationPolicy,
}

impl GenerationParameters {
    #[must_use]
    pub fn new(
        max_output_tokens: NonZeroU32,
        sampling: SamplingParameters,
        stop_sequences: Vec<StopSequence>,
        end_of_generation: EndOfGenerationPolicy,
    ) -> Self {
        Self {
            max_output_tokens,
            sampling,
            stop_sequences,
            end_of_generation,
        }
    }

    #[must_use]
    pub fn max_output_tokens(&self) -> NonZeroU32 {
        self.max_output_tokens
    }

    #[must_use]
    pub fn sampling(&self) -> &SamplingParameters {
        &self.sampling
    }

    #[must_use]
    pub fn stop_sequences(&self) -> &[StopSequence] {
        &self.stop_sequences
    }

    #[must_use]
    pub fn end_of_generation(&self) -> EndOfGenerationPolicy {
        self.end_of_generation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SamplingParameters {
    temperature: Temperature,
    top_p: TopP,
    seed: u32,
}

impl SamplingParameters {
    #[must_use]
    pub fn new(temperature: Temperature, top_p: TopP, seed: u32) -> Self {
        Self {
            temperature,
            top_p,
            seed,
        }
    }

    #[must_use]
    pub fn temperature(&self) -> Temperature {
        self.temperature
    }

    #[must_use]
    pub fn top_p(&self) -> TopP {
        self.top_p
    }

    #[must_use]
    pub fn seed(&self) -> u32 {
        self.seed
    }
}

macro_rules! bounded_float {
    ($name:ident, $field:literal, $minimum:expr, $maximum:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
        #[serde(transparent)]
        pub struct $name(f32);

        impl $name {
            pub fn try_new(value: f32) -> Result<Self, InferenceRequestError> {
                if !value.is_finite() || !($minimum..=$maximum).contains(&value) {
                    return Err(InferenceRequestError::OutOfRange {
                        field: $field,
                        minimum: $minimum,
                        maximum: $maximum,
                    });
                }
                Ok(Self(value))
            }

            #[must_use]
            pub fn get(self) -> f32 {
                self.0
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                Self::try_new(f32::deserialize(deserializer)?).map_err(D::Error::custom)
            }
        }
    };
}

bounded_float!(Temperature, "temperature", 0.0, 2.0);
bounded_float!(TopP, "top_p", 0.0, 1.0);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StopSequence(String);

impl StopSequence {
    pub fn try_new(value: String) -> Result<Self, InferenceRequestError> {
        if value.is_empty() {
            return Err(InferenceRequestError::Empty {
                field: "stop sequence",
            });
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl serde::Serialize for StopSequence {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> serde::Deserialize<'de> for StopSequence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::try_new(String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndOfGenerationPolicy {
    StopAtModelEnd,
    IgnoreModelEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptReusePolicy {
    Disabled,
    Allowed,
}

#[cfg(test)]
mod tests {
    use serde_json::Map;

    use super::*;

    fn definition(name: &str) -> ToolDefinition {
        ToolDefinition::new(
            ToolName::try_new(name).expect("valid name"),
            None,
            JsonObject::new(Map::new()),
        )
    }

    #[test]
    fn bounded_sampling_values_reject_invalid_values() {
        assert!(Temperature::try_new(f32::NAN).is_err());
        assert!(Temperature::try_new(2.1).is_err());
        assert!(TopP::try_new(-0.1).is_err());
        assert_eq!(TopP::try_new(1.0).expect("valid").get(), 1.0);
    }

    #[test]
    fn tool_configuration_rejects_duplicate_names() {
        assert!(matches!(
            ToolConfiguration::try_new(
                vec![definition("search"), definition("search")],
                ToolChoice::Auto,
                ToolParallelism::Parallel,
            ),
            Err(InferenceRequestError::DuplicateToolName { .. })
        ));
    }
}
