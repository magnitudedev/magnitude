use std::collections::BTreeMap;

use crate::GenerationMetrics;

use super::{
    InferenceRequestError, JsonObject, NonEmptyText, StopSequence, ToolCall, ToolCallId, ToolName,
};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferenceOutput {
    reasoning: Option<NonEmptyText>,
    text: Option<NonEmptyText>,
    tool_calls: Vec<ToolCall>,
}

impl InferenceOutput {
    #[must_use]
    pub fn new(
        reasoning: Option<NonEmptyText>,
        text: Option<NonEmptyText>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            reasoning,
            text,
            tool_calls,
        }
    }

    #[must_use]
    pub fn reasoning(&self) -> Option<&NonEmptyText> {
        self.reasoning.as_ref()
    }

    #[must_use]
    pub fn text(&self) -> Option<&NonEmptyText> {
        self.text.as_ref()
    }

    #[must_use]
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferenceResult {
    output: InferenceOutput,
    usage: TokenUsage,
    termination: Termination,
    metrics: GenerationMetrics,
}

/// Terminal execution facts emitted after the canonical output event stream closes.
///
/// Output is intentionally absent: `OutputJournal` is the sole constructor of aggregate output.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferenceCompletion {
    usage: TokenUsage,
    termination: Termination,
    metrics: GenerationMetrics,
}

impl InferenceCompletion {
    #[must_use]
    pub fn new(usage: TokenUsage, termination: Termination, metrics: GenerationMetrics) -> Self {
        Self {
            usage,
            termination,
            metrics,
        }
    }

    #[must_use]
    pub fn into_result(self, output: InferenceOutput) -> InferenceResult {
        InferenceResult::new(output, self.usage, self.termination, self.metrics)
    }

    #[must_use]
    pub fn usage(&self) -> &TokenUsage {
        &self.usage
    }

    #[must_use]
    pub fn termination(&self) -> &Termination {
        &self.termination
    }

    #[must_use]
    pub fn metrics(&self) -> &GenerationMetrics {
        &self.metrics
    }
}

impl InferenceResult {
    #[must_use]
    pub fn new(
        output: InferenceOutput,
        usage: TokenUsage,
        termination: Termination,
        metrics: GenerationMetrics,
    ) -> Self {
        Self {
            output,
            usage,
            termination,
            metrics,
        }
    }

    #[must_use]
    pub fn output(&self) -> &InferenceOutput {
        &self.output
    }

    #[must_use]
    pub fn usage(&self) -> &TokenUsage {
        &self.usage
    }

    #[must_use]
    pub fn termination(&self) -> &Termination {
        &self.termination
    }

    #[must_use]
    pub fn metrics(&self) -> &GenerationMetrics {
        &self.metrics
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    input_tokens: u64,
    cached_input_tokens: u64,
    output_tokens: u64,
    reasoning_output_tokens: u64,
}

impl TokenUsage {
    #[must_use]
    pub fn new(
        input_tokens: u64,
        cached_input_tokens: u64,
        output_tokens: u64,
        reasoning_output_tokens: u64,
    ) -> Self {
        Self {
            input_tokens,
            cached_input_tokens,
            output_tokens,
            reasoning_output_tokens,
        }
    }

    #[must_use]
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens
    }

    #[must_use]
    pub fn cached_input_tokens(&self) -> u64 {
        self.cached_input_tokens
    }

    #[must_use]
    pub fn output_tokens(&self) -> u64 {
        self.output_tokens
    }

    #[must_use]
    pub fn reasoning_output_tokens(&self) -> u64 {
        self.reasoning_output_tokens
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Termination {
    Natural,
    StopSequence { sequence: StopSequence },
    OutputLimit,
    ToolCalls,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceOutputEvent {
    Started,
    ReasoningDelta {
        text: NonEmptyText,
    },
    TextDelta {
        text: NonEmptyText,
    },
    ToolCallStarted {
        index: usize,
        id: ToolCallId,
        name: ToolName,
    },
    ToolInputDelta {
        index: usize,
        json_fragment: NonEmptyText,
    },
    ToolCallFinished {
        index: usize,
    },
}

/// Semantic output and execution progress share a transport without making progress part of the
/// generated output model.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceObservationEvent {
    Output { event: InferenceOutputEvent },
    Progress { progress: crate::InferenceProgress },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferenceObservation {
    event: InferenceObservationEvent,
    timings: Option<crate::GenerationSnapshot>,
}

impl InferenceObservation {
    #[must_use]
    pub fn new(
        event: InferenceObservationEvent,
        timings: Option<crate::GenerationSnapshot>,
    ) -> Self {
        Self { event, timings }
    }

    #[must_use]
    pub fn event(&self) -> &InferenceObservationEvent {
        &self.event
    }

    #[must_use]
    pub fn timings(&self) -> Option<&crate::GenerationSnapshot> {
        self.timings.as_ref()
    }

    #[must_use]
    pub fn into_parts(self) -> (InferenceObservationEvent, Option<crate::GenerationSnapshot>) {
        (self.event, self.timings)
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum OutputJournalError {
    #[error("output stream started more than once")]
    DuplicateStart,
    #[error("output event arrived before stream start")]
    NotStarted,
    #[error("output stream returned to an earlier semantic phase")]
    PhaseRegression,
    #[error("tool call index {index} is not the next ordered call")]
    UnexpectedToolIndex { index: usize },
    #[error("tool call index {index} is not open")]
    ToolCallNotOpen { index: usize },
    #[error("another tool call is already open")]
    ToolCallAlreadyOpen,
    #[error("tool input for call index {index} is not a JSON object: {message}")]
    InvalidToolInput { index: usize, message: String },
    #[error("output stream terminated with an open tool call")]
    OpenToolCall,
    #[error("output stream terminated before it started")]
    MissingStart,
    #[error(transparent)]
    InvalidOutput(#[from] InferenceRequestError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OutputPhase {
    Start,
    Reasoning,
    Text,
    Tools,
}

#[derive(Debug)]
struct OpenToolCall {
    id: ToolCallId,
    name: ToolName,
    input: String,
}

#[derive(Debug, Default)]
pub struct OutputJournal {
    started: bool,
    phase: Option<OutputPhase>,
    reasoning: String,
    text: String,
    calls: Vec<Option<ToolCall>>,
    open_calls: BTreeMap<usize, OpenToolCall>,
}

impl OutputJournal {
    pub fn push(&mut self, event: &InferenceOutputEvent) -> Result<(), OutputJournalError> {
        match event {
            InferenceOutputEvent::Started => {
                if self.started {
                    return Err(OutputJournalError::DuplicateStart);
                }
                self.started = true;
                self.phase = Some(OutputPhase::Start);
            }
            InferenceOutputEvent::ReasoningDelta { text } => {
                self.advance(OutputPhase::Reasoning)?;
                self.reasoning.push_str(text.as_str());
            }
            InferenceOutputEvent::TextDelta { text } => {
                self.advance(OutputPhase::Text)?;
                self.text.push_str(text.as_str());
            }
            InferenceOutputEvent::ToolCallStarted { index, id, name } => {
                self.advance(OutputPhase::Tools)?;
                if *index != self.calls.len() {
                    return Err(OutputJournalError::UnexpectedToolIndex { index: *index });
                }
                if self
                    .open_calls
                    .insert(
                        *index,
                        OpenToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            input: String::new(),
                        },
                    )
                    .is_some()
                {
                    return Err(OutputJournalError::ToolCallAlreadyOpen);
                }
                self.calls.push(None);
            }
            InferenceOutputEvent::ToolInputDelta {
                index,
                json_fragment,
            } => {
                let Some(call) = self.open_calls.get_mut(index) else {
                    return Err(OutputJournalError::ToolCallNotOpen { index: *index });
                };
                call.input.push_str(json_fragment.as_str());
            }
            InferenceOutputEvent::ToolCallFinished { index } => {
                let Some(call) = self.open_calls.remove(index) else {
                    return Err(OutputJournalError::ToolCallNotOpen { index: *index });
                };
                let input = serde_json::from_str::<JsonObject>(&call.input).map_err(|error| {
                    OutputJournalError::InvalidToolInput {
                        index: *index,
                        message: error.to_string(),
                    }
                })?;
                self.calls[*index] = Some(ToolCall::new(call.id, call.name, input));
            }
        }
        Ok(())
    }

    pub fn finish(self) -> Result<InferenceOutput, OutputJournalError> {
        if !self.started {
            return Err(OutputJournalError::MissingStart);
        }
        if !self.open_calls.is_empty() {
            return Err(OutputJournalError::OpenToolCall);
        }
        let reasoning = optional_text(self.reasoning, "reasoning output")?;
        let text = optional_text(self.text, "text output")?;
        let calls = self
            .calls
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(OutputJournalError::OpenToolCall)?;
        Ok(InferenceOutput::new(reasoning, text, calls))
    }

    fn advance(&mut self, next: OutputPhase) -> Result<(), OutputJournalError> {
        if !self.started {
            return Err(OutputJournalError::NotStarted);
        }
        if self.phase.is_some_and(|current| next < current) {
            return Err(OutputJournalError::PhaseRegression);
        }
        self.phase = Some(next);
        Ok(())
    }
}

fn optional_text(
    value: String,
    field: &'static str,
) -> Result<Option<NonEmptyText>, InferenceRequestError> {
    if value.is_empty() {
        Ok(None)
    } else {
        NonEmptyText::try_new(value, field).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(value: &str) -> NonEmptyText {
        NonEmptyText::try_new(value, "test text").expect("nonempty test text")
    }

    #[test]
    fn journal_constructs_the_fixed_output_shape() {
        let mut journal = OutputJournal::default();
        for event in [
            InferenceOutputEvent::Started,
            InferenceOutputEvent::ReasoningDelta {
                text: text("think"),
            },
            InferenceOutputEvent::TextDelta {
                text: text("answer"),
            },
            InferenceOutputEvent::ToolCallStarted {
                index: 0,
                id: ToolCallId::try_new("call_1").expect("valid"),
                name: ToolName::try_new("search").expect("valid"),
            },
            InferenceOutputEvent::ToolInputDelta {
                index: 0,
                json_fragment: text("{\"q\":\"rust\"}"),
            },
            InferenceOutputEvent::ToolCallFinished { index: 0 },
        ] {
            journal.push(&event).expect("valid event");
        }
        let output = journal.finish().expect("valid output");
        assert_eq!(output.reasoning().expect("reasoning").as_str(), "think");
        assert_eq!(output.text().expect("text").as_str(), "answer");
        assert_eq!(output.tool_calls()[0].input().as_map()["q"], "rust");
    }

    #[test]
    fn journal_rejects_phase_regression_and_incomplete_calls() {
        let mut journal = OutputJournal::default();
        journal.push(&InferenceOutputEvent::Started).expect("start");
        journal
            .push(&InferenceOutputEvent::TextDelta {
                text: text("answer"),
            })
            .expect("text");
        assert_eq!(
            journal.push(&InferenceOutputEvent::ReasoningDelta { text: text("late") }),
            Err(OutputJournalError::PhaseRegression)
        );

        let mut journal = OutputJournal::default();
        journal.push(&InferenceOutputEvent::Started).expect("start");
        journal
            .push(&InferenceOutputEvent::ToolCallStarted {
                index: 0,
                id: ToolCallId::try_new("call").expect("valid"),
                name: ToolName::try_new("tool").expect("valid"),
            })
            .expect("tool start");
        assert!(matches!(
            journal.finish(),
            Err(OutputJournalError::OpenToolCall)
        ));
    }

    #[test]
    fn journal_accepts_interleaved_parallel_tool_input() {
        let mut journal = OutputJournal::default();
        for event in [
            InferenceOutputEvent::Started,
            InferenceOutputEvent::ToolCallStarted {
                index: 0,
                id: ToolCallId::try_new("first").expect("valid"),
                name: ToolName::try_new("one").expect("valid"),
            },
            InferenceOutputEvent::ToolCallStarted {
                index: 1,
                id: ToolCallId::try_new("second").expect("valid"),
                name: ToolName::try_new("two").expect("valid"),
            },
            InferenceOutputEvent::ToolInputDelta {
                index: 1,
                json_fragment: text("{}"),
            },
            InferenceOutputEvent::ToolInputDelta {
                index: 0,
                json_fragment: text("{}"),
            },
            InferenceOutputEvent::ToolCallFinished { index: 1 },
            InferenceOutputEvent::ToolCallFinished { index: 0 },
        ] {
            journal.push(&event).expect("valid parallel event");
        }
        let output = journal.finish().expect("valid output");
        let calls = output.tool_calls();
        assert_eq!(calls[0].id().as_str(), "first");
        assert_eq!(calls[1].id().as_str(), "second");
    }

    #[test]
    fn canonical_deltas_cannot_be_empty() {
        for field in ["reasoning", "text", "tool input"] {
            assert_eq!(
                NonEmptyText::try_new("", field),
                Err(InferenceRequestError::Empty { field })
            );
        }
    }
}
