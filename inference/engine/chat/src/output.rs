//! Protocol-neutral generation output: the semantic event stream every
//! protocol frames, terminal execution facts, and the journal that assembles
//! non-streaming output from exactly the same events.
use super::request::ToolCall;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Semantic output in publication order. A stream starts once, then moves
/// through reasoning, text and tool calls without returning to an earlier phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputEvent {
    Started,
    ReasoningDelta(String),
    TextDelta(String),
    ToolCallStarted {
        index: usize,
        id: String,
        name: String,
    },
    ToolInputDelta {
        index: usize,
        fragment: String,
    },
    ToolCallFinished {
        index: usize,
    },
}

/// Request progress before output: host preparation, engine admission,
/// prompt processing, then generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    Preparing,
    Queued,
    Prefill {
        completed_tokens: u64,
        total_tokens: u64,
        cached_tokens: u64,
    },
    Generating,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Termination {
    Natural,
    StopSequence(String),
    OutputLimit,
    ToolCalls,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_output_tokens: u64,
}

/// Request timing. Prompt and decode durations are engine-measured physical
/// work at completion; progressive snapshots report host-observed elapsed
/// time. Draft counters are present only when a speculative method ran.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenerationTimings {
    pub prompt_ms: f64,
    pub decode_ms: f64,
    pub time_to_first_token_ms: f64,
    pub sampler_ms: f64,
    pub parser_ms: f64,
    pub draft_tokens: u64,
    pub accepted_draft_tokens: u64,
}

/// Cumulative request progress attached to output when per-token timings are
/// requested.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TimingSnapshot {
    pub cached_prompt_tokens: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub timings: GenerationTimings,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Completion {
    pub usage: TokenUsage,
    pub termination: Termination,
    pub timings: GenerationTimings,
}

impl Completion {
    pub fn snapshot(&self) -> TimingSnapshot {
        TimingSnapshot {
            cached_prompt_tokens: self.usage.cached_input_tokens,
            prompt_tokens: self.usage.input_tokens,
            generated_tokens: self.usage.output_tokens,
            timings: self.timings,
        }
    }
}

/// Aggregate output built only by [`OutputJournal`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Output {
    pub reasoning: Option<String>,
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JournalError {
    DuplicateStart,
    NotStarted,
    PhaseRegression,
    UnexpectedToolIndex { index: usize },
    ToolCallNotOpen { index: usize },
    ToolCallAlreadyOpen,
    InvalidToolInput { index: usize, message: String },
    OpenToolCall,
    MissingStart,
}

impl std::fmt::Display for JournalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateStart => formatter.write_str("output stream started more than once"),
            Self::NotStarted => formatter.write_str("output event arrived before stream start"),
            Self::PhaseRegression => {
                formatter.write_str("output stream returned to an earlier semantic phase")
            }
            Self::UnexpectedToolIndex { index } => {
                write!(formatter, "tool call index {index} is not the next ordered call")
            }
            Self::ToolCallNotOpen { index } => write!(formatter, "tool call index {index} is not open"),
            Self::ToolCallAlreadyOpen => formatter.write_str("another tool call is already open"),
            Self::InvalidToolInput { index, message } => write!(
                formatter,
                "tool input for call index {index} is not a JSON object: {message}"
            ),
            Self::OpenToolCall => formatter.write_str("output stream terminated with an open tool call"),
            Self::MissingStart => formatter.write_str("output stream terminated before it started"),
        }
    }
}

impl std::error::Error for JournalError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    Start,
    Reasoning,
    Text,
    Tools,
}

#[derive(Debug)]
struct OpenCall {
    id: String,
    name: String,
    input: String,
}

/// Validates the semantic stream and assembles its aggregate output.
#[derive(Debug, Default)]
pub struct OutputJournal {
    phase: Option<Phase>,
    reasoning: String,
    text: String,
    calls: Vec<Option<ToolCall>>,
    open: BTreeMap<usize, OpenCall>,
}

impl OutputJournal {
    pub fn push(&mut self, event: &OutputEvent) -> Result<(), JournalError> {
        match event {
            OutputEvent::Started => {
                if self.phase.is_some() {
                    return Err(JournalError::DuplicateStart);
                }
                self.phase = Some(Phase::Start);
            }
            OutputEvent::ReasoningDelta(text) => {
                self.advance(Phase::Reasoning)?;
                self.reasoning.push_str(text);
            }
            OutputEvent::TextDelta(text) => {
                self.advance(Phase::Text)?;
                self.text.push_str(text);
            }
            OutputEvent::ToolCallStarted { index, id, name } => {
                self.advance(Phase::Tools)?;
                if *index != self.calls.len() {
                    return Err(JournalError::UnexpectedToolIndex { index: *index });
                }
                let call = OpenCall {
                    id: id.clone(),
                    name: name.clone(),
                    input: String::new(),
                };
                if self.open.insert(*index, call).is_some() {
                    return Err(JournalError::ToolCallAlreadyOpen);
                }
                self.calls.push(None);
            }
            OutputEvent::ToolInputDelta { index, fragment } => {
                let call = self
                    .open
                    .get_mut(index)
                    .ok_or(JournalError::ToolCallNotOpen { index: *index })?;
                call.input.push_str(fragment);
            }
            OutputEvent::ToolCallFinished { index } => {
                let call = self
                    .open
                    .remove(index)
                    .ok_or(JournalError::ToolCallNotOpen { index: *index })?;
                let arguments = serde_json::from_str::<Map<String, Value>>(&call.input)
                    .map_err(|error| JournalError::InvalidToolInput {
                        index: *index,
                        message: error.to_string(),
                    })?;
                self.calls[*index] = Some(ToolCall {
                    id: call.id,
                    name: call.name,
                    arguments,
                });
            }
        }
        Ok(())
    }

    pub fn finish(self) -> Result<Output, JournalError> {
        if self.phase.is_none() {
            return Err(JournalError::MissingStart);
        }
        if !self.open.is_empty() {
            return Err(JournalError::OpenToolCall);
        }
        let tool_calls = self
            .calls
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(JournalError::OpenToolCall)?;
        Ok(Output {
            reasoning: (!self.reasoning.is_empty()).then_some(self.reasoning),
            text: (!self.text.is_empty()).then_some(self.text),
            tool_calls,
        })
    }

    fn advance(&mut self, next: Phase) -> Result<(), JournalError> {
        let current = self.phase.ok_or(JournalError::NotStarted)?;
        if next < current {
            return Err(JournalError::PhaseRegression);
        }
        self.phase = Some(next);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_constructs_the_fixed_output_shape() {
        let mut journal = OutputJournal::default();
        for event in [
            OutputEvent::Started,
            OutputEvent::ReasoningDelta("think".into()),
            OutputEvent::TextDelta("answer".into()),
            OutputEvent::ToolCallStarted {
                index: 0,
                id: "call_1".into(),
                name: "search".into(),
            },
            OutputEvent::ToolInputDelta {
                index: 0,
                fragment: "{\"q\":\"rust\"}".into(),
            },
            OutputEvent::ToolCallFinished { index: 0 },
        ] {
            journal.push(&event).unwrap();
        }
        let output = journal.finish().unwrap();
        assert_eq!(output.reasoning.as_deref(), Some("think"));
        assert_eq!(output.text.as_deref(), Some("answer"));
        assert_eq!(output.tool_calls[0].arguments["q"], "rust");
    }

    #[test]
    fn journal_rejects_phase_regression_and_incomplete_calls() {
        let mut journal = OutputJournal::default();
        journal.push(&OutputEvent::Started).unwrap();
        journal.push(&OutputEvent::TextDelta("answer".into())).unwrap();
        assert_eq!(
            journal.push(&OutputEvent::ReasoningDelta("late".into())),
            Err(JournalError::PhaseRegression)
        );
        let mut journal = OutputJournal::default();
        journal.push(&OutputEvent::Started).unwrap();
        journal
            .push(&OutputEvent::ToolCallStarted {
                index: 0,
                id: "call".into(),
                name: "tool".into(),
            })
            .unwrap();
        assert_eq!(journal.finish(), Err(JournalError::OpenToolCall));
    }

    #[test]
    fn journal_accepts_interleaved_parallel_tool_input() {
        let mut journal = OutputJournal::default();
        for event in [
            OutputEvent::Started,
            OutputEvent::ToolCallStarted {
                index: 0,
                id: "first".into(),
                name: "one".into(),
            },
            OutputEvent::ToolCallStarted {
                index: 1,
                id: "second".into(),
                name: "two".into(),
            },
            OutputEvent::ToolInputDelta {
                index: 1,
                fragment: "{}".into(),
            },
            OutputEvent::ToolInputDelta {
                index: 0,
                fragment: "{}".into(),
            },
            OutputEvent::ToolCallFinished { index: 1 },
            OutputEvent::ToolCallFinished { index: 0 },
        ] {
            journal.push(&event).unwrap();
        }
        let output = journal.finish().unwrap();
        assert_eq!(output.tool_calls[0].id, "first");
        assert_eq!(output.tool_calls[1].id, "second");
    }
}
