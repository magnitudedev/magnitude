//! The protocol-neutral decision request: the superset both wire formats map
//! into. A request is data only; what a model serves of it is its type's,
//! checked when the request is prepared ([`crate::Decider::prepare`]).

use crate::json::Value;
use magnitude_chat::request::ImageInput;

/// One decision request: a state, the images it carries, and named
/// questions answered about it.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRequest {
    /// The state: text (a string), JSON, or absent ([`Value::Null`]). Its
    /// strings may place images ([`crate::json::Segment::Image`]).
    pub state: Value,
    /// Every distinct image of the request; segments and `front` name them
    /// by index. Each placement is another copy of the image to the model.
    pub images: Vec<ImageInput>,
    /// Images read before the state, in order (the compatibility `images`
    /// list and chat-message images).
    pub front: Vec<usize>,
    /// The questions, in request order.
    pub questions: Vec<Question>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    /// What to decide; absent ([`Value::Null`]) leaves the type's default.
    pub instructions: Value,
    pub kind: QuestionKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum QuestionKind {
    /// Whether the instructions hold: P(true). The descriptions say what
    /// true and false mean; absent ([`Value::Null`]) leaves the type's.
    Predicate { truthy: Value, falsy: Value },
    /// One of ordered options; with `multiple`, each option's own P(yes).
    Choice {
        options: Vec<ChoiceOption>,
        multiple: bool,
    },
    /// One of ordered levels, lowest first; each level is its description.
    Score { levels: Vec<Value> },
    /// Where one thing is in the picture.
    Box,
    /// Every instance of a thing in the picture, at most `max` (the type's
    /// default when absent).
    Boxes { max: Option<u32> },
    /// A span read from the state.
    Read { value: ReadValue },
    /// A number estimated over `[low, high]`, on a grid `step` apart when
    /// given.
    Estimate {
        low: f64,
        high: f64,
        step: Option<f64>,
        unit: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceOption {
    pub key: String,
    /// Absent ([`Value::Null`]) shows the option by its key alone.
    pub description: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReadValue {
    Text,
    Number { unit: Option<String> },
    Date,
}

/// A question kind's name, as capabilities and refusals report it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KindName {
    Predicate,
    Choice,
    MultipleChoice,
    Score,
    Box,
    Boxes,
    ReadText,
    ReadNumber,
    ReadDate,
    Estimate,
}

impl QuestionKind {
    pub fn name(&self) -> KindName {
        match self {
            Self::Predicate { .. } => KindName::Predicate,
            Self::Choice {
                multiple: false, ..
            } => KindName::Choice,
            Self::Choice { multiple: true, .. } => KindName::MultipleChoice,
            Self::Score { .. } => KindName::Score,
            Self::Box => KindName::Box,
            Self::Boxes { .. } => KindName::Boxes,
            Self::Read {
                value: ReadValue::Text,
            } => KindName::ReadText,
            Self::Read {
                value: ReadValue::Number { .. },
            } => KindName::ReadNumber,
            Self::Read {
                value: ReadValue::Date,
            } => KindName::ReadDate,
            Self::Estimate { .. } => KindName::Estimate,
        }
    }
}

impl KindName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Predicate => "predicate",
            Self::Choice => "choice",
            Self::MultipleChoice => "multiple_choice",
            Self::Score => "score",
            Self::Box => "box",
            Self::Boxes => "boxes",
            Self::ReadText => "text",
            Self::ReadNumber => "number",
            Self::ReadDate => "date",
            Self::Estimate => "estimate",
        }
    }
}

/// Where a request may place images.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Placement {
    /// Before the state ([`DecisionRequest::front`]).
    Front,
    /// Inside the state's strings.
    State,
    /// Inside a question's instructions.
    Instructions,
    /// Inside a question's option or level descriptions.
    Options,
}

impl Placement {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Front => "front",
            Self::State => "state",
            Self::Instructions => "instructions",
            Self::Options => "options",
        }
    }
}
