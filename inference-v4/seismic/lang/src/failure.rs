//! A stopped source operation, identified by its existing checked event.
use crate::entry::{CheckReason, SemanticEventId, SemanticEventKind, SemanticFunction};
use crate::ids::{NodeId, StableFunctionId};
use crate::reference_math::ScalarFailure;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceFailureCause {
    Scalar(ScalarFailure),
    Check(CheckReason),
}
impl SourceFailureCause {
    pub fn message(&self) -> &str {
        match self {
            Self::Scalar(cause) => cause.message(),
            Self::Check(CheckReason::IndexBound) => "index out of bounds",
            Self::Check(CheckReason::RangeOrder) => "range endpoints out of order",
            Self::Check(CheckReason::Custom(message)) => message,
        }
    }
}

/// Canonical identity of an existing checked event. Its namespace is the
/// content-derived body; local positions come from that body's actual graph.
/// Fresh entry/program handles never alter this identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceEvent {
    body: StableFunctionId,
    position: [u32; 3],
}
impl SourceEvent {
    pub fn body(self) -> StableFunctionId {
        self.body
    }
    pub fn position(self) -> [u32; 3] {
        self.position
    }
    fn at(function: &SemanticFunction, event: SemanticEventId) -> Self {
        assert_eq!(function.id(), event.node().region().function());
        Self {
            body: function.stable(),
            position: [
                event.node().region().index() as u32,
                event.node().ordinal() as u32,
                event.ordinal(),
            ],
        }
    }
}

/// Current public failure semantics expose the event and cause, not a trace of
/// loop/participant occurrences. Native status combines those occurrences at
/// this same event. Diagnostic path/line information is separate display data.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceFailure {
    pub event: SourceEvent,
    pub cause: SourceFailureCause,
}
impl SourceFailure {
    pub fn at(function: &SemanticFunction, node: NodeId, cause: SourceFailureCause) -> Self {
        let event = function
            .node(node)
            .events()
            .iter()
            .find(|event| matches!(event.kind(), SemanticEventKind::MayFail))
            .expect("source failure operation must own its checked MayFail event")
            .id();
        Self {
            event: SourceEvent::at(function, event),
            cause,
        }
    }
}
impl std::fmt::Display for SourceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.cause.message())
    }
}

/// Terminal source execution. Service/device failures are outside this sum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceTermination<T, F = SourceFailure> {
    Returned(T),
    Failed(F),
}
