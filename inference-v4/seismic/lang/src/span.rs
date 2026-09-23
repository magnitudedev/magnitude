//! Byte spans and checker-internal diagnostics.

use crate::checked::DiagnosticRule;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Span {
        Span {
            start: start as u32,
            end: end as u32,
        }
    }

    pub fn to(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// A diagnostic of one source file, before it is located in the module's
/// source set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Diagnostic {
    pub span: Span,
    pub rule: DiagnosticRule,
    pub message: String,
}

impl Diagnostic {
    pub(crate) fn with_rule(
        rule: DiagnosticRule,
        span: Span,
        message: impl Into<String>,
    ) -> Diagnostic {
        Diagnostic {
            span,
            rule,
            message: message.into(),
        }
    }

    pub(crate) fn new(span: Span, message: impl Into<String>) -> Diagnostic {
        Diagnostic::with_rule(DiagnosticRule::Type, span, message)
    }
}
