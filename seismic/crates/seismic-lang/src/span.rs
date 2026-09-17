//! Byte spans and source positions.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Span {
        Span { start: start as u32, end: end as u32 }
    }

    pub fn to(self, other: Span) -> Span {
        Span { start: self.start.min(other.start), end: self.end.max(other.end) }
    }
}

/// Line and column (both 1-based) for a byte offset.
pub fn line_col(text: &str, offset: u32) -> (usize, usize) {
    let offset = (offset as usize).min(text.len());
    let before = &text[..offset];
    let line = before.matches('\n').count() + 1;
    let col = before.rfind('\n').map(|i| offset - i).unwrap_or(offset + 1);
    (line, col)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub span: Span,
    pub message: String,
}

impl Diagnostic {
    pub fn new(span: Span, message: impl Into<String>) -> Diagnostic {
        Diagnostic { span, message: message.into() }
    }

    pub fn render(&self, path: &str, text: &str) -> String {
        let (line, col) = line_col(text, self.span.start);
        let source_line = text.lines().nth(line - 1).unwrap_or("");
        let width = (self.span.end.saturating_sub(self.span.start)).max(1) as usize;
        format!(
            "{path}:{line}:{col}: {}\n  {}\n  {}{}",
            self.message,
            source_line,
            " ".repeat(col - 1),
            "^".repeat(width.min(source_line.len().saturating_sub(col - 1).max(1)))
        )
    }
}
