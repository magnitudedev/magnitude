use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryRule {
    Causal,
    Indivisible,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSpan {
    pub start: usize,
    pub end: usize,
    pub identity: String,
    pub boundaries: BoundaryRule,
    pub language_history: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawInputLayout")]
pub struct InputLayout {
    count: usize,
    spans: Vec<InputSpan>,
}

/// Deserialization revalidates through [`InputLayout::new`].
#[derive(Deserialize)]
struct RawInputLayout {
    count: usize,
    spans: Vec<InputSpan>,
}

impl TryFrom<RawInputLayout> for InputLayout {
    type Error = String;
    fn try_from(raw: RawInputLayout) -> Result<Self, String> {
        Self::new(raw.count, raw.spans)
    }
}

impl InputLayout {
    pub fn new(count: usize, spans: Vec<InputSpan>) -> Result<Self, String> {
        if count > i32::MAX as usize {
            return Err("input exceeds logical position range".into());
        }
        let mut previous = 0;
        for span in &spans {
            if span.identity.is_empty()
                || span.start < previous
                || span.end <= span.start
                || span.end > count
            {
                return Err("conditioned spans must be identified, ordered, disjoint, and inside the prompt".into());
            }
            previous = span.end;
        }
        Ok(Self { count, spans })
    }
    pub fn count(&self) -> usize {
        self.count
    }
    pub fn spans(&self) -> &[InputSpan] {
        &self.spans
    }
    pub fn boundary(&self, position: usize) -> bool {
        position <= i32::MAX as usize
            && !self.spans.iter().any(|span| {
                span.boundaries == BoundaryRule::Indivisible
                    && span.start < position
                    && position < span.end
            })
    }
    /// The first position at or after `position` a chunk may end at: the
    /// end of the indivisible span holding it inside, or itself.
    pub fn next_boundary(&self, position: usize) -> usize {
        self.spans
            .iter()
            .find(|span| {
                span.boundaries == BoundaryRule::Indivisible
                    && span.start < position
                    && position < span.end
            })
            .map_or(position, |span| span.end)
    }

    /// The service allowance is soft. A physical capacity check still applies to
    /// the returned unit, including when the first indivisible span exceeds it.
    pub fn chunk_end(
        &self,
        start: usize,
        available_end: usize,
        allowance: usize,
    ) -> Result<usize, String> {
        if !self.boundary(start)
            || !self.boundary(available_end)
            || allowance == 0
            || available_end <= start
        {
            return Err(
                "chunk requires legal boundaries, available input, and positive allowance".into(),
            );
        }
        let mut end = start.saturating_add(allowance).min(available_end);
        for span in &self.spans {
            if span.boundaries == BoundaryRule::Indivisible && span.start < end && end < span.end {
                end = if span.start > start {
                    span.start
                } else {
                    span.end
                };
                break;
            }
        }
        Ok(end)
    }
    pub fn language(&self, position: usize) -> Result<bool, String> {
        if position > i32::MAX as usize {
            return Err("invalid language history position".into());
        }
        Ok(self
            .spans
            .iter()
            .filter(|s| s.start <= position && position < s.end)
            .all(|s| s.language_history))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk ends past an indivisible span; a causal span splits anywhere.
    #[test]
    fn indivisible_spans_bound_chunk_ends() {
        let span = |start, end, boundaries| InputSpan {
            start,
            end,
            identity: format!("{start}"),
            boundaries,
            language_history: false,
        };
        let layout = InputLayout::new(
            12,
            vec![
                span(2, 6, BoundaryRule::Indivisible),
                span(8, 10, BoundaryRule::Causal),
            ],
        )
        .unwrap();
        assert_eq!(
            (0..=12)
                .map(|position| layout.next_boundary(position))
                .collect::<Vec<_>>(),
            [0, 1, 2, 6, 6, 6, 6, 7, 8, 9, 10, 11, 12]
        );
    }
}
