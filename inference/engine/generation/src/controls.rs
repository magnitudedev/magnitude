//! Request-local generation controls that restrict selection independently of
//! any output grammar: end-of-generation suppression and the hard reasoning
//! budget. Both compose with the request's grammar constraint as one
//! [`Constraint`], so every selection path (prefill completion, forced runs,
//! proposals and verification) observes them without a second mechanism.
use crate::{Constraint, TokenId};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, sync::Arc};

/// Whether the model's end-of-generation tokens terminate the request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndOfGeneration {
    /// A selected stop token finishes the request.
    #[default]
    Stop,
    /// Stop tokens are never selected; the request ends at its token or
    /// context limit (`ignore_eos`).
    Suppress,
}

/// An engine-enforced cap on reasoning tokens. Reasoning is delimited by the
/// template's start and end tag token sequences; once `tokens` reasoning
/// tokens have been accepted without the model closing reasoning itself, the
/// end sequence is forced. There is no automatic mapping from reasoning effort
/// to a budget: a budget exists only when the caller requested one.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningBudget {
    pub tokens: u32,
    pub start: Vec<TokenId>,
    pub end: Vec<TokenId>,
    /// The prompt already opened reasoning (the generation prefix ends with the
    /// start tag), so counting begins with the first generated token.
    pub open: bool,
}

impl ReasoningBudget {
    pub fn validate(&self, vocabulary: usize) -> Result<(), String> {
        if self.tokens == 0
            || self.start.is_empty()
            || self.end.is_empty()
            || self
                .start
                .iter()
                .chain(&self.end)
                .any(|token| token.0 as usize >= vocabulary)
        {
            return Err("reasoning budget requires positive tokens and in-vocabulary tags".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Reasoning has not started; the start tag may still open it.
    Before,
    /// Reasoning is open with this many counted tokens.
    Reasoning(u32),
    /// The budget is spent; this many end-tag tokens were already forced.
    Closing(usize),
    /// Reasoning closed; the budget no longer restricts selection.
    After,
}

#[derive(Clone)]
struct Budget {
    spec: Arc<ReasoningBudget>,
    phase: Phase,
    /// Recent tokens, bounded by the longest tag, for tag recognition.
    recent: Vec<TokenId>,
}

impl Budget {
    fn new(spec: ReasoningBudget) -> Self {
        let phase = if spec.open {
            Phase::Reasoning(0)
        } else {
            Phase::Before
        };
        Self {
            spec: Arc::new(spec),
            phase,
            recent: Vec::new(),
        }
    }

    fn forced_token(&self) -> Option<TokenId> {
        match self.phase {
            Phase::Closing(index) => Some(self.spec.end[index]),
            _ => None,
        }
    }

    fn accept(&mut self, token: TokenId) -> Result<(), String> {
        let window = self.spec.start.len().max(self.spec.end.len());
        self.recent.push(token);
        if self.recent.len() > window {
            self.recent.remove(0);
        }
        self.phase = match self.phase {
            Phase::Before if self.recent.ends_with(&self.spec.start) => Phase::Reasoning(0),
            Phase::Before => Phase::Before,
            Phase::Reasoning(_) if self.recent.ends_with(&self.spec.end) => Phase::After,
            Phase::Reasoning(count) => {
                let count = count + 1;
                if count >= self.spec.tokens {
                    // A partially emitted end tag is completed, not restarted.
                    let started = (1..self.spec.end.len())
                        .rev()
                        .find(|length| self.recent.ends_with(&self.spec.end[..*length]))
                        .unwrap_or(0);
                    Phase::Closing(started)
                } else {
                    Phase::Reasoning(count)
                }
            }
            Phase::Closing(index) => {
                if token != self.spec.end[index] {
                    return Err("token violates forced reasoning closure".into());
                }
                if index + 1 == self.spec.end.len() {
                    Phase::After
                } else {
                    Phase::Closing(index + 1)
                }
            }
            Phase::After => Phase::After,
        };
        Ok(())
    }

    /// Tokens of an inner forced run that the budget still admits unchanged.
    fn admits(&self, tokens: &[TokenId]) -> usize {
        let mut preview = self.clone();
        for (index, token) in tokens.iter().enumerate() {
            if preview.forced_token().is_some() || preview.accept(*token).is_err() {
                return index;
            }
        }
        tokens.len()
    }
}

/// The request's grammar (if any) composed with its generation controls.
pub(crate) struct Controlled {
    inner: Option<Box<dyn Constraint>>,
    suppressed: Option<Arc<BTreeSet<TokenId>>>,
    budget: Option<Budget>,
    vocabulary: usize,
    position: usize,
}

impl Controlled {
    /// `None` when no control restricts selection, leaving the grammar as is.
    pub(crate) fn compose(
        inner: Option<Box<dyn Constraint>>,
        end_of_generation: EndOfGeneration,
        stop_tokens: &BTreeSet<TokenId>,
        never: &BTreeSet<TokenId>,
        budget: Option<&ReasoningBudget>,
        vocabulary: usize,
    ) -> Result<Option<Box<dyn Constraint>>, String> {
        // The tokens selection excludes: the model's never-selected tokens,
        // plus its stop tokens when end of generation is suppressed.
        let suppressed: BTreeSet<TokenId> = match end_of_generation {
            EndOfGeneration::Suppress => never.union(stop_tokens).copied().collect(),
            EndOfGeneration::Stop => never.clone(),
        };
        if suppressed.is_empty() && budget.is_none() {
            return Ok(inner);
        }
        if let Some(budget) = budget {
            budget.validate(vocabulary)?;
            if budget.end.iter().any(|token| suppressed.contains(token)) {
                return Err("reasoning end tag cannot use a suppressed token".into());
            }
        }
        let position = inner.as_ref().map_or(0, |inner| inner.position());
        Ok(Some(Box::new(Self {
            inner,
            suppressed: (!suppressed.is_empty()).then(|| Arc::new(suppressed)),
            budget: budget.cloned().map(Budget::new),
            vocabulary,
            position,
        })))
    }
}

fn single(vocabulary: usize, token: TokenId) -> Arc<[u32]> {
    let mut mask = vec![0u32; vocabulary.div_ceil(32)];
    mask[token.0 as usize / 32] |= 1 << (token.0 % 32);
    mask.into()
}

impl Constraint for Controlled {
    fn position(&self) -> usize {
        self.position
    }

    fn fork(&self) -> Box<dyn Constraint> {
        Box::new(Self {
            inner: self.inner.as_ref().map(|inner| inner.fork()),
            suppressed: self.suppressed.clone(),
            budget: self.budget.clone(),
            vocabulary: self.vocabulary,
            position: self.position,
        })
    }

    fn stage(&self, tokens: &[TokenId]) -> Result<Box<dyn Constraint>, String> {
        if let Some(suppressed) = &self.suppressed {
            if tokens.iter().any(|token| suppressed.contains(token)) {
                return Err("suppressed token was selected".into());
            }
        }
        let mut budget = self.budget.clone();
        if let Some(budget) = budget.as_mut() {
            for token in tokens {
                budget.accept(*token)?;
            }
        }
        let inner = self
            .inner
            .as_ref()
            .map(|inner| inner.stage(tokens))
            .transpose()?;
        Ok(Box::new(Self {
            inner,
            suppressed: self.suppressed.clone(),
            budget,
            vocabulary: self.vocabulary,
            position: self.position + tokens.len(),
        }))
    }

    fn forced(&self, limit: usize) -> Result<Vec<TokenId>, String> {
        if limit == 0 {
            return Err("forced-token allowance must be positive".into());
        }
        if let Some(budget) = &self.budget {
            if let Phase::Closing(index) = budget.phase {
                return Ok(budget.spec.end[index..]
                    .iter()
                    .take(limit)
                    .copied()
                    .collect());
            }
        }
        let Some(inner) = &self.inner else {
            return Ok(Vec::new());
        };
        let mut forced = inner.forced(limit)?;
        if let Some(suppressed) = &self.suppressed {
            if let Some(end) = forced.iter().position(|token| suppressed.contains(token)) {
                forced.truncate(end);
            }
        }
        if let Some(budget) = &self.budget {
            forced.truncate(budget.admits(&forced));
        }
        Ok(forced)
    }

    fn mask(&self) -> Result<Arc<[u32]>, String> {
        if let Some(token) = self.budget.as_ref().and_then(Budget::forced_token) {
            return Ok(single(self.vocabulary, token));
        }
        let mut mask = match &self.inner {
            Some(inner) => inner.mask()?.to_vec(),
            None => {
                let mut mask = vec![u32::MAX; self.vocabulary.div_ceil(32)];
                if self.vocabulary % 32 != 0 {
                    *mask.last_mut().unwrap() = (1u32 << (self.vocabulary % 32)) - 1;
                }
                mask
            }
        };
        if let Some(suppressed) = &self.suppressed {
            for token in suppressed.iter() {
                if let Some(word) = mask.get_mut(token.0 as usize / 32) {
                    *word &= !(1 << (token.0 % 32));
                }
            }
        }
        Ok(mask.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed(mask: &[u32]) -> Vec<u32> {
        (0..mask.len() as u32 * 32)
            .filter(|id| mask[*id as usize / 32] & (1 << (id % 32)) != 0)
            .collect()
    }

    fn budget(tokens: u32, open: bool) -> ReasoningBudget {
        ReasoningBudget {
            tokens,
            start: vec![TokenId(10)],
            end: vec![TokenId(11), TokenId(12)],
            open,
        }
    }

    fn controlled(
        end_of_generation: EndOfGeneration,
        budget: Option<ReasoningBudget>,
    ) -> Box<dyn Constraint> {
        Controlled::compose(
            None,
            end_of_generation,
            &BTreeSet::from([TokenId(13)]),
            &BTreeSet::new(),
            budget.as_ref(),
            14,
        )
        .unwrap()
        .unwrap()
    }

    #[test]
    fn uncontrolled_requests_keep_their_grammar_untouched() {
        assert!(Controlled::compose(
            None,
            EndOfGeneration::Stop,
            &BTreeSet::from([TokenId(13)]),
            &BTreeSet::new(),
            None,
            14
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn never_selected_tokens_are_masked_whether_or_not_generation_can_end() {
        for end_of_generation in [EndOfGeneration::Stop, EndOfGeneration::Suppress] {
            let state = Controlled::compose(
                None,
                end_of_generation,
                &BTreeSet::from([TokenId(13)]),
                &BTreeSet::from([TokenId(3), TokenId(7)]),
                None,
                14,
            )
            .unwrap()
            .unwrap();
            let mut expected: Vec<u32> = (0..14).filter(|id| ![3, 7].contains(id)).collect();
            if end_of_generation == EndOfGeneration::Suppress {
                expected.retain(|id| *id != 13);
            }
            assert_eq!(allowed(&state.mask().unwrap()), expected);
            assert!(state.stage(&[TokenId(7)]).is_err());
        }
    }

    #[test]
    fn suppression_removes_only_stop_tokens_from_selection() {
        let state = controlled(EndOfGeneration::Suppress, None);
        assert_eq!(allowed(&state.mask().unwrap()), (0..13).collect::<Vec<_>>());
        assert!(state.stage(&[TokenId(13)]).is_err());
        assert_eq!(state.stage(&[TokenId(1)]).unwrap().position(), 1);
    }

    #[test]
    fn spent_open_budget_forces_the_end_tag_then_releases_selection() {
        let mut state = controlled(EndOfGeneration::Stop, Some(budget(2, true)));
        state = state.stage(&[TokenId(1)]).unwrap();
        assert_eq!(allowed(&state.mask().unwrap()).len(), 14);
        state = state.stage(&[TokenId(2)]).unwrap();
        assert_eq!(allowed(&state.mask().unwrap()), vec![11]);
        assert_eq!(state.forced(8).unwrap(), vec![TokenId(11), TokenId(12)]);
        assert!(state.stage(&[TokenId(3)]).is_err());
        state = state.stage(&[TokenId(11), TokenId(12)]).unwrap();
        assert_eq!(allowed(&state.mask().unwrap()).len(), 14);
        assert_eq!(state.position(), 4);
    }

    #[test]
    fn budget_counts_only_after_the_model_opens_reasoning() {
        let mut state = controlled(EndOfGeneration::Stop, Some(budget(1, false)));
        state = state.stage(&[TokenId(1), TokenId(2)]).unwrap();
        assert_eq!(allowed(&state.mask().unwrap()).len(), 14);
        state = state.stage(&[TokenId(10), TokenId(4)]).unwrap();
        assert_eq!(allowed(&state.mask().unwrap()), vec![11]);
    }

    #[test]
    fn model_closing_reasoning_within_budget_ends_the_restriction() {
        let mut state = controlled(EndOfGeneration::Stop, Some(budget(3, true)));
        state = state
            .stage(&[TokenId(1), TokenId(11), TokenId(12), TokenId(5), TokenId(6)])
            .unwrap();
        assert_eq!(allowed(&state.mask().unwrap()).len(), 14);
    }

    #[test]
    fn a_partially_emitted_end_tag_is_completed_when_the_budget_runs_out() {
        let mut state = controlled(EndOfGeneration::Stop, Some(budget(2, true)));
        state = state.stage(&[TokenId(1), TokenId(11)]).unwrap();
        assert_eq!(allowed(&state.mask().unwrap()), vec![12]);
    }
}
