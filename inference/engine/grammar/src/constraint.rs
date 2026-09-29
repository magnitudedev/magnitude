//! Per-request grammar state for generation.
use crate::{
    vocabulary::{allowed, check, mask, TokenTable},
    GrammarError, TokenId,
};
use llguidance::Matcher;
use magnitude_generation::Constraint;
use std::{cell::RefCell, sync::Arc};

/// A bound grammar at an accepted position. Staging and forking copy the
/// parser state and share the request's lexer, which only caches DFA states.
pub struct GrammarConstraint {
    matcher: RefCell<Matcher>,
    cached_mask: RefCell<Option<Arc<[u32]>>>,
    table: Arc<dyn TokenTable>,
    usable: Arc<[u32]>,
    position: usize,
    terminal: bool,
}

impl GrammarConstraint {
    pub(crate) fn new(
        matcher: Matcher,
        mask: Arc<[u32]>,
        table: Arc<dyn TokenTable>,
        usable: Arc<[u32]>,
    ) -> Self {
        Self {
            matcher: RefCell::new(matcher),
            cached_mask: RefCell::new(Some(mask)),
            table,
            usable,
            position: 0,
            terminal: false,
        }
    }
    pub fn fork(&self) -> Self {
        Self {
            matcher: RefCell::new(self.matcher.borrow().clone()),
            cached_mask: RefCell::new(self.cached_mask.borrow().clone()),
            table: self.table.clone(),
            usable: self.usable.clone(),
            position: self.position,
            terminal: self.terminal,
        }
    }
    pub fn accepting(&self) -> Result<bool, GrammarError> {
        self.matcher
            .borrow_mut()
            .is_accepting()
            .map_err(|error| GrammarError::Binding(error.to_string()))
    }
    pub fn stopped(&self) -> bool {
        self.terminal
    }
    pub fn advance(&self, tokens: &[TokenId]) -> Result<Self, GrammarError> {
        let violation = |message: &str| Err(GrammarError::Violation(message.into()));
        if self.terminal || tokens.is_empty() {
            return violation("a transition needs tokens and a live matcher");
        }
        if tokens
            .iter()
            .any(|token| token.0 as usize >= self.table.len() || !allowed(&self.usable, *token))
        {
            return violation("the transition contains unusable tokens");
        }
        let stops = self.table.stop_tokens();
        if tokens[..tokens.len() - 1]
            .iter()
            .any(|token| stops.contains(token))
        {
            return violation("the transition continues after a stop token");
        }
        let terminal = stops.contains(tokens.last().unwrap());
        let content = &tokens[..tokens.len() - usize::from(terminal)];
        let mut matcher = self.matcher.borrow().clone();
        if !content.is_empty() {
            let ids = content.iter().map(|token| token.0).collect::<Vec<_>>();
            let accepted = matcher
                .validate_tokens(&ids)
                .map_err(|error| GrammarError::Binding(error.to_string()))?;
            if accepted != ids.len() {
                return violation("the tokens violate the grammar");
            }
            matcher
                .consume_tokens(&ids)
                .map_err(|error| GrammarError::Binding(error.to_string()))?;
        }
        if terminal {
            let stop = *tokens.last().unwrap();
            let next = mask(&mut matcher, &self.usable)?;
            let accepting = matcher
                .is_accepting()
                .map_err(|error| GrammarError::Binding(error.to_string()))?;
            if !accepting || !allowed(&next, stop) {
                return violation("the grammar does not end here");
            }
            // A finite grammar can stop as soon as its last content token is
            // consumed. Its mask still admits the engine's stop token, but
            // llguidance rejects another consume on that parser.
            if !matcher.is_stopped() {
                matcher
                    .consume_token(stop.0)
                    .map_err(|error| GrammarError::Binding(error.to_string()))?;
            }
        }
        check(&matcher)?;
        Ok(Self {
            matcher: RefCell::new(matcher),
            cached_mask: RefCell::new(None),
            table: self.table.clone(),
            usable: self.usable.clone(),
            position: self.position + tokens.len(),
            terminal,
        })
    }
}

impl Constraint for GrammarConstraint {
    fn fork(&self) -> Box<dyn Constraint> {
        Box::new(GrammarConstraint::fork(self))
    }
    fn position(&self) -> usize {
        self.position
    }
    fn stage(&self, tokens: &[TokenId]) -> Result<Box<dyn Constraint>, String> {
        Ok(Box::new(self.advance(tokens).map_err(|e| e.to_string())?))
    }
    fn forced(&self, limit: usize) -> Result<Vec<TokenId>, String> {
        if limit == 0 {
            return Err("forced-token allowance must be positive".into());
        }
        if self.terminal {
            return Ok(Vec::new());
        }
        let mut matcher = self.matcher.borrow_mut();
        let forced = matcher.compute_ff_tokens();
        check(&matcher).map_err(|e| e.to_string())?;
        let stops = self.table.stop_tokens();
        Ok(forced
            .into_iter()
            .map(TokenId)
            .take(limit)
            .take_while(|token| !stops.contains(token))
            .collect())
    }
    fn mask(&self) -> Result<Arc<[u32]>, String> {
        if let Some(mask) = self.cached_mask.borrow().as_ref() {
            return Ok(mask.clone());
        }
        let mask = mask(&mut self.matcher.borrow_mut(), &self.usable).map_err(|e| e.to_string())?;
        *self.cached_mask.borrow_mut() = Some(mask.clone());
        Ok(mask)
    }
}
