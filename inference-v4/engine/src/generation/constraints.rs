//! Exact vocabulary binding and independent speculative llguidance matchers.
use super::{grammar, Constraint, Generation, Options};
use crate::{
    chat::{ConstraintPlan, PreparedInput},
    inputs::{ByteBpeTokenizer, InputLayout, PieceKind, SpecialTokens, TokenId},
};
use llguidance::{
    api::TopLevelGrammar,
    toktrie::{TokEnv, TokRxInfo, TokTrie, TokenizerEnv},
    Matcher, ParserFactory,
};
use std::{cell::RefCell, collections::VecDeque, sync::Arc};

struct Environment {
    tokenizer: Arc<ByteBpeTokenizer>,
    trie: TokTrie,
}
impl TokenizerEnv for Environment {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }
    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        self.trie.tokenize_with_greedy_fallback(bytes, |text| {
            self.tokenizer
                .encode(text, SpecialTokens::Recognize)
                .expect("validated byte BPE tokenization failed inside llguidance")
                .into_iter()
                .map(|t| t.0)
                .collect()
        })
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CacheLimits {
    pub entries: usize,
    pub bytes: usize,
}
struct Cached {
    plan: ConstraintPlan,
    matcher: Matcher,
    mask: Arc<[u32]>,
    bytes: usize,
}
pub struct Vocabulary {
    tokenizer: Arc<ByteBpeTokenizer>,
    factory: ParserFactory,
    usable: Arc<[u32]>,
    projection: usize,
    cache: VecDeque<Cached>,
    cache_bytes: usize,
    limits: CacheLimits,
    cache_hits: u64,
}
impl Vocabulary {
    pub fn new(
        tokenizer: Arc<ByteBpeTokenizer>,
        projection: usize,
        limits: CacheLimits,
    ) -> Result<Self, String> {
        if projection < tokenizer.vocabulary()
            || projection > u32::MAX as usize
            || tokenizer.stop_tokens().is_empty()
        {
            return Err(
                "constraint vocabulary requires model-sized projection and explicit EOS identities"
                    .into(),
            );
        }
        let mut words = Vec::with_capacity(projection);
        let mut usable = vec![0u32; projection.div_ceil(32)];
        for id in 0..tokenizer.vocabulary() {
            let token = TokenId(id as u32);
            let bytes = tokenizer.piece(token, false)?;
            if tokenizer.kind(token)? == PieceKind::Unused || bytes.starts_with(&[0xff]) {
                if tokenizer.stop_tokens().contains(&token) {
                    return Err("EOS cannot be an unusable token".into());
                }
                words.push(Vec::new());
            } else {
                words.push(bytes.to_vec());
                usable[id / 32] |= 1 << (id % 32);
            }
        }
        words.resize_with(projection, Vec::new);
        let eos = tokenizer
            .stop_tokens()
            .iter()
            .map(|id| id.0)
            .collect::<Vec<_>>();
        let trie =
            TokTrie::from(&TokRxInfo::new(projection as u32, eos[0]), &words).with_eos_tokens(&eos);
        let env: TokEnv = Arc::new(Environment {
            tokenizer: tokenizer.clone(),
            trie,
        });
        let mut factory = ParserFactory::new_simple(&env).map_err(|e| e.to_string())?;
        factory.quiet();
        let parser_limits = factory.limits_mut();
        parser_limits.max_lexer_states = 8192;
        parser_limits.max_grammar_size = 100_000;
        parser_limits.verbose_errors = false;
        Ok(Self {
            tokenizer,
            factory,
            usable: usable.into(),
            projection,
            cache: VecDeque::new(),
            cache_bytes: 0,
            limits,
            cache_hits: 0,
        })
    }
    pub fn projection(&self) -> usize {
        self.projection
    }
    pub fn base_mask(&self) -> Arc<[u32]> {
        self.usable.clone()
    }
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits
    }
    pub fn cached_entries(&self) -> usize {
        self.cache.len()
    }
    pub fn cached_bytes(&self) -> usize {
        self.cache_bytes
    }
    /// Execution-owner preparation before numerical admission. Never construct a
    /// logical request from a host plan whose identities or grammar cannot bind.
    /// This text-only boundary uses a causal layout; media admission must supply
    /// its separately prepared semantic spans.
    pub fn prepare_generation(
        &mut self,
        input: &PreparedInput,
        options: Options,
    ) -> Result<Generation, String> {
        self.prepare_generation_with_layout(
            input,
            options,
            InputLayout::new(input.tokens.len(), vec![])?,
        )
    }
    /// The model adapter supplies semantic spans before constraint/generation
    /// binding. This metadata carries no live device resources.
    pub fn prepare_generation_with_layout(
        &mut self,
        input: &PreparedInput,
        options: Options,
        layout: InputLayout,
    ) -> Result<Generation, String> {
        if input.artifact_identity != self.tokenizer.artifact_identity()
            || input.tokenizer_identity != self.tokenizer.identity()
            || options.vocabulary != self.projection
            || &options.stop_tokens != self.tokenizer.stop_tokens()
            || input.tokens.iter().any(|t| {
                t.0 as usize >= self.tokenizer.vocabulary()
                    || self.tokenizer.kind(*t).ok() == Some(PieceKind::Unused)
            })
        {
            return Err(
                "prepared input or generation options do not match the execution vocabulary".into(),
            );
        }
        let constraint = input
            .constraint
            .as_ref()
            .map(|plan| {
                self.bind(plan)
                    .map(|state| Box::new(state) as Box<dyn Constraint>)
            })
            .transpose()?;
        Generation::new(input.tokens.clone(), layout, options, constraint)
    }
    pub fn bind(&mut self, plan: &ConstraintPlan) -> Result<State, String> {
        if plan.artifact_identity != self.tokenizer.artifact_identity()
            || plan.tokenizer_identity != self.tokenizer.identity()
            || plan.template_identity.is_empty()
            || plan.converter_identity != grammar::CONVERTER_IDENTITY
        {
            return Err("constraint plan has incompatible artifact, tokenizer, template, or converter identity".into());
        }
        if plan.initial_prefix.len() > 1024 * 1024 {
            return Err("grammar initial prefix exceeds size limit".into());
        }
        if let Some(index) = self.cache.iter().position(|entry| &entry.plan == plan) {
            let entry = self.cache.remove(index).unwrap();
            self.cache_hits += 1;
            let state = self.state(entry.matcher.deep_clone(), entry.mask.clone());
            self.cache.push_back(entry);
            return Ok(state);
        }
        let lark = grammar::to_lark(&plan.gbnf)?;
        let mut matcher = Matcher::new(
            self.factory
                .create_parser(TopLevelGrammar::from_lark(lark.clone())),
        );
        check(&matcher)?;
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(format!(
                "constraint compilation warnings: {}",
                warnings.join("; ")
            ));
        }
        let prefix = self
            .tokenizer
            .encode(&plan.initial_prefix, SpecialTokens::Recognize)?;
        let mut bytes = Vec::new();
        for token in &prefix {
            bytes.extend_from_slice(self.tokenizer.piece(*token, false)?);
        }
        if bytes != plan.initial_prefix.as_bytes()
            || prefix
                .iter()
                .any(|t| self.tokenizer.stop_tokens().contains(t) || !allowed(&self.usable, *t))
        {
            return Err(
                "grammar prefix is not exactly representable as usable non-EOS tokens".into(),
            );
        }
        let prefix = prefix.into_iter().map(|t| t.0).collect::<Vec<_>>();
        if !prefix.is_empty() {
            if matcher
                .validate_tokens(&prefix)
                .map_err(|e| e.to_string())?
                != prefix.len()
            {
                return Err("grammar rejects its initial prefix".into());
            }
            matcher.consume_tokens(&prefix).map_err(|e| e.to_string())?;
        }
        let mask = mask(&mut matcher, &self.usable)?;
        let charged = serde_json::to_vec(plan).map_err(|e| e.to_string())?.len()
            + lark.len()
            + mask.len() * 4;
        if self.limits.entries > 0 && charged <= self.limits.bytes {
            while self.cache.len() >= self.limits.entries
                || self.cache_bytes + charged > self.limits.bytes
            {
                self.cache_bytes -= self.cache.pop_front().unwrap().bytes;
            }
            self.cache.push_back(Cached {
                plan: plan.clone(),
                matcher: matcher.deep_clone(),
                mask: mask.clone(),
                bytes: charged,
            });
            self.cache_bytes += charged;
        }
        Ok(self.state(matcher, mask))
    }
    fn state(&self, matcher: Matcher, mask: Arc<[u32]>) -> State {
        State {
            matcher: RefCell::new(matcher),
            cached_mask: RefCell::new(Some(mask)),
            tokenizer: self.tokenizer.clone(),
            usable: self.usable.clone(),
            position: 0,
            terminal: false,
        }
    }
}
fn check(matcher: &Matcher) -> Result<(), String> {
    match matcher.get_error() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
fn allowed(mask: &[u32], token: TokenId) -> bool {
    mask.get(token.0 as usize / 32)
        .is_some_and(|word| word & (1 << (token.0 % 32)) != 0)
}
fn mask(matcher: &mut Matcher, usable: &[u32]) -> Result<Arc<[u32]>, String> {
    let mask = matcher.compute_mask_or_eos().map_err(|e| e.to_string())?;
    check(matcher)?;
    if mask.as_slice().len() < usable.len() {
        return Err("constraint mask does not cover projection vocabulary".into());
    }
    Ok(mask
        .as_slice()
        .iter()
        .zip(usable)
        .map(|(actual, usable)| actual & usable)
        .collect::<Vec<_>>()
        .into())
}

pub struct State {
    matcher: RefCell<Matcher>,
    cached_mask: RefCell<Option<Arc<[u32]>>>,
    tokenizer: Arc<ByteBpeTokenizer>,
    usable: Arc<[u32]>,
    position: usize,
    terminal: bool,
}
impl State {
    pub fn fork(&self) -> Self {
        Self {
            matcher: RefCell::new(self.matcher.borrow().deep_clone()),
            cached_mask: RefCell::new(self.cached_mask.borrow().clone()),
            tokenizer: self.tokenizer.clone(),
            usable: self.usable.clone(),
            position: self.position,
            terminal: self.terminal,
        }
    }
    pub fn accepting(&self) -> Result<bool, String> {
        self.matcher
            .borrow_mut()
            .is_accepting()
            .map_err(|e| e.to_string())
    }
    pub fn stopped(&self) -> bool {
        self.terminal
    }
    pub fn advance(&self, tokens: &[TokenId]) -> Result<Self, String> {
        if self.terminal || tokens.is_empty() {
            return Err("constraint transition requires nonempty tokens and a live matcher".into());
        }
        if tokens
            .iter()
            .any(|t| t.0 as usize >= self.tokenizer.vocabulary() || !allowed(&self.usable, *t))
        {
            return Err("constraint transition contains unusable token IDs".into());
        }
        if tokens[..tokens.len() - 1]
            .iter()
            .any(|t| self.tokenizer.stop_tokens().contains(t))
        {
            return Err("constraint transition continues after EOS".into());
        }
        let terminal = self
            .tokenizer
            .stop_tokens()
            .contains(tokens.last().unwrap());
        let content = &tokens[..tokens.len() - usize::from(terminal)];
        let mut matcher = self.matcher.borrow().deep_clone();
        if !content.is_empty() {
            let ids = content.iter().map(|t| t.0).collect::<Vec<_>>();
            if matcher.validate_tokens(&ids).map_err(|e| e.to_string())? != ids.len() {
                return Err("tokens violate prepared constraint".into());
            }
            matcher.consume_tokens(&ids).map_err(|e| e.to_string())?;
        }
        if terminal {
            let eos = *tokens.last().unwrap();
            let next = mask(&mut matcher, &self.usable)?;
            if !matcher.is_accepting().map_err(|e| e.to_string())? || !allowed(&next, eos) {
                return Err("EOS violates prepared constraint".into());
            }
            // A finite grammar can stop as soon as its last content token is
            // consumed. Its EOS mask still authorizes the engine's terminal
            // token, but llguidance rejects another consume on that parser.
            if !matcher.is_stopped() {
                matcher.consume_token(eos.0).map_err(|e| e.to_string())?;
            }
        }
        check(&matcher)?;
        Ok(Self {
            matcher: RefCell::new(matcher),
            cached_mask: RefCell::new(None),
            tokenizer: self.tokenizer.clone(),
            usable: self.usable.clone(),
            position: self.position + tokens.len(),
            terminal,
        })
    }
}
impl Constraint for State {
    fn fork(&self) -> Box<dyn Constraint> {
        Box::new(State::fork(self))
    }
    fn position(&self) -> usize {
        self.position
    }
    fn stage(&self, tokens: &[TokenId]) -> Result<Box<dyn Constraint>, String> {
        Ok(Box::new(self.advance(tokens)?))
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
        check(&matcher)?;
        drop(matcher);
        let tokens = forced
            .into_iter()
            .map(TokenId)
            .take(limit)
            .take_while(|t| !self.tokenizer.stop_tokens().contains(t))
            .collect::<Vec<_>>();
        if !tokens.is_empty() {
            self.advance(&tokens)?;
        }
        Ok(tokens)
    }
    fn mask(&self) -> Result<Arc<[u32]>, String> {
        if let Some(mask) = self.cached_mask.borrow().as_ref() {
            return Ok(mask.clone());
        }
        let mask = mask(&mut self.matcher.borrow_mut(), &self.usable)?;
        *self.cached_mask.borrow_mut() = Some(mask.clone());
        Ok(mask)
    }
}
