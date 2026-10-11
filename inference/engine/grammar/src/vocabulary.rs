//! A model vocabulary prepared for matching: the llguidance parser factory
//! over the token trie, and a cache of bound grammars.
use crate::{constraint::GrammarConstraint, Grammar, GrammarError, TokenId};
use llguidance::{
    api::TopLevelGrammar,
    toktrie::{TokEnv, TokRxInfo, TokTrie, TokenizerEnv},
    Matcher, ParserFactory,
};
use std::{
    collections::{BTreeSet, VecDeque},
    sync::Arc,
};

/// Tokenizer facts binding needs.
pub trait TokenTable: Send + Sync {
    fn len(&self) -> usize;
    /// The bytes a token spells; `None` for tokens that are never generated.
    fn bytes(&self, token: TokenId) -> Option<&[u8]>;
    fn stop_tokens(&self) -> &BTreeSet<TokenId>;
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, String>;
}

struct Environment {
    table: Arc<dyn TokenTable>,
    trie: TokTrie,
}

impl TokenizerEnv for Environment {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }
    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        self.trie.tokenize_with_greedy_fallback(bytes, |text| {
            self.table
                .encode(text)
                .expect("the token table encodes text it spelled")
                .into_iter()
                .map(|token| token.0)
                .collect()
        })
    }
}

/// Lexer derivatives llguidance may build for one step, ten times its default. Flattened tool
/// schemas start lexeme sets whose states need more construction than the default allows, while
/// a mask over them still takes milliseconds.
const LEXER_FUEL: u64 = 2_000_000;

#[derive(Clone, Copy, Debug)]
pub struct CacheLimits {
    pub entries: usize,
    pub bytes: usize,
}

struct Cached {
    grammar: Grammar,
    prefix: Vec<TokenId>,
    matcher: Matcher,
    mask: Arc<[u32]>,
    bytes: usize,
}

pub struct Vocabulary {
    table: Arc<dyn TokenTable>,
    factory: ParserFactory,
    usable: Arc<[u32]>,
    projection: usize,
    cache: VecDeque<Cached>,
    cache_bytes: usize,
    limits: CacheLimits,
    cache_hits: u64,
}

pub(crate) fn allowed(mask: &[u32], token: TokenId) -> bool {
    mask.get(token.0 as usize / 32)
        .is_some_and(|word| word & (1 << (token.0 % 32)) != 0)
}

pub(crate) fn check(matcher: &Matcher) -> Result<(), GrammarError> {
    match matcher.get_error() {
        Some(error) => Err(GrammarError::Binding(error)),
        None => Ok(()),
    }
}

/// The matcher's mask restricted to tokens the model may generate.
pub(crate) fn mask(matcher: &mut Matcher, usable: &[u32]) -> Result<Arc<[u32]>, GrammarError> {
    let mask = matcher
        .compute_mask_or_eos()
        .map_err(|error| GrammarError::Binding(error.to_string()))?;
    check(matcher)?;
    if mask.as_slice().len() < usable.len() {
        return Err(GrammarError::Binding(
            "the matcher's mask does not cover the projection".into(),
        ));
    }
    Ok(mask
        .as_slice()
        .iter()
        .zip(usable)
        .map(|(actual, usable)| actual & usable)
        .collect::<Vec<_>>()
        .into())
}

impl Vocabulary {
    /// `projection` is the model's output width, at least the table's size;
    /// tokens beyond the table are never generated.
    pub fn new(
        table: Arc<dyn TokenTable>,
        projection: usize,
        limits: CacheLimits,
    ) -> Result<Self, GrammarError> {
        if projection < table.len()
            || projection > u32::MAX as usize
            || table.stop_tokens().is_empty()
        {
            return Err(GrammarError::Binding(
                "a vocabulary needs a model-sized projection and explicit stop tokens".into(),
            ));
        }
        let mut words = Vec::with_capacity(projection);
        let mut usable = vec![0u32; projection.div_ceil(32)];
        for id in 0..table.len() {
            let token = TokenId(id as u32);
            let stop = table.stop_tokens().contains(&token);
            match table.bytes(token) {
                None if stop => {
                    return Err(GrammarError::Binding(
                        "a stop token cannot be an unusable token".into(),
                    ))
                }
                None => words.push(Vec::new()),
                // A stop token ends generation; it never spells its text.
                // The matcher admits it only where the grammar accepts.
                Some(_) if stop => {
                    words.push(Vec::new());
                    usable[id / 32] |= 1 << (id % 32);
                }
                Some(bytes) => {
                    words.push(bytes.to_vec());
                    usable[id / 32] |= 1 << (id % 32);
                }
            }
        }
        words.resize_with(projection, Vec::new);
        let stops = table
            .stop_tokens()
            .iter()
            .map(|token| token.0)
            .collect::<Vec<_>>();
        let trie = TokTrie::from(&TokRxInfo::new(projection as u32, stops[0]), &words)
            .with_eos_tokens(&stops);
        let env: TokEnv = Arc::new(Environment {
            table: table.clone(),
            trie,
        });
        let mut factory = ParserFactory::new_simple(&env)
            .map_err(|error| GrammarError::Binding(error.to_string()))?;
        factory.quiet();
        let parser = factory.limits_mut();
        parser.verbose_errors = false;
        parser.initial_lexer_fuel = LEXER_FUEL;
        parser.step_lexer_fuel = LEXER_FUEL;
        Ok(Self {
            table,
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
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits
    }
    pub fn cached_entries(&self) -> usize {
        self.cache.len()
    }
    pub fn cached_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// A matcher for `grammar` that has consumed `prefix`: the grammar's
    /// leading text the prompt already contains.
    pub fn bind(
        &mut self,
        grammar: &Grammar,
        prefix: &[TokenId],
    ) -> Result<GrammarConstraint, GrammarError> {
        if let Some(index) = self
            .cache
            .iter()
            .position(|entry| &entry.grammar == grammar && entry.prefix == prefix)
        {
            let entry = self.cache.remove(index).unwrap();
            self.cache_hits += 1;
            // A private lexer per request: lexer state accumulates, and must
            // not grow across requests sharing a cached grammar.
            let state = self.constraint(entry.matcher.deep_clone(), entry.mask.clone());
            self.cache.push_back(entry);
            return Ok(state);
        }
        if prefix
            .iter()
            .any(|token| self.table.stop_tokens().contains(token) || !allowed(&self.usable, *token))
        {
            return Err(GrammarError::Binding(
                "a grammar prefix must consist of usable tokens other than stop tokens".into(),
            ));
        }
        let mut matcher = Matcher::new(
            self.factory
                .create_parser(TopLevelGrammar::from_lark(grammar.lark().to_owned())),
        );
        check(&matcher)?;
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(GrammarError::Binding(format!(
                "grammar warnings: {}",
                warnings.join("; ")
            )));
        }
        if !prefix.is_empty() {
            let ids = prefix.iter().map(|token| token.0).collect::<Vec<_>>();
            let accepted = matcher
                .validate_tokens(&ids)
                .map_err(|error| GrammarError::Binding(error.to_string()))?;
            if accepted != ids.len() {
                return Err(GrammarError::Binding(
                    "the grammar rejects its prefix".into(),
                ));
            }
            matcher
                .consume_tokens(&ids)
                .map_err(|error| GrammarError::Binding(error.to_string()))?;
        }
        let mask = mask(&mut matcher, &self.usable)?;
        let charged = grammar.lark().len() + prefix.len() * 4 + mask.len() * 4;
        if self.limits.entries > 0 && charged <= self.limits.bytes {
            while self.cache.len() >= self.limits.entries
                || self.cache_bytes + charged > self.limits.bytes
            {
                self.cache_bytes -= self.cache.pop_front().unwrap().bytes;
            }
            self.cache.push_back(Cached {
                grammar: grammar.clone(),
                prefix: prefix.to_vec(),
                matcher: matcher.deep_clone(),
                mask: mask.clone(),
                bytes: charged,
            });
            self.cache_bytes += charged;
        }
        Ok(self.constraint(matcher, mask))
    }

    fn constraint(&self, matcher: Matcher, mask: Arc<[u32]>) -> GrammarConstraint {
        GrammarConstraint::new(matcher, mask, self.table.clone(), self.usable.clone())
    }
}
