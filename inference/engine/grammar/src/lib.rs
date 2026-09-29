//! Constrained decoding: GBNF compiled to an llguidance grammar, bound to a
//! model vocabulary, and matched per token.
//!
//! Compilation turns GBNF into a lexical plan. Regular rules become
//! terminals defined by reference; recursive rules become Earley rules whose
//! skeletons flatten non-recursive structure within a size allowance; a
//! lexeme is every regular path between two parse boundaries. Each lexeme is
//! certified exact under llguidance's greedy lexing or rendered one character
//! at a time, so the result accepts exactly the GBNF language and its size is
//! bounded by a fixed multiple of the source.
mod cache;
mod certify;
mod constraint;
mod elimination;
mod gbnf;
mod lark;
mod lexeme;
mod network;
mod terminal;
mod vocabulary;

pub use cache::CompileCache;
pub use constraint::GrammarConstraint;
pub use magnitude_generation::TokenId;
pub use vocabulary::{CacheLimits, TokenTable, Vocabulary};

use serde::{Deserialize, Serialize};
use std::{fmt, time::Instant};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrammarError {
    /// The source is not admissible GBNF.
    Syntax(String),
    /// The source is well formed but its language cannot be used.
    Language(String),
    /// The grammar cannot be bound to the vocabulary or its prefix.
    Binding(String),
    /// Tokens outside the grammar's language.
    Violation(String),
}

impl fmt::Display for GrammarError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(message) => write!(formatter, "invalid GBNF: {message}"),
            Self::Language(message) => write!(formatter, "unusable grammar: {message}"),
            Self::Binding(message) => write!(formatter, "grammar binding failed: {message}"),
            Self::Violation(message) => write!(formatter, "grammar violation: {message}"),
        }
    }
}

impl std::error::Error for GrammarError {}

/// A compiled, tokenizer-independent grammar.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Grammar {
    lark: String,
}

impl Grammar {
    /// The llguidance Lark program.
    pub fn lark(&self) -> &str {
        &self.lark
    }
}

/// What compilation produced and where it had to degrade.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CompileReport {
    pub gbnf_bytes: usize,
    pub lark_bytes: usize,
    pub symbols: usize,
    pub earley_rules: usize,
    pub terminals: usize,
    pub lexemes: usize,
    /// Right-linear components compiled to terminals.
    pub scanners: usize,
    /// Structural rules kept as calls because flattening exceeded the
    /// allowance.
    pub unflattened: usize,
    /// Skeleton states kept as boundaries because eliminating them exceeded
    /// the allowance.
    pub hubs: usize,
    /// Lexemes rendered one character at a time.
    pub character_lexemes: usize,
    pub certification_questions: usize,
    pub exhausted_questions: usize,
    pub compile_us: u64,
}

pub struct Compiled {
    pub grammar: Grammar,
    pub report: CompileReport,
}

pub fn compile(gbnf: &str) -> Result<Compiled, GrammarError> {
    let started = Instant::now();
    let rules = gbnf::parse(gbnf)?;
    let network = network::Network::build(&rules)?;
    let mut terms = terminal::Terms::new();
    let mut allowance = terminal::Allowance::new(network.symbols);
    let (classes, scanners) = terminal::classify(&network, &mut terms, &mut allowance);
    let mut plan = lexeme::Plan::build(&network, &classes, &mut terms, &mut allowance);
    let certification = certify::certify(&mut plan, &mut terms);
    let rendered = lark::render(&plan, &terms);
    Ok(Compiled {
        report: CompileReport {
            gbnf_bytes: gbnf.len(),
            lark_bytes: rendered.lark.len(),
            symbols: network.symbols,
            earley_rules: plan.rules.len(),
            terminals: rendered.terminals,
            lexemes: rendered.lexemes,
            scanners,
            unflattened: plan.unflattened,
            hubs: plan.hubs,
            character_lexemes: certification.characters,
            certification_questions: certification.questions,
            exhausted_questions: certification.exhausted,
            compile_us: started.elapsed().as_micros() as u64,
        },
        grammar: Grammar {
            lark: rendered.lark,
        },
    })
}

/// The same language with every lexeme at character level: exact by
/// construction, the reference the compiled form is tested against.
#[cfg(test)]
pub(crate) fn compile_characters(gbnf: &str) -> Result<Grammar, GrammarError> {
    let rules = gbnf::parse(gbnf)?;
    let network = network::Network::build(&rules)?;
    let mut terms = terminal::Terms::new();
    let mut allowance = terminal::Allowance::new(network.symbols);
    let (classes, _) = terminal::classify(&network, &mut terms, &mut allowance);
    let mut plan = lexeme::Plan::build(&network, &classes, &mut terms, &mut allowance);
    loop {
        let view = &terms;
        let lexemes = plan
            .rules
            .iter()
            .enumerate()
            .flat_map(|(rule, earley)| {
                earley
                    .edges
                    .iter()
                    .enumerate()
                    .filter_map(move |(edge, e)| match e.kind {
                        lexeme::EdgeKind::Lexeme { lexeme, .. }
                            if !view.single_character(lexeme) =>
                        {
                            Some((rule, edge))
                        }
                        _ => None,
                    })
            })
            .collect::<Vec<_>>();
        if lexemes.is_empty() {
            break;
        }
        for (rule, edge) in lexemes {
            plan.characters(rule, edge, &mut terms);
        }
    }
    Ok(Grammar {
        lark: lark::render(&plan, &terms).lark,
    })
}

#[cfg(test)]
mod tests;
