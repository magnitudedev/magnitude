//! The Seismic language: lexer, parser, AST and printer.

pub mod ast;
pub mod check;
pub mod hir;
pub mod interp;
pub mod lower;
pub mod intrinsics;
pub mod lexer;
pub mod parser;
pub mod plan;
pub mod printer;
pub mod program;
pub mod repr;
pub mod rewrite;
pub mod span;
pub mod split;
pub mod sym;
pub mod types;
pub mod widen;
pub mod token;

pub use parser::parse;
pub use printer::print;
pub use span::{Diagnostic, Span};

/// The scope a file's extension selects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    Portable,
    Backend(String),
}

/// Parse a file name of the form `<name>.seismic.<scope>`.
pub fn scope_of_path(path: &str) -> Option<Scope> {
    let file = path.rsplit('/').next()?;
    let (_, rest) = file.split_once(".seismic.")?;
    if rest.is_empty() || rest.contains('.') {
        return None;
    }
    Some(if rest == "portable" { Scope::Portable } else { Scope::Backend(rest.to_string()) })
}
