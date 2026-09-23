//! Surface syntax: tokens, lexer, AST and parser. The canonical printer
//! exists only to test the parser's round trip.
pub mod ast;
pub(crate) mod lexer;
pub(crate) mod parser;
#[cfg(test)]
mod printer;
pub(crate) mod token;

pub(crate) use parser::parse;
