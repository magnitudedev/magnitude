//! Tokens of structured Seismic.

use crate::span::Span;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Name(String),
    Int(u64),
    Float(f64),
    Kw(Kw),
    Op(Op),
    Newline,
    Indent,
    Dedent,
    Eof,
}

/// Reserved words. Structural words are reserved everywhere; a function cannot shadow them.
/// Contextual words are ordinary names to the lexer and recognized by position in the
/// parser: `tensor`/`view`/`index` (type position), `out`/`inout` (before a parameter
/// name), `alias` (after a parameter list), `to` (after a `publish` value), `identity`
/// (after a `merge` binder list).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kw {
    Fn,
    Lower,
    Requires,
    Where,
    For,
    In,
    If,
    Else,
    And,
    Or,
    Not,
    True,
    False,
    Inf,
    Tile,
    Void,
    Let,
    Mut,
    Parallel,
    Ordered,
    Pipeline,
    Stage,
    Yield,
    Return,
    Merge,
    Publish,
}

impl Kw {
    pub const ALL: [Kw; 26] = [
        Kw::Fn,
        Kw::Lower,
        Kw::Requires,
        Kw::Where,
        Kw::For,
        Kw::In,
        Kw::If,
        Kw::Else,
        Kw::And,
        Kw::Or,
        Kw::Not,
        Kw::True,
        Kw::False,
        Kw::Inf,
        Kw::Tile,
        Kw::Void,
        Kw::Let,
        Kw::Mut,
        Kw::Parallel,
        Kw::Ordered,
        Kw::Pipeline,
        Kw::Stage,
        Kw::Yield,
        Kw::Return,
        Kw::Merge,
        Kw::Publish,
    ];

    pub fn from_name(name: &str) -> Option<Kw> {
        Kw::ALL.iter().copied().find(|kw| kw.text() == name)
    }

    pub fn text(self) -> &'static str {
        match self {
            Kw::Fn => "fn",
            Kw::Lower => "lower",
            Kw::Requires => "requires",
            Kw::Where => "where",
            Kw::For => "for",
            Kw::In => "in",
            Kw::If => "if",
            Kw::Else => "else",
            Kw::And => "and",
            Kw::Or => "or",
            Kw::Not => "not",
            Kw::True => "true",
            Kw::False => "false",
            Kw::Inf => "inf",
            Kw::Tile => "tile",
            Kw::Void => "void",
            Kw::Let => "let",
            Kw::Mut => "mut",
            Kw::Parallel => "parallel",
            Kw::Ordered => "ordered",
            Kw::Pipeline => "pipeline",
            Kw::Stage => "stage",
            Kw::Yield => "yield",
            Kw::Return => "return",
            Kw::Merge => "merge",
            Kw::Publish => "publish",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Shl,
    Shr,
    Amp,
    Pipe,
    Caret,
    Tilde,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Assign,
    PlusAssign,
    MinusAssign,
    StarAssign,
    Colon,
    Comma,
    Semi,
    Dot,
    DotDot,
    Arrow,
    LParen,
    RParen,
    LBracket,
    RBracket,
}

impl Op {
    pub fn text(self) -> &'static str {
        match self {
            Op::Plus => "+",
            Op::Minus => "-",
            Op::Star => "*",
            Op::Slash => "/",
            Op::Percent => "%",
            Op::Shl => "<<",
            Op::Shr => ">>",
            Op::Amp => "&",
            Op::Pipe => "|",
            Op::Caret => "^",
            Op::Tilde => "~",
            Op::EqEq => "==",
            Op::Ne => "!=",
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Gt => ">",
            Op::Ge => ">=",
            Op::Assign => "=",
            Op::PlusAssign => "+=",
            Op::MinusAssign => "-=",
            Op::StarAssign => "*=",
            Op::Colon => ":",
            Op::Comma => ",",
            Op::Semi => ";",
            Op::Dot => ".",
            Op::DotDot => "..",
            Op::Arrow => "->",
            Op::LParen => "(",
            Op::RParen => ")",
            Op::LBracket => "[",
            Op::RBracket => "]",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

impl Tok {
    pub fn describe(&self) -> String {
        match self {
            Tok::Name(n) => format!("name `{n}`"),
            Tok::Int(v) => format!("integer `{v}`"),
            Tok::Float(v) => format!("number `{v}`"),
            Tok::Kw(k) => format!("`{}`", k.text()),
            Tok::Op(o) => format!("`{}`", o.text()),
            Tok::Newline => "end of line".into(),
            Tok::Indent => "indent".into(),
            Tok::Dedent => "dedent".into(),
            Tok::Eof => "end of file".into(),
        }
    }
}
