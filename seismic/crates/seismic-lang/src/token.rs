//! Tokens of the Seismic language.

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kw {
    Fn,
    Construct,
    Lower,
    Portable,
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
}

impl Kw {
    pub fn from_name(name: &str) -> Option<Kw> {
        Some(match name {
            "fn" => Kw::Fn,
            "construct" => Kw::Construct,
            "lower" => Kw::Lower,
            "portable" => Kw::Portable,
            "for" => Kw::For,
            "in" => Kw::In,
            "if" => Kw::If,
            "else" => Kw::Else,
            "and" => Kw::And,
            "or" => Kw::Or,
            "not" => Kw::Not,
            "true" => Kw::True,
            "false" => Kw::False,
            "inf" => Kw::Inf,
            "tile" => Kw::Tile,
            _ => return None,
        })
    }

    pub fn text(self) -> &'static str {
        match self {
            Kw::Fn => "fn",
            Kw::Construct => "construct",
            Kw::Lower => "lower",
            Kw::Portable => "portable",
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
