//! Indentation-sensitive lexer.
//!
//! Emits `Newline`, `Indent` and `Dedent` tokens like Python's tokenizer.
//! Newlines inside brackets are ignored. Comments start with `#`.

use crate::span::{Diagnostic, Span};
use crate::token::{Kw, Op, Tok, Token};

pub fn lex(text: &str) -> Result<Vec<Token>, Diagnostic> {
    Lexer { text, bytes: text.as_bytes(), pos: 0, tokens: Vec::new(), indents: vec![0], depth: 0, at_line_start: true }
        .run()
}

struct Lexer<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    tokens: Vec<Token>,
    indents: Vec<usize>,
    depth: usize,
    at_line_start: bool,
}

impl<'a> Lexer<'a> {
    fn run(mut self) -> Result<Vec<Token>, Diagnostic> {
        while self.pos < self.bytes.len() {
            if self.at_line_start && self.depth == 0 {
                self.handle_line_start()?;
                continue;
            }
            let c = self.bytes[self.pos];
            match c {
                b' ' => self.pos += 1,
                b'\t' => return Err(Diagnostic::new(Span::new(self.pos, self.pos + 1), "tabs are not allowed; use spaces")),
                b'\r' => self.pos += 1,
                b'#' => self.skip_comment(),
                b'\n' => {
                    self.pos += 1;
                    if self.depth == 0 {
                        self.push_newline();
                        self.at_line_start = true;
                    }
                }
                b'0'..=b'9' => self.number()?,
                b'A'..=b'Z' | b'a'..=b'z' | b'_' => self.name(),
                _ => self.operator()?,
            }
        }
        if self.depth != 0 {
            return Err(Diagnostic::new(Span::new(self.pos, self.pos), "unclosed bracket at end of file"));
        }
        self.push_newline();
        while self.indents.len() > 1 {
            self.indents.pop();
            self.tokens.push(Token { tok: Tok::Dedent, span: Span::new(self.pos, self.pos) });
        }
        self.tokens.push(Token { tok: Tok::Eof, span: Span::new(self.pos, self.pos) });
        Ok(self.tokens)
    }

    fn push_newline(&mut self) {
        if matches!(self.tokens.last().map(|t| &t.tok), Some(Tok::Newline) | Some(Tok::Indent) | None) {
            return;
        }
        self.tokens.push(Token { tok: Tok::Newline, span: Span::new(self.pos.saturating_sub(1), self.pos) });
    }

    fn handle_line_start(&mut self) -> Result<(), Diagnostic> {
        self.at_line_start = false;
        // Measure indentation; skip blank and comment-only lines entirely.
        let start = self.pos;
        let mut width = 0;
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' => {
                    width += 1;
                    self.pos += 1;
                }
                b'\t' => return Err(Diagnostic::new(Span::new(self.pos, self.pos + 1), "tabs are not allowed; use spaces")),
                b'\r' => self.pos += 1,
                _ => break,
            }
        }
        if self.pos >= self.bytes.len() {
            return Ok(());
        }
        match self.bytes[self.pos] {
            b'\n' => {
                self.pos += 1;
                self.at_line_start = true;
                return Ok(());
            }
            b'#' => {
                self.skip_comment();
                if self.pos < self.bytes.len() && self.bytes[self.pos] == b'\n' {
                    self.pos += 1;
                }
                self.at_line_start = true;
                return Ok(());
            }
            _ => {}
        }
        let current = *self.indents.last().unwrap();
        if width > current {
            self.indents.push(width);
            self.tokens.push(Token { tok: Tok::Indent, span: Span::new(start, self.pos) });
        } else {
            while width < *self.indents.last().unwrap() {
                self.indents.pop();
                self.tokens.push(Token { tok: Tok::Dedent, span: Span::new(start, self.pos) });
            }
            if width != *self.indents.last().unwrap() {
                return Err(Diagnostic::new(Span::new(start, self.pos), "inconsistent indentation"));
            }
        }
        Ok(())
    }

    fn skip_comment(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
            self.pos += 1;
        }
    }

    fn number(&mut self) -> Result<(), Diagnostic> {
        let start = self.pos;
        if self.bytes[self.pos] == b'0' && self.pos + 1 < self.bytes.len() && (self.bytes[self.pos + 1] == b'x' || self.bytes[self.pos + 1] == b'X') {
            self.pos += 2;
            let digits = self.pos;
            while self.pos < self.bytes.len() && (self.bytes[self.pos].is_ascii_hexdigit() || self.bytes[self.pos] == b'_') {
                self.pos += 1;
            }
            let s: String = self.text[digits..self.pos].chars().filter(|c| *c != '_').collect();
            let value = u64::from_str_radix(&s, 16).map_err(|_| Diagnostic::new(Span::new(start, self.pos), "invalid hexadecimal literal"))?;
            self.tokens.push(Token { tok: Tok::Int(value), span: Span::new(start, self.pos) });
            return Ok(());
        }
        let mut is_float = false;
        while self.pos < self.bytes.len() && (self.bytes[self.pos].is_ascii_digit() || self.bytes[self.pos] == b'_') {
            self.pos += 1;
        }
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'.' && self.pos + 1 < self.bytes.len() && self.bytes[self.pos + 1].is_ascii_digit() {
            is_float = true;
            self.pos += 1;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos < self.bytes.len() && (self.bytes[self.pos] == b'e' || self.bytes[self.pos] == b'E') {
            let save = self.pos;
            self.pos += 1;
            if self.pos < self.bytes.len() && (self.bytes[self.pos] == b'+' || self.bytes[self.pos] == b'-') {
                self.pos += 1;
            }
            if self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                is_float = true;
                while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
            } else {
                self.pos = save;
            }
        }
        let s: String = self.text[start..self.pos].chars().filter(|c| *c != '_').collect();
        let span = Span::new(start, self.pos);
        if is_float {
            let value: f64 = s.parse().map_err(|_| Diagnostic::new(span, "invalid number literal"))?;
            self.tokens.push(Token { tok: Tok::Float(value), span });
        } else {
            let value: u64 = s.parse().map_err(|_| Diagnostic::new(span, "integer literal out of range"))?;
            self.tokens.push(Token { tok: Tok::Int(value), span });
        }
        Ok(())
    }

    fn name(&mut self) {
        let start = self.pos;
        while self.pos < self.bytes.len() && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_') {
            self.pos += 1;
        }
        let name = &self.text[start..self.pos];
        let tok = match Kw::from_name(name) {
            Some(kw) => Tok::Kw(kw),
            None => Tok::Name(name.to_string()),
        };
        self.tokens.push(Token { tok, span: Span::new(start, self.pos) });
    }

    fn operator(&mut self) -> Result<(), Diagnostic> {
        let start = self.pos;
        let rest = &self.bytes[self.pos..];
        let two = |a: u8, b: u8| rest.len() >= 2 && rest[0] == a && rest[1] == b;
        let (op, len) = if two(b'<', b'<') {
            (Op::Shl, 2)
        } else if two(b'>', b'>') {
            (Op::Shr, 2)
        } else if two(b'=', b'=') {
            (Op::EqEq, 2)
        } else if two(b'!', b'=') {
            (Op::Ne, 2)
        } else if two(b'<', b'=') {
            (Op::Le, 2)
        } else if two(b'>', b'=') {
            (Op::Ge, 2)
        } else if two(b'+', b'=') {
            (Op::PlusAssign, 2)
        } else if two(b'-', b'=') {
            (Op::MinusAssign, 2)
        } else if two(b'*', b'=') {
            (Op::StarAssign, 2)
        } else if two(b'-', b'>') {
            (Op::Arrow, 2)
        } else {
            let op = match rest[0] {
                b'+' => Op::Plus,
                b'-' => Op::Minus,
                b'*' => Op::Star,
                b'/' => Op::Slash,
                b'%' => Op::Percent,
                b'&' => Op::Amp,
                b'|' => Op::Pipe,
                b'^' => Op::Caret,
                b'~' => Op::Tilde,
                b'<' => Op::Lt,
                b'>' => Op::Gt,
                b'=' => Op::Assign,
                b':' => Op::Colon,
                b',' => Op::Comma,
                b';' => Op::Semi,
                b'.' => Op::Dot,
                b'(' => Op::LParen,
                b')' => Op::RParen,
                b'[' => Op::LBracket,
                b']' => Op::RBracket,
                other => {
                    let ch = self.text[self.pos..].chars().next().unwrap_or(other as char);
                    return Err(Diagnostic::new(Span::new(start, start + ch.len_utf8()), format!("unexpected character `{ch}`")));
                }
            };
            (op, 1)
        };
        match op {
            Op::LParen | Op::LBracket => self.depth += 1,
            Op::RParen | Op::RBracket => {
                if self.depth == 0 {
                    return Err(Diagnostic::new(Span::new(start, start + 1), format!("unmatched `{}`", op.text())));
                }
                self.depth -= 1;
            }
            _ => {}
        }
        self.pos += len;
        self.tokens.push(Token { tok: Tok::Op(op), span: Span::new(start, self.pos) });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(text: &str) -> Vec<Tok> {
        lex(text).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn indentation_produces_indent_and_dedent() {
        let toks = kinds("fn a():\n  x = 1\n  if x:\n    y = 2\nfn b():\n  z = 3\n");
        let count = |k: &Tok| toks.iter().filter(|t| *t == k).count();
        assert_eq!(count(&Tok::Indent), 3);
        assert_eq!(count(&Tok::Dedent), 3);
    }

    #[test]
    fn brackets_join_lines() {
        let toks = kinds("x = f(1,\n  2)\n");
        assert_eq!(toks.iter().filter(|t| t.tok_is_newline()).count(), 1);
        assert!(!toks.contains(&Tok::Indent));
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let toks = kinds("# c\n\nfn a():  # trailing\n\n  # inner\n  x = 1\n");
        assert_eq!(toks.iter().filter(|t| **t == Tok::Indent).count(), 1);
    }

    #[test]
    fn numbers() {
        assert_eq!(kinds("1 2.5 1e-5 0x1F")[..4], [Tok::Int(1), Tok::Float(2.5), Tok::Float(1e-5), Tok::Int(31)]);
    }

    #[test]
    fn tabs_rejected() {
        assert!(lex("fn a():\n\tx = 1\n").is_err());
    }

    impl Tok {
        fn tok_is_newline(&self) -> bool {
            matches!(self, Tok::Newline)
        }
    }
}
