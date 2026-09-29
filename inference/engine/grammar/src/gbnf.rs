//! GBNF syntax. Parsing checks the source before any lowering, so undefined
//! rules and malformed escapes cannot be disguised by later transformations.
//! One-character classes stay Unicode scalar sets, including escaped controls.
use crate::GrammarError;
use std::collections::{BTreeMap, BTreeSet};

const MAX_SOURCE: usize = 8 * 1024 * 1024;
const MAX_NODES: usize = 100_000;
pub(crate) const MAX_DEPTH: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Expr {
    Literal(String),
    Class {
        negated: bool,
        ranges: Vec<(char, char)>,
    },
    Any,
    Reference(String),
    Sequence(Vec<Expr>),
    Alternative(Vec<Expr>),
    Repeat(Box<Expr>, u32, Option<u32>),
}

pub(crate) type Rules = BTreeMap<String, Expr>;

impl Expr {
    pub(crate) fn references(&self, names: &mut BTreeSet<String>) {
        match self {
            Self::Reference(name) => {
                names.insert(name.clone());
            }
            Self::Sequence(parts) | Self::Alternative(parts) => {
                for part in parts {
                    part.references(names);
                }
            }
            Self::Repeat(part, _, _) => part.references(names),
            Self::Literal(_) | Self::Class { .. } | Self::Any => {}
        }
    }
}

pub(crate) fn parse(source: &str) -> Result<Rules, GrammarError> {
    Parser::parse(source).map_err(GrammarError::Syntax)
}

struct Parser<'a> {
    source: &'a str,
    at: usize,
    nodes: usize,
}

impl<'a> Parser<'a> {
    fn parse(source: &'a str) -> Result<Rules, String> {
        if source.is_empty() || source.len() > MAX_SOURCE {
            return Err("GBNF source must be between 1 byte and 8 MiB".into());
        }
        let mut parser = Self {
            source,
            at: 0,
            nodes: 0,
        };
        let mut rules = Rules::new();
        parser.whitespace(true);
        while parser.peek().is_some() {
            let name = parser.name()?;
            parser.whitespace(false);
            if !parser.source[parser.at..].starts_with("::=") {
                return parser.error("expected ::=");
            }
            parser.at += 3;
            parser.whitespace(true);
            let body = parser.alternatives(false, 0)?;
            if rules.insert(name, body).is_some() {
                return parser.error("duplicate rule");
            }
            parser.whitespace(true);
        }
        if !rules.contains_key("root") {
            return Err("GBNF grammar has no root rule".into());
        }
        let mut references = BTreeSet::new();
        for expr in rules.values() {
            expr.references(&mut references);
        }
        for name in references {
            if !rules.contains_key(&name) {
                return Err(format!("undefined GBNF rule: {name}"));
            }
        }
        Ok(rules)
    }
    fn error<T>(&self, message: &str) -> Result<T, String> {
        Err(format!("GBNF byte {}: {message}", self.at))
    }
    fn peek(&self) -> Option<char> {
        self.source[self.at..].chars().next()
    }
    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.at += c.len_utf8();
        Some(c)
    }
    fn whitespace(&mut self, newlines: bool) {
        loop {
            match self.peek() {
                Some(' ' | '\t') => {
                    self.next();
                }
                Some('\r' | '\n') if newlines => {
                    self.next();
                }
                Some('#') => {
                    while self.peek().is_some_and(|c| c != '\r' && c != '\n') {
                        self.next();
                    }
                }
                _ => break,
            }
        }
    }
    fn name(&mut self) -> Result<String, String> {
        let start = self.at;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            self.next();
        }
        if start == self.at {
            self.error("expected rule name")
        } else {
            Ok(self.source[start..self.at].into())
        }
    }
    fn node(&mut self, node: Expr) -> Result<Expr, String> {
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return self.error("grammar exceeds node limit");
        }
        Ok(node)
    }
    fn alternatives(&mut self, nested: bool, depth: usize) -> Result<Expr, String> {
        if depth > MAX_DEPTH {
            return self.error("grammar nesting exceeds limit");
        }
        let mut options = vec![self.sequence(nested, depth)?];
        loop {
            let before = self.at;
            self.whitespace(true);
            if self.peek() != Some('|') {
                self.at = before;
                break;
            }
            self.next();
            self.whitespace(true);
            options.push(self.sequence(nested, depth)?);
        }
        if options.len() == 1 {
            Ok(options.pop().unwrap())
        } else {
            self.node(Expr::Alternative(options))
        }
    }
    fn sequence(&mut self, nested: bool, depth: usize) -> Result<Expr, String> {
        let mut parts = Vec::new();
        loop {
            self.whitespace(nested);
            match self.peek() {
                None | Some('|' | ')' | '\r' | '\n') => break,
                _ => {}
            }
            let mut part = match self.next().unwrap() {
                '"' => {
                    let mut text = String::new();
                    while self.peek() != Some('"') {
                        if self.peek().is_none() {
                            return self.error("unterminated literal");
                        }
                        text.push(self.character()?);
                    }
                    self.next();
                    self.node(Expr::Literal(text))?
                }
                '[' => {
                    let negated = self.peek() == Some('^');
                    if negated {
                        self.next();
                    }
                    let mut ranges = Vec::new();
                    while self.peek() != Some(']') {
                        if self.peek().is_none() {
                            return self.error("unterminated character class");
                        }
                        let start = self.character()?;
                        let end = if self.peek() == Some('-')
                            && !self.source[self.at..].starts_with("-]")
                        {
                            self.next();
                            self.character()?
                        } else {
                            start
                        };
                        if start > end {
                            return self.error("reversed character range");
                        }
                        ranges.push((start, end));
                    }
                    self.next();
                    if ranges.is_empty() {
                        return self.error("empty character class");
                    }
                    self.node(Expr::Class { negated, ranges })?
                }
                '.' => self.node(Expr::Any)?,
                '(' => {
                    self.whitespace(true);
                    let expr = self.alternatives(true, depth + 1)?;
                    self.whitespace(true);
                    if self.next() != Some(')') {
                        return self.error("expected closing parenthesis");
                    }
                    expr
                }
                c if c.is_ascii_alphanumeric() || c == '_' || c == '-' => {
                    self.at -= c.len_utf8();
                    let name = self.name()?;
                    self.node(Expr::Reference(name))?
                }
                _ => return self.error("unexpected production character"),
            };
            self.whitespace(false);
            let repeat = match self.peek() {
                Some('*') => {
                    self.next();
                    Some((0, None))
                }
                Some('+') => {
                    self.next();
                    Some((1, None))
                }
                Some('?') => {
                    self.next();
                    Some((0, Some(1)))
                }
                Some('{') => {
                    self.next();
                    self.whitespace(false);
                    let min = self.number()?;
                    self.whitespace(false);
                    let max = if self.peek() == Some(',') {
                        self.next();
                        self.whitespace(false);
                        if self.peek() == Some('}') {
                            None
                        } else {
                            Some(self.number()?)
                        }
                    } else {
                        Some(min)
                    };
                    self.whitespace(false);
                    if self.next() != Some('}') || max.is_some_and(|max| max < min) {
                        return self.error("invalid repetition bounds");
                    }
                    Some((min, max))
                }
                _ => None,
            };
            if let Some((min, max)) = repeat {
                part = self.node(Expr::Repeat(Box::new(part), min, max))?;
            }
            parts.push(part);
        }
        if parts.len() == 1 {
            Ok(parts.pop().unwrap())
        } else {
            self.node(Expr::Sequence(parts))
        }
    }
    fn number(&mut self) -> Result<u32, String> {
        let start = self.at;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.next();
        }
        self.source[start..self.at]
            .parse()
            .map_err(|_| format!("GBNF byte {start}: invalid repetition count"))
    }
    fn character(&mut self) -> Result<char, String> {
        let c = self.next().ok_or("unterminated GBNF character")?;
        if c == '\r' || c == '\n' {
            return self.error("literal newlines must be escaped");
        }
        if c != '\\' {
            return Ok(c);
        }
        match self.next().ok_or("unterminated GBNF escape")? {
            'n' => Ok('\n'),
            'r' => Ok('\r'),
            't' => Ok('\t'),
            c @ ('"' | '\\' | '[' | ']' | '-' | '/') => Ok(c),
            c @ ('x' | 'u' | 'U') => {
                let digits = match c {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                let mut scalar = 0;
                for _ in 0..digits {
                    let digit = self
                        .next()
                        .and_then(|c| c.to_digit(16))
                        .ok_or("invalid GBNF Unicode escape")?;
                    scalar = scalar * 16 + digit;
                }
                char::from_u32(scalar).ok_or_else(|| "GBNF escape is not a Unicode scalar".into())
            }
            _ => self.error("unknown character escape"),
        }
    }
}
