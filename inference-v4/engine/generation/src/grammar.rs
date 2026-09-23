//! Family-neutral GBNF conversion over parsed productions. Renaming never touches literal text;
//! one-character classes remain Unicode scalar sets, including escaped controls.
mod regular;
use std::collections::{BTreeMap, BTreeSet};

const MAX_SOURCE: usize = 8 * 1024 * 1024;
const MAX_NODES: usize = 100_000;
const MAX_DEPTH: usize = 256;
pub const CONVERTER_IDENTITY: &str = "magnitude-gbnf-rust-1-llguidance-1.8.0";

#[derive(Clone, Debug)]
enum Expr {
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
type Rules = BTreeMap<String, Expr>;
impl Expr {
    fn references(&self, names: &mut BTreeSet<String>) {
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
            _ => {}
        }
    }
    fn rename(&mut self, names: &BTreeMap<String, String>) {
        match self {
            Self::Reference(name) => *name = names[name].clone(),
            Self::Sequence(parts) | Self::Alternative(parts) => {
                for part in parts {
                    part.rename(names);
                }
            }
            Self::Repeat(part, _, _) => part.rename(names),
            _ => {}
        }
    }
    fn render(&self) -> String {
        match self {
            Self::Literal(text) => serde_json::to_string(text).unwrap(),
            Self::Class { negated, ranges } => format!(
                "/[{}{}]/",
                if *negated { "^" } else { "" },
                ranges
                    .iter()
                    .map(|(a, b)| {
                        if a == b {
                            format!("\\x{{{:x}}}", *a as u32)
                        } else {
                            format!("\\x{{{:x}}}-\\x{{{:x}}}", *a as u32, *b as u32)
                        }
                    })
                    .collect::<String>()
            ),
            Self::Any => "/[\\x{0}-\\x{10ffff}]/".into(),
            Self::Reference(name) => name.clone(),
            Self::Sequence(parts) => {
                if parts.is_empty() {
                    "\"\"".into()
                } else {
                    parts.iter().map(Self::render).collect::<Vec<_>>().join(" ")
                }
            }
            Self::Alternative(parts) => format!(
                "({})",
                parts
                    .iter()
                    .map(Self::render)
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            Self::Repeat(part, min, max) => format!(
                "({}){}",
                part.render(),
                match (*min, *max) {
                    (0, None) => "*".into(),
                    (1, None) => "+".into(),
                    (0, Some(1)) => "?".into(),
                    (min, Some(max)) if min == max => format!("{{{min}}}"),
                    (min, Some(max)) => format!("{{{min},{max}}}"),
                    (min, None) => format!("{{{min},}}"),
                }
            ),
        }
    }
}

/// Convert admitted GBNF to llguidance Lark. The source is parsed and checked
/// before transformations, so undefined rules and malformed escapes cannot be
/// disguised by renaming. No Python executable is used at runtime.
pub fn to_lark(source: &str) -> Result<String, String> {
    let mut rules = Parser::parse(source)?;
    let names: BTreeMap<_, _> = rules
        .keys()
        .enumerate()
        .map(|(index, name)| {
            (
                name.clone(),
                if name == "root" {
                    "root".into()
                } else {
                    format!("g{index}")
                },
            )
        })
        .collect();
    rules = rules
        .into_iter()
        .map(|(name, mut expr)| {
            expr.rename(&names);
            (names[&name].clone(), expr)
        })
        .collect();
    if let Some(whole) = regular::whole_completion(&rules) {
        return Ok(format!(
            "%llguidance {{}}\nstart: WHOLE_COMPLETION\nWHOLE_COMPLETION: {whole}\n"
        ));
    }
    regular::orient(&mut rules);
    let mut output = String::from("%llguidance {}\nstart: root\n");
    for (name, expr) in rules {
        output.push_str(&format!("{name}: {}\n", expr.render()));
    }
    Ok(output)
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
