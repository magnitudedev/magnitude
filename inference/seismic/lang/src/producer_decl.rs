//! The text of a backend's producer declaration (`producers.decl`): the one
//! statement of which axis values its producers decode. `seismic-lang`
//! reads it for form admission (`producers`) and its build script renders
//! the backend's predicates from it, so both parse it here.
//!
//! ```text
//! producer NAME                      # decode functions of an axis value
//! codes=KIND|.. planes=LOW.HIGH|..   # one pattern per line
//! let NAME = EXPR                    # a name for an expression
//! require EXPR                       # a limit of the weights it reads
//! form NAME uses NAME codes=.. locals=L|..
//!                                    # a form over a producer's codes
//! form NAME                          # a form with its own patterns, each
//! codes=.. planes=.. locals=..       #   also naming the locals it reads
//! ```
//!
//! A field lists the values a pattern takes, `*` or absent for any. `locals`
//! lists `direct`, `plane`, `plane:KIND` or `carried`. A form's `uses`
//! takes the producer's patterns (narrowed to its `codes`, where it names
//! them) and requirements, read with its `locals`. `enter` states how a
//! lane producer's codes enter its lanes: as `pairs` at any zero from the
//! code's own, or as `bytes` at the code's own zero, or at `bytes:ZERO`;
//! a producer of bytes also serves pairs at any zero (`Lanes::enter_at`).
//! `values=i8` takes only weights whose values are signed bytes.
//!
//! A requirement is an expression in C preprocessor syntax (`|| && & == !=
//! < <= > >= + - * %`, `!` and parentheses, on integers) over the axes of a
//! weight's native decode ABI (`CODES`, `COEFF_GROUP`, .., [`Axis`]) and the
//! values of its enumerated axes (`CODES:OFFSET`, `COEFF:DIRECT`,
//! `BIAS_KIND:MINIMUM`), each numbered as the ABI numbers it; a weight a
//! producer reads meets every requirement of the producer.

use std::collections::HashMap;

/// How a lane producer's codes enter its lanes: as pairs, at any zero from
/// the code's own, or as bytes, at the code's own zero or at `zero`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub bytes: bool,
    pub zero: Option<u32>,
}

/// A set of axis values: each field lists the values it takes, empty for any.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pattern {
    pub kinds: Vec<String>,
    pub planes: Vec<(u32, u32)>,
    /// `(placement, kind)`: `plane:unsigned` is `("plane", Some("unsigned"))`.
    pub locals: Vec<(String, Option<String>)>,
    pub entry: Option<Entry>,
    /// Whether it takes only weights whose values are signed bytes
    /// (`values=i8`).
    pub signed_bytes: bool,
}

/// One form's producers: its patterns, the producer's codes crossed with
/// the form's locals where it `uses` one, and their requirements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Form {
    pub name: String,
    pub patterns: Vec<Pattern>,
    pub requires: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Producer {
    pub name: String,
    pub patterns: Vec<Pattern>,
    pub requires: Vec<Expr>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Declaration {
    pub producers: Vec<Producer>,
    pub forms: Vec<Form>,
}

/// An axis of a weight's native decode ABI that a requirement reads
/// (`SEISMIC_AXIS_<NAME>` in a backend's axes tuple), and a native
/// declaration (`axis(W, NAME)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Axis {
    Codes,
    CodeLowBits,
    CodeHighBits,
    CodesOffset,
    Coeff,
    CoeffBias,
    CoeffGroup,
    CoeffFactorGroup,
    CoeffLocal,
    CoeffLocalBits,
    CoeffCarried,
    LogicalGroup,
}

impl Axis {
    pub const ALL: [Self; 12] = [
        Self::Codes,
        Self::CodeLowBits,
        Self::CodeHighBits,
        Self::CodesOffset,
        Self::Coeff,
        Self::CoeffBias,
        Self::CoeffGroup,
        Self::CoeffFactorGroup,
        Self::CoeffLocal,
        Self::CoeffLocalBits,
        Self::CoeffCarried,
        Self::LogicalGroup,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Codes => "CODES",
            Self::CodeLowBits => "CODE_LOW_BITS",
            Self::CodeHighBits => "CODE_HIGH_BITS",
            Self::CodesOffset => "CODES_OFFSET",
            Self::Coeff => "COEFF",
            Self::CoeffBias => "COEFF_BIAS",
            Self::CoeffGroup => "COEFF_GROUP",
            Self::CoeffFactorGroup => "COEFF_FACTOR_GROUP",
            Self::CoeffLocal => "COEFF_LOCAL",
            Self::CoeffLocalBits => "COEFF_LOCAL_BITS",
            Self::CoeffCarried => "COEFF_CARRIED",
            Self::LogicalGroup => "LOGICAL_GROUP",
        }
    }

    /// The axis the ABI names `name`.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|axis| axis.name() == name)
    }
}

/// An enumerated axis whose values a requirement names (`CODES:OFFSET` is
/// `SEISMIC_CODES_OFFSET`): the kinds of codes (also those of local codes),
/// of coefficients and of a bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Codes,
    Coeff,
    BiasKind,
}

impl Domain {
    pub const ALL: [Self; 3] = [Self::Codes, Self::Coeff, Self::BiasKind];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Codes => "CODES",
            Self::Coeff => "COEFF",
            Self::BiasKind => "BIAS_KIND",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Or,
    And,
    BitAnd,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Rem,
}

impl Op {
    /// The operator of `token` with its binding (higher binds tighter), as
    /// C binds it.
    fn of(token: &str) -> Option<(Self, u8)> {
        Some(match token {
            "||" => (Self::Or, 1),
            "&&" => (Self::And, 2),
            "&" => (Self::BitAnd, 3),
            "==" => (Self::Eq, 4),
            "!=" => (Self::Ne, 4),
            "<" => (Self::Lt, 5),
            "<=" => (Self::Le, 5),
            ">" => (Self::Gt, 5),
            ">=" => (Self::Ge, 5),
            "+" => (Self::Add, 6),
            "-" => (Self::Sub, 6),
            "*" => (Self::Mul, 7),
            "%" => (Self::Rem, 7),
            _ => return None,
        })
    }

    fn token(self) -> &'static str {
        match self {
            Self::Or => "||",
            Self::And => "&&",
            Self::BitAnd => "&",
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Rem => "%",
        }
    }
}

/// A requirement's expression, its `let` names substituted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Number(i64),
    Axis(Axis),
    Constant(Domain, String),
    Not(Box<Expr>),
    Binary(Op, Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Its value, as the C preprocessor evaluates it, where `axis` gives
    /// each axis's value and `constant` each enumerated value's.
    // The crate evaluates requirements; the build only renders them.
    #[allow(dead_code)]
    pub fn value(&self, axis: &impl Fn(Axis) -> i64, constant: &impl Fn(Domain, &str) -> i64) -> i64 {
        match self {
            Self::Number(value) => *value,
            Self::Axis(name) => axis(*name),
            Self::Constant(domain, name) => constant(*domain, name),
            Self::Not(operand) => i64::from(operand.value(axis, constant) == 0),
            Self::Binary(op, left, right) => {
                let left = left.value(axis, constant);
                // `||` and `&&` read their right operand only where it decides.
                match op {
                    Op::Or if left != 0 => return 1,
                    Op::And if left == 0 => return 0,
                    _ => {}
                }
                let right = right.value(axis, constant);
                match op {
                    Op::Or | Op::And => i64::from(right != 0),
                    Op::BitAnd => left & right,
                    Op::Eq => i64::from(left == right),
                    Op::Ne => i64::from(left != right),
                    Op::Lt => i64::from(left < right),
                    Op::Le => i64::from(left <= right),
                    Op::Gt => i64::from(left > right),
                    Op::Ge => i64::from(left >= right),
                    Op::Add => left + right,
                    Op::Sub => left - right,
                    Op::Mul => left * right,
                    Op::Rem => left % right,
                }
            }
        }
    }

    /// Its text in C preprocessor syntax, every operation parenthesized,
    /// where `axis` renders each axis and `constant` each enumerated value.
    // The build renders requirements; the crate only evaluates them.
    #[allow(dead_code)]
    pub fn render(&self, axis: &impl Fn(Axis) -> String, constant: &impl Fn(Domain, &str) -> String) -> String {
        match self {
            Self::Number(value) => value.to_string(),
            Self::Axis(name) => axis(*name),
            Self::Constant(domain, name) => constant(*domain, name),
            Self::Not(operand) => format!("(!{})", operand.render(axis, constant)),
            Self::Binary(op, left, right) => {
                format!("({} {} {})", left.render(axis, constant), op.token(), right.render(axis, constant))
            }
        }
    }
}

/// An expression's tokens: numbers, names (`A` or `A:B`) and operators.
fn tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c.is_ascii_alphanumeric() || c == '_' {
            let mut token = String::new();
            while let Some(&c) = chars.peek().filter(|c| c.is_ascii_alphanumeric() || **c == '_' || **c == ':') {
                token.push(c);
                chars.next();
            }
            tokens.push(token);
        } else {
            chars.next();
            let pair = chars
                .peek()
                .map(|next| format!("{c}{next}"))
                .filter(|pair| ["||", "&&", "==", "!=", "<=", ">="].contains(&pair.as_str()));
            if let Some(pair) = pair {
                tokens.push(pair);
                chars.next();
            } else {
                assert!("&<>+-*%!()".contains(c), "no operator `{c}` in a requirement: `{text}`");
                tokens.push(c.to_string());
            }
        }
    }
    tokens
}

/// A requirement's parser over its tokens, with the declaration's `let`
/// names.
struct Parser<'a> {
    tokens: Vec<String>,
    at: usize,
    lets: &'a HashMap<String, Expr>,
    text: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.at).map(String::as_str)
    }

    fn next(&mut self) -> String {
        let token = self.tokens.get(self.at).unwrap_or_else(|| panic!("a requirement ends early: `{}`", self.text)).clone();
        self.at += 1;
        token
    }

    /// An expression of operators binding at least `binding`.
    fn expression(&mut self, binding: u8) -> Expr {
        let mut left = self.operand();
        while let Some((op, bound)) = self.peek().and_then(Op::of).filter(|(_, bound)| *bound >= binding) {
            self.next();
            let right = self.expression(bound + 1);
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        left
    }

    fn operand(&mut self) -> Expr {
        let token = self.next();
        match token.as_str() {
            "!" => Expr::Not(Box::new(self.operand())),
            "(" => {
                let inner = self.expression(0);
                assert_eq!(self.next(), ")", "an unclosed parenthesis in a requirement: `{}`", self.text);
                inner
            }
            _ if token.starts_with(|c: char| c.is_ascii_digit()) => {
                Expr::Number(token.parse().unwrap_or_else(|_| panic!("no number `{token}` in `{}`", self.text)))
            }
            _ => match token.split_once(':') {
                Some((domain, name)) => {
                    let domain = Domain::ALL
                        .into_iter()
                        .find(|candidate| candidate.name() == domain)
                        .unwrap_or_else(|| panic!("no enumerated axis `{domain}` in `{}`", self.text));
                    Expr::Constant(domain, name.to_owned())
                }
                None => self.lets.get(&token).cloned().unwrap_or_else(|| {
                    Expr::Axis(
                        Axis::named(&token).unwrap_or_else(|| panic!("no axis or name `{token}` in `{}`", self.text)),
                    )
                }),
            },
        }
    }
}

fn expression(text: &str, lets: &HashMap<String, Expr>) -> Expr {
    let mut parser = Parser { tokens: tokens(text), at: 0, lets, text };
    let expression = parser.expression(0);
    assert!(parser.peek().is_none(), "a requirement has text past its expression: `{text}`");
    expression
}

fn pattern(fields: &str) -> Pattern {
    let mut pattern = Pattern::default();
    for field in fields.split_whitespace() {
        let (name, values) = field
            .split_once('=')
            .unwrap_or_else(|| panic!("a producer pattern field is `name=values`, not `{field}`"));
        for value in values.split('|').filter(|value| *value != "*") {
            match name {
                "codes" => pattern.kinds.push(value.to_owned()),
                "planes" => {
                    let (low, high) = value.split_once('.').expect("planes are `low.high`");
                    pattern.planes.push((low.parse().expect("low bits"), high.parse().expect("high bits")));
                }
                "locals" => {
                    let (placement, kind) = match value.split_once(':') {
                        Some((placement, kind)) => (placement, Some(kind.to_owned())),
                        None => (value, None),
                    };
                    assert!(["direct", "plane", "carried"].contains(&placement), "no local placement `{placement}`");
                    pattern.locals.push((placement.to_owned(), kind));
                }
                "enter" => {
                    let (layout, zero) = match value.split_once(':') {
                        Some((layout, zero)) => (layout, Some(zero.parse().expect("an entry's zero is a number"))),
                        None => (value, None),
                    };
                    let bytes = match layout {
                        "pairs" => false,
                        "bytes" => true,
                        other => panic!("no entry `{other}`: codes enter as `pairs` or `bytes`"),
                    };
                    assert!(pattern.entry.is_none(), "a pattern's codes enter one way");
                    pattern.entry = Some(Entry { bytes, zero });
                }
                "values" => {
                    assert_eq!(value, "i8", "a pattern limits its values to `i8` or not at all");
                    pattern.signed_bytes = true;
                }
                other => panic!("no producer pattern field `{other}`"),
            }
        }
    }
    pattern
}

enum Section {
    None,
    Producer,
    Form,
}

pub fn parse(text: &str) -> Declaration {
    let mut declaration = Declaration::default();
    let mut lets = HashMap::new();
    let mut section = Section::None;
    for line in text.lines().map(|line| line.split('#').next().unwrap_or("").trim()).filter(|line| !line.is_empty()) {
        if let Some(name) = line.strip_prefix("producer ") {
            declaration.producers.push(Producer { name: name.trim().to_owned(), patterns: Vec::new(), requires: Vec::new() });
            section = Section::Producer;
        } else if let Some(definition) = line.strip_prefix("let ") {
            let (name, value) = definition.split_once('=').expect("`let NAME = EXPR`");
            let name = name.trim().to_owned();
            assert!(Axis::ALL.iter().all(|axis| axis.name() != name), "`let {name}` names an axis");
            let value = expression(value, &lets);
            assert!(lets.insert(name.clone(), value).is_none(), "`let {name}` is stated twice");
        } else if let Some(requirement) = line.strip_prefix("require ") {
            let requirement = expression(requirement, &lets);
            match section {
                Section::Producer => declaration.producers.last_mut().expect("a producer").requires.push(requirement),
                Section::Form => declaration.forms.last_mut().expect("a form").requires.push(requirement),
                Section::None => panic!("a requirement follows its producer or form: `{line}`"),
            }
        } else if let Some(head) = line.strip_prefix("form ") {
            let mut words = head.split_whitespace();
            let name = words.next().expect("a form has a name").to_owned();
            let rest: Vec<&str> = words.collect();
            let (patterns, requires) = if let Some(["uses", used, fields @ ..]) = Some(rest.as_slice()).filter(|rest| rest.first() == Some(&"uses")) {
                let producer = declaration
                    .producers
                    .iter()
                    .find(|producer| producer.name == *used)
                    .unwrap_or_else(|| panic!("form `{name}` uses no producer `{used}`"));
                let read = pattern(&fields.join(" "));
                assert!(
                    read.planes.is_empty() && read.entry.is_none() && !read.signed_bytes,
                    "form `{name}` narrows its producer's codes and names its locals alone"
                );
                // The producer's patterns of the codes the form names, each
                // narrowed to them.
                let patterns = producer
                    .patterns
                    .iter()
                    .filter_map(|codes| {
                        let kinds = match (read.kinds.is_empty(), codes.kinds.is_empty()) {
                            (true, _) => codes.kinds.clone(),
                            (false, true) => read.kinds.clone(),
                            (false, false) => codes.kinds.iter().filter(|kind| read.kinds.contains(kind)).cloned().collect(),
                        };
                        (read.kinds.is_empty() || !kinds.is_empty())
                            .then(|| Pattern { kinds, locals: read.locals.clone(), ..codes.clone() })
                    })
                    .collect::<Vec<_>>();
                assert!(!patterns.is_empty(), "form `{name}` takes none of `{used}`'s codes");
                section = Section::None;
                (patterns, producer.requires.clone())
            } else {
                assert!(rest.is_empty(), "`form {name}` is followed by `uses` or its patterns");
                section = Section::Form;
                (Vec::new(), Vec::new())
            };
            assert!(declaration.forms.iter().all(|form| form.name != name), "form `{name}` is declared twice");
            declaration.forms.push(Form { name, patterns, requires });
        } else {
            let pattern = pattern(line);
            match section {
                Section::Producer => declaration.producers.last_mut().expect("a producer").patterns.push(pattern),
                Section::Form => declaration.forms.last_mut().expect("a form").patterns.push(pattern),
                Section::None => panic!("a pattern follows its producer or form: `{line}`"),
            }
        }
    }
    for producer in &declaration.producers {
        let entered = producer.patterns.iter().filter(|pattern| pattern.entry.is_some()).count();
        assert!(
            entered == 0 || entered == producer.patterns.len(),
            "producer `{}` states how every pattern's codes enter, or none",
            producer.name
        );
    }
    declaration
}
