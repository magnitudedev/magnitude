//! Recursive-descent parser with Pratt expression parsing.
use super::ast::*;
use super::lexer::lex;
use super::token::{Kw, Op, Tok, Token};
use crate::check::ir::MAX_NESTING;
use crate::checked::{ConstantBody, ConstantFunction, DiagnosticRule, Encoded};
use crate::span::{Diagnostic, Span};
use std::collections::BTreeMap;

/// Parse one file of a source set whose module constants are `constants`
/// and constant functions `functions` (every file's `const` declarations): a
/// name of a constant in an expression is its integer literal, and a binding
/// may not take one's name; a call of a constant function is its body at the
/// call's arguments.
pub(crate) fn parse(
    text: &str,
    constants: &BTreeMap<String, u64>,
    functions: &BTreeMap<String, ConstantFunction>,
) -> Result<File, Diagnostic> {
    let tokens = lex(text)?;
    Parser { tokens, pos: 0, constants: constants.clone(), functions: functions.clone(), open: 0 }.file()
}

/// `body`, a constant function's, as an expression at `span`, each of its
/// parameters (`parameters`, in order) the argument in its place.
fn called(body: &ConstantBody, parameters: &[String], arguments: &[Expr], span: Span) -> Expr {
    let binary = |op, left: &ConstantBody, right: &ConstantBody| ExprKind::Binary {
        op,
        lhs: Box::new(called(left, parameters, arguments, span)),
        rhs: Box::new(called(right, parameters, arguments, span)),
    };
    let kind = match body {
        ConstantBody::Constant(value) => ExprKind::Int(*value),
        ConstantBody::Parameter(name) => {
            let parameter = parameters
                .iter()
                .position(|parameter| parameter == name)
                .unwrap_or_else(|| panic!("a constant function reads only its parameters, not `{name}`"));
            return arguments[parameter].clone();
        }
        ConstantBody::Add(left, right) => binary(BinaryOp::Add, left, right),
        ConstantBody::Sub(left, right) => binary(BinaryOp::Sub, left, right),
        ConstantBody::Mul(left, right) => binary(BinaryOp::Mul, left, right),
        ConstantBody::Div(left, right) => binary(BinaryOp::Div, left, right),
        ConstantBody::Rem(left, right) => binary(BinaryOp::Rem, left, right),
    };
    Expr { kind, span }
}

/// Binding powers, shared with the printer. Binary operators use twice their `precedence()`
/// so that `..` fits between the comparisons and every arithmetic and bit operator.
pub(super) fn binary_bp(op: BinaryOp) -> u8 {
    op.precedence() * 2
}
pub(super) const RANGE_BP: u8 = 9;
pub(super) const NOT_BP: u8 = NOT_PRECEDENCE * 2;
pub(super) const UNARY_BP: u8 = UNARY_PRECEDENCE * 2;

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    constants: BTreeMap<String, u64>,
    functions: BTreeMap<String, ConstantFunction>,
    /// The statements, expressions, patterns and types open on the parse's
    /// path ([`Parser::nested`]).
    open: usize,
}

/// How deep `expr` nests: its levels on its deepest path, each expression
/// one, as a checked body counts them. Measured without recursion, so a
/// tree of any depth is measured on the parser's stack.
fn nesting(expr: &Expr) -> usize {
    let mut deepest = 0;
    let mut open: Vec<(&Expr, usize)> = vec![(expr, 1)];
    while let Some((expr, level)) = open.pop() {
        deepest = deepest.max(level);
        let below: Vec<&Expr> = match &expr.kind {
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Inf | ExprKind::Bool(_) | ExprKind::Name(_) => Vec::new(),
            ExprKind::Tuple(items) => items.iter().collect(),
            ExprKind::Range { lo, hi } => vec![&**lo, &**hi],
            ExprKind::Tensor { shape, .. } => shape.iter().collect(),
            ExprKind::Call { callee, bindings, args } => std::iter::once(&**callee)
                .chain(bindings.iter().map(|(_, value)| value))
                .chain(args.iter().map(|arg| &arg.value))
                .collect(),
            ExprKind::Index { base, indices } => std::iter::once(&**base)
                .chain(indices.iter().flat_map(|index| match index {
                    Index::Expr(value) => vec![value],
                    Index::Slice { start, end } => start.iter().chain(end).collect(),
                }))
                .collect(),
            ExprKind::Attr { base, .. } => vec![&**base],
            ExprKind::Unary { expr, .. } => vec![&**expr],
            ExprKind::Binary { lhs, rhs, .. } => vec![&**lhs, &**rhs],
        };
        open.extend(below.into_iter().map(|child| (child, level + 1)));
    }
    deepest
}

type PResult<T> = Result<T, Diagnostic>;

fn binary_op(tok: &Tok) -> Option<BinaryOp> {
    Some(match tok {
        Tok::Kw(Kw::Or) => BinaryOp::Or,
        Tok::Kw(Kw::And) => BinaryOp::And,
        Tok::Op(Op::EqEq) => BinaryOp::Eq,
        Tok::Op(Op::Ne) => BinaryOp::Ne,
        Tok::Op(Op::Lt) => BinaryOp::Lt,
        Tok::Op(Op::Le) => BinaryOp::Le,
        Tok::Op(Op::Gt) => BinaryOp::Gt,
        Tok::Op(Op::Ge) => BinaryOp::Ge,
        Tok::Op(Op::Pipe) => BinaryOp::BitOr,
        Tok::Op(Op::Caret) => BinaryOp::BitXor,
        Tok::Op(Op::Amp) => BinaryOp::BitAnd,
        Tok::Op(Op::Shl) => BinaryOp::Shl,
        Tok::Op(Op::Shr) => BinaryOp::Shr,
        Tok::Op(Op::Plus) => BinaryOp::Add,
        Tok::Op(Op::Minus) => BinaryOp::Sub,
        Tok::Op(Op::Star) => BinaryOp::Mul,
        Tok::Op(Op::Slash) => BinaryOp::Div,
        Tok::Op(Op::Percent) => BinaryOp::Rem,
        _ => return None,
    })
}

fn is_place(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Name(_) | ExprKind::Index { .. } | ExprKind::Attr { .. } => true,
        ExprKind::Tuple(items) => items.iter().all(is_place),
        _ => false,
    }
}

/// Top-level `and` conjuncts of a `where` clause.
fn conjuncts(e: Expr, out: &mut Vec<Expr>) {
    match e.kind {
        ExprKind::Binary {
            op: BinaryOp::And,
            lhs,
            rhs,
        } => {
            conjuncts(*lhs, out);
            conjuncts(*rhs, out);
        }
        _ => out.push(e),
    }
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn peek_at(&self, offset: usize) -> &Tok {
        let i = (self.pos + offset).min(self.tokens.len() - 1);
        &self.tokens[i].tok
    }

    fn span(&self) -> Span {
        self.tokens[self.pos].span
    }

    fn prev_span(&self) -> Span {
        self.tokens[self.pos.saturating_sub(1)].span
    }

    fn bump(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn at_op(&self, op: Op) -> bool {
        matches!(self.peek(), Tok::Op(o) if *o == op)
    }

    fn at_kw(&self, kw: Kw) -> bool {
        matches!(self.peek(), Tok::Kw(k) if *k == kw)
    }

    /// A contextual word: an ordinary name recognized by position.
    fn at_word(&self, word: &str) -> bool {
        matches!(self.peek(), Tok::Name(n) if n == word)
    }

    fn at_line_end(&self) -> bool {
        matches!(self.peek(), Tok::Newline | Tok::Dedent | Tok::Eof)
    }

    fn eat_op(&mut self, op: Op) -> bool {
        if self.at_op(op) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn eat_kw(&mut self, kw: Kw) -> bool {
        if self.at_kw(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_op(&mut self, op: Op) -> PResult<Span> {
        if self.at_op(op) {
            Ok(self.bump().span)
        } else {
            Err(self.error(format!(
                "expected `{}`, found {}",
                op.text(),
                self.peek().describe()
            )))
        }
    }

    fn expect_kw(&mut self, kw: Kw) -> PResult<Span> {
        if self.at_kw(kw) {
            Ok(self.bump().span)
        } else {
            Err(self.error(format!(
                "expected `{}`, found {}",
                kw.text(),
                self.peek().describe()
            )))
        }
    }

    fn expect_name(&mut self) -> PResult<Ident> {
        match self.peek().clone() {
            Tok::Name(name) => {
                let span = self.bump().span;
                Ok(Ident { name, span })
            }
            Tok::Kw(kw) => Err(self.error(format!(
                "expected a name, found `{}`, which is reserved",
                kw.text()
            ))),
            other => Err(self.error(format!("expected a name, found {}", other.describe()))),
        }
    }

    fn expect_newline(&mut self) -> PResult<()> {
        match self.peek() {
            Tok::Newline => {
                self.bump();
                Ok(())
            }
            Tok::Eof | Tok::Dedent => Ok(()),
            other => Err(self.error(format!("expected end of line, found {}", other.describe()))),
        }
    }

    fn error(&self, message: String) -> Diagnostic {
        Diagnostic::with_rule(DiagnosticRule::Syntax, self.span(), message)
    }

    /// `parse` with one more statement, expression, pattern or type open on
    /// the parse's path: refused past [`MAX_NESTING`], which bounds every
    /// walk of a body (this parse, the checker's, monomorphization's, the
    /// encode's and the decode's) by construction. Every recursion of the
    /// parser opens one; a chain that nests without it recursing (binary
    /// operators, postfix accesses) is measured where its expression ends
    /// ([`Parser::expr`]).
    fn nested<T>(&mut self, parse: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<T> {
        if self.open == MAX_NESTING {
            return Err(self.too_deep(self.span()));
        }
        self.open += 1;
        let parsed = parse(self);
        self.open -= 1;
        parsed
    }

    fn too_deep(&self, span: Span) -> Diagnostic {
        Diagnostic::with_rule(
            DiagnosticRule::Nesting,
            span,
            format!("nested past the {MAX_NESTING} statements and expressions that bound a program's stack"),
        )
    }

    /// `item ("," item)*`
    fn comma_list<T>(&mut self, mut item: impl FnMut(&mut Self) -> PResult<T>) -> PResult<Vec<T>> {
        let mut items = vec![item(self)?];
        while self.eat_op(Op::Comma) {
            items.push(item(self)?);
        }
        Ok(items)
    }

    // ---- declarations ----

    fn file(&mut self) -> PResult<File> {
        let mut decls = Vec::new();
        let mut constants = Vec::new();
        let mut constant_functions = Vec::new();
        loop {
            while matches!(self.peek(), Tok::Newline) {
                self.bump();
            }
            match self.peek() {
                Tok::Eof => break,
                Tok::Kw(Kw::Const) if matches!(self.peek_at(2), Tok::Op(Op::LParen)) => {
                    constant_functions.push(self.const_fn_decl()?)
                }
                Tok::Kw(Kw::Const) => constants.push(self.const_decl()?),
                Tok::Kw(Kw::Fn) => decls.push(Decl::Fn(self.fn_decl()?)),
                Tok::Kw(Kw::Lower) => decls.push(Decl::Lower(self.lower_decl()?)),
                Tok::Kw(Kw::Native) => decls.push(Decl::Native(self.native_decl()?)),
                Tok::Name(word) if word == "projection" => {
                    decls.push(Decl::Projection(self.projection_decl()?))
                }
                Tok::Indent => {
                    return Err(self.error(
                        "unexpected indentation; declarations start at the left margin".into(),
                    ));
                }
                other => {
                    return Err(self.error(format!(
                        "expected `const`, `fn`, `lower`, `native`, or `projection`, found {}",
                        other.describe()
                    )));
                }
            }
        }
        Ok(File { decls, constants, constant_functions })
    }

    /// `const NAME = INTEGER`.
    fn const_decl(&mut self) -> PResult<ConstDecl> {
        let start = self.expect_kw(Kw::Const)?;
        let name = self.expect_name()?;
        self.expect_op(Op::Assign)?;
        let value = match self.peek().clone() {
            Tok::Int(value) => {
                self.bump();
                value
            }
            other => {
                return Err(self.error(format!(
                    "a module constant is an integer literal, found {}",
                    other.describe()
                )));
            }
        };
        let span = start.to(self.prev_span());
        self.expect_newline()?;
        Ok(ConstDecl { name, value, span })
    }

    /// `const NAME(PARAMETER, ..) = EXPR`.
    fn const_fn_decl(&mut self) -> PResult<ConstFnDecl> {
        let start = self.expect_kw(Kw::Const)?;
        let name = self.expect_name()?;
        self.expect_op(Op::LParen)?;
        let parameters = if self.at_op(Op::RParen) { Vec::new() } else { self.comma_list(Self::binding_name)? };
        self.expect_op(Op::RParen)?;
        self.expect_op(Op::Assign)?;
        let body = self.expr()?;
        let span = start.to(self.prev_span());
        self.expect_newline()?;
        Ok(ConstFnDecl { name, parameters, body, span })
    }

    /// A name a declaration binds (a parameter, a dimension, a `let` or
    /// `for` target, a native term, parameter or scratch buffer): never a
    /// module constant's.
    fn binding_name(&mut self) -> PResult<Ident> {
        let name = self.expect_name()?;
        if self.constants.contains_key(&name.name) {
            return Err(Diagnostic::with_rule(
                DiagnosticRule::Resolution,
                name.span,
                format!("`{}` is a module constant and cannot be bound", name.name),
            ));
        }
        Ok(name)
    }

    fn fn_decl(&mut self) -> PResult<FnDecl> {
        let start = self.span();
        self.expect_kw(Kw::Fn)?;
        let name = self.expect_name()?;
        let mut continued = false;
        let mut signature = self.signature(&mut continued, true)?;
        self.continue_header(&mut continued, true);
        if self.at_kw(Kw::For) {
            return Err(self.error(
                "backend code is written as `lower NAME … for BACKEND`; a `fn` is portable".into(),
            ));
        }
        let requires = self.requires_clause(&mut continued)?;
        signature.predicates = self.where_clause(&mut continued, true)?;
        self.expect_op(Op::Colon)?;
        let body = self.body(continued)?;
        Ok(FnDecl {
            signature,
            name,
            requires,
            body,
            span: start.to(self.prev_span()),
        })
    }

    fn lower_decl(&mut self) -> PResult<LowerDecl> {
        let start = self.expect_kw(Kw::Lower)?;
        let name = self.expect_name()?;
        let mut continued = false;
        self.continue_header(&mut continued, true);
        let signature = self.signature(&mut continued, true)?;
        self.continue_header(&mut continued, true);
        if !self.at_kw(Kw::For) {
            return Err(self.error(format!(
                "expected `for <target>` in a lowering, found {}",
                self.peek().describe()
            )));
        }
        self.bump();
        let target = self.expect_name()?;
        let requires = self.requires_clause(&mut continued)?;
        let predicates = self.where_clause(&mut continued, true)?;
        self.expect_op(Op::Colon)?;
        let body = self.body(continued)?;
        Ok(LowerDecl {
            name,
            signature,
            target,
            requires,
            predicates,
            body,
            span: start.to(self.prev_span()),
        })
    }

    fn native_decl(&mut self) -> PResult<NativeDecl> {
        let start = self.expect_kw(Kw::Native)?;
        let function = self.expect_name()?;
        self.expect_kw(Kw::For)?;
        let target = self.expect_name()?;
        if !self.at_word("from") {
            return Err(self.error(format!(
                "expected `from <source>`, found {}",
                self.peek().describe()
            )));
        }
        self.bump();
        let source = match self.peek().clone() {
            Tok::String(value) => {
                self.bump();
                value
            }
            other => {
                return Err(self.error(format!(
                    "expected a native source string, found {}",
                    other.describe()
                )));
            }
        };
        self.expect_op(Op::Colon)?;
        self.native_block_start("native declaration")?;
        let mut statics = Vec::new();
        if self.at_word("static") {
            self.bump();
            self.expect_op(Op::LParen)?;
            statics = self.comma_list(Self::expect_name)?;
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
        }
        let mut params = Vec::new();
        if self.at_word("params") {
            self.bump();
            self.expect_op(Op::LParen)?;
            params = self.comma_list(Self::native_param)?;
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
        }
        let mut elements = Vec::new();
        if self.at_word("elements") {
            self.bump();
            self.expect_op(Op::LParen)?;
            elements = self.comma_list(Self::native_elements)?;
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
        }
        let mut records = Vec::new();
        if self.at_word("records") {
            self.bump();
            self.expect_op(Op::LParen)?;
            records = self.comma_list(Self::native_record)?;
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
        }
        let mut terms = Vec::new();
        while matches!(self.peek(), Tok::Kw(Kw::Let)) {
            let begin = self.bump().span;
            let name = self.binding_name()?;
            self.expect_op(Op::Assign)?;
            let value = self.expr()?;
            self.expect_newline()?;
            terms.push(NativeTermDecl {
                name,
                value,
                span: begin.to(self.prev_span()),
            });
        }
        let mut constraint = None;
        if self.eat_kw(Kw::Where) {
            constraint = Some(self.expr()?);
            self.expect_newline()?;
        }
        let mut scratch = Vec::new();
        let mut project = None;
        loop {
            if self.at_word("scratch") {
                scratch.push(self.native_scratch()?);
                self.expect_newline()?;
            } else if self.at_word("project") && project.is_none() {
                let begin = self.bump().span;
                let projection = self.expect_name()?;
                // `project NAME emits (..)`: a producer's line, which takes
                // only the projection's emitted buffers.
                if self.at_word("emits") && matches!(self.peek_at(1), Tok::Op(Op::LParen)) {
                    let variant = Ident { name: "emits".to_owned(), span: self.span() };
                    let emits = self.project_emits()?;
                    project = Some(NativeProjectDecl {
                        projection,
                        variant,
                        weights: Vec::new(),
                        weight_overs: Vec::new(),
                        weight_planes: Vec::new(),
                        over: None,
                        served: Vec::new(),
                        shared: Vec::new(),
                        down: None,
                        groups: None,
                        parts: None,
                        mappings: Vec::new(),
                        gathered: None,
                        rows: None,
                        publishes: None,
                        ordered: Vec::new(),
                        guard: Vec::new(),
                        restated: Vec::new(),
                        operands: Vec::new(),
                        emits,
                        scratch_before: scratch.len(),
                        span: begin.to(self.prev_span()),
                    });
                    self.expect_newline()?;
                    continue;
                }
                let variant = self.expect_name()?;
                self.expect_op(Op::LParen)?;
                let mut weights = Vec::new();
                let mut weight_overs = Vec::new();
                let mut weight_planes = Vec::new();
                let mut served = Vec::new();
                let mut grouped = false;
                loop {
                    // A plane set in place of a weight: `planes (PLANE, ..)
                    // over (ROWS, WIDTH)`.
                    if self.at_word("planes") && matches!(self.peek_at(1), Tok::Op(Op::LParen)) {
                        let name = Ident { name: "planes".to_owned(), span: self.bump().span };
                        self.expect_op(Op::LParen)?;
                        weight_planes.push(self.comma_list(Self::expect_name)?);
                        self.expect_op(Op::RParen)?;
                        weights.push(name);
                        weight_overs.push(self.project_over()?);
                        served.push(1);
                    } else if self.eat_op(Op::LParen) {
                        let pair = self.comma_list(|parser| {
                            let name = parser.expect_name()?;
                            Ok((name, parser.project_over()?))
                        })?;
                        self.expect_op(Op::RParen)?;
                        served.push(pair.len());
                        for (name, over) in pair {
                            weights.push(name);
                            weight_overs.push(over);
                            weight_planes.push(Vec::new());
                        }
                        grouped = true;
                    } else {
                        weights.push(self.expect_name()?);
                        weight_overs.push(self.project_over()?);
                        weight_planes.push(Vec::new());
                        served.push(1);
                    }
                    if !self.eat_op(Op::Comma) {
                        break;
                    }
                }
                if !grouped {
                    served.clear();
                }
                self.expect_op(Op::RParen)?;
                let over = self.project_over()?;
                // A routed projection's second weights: the shared expert's
                // (`shared (..)`) or the expansion's down weight (`down (..)`).
                let mut shared = Vec::new();
                if self.at_word("shared") {
                    self.bump();
                    self.expect_op(Op::LParen)?;
                    shared = self.comma_list(Self::expect_name)?;
                    self.expect_op(Op::RParen)?;
                }
                let down = if self.at_word("down") {
                    self.bump();
                    self.expect_op(Op::LParen)?;
                    let down = self.expect_name()?;
                    self.expect_op(Op::RParen)?;
                    Some(down)
                } else {
                    None
                };
                let groups = if self.at_word("groups") {
                    self.bump();
                    Some(self.expect_name()?)
                } else {
                    None
                };
                let parts = if self.at_word("parts") {
                    self.bump();
                    Some(self.expect_name()?)
                } else {
                    None
                };
                let gathered = if self.at_word("gathered") {
                    self.bump();
                    Some(self.expect_name()?)
                } else {
                    None
                };
                let rows = if self.at_word("rows") {
                    self.bump();
                    Some(self.expect_name()?)
                } else {
                    None
                };
                let publishes = if self.at_word("publishes") {
                    let begin = self.bump().span;
                    self.expect_op(Op::LParen)?;
                    let result = match (self.at_word("result"), self.peek_at(1).clone()) {
                        (true, Tok::Int(ordinal)) => {
                            self.bump();
                            self.bump();
                            ordinal
                        }
                        _ => {
                            return Err(self.error(format!(
                                "a published result is named `result N`, found {}",
                                self.peek().describe()
                            )));
                        }
                    };
                    self.expect_op(Op::RParen)?;
                    Some(NativePublishDecl { result, span: begin.to(self.prev_span()) })
                } else {
                    None
                };
                let ordered = if self.at_word("ordered") {
                    self.bump();
                    self.expect_op(Op::LParen)?;
                    let ordered = self.comma_list(Self::expect_name)?;
                    self.expect_op(Op::RParen)?;
                    ordered
                } else {
                    Vec::new()
                };
                let operands = if self.at_word("operand") {
                    self.bump();
                    self.expect_op(Op::LParen)?;
                    let operands = self.comma_list(|parser| {
                        let source = if parser.at_word("emitted") {
                            OperandSourceDecl::Emitted(parser.bump().span)
                        } else if parser.at_word("values") {
                            parser.bump();
                            parser.expect_op(Op::LParen)?;
                            let function = parser.expect_name()?;
                            parser.expect_op(Op::RParen)?;
                            OperandSourceDecl::Values(function)
                        } else {
                            let parameter = parser.expect_name()?;
                            let at = if parser.at_word("at") {
                                parser.bump();
                                parser.expect_op(Op::LParen)?;
                                let at = parser.expect_name()?;
                                parser.expect_op(Op::RParen)?;
                                Some(at)
                            } else {
                                None
                            };
                            let within = if parser.at_word("within") {
                                parser.bump();
                                parser.expect_op(Op::LParen)?;
                                let first = parser.expect_name()?;
                                parser.expect_op(Op::Comma)?;
                                let count = parser.expect_name()?;
                                parser.expect_op(Op::RParen)?;
                                Some([first, count])
                            } else {
                                None
                            };
                            let norm = parser.operand_norm()?;
                            OperandSourceDecl::Parameter { parameter, at, within, norm }
                        };
                        let tiles = parser.operand_clause("tiles")?;
                        Ok(NativeOperandDecl { source, tiles })
                    })?;
                    self.expect_op(Op::RParen)?;
                    operands
                } else {
                    Vec::new()
                };
                let emits = if self.at_word("emits") { self.project_emits()? } else { Vec::new() };
                project = Some(NativeProjectDecl {
                    projection,
                    variant,
                    weights,
                    weight_overs,
                    weight_planes,
                    over,
                    served,
                    shared,
                    down,
                    groups,
                    parts,
                    mappings: Vec::new(),
                    gathered,
                    rows,
                    publishes,
                    ordered,
                    guard: Vec::new(),
                    restated: Vec::new(),
                    operands,
                    emits,
                    scratch_before: scratch.len(),
                    span: begin.to(self.prev_span()),
                });
                self.expect_newline()?;
            } else if project.is_some() && (self.at_kw(Kw::If) || self.at_word("unless")) {
                let conditions = self.native_conditions()?;
                if let Some(project) = project.as_mut() {
                    project.guard.extend(conditions);
                }
            } else if self.at_word("restate") && project.is_some() {
                self.bump();
                // `restate MAPPING` takes a projection's mapping.
                if matches!(self.peek(), Tok::Name(_)) && matches!(self.peek_at(1), Tok::Newline) {
                    let mapping = self.expect_name()?;
                    if let Some(project) = project.as_mut() {
                        project.mappings.push(mapping);
                    }
                    self.expect_newline()?;
                    continue;
                }
                // `restate NAME in ..` (or with a flag first) restates an entry-wide parameter;
                // `restate LAUNCH NAME in ..` a launch's.
                let entry_wide = self.at_word("arithmetic")
                    || self.at_word("form")
                    || matches!(self.peek_at(1), Tok::Kw(Kw::In));
                let launch = if entry_wide { None } else { Some(self.expect_name()?) };
                let param = self.native_param()?;
                if let Some(project) = project.as_mut() {
                    project.restated.push(NativeRestateDecl { launch, param });
                }
                self.expect_newline()?;
            } else {
                break;
            }
        }
        let mut launches = Vec::new();
        let mut repeat = None;
        loop {
            if self.at_word("launch") {
                launches.push(self.native_launch()?);
            } else if self.at_word("repeat") && repeat.is_none() {
                let begin = self.bump().span;
                self.expect_op(Op::LParen)?;
                let count = self.expr()?;
                self.expect_op(Op::RParen)?;
                self.expect_op(Op::Colon)?;
                self.native_block_start("repeat")?;
                let first = launches.len();
                while self.at_word("launch") {
                    launches.push(self.native_launch()?);
                }
                if launches.len() == first {
                    return Err(self.error(format!(
                        "expected `launch <kernel>:` in the repeat block, found {}",
                        self.peek().describe()
                    )));
                }
                self.native_block_end("repeat")?;
                repeat = Some(NativeRepeatDecl {
                    count,
                    first,
                    launches: launches.len() - first,
                    span: begin.to(self.prev_span()),
                });
            } else {
                break;
            }
        }
        // A projecting declaration's launches are its projection's; it may
        // add none of its own.
        if launches.is_empty() && project.is_none() {
            return Err(self.error(format!(
                "expected `launch <kernel>:`, found {}; a native declaration lists `static`, `params`, `elements`, `let`, `where`, `scratch`, then one or more launches",
                self.peek().describe()
            )));
        }
        self.native_block_end("native declaration")?;
        Ok(NativeDecl {
            function,
            target,
            source,
            statics,
            params,
            elements,
            records,
            terms,
            constraint,
            scratch,
            project,
            projected_rows: None,
            projected_segments: None,
            projected_ordered: Vec::new(),
            formation: None,
            launches,
            repeat,
            span: start.to(self.prev_span()),
        })
    }

    /// `scratch NAME bytes (EXPR) [sync] [when CONDITION]`, before its line's end.
    fn native_scratch(&mut self) -> PResult<NativeScratchDecl> {
        let begin = self.bump().span;
        let name = self.binding_name()?;
        if !self.at_word("bytes") {
            return Err(self.error(format!(
                "expected `bytes (<size>)` after the scratch name, found {}",
                self.peek().describe()
            )));
        }
        self.bump();
        self.expect_op(Op::LParen)?;
        let bytes = self.expr()?;
        self.expect_op(Op::RParen)?;
        let sync = self.at_word("sync");
        if sync {
            self.bump();
        }
        let when = self.native_when()?;
        Ok(NativeScratchDecl {
            name,
            bytes,
            sync,
            when,
            operand: None,
            emission: None,
            span: begin.to(self.prev_span()),
        })
    }

    /// A buffer of a projection: a native one (`native_scratch`), then
    /// `operand [rows] [when PAIRS] [formed (COND)] [nvfp4 (COND)]` where it
    /// holds a form of an operand of the entry (`ScratchOperandDecl`). An
    /// entry's own buffers hold none: the operator owns every form.
    fn projection_scratch(&mut self) -> PResult<NativeScratchDecl> {
        let mut scratch = self.native_scratch()?;
        if self.at_word("operand") {
            self.bump();
            let rows = self.at_word("rows");
            if rows {
                self.bump();
            }
            let when = self.native_when()?;
            let formed = self.operand_clause("formed")?;
            let nvfp4 = self.operand_clause("nvfp4")?;
            scratch.operand = Some(ScratchOperandDecl::Declared { rows, when, formed, nvfp4 });
            scratch.span = scratch.span.to(self.prev_span());
        }
        Ok(scratch)
    }

    /// An optional `WORD (CONDITION)` of an operand buffer or of a `project`
    /// line's operand.
    fn operand_clause(&mut self, word: &str) -> PResult<Option<Expr>> {
        if !self.at_word(word) {
            return Ok(None);
        }
        self.bump();
        self.expect_op(Op::LParen)?;
        let condition = self.expr()?;
        self.expect_op(Op::RParen)?;
        Ok(Some(condition))
    }

    /// An optional `normed (WEIGHT, EPSILON)` of an operand.
    fn operand_norm(&mut self) -> PResult<Option<OperandNormDecl>> {
        if !self.at_word("normed") {
            return Ok(None);
        }
        let begin = self.bump().span;
        self.expect_op(Op::LParen)?;
        let weight = self.expect_name()?;
        self.expect_op(Op::Comma)?;
        let epsilon = self.expect_name()?;
        self.expect_op(Op::RParen)?;
        Ok(Some(OperandNormDecl { weight, epsilon, span: begin.to(self.prev_span()) }))
    }

    /// `emits (result N over (ROWS, WIDTH), ..)` of a `project` line.
    fn project_emits(&mut self) -> PResult<Vec<NativeEmitDecl>> {
        self.bump();
        self.expect_op(Op::LParen)?;
        let emits = self.comma_list(|parser| {
            let begin = parser.span();
            let result = match (parser.at_word("result"), parser.peek_at(1).clone()) {
                (true, Tok::Int(ordinal)) => {
                    parser.bump();
                    parser.bump();
                    ordinal
                }
                _ => {
                    return Err(parser.error(format!(
                        "an emitted operand names its result, `result N over (ROWS, WIDTH)`, found {}",
                        parser.peek().describe()
                    )));
                }
            };
            let Some(over) = parser.project_over()? else {
                return Err(parser.error(format!(
                    "expected `over (ROWS, WIDTH)` after `result {result}`, found {}",
                    parser.peek().describe()
                )));
            };
            let norm = parser.operand_norm()?;
            Ok(NativeEmitDecl { result, over, norm, span: begin.to(parser.prev_span()) })
        })?;
        self.expect_op(Op::RParen)?;
        Ok(emits)
    }

    /// An optional `over (ROWS, WIDTH)` of a `project` line.
    fn project_over(&mut self) -> PResult<Option<NativeProjectOver>> {
        if !self.at_word("over") {
            return Ok(None);
        }
        self.bump();
        self.expect_op(Op::LParen)?;
        let rows = self.expr()?;
        self.expect_op(Op::Comma)?;
        let width = self.expr()?;
        self.expect_op(Op::RParen)?;
        Ok(Some(NativeProjectOver { rows, width }))
    }

    /// An optional `for VARIANT | ..`: the projection variants an item
    /// belongs to (none: every one).
    fn variants(&mut self) -> PResult<Vec<Ident>> {
        let mut variants = Vec::new();
        if self.eat_kw(Kw::For) {
            variants.push(self.expect_name()?);
            while self.eat_op(Op::Pipe) {
                variants.push(self.expect_name()?);
            }
        }
        Ok(variants)
    }

    fn gated<T>(&mut self, item: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<Gated<T>> {
        let item = item(self)?;
        let variants = self.variants()?;
        Ok(Gated { item, variants })
    }

    /// `projection NAME for TARGET:` and its indented body
    /// ([`ProjectionDecl`]).
    fn projection_decl(&mut self) -> PResult<ProjectionDecl> {
        let start = self.bump().span;
        let name = self.expect_name()?;
        self.expect_kw(Kw::For)?;
        let target = self.expect_name()?;
        self.expect_op(Op::Colon)?;
        self.native_block_start("projection")?;
        let mut terms = Vec::new();
        while matches!(self.peek(), Tok::Kw(Kw::Let)) {
            terms.push(self.gated(|parser| {
                let begin = parser.bump().span;
                let name = parser.binding_name()?;
                parser.expect_op(Op::Assign)?;
                let value = parser.expr()?;
                Ok(NativeTermDecl {
                    name,
                    value,
                    span: begin.to(parser.prev_span()),
                })
            })?);
            self.expect_newline()?;
        }
        let mut params = Vec::new();
        if self.at_word("params") {
            self.bump();
            self.expect_op(Op::LParen)?;
            params = self.comma_list(|parser| parser.gated(Self::native_param))?;
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
        }
        let mut constraints = Vec::new();
        while self.eat_kw(Kw::Where) {
            constraints.push(self.gated(Self::expr)?);
            self.expect_newline()?;
        }
        let mut scratch = Vec::new();
        loop {
            if self.at_word("scratch") {
                scratch.push(ProjectionScratchDecl::One(self.gated(Self::projection_scratch)?));
                self.expect_newline()?;
            } else if self.at_word("each") {
                self.bump();
                let formal = self.expect_name()?;
                if formal.name != "W" {
                    return Err(Diagnostic::with_rule(
                        crate::checked::DiagnosticRule::Syntax,
                        formal.span,
                        format!("`each` ranges over the weights, `W`, not `{}`", formal.name),
                    ));
                }
                self.expect_op(Op::Colon)?;
                self.native_block_start("each")?;
                let mut each = Vec::new();
                while self.at_word("scratch") {
                    each.push(self.gated(Self::projection_scratch)?);
                    self.expect_newline()?;
                }
                if each.is_empty() {
                    return Err(self.error(format!(
                        "expected `scratch` in the `each W` block, found {}",
                        self.peek().describe()
                    )));
                }
                self.native_block_end("each")?;
                scratch.push(ProjectionScratchDecl::Each(each));
            } else {
                break;
            }
        }
        let mut launches = Vec::new();
        while self.at_word("launch") {
            launches.push(self.launch(true)?);
        }
        let mut mappings = Vec::new();
        while self.at_word("mapping") {
            let begin = self.bump().span;
            let name = self.expect_name()?;
            self.expect_op(Op::Colon)?;
            self.native_block_start("mapping")?;
            let mut restated = Vec::new();
            while self.at_word("restate") {
                self.bump();
                let entry_wide = self.at_word("arithmetic")
                    || self.at_word("form")
                    || matches!(self.peek_at(1), Tok::Kw(Kw::In));
                let launch = if entry_wide { None } else { Some(self.expect_name()?) };
                let param = self.native_param()?;
                restated.push(NativeRestateDecl { launch, param });
                self.expect_newline()?;
            }
            if restated.is_empty() {
                return Err(self.error(format!(
                    "expected `restate` in the `mapping {}` block, found {}",
                    name.name,
                    self.peek().describe()
                )));
            }
            self.native_block_end("mapping")?;
            mappings.push(ProjectionMapping { name, restated, span: begin.to(self.prev_span()) });
        }
        self.native_block_end("projection")?;
        Ok(ProjectionDecl {
            name,
            target,
            terms,
            params,
            constraints,
            scratch,
            launches,
            mappings,
            span: start.to(self.prev_span()),
        })
    }

    fn native_block_start(&mut self, what: &str) -> PResult<()> {
        self.expect_newline()?;
        if !matches!(self.peek(), Tok::Indent) {
            return Err(self.error(format!("expected an indented {what} body")));
        }
        self.bump();
        Ok(())
    }

    fn native_block_end(&mut self, what: &str) -> PResult<()> {
        match self.peek() {
            Tok::Dedent => {
                self.bump();
                Ok(())
            }
            Tok::Eof => Ok(()),
            other => Err(self.error(format!("unexpected {} in {what}", other.describe()))),
        }
    }

    /// `[arithmetic] [form] NAME in [V, ..] default DEFAULT`, the default
    /// `EXPR`, `least where CONDITION` or `greatest where CONDITION`.
    fn native_param(&mut self) -> PResult<NativeParamDecl> {
        let begin = self.span();
        let mut arithmetic = false;
        let mut form = false;
        loop {
            if self.at_word("arithmetic") && !arithmetic {
                self.bump();
                arithmetic = true;
            } else if self.at_word("form") && !form {
                self.bump();
                form = true;
            } else {
                break;
            }
        }
        let name = self.binding_name()?;
        self.expect_kw(Kw::In)?;
        self.expect_op(Op::LBracket)?;
        let values = self.comma_list(|parser| match parser.peek().clone() {
            Tok::Int(value) => {
                parser.bump();
                Ok(value)
            }
            other => Err(parser.error(format!(
                "a native parameter domain lists integer literals, found {}",
                other.describe()
            ))),
        })?;
        self.expect_op(Op::RBracket)?;
        if !self.at_word("default") {
            return Err(self.error(format!(
                "expected `default` after the values of `{}`: every native parameter states its default (`default EXPR`, `default least where CONDITION` or `default greatest where CONDITION`), found {}",
                name.name,
                self.peek().describe()
            )));
        }
        self.bump();
        let default = if self.at_word("least") || self.at_word("greatest") {
            let least = self.at_word("least");
            self.bump();
            self.expect_kw(Kw::Where)?;
            let condition = self.expr()?;
            if least {
                NativeDefaultDecl::Least(condition)
            } else {
                NativeDefaultDecl::Greatest(condition)
            }
        } else {
            NativeDefaultDecl::Value(self.expr()?)
        };
        Ok(NativeParamDecl {
            name,
            arithmetic,
            form,
            values,
            default,
            span: begin.to(self.prev_span()),
        })
    }

    /// `NAME in [DTYPE, ..]`
    fn native_elements(&mut self) -> PResult<NativeElementsDecl> {
        let begin = self.span();
        let name = self.expect_name()?;
        self.expect_kw(Kw::In)?;
        self.expect_op(Op::LBracket)?;
        let dtypes = self.comma_list(Self::expect_name)?;
        self.expect_op(Op::RBracket)?;
        Ok(NativeElementsDecl {
            name,
            dtypes,
            span: begin.to(self.prev_span()),
        })
    }

    /// A record port: `NAME` (a parameter) or `result N`.
    fn native_record(&mut self) -> PResult<NativeRecordDecl> {
        let name = self.expect_name()?;
        match (name.name.as_str(), self.peek().clone()) {
            ("result", Tok::Int(ordinal)) => {
                self.bump();
                Ok(NativeRecordDecl::Result {
                    ordinal,
                    span: name.span.to(self.prev_span()),
                })
            }
            _ => Ok(NativeRecordDecl::Parameter(name)),
        }
    }

    /// An optional `when CONDITION` (`when` is a contextual word).
    fn native_when(&mut self) -> PResult<Option<Expr>> {
        if !self.at_word("when") {
            return Ok(None);
        }
        self.bump();
        Ok(Some(self.expr()?))
    }

    /// `launch KERNEL [before LAUNCH] [resident (BARRIER)] [when CONDITION]:`
    /// followed by its indented geometry.
    fn native_launch(&mut self) -> PResult<NativeLaunchDecl> {
        Ok(self.launch(false)?.launch)
    }

    /// A native launch or (`projection`) a projection's: `launch NAME [for
    /// VARIANT] [when CONDITION]:`, its parameters each `[for VARIANT]`.
    fn launch(&mut self, projection: bool) -> PResult<ProjectionLaunchDecl> {
        let begin = self.bump().span;
        let kernel = self.expect_name()?;
        let variants = if projection { self.variants()? } else { Vec::new() };
        let before = if !projection && self.at_word("before") {
            self.bump();
            Some(self.expect_name()?)
        } else {
            None
        };
        let resident = if self.at_word("resident") {
            self.bump();
            self.expect_op(Op::LParen)?;
            let barrier = self.expect_name()?;
            self.expect_op(Op::RParen)?;
            Some(barrier)
        } else {
            None
        };
        let when = self.native_when()?;
        self.expect_op(Op::Colon)?;
        self.native_block_start("launch")?;
        let mut params = Vec::new();
        let mut param_variants = Vec::new();
        if self.at_word("params") {
            self.bump();
            self.expect_op(Op::LParen)?;
            for param in self.comma_list(|parser| {
                let param = parser.native_param()?;
                let variants = if projection { parser.variants()? } else { Vec::new() };
                Ok((param, variants))
            })? {
                params.push(param.0);
                param_variants.push(param.1);
            }
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
        }
        let mut constraint = None;
        if self.eat_kw(Kw::Where) {
            constraint = Some(self.expr()?);
            self.expect_newline()?;
        }
        let reads = if self.at_word("reads") {
            self.bump();
            self.expect_op(Op::LParen)?;
            let reads = self.comma_list(Self::expect_name)?;
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
            reads
        } else {
            Vec::new()
        };
        let alike = if self.at_word("alike") {
            self.bump();
            self.expect_op(Op::LParen)?;
            let alike = self.expect_name()?;
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
            Some(alike)
        } else {
            None
        };
        let staging = if self.at_word("staging") {
            self.bump();
            self.expect_newline()?;
            true
        } else {
            false
        };
        let records = if self.at_word("records") {
            self.bump();
            self.expect_op(Op::LParen)?;
            let records = self.comma_list(Self::native_record)?;
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
            Some(records)
        } else {
            None
        };
        let threadgroups = self.native_launch_property("threadgroups")?;
        self.expect_newline()?;
        let grid = if self.at_word("grid") {
            self.bump();
            self.expect_op(Op::LParen)?;
            let grid = self.expect_name()?;
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
            Some(grid)
        } else {
            None
        };
        let conditions = self.native_conditions()?;
        let threads_per_threadgroup = self.native_launch_property("threads_per_threadgroup")?;
        self.expect_newline()?;
        let shared_bytes = if self.at_word("shared_bytes") {
            self.bump();
            self.expect_op(Op::LParen)?;
            let bytes = self.expr()?;
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
            Some(bytes)
        } else {
            None
        };
        self.native_block_end("launch")?;
        let launch = NativeLaunchDecl {
            kernel,
            before,
            resident,
            when,
            params,
            constraint,
            reads,
            alike,
            staging,
            records,
            threadgroups,
            grid,
            conditions,
            threads_per_threadgroup,
            shared_bytes,
            span: begin.to(self.prev_span()),
        };
        Ok(ProjectionLaunchDecl {
            launch,
            variants,
            param_variants,
        })
    }

    /// The `if (WORD)` and `unless (WORD)` lines at the cursor, a launch's or
    /// a `project` line's.
    fn native_conditions(&mut self) -> PResult<Vec<NativeConditionDecl>> {
        let mut conditions = Vec::new();
        loop {
            let encoded = if self.at_kw(Kw::If) {
                Encoded::WhileSet
            } else if self.at_word("unless") {
                Encoded::WhileClear
            } else {
                return Ok(conditions);
            };
            self.bump();
            self.expect_op(Op::LParen)?;
            let word = self.expect_name()?;
            let index = if self.eat_op(Op::LBracket) {
                let index = self.expr()?;
                self.expect_op(Op::RBracket)?;
                Some(index)
            } else {
                None
            };
            self.expect_op(Op::RParen)?;
            self.expect_newline()?;
            conditions.push(NativeConditionDecl { encoded, word, index });
        }
    }

    fn native_launch_property(&mut self, expected: &str) -> PResult<[Expr; 3]> {
        if !self.at_word(expected) {
            return Err(self.error(format!(
                "expected `{expected} (x, y, z)`, found {}",
                self.peek().describe()
            )));
        }
        self.bump();
        self.expect_op(Op::LParen)?;
        let x = self.expr()?;
        self.expect_op(Op::Comma)?;
        let y = self.expr()?;
        self.expect_op(Op::Comma)?;
        let z = self.expr()?;
        self.expect_op(Op::RParen)?;
        Ok([x, y, z])
    }

    /// A header may continue on one deeper-indented line (and further lines at that
    /// indentation) starting with `->`, `for`, `requires` or `where`. The
    /// `Indent` consumed here is closed by `body` or `end_header`.
    fn continue_header(&mut self, continued: &mut bool, lower: bool) {
        if !matches!(self.peek(), Tok::Newline) {
            return;
        }
        let skip = if *continued {
            1
        } else if matches!(self.peek_at(1), Tok::Indent) {
            2
        } else {
            return;
        };
        let continues = match self.peek_at(skip) {
            Tok::Op(Op::Arrow) | Tok::Kw(Kw::Requires) | Tok::Kw(Kw::Where) => true,
            Tok::Kw(Kw::For) => lower,
            _ => false,
        };
        if continues {
            self.pos += skip;
            *continued = true;
        }
    }

    /// `[Shape, ..](params) [-> type]`; predicates are filled by the caller.
    fn signature(&mut self, continued: &mut bool, lower: bool) -> PResult<Signature> {
        let mut shape = Vec::new();
        if self.eat_op(Op::LBracket) {
            shape = self.comma_list(Self::binding_name)?;
            self.expect_op(Op::RBracket)?;
        }
        self.expect_op(Op::LParen)?;
        let mut params = Vec::new();
        while !self.at_op(Op::RParen) {
            params.push(self.param()?);
            if !self.eat_op(Op::Comma) {
                break;
            }
        }
        self.expect_op(Op::RParen)?;
        self.continue_header(continued, lower);
        let mut result = None;
        if self.eat_op(Op::Arrow) {
            let ty = self.type_expr()?;
            // `-> void` and an absent result are the same signature.
            if ty.kind != TypeKind::Void {
                result = Some(ty);
            }
        }
        Ok(Signature {
            shape,
            params,
            result,
            predicates: Vec::new(),
        })
    }

    fn where_clause(&mut self, continued: &mut bool, lower: bool) -> PResult<Vec<Expr>> {
        self.continue_header(continued, lower);
        let mut predicates = Vec::new();
        if self.eat_kw(Kw::Where) {
            conjuncts(self.expr()?, &mut predicates);
        }
        Ok(predicates)
    }

    fn requires_clause(&mut self, continued: &mut bool) -> PResult<Vec<CapabilityPath>> {
        self.continue_header(continued, true);
        if !self.eat_kw(Kw::Requires) {
            return Ok(Vec::new());
        }
        self.comma_list(|parser| {
            let backend = parser.expect_name()?;
            parser.expect_op(Op::Dot)?;
            let capability = parser.expect_name()?;
            let span = backend.span.to(capability.span);
            Ok(CapabilityPath {
                backend,
                capability,
                span,
            })
        })
    }

    fn param(&mut self) -> PResult<Param> {
        let name = self.binding_name()?;
        self.expect_op(Op::Colon)?;
        let ty = self.type_expr()?;
        Ok(Param { name, ty })
    }

    fn type_expr(&mut self) -> PResult<TypeExpr> {
        let start = self.span();
        let kind = match self.peek().clone() {
            Tok::Op(Op::Amp) => {
                self.bump();
                // `host`, `await` and `slab` are contextual: after `&`, only
                // `mut` or `tensor` could follow otherwise.
                let qualifier = ["host", "await", "slab"]
                    .into_iter()
                    .find(|word| self.at_word(word));
                if qualifier.is_some() {
                    self.bump();
                }
                let mutable = self.eat_kw(Kw::Mut);
                let tensor = self.expect_name()?;
                if tensor.name != "tensor" || !self.at_op(Op::LBracket) {
                    return Err(Diagnostic::with_rule(
                        DiagnosticRule::Syntax,
                        tensor.span,
                        "a borrow type is `&tensor[...] T` or `&mut tensor[...] T`, optionally qualified `&host`, `&await` or `&slab`",
                    ));
                }
                self.shaped(match (qualifier, mutable) {
                    (Some("host"), write) => ShapedHead::HostTensor { write },
                    (Some("await"), write) => ShapedHead::AwaitTensor { write },
                    (Some("slab"), write) => ShapedHead::SlabTensor { write },
                    (Some(_), _) => unreachable!("a borrow qualifier is one of its words"),
                    (None, true) => ShapedHead::MutTensor,
                    (None, false) => ShapedHead::SharedTensor,
                })?
            }
            Tok::Kw(Kw::Void) => {
                self.bump();
                TypeKind::Void
            }
            Tok::Op(Op::LParen) => {
                self.bump();
                let mut items = self.comma_list(|parser| parser.nested(Self::type_expr))?;
                self.expect_op(Op::RParen)?;
                if items.len() == 1 {
                    return Ok(items.remove(0));
                }
                TypeKind::Tuple(items)
            }
            Tok::Name(word) => {
                let name = self.expect_name()?;
                match self.peek() {
                    Tok::Op(Op::LBracket) if word == "tensor" => self.shaped(ShapedHead::Tensor)?,
                    Tok::Op(Op::LBracket) if word == "index" => {
                        self.bump();
                        let bound = self.expr()?;
                        self.expect_op(Op::RBracket)?;
                        TypeKind::Index(Box::new(bound))
                    }
                    Tok::Op(Op::LBracket) if word == "range" => {
                        self.bump();
                        let bound = self.expr()?;
                        self.expect_op(Op::RBracket)?;
                        TypeKind::Range(Box::new(bound))
                    }
                    Tok::Op(Op::LBracket) => {
                        return Err(self.error(format!(
                            "`{word}` takes no shape; the shaped type is `tensor`"
                        )));
                    }
                    Tok::Op(Op::Dot) => {
                        return Err(Diagnostic::with_rule(
                            DiagnosticRule::Syntax,
                            name.span,
                            "backend-native types are compiler-internal and cannot appear in source",
                        ));
                    }
                    _ => TypeKind::Scalar(name),
                }
            }
            other => return Err(self.error(format!("expected a type, found {}", other.describe()))),
        };
        Ok(TypeExpr {
            kind,
            span: start.to(self.prev_span()),
        })
    }

    /// `[shape] elem` after a shaped head.
    fn shaped(&mut self, head: ShapedHead) -> PResult<TypeKind> {
        let (shape, elem) = self.shape_and_elem()?;
        Ok(TypeKind::Shaped { head, shape, elem })
    }

    fn shape_and_elem(&mut self) -> PResult<(Vec<Expr>, Ident)> {
        self.expect_op(Op::LBracket)?;
        let shape = self.comma_list(Self::expr)?;
        self.expect_op(Op::RBracket)?;
        if !matches!(self.peek(), Tok::Name(_)) {
            return Err(self.error(format!(
                "expected an element type after the shape, found {}",
                self.peek().describe()
            )));
        }
        Ok((shape, self.expect_name()?))
    }

    // ---- blocks ----

    /// A block after `:`. Either an indented suite or simple statements on the same line.
    fn block(&mut self) -> PResult<Block> {
        if matches!(self.peek(), Tok::Newline) {
            self.bump();
            if !matches!(self.peek(), Tok::Indent) {
                return Err(self.error("expected an indented block".into()));
            }
            self.bump();
            return self.suite();
        }
        let start = self.span();
        let mut stmts = Vec::new();
        self.simple_statements(&mut stmts)?;
        Ok(Block {
            stmts,
            span: start.to(self.prev_span()),
        })
    }

    /// Statement lines up to and including the `Dedent` closing an already-open `Indent`.
    fn suite(&mut self) -> PResult<Block> {
        let start = self.span();
        let mut stmts = Vec::new();
        loop {
            match self.peek() {
                Tok::Newline => {
                    self.bump();
                }
                Tok::Dedent | Tok::Eof => break,
                _ => self.statement_line(&mut stmts)?,
            }
        }
        if stmts.is_empty() {
            return Err(self.error("expected a statement".into()));
        }
        let span = start.to(self.prev_span());
        if matches!(self.peek(), Tok::Dedent) {
            self.bump();
        }
        Ok(Block { stmts, span })
    }

    /// A declaration body after the header's `:`. A continued header already opened the
    /// block, so the body carries on at the continuation's indentation.
    fn body(&mut self, continued: bool) -> PResult<Block> {
        if continued {
            self.suite()
        } else {
            self.block()
        }
    }

    // ---- statements ----

    fn statement_line(&mut self, out: &mut Vec<Stmt>) -> PResult<()> {
        let start = self.span();
        let kind = match self.peek() {
            Tok::Kw(Kw::For) => {
                self.bump();
                self.nested(|parser| parser.logical_for(false))?
            }
            Tok::Kw(Kw::Parallel) if self.peek_at(1) == &Tok::Kw(Kw::For) => {
                self.bump();
                self.bump();
                self.nested(|parser| parser.logical_for(true))?
            }
            Tok::Kw(Kw::If) => self.nested(Self::if_stmt)?,
            Tok::Kw(Kw::Else) => return Err(self.error("`else` without a matching `if`".into())),
            Tok::Indent => return Err(self.error("unexpected indentation".into())),
            _ => return self.simple_statements(out),
        };
        out.push(Stmt {
            kind,
            span: start.to(self.prev_span()),
        });
        Ok(())
    }

    /// Parse the common tail of ordered `for` and independent `parallel for`.
    fn logical_for(&mut self, parallel: bool) -> PResult<StmtKind> {
        let targets = self.comma_list(Self::binding_name)?;
        self.expect_kw(Kw::In)?;
        let iter = self.expr()?;
        self.expect_op(Op::Colon)?;
        Ok(StmtKind::For {
            parallel,
            targets,
            iter,
            body: self.block()?,
        })
    }

    /// At `if`. `else if` nests an `if` as the only statement of the `else` block.
    fn if_stmt(&mut self) -> PResult<StmtKind> {
        self.expect_kw(Kw::If)?;
        let cond = self.expr()?;
        self.expect_op(Op::Colon)?;
        let then = self.block()?;
        let mut els = None;
        if self.eat_kw(Kw::Else) {
            els = Some(if self.at_kw(Kw::If) {
                let start = self.span();
                let kind = self.nested(Self::if_stmt)?;
                let span = start.to(self.prev_span());
                Block {
                    stmts: vec![Stmt { kind, span }],
                    span,
                }
            } else {
                self.expect_op(Op::Colon)?;
                self.block()?
            });
        }
        Ok(StmtKind::If { cond, then, els })
    }

    /// `a`, `a, b`, `(a, b)`, nested.
    fn pattern(&mut self) -> PResult<Pattern> {
        let mut items = self.comma_list(Self::pattern_atom)?;
        Ok(if items.len() == 1 {
            items.remove(0)
        } else {
            Pattern::Tuple(items)
        })
    }

    fn pattern_atom(&mut self) -> PResult<Pattern> {
        if self.eat_op(Op::LParen) {
            let inner = self.nested(Self::pattern)?;
            self.expect_op(Op::RParen)?;
            return Ok(inner);
        }
        Ok(Pattern::Name(self.binding_name()?))
    }

    /// `simple (";" simple)* NEWLINE`. A statement whose value is a region ends its own line.
    fn simple_statements(&mut self, out: &mut Vec<Stmt>) -> PResult<()> {
        loop {
            let (stmt, ended) = self.simple_statement()?;
            out.push(stmt);
            if ended {
                return Ok(());
            }
            if !self.eat_op(Op::Semi) || self.at_line_end() {
                break;
            }
        }
        self.expect_newline()
    }

    /// The statement and whether a region value already consumed the end of its line.
    fn simple_statement(&mut self) -> PResult<(Stmt, bool)> {
        self.nested(Self::open_simple_statement)
    }

    /// [`Parser::simple_statement`], open on the parse's path.
    fn open_simple_statement(&mut self) -> PResult<(Stmt, bool)> {
        let start = self.span();
        let mut ended = false;
        let kind = match self.peek() {
            Tok::Kw(Kw::Let) => {
                self.bump();
                let mutable = self.eat_kw(Kw::Mut);
                let pattern = self.pattern()?;
                self.expect_op(Op::Assign)?;
                let value = self.value(&mut ended)?;
                StmtKind::Let {
                    mutable,
                    pattern,
                    value,
                }
            }
            Tok::Kw(Kw::Return) => {
                self.bump();
                let values = if self.at_line_end() || self.at_op(Op::Semi) {
                    Vec::new()
                } else {
                    self.comma_list(Self::expr)?
                };
                StmtKind::Return(values)
            }
            _ => {
                let target = self.expr()?;
                let op = match self.peek() {
                    Tok::Op(Op::Assign) => Some(AssignOp::Assign),
                    Tok::Op(Op::PlusAssign) => Some(AssignOp::Add),
                    Tok::Op(Op::MinusAssign) => Some(AssignOp::Sub),
                    Tok::Op(Op::StarAssign) => Some(AssignOp::Mul),
                    _ => None,
                };
                match op {
                    Some(op) => {
                        if !is_place(&target) {
                            return Err(Diagnostic::with_rule(
                                DiagnosticRule::Syntax,
                                target.span,
                                "cannot assign to this expression; a target is a name, an indexed place or a tuple of them",
                            ));
                        }
                        self.bump();
                        StmtKind::Assign {
                            target,
                            op,
                            value: self.expr()?,
                        }
                    }
                    None => StmtKind::Expr(target),
                }
            }
        };
        Ok((
            Stmt {
                kind,
                span: start.to(self.prev_span()),
            },
            ended,
        ))
    }

    /// The value of `let`: an expression.
    fn value(&mut self, ended: &mut bool) -> PResult<Expr> {
        *ended = false;
        self.expr()
    }

    // ---- expressions ----

    /// An expression, refused where it nests, under what is open above it,
    /// past [`MAX_NESTING`].
    fn expr(&mut self) -> PResult<Expr> {
        let expr = self.nested(|parser| parser.expr_bp(0))?;
        if self.open + nesting(&expr) > MAX_NESTING {
            return Err(self.too_deep(expr.span));
        }
        Ok(expr)
    }

    fn expr_bp(&mut self, min_bp: u8) -> PResult<Expr> {
        let mut lhs = self.unary()?;
        loop {
            let op = binary_op(self.peek());
            let bp = match op {
                Some(op) => binary_bp(op),
                None if self.at_op(Op::DotDot) => RANGE_BP,
                None => break,
            };
            if bp <= min_bp {
                break;
            }
            self.bump();
            let rhs = self.expr_bp(bp)?;
            let span = lhs.span.to(rhs.span);
            let kind = match op {
                Some(op) => ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                None => {
                    if self.at_op(Op::DotDot) {
                        return Err(self.error("`..` does not chain; a domain is `lo..hi`".into()));
                    }
                    ExprKind::Range {
                        lo: Box::new(lhs),
                        hi: Box::new(rhs),
                    }
                }
            };
            lhs = Expr { kind, span };
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> PResult<Expr> {
        let (op, bp) = match self.peek() {
            Tok::Op(Op::Minus) => (UnaryOp::Neg, UNARY_BP),
            Tok::Op(Op::Tilde) => (UnaryOp::BitNot, UNARY_BP),
            Tok::Kw(Kw::Not) => (UnaryOp::Not, NOT_BP),
            _ => return self.postfix(),
        };
        let start = self.bump().span;
        let expr = self.nested(|parser| parser.expr_bp(bp - 1))?;
        let span = start.to(expr.span);
        Ok(Expr {
            kind: ExprKind::Unary {
                op,
                expr: Box::new(expr),
            },
            span,
        })
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let mut e = self.primary()?;
        loop {
            let start = e.span;
            let kind = match self.peek() {
                // A constant function's call is its body at the arguments.
                Tok::Op(Op::LParen)
                    if matches!(&e.kind, ExprKind::Name(name) if self.functions.contains_key(&name.name)) =>
                {
                    let ExprKind::Name(name) = &e.kind else { unreachable!("the guard matched a name") };
                    let function = self.functions[&name.name].clone();
                    let args = self.args()?;
                    let span = start.to(self.prev_span());
                    if args.len() != function.parameters().len() || args.iter().any(|arg| arg.name.is_some()) {
                        return Err(Diagnostic::with_rule(
                            DiagnosticRule::Resolution,
                            span,
                            format!(
                                "`{}` takes {} positional arguments ({})",
                                function.name(),
                                function.parameters().len(),
                                function.parameters().join(", ")
                            ),
                        ));
                    }
                    let arguments = args.into_iter().map(|arg| arg.value).collect::<Vec<_>>();
                    e = called(function.body(), function.parameters(), &arguments, span);
                    continue;
                }
                Tok::Op(Op::LParen) => ExprKind::Call {
                    callee: Box::new(e),
                    bindings: Vec::new(),
                    args: self.args()?,
                },
                Tok::Op(Op::LBracket)
                    if matches!(self.peek_at(1), Tok::Name(_))
                        && matches!(self.peek_at(2), Tok::Op(Op::Assign)) =>
                {
                    // Shape bindings on a call: `f[R = 64, S = 192](...)`.
                    self.bump();
                    let bindings = self.comma_list(|p| {
                        let param = p.expect_name()?;
                        p.expect_op(Op::Assign)?;
                        Ok((param, p.expr()?))
                    })?;
                    self.expect_op(Op::RBracket)?;
                    if !self.at_op(Op::LParen) {
                        return Err(self.error(format!(
                            "expected call arguments after shape bindings, found {}",
                            self.peek().describe()
                        )));
                    }
                    ExprKind::Call {
                        callee: Box::new(e),
                        bindings,
                        args: self.args()?,
                    }
                }
                Tok::Op(Op::LBracket) => {
                    self.bump();
                    let indices = self.comma_list(Self::index)?;
                    self.expect_op(Op::RBracket)?;
                    ExprKind::Index {
                        base: Box::new(e),
                        indices,
                    }
                }
                Tok::Op(Op::Dot) => {
                    self.bump();
                    ExprKind::Attr {
                        base: Box::new(e),
                        name: self.expect_name()?,
                    }
                }
                _ => return Ok(e),
            };
            e = Expr {
                kind,
                span: start.to(self.prev_span()),
            };
        }
    }

    /// At `(`: `(arg, name=value, ..)` with an optional trailing comma.
    fn args(&mut self) -> PResult<Vec<Arg>> {
        self.expect_op(Op::LParen)?;
        let mut args = Vec::new();
        while !self.at_op(Op::RParen) {
            let name = match (self.peek(), self.peek_at(1)) {
                (Tok::Name(_), Tok::Op(Op::Assign)) => {
                    let name = self.expect_name()?;
                    self.bump();
                    Some(name)
                }
                _ => None,
            };
            args.push(Arg {
                name,
                value: self.expr()?,
            });
            if !self.eat_op(Op::Comma) {
                break;
            }
        }
        self.expect_op(Op::RParen)?;
        Ok(args)
    }

    /// `expr`, `:`, `lo:`, `:hi`, `lo:hi`
    fn index(&mut self) -> PResult<Index> {
        let start = if self.at_op(Op::Colon) {
            None
        } else {
            Some(self.expr()?)
        };
        if self.eat_op(Op::Colon) {
            let end = if matches!(self.peek(), Tok::Op(Op::Comma | Op::RBracket)) {
                None
            } else {
                Some(self.expr()?)
            };
            return Ok(Index::Slice { start, end });
        }
        match start {
            Some(e) => Ok(Index::Expr(e)),
            None => Err(self.error(format!(
                "expected an index, found {}",
                self.peek().describe()
            ))),
        }
    }

    fn primary(&mut self) -> PResult<Expr> {
        let span = self.span();
        let kind = match self.peek().clone() {
            Tok::Int(v) => ExprKind::Int(v),
            Tok::Float(v) => ExprKind::Float(v),
            Tok::Kw(Kw::Inf) => ExprKind::Inf,
            Tok::Kw(Kw::True) => ExprKind::Bool(true),
            Tok::Kw(Kw::False) => ExprKind::Bool(false),
            Tok::Name(name) if name == "tensor" => {
                self.bump();
                let (shape, elem) = self.shape_and_elem()?;
                return Ok(Expr {
                    kind: ExprKind::Tensor { shape, elem },
                    span: span.to(self.prev_span()),
                });
            }
            // A module constant is its integer literal.
            Tok::Name(name) => match self.constants.get(&name) {
                Some(value) => ExprKind::Int(*value),
                None => ExprKind::Name(Ident { name, span }),
            },
            Tok::Op(Op::LParen) => {
                self.bump();
                if self.at_op(Op::RParen) {
                    return Err(self.error(
                        "expected an expression, found `)`; there is no empty tuple".into(),
                    ));
                }
                let first = self.expr()?;
                if self.eat_op(Op::RParen) {
                    return Ok(first);
                }
                let mut items = vec![first];
                while self.eat_op(Op::Comma) {
                    if self.at_op(Op::RParen) {
                        break;
                    }
                    items.push(self.expr()?);
                }
                let end = self.expect_op(Op::RParen)?;
                return Ok(Expr {
                    kind: ExprKind::Tuple(items),
                    span: span.to(end),
                });
            }
            other => {
                return Err(self.error(format!(
                    "expected an expression, found {}",
                    other.describe()
                )));
            }
        };
        self.bump();
        Ok(Expr { kind, span })
    }
}

#[cfg(test)]
mod tests {
    use super::super::ast::structure;
    use super::super::printer::print;
    use super::*;

    /// A file parsed with its own constants, as the first of `check_closed`'s two passes
    /// gives them.
    fn parse(text: &str) -> Result<File, Diagnostic> {
        let constants = super::parse(text, &BTreeMap::new(), &BTreeMap::new())?
            .constants
            .iter()
            .map(|constant| (constant.name.name.clone(), constant.value))
            .collect();
        super::parse(text, &constants, &BTreeMap::new())
    }

    /// A module constant is the integer it names wherever an expression names it, a
    /// `where` clause's included, and no binding may take its name.
    #[test]
    fn module_constants_are_their_integers() {
        let text = "const CHOICES = 4\n\nfn f[C](control: &tensor[C] i32) -> i32 where C >= CHOICES + 1:\n    return control[CHOICES]\n";
        let file = round_trip(text);
        assert_eq!(file.constants.len(), 1);
        assert_eq!((file.constants[0].name.name.as_str(), file.constants[0].value), ("CHOICES", 4));
        let f = only_fn(&file);
        assert!(format!("{:?}", f.signature.predicates).contains("Int(4)"));
        assert!(format!("{:?}", f.body).contains("Int(4)") && !format!("{:?}", f.body).contains("CHOICES"));
        for bound in [
            "const N = 1\n\nfn f[N](x: &tensor[N] i32):\n    return\n",
            "const N = 1\n\nfn f(N: i32):\n    return\n",
            "const N = 1\n\nfn f():\n    let N = 2\n",
            "const N = 1\n\nfn f():\n    for N in 0..2:\n        return\n",
            "const BLOCK = 64\n\nnative f for metal from \"f.metal\":\n    params (BLOCK in [32, 64] default 32)\n    launch f:\n        threadgroups (ceil_div(N, BLOCK), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
            "const BLOCK = 64\n\nnative f for metal from \"f.metal\":\n    launch f:\n        params (BLOCK in [32, 64] default 32)\n        threadgroups (ceil_div(N, BLOCK), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
            "const BLOCK = 64\n\nnative f for metal from \"f.metal\":\n    scratch BLOCK bytes (4)\n    launch f:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
        ] {
            assert!(parse(bound).unwrap_err().message.contains("module constant"), "{bound}");
        }
        assert!(parse("const N = M\n").is_err());
    }

    /// A module constant function's body reads module constants as their
    /// integers, prints back to itself, and binds no constant's name.
    #[test]
    fn module_constant_functions_round_trip() {
        let file = round_trip("const WIDE = 64\nconst GROUPS(STAGED) = 1 + STAGED * (WIDE - 1)\n");
        let [function] = file.constant_functions.as_slice() else {
            panic!("one constant function");
        };
        assert_eq!(function.name.name, "GROUPS");
        assert_eq!(function.parameters.iter().map(|parameter| parameter.name.as_str()).collect::<Vec<_>>(), ["STAGED"]);
        assert!(format!("{:?}", function.body).contains("Int(64)") && !format!("{:?}", function.body).contains("WIDE"));
        assert!(parse("const N = 1\nconst F(N) = N\n").unwrap_err().message.contains("module constant"));
    }


    fn round_trip(text: &str) -> File {
        let file = parse(text).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
        let printed = print(&file);
        let again = parse(&printed).unwrap_or_else(|d| panic!("{d:?}\n{printed}"));
        assert_eq!(structure(&file), structure(&again), "printed:\n{printed}");
        assert_eq!(print(&again), printed);
        file
    }

    fn only_fn(file: &File) -> &FnDecl {
        match file.decls.as_slice() {
            [Decl::Fn(f)] => f,
            other => panic!("expected one fn, found {other:?}"),
        }
    }

    #[test]
    fn logical_sources_round_trip() {
        let file = round_trip(
            "fn update[M, N](x: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + 1.0\n    return output\n\nlower update[M, N](x: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32 for cpu:\n    let mut output = result\n    for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + 1.0\n    return output\n",
        );
        assert_eq!(file.decls.len(), 2);
        let Decl::Fn(function) = &file.decls[0] else {
            panic!("expected fn")
        };
        assert!(matches!(
            function.signature.params[0].ty.kind,
            TypeKind::Shaped {
                head: ShapedHead::SharedTensor,
                ..
            }
        ));
        assert!(function.signature.result.is_some());
        assert!(matches!(
            function.body.stmts[1].kind,
            StmtKind::For { parallel: true, .. }
        ));
        assert!(matches!(file.decls[1], Decl::Lower(_)));
    }

    #[test]
    fn native_declarations_round_trip_without_repeating_the_signature() {
        let file = round_trip(
            "fn scale[N](x: &tensor[N] f32, factor: f32, output: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        output[i] = x[i] * factor\n\nnative scale for metal from \"native/scale.metal\":\n    launch scale:\n        threadgroups (ceil_div(N, 256), 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n",
        );
        let [Decl::Fn(_), Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one portable function and one native implementation")
        };
        assert_eq!(native.function.name, "scale");
        assert_eq!(native.target.name, "metal");
        assert_eq!(native.source, "native/scale.metal");
        assert_eq!(native.launches.len(), 1);
    }

    #[test]
    fn cpu_element_coverage_round_trips() {
        let file = round_trip(
            "native copy for cpu from \"copy.rs\":\n    params (ROWS in [4, 1] default 4)\n    elements (A in [f32, u32], B in [bf16])\n    launch copy:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert_eq!(native.elements.len(), 2);
        assert_eq!(native.elements[0].name.name, "A");
        assert_eq!(
            native.elements[0]
                .dtypes
                .iter()
                .map(|dtype| dtype.name.as_str())
                .collect::<Vec<_>>(),
            ["f32", "u32"]
        );
    }

    #[test]
    fn specialized_native_declarations_round_trip() {
        let file = round_trip(
            "native scale for cuda from \"scale.cu\":\n    static (N)\n    params (arithmetic form PARTS in [1, 2, 4] default 1, WIDTH in [64, 128] default 64)\n    where PARTS * WIDTH <= N and WIDTH >= 64\n    scratch partials bytes (PARTS * N * 4)\n    launch scale_partial:\n        threadgroups (ceil_div(N, WIDTH), PARTS, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n        shared_bytes (max(WIDTH * 4, 256))\n    launch scale_merge:\n        threadgroups (ceil_div(N, 256), 1, 1)\n        threads_per_threadgroup (min(N, 256), 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert_eq!(native.statics.len(), 1);
        assert_eq!(native.params.len(), 2);
        assert!(native.params[0].arithmetic);
        assert!(native.params[0].form);
        assert!(matches!(
            native
                .constraint
                .as_ref()
                .map(|constraint| &constraint.kind),
            Some(ExprKind::Binary {
                op: BinaryOp::And,
                ..
            })
        ));
        assert_eq!(native.scratch.len(), 1);
        assert_eq!(native.launches.len(), 2);
        assert!(native.launches[0].shared_bytes.is_some());
    }

    #[test]
    fn launch_parameters_round_trip_with_reused_names() {
        let file = round_trip(
            "native project for metal from \"project.metal\":\n    params (BATCH_FROM in [5, 3] default 5)\n    launch gemv when O < BATCH_FROM:\n        params (SIMDGROUPS in [16, 8] default 16, arithmetic form ROWS in [1, 2] default 1)\n        threadgroups (ceil_div(H, SIMDGROUPS * ROWS), 1, 1)\n        threads_per_threadgroup (SIMDGROUPS * 32, 1, 1)\n    launch batch when O >= BATCH_FROM:\n        params (SIMDGROUPS in [8, 4] default 8)\n        threadgroups (ceil_div(H, SIMDGROUPS), 1, 1)\n        threads_per_threadgroup (SIMDGROUPS * 32, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert_eq!(native.launches[0].params[0].name.name, "SIMDGROUPS");
        assert!(matches!(
            native.launches[0].params[1].default,
            NativeDefaultDecl::Value(Expr { kind: ExprKind::Int(1), .. })
        ));
        assert!(native.launches[0].params[1].arithmetic);
        assert!(native.launches[0].params[1].form);
        assert_eq!(native.launches[1].params[0].name.name, "SIMDGROUPS");
    }

    #[test]
    fn device_defaults_and_launch_conditions_round_trip() {
        let file = round_trip(
            "native project for cuda from \"project.cu\":\n    static (KV)\n    params (PARTS in [12, 24, 48] default least where KV * PARTS >= device.units)\n    launch gemv:\n        params (WARPS in [1, 2, 4] default greatest where WARPS * 32 <= device.subgroup_width, MATRIX in [0, 1] default device.matrix)\n        where MATRIX == 0 or device.matrix\n        threadgroups (KV, PARTS, 1)\n        threads_per_threadgroup (WARPS * 32, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert!(matches!(native.params[0].default, NativeDefaultDecl::Least(_)));
        assert!(matches!(native.launches[0].params[0].default, NativeDefaultDecl::Greatest(_)));
        assert!(matches!(
            native.launches[0].params[1].default,
            NativeDefaultDecl::Value(Expr { kind: ExprKind::Attr { .. }, .. })
        ));
        assert!(native.launches[0].constraint.is_some());
    }

    #[test]
    fn a_launch_grid_round_trips() {
        let file = round_trip(
            "native fill for metal from \"fill.metal\":\n    launch fill:\n        threadgroups (32, M, 1)\n        grid (groups)\n        threads_per_threadgroup (256, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert_eq!(native.launches[0].grid.as_ref().map(|grid| grid.name.as_str()), Some("groups"));
    }

    #[test]
    fn a_launch_condition_round_trips() {
        for (clause, encoded) in [("unless", Encoded::WhileClear), ("if", Encoded::WhileSet)] {
            let file = round_trip(&format!(
                "native fill for metal from \"fill.metal\":\n    launch fill:\n        threadgroups (32, M, 1)\n        grid (groups)\n        {clause} (whole)\n        threads_per_threadgroup (256, 1, 1)\n",
            ));
            let [Decl::Native(native)] = file.decls.as_slice() else {
                panic!("expected one native implementation")
            };
            let conditions = native.launches[0]
                .conditions
                .iter()
                .map(|condition| {
                let index = condition.index.as_ref().map(|index| match index.kind {
                    ExprKind::Int(index) => index,
                    _ => panic!("a constant index"),
                });
                (condition.encoded, condition.word.name.as_str(), index)
            })
                .collect::<Vec<_>>();
            assert_eq!(conditions, [(encoded, "whole", None)], "{clause}");
        }
    }

    #[test]
    fn a_launch_condition_conjunction_round_trips() {
        let file = round_trip(
            "native fill for metal from \"fill.metal\":\n    launch fill:\n        threadgroups (32, M, 1)\n        unless (whole)\n        if (next)\n        threads_per_threadgroup (256, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        let conditions = native.launches[0]
            .conditions
            .iter()
            .map(|condition| {
                let index = condition.index.as_ref().map(|index| match index.kind {
                    ExprKind::Int(index) => index,
                    _ => panic!("a constant index"),
                });
                (condition.encoded, condition.word.name.as_str(), index)
            })
            .collect::<Vec<_>>();
        assert_eq!(conditions, [(Encoded::WhileClear, "whole", None), (Encoded::WhileSet, "next", None)]);
    }

    #[test]
    fn an_indexed_launch_condition_round_trips() {
        let file = round_trip(
            "const WHOLE = 2\n\nnative fill for metal from \"fill.metal\":\n    launch fill:\n        threadgroups (32, M, 1)\n        unless (policy[WHOLE])\n        if (next[0])\n        threads_per_threadgroup (256, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        let conditions = native.launches[0]
            .conditions
            .iter()
            .map(|condition| {
                let index = condition.index.as_ref().map(|index| match index.kind {
                    ExprKind::Int(index) => index,
                    _ => panic!("a constant index"),
                });
                (condition.encoded, condition.word.name.as_str(), index)
            })
            .collect::<Vec<_>>();
        assert_eq!(conditions, [(Encoded::WhileClear, "policy", Some(2)), (Encoded::WhileSet, "next", Some(0))]);
    }

    #[test]
    fn launch_kernel_reads_round_trip() {
        let file = round_trip(
            "native project for cuda from \"project.cu\":\n    params (arithmetic INT8 in [0, 1] default 0)\n    launch project:\n        reads (INT8)\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert_eq!(native.launches[0].reads[0].name, "INT8");
    }

    #[test]
    fn conditional_native_launches_and_scratch_round_trip() {
        let file = round_trip(
            "native rows for metal from \"rows.metal\":\n    params (SPLIT in [1, 2] default 1)\n    where SPLIT == 1 or (SPLIT > 1 and SPLIT < 4)\n    scratch normalized bytes (O * H * 2) when O > 8\n    scratch partials bytes (SPLIT * O * 4) when (SPLIT > 1 or O >= 9) and O != 0\n    launch rows_normalize when O > 8:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch rows_gemv when O <= 2 or (INT8 == 1 and O <= 8):\n        threadgroups (ceil_div(F, 8), 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n        shared_bytes (O * 72)\n    launch rows_merge:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert!(matches!(
            native
                .constraint
                .as_ref()
                .map(|constraint| &constraint.kind),
            Some(ExprKind::Binary {
                op: BinaryOp::Or,
                ..
            })
        ));
        assert!(native.scratch.iter().all(|scratch| scratch.when.is_some()));
        assert!(matches!(
            native.scratch[1].when.as_ref().map(|when| &when.kind),
            Some(ExprKind::Binary {
                op: BinaryOp::And,
                ..
            })
        ));
        assert!(native.launches[0].when.is_some());
        assert!(matches!(
            native.launches[1].when.as_ref().map(|when| &when.kind),
            Some(ExprKind::Binary {
                op: BinaryOp::Or,
                ..
            })
        ));
        assert!(native.launches[2].when.is_none());
        let printed = print(&file);
        assert!(
            printed.contains("when (SPLIT > 1 or O >= 9) and O != 0\n"),
            "{printed}"
        );
        assert!(
            printed.contains("launch rows_gemv when O <= 2 or INT8 == 1 and O <= 8:\n"),
            "{printed}"
        );
    }

    #[test]
    fn native_terms_round_trip() {
        let file = round_trip(
            "native rows for metal from \"rows.metal\":\n    params (SPLIT in [1, 2] default 1)\n    let PARTS = max(1, SPLIT * O)\n    let SPARE = (8 - PARTS) * O\n    where PARTS <= 8\n    scratch partials bytes (PARTS * 4)\n    launch rows:\n        threadgroups (ceil_div(SPARE, 4), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert_eq!(
            native
                .terms
                .iter()
                .map(|term| term.name.name.as_str())
                .collect::<Vec<_>>(),
            ["PARTS", "SPARE"]
        );
    }

    #[test]
    fn projections_and_project_lines_round_trip() {
        let file = round_trip(
            "projection lanes for metal:\n    let SPLITS = 1 for paired\n    let SPLITS = SPLIT for single\n    params (SPLIT in [1, 2] default 1 for single, form WIDE in [0, 1] default 0)\n    where (WIDE == 0 or each(W, form(W, INT8) == 1))\n    where K % 2 == 0 for paired\n    scratch partials bytes (SPLITS * M * N * 4) when M > 4 for single\n    each W:\n        scratch factors bytes (N * 4)\n        scratch scales bytes (N) when WIDE == 1 for paired\n    launch gemv for single when M < 2:\n        params (ROWS in [1, 2] default 1 for paired, ROWS in [2, 1] default 2 for single)\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n\nnative rows for metal from \"rows.metal\":\n    where O > 0\n    scratch staged bytes (O * 4)\n    project lanes paired (gate_weight, up_weight)\n    scratch after bytes (O)\n    launch rows_stage before gemv when O > 1:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
        );
        let [Decl::Projection(projection), Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected a projection and a native implementation")
        };
        assert_eq!(projection.launches[0].param_variants.len(), 2);
        let project = native.project.as_ref().expect("the native projects");
        assert_eq!((project.scratch_before, project.weights.len()), (1, 2));
        assert_eq!(native.launches[0].before.as_ref().map(|before| before.name.as_str()), Some("gemv"));
        assert!(parse("native rows for metal from \"rows.metal\":\n    launch rows for single:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n").is_err());
    }

    /// A plane set in place of a weight (`planes (PLANE, ..) over (ROWS,
    /// WIDTH)`).
    #[test]
    fn plane_sets_round_trip() {
        let file = round_trip(
            "native planes for cuda from \"planes.cu\":\n    project dense single (planes (top, bit3, rest, scales) over (V, D)) rows rms\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else { panic!("expected a native implementation") };
        let project = native.project.as_ref().expect("the native projects");
        let planes: Vec<&str> = project.weight_planes[0].iter().map(|plane| plane.name.as_str()).collect();
        assert_eq!((project.weights.len(), planes), (1, vec!["top", "bit3", "rest", "scales"]));
        assert!(project.weight_overs[0].is_some());
    }

    /// A `project` line names the result its formed rows are (`publishes
    /// (result N)`) and holds its launches under `if` and `unless` lines.
    #[test]
    fn a_published_result_and_a_launch_guard_round_trip() {
        let file = round_trip(
            "native route for metal from \"route.metal\":\n    project dense segmented (router, gate) rows rms publishes (result 1)\n    unless (policy[2])\n    if (next)\n    restate SIMDGROUPS in [2, 4] default 2\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else { panic!("expected a native implementation") };
        let project = native.project.as_ref().expect("the native projects");
        assert_eq!(project.publishes.as_ref().map(|publishes| publishes.result), Some(1));
        let guard: Vec<(Encoded, &str)> =
            project.guard.iter().map(|condition| (condition.encoded, condition.word.name.as_str())).collect();
        assert_eq!(guard, [(Encoded::WhileClear, "policy"), (Encoded::WhileSet, "next")]);
        assert_eq!(project.restated.len(), 1);
        let file = round_trip("native route for metal from \"route.metal\":\n    project dense single (weight) rows rms ordered (weight)\n");
        let [Decl::Native(native)] = file.decls.as_slice() else { panic!("expected a native implementation") };
        assert!(native.project.as_ref().is_some_and(|project| project.ordered.len() == 1));
        assert!(parse("native route for metal from \"route.metal\":\n    project dense single (weight) rows rms publishes (rows 1)\n").is_err());
    }

    /// An operand buffer's clauses (`operand [rows] [when PAIRS] [formed
    /// (COND)] [nvfp4 (COND)]`), on a projection's buffers alone, and the
    /// `operand`/`emits` of a `project` line.
    #[test]
    fn operand_clauses_round_trip() {
        let file = round_trip(
            "projection quantized for cpu:\n    scratch quantized bytes (M * K) operand when (G * M == 1) formed (INT8 == 1) for tiled\n    scratch staged bytes (M * K * 4) when M > 0 operand rows when (NM > 0) nvfp4 (operands(W) == 1) for tiled\n    scratch rows bytes (M * K * 4) operand rows for emits\n\nnative rows for cpu from \"rows.rs\":\n    project quantized rows (W, V over (R, N)) over (M, K) operand (x at (picked) within (first, count) normed (norm, eps) tiles (M == 1), emitted) emits (result 0 over (M, N))\n    launch rows_project:\n        threadgroups (M, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n",
        );
        let [Decl::Projection(projection), Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected a projection and a native implementation")
        };
        let ProjectionScratchDecl::One(quantized) = &projection.scratch[0] else { panic!("one buffer") };
        assert!(matches!(
            &quantized.item.operand,
            Some(ScratchOperandDecl::Declared { rows: false, when: Some(_), formed: Some(_), nvfp4: None })
        ));
        let ProjectionScratchDecl::One(staged) = &projection.scratch[1] else { panic!("one buffer") };
        assert!(matches!(
            &staged.item.operand,
            Some(ScratchOperandDecl::Declared { rows: true, when: Some(_), formed: None, nvfp4: Some(_) })
        ));
        let project = native.project.as_ref().expect("a project line");
        assert!(matches!(
            project.operands.as_slice(),
            [
                NativeOperandDecl {
                    source: OperandSourceDecl::Parameter { at: Some(at), within: Some(_), norm: Some(_), .. },
                    tiles: Some(_),
                },
                NativeOperandDecl { source: OperandSourceDecl::Emitted(_), tiles: None },
            ] if at.name == "picked"
        ));
        let computed = "native rows for cpu from \"rows.rs\":\n    project quantized tiled (W) over (M, K) operand (values (joined))\n";
        let computed = parse(computed).expect("a computed operand parses");
        let [Decl::Native(native)] = computed.decls.as_slice() else { panic!("one native") };
        assert!(matches!(
            native.project.as_ref().map(|project| &project.operands[..]),
            Some([NativeOperandDecl { source: OperandSourceDecl::Values(function), .. }]) if function.name == "joined"
        ));
        let malformed = "projection quantized for cpu:\n    scratch staged bytes (4) operand rows formed INT8 == 1\n";
        assert!(parse(malformed).unwrap_err().message.contains("expected `(`"));
        // An entry's buffer holds no operand form: the operator owns them.
        let owned = "native rows for cpu from \"rows.rs\":\n    scratch staged bytes (4) operand rows\n    launch rows_stage:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n";
        assert!(parse(owned).is_err());
    }

    #[test]
    fn repeated_native_launches_round_trip() {
        let file = round_trip(
            "native rows for metal from \"rows.metal\":\n    scratch window bytes (64)\n    launch rows_prepare:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    repeat (ceil_div(O, 64)):\n        launch rows_decode when O > 8:\n            threadgroups (2, 1, 1)\n            threads_per_threadgroup (32, 1, 1)\n        launch rows_attend:\n            threadgroups (O, 1, 1)\n            threads_per_threadgroup (32, 1, 1)\n    launch rows_merge:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert_eq!(native.launches.len(), 4);
        let repeat = native.repeat.as_ref().unwrap();
        assert_eq!((repeat.first, repeat.launches), (1, 2));
    }

    #[test]
    fn continued_headers() {
        let file = round_trip(
            "fn row_dot[N](x: tensor[N] f32, w: tensor[N] f32) -> f32\n    where N >= 2 and N % 2 == 0:\n    let mut result = f32(0.0)\n    for pair in 0..(N / 2):\n        result = fma(x[2 * pair], w[2 * pair], result)\n    return result\n\n\
             lower row_dot[N](x: tensor[N] f32, w: tensor[N] f32) -> f32\n    for cpu where N >= 1:\n    return row_dot_cpu(x, w)\n\n\
             fn update[M](a: &tensor[M, M] f32,\n             acc: tensor[M, M] f32) -> tensor[M, M] f32:\n    let mut result = acc\n    for i in 0..M:\n        for j in 0..M:\n            result[i, j] = result[i, j] + a[i, j]\n    return result\n\n\
             lower update[M](a: &tensor[M, M] f32, acc: tensor[M, M] f32) -> tensor[M, M] f32\n    for metal\n    requires metal.matrix\n    where M == 8 and full(M):\n    return metal.matrix.matmul(a, acc, accumulation=f32)\n\n\
             fn prepare[R, K](x: &tensor[R, K] bf16, pos: index[K])\n    -> tensor[R, K] f32 requires metal.matrix:\n    return metal.matrix.matmul(x, x, accumulation=f32)\n",
        );
        assert_eq!(file.decls.len(), 5);
        let Decl::Fn(f) = &file.decls[0] else {
            panic!("expected fn")
        };
        assert_eq!(f.signature.predicates.len(), 2);
        assert_eq!(f.body.stmts.len(), 3);
        let Decl::Lower(l) = &file.decls[1] else {
            panic!("expected lower")
        };
        assert!(
            l.predicates.len() == 1
                && l.signature.predicates.is_empty()
                && l.signature.result.is_some()
        );
        let Decl::Fn(f) = &file.decls[2] else {
            panic!("expected fn")
        };
        assert!(f.body.stmts.len() == 3 && f.signature.result.is_some());
        let Decl::Lower(l) = &file.decls[3] else {
            panic!("expected lower")
        };
        assert_eq!(l.predicates.len(), 2);
        assert_eq!(l.requires.len(), 1);
        assert_eq!(l.body.stmts.len(), 1);
        let Decl::Fn(f) = &file.decls[4] else {
            panic!("expected fn")
        };
        assert!(matches!(
            &f.signature.result,
            Some(TypeExpr {
                kind: TypeKind::Shaped { .. },
                ..
            })
        ));
        assert_eq!(f.requires.len(), 1);
    }

    #[test]
    fn ownership_and_contextual_words() {
        let file = round_trip(
            "fn f[M](owned: tensor[M] f32, shared: &tensor[M] f32, exclusive: &mut tensor[M] f32, to: f32) -> tensor[M] f32:\n    exclusive[0] = shared[0] + to\n    return owned\n",
        );
        let f = only_fn(&file);
        assert!(matches!(
            f.signature.params[0].ty.kind,
            TypeKind::Shaped {
                head: ShapedHead::Tensor,
                ..
            }
        ));
        assert!(matches!(
            f.signature.params[1].ty.kind,
            TypeKind::Shaped {
                head: ShapedHead::SharedTensor,
                ..
            }
        ));
        assert!(matches!(
            f.signature.params[2].ty.kind,
            TypeKind::Shaped {
                head: ShapedHead::MutTensor,
                ..
            }
        ));
        assert_eq!(f.signature.params[3].name.name, "to");
    }

    #[test]
    fn host_ports_round_trip() {
        let file = round_trip(
            "fn request[N](read: &host tensor[N] i32, written: &host mut tensor[N] i32, out: &mut tensor[N] i32):\n    out[0] = 1\n",
        );
        let function = only_fn(&file);
        for (index, write) in [(0, false), (1, true)] {
            assert!(matches!(
                function.signature.params[index].ty.kind,
                TypeKind::Shaped {
                    head: ShapedHead::HostTensor { write: w },
                    ..
                } if w == write
            ));
        }
    }

    #[test]
    fn logical_ownership_ranges_and_loops_round_trip() {
        let file = round_trip(
            "fn update[N](owned: tensor[N] f32, shared: &tensor[N] f32, exclusive: &mut tensor[N] f32, at: index[N], span: range[N]) -> tensor[N] f32:\n    for i in 0..N:\n        exclusive[i] = shared[i]\n    parallel for i in 0..N:\n        exclusive[i] = shared[i]\n    return owned\n",
        );
        let function = only_fn(&file);
        assert!(matches!(
            function.signature.params[0].ty.kind,
            TypeKind::Shaped {
                head: ShapedHead::Tensor,
                ..
            }
        ));
        assert!(matches!(
            function.signature.params[1].ty.kind,
            TypeKind::Shaped {
                head: ShapedHead::SharedTensor,
                ..
            }
        ));
        assert!(matches!(
            function.signature.params[2].ty.kind,
            TypeKind::Shaped {
                head: ShapedHead::MutTensor,
                ..
            }
        ));
        assert!(matches!(
            function.signature.params[4].ty.kind,
            TypeKind::Range(_)
        ));
        assert!(matches!(
            function.body.stmts[0].kind,
            StmtKind::For {
                parallel: false,
                ..
            }
        ));
        assert!(matches!(
            function.body.stmts[1].kind,
            StmtKind::For { parallel: true, .. }
        ));
    }

    #[test]
    fn returned_values_and_loop_forms() {
        let file = round_trip(
            "fn prefix[K](x: &tensor[K] f32, result: tensor[K] f32) -> tensor[K] f32:\n    let mut output = result\n    let mut running = f32(0.0)\n    for i in 0..K:\n        running = running + x[i]\n        output[i] = running\n    return output\n",
        );
        let body = &only_fn(&file).body;
        assert!(matches!(
            body.stmts[2].kind,
            StmtKind::For {
                parallel: false,
                ..
            }
        ));
        assert!(matches!(body.stmts[3].kind, StmtKind::Return(_)));
    }

    #[test]
    fn logical_parallel_and_statements() {
        let file = round_trip(
            "fn transform[N](x: &tensor[N] f32, result: tensor[N] f32, enabled: bool) -> tensor[N] f32 where N >= 1:\n    let mut output = result\n    parallel for i in 0..N:\n        if enabled: output[i] = x[i] + 1.0; output[i] *= 2.0\n        else if N > 1: output[i] = x[i] - 1.0\n        else:\n            output[i] = x[i]\n    return output\n",
        );
        let body = &only_fn(&file).body;
        let StmtKind::For {
            body: inline,
            parallel: true,
            ..
        } = &body.stmts[1].kind
        else {
            panic!("expected for")
        };
        let StmtKind::If { els: Some(els), .. } = &inline.stmts[0].kind else {
            panic!("expected if")
        };
        assert!(matches!(
            els.stmts.as_slice(),
            [Stmt {
                kind: StmtKind::If { els: Some(_), .. },
                ..
            }]
        ));
    }

    #[test]
    fn ranges_and_diagnostics() {
        let file = round_trip(
            "fn f[C](x: tensor[C] f32):\n    for i in 0..C - 1 | 1:\n        g(i, 1.0..2.5)\n",
        );
        let StmtKind::For { iter, .. } = &only_fn(&file).body.stmts[0].kind else {
            panic!("expected for")
        };
        let ExprKind::Range { lo, hi } = &iter.kind else {
            panic!("expected range")
        };
        assert!(
            matches!(lo.kind, ExprKind::Int(0))
                && matches!(
                    hi.kind,
                    ExprKind::Binary {
                        op: BinaryOp::BitOr,
                        ..
                    }
                )
        );

        let text = "fn f(x: f32):\n    let for = x\n";
        let err = parse(text).unwrap_err();
        assert_eq!(&text[err.span.start as usize..err.span.end as usize], "for");
        assert_eq!(err.rule, DiagnosticRule::Syntax);
        assert!(parse("fn f():\n    a..b..c\n").is_err());
        assert!(parse("fn f():\n    f(x) = 1\n").is_err());
        assert!(parse("lower f for cpu:\n    g()\n").is_err());
        assert!(parse("fn f(x: f32) -> f32\n    where x > 0 = portable\n").is_err());
    }

    #[test]
    fn backend_functions_are_written_as_lowerings() {
        let text = "fn h(x: f32) -> f32 for cpu:\n    return x\n";
        let err = parse(text).unwrap_err();
        assert_eq!(err.rule, DiagnosticRule::Syntax);
        assert_eq!(&text[err.span.start as usize..err.span.end as usize], "for");
        assert!(
            err.message
                .contains("written as `lower NAME … for BACKEND`")
        );
    }

    #[test]
    fn retired_source_forms_are_hard_errors() {
        for source in [
            "fn f[N](out x: tensor[N] f32):\n    return\n",
            "fn f[N](inout x: tensor[N] f32):\n    return\n",
            "fn f[N](x: view[N] f32):\n    return\n",
            "fn f[N](x: tile[N] f32):\n    return\n",
            "fn f[N](x: tensor[N] f32) alias(x, x):\n    return\n",
            "fn f[N](x: tensor[N] f32):\n    let y = tile[N] f32\n",
            "fn f[N](x: tensor[N] f32):\n    publish x to x\n",
            "fn f[N](x: tensor[N] f32):\n    yield x\n",
            "fn f[N](x: tensor[N] f32):\n    parallel [i] in 0..N:\n        g(i)\n",
            "fn f[N](x: tensor[N] f32):\n    ordered [i] in 0..N:\n        g(i)\n",
            "fn f[N](x: tensor[N] f32):\n    pipeline [i] in 0..N:\n        g(i)\n",
            "fn f():\n    stage s:\n        g()\n",
            "fn f():\n    merge (a, b) identity 0:\n        return\n",
            "fn f(x: metal.fragment):\n    return\n",
        ] {
            assert!(
                parse(source).is_err(),
                "retired syntax parsed successfully:\n{source}"
            );
        }
    }
}
