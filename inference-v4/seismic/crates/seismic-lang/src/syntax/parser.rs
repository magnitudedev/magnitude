//! Recursive-descent parser with Pratt expression parsing.
use super::ast::*;
use super::lexer::lex;
use super::token::{Kw, Op, Tok, Token};
use crate::span::{Diagnostic, Span};

pub fn parse(text: &str) -> Result<File, Diagnostic> {
    let tokens = lex(text)?;
    Parser { tokens, pos: 0 }.file()
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

fn region_mode(tok: &Tok) -> Option<RegionMode> {
    match tok {
        Tok::Kw(Kw::Parallel) => Some(RegionMode::Parallel),
        Tok::Kw(Kw::Ordered) => Some(RegionMode::Ordered),
        Tok::Kw(Kw::Pipeline) => Some(RegionMode::Pipeline),
        _ => None,
    }
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

    fn expect_word(&mut self, word: &str) -> PResult<Span> {
        if self.at_word(word) {
            Ok(self.bump().span)
        } else {
            Err(self.error(format!(
                "expected `{word}`, found {}",
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
        Diagnostic::new(self.span(), message)
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
        loop {
            while matches!(self.peek(), Tok::Newline) {
                self.bump();
            }
            match self.peek() {
                Tok::Eof => break,
                Tok::Kw(Kw::Fn) => decls.push(Decl::Fn(self.fn_decl()?)),
                Tok::Kw(Kw::Lower) => decls.push(Decl::Lower(self.lower_decl()?)),
                Tok::Indent => {
                    return Err(self.error(
                        "unexpected indentation; declarations start at the left margin".into(),
                    ))
                }
                other => {
                    return Err(self.error(format!(
                        "expected `fn` or `lower`, found {}",
                        other.describe()
                    )))
                }
            }
        }
        Ok(File { decls })
    }

    fn fn_decl(&mut self) -> PResult<FnDecl> {
        let start = self.span();
        self.expect_kw(Kw::Fn)?;
        let name = self.expect_name()?;
        let mut continued = false;
        let mut signature = self.signature(&mut continued, true)?;
        self.continue_header(&mut continued, true);
        let target = if self.eat_kw(Kw::For) {
            Some(self.expect_name()?)
        } else {
            None
        };
        signature.predicates = self.where_clause(&mut continued, true)?;
        self.expect_op(Op::Colon)?;
        let body = self.body(continued)?;
        Ok(FnDecl {
            signature,
            name,
            target,
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
        let predicates = self.where_clause(&mut continued, true)?;
        self.expect_op(Op::Colon)?;
        let body = self.body(continued)?;
        Ok(LowerDecl {
            name,
            signature,
            target,
            predicates,
            body,
            span: start.to(self.prev_span()),
        })
    }

    /// A header may continue on one deeper-indented line (and further lines at that
    /// indentation) starting with `alias`, `->`, `where` or, in a lowering, `for`. The
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
            Tok::Op(Op::Arrow) | Tok::Kw(Kw::Where) => true,
            Tok::Kw(Kw::For) => lower,
            Tok::Name(n) => n == "alias" && matches!(self.peek_at(skip + 1), Tok::Op(Op::LParen)),
            _ => false,
        };
        if continues {
            self.pos += skip;
            *continued = true;
        }
    }

    /// `[Shape, ..](params) [alias(a, b), ..] [-> type]`; predicates are filled by the caller.
    fn signature(&mut self, continued: &mut bool, lower: bool) -> PResult<Signature> {
        let mut shape = Vec::new();
        if self.eat_op(Op::LBracket) {
            shape = self.comma_list(Self::expect_name)?;
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
        let mut aliases = Vec::new();
        if self.at_alias() {
            loop {
                self.bump();
                self.expect_op(Op::LParen)?;
                let a = self.expect_name()?;
                self.expect_op(Op::Comma)?;
                let b = self.expect_name()?;
                self.expect_op(Op::RParen)?;
                aliases.push((a, b));
                if !self.eat_op(Op::Comma) {
                    break;
                }
                if !self.at_alias() {
                    return Err(self.error(format!(
                        "expected `alias(a, b)`, found {}",
                        self.peek().describe()
                    )));
                }
            }
        }
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
            aliases,
            result,
            predicates: Vec::new(),
        })
    }

    fn at_alias(&self) -> bool {
        self.at_word("alias") && matches!(self.peek_at(1), Tok::Op(Op::LParen))
    }

    fn where_clause(&mut self, continued: &mut bool, lower: bool) -> PResult<Vec<Expr>> {
        self.continue_header(continued, lower);
        let mut predicates = Vec::new();
        if self.eat_kw(Kw::Where) {
            conjuncts(self.expr()?, &mut predicates);
        }
        Ok(predicates)
    }

    /// `[out | inout] name: type`. `out`/`inout` are modes only before a parameter name, so
    /// `out: tensor[N] f32` is a read-only parameter named `out`.
    fn param(&mut self) -> PResult<Param> {
        let mode = match (self.peek(), self.peek_at(1)) {
            (Tok::Name(m), Tok::Name(_)) if m == "out" => Mode::Out,
            (Tok::Name(m), Tok::Name(_)) if m == "inout" => Mode::Inout,
            _ => Mode::In,
        };
        if mode != Mode::In {
            self.bump();
        }
        let name = self.expect_name()?;
        self.expect_op(Op::Colon)?;
        let ty = self.type_expr()?;
        Ok(Param { mode, name, ty })
    }

    fn type_expr(&mut self) -> PResult<TypeExpr> {
        let start = self.span();
        let kind = match self.peek().clone() {
            Tok::Kw(Kw::Void) => {
                self.bump();
                TypeKind::Void
            }
            Tok::Kw(Kw::Tile) => {
                self.bump();
                self.shaped(ShapedHead::Tile)?
            }
            Tok::Op(Op::LParen) => {
                self.bump();
                let mut items = self.comma_list(Self::type_expr)?;
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
                    Tok::Op(Op::LBracket) if word == "view" => self.shaped(ShapedHead::View)?,
                    Tok::Op(Op::LBracket) if word == "index" => {
                        self.bump();
                        let bound = self.expr()?;
                        self.expect_op(Op::RBracket)?;
                        TypeKind::Index(Box::new(bound))
                    }
                    Tok::Op(Op::LBracket) => {
                        return Err(self.error(format!(
                            "`{word}` takes no shape; shaped types are `tensor`, `view` and `tile`"
                        )))
                    }
                    Tok::Op(Op::Dot) => {
                        self.bump();
                        let ty = self.expect_name()?;
                        let mut args = Vec::new();
                        if self.eat_op(Op::LParen) {
                            if !self.at_op(Op::RParen) {
                                args = self.comma_list(Self::expr)?;
                            }
                            self.expect_op(Op::RParen)?;
                        }
                        TypeKind::Native {
                            target: name,
                            name: ty,
                            args,
                        }
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
                let targets = self.comma_list(Self::expect_name)?;
                self.expect_kw(Kw::In)?;
                let iter = self.expr()?;
                self.expect_op(Op::Colon)?;
                StmtKind::For {
                    targets,
                    iter,
                    body: self.block()?,
                }
            }
            Tok::Kw(Kw::If) => self.if_stmt()?,
            Tok::Kw(Kw::Stage) => {
                self.bump();
                let name = self.expect_name()?;
                let mut ports = Vec::new();
                if self.eat_op(Op::LParen) {
                    if !self.at_op(Op::RParen) {
                        ports = self.comma_list(Self::expect_name)?;
                    }
                    self.expect_op(Op::RParen)?;
                }
                self.expect_op(Op::Colon)?;
                StmtKind::Stage {
                    name,
                    ports,
                    body: self.block()?,
                }
            }
            Tok::Kw(Kw::Parallel | Kw::Ordered | Kw::Pipeline) => StmtKind::Region(self.region()?),
            Tok::Kw(Kw::Else) => return Err(self.error("`else` without a matching `if`".into())),
            Tok::Kw(Kw::Merge) => {
                return Err(self.error(
                    "`merge` must directly follow a `parallel` region at the same indentation"
                        .into(),
                ))
            }
            Tok::Indent => return Err(self.error("unexpected indentation".into())),
            _ => return self.simple_statements(out),
        };
        out.push(Stmt {
            kind,
            span: start.to(self.prev_span()),
        });
        Ok(())
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
                let kind = self.if_stmt()?;
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

    /// At a region mode: `mode [binders] in source: block [merge (l, r) identity e: block]`.
    fn region(&mut self) -> PResult<Region> {
        let start = self.span();
        let mode = match region_mode(self.peek()) {
            Some(mode) => mode,
            None => {
                return Err(self.error(format!(
                    "expected `parallel`, `ordered` or `pipeline`, found {}",
                    self.peek().describe()
                )))
            }
        };
        self.bump();
        self.expect_op(Op::LBracket)?;
        let binders = self.comma_list(Self::expect_name)?;
        self.expect_op(Op::RBracket)?;
        self.expect_kw(Kw::In)?;
        let source = self.expr()?;
        let source_span = source.span;
        let sources = match source.kind {
            ExprKind::Tuple(members) => members,
            _ => vec![source],
        };
        if sources.len() > 1 && sources.len() != binders.len() {
            return Err(Diagnostic::new(
                source_span,
                format!(
                    "a product of {} domains needs {} binders, found {}",
                    sources.len(),
                    sources.len(),
                    binders.len()
                ),
            ));
        }
        self.expect_op(Op::Colon)?;
        let body = self.block()?;
        let mut merge = None;
        if self.at_kw(Kw::Merge) {
            if mode != RegionMode::Parallel {
                return Err(
                    self.error("`merge` combines the results of a `parallel` region".into())
                );
            }
            let merge_start = self.bump().span;
            self.expect_op(Op::LParen)?;
            let left = self.pattern_atom()?;
            self.expect_op(Op::Comma)?;
            let right = self.pattern_atom()?;
            self.expect_op(Op::RParen)?;
            self.expect_word("identity")?;
            let identity = self.expr()?;
            self.expect_op(Op::Colon)?;
            let body = self.block()?;
            merge = Some(Merge {
                left,
                right,
                identity,
                body,
                span: merge_start.to(self.prev_span()),
            });
        }
        Ok(Region {
            mode,
            binders,
            sources,
            body,
            merge,
            span: start.to(self.prev_span()),
        })
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
            let inner = self.pattern()?;
            self.expect_op(Op::RParen)?;
            return Ok(inner);
        }
        Ok(Pattern::Name(self.expect_name()?))
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
            Tok::Kw(Kw::Publish) => {
                self.bump();
                let value = self.expr()?;
                self.expect_word("to")?;
                StmtKind::Publish {
                    value,
                    destination: self.expr()?,
                }
            }
            Tok::Kw(kw @ (Kw::Yield | Kw::Return)) => {
                let kw = *kw;
                self.bump();
                let values = if kw == Kw::Return && (self.at_line_end() || self.at_op(Op::Semi)) {
                    Vec::new()
                } else if region_mode(self.peek()).is_some() {
                    vec![self.value(&mut ended)?]
                } else {
                    self.comma_list(Self::expr)?
                };
                if kw == Kw::Yield {
                    StmtKind::Yield(values)
                } else {
                    StmtKind::Return(values)
                }
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
                            return Err(Diagnostic::new(target.span, "cannot assign to this expression; a target is a name, an indexed place or a tuple of them"));
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

    /// The value of `let`/`yield`/`return`: an expression or a result-producing region.
    fn value(&mut self, ended: &mut bool) -> PResult<Expr> {
        match region_mode(self.peek()) {
            Some(RegionMode::Pipeline) => {
                Err(self.error("a `pipeline` region produces no result".into()))
            }
            Some(_) => {
                let region = self.region()?;
                *ended = true;
                let span = region.span;
                Ok(Expr {
                    kind: ExprKind::Region(Box::new(region)),
                    span,
                })
            }
            None => self.expr(),
        }
    }

    // ---- expressions ----

    fn expr(&mut self) -> PResult<Expr> {
        self.expr_bp(0)
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
        let expr = self.expr_bp(bp - 1)?;
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
            Tok::Name(name) => ExprKind::Name(Ident { name, span }),
            Tok::Kw(Kw::Tile) => {
                self.bump();
                let (shape, elem) = self.shape_and_elem()?;
                return Ok(Expr {
                    kind: ExprKind::Tile { shape, elem },
                    span: span.to(self.prev_span()),
                });
            }
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
            Tok::Kw(Kw::Parallel | Kw::Ordered | Kw::Pipeline) => {
                return Err(self.error(
                    "a region is a statement, or the whole value of `let`, `yield` or `return`"
                        .into(),
                ));
            }
            other => {
                return Err(self.error(format!(
                    "expected an expression, found {}",
                    other.describe()
                )))
            }
        };
        self.bump();
        Ok(Expr { kind, span })
    }
}

#[cfg(test)]
mod tests {
    use super::super::printer::print;
    use super::*;

    /// Debug rendering with every `Span { .. }` removed, for span-insensitive comparison.
    fn shape(file: &File) -> String {
        let text = format!("{file:?}");
        let mut out = String::new();
        let mut rest = text.as_str();
        while let Some(i) = rest.find("Span {") {
            out.push_str(&rest[..i]);
            rest = &rest[i..];
            rest = &rest[rest.find('}').map_or(rest.len(), |j| j + 1)..];
        }
        out + rest
    }

    fn round_trip(text: &str) -> File {
        let file = parse(text).unwrap_or_else(|d| panic!("{}", d.render("source", text)));
        let printed = print(&file);
        let again = parse(&printed)
            .unwrap_or_else(|d| panic!("{}\n{printed}", d.render("printed", &printed)));
        assert_eq!(shape(&file), shape(&again), "printed:\n{printed}");
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
    fn reference_sources_round_trip() {
        let rms = round_trip(include_str!(
            "../../../../../seismic-std/lib/kernels/rms_norm.seismic"
        ));
        assert!(rms.decls.len() >= 3);
        let Decl::Fn(f) = &rms.decls[0] else {
            panic!("expected fn")
        };
        assert!(f.target.is_none());
        assert_eq!(
            f.signature
                .params
                .iter()
                .map(|p| p.mode)
                .collect::<Vec<_>>(),
            [Mode::In, Mode::In, Mode::Out, Mode::In]
        );

        let linear = round_trip(include_str!(
            "../../../../../seismic-std/lib/kernels/linear.seismic"
        ));
        let Decl::Fn(f) = &linear.decls[0] else {
            panic!("expected fn")
        };
        let Block { stmts, .. } = &f.body;
        let StmtKind::Region(region) = &stmts[0].kind else {
            panic!("expected region")
        };
        assert_eq!(
            (region.mode, region.binders.len(), region.sources.len()),
            (RegionMode::Parallel, 2, 2)
        );
        assert!(matches!(region.sources[0].kind, ExprKind::Range { .. }));

        let matmul = round_trip(include_str!(
            "../../../../../seismic-std/lib/constructs/matmul.seismic"
        ));
        assert_eq!(only_fn(&matmul).signature.params[2].mode, Mode::Inout);
    }

    #[test]
    fn continued_headers() {
        let file = round_trip(
            "fn row_dot[N](x: tensor[N] f32, w: tensor[N] f32) -> f32\n    where N >= 2 and N % 2 == 0:\n    let mut result = f32(0.0)\n    for pair in 0..(N / 2):\n        result = fma(x[2 * pair], w[2 * pair], result)\n    return result\n\n\
             lower row_dot[N](x: tensor[N] f32, w: tensor[N] f32) -> f32\n    for cpu where N >= 1:\n    return row_dot_cpu(x, w)\n\n\
             fn update[M](a: tile[M, M] f32,\n             inout acc: tile[M, M] f32) -> void:\n    acc += a\n\n\
             lower update[M](a: tile[M, M] f32, inout acc: tile[M, M] f32) -> void\n    for metal\n    where M == 8 and full(M):\n    let mut left = metal.simdgroup_matrix(f32)\n    metal.simdgroup_load(left, a, 0, 0)\n\n\
             fn prepare[R, K](x: view[R, K] bf16, pos: index[K])\n    -> (tile[R, K] f32, metal.simdgroup_matrix(f32)) for metal:\n    return f32(x), native(x)\n",
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
        assert!(f.body.stmts.len() == 1 && f.signature.result.is_none());
        let Decl::Lower(l) = &file.decls[3] else {
            panic!("expected lower")
        };
        assert_eq!(l.predicates.len(), 2);
        assert_eq!(l.body.stmts.len(), 2);
        let Decl::Fn(f) = &file.decls[4] else {
            panic!("expected fn")
        };
        assert!(
            matches!(&f.signature.result, Some(TypeExpr { kind: TypeKind::Tuple(items), .. }) if items.len() == 2)
        );
    }

    #[test]
    fn modes_aliases_and_contextual_words() {
        let file = round_trip(
            "fn f(out out: tensor[M] f32, out: tensor[M] f32, inout inout: tile[M] T, to: f32) alias(out, inout), alias(to, out) -> f32:\n    publish to to out[:]\n    return to\n",
        );
        let f = only_fn(&file);
        assert!(f.target.is_none());
        let params: Vec<_> = f
            .signature
            .params
            .iter()
            .map(|p| (p.mode, p.name.name.as_str()))
            .collect();
        assert_eq!(
            params,
            [
                (Mode::Out, "out"),
                (Mode::In, "out"),
                (Mode::Inout, "inout"),
                (Mode::In, "to")
            ]
        );
        assert_eq!(f.signature.aliases.len(), 2);
    }

    #[test]
    fn region_results_merge_and_stages() {
        let file = round_trip(
            "fn sum[K](x: tensor[K] f32, out y: tensor[1] f32):\n    stage prepare:\n        let total = parallel [part] in 0..K:\n            yield reduce(f32(x[part]), 0, sum)\n        merge (left, right) identity f32(0.0):\n            yield left + right\n        let mut running = f32(-inf)\n        let checkpoints = ordered [p, q] in rectangles:\n            let (m, (l, a)) = rectangles[p, q]\n            (running, m) = (running + m, l * a)\n            yield running, m\n        yield total, checkpoints\n\n    stage finish(total, checkpoints):\n        publish total to y[0]\n",
        );
        let body = &only_fn(&file).body;
        let StmtKind::Stage { body: prepare, .. } = &body.stmts[0].kind else {
            panic!("expected stage")
        };
        assert_eq!(prepare.stmts.len(), 4);
        let StmtKind::Let {
            value:
                Expr {
                    kind: ExprKind::Region(region),
                    ..
                },
            ..
        } = &prepare.stmts[0].kind
        else {
            panic!("expected region")
        };
        assert!(region.merge.is_some());
        assert!(matches!(&body.stmts[1].kind, StmtKind::Stage { ports, .. } if ports.len() == 2));
        assert!(parse(
            "fn f():\n    let t = pipeline [k] in 0..K:\n        stage a:\n            g()\n"
        )
        .is_err());
        assert!(
            parse("fn f():\n    parallel [a, b] in (0..M, 0..N, 0..K):\n        g()\n").is_err()
        );
    }

    #[test]
    fn pipeline_and_statements() {
        let file = round_trip(
            "fn gp[N, K](x: tensor[1, K] bf16, gate: tensor[N, K] q4g64, out: tensor[1, N] bf16) where N >= 1:\n    parallel [cols] in 0..N:\n        let mut g = zeros_like(out[:, cols], dtype=f32)\n\n        pipeline [k] in 0..K:\n            stage prepare:\n                let a, gw = prepare_inputs[R = 64](x[:, k], gate[cols, k])\n                yield a, gw\n\n            stage accumulate(a, gw):\n                matmul(a, gw, into=g)\n\n        let y = g / (1.0 + exp(-g)) * g.T\n        let t = tile[2, K - 1] f32\n        for i, j in owned(t): t[i, j] = 0.0; g[i, j] += 3.402823466e38\n        if not (K > 1 or N == 2) and K % 2 == 0: return\n        else if K << 1 > 4: y[0:1, 1:] *= 1e-30\n        else:\n            y[:2] -= -(1 - 2) - 3\n        publish bf16(y) to out[:, cols]\n",
        );
        let body = &only_fn(&file).body;
        let StmtKind::Region(outer) = &body.stmts[0].kind else {
            panic!("expected region")
        };
        assert_eq!(outer.body.stmts.len(), 7);
        let StmtKind::Region(pipeline) = &outer.body.stmts[1].kind else {
            panic!("expected pipeline")
        };
        assert_eq!(
            (pipeline.mode, pipeline.body.stmts.len()),
            (RegionMode::Pipeline, 2)
        );
        let StmtKind::For { body: inline, .. } = &outer.body.stmts[4].kind else {
            panic!("expected for")
        };
        assert_eq!(inline.stmts.len(), 2);
        let StmtKind::If { els: Some(els), .. } = &outer.body.stmts[5].kind else {
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

        let text = "fn f(x: f32):\n    let tile = x\n";
        let err = parse(text).unwrap_err();
        assert_eq!(
            &text[err.span.start as usize..err.span.end as usize],
            "tile"
        );
        assert!(parse("fn f():\n    a..b..c\n").is_err());
        assert!(parse("fn f():\n    f(x) = 1\n").is_err());
        assert!(parse("lower f for cpu:\n    g()\n").is_err());
        assert!(parse("fn f(x: f32) -> f32\n    where x > 0 = portable\n").is_err());
    }
}
