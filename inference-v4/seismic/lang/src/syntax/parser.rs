//! Recursive-descent parser with Pratt expression parsing.
use super::ast::*;
use super::lexer::lex;
use super::token::{Kw, Op, Tok, Token};
use crate::checked::DiagnosticRule;
use crate::span::{Diagnostic, Span};

pub(crate) fn parse(text: &str) -> Result<File, Diagnostic> {
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
                Tok::Kw(Kw::Native) => decls.push(Decl::Native(self.native_decl()?)),
                Tok::Indent => {
                    return Err(self.error(
                        "unexpected indentation; declarations start at the left margin".into(),
                    ));
                }
                other => {
                    return Err(self.error(format!(
                        "expected `fn`, `lower`, or `native`, found {}",
                        other.describe()
                    )));
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
        let mut constraint = None;
        if self.eat_kw(Kw::Where) {
            constraint = Some(self.expr()?);
            self.expect_newline()?;
        }
        let mut scratch = Vec::new();
        while self.at_word("scratch") {
            let begin = self.bump().span;
            let name = self.expect_name()?;
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
            let when = self.native_when()?;
            scratch.push(NativeScratchDecl {
                name,
                bytes,
                when,
                span: begin.to(self.prev_span()),
            });
            self.expect_newline()?;
        }
        let mut launches = Vec::new();
        while self.at_word("launch") {
            launches.push(self.native_launch()?);
        }
        if launches.is_empty() {
            return Err(self.error(format!(
                "expected `launch <kernel>:`, found {}; a native declaration lists `static`, `params`, `elements`, `where`, `scratch`, then one or more launches",
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
            constraint,
            scratch,
            launches,
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

    /// `[arithmetic] NAME in [V, ..]`
    fn native_param(&mut self) -> PResult<NativeParamDecl> {
        let begin = self.span();
        let arithmetic = matches!(self.peek_at(1), Tok::Name(_)) && self.at_word("arithmetic");
        if arithmetic {
            self.bump();
        }
        let name = self.expect_name()?;
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
        Ok(NativeParamDecl {
            name,
            arithmetic,
            values,
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

    /// An optional `when CONDITION` (`when` is a contextual word).
    fn native_when(&mut self) -> PResult<Option<Expr>> {
        if !self.at_word("when") {
            return Ok(None);
        }
        self.bump();
        Ok(Some(self.expr()?))
    }

    /// `launch KERNEL [when CONDITION]:` followed by its indented geometry.
    fn native_launch(&mut self) -> PResult<NativeLaunchDecl> {
        let begin = self.bump().span;
        let kernel = self.expect_name()?;
        let when = self.native_when()?;
        self.expect_op(Op::Colon)?;
        self.native_block_start("launch")?;
        let threadgroups = self.native_launch_property("threadgroups")?;
        self.expect_newline()?;
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
        Ok(NativeLaunchDecl {
            kernel,
            when,
            threadgroups,
            threads_per_threadgroup,
            shared_bytes,
            span: begin.to(self.prev_span()),
        })
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
        let name = self.expect_name()?;
        self.expect_op(Op::Colon)?;
        let ty = self.type_expr()?;
        Ok(Param { name, ty })
    }

    fn type_expr(&mut self) -> PResult<TypeExpr> {
        let start = self.span();
        let kind = match self.peek().clone() {
            Tok::Op(Op::Amp) => {
                self.bump();
                let mutable = self.eat_kw(Kw::Mut);
                let tensor = self.expect_name()?;
                if tensor.name != "tensor" || !self.at_op(Op::LBracket) {
                    return Err(Diagnostic::with_rule(
                        DiagnosticRule::Syntax,
                        tensor.span,
                        "a borrow type is `&tensor[...] T` or `&mut tensor[...] T`",
                    ));
                }
                self.shaped(if mutable {
                    ShapedHead::MutTensor
                } else {
                    ShapedHead::SharedTensor
                })?
            }
            Tok::Kw(Kw::Void) => {
                self.bump();
                TypeKind::Void
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
                self.logical_for(false)?
            }
            Tok::Kw(Kw::Parallel) if self.peek_at(1) == &Tok::Kw(Kw::For) => {
                self.bump();
                self.bump();
                self.logical_for(true)?
            }
            Tok::Kw(Kw::If) => self.if_stmt()?,
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
        let targets = self.comma_list(Self::expect_name)?;
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
            Tok::Name(name) if name == "tensor" => {
                self.bump();
                let (shape, elem) = self.shape_and_elem()?;
                return Ok(Expr {
                    kind: ExprKind::Tensor { shape, elem },
                    span: span.to(self.prev_span()),
                });
            }
            Tok::Name(name) => ExprKind::Name(Ident { name, span }),
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
        let file = parse(text).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
        let printed = print(&file);
        let again = parse(&printed).unwrap_or_else(|d| panic!("{d:?}\n{printed}"));
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
            "native copy for cpu from \"copy.rs\":\n    params (ROWS in [4, 1])\n    elements (A in [f32, u32], B in [bf16])\n    launch copy:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert_eq!(native.elements.len(), 2);
        assert_eq!(native.elements[0].name.name, "A");
        assert_eq!(
            native.elements[0].dtypes.iter().map(|dtype| dtype.name.as_str()).collect::<Vec<_>>(),
            ["f32", "u32"]
        );
    }

    #[test]
    fn specialized_native_declarations_round_trip() {
        let file = round_trip(
            "native scale for cuda from \"scale.cu\":\n    static (N)\n    params (arithmetic PARTS in [1, 2, 4], WIDTH in [64, 128])\n    where PARTS * WIDTH <= N and WIDTH >= 64\n    scratch partials bytes (PARTS * N * 4)\n    launch scale_partial:\n        threadgroups (ceil_div(N, WIDTH), PARTS, 1)\n        threads_per_threadgroup (WIDTH, 1, 1)\n        shared_bytes (max(WIDTH * 4, 256))\n    launch scale_merge:\n        threadgroups (ceil_div(N, 256), 1, 1)\n        threads_per_threadgroup (min(N, 256), 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert_eq!(native.statics.len(), 1);
        assert_eq!(native.params.len(), 2);
        assert!(native.params[0].arithmetic);
        assert!(matches!(
            native.constraint.as_ref().map(|constraint| &constraint.kind),
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
    fn conditional_native_launches_and_scratch_round_trip() {
        let file = round_trip(
            "native rows for metal from \"rows.metal\":\n    params (SPLIT in [1, 2])\n    where SPLIT == 1 or (SPLIT > 1 and SPLIT < 4)\n    scratch normalized bytes (O * H * 2) when O > 8\n    scratch partials bytes (SPLIT * O * 4) when (SPLIT > 1 or O >= 9) and O != 0\n    launch rows_normalize when O > 8:\n        threadgroups (O, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n    launch rows_gemv when O <= 2 or (INT8 == 1 and O <= 8):\n        threadgroups (ceil_div(F, 8), 1, 1)\n        threads_per_threadgroup (128, 1, 1)\n        shared_bytes (O * 72)\n    launch rows_merge:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n",
        );
        let [Decl::Native(native)] = file.decls.as_slice() else {
            panic!("expected one native implementation")
        };
        assert!(matches!(
            native.constraint.as_ref().map(|constraint| &constraint.kind),
            Some(ExprKind::Binary { op: BinaryOp::Or, .. })
        ));
        assert!(native.scratch.iter().all(|scratch| scratch.when.is_some()));
        assert!(matches!(
            native.scratch[1].when.as_ref().map(|when| &when.kind),
            Some(ExprKind::Binary { op: BinaryOp::And, .. })
        ));
        assert!(native.launches[0].when.is_some());
        assert!(matches!(
            native.launches[1].when.as_ref().map(|when| &when.kind),
            Some(ExprKind::Binary { op: BinaryOp::Or, .. })
        ));
        assert!(native.launches[2].when.is_none());
        let printed = print(&file);
        assert!(
            printed.contains("when (SPLIT > 1 or O >= 9) and O != 0\n"),
            "{printed}"
        );
        assert!(printed.contains("launch rows_gemv when O <= 2 or INT8 == 1 and O <= 8:\n"), "{printed}");
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
        assert!(err
            .message
            .contains("written as `lower NAME … for BACKEND`"));
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
