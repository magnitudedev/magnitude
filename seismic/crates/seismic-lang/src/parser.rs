//! Recursive-descent parser with Pratt expression parsing.

use crate::ast::*;
use crate::lexer::lex;
use crate::span::{Diagnostic, Span};
use crate::token::{Kw, Op, Tok, Token};

pub fn parse(text: &str) -> Result<File, Diagnostic> {
    let tokens = lex(text)?;
    Parser { tokens, pos: 0 }.file()
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

type PResult<T> = Result<T, Diagnostic>;

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

    fn eat_op(&mut self, op: Op) -> bool {
        if self.at_op(op) {
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
            Err(self.error(format!("expected `{}`, found {}", op.text(), self.peek().describe())))
        }
    }

    fn expect_kw(&mut self, kw: Kw) -> PResult<Span> {
        if self.at_kw(kw) {
            Ok(self.bump().span)
        } else {
            Err(self.error(format!("expected `{}`, found {}", kw.text(), self.peek().describe())))
        }
    }

    fn expect_name(&mut self) -> PResult<Ident> {
        match self.peek().clone() {
            Tok::Name(name) => {
                let span = self.bump().span;
                Ok(Ident { name, span })
            }
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

    // ---- declarations ----

    fn file(&mut self) -> PResult<File> {
        let mut decls = Vec::new();
        loop {
            while matches!(self.peek(), Tok::Newline) {
                self.bump();
            }
            match self.peek() {
                Tok::Eof => break,
                Tok::Kw(Kw::Fn) => decls.push(Decl::Fn(self.fn_decl(Kw::Fn)?)),
                Tok::Kw(Kw::Construct) => decls.push(Decl::Construct(self.fn_decl(Kw::Construct)?)),
                Tok::Kw(Kw::Lower) => decls.push(Decl::Lower(self.lower_decl()?)),
                other => return Err(self.error(format!("expected `fn`, `construct` or `lower`, found {}", other.describe()))),
            }
        }
        Ok(File { decls })
    }

    fn fn_decl(&mut self, kw: Kw) -> PResult<FnDecl> {
        let start = self.expect_kw(kw)?;
        let name = self.expect_name()?;
        let mut shape = Vec::new();
        if self.eat_op(Op::LBracket) {
            loop {
                shape.push(self.expect_name()?);
                if !self.eat_op(Op::Comma) {
                    break;
                }
            }
            self.expect_op(Op::RBracket)?;
        }
        self.expect_op(Op::LParen)?;
        let mut params = Vec::new();
        if !self.at_op(Op::RParen) {
            loop {
                let pname = self.expect_name()?;
                self.expect_op(Op::Colon)?;
                let ty = self.type_expr()?;
                params.push(Param { name: pname, ty });
                if !self.eat_op(Op::Comma) {
                    break;
                }
            }
        }
        self.expect_op(Op::RParen)?;
        self.expect_op(Op::Colon)?;
        let body = self.block()?;
        let span = start.to(body.span);
        Ok(FnDecl { name, shape, params, body, span })
    }

    fn lower_decl(&mut self) -> PResult<LowerDecl> {
        let start = self.expect_kw(Kw::Lower)?;
        let name = self.expect_name()?;
        // `lower name: portable` records that the composition is the realization here.
        if self.eat_op(Op::Colon) {
            if self.at_kw(Kw::Portable) {
                let end = self.bump().span;
                self.expect_newline()?;
                return Ok(LowerDecl { name, shape: Vec::new(), params: Vec::new(), body: None, span: start.to(end) });
            }
            return Err(Diagnostic::new(name.span, "a lowering restates the construct's signature: `lower name[shape](params):`"));
        }
        // Otherwise the block restates the construct's signature, checked against it.
        let mut shape = Vec::new();
        if self.eat_op(Op::LBracket) {
            loop {
                shape.push(self.expect_name()?);
                if !self.eat_op(Op::Comma) {
                    break;
                }
            }
            self.expect_op(Op::RBracket)?;
        }
        self.expect_op(Op::LParen)?;
        let mut params = Vec::new();
        if !self.at_op(Op::RParen) {
            loop {
                let pname = self.expect_name()?;
                self.expect_op(Op::Colon)?;
                let ty = self.type_expr()?;
                params.push(Param { name: pname, ty });
                if !self.eat_op(Op::Comma) {
                    break;
                }
            }
        }
        self.expect_op(Op::RParen)?;
        self.expect_op(Op::Colon)?;
        let body = self.block()?;
        let span = start.to(body.span);
        Ok(LowerDecl { name, shape, params, body: Some(body), span })
    }

    fn type_expr(&mut self) -> PResult<TypeExpr> {
        let head = if self.at_kw(Kw::Tile) {
            let span = self.bump().span;
            Ident { name: "tile".to_string(), span }
        } else {
            self.expect_name()?
        };
        let mut span = head.span;
        let mut shape = Vec::new();
        if self.eat_op(Op::LBracket) {
            loop {
                shape.push(self.expr()?);
                if !self.eat_op(Op::Comma) {
                    break;
                }
            }
            span = span.to(self.expect_op(Op::RBracket)?);
        }
        let elem = match self.peek() {
            Tok::Name(_) => {
                let e = self.expect_name()?;
                span = span.to(e.span);
                Some(e)
            }
            _ => None,
        };
        Ok(TypeExpr { head, shape, elem, span })
    }

    // ---- statements ----

    /// A block after `:`. Either an indented suite or simple statements on the same line.
    fn block(&mut self) -> PResult<Block> {
        if matches!(self.peek(), Tok::Newline) {
            self.bump();
            let start = self.span();
            if !matches!(self.peek(), Tok::Indent) {
                return Err(self.error("expected an indented block".into()));
            }
            self.bump();
            let mut stmts = Vec::new();
            while !matches!(self.peek(), Tok::Dedent | Tok::Eof) {
                if matches!(self.peek(), Tok::Newline) {
                    self.bump();
                    continue;
                }
                self.statement_line(&mut stmts)?;
            }
            let end = self.span();
            if matches!(self.peek(), Tok::Dedent) {
                self.bump();
            }
            return Ok(Block { span: start.to(end), stmts });
        }
        let start = self.span();
        let mut stmts = Vec::new();
        self.simple_statements(&mut stmts)?;
        let span = start.to(self.prev_span());
        Ok(Block { stmts, span })
    }

    fn statement_line(&mut self, out: &mut Vec<Stmt>) -> PResult<()> {
        match self.peek() {
            Tok::Kw(Kw::For) => {
                let start = self.bump().span;
                let mut targets = vec![self.expect_name()?];
                while self.eat_op(Op::Comma) {
                    targets.push(self.expect_name()?);
                }
                self.expect_kw(Kw::In)?;
                let iter = self.expr()?;
                self.expect_op(Op::Colon)?;
                let body = self.block()?;
                let span = start.to(body.span);
                out.push(Stmt { kind: StmtKind::For { targets, iter, body }, span });
                Ok(())
            }
            Tok::Kw(Kw::If) => {
                let start = self.bump().span;
                let cond = self.expr()?;
                self.expect_op(Op::Colon)?;
                let then = self.block()?;
                let mut span = start.to(then.span);
                let mut els = None;
                if self.at_kw(Kw::Else) {
                    self.bump();
                    self.expect_op(Op::Colon)?;
                    let b = self.block()?;
                    span = span.to(b.span);
                    els = Some(b);
                }
                out.push(Stmt { kind: StmtKind::If { cond, then, els }, span });
                Ok(())
            }
            _ => self.simple_statements(out),
        }
    }

    /// `simple (";" simple)* NEWLINE`
    fn simple_statements(&mut self, out: &mut Vec<Stmt>) -> PResult<()> {
        loop {
            out.push(self.simple_statement()?);
            if self.eat_op(Op::Semi) {
                if matches!(self.peek(), Tok::Newline | Tok::Eof | Tok::Dedent) {
                    break;
                }
                continue;
            }
            break;
        }
        self.expect_newline()
    }

    fn simple_statement(&mut self) -> PResult<Stmt> {
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
                self.bump();
                let value = self.expr()?;
                let span = target.span.to(value.span);
                Ok(Stmt { kind: StmtKind::Assign { target, op, value }, span })
            }
            None => {
                let span = target.span;
                Ok(Stmt { kind: StmtKind::Expr(target), span })
            }
        }
    }

    // ---- expressions ----

    pub(crate) fn expr(&mut self) -> PResult<Expr> {
        self.expr_bp(0)
    }

    fn expr_bp(&mut self, min_bp: u8) -> PResult<Expr> {
        let mut lhs = self.unary()?;
        loop {
            let op = match self.peek() {
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
                _ => break,
            };
            let bp = op.precedence();
            if bp <= min_bp {
                break;
            }
            self.bump();
            let rhs = self.expr_bp(bp)?;
            let span = lhs.span.to(rhs.span);
            lhs = Expr { kind: ExprKind::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }, span };
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> PResult<Expr> {
        let (op, bp) = match self.peek() {
            Tok::Op(Op::Minus) => (UnaryOp::Neg, UNARY_PRECEDENCE),
            Tok::Op(Op::Tilde) => (UnaryOp::BitNot, UNARY_PRECEDENCE),
            Tok::Kw(Kw::Not) => (UnaryOp::Not, NOT_PRECEDENCE),
            _ => return self.postfix(),
        };
        let start = self.bump().span;
        let expr = self.expr_bp(bp - 1)?;
        let span = start.to(expr.span);
        Ok(Expr { kind: ExprKind::Unary { op, expr: Box::new(expr) }, span })
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let mut e = self.primary()?;
        loop {
            match self.peek() {
                Tok::Op(Op::LParen) => {
                    self.bump();
                    let args = self.args()?;
                    let end = self.expect_op(Op::RParen)?;
                    let span = e.span.to(end);
                    e = Expr { kind: ExprKind::Call { callee: Box::new(e), bindings: Vec::new(), args }, span };
                }
                Tok::Op(Op::LBracket) if matches!(self.peek_at(1), Tok::Name(_)) && matches!(self.peek_at(2), Tok::Op(Op::Assign)) => {
                    // Shape bindings on a call: `f[R = 64, S = 192](...)`.
                    self.bump();
                    let mut bindings = Vec::new();
                    loop {
                        let param = self.expect_name()?;
                        self.expect_op(Op::Assign)?;
                        let value = self.expr()?;
                        bindings.push((param, value));
                        if !self.eat_op(Op::Comma) {
                            break;
                        }
                    }
                    self.expect_op(Op::RBracket)?;
                    self.expect_op(Op::LParen)?;
                    let args = self.args()?;
                    let end = self.expect_op(Op::RParen)?;
                    let span = e.span.to(end);
                    e = Expr { kind: ExprKind::Call { callee: Box::new(e), bindings, args }, span };
                }
                Tok::Op(Op::LBracket) => {
                    self.bump();
                    let indices = self.indices()?;
                    let end = self.expect_op(Op::RBracket)?;
                    let span = e.span.to(end);
                    e = Expr { kind: ExprKind::Index { base: Box::new(e), indices }, span };
                }
                Tok::Op(Op::Dot) => {
                    self.bump();
                    let name = self.expect_name()?;
                    let span = e.span.to(name.span);
                    e = Expr { kind: ExprKind::Attr { base: Box::new(e), name }, span };
                }
                _ => return Ok(e),
            }
        }
    }

    fn args(&mut self) -> PResult<Vec<Arg>> {
        let mut args = Vec::new();
        if self.at_op(Op::RParen) {
            return Ok(args);
        }
        loop {
            let name = match (self.peek(), self.peek_at(1)) {
                (Tok::Name(n), Tok::Op(Op::Assign)) => {
                    let n = n.clone();
                    let span = self.bump().span;
                    self.bump();
                    Some(Ident { name: n, span })
                }
                _ => None,
            };
            let value = self.expr()?;
            args.push(Arg { name, value });
            if !self.eat_op(Op::Comma) {
                break;
            }
            if self.at_op(Op::RParen) {
                break;
            }
        }
        Ok(args)
    }

    fn indices(&mut self) -> PResult<Vec<Index>> {
        let mut out = Vec::new();
        loop {
            let start = if self.at_op(Op::Colon) { None } else { Some(self.expr()?) };
            if self.eat_op(Op::Colon) {
                let end = if matches!(self.peek(), Tok::Op(Op::Comma) | Tok::Op(Op::RBracket)) { None } else { Some(self.expr()?) };
                out.push(Index::Slice { start, end });
            } else {
                out.push(Index::Expr(start.expect("non-slice index has an expression")));
            }
            if !self.eat_op(Op::Comma) {
                break;
            }
        }
        Ok(out)
    }

    fn primary(&mut self) -> PResult<Expr> {
        let span = self.span();
        match self.peek().clone() {
            Tok::Int(v) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Int(v), span })
            }
            Tok::Float(v) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Float(v), span })
            }
            Tok::Kw(Kw::Inf) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Inf, span })
            }
            Tok::Kw(Kw::True) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Bool(true), span })
            }
            Tok::Kw(Kw::False) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Bool(false), span })
            }
            Tok::Kw(Kw::Tile) => {
                self.bump();
                self.expect_op(Op::LBracket)?;
                let mut shape = Vec::new();
                loop {
                    shape.push(self.expr()?);
                    if !self.eat_op(Op::Comma) {
                        break;
                    }
                }
                self.expect_op(Op::RBracket)?;
                let dtype = self.expect_name()?;
                let span = span.to(dtype.span);
                Ok(Expr { kind: ExprKind::Tile { shape, dtype }, span })
            }
            Tok::Name(name) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Name(Ident { name, span }), span })
            }
            Tok::Op(Op::LParen) => {
                self.bump();
                // Lambda: `(a, b) -> expr` or `() -> expr`.
                if self.looks_like_lambda() {
                    let mut params = Vec::new();
                    while !self.at_op(Op::RParen) {
                        params.push(self.expect_name()?);
                        if !self.eat_op(Op::Comma) {
                            break;
                        }
                    }
                    self.expect_op(Op::RParen)?;
                    self.expect_op(Op::Arrow)?;
                    let body = self.expr()?;
                    let span = span.to(body.span);
                    return Ok(Expr { kind: ExprKind::Lambda { params, body: Box::new(body) }, span });
                }
                let first = self.expr()?;
                if self.at_op(Op::RParen) {
                    self.bump();
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
                Ok(Expr { kind: ExprKind::Tuple(items), span: span.to(end) })
            }
            other => Err(self.error(format!("expected an expression, found {}", other.describe()))),
        }
    }

    /// After `(`: names separated by commas, `)`, then `->`.
    fn looks_like_lambda(&self) -> bool {
        let mut i = 0;
        loop {
            match self.peek_at(i) {
                Tok::Op(Op::RParen) => return matches!(self.peek_at(i + 1), Tok::Op(Op::Arrow)),
                Tok::Name(_) => {
                    i += 1;
                    match self.peek_at(i) {
                        Tok::Op(Op::Comma) => i += 1,
                        Tok::Op(Op::RParen) => continue,
                        _ => return false,
                    }
                }
                _ => return false,
            }
        }
    }
}
