//! Canonical printer: `parse(print(parse(x))) == parse(x)`.
//!
//! Headers print on one line, blocks indent by four spaces, comments are not preserved.
use super::ast::*;
use super::parser::{binary_bp, NOT_BP, RANGE_BP, UNARY_BP};
use std::fmt::Write;

pub(crate) fn print(file: &File) -> String {
    let mut p = Printer {
        out: String::new(),
        level: 0,
    };
    for (i, decl) in file.decls.iter().enumerate() {
        if i > 0 {
            p.out.push('\n');
        }
        p.decl(decl);
    }
    p.out
}

struct Printer {
    out: String,
    level: usize,
}

impl Printer {
    fn indent(&mut self) {
        for _ in 0..self.level {
            self.out.push_str("    ");
        }
    }

    fn list<T>(&mut self, items: &[T], mut item: impl FnMut(&mut Self, &T)) {
        for (i, it) in items.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            item(self, it);
        }
    }

    fn names(&mut self, names: &[Ident]) {
        self.list(names, |p, n| p.out.push_str(&n.name));
    }

    // ---- declarations ----

    fn decl(&mut self, decl: &Decl) {
        match decl {
            Decl::Fn(f) => {
                let _ = write!(self.out, "fn {}", f.name.name);
                self.signature(&f.signature);
                self.requires(&f.requires);
                self.predicates(&f.signature.predicates);
                self.out.push_str(":\n");
                self.block(&f.body);
            }
            Decl::Lower(l) => {
                let _ = write!(self.out, "lower {}", l.name.name);
                self.signature(&l.signature);
                let _ = write!(self.out, " for {}", l.target.name);
                self.requires(&l.requires);
                self.predicates(&l.predicates);
                self.out.push_str(":\n");
                self.block(&l.body);
            }
            Decl::Native(n) => {
                let escaped = n.source.replace('\\', "\\\\").replace('"', "\\\"");
                let _ = write!(
                    self.out,
                    "native {} for {} from \"{}\":\n",
                    n.function.name, n.target.name, escaped
                );
                self.level += 1;
                if !n.statics.is_empty() {
                    self.indent();
                    self.out.push_str("static (");
                    self.names(&n.statics);
                    self.out.push_str(")\n");
                }
                if !n.params.is_empty() {
                    self.indent();
                    self.native_params(&n.params);
                }
                if !n.elements.is_empty() {
                    self.indent();
                    self.out.push_str("elements (");
                    self.list(&n.elements, |p, elements| {
                        let _ = write!(p.out, "{} in [", elements.name.name);
                        p.names(&elements.dtypes);
                        p.out.push(']');
                    });
                    self.out.push_str(")\n");
                }
                if let Some(constraint) = &n.constraint {
                    self.indent();
                    self.out.push_str("where ");
                    self.expr(constraint, 0);
                    self.out.push('\n');
                }
                for scratch in &n.scratch {
                    self.indent();
                    let _ = write!(self.out, "scratch {} bytes (", scratch.name.name);
                    self.expr(&scratch.bytes, 0);
                    self.out.push(')');
                    if let Some(when) = &scratch.when {
                        self.out.push_str(" when ");
                        self.expr(when, 0);
                    }
                    self.out.push('\n');
                }
                for launch in &n.launches {
                    self.indent();
                    let _ = write!(self.out, "launch {}", launch.kernel.name);
                    if let Some(when) = &launch.when {
                        self.out.push_str(" when ");
                        self.expr(when, 0);
                    }
                    self.out.push_str(":\n");
                    self.level += 1;
                    if !launch.params.is_empty() {
                        self.indent();
                        self.native_params(&launch.params);
                    }
                    if !launch.reads.is_empty() {
                        self.indent();
                        self.out.push_str("reads (");
                        self.list(&launch.reads, |p, name| p.out.push_str(&name.name));
                        self.out.push_str(")\n");
                    }
                    self.indent();
                    self.out.push_str("threadgroups (");
                    self.list(&launch.threadgroups, |p, expr| p.expr(expr, 0));
                    self.out.push_str(")\n");
                    self.indent();
                    self.out.push_str("threads_per_threadgroup (");
                    self.list(&launch.threads_per_threadgroup, |p, expr| p.expr(expr, 0));
                    self.out.push_str(")\n");
                    if let Some(bytes) = &launch.shared_bytes {
                        self.indent();
                        self.out.push_str("shared_bytes (");
                        self.expr(bytes, 0);
                        self.out.push_str(")\n");
                    }
                    self.level -= 1;
                }
                self.level -= 1;
            }
        }
    }

    fn native_params(&mut self, params: &[NativeParamDecl]) {
        self.out.push_str("params (");
        self.list(params, |p, param| {
            if param.code {
                p.out.push_str("code ");
            }
            if param.arithmetic {
                p.out.push_str("arithmetic ");
            }
            let values = param
                .values
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let _ = write!(p.out, "{} in [{values}]", param.name.name);
        });
        self.out.push_str(")\n");
    }

    fn signature(&mut self, s: &Signature) {
        if !s.shape.is_empty() {
            self.out.push('[');
            self.names(&s.shape);
            self.out.push(']');
        }
        self.out.push('(');
        self.list(&s.params, |p, param| {
            let _ = write!(p.out, "{}: ", param.name.name);
            p.ty(&param.ty);
        });
        self.out.push(')');
        if let Some(result) = &s.result {
            self.out.push_str(" -> ");
            self.ty(result);
        }
    }

    fn predicates(&mut self, predicates: &[Expr]) {
        for (i, predicate) in predicates.iter().enumerate() {
            self.out.push_str(if i == 0 { " where " } else { " and " });
            self.expr(predicate, binary_bp(BinaryOp::And));
        }
    }

    fn requires(&mut self, capabilities: &[CapabilityPath]) {
        if capabilities.is_empty() {
            return;
        }
        self.out.push_str(" requires ");
        self.list(capabilities, |p, capability| {
            let _ = write!(
                p.out,
                "{}.{}",
                capability.backend.name, capability.capability.name
            );
        });
    }

    fn ty(&mut self, ty: &TypeExpr) {
        match &ty.kind {
            TypeKind::Scalar(name) => self.out.push_str(&name.name),
            TypeKind::Index(bound) => {
                self.out.push_str("index[");
                self.expr(bound, 0);
                self.out.push(']');
            }
            TypeKind::Range(bound) => {
                self.out.push_str("range[");
                self.expr(bound, 0);
                self.out.push(']');
            }
            TypeKind::Shaped { head, shape, elem } => {
                self.out.push_str(match head {
                    ShapedHead::Tensor => "tensor",
                    ShapedHead::SharedTensor => "&tensor",
                    ShapedHead::MutTensor => "&mut tensor",
                });
                self.shape_and_elem(shape, elem);
            }
            TypeKind::Tuple(items) => {
                self.out.push('(');
                self.list(items, |p, item| p.ty(item));
                self.out.push(')');
            }
            TypeKind::Void => self.out.push_str("void"),
        }
    }

    fn shape_and_elem(&mut self, shape: &[Expr], elem: &Ident) {
        self.out.push('[');
        self.list(shape, |p, e| p.expr(e, 0));
        let _ = write!(self.out, "] {}", elem.name);
    }

    // ---- statements ----

    fn block(&mut self, block: &Block) {
        self.level += 1;
        for stmt in &block.stmts {
            self.indent();
            self.stmt(stmt);
        }
        self.level -= 1;
    }

    /// After `:`; the header's indentation is already written.
    fn suite(&mut self, block: &Block) {
        self.out.push_str(":\n");
        self.block(block);
    }

    fn pattern(&mut self, pattern: &Pattern) {
        match pattern {
            Pattern::Name(name) => self.out.push_str(&name.name),
            Pattern::Tuple(items) => {
                self.out.push('(');
                self.list(items, |p, item| p.pattern(item));
                self.out.push(')');
            }
        }
    }

    /// A trailing expression value.
    fn value(&mut self, value: &Expr) {
        self.expr(value, 0);
        self.out.push('\n');
    }

    fn values(&mut self, keyword: &str, values: &[Expr]) {
        self.out.push_str(keyword);
        if !values.is_empty() {
            self.out.push(' ');
        }
        self.list(values, |p, v| p.expr(v, 0));
        self.out.push('\n');
    }

    fn stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let {
                mutable,
                pattern,
                value,
            } => {
                self.out
                    .push_str(if *mutable { "let mut " } else { "let " });
                self.pattern(pattern);
                self.out.push_str(" = ");
                self.value(value);
            }
            StmtKind::Assign { target, op, value } => {
                self.expr(target, 0);
                let _ = write!(self.out, " {} ", op.text());
                self.value(value);
            }
            StmtKind::For {
                parallel,
                targets,
                iter,
                body,
            } => {
                self.out
                    .push_str(if *parallel { "parallel for " } else { "for " });
                self.names(targets);
                self.out.push_str(" in ");
                self.expr(iter, 0);
                self.suite(body);
            }
            StmtKind::If { cond, then, els } => self.if_stmt(cond, then, els.as_ref()),
            StmtKind::Return(values) => self.values("return", values),
            StmtKind::Expr(e) => self.value(e),
        }
    }

    /// An `else` block holding exactly one `if` prints as `else if`.
    fn if_stmt(&mut self, cond: &Expr, then: &Block, els: Option<&Block>) {
        self.out.push_str("if ");
        self.expr(cond, 0);
        self.suite(then);
        let Some(els) = els else { return };
        self.indent();
        match els.stmts.as_slice() {
            [Stmt {
                kind: StmtKind::If { cond, then, els },
                ..
            }] => {
                self.out.push_str("else ");
                self.if_stmt(cond, then, els.as_ref());
            }
            _ => {
                self.out.push_str("else");
                self.suite(els);
            }
        }
    }

    // ---- expressions ----

    fn float(&mut self, v: f64) {
        let magnitude = v.abs();
        let _ = if v != 0.0 && !(1e-4..1e16).contains(&magnitude) {
            write!(self.out, "{v:e}")
        } else if v.fract() == 0.0 {
            write!(self.out, "{v:.1}")
        } else {
            write!(self.out, "{v}")
        };
    }

    /// `min_bp` is the binding power of the enclosing operator; parenthesize when ours is not higher.
    fn expr(&mut self, e: &Expr, min_bp: u8) {
        match &e.kind {
            ExprKind::Int(v) => {
                let _ = write!(self.out, "{v}");
            }
            ExprKind::Float(v) => self.float(*v),
            ExprKind::Inf => self.out.push_str("inf"),
            ExprKind::Bool(b) => self.out.push_str(if *b { "true" } else { "false" }),
            ExprKind::Name(n) => self.out.push_str(&n.name),
            ExprKind::Tuple(items) => {
                self.out.push('(');
                self.list(items, |p, item| p.expr(item, 0));
                if items.len() == 1 {
                    self.out.push(',');
                }
                self.out.push(')');
            }
            ExprKind::Range { lo, hi } => {
                let paren = RANGE_BP <= min_bp;
                if paren {
                    self.out.push('(');
                }
                self.expr(lo, RANGE_BP);
                self.out.push_str("..");
                self.expr(hi, RANGE_BP);
                if paren {
                    self.out.push(')');
                }
            }
            ExprKind::Tensor { shape, elem } => {
                self.out.push_str("tensor");
                self.shape_and_elem(shape, elem);
            }
            ExprKind::Call {
                callee,
                bindings,
                args,
            } => {
                self.expr(callee, UNARY_BP);
                if !bindings.is_empty() {
                    self.out.push('[');
                    self.list(bindings, |p, (param, value)| {
                        let _ = write!(p.out, "{} = ", param.name);
                        p.expr(value, 0);
                    });
                    self.out.push(']');
                }
                self.out.push('(');
                self.list(args, |p, arg| {
                    if let Some(name) = &arg.name {
                        let _ = write!(p.out, "{}=", name.name);
                    }
                    p.expr(&arg.value, 0);
                });
                self.out.push(')');
            }
            ExprKind::Index { base, indices } => {
                self.expr(base, UNARY_BP);
                self.out.push('[');
                self.list(indices, |p, index| match index {
                    Index::Expr(e) => p.expr(e, 0),
                    Index::Slice { start, end } => {
                        if let Some(start) = start {
                            p.expr(start, 0);
                        }
                        p.out.push(':');
                        if let Some(end) = end {
                            p.expr(end, 0);
                        }
                    }
                });
                self.out.push(']');
            }
            ExprKind::Attr { base, name } => {
                self.expr(base, UNARY_BP);
                let _ = write!(self.out, ".{}", name.name);
            }
            ExprKind::Unary { op, expr } => {
                let bp = if *op == UnaryOp::Not {
                    NOT_BP
                } else {
                    UNARY_BP
                };
                let paren = bp <= min_bp;
                if paren {
                    self.out.push('(');
                }
                self.out.push_str(op.text());
                self.expr(expr, bp - 1);
                if paren {
                    self.out.push(')');
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let bp = binary_bp(*op);
                let paren = bp <= min_bp;
                if paren {
                    self.out.push('(');
                }
                self.expr(lhs, bp - 1);
                let _ = write!(self.out, " {} ", op.text());
                self.expr(rhs, bp);
                if paren {
                    self.out.push(')');
                }
            }
        }
    }
}
