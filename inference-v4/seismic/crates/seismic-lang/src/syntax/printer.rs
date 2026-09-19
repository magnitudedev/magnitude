//! Canonical printer: `parse(print(parse(x))) == parse(x)`.
//!
//! Headers print on one line, blocks indent by four spaces, comments are not preserved.
use super::ast::*;
use super::parser::{binary_bp, NOT_BP, RANGE_BP, UNARY_BP};
use std::fmt::Write;

pub fn print(file: &File) -> String {
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

pub fn expr_to_string(e: &Expr) -> String {
    let mut p = Printer {
        out: String::new(),
        level: 0,
    };
    p.expr(e, 0);
    p.out
}

pub fn type_to_string(ty: &TypeExpr) -> String {
    let mut p = Printer {
        out: String::new(),
        level: 0,
    };
    p.ty(ty);
    p.out
}

struct Printer {
    out: String,
    level: usize,
}

fn is_region(e: &Expr) -> bool {
    matches!(e.kind, ExprKind::Region(_))
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
                if f.admit {
                    self.out.push_str("admit ");
                }
                let _ = write!(self.out, "fn {}", f.name.name);
                self.signature(&f.signature);
                if let Some(target) = &f.target {
                    let _ = write!(self.out, " for {}", target.name);
                }
                self.predicates(&f.signature.predicates);
                self.out.push_str(":\n");
                self.block(&f.body);
            }
            Decl::Lower(l) => {
                let _ = write!(self.out, "lower {}", l.name.name);
                self.signature(&l.signature);
                let _ = write!(self.out, " for {}", l.target.name);
                self.predicates(&l.predicates);
                self.out.push_str(":\n");
                self.block(&l.body);
            }
        }
    }

    fn signature(&mut self, s: &Signature) {
        if !s.shape.is_empty() {
            self.out.push('[');
            self.names(&s.shape);
            self.out.push(']');
        }
        self.out.push('(');
        self.list(&s.params, |p, param| {
            p.out.push_str(match param.mode {
                Mode::In => "",
                Mode::Out => "out ",
                Mode::Inout => "inout ",
            });
            let _ = write!(p.out, "{}: ", param.name.name);
            p.ty(&param.ty);
        });
        self.out.push(')');
        if !s.aliases.is_empty() {
            self.out.push(' ');
            self.list(&s.aliases, |p, (a, b)| {
                let _ = write!(p.out, "alias({}, {})", a.name, b.name);
            });
        }
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

    fn ty(&mut self, ty: &TypeExpr) {
        match &ty.kind {
            TypeKind::Scalar(name) => self.out.push_str(&name.name),
            TypeKind::Index(bound) => {
                self.out.push_str("index[");
                self.expr(bound, 0);
                self.out.push(']');
            }
            TypeKind::Shaped { head, shape, elem } => {
                self.out.push_str(match head {
                    ShapedHead::Tensor => "tensor",
                    ShapedHead::View => "view",
                    ShapedHead::Tile => "tile",
                });
                self.shape_and_elem(shape, elem);
            }
            TypeKind::Tuple(items) => {
                self.out.push('(');
                self.list(items, |p, item| p.ty(item));
                self.out.push(')');
            }
            TypeKind::Void => self.out.push_str("void"),
            TypeKind::Native { target, name, args } => {
                let _ = write!(self.out, "{}.{}", target.name, name.name);
                if !args.is_empty() {
                    self.out.push('(');
                    self.list(args, |p, arg| p.expr(arg, 0));
                    self.out.push(')');
                }
            }
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

    /// A trailing value; a region value ends its own line.
    fn value(&mut self, value: &Expr) {
        self.expr(value, 0);
        if !is_region(value) {
            self.out.push('\n');
        }
    }

    fn values(&mut self, keyword: &str, values: &[Expr]) {
        self.out.push_str(keyword);
        if !values.is_empty() {
            self.out.push(' ');
        }
        self.list(values, |p, v| p.expr(v, 0));
        if !values.last().is_some_and(is_region) {
            self.out.push('\n');
        }
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
            StmtKind::Region(region) => self.region(region),
            StmtKind::Stage { name, ports, body } => {
                let _ = write!(self.out, "stage {}", name.name);
                if !ports.is_empty() {
                    self.out.push('(');
                    self.names(ports);
                    self.out.push(')');
                }
                self.suite(body);
            }
            StmtKind::For {
                targets,
                iter,
                body,
            } => {
                self.out.push_str("for ");
                self.names(targets);
                self.out.push_str(" in ");
                self.expr(iter, 0);
                self.suite(body);
            }
            StmtKind::If { cond, then, els } => self.if_stmt(cond, then, els.as_ref()),
            StmtKind::Publish { value, destination } => {
                self.out.push_str("publish ");
                self.expr(value, 0);
                self.out.push_str(" to ");
                self.value(destination);
            }
            StmtKind::Yield(values) => self.values("yield", values),
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

    /// Header, body and `merge` clause at the current level; ends with a newline.
    fn region(&mut self, region: &Region) {
        self.out.push_str(match region.mode {
            RegionMode::Parallel => "parallel [",
            RegionMode::Ordered => "ordered [",
            RegionMode::Pipeline => "pipeline [",
        });
        self.names(&region.binders);
        self.out.push_str("] in ");
        match region.sources.as_slice() {
            [source] => self.expr(source, 0),
            sources => {
                self.out.push('(');
                self.list(sources, |p, s| p.expr(s, 0));
                self.out.push(')');
            }
        }
        self.suite(&region.body);
        if let Some(merge) = &region.merge {
            self.indent();
            self.out.push_str("merge (");
            self.pattern(&merge.left);
            self.out.push_str(", ");
            self.pattern(&merge.right);
            self.out.push_str(") identity ");
            self.expr(&merge.identity, 0);
            self.suite(&merge.body);
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
            ExprKind::Tile { shape, elem } => {
                self.out.push_str("tile");
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
            ExprKind::Region(region) => self.region(region),
        }
    }
}
