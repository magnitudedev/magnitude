//! Canonical pretty-printer. `parse(print(parse(x))) == parse(x)` for every valid file.

use crate::ast::*;
use std::fmt::Write;

pub fn print(file: &File) -> String {
    let mut out = String::new();
    for (i, decl) in file.decls.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        print_decl(&mut out, decl);
    }
    out
}

fn print_decl(out: &mut String, decl: &Decl) {
    match decl {
        Decl::Fn(f) => print_fn(out, "fn", f),
        Decl::Construct(f) => print_fn(out, "construct", f),
        Decl::Lower(l) => {
            let _ = write!(out, "lower {}", l.name.name);
            match &l.body {
                None => out.push_str(": portable\n"),
                Some(b) => {
                    if !l.shape.is_empty() {
                        out.push('[');
                        out.push_str(&l.shape.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", "));
                        out.push(']');
                    }
                    out.push('(');
                    for (i, p) in l.params.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        let _ = write!(out, "{}: ", p.name.name);
                        print_type(out, &p.ty);
                    }
                    out.push_str("):\n");
                    print_block(out, b, 1);
                }
            }
        }
    }
}

fn print_fn(out: &mut String, kw: &str, f: &FnDecl) {
    let _ = write!(out, "{kw} {}", f.name.name);
    if !f.shape.is_empty() {
        out.push('[');
        out.push_str(&f.shape.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", "));
        out.push(']');
    }
    out.push('(');
    for (i, p) in f.params.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{}: ", p.name.name);
        print_type(out, &p.ty);
    }
    out.push_str("):\n");
    print_block(out, &f.body, 1);
}

pub fn print_type(out: &mut String, ty: &TypeExpr) {
    out.push_str(&ty.head.name);
    if !ty.shape.is_empty() {
        out.push('[');
        for (i, e) in ty.shape.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            print_expr(out, e, 0);
        }
        out.push(']');
    }
    if let Some(e) = &ty.elem {
        out.push(' ');
        out.push_str(&e.name);
    }
}

fn indent(out: &mut String, level: usize) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn print_block(out: &mut String, block: &Block, level: usize) {
    for stmt in &block.stmts {
        print_stmt(out, stmt, level);
    }
}

fn print_stmt(out: &mut String, stmt: &Stmt, level: usize) {
    indent(out, level);
    match &stmt.kind {
        StmtKind::For { targets, iter, body } => {
            out.push_str("for ");
            out.push_str(&targets.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "));
            out.push_str(" in ");
            print_expr(out, iter, 0);
            out.push_str(":\n");
            print_block(out, body, level + 1);
        }
        StmtKind::If { cond, then, els } => {
            out.push_str("if ");
            print_expr(out, cond, 0);
            out.push_str(":\n");
            print_block(out, then, level + 1);
            if let Some(e) = els {
                indent(out, level);
                out.push_str("else:\n");
                print_block(out, e, level + 1);
            }
        }
        StmtKind::Assign { target, op, value } => {
            print_expr(out, target, 0);
            let _ = write!(out, " {} ", op.text());
            print_expr(out, value, 0);
            out.push('\n');
        }
        StmtKind::Expr(e) => {
            print_expr(out, e, 0);
            out.push('\n');
        }
    }
}

pub fn expr_to_string(e: &Expr) -> String {
    let mut s = String::new();
    print_expr(&mut s, e, 0);
    s
}

/// `min_bp` is the binding power of the enclosing operator; parenthesize when ours is not higher.
fn print_expr(out: &mut String, e: &Expr, min_bp: u8) {
    match &e.kind {
        ExprKind::Int(v) => {
            let _ = write!(out, "{v}");
        }
        ExprKind::Float(v) => {
            if v.fract() == 0.0 && v.is_finite() && v.abs() < 1e16 {
                let _ = write!(out, "{v:.1}");
            } else {
                let _ = write!(out, "{v}");
            }
        }
        ExprKind::Inf => out.push_str("inf"),
        ExprKind::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        ExprKind::Name(n) => out.push_str(&n.name),
        ExprKind::Tuple(items) => {
            out.push('(');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                print_expr(out, item, 0);
            }
            if items.len() == 1 {
                out.push(',');
            }
            out.push(')');
        }
        ExprKind::Tile { shape, dtype } => {
            out.push_str("tile[");
            for (i, s) in shape.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                print_expr(out, s, 0);
            }
            let _ = write!(out, "] {}", dtype.name);
        }
        ExprKind::Call { callee, bindings, args } => {
            print_expr(out, callee, UNARY_PRECEDENCE);
            if !bindings.is_empty() {
                out.push('[');
                for (i, (p, v)) in bindings.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{} = ", p.name);
                    print_expr(out, v, 0);
                }
                out.push(']');
            }
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                if let Some(n) = &a.name {
                    let _ = write!(out, "{}=", n.name);
                }
                print_expr(out, &a.value, 0);
            }
            out.push(')');
        }
        ExprKind::Index { base, indices } => {
            print_expr(out, base, UNARY_PRECEDENCE);
            out.push('[');
            for (i, idx) in indices.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                match idx {
                    Index::Expr(e) => print_expr(out, e, 0),
                    Index::Slice { start, end } => {
                        if let Some(s) = start {
                            print_expr(out, s, 0);
                        }
                        out.push(':');
                        if let Some(e) = end {
                            print_expr(out, e, 0);
                        }
                    }
                }
            }
            out.push(']');
        }
        ExprKind::Attr { base, name } => {
            print_expr(out, base, UNARY_PRECEDENCE);
            let _ = write!(out, ".{}", name.name);
        }
        ExprKind::Unary { op, expr } => {
            let bp = if *op == UnaryOp::Not { NOT_PRECEDENCE } else { UNARY_PRECEDENCE };
            let paren = bp <= min_bp;
            if paren {
                out.push('(');
            }
            out.push_str(op.text());
            print_expr(out, expr, bp - 1);
            if paren {
                out.push(')');
            }
        }
        ExprKind::Binary { op, lhs, rhs } => {
            let bp = op.precedence();
            let paren = bp <= min_bp;
            if paren {
                out.push('(');
            }
            print_expr(out, lhs, bp - 1);
            let _ = write!(out, " {} ", op.text());
            print_expr(out, rhs, bp);
            if paren {
                out.push(')');
            }
        }
        ExprKind::Lambda { params, body } => {
            let paren = min_bp > 0;
            if paren {
                out.push('(');
            }
            out.push('(');
            out.push_str(&params.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", "));
            out.push_str(") -> ");
            print_expr(out, body, 0);
            if paren {
                out.push(')');
            }
        }
    }
}
