//! Portable-algorithm work derived from checked IR. These are semantic operations,
//! not issued instructions or necessary lower-bound obligations. A stream is taken
//! as one whole-axis piece, matching reference semantics; splitting is realization work.

use crate::quantity::Count;
use seismic_lang::ast::{AssignOp, BinaryOp};
use seismic_lang::ir::*;
use seismic_lang::program::Program;
use seismic_lang::sym::{Atom, Sym};
use seismic_lang::types::{DType, Elem, Ty};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkKind {
    Binary {
        operator: String,
        dtype: DType,
    },
    Unary {
        operator: String,
        dtype: DType,
    },
    Builtin {
        operation: Builtin,
        dtype: DType,
    },
    Conversion {
        from: DType,
        to: DType,
    },
    /// One reduction per output element, with the input extent retained. Backend
    /// rules decide its comparisons/additions/communication under precision rules.
    Reduction {
        contract: seismic_lang::reduction::Contract,
        extent: Count,
    },
    Decode {
        representation: String,
    },
}

#[derive(Clone, Debug)]
pub struct WorkTerm {
    pub kind: WorkKind,
    pub count: Count,
    pub function: String,
    pub byte_offset: u32,
    pub conditions: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct WorkAccount {
    pub terms: Vec<WorkTerm>,
    pub unavailable: Vec<String>,
}

impl WorkAccount {
    pub fn is_exact(&self) -> bool {
        self.unavailable.is_empty()
            && self.terms.iter().all(|t| {
                matches!(t.count, Count::Exact(_))
                    && t.conditions.is_empty()
                    && !matches!(
                        t.kind,
                        WorkKind::Reduction {
                            extent: Count::Unknown { .. } | Count::Interval(_),
                            ..
                        }
                    )
            })
    }
}

#[derive(Clone, Default)]
struct Bindings {
    shapes: HashMap<String, i64>,
    elements: HashMap<String, Elem>,
}

pub fn derive(
    program: &Program,
    function: &str,
    shapes: &HashMap<String, i64>,
) -> Result<WorkAccount, String> {
    derive_specialized(program, function, shapes, &HashMap::new())
}

pub fn derive_specialized(
    program: &Program,
    function: &str,
    shapes: &HashMap<String, i64>,
    elements: &HashMap<String, Elem>,
) -> Result<WorkAccount, String> {
    let f = program
        .functions
        .iter()
        .find(|f| f.name == function)
        .ok_or_else(|| format!("no function `{function}`"))?;
    seismic_lang::program::validate_element_bindings(f, elements)?;
    for name in &f.shape_params {
        let value = shapes
            .get(name)
            .ok_or_else(|| format!("unbound shape `{name}`"))?;
        if *value < 0 {
            return Err(format!("shape `{name}` must be nonnegative"));
        }
    }
    let mut walker = Walker {
        program,
        account: WorkAccount::default(),
        stack: Vec::new(),
        conditions: Vec::new(),
    };
    walker.function(
        f,
        &Bindings {
            shapes: shapes.clone(),
            elements: elements.clone(),
        },
        &Count::Exact(1),
    );
    Ok(walker.account)
}

struct Walker<'a> {
    program: &'a Program,
    account: WorkAccount,
    stack: Vec<String>,
    conditions: Vec<String>,
}

fn count(s: &Sym, b: &Bindings) -> Count {
    match s.eval(&|p| b.shapes.get(p).copied()) {
        Some(n) if n >= 0 => Count::Exact(n as u64),
        _ => Count::unknown(format!("unresolved or invalid extent `{s}`")),
    }
}

fn product(a: &Count, b: &Count) -> Count {
    match (a, b) {
        (Count::Exact(n), _) => b.scale(*n),
        (_, Count::Exact(n)) => a.scale(*n),
        _ => Count::unknown(format!("unresolved multiplicity: {a:?} * {b:?}")),
    }
}

fn elements(ty: &Ty, b: &Bindings) -> Count {
    ty.shaped()
        .map(|t| {
            t.shape
                .iter()
                .fold(Count::Exact(1), |a, s| product(&a, &count(s, b)))
        })
        .unwrap_or(Count::Exact(1))
}

fn element(elem: &Elem, b: &Bindings) -> Elem {
    if let Elem::Param(name) = elem {
        b.elements
            .get(name)
            .cloned()
            .unwrap_or_else(|| elem.clone())
    } else {
        elem.clone()
    }
}

fn dtype(ty: &Ty, b: &Bindings) -> Option<DType> {
    match ty {
        Ty::Scalar(d) => Some(*d),
        _ => ty.shaped().and_then(|s| element(&s.elem, b).read_dtype()),
    }
}

impl Walker<'_> {
    fn conversion(
        &mut self,
        from: Option<DType>,
        to: Option<DType>,
        n: Count,
        f: &Function,
        e: &Expr,
    ) {
        match (from, to) {
            (Some(from), Some(to)) if from != to => {
                self.term(WorkKind::Conversion { from, to }, n, f, e)
            }
            (Some(_), Some(_)) => {}
            _ => self.account.unavailable.push(format!(
                "unresolved conversion in {}:{}",
                f.name, e.span.start
            )),
        }
    }
    fn function(&mut self, f: &Function, b: &Bindings, multiplicity: &Count) {
        if self.stack.contains(&f.name) {
            self.account
                .unavailable
                .push(format!("recursive work derivation at {}", f.name));
            return;
        }
        self.stack.push(f.name.clone());
        self.block(&f.body, f, b, multiplicity);
        self.stack.pop();
    }

    fn term(&mut self, kind: WorkKind, n: Count, f: &Function, e: &Expr) {
        if n == Count::Exact(0) {
            return;
        }
        self.account.terms.push(WorkTerm {
            kind,
            count: n,
            function: f.name.clone(),
            byte_offset: e.span.start,
            conditions: self.conditions.clone(),
        });
    }

    fn block(&mut self, stmts: &[Stmt], f: &Function, b: &Bindings, mult: &Count) {
        if *mult == Count::Exact(0) {
            return;
        }
        for s in stmts {
            match &s.kind {
                StmtKind::Parallel { extents, body, .. } => {
                    let n = extents
                        .iter()
                        .fold(mult.clone(), |a, d| product(&a, &count(d, b)));
                    self.block(body, f, b, &n);
                }
                StmtKind::Owned { tile, body, .. } => {
                    self.block(body, f, b, &product(mult, &elements(&tile.ty, b)))
                }
                StmtKind::Range { lo, hi, body, .. } => {
                    let n = match (
                        lo.eval(&|p| b.shapes.get(p).copied()),
                        hi.eval(&|p| b.shapes.get(p).copied()),
                    ) {
                        (Some(lo), Some(hi)) => match hi.checked_sub(lo) {
                            Some(n) => Count::Exact(n.max(0) as u64),
                            None => Count::unknown("range length overflow"),
                        },
                        _ => Count::unknown(format!("dependent range `{lo}..{hi}`")),
                    };
                    self.block(body, f, b, &product(mult, &n));
                }
                StmtKind::LoadLoop {
                    views,
                    axis,
                    piece,
                    body,
                    ..
                } => {
                    let mut inner = b.clone();
                    let mut extent = None;
                    for view in views {
                        self.expr(view, f, b, mult);
                        let n = view
                            .ty
                            .shaped()
                            .and_then(|t| t.shape.get(*axis))
                            .and_then(|s| s.eval(&|p| b.shapes.get(p).copied()));
                        match (extent, n) {
                            (Some(a), Some(v)) if a != v => self
                                .account
                                .unavailable
                                .push(format!("inconsistent stream extents in {}", f.name)),
                            (_, Some(v)) => extent = Some(v),
                            _ => {
                                self.account.unavailable.push(format!(
                                    "runtime stream extent in {} at {}",
                                    f.name, s.span.start
                                ));
                            }
                        }
                    }
                    if let (Atom::Param(name), Some(n)) = (piece, extent) {
                        inner.shapes.insert(name.clone(), n);
                    }
                    if extent != Some(0) {
                        if extent.is_none() {
                            self.conditions.push(format!(
                                "{}:{} stream axis is nonempty",
                                f.name, s.span.start
                            ));
                        }
                        self.block(body, f, &inner, mult);
                        if extent.is_none() {
                            self.conditions.pop();
                        }
                    }
                }
                StmtKind::If { cond, then, els } => {
                    self.expr(cond, f, b, mult);
                    if let ExprKind::Bool(value) = cond.kind {
                        self.block(if value { then } else { els }, f, b, mult);
                    } else {
                        for (label, body) in [("true", then), ("false", els)] {
                            self.conditions
                                .push(format!("{}:{}={label}", f.name, cond.span.start));
                            self.block(body, f, b, mult);
                            self.conditions.pop();
                        }
                    }
                }
                StmtKind::Assign { target, op, value } => {
                    self.expr(value, f, b, mult);
                    let publication_dtype = match &target.kind {
                        ExprKind::Index { base, .. } if matches!(target.ty, Ty::Scalar(_)) => {
                            dtype(&base.ty, b)
                        }
                        _ => dtype(&target.ty, b),
                    };
                    if matches!(target.ty, Ty::Scalar(_) | Ty::Tile(_)) {
                        self.conversion(
                            dtype(&value.ty, b),
                            publication_dtype,
                            product(mult, &elements(&target.ty, b)),
                            f,
                            target,
                        );
                    }
                    // Address expressions are separate from loading the target value.
                    if let ExprKind::Index { indices, .. } = &target.kind {
                        self.indices(indices, f, b, mult);
                    }
                    if *op != AssignOp::Assign {
                        self.conversion(
                            publication_dtype,
                            dtype(&target.ty, b),
                            mult.clone(),
                            f,
                            target,
                        );
                        if let Some(d) = dtype(&target.ty, b) {
                            let operator = match op {
                                AssignOp::Add => "+",
                                AssignOp::Sub => "-",
                                AssignOp::Mul => "*",
                                _ => unreachable!(),
                            };
                            self.term(
                                WorkKind::Binary {
                                    operator: operator.into(),
                                    dtype: d,
                                },
                                product(mult, &elements(&target.ty, b)),
                                f,
                                target,
                            );
                        } else {
                            self.account
                                .unavailable
                                .push(format!("unresolved assignment type in {}", f.name));
                        }
                    }
                }
                StmtKind::Expr(e) => self.expr(e, f, b, mult),
                StmtKind::Lanes { .. } => self
                    .account
                    .unavailable
                    .push("lane execution belongs to realization accounting".into()),
            }
        }
    }

    fn indices(&mut self, indices: &[Index], f: &Function, b: &Bindings, mult: &Count) {
        for i in indices {
            match i {
                Index::Point(e) => self.expr(e, f, b, mult),
                Index::Slice { start, end } => {
                    if let Some(e) = start {
                        self.expr(e, f, b, mult);
                    }
                    if let Some(e) = end {
                        self.expr(e, f, b, mult);
                    }
                }
            }
        }
    }

    fn expr(&mut self, e: &Expr, f: &Function, b: &Bindings, mult: &Count) {
        match &e.kind {
            ExprKind::Call {
                callee,
                shape_args,
                elem_args,
                args,
            } => {
                for a in args {
                    self.expr(a, f, b, mult);
                }
                if let Some(g) = self.program.functions.iter().find(|g| g.name == *callee) {
                    let mut inner = Bindings::default();
                    for (p, s) in g.shape_params.iter().zip(shape_args) {
                        if let Some(n) = s.eval(&|p| b.shapes.get(p).copied()) {
                            inner.shapes.insert(p.clone(), n);
                        }
                    }
                    for (p, e) in g.elem_params.iter().zip(elem_args) {
                        inner.elements.insert(p.clone(), element(e, b));
                    }
                    self.function(g, &inner, mult);
                } else {
                    self.account
                        .unavailable
                        .push(format!("missing function `{callee}`"));
                }
            }
            ExprKind::Builtin { name, args } => {
                for a in args {
                    self.expr(a, f, b, mult);
                }
                match name {
                    Builtin::Reshape | Builtin::Load | Builtin::Extent => {}
                    Builtin::Store => self.conversion(
                        dtype(&args[0].ty, b),
                        dtype(&args[1].ty, b),
                        product(mult, &elements(&args[1].ty, b)),
                        f,
                        e,
                    ),
                    Builtin::Atomic => self.account.unavailable.push(format!(
                        "atomic operation accounting at {}:{}",
                        f.name, e.span.start
                    )),
                    Builtin::Reduce => {
                        let axis = args[1]
                            .sym
                            .as_ref()
                            .and_then(Sym::as_constant)
                            .and_then(|v| usize::try_from(v).ok());
                        let op = if let ExprKind::Int(op) = args[2].kind {
                            ReduceOp::from_tag(op)
                        } else {
                            None
                        };
                        if let (Some(axis), Some(op), Some(sh), Some(d)) =
                            (axis, op, args[0].ty.shaped(), dtype(&args[0].ty, b))
                        {
                            if let Some(extent) = sh.shape.get(axis) {
                                self.term(
                                    WorkKind::Reduction {
                                        contract: seismic_lang::reduction::Contract::new(
                                            op,
                                            d,
                                            matches!(
                                                args.get(3).map(|e| &e.kind),
                                                Some(ExprKind::Bool(true))
                                            ),
                                        ),
                                        extent: count(extent, b),
                                    },
                                    product(mult, &elements(&e.ty, b)),
                                    f,
                                    e,
                                );
                            }
                        } else {
                            self.account
                                .unavailable
                                .push("unresolved reduction contract".into());
                        }
                    }
                    _ => {
                        if let Some(d) = dtype(&e.ty, b) {
                            for a in args {
                                self.conversion(
                                    dtype(&a.ty, b),
                                    Some(d),
                                    product(mult, &elements(&a.ty, b)),
                                    f,
                                    a,
                                );
                            }
                            self.term(
                                WorkKind::Builtin {
                                    operation: *name,
                                    dtype: d,
                                },
                                product(mult, &elements(&e.ty, b)),
                                f,
                                e,
                            );
                        } else {
                            self.account
                                .unavailable
                                .push("unresolved arithmetic type".into());
                        }
                    }
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                self.expr(lhs, f, b, mult);
                self.expr(rhs, f, b, mult);
                let common = dtype(&lhs.ty, b)
                    .zip(dtype(&rhs.ty, b))
                    .and_then(|(a, b)| DType::promote(a, b));
                if let Some(d) = common {
                    self.conversion(
                        dtype(&lhs.ty, b),
                        Some(d),
                        product(mult, &elements(&lhs.ty, b)),
                        f,
                        lhs,
                    );
                    self.conversion(
                        dtype(&rhs.ty, b),
                        Some(d),
                        product(mult, &elements(&rhs.ty, b)),
                        f,
                        rhs,
                    );
                    self.term(
                        WorkKind::Binary {
                            operator: op.text().into(),
                            dtype: d,
                        },
                        product(mult, &elements(&e.ty, b)),
                        f,
                        e,
                    );
                } else {
                    self.account
                        .unavailable
                        .push("unresolved binary operand type".into());
                }
                if matches!(op, BinaryOp::And | BinaryOp::Or) {
                    self.account
                        .unavailable
                        .push("short-circuit operand evaluation needs path accounting".into());
                }
            }
            ExprKind::Unary { op, expr } => {
                self.expr(expr, f, b, mult);
                if let Some(d) = dtype(&expr.ty, b) {
                    self.term(
                        WorkKind::Unary {
                            operator: format!("{op:?}"),
                            dtype: d,
                        },
                        product(mult, &elements(&e.ty, b)),
                        f,
                        e,
                    );
                }
            }
            ExprKind::Cast { dtype: to, expr } => {
                self.expr(expr, f, b, mult);
                if let Some(from) = dtype(&expr.ty, b) {
                    if from != *to {
                        self.term(
                            WorkKind::Conversion { from, to: *to },
                            product(mult, &elements(&e.ty, b)),
                            f,
                            e,
                        );
                    }
                } else {
                    self.account
                        .unavailable
                        .push("unresolved conversion operand type".into());
                }
            }
            ExprKind::Index { base, indices } => {
                self.expr(base, f, b, mult);
                self.indices(indices, f, b, mult);
                if matches!(e.ty, Ty::Scalar(_)) {
                    self.conversion(dtype(&base.ty, b), dtype(&e.ty, b), mult.clone(), f, e);
                    if let Some(sh) = base.ty.shaped() {
                        if let Elem::Repr(representation) = element(&sh.elem, b) {
                            self.term(WorkKind::Decode { representation }, mult.clone(), f, e);
                        }
                    }
                }
            }
            ExprKind::Load { view: base, .. }
            | ExprKind::Transpose(base)
            | ExprKind::Accessor { base, .. } => self.expr(base, f, b, mult),
            ExprKind::Tuple(items) => {
                for item in items {
                    self.expr(item, f, b, mult);
                }
            }
            ExprKind::Intrinsic { .. } | ExprKind::Lanes { .. } => self
                .account
                .unavailable
                .push("backend intrinsic needs realization accounting".into()),
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Bool(_)
            | ExprKind::ShapeParam(_)
            | ExprKind::Var(_)
            | ExprKind::TileAlloc { .. } => {}
        }
    }
}
