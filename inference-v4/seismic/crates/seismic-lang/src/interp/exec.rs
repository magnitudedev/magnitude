//! Frames, definition selection, calls, statements, regions, stages and merges.
use super::value::{Backing, Flow, Piece, ResultVal, Shaped, Value};
use super::{round_to, Arg, Interpreter};
use crate::family::Workload;
use crate::sir::{
    Block, Body, CallId, ContractFamily, DefId, DefKind, Definition, Expr, ExprKind, Merge,
    Pattern, Predicate, Region, RegionSource, SliceParent, Stage, Stmt, StmtKind, VarKind,
};
use crate::span::{line_col, Span};
use crate::sym::Sym;
use crate::syntax::ast::AssignOp;
use crate::sir::Mode;
use crate::types::{DType, Elem};
use crate::types::{Extent, SliceId, Ty};
use std::collections::HashMap;
use std::rc::Rc;

pub(super) struct Frame<'a> {
    pub def: &'a Definition,
    pub body: &'a Body,
    pub vars: Vec<Option<Value>>,
    /// Current piece of every slice binder in scope, by `SliceId`.
    pub slices: Vec<Option<Piece>>,
    /// Shape parameters; one bound to a structural extent holds the current valid extent.
    pub shapes: HashMap<String, i64>,
    /// Shape parameters bound to a structural extent: the caller's piece (capacity, domain).
    pub caps: HashMap<String, Piece>,
    pub elems: HashMap<String, Elem>,
    /// Realized lengths of runtime-bounded range views, by the checker's `@dyn#n` atom.
    pub dynamic: HashMap<String, i64>,
}

impl<'a> Frame<'a> {
    /// Symbols name shape parameters, or the runtime value of an integer body variable as
    /// `name#VarId` (the checker's variable atoms); a bare name denotes the innermost live
    /// integer variable of that name.
    fn lookup(&self, name: &str) -> Option<i64> {
        if let Some(v) = self.shapes.get(name).or_else(|| self.dynamic.get(name)) {
            return Some(*v);
        }
        let integer = |i: usize| match self.vars.get(i) {
            Some(Some(Value::Scalar(d, x))) if d.is_int() => Some(*x as i64),
            _ => None,
        };
        if let Some((base, id)) = name.rsplit_once('#') {
            if let Some(id) = id
                .parse::<usize>()
                .ok()
                .filter(|id| self.body.vars.get(*id).is_some_and(|v| v.name == base))
            {
                return integer(id);
            }
        }
        self.body
            .vars
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, var)| var.name == name)
            .find_map(|(i, _)| integer(i))
    }

    pub fn sym(&self, s: &Sym) -> Result<i64, String> {
        s.eval(&|n| self.lookup(n))
            .ok_or_else(|| format!("cannot evaluate `{s}` here"))
    }

    pub fn elem(&self, e: &Elem) -> Elem {
        match e {
            Elem::Param(p) => self.elems.get(p).cloned().unwrap_or_else(|| e.clone()),
            other => other.clone(),
        }
    }

    /// A predicate on a structurally bound parameter constrains the capacity (R1).
    fn holds(&self, p: &Predicate) -> Result<bool, String> {
        let env = |n: &str| {
            self.caps
                .get(n)
                .map(|piece| piece.width)
                .or_else(|| self.lookup(n))
        };
        let eval = |s: &Sym| {
            s.eval(&env)
                .ok_or_else(|| format!("cannot evaluate predicate operand `{s}`"))
        };
        Ok(match p {
            Predicate::NonNegative(s) => eval(s)? >= 0,
            Predicate::Zero(s) => eval(s)? == 0,
            Predicate::NonZero(s) => eval(s)? != 0,
            Predicate::Full(name) => self
                .caps
                .get(name)
                .is_none_or(|piece| piece.width > 0 && piece.domain % piece.width == 0),
        })
    }
}

fn same_backing(a: &Shaped, b: &Shaped) -> bool {
    match (&a.backing, &b.backing) {
        (Backing::Owned(x), Backing::Owned(y)) => Rc::ptr_eq(x, y),
        (Backing::Tensor(x), Backing::Tensor(y)) => x == y,
        _ => false,
    }
}

impl<'a> Interpreter<'a> {
    // ----- entry and selection ---------------------------------------------------------

    pub(super) fn run_entry(
        &mut self,
        name: &str,
        args: &[Arg],
        workload: &Workload,
    ) -> Result<(), String> {
        let program = self.program;
        let family = program.resolve_family(name)?;
        let families = [family];
        let shapes: HashMap<String, i64> = workload
            .shapes
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let elems: HashMap<String, Elem> = workload
            .elems
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let make = |d: &'a Definition| -> Result<Frame<'a>, String> {
            if let Some(missing) = d.shape_params.iter().find(|p| !shapes.contains_key(*p)) {
                return Err(format!("workload does not bind shape parameter {missing}"));
            }
            let values = self.entry_values(d, args, &shapes)?;
            self.instantiate(d, shapes.clone(), HashMap::new(), elems.clone(), values)
        };
        let mut frame = self.select(name, &families, &make)?;
        let body = frame.body;
        self.block(&body.block, &mut frame)?;
        Ok(())
    }

    fn entry_values(
        &self,
        d: &Definition,
        args: &[Arg],
        shapes: &HashMap<String, i64>,
    ) -> Result<Vec<Value>, String> {
        if args.len() != d.params.len() {
            return Err(format!(
                "`{}` takes {} arguments, {} given",
                d.name,
                d.params.len(),
                args.len()
            ));
        }
        let mut values = Vec::with_capacity(args.len());
        for (i, (arg, param)) in args.iter().zip(&d.params).enumerate() {
            let scalar = |dtype: DType, bound: Option<u64>, v: f64| -> Result<Value, String> {
                let mut parameter = crate::abi::ScalarParameter::plain(&param.name, dtype);
                parameter.index_bound = bound;
                crate::abi::ScalarLayout::words(&[parameter])?.encode(&[v])?;
                Ok(Value::Scalar(
                    dtype,
                    if dtype.is_float() {
                        round_to(dtype, v)
                    } else {
                        v
                    },
                ))
            };
            values.push(match (arg, &param.ty) {
                (Arg::Tensor(id), Ty::Tensor(_) | Ty::View(_)) => {
                    let t = self.tensors.get(*id).ok_or_else(|| {
                        format!("argument {i} names tensor {id}, which does not exist")
                    })?;
                    Value::View(Shaped::tensor(*id, t.shape()))
                }
                (Arg::Scalar(v), Ty::Scalar(dtype)) => scalar(*dtype, None, *v)?,
                (Arg::Scalar(v), Ty::Index(bound)) => {
                    let bound = bound
                        .eval(&|n| shapes.get(n).copied())
                        .and_then(|n| u64::try_from(n).ok())
                        .ok_or_else(|| format!("unresolved index bound for `{}`", param.name))?;
                    scalar(DType::I32, Some(bound), *v)?
                }
                (Arg::Range(start, end), Ty::Range(bound)) => {
                    let bound = bound.eval(&|n| shapes.get(n).copied()).and_then(|n| u64::try_from(n).ok())
                        .ok_or_else(|| format!("unresolved range bound for `{}`", param.name))?;
                    let mut first = crate::abi::ScalarParameter::plain(format!("{}_start", param.name), DType::I32);
                    first.range = Some(crate::abi::RangeScalar { parameter: param.name.clone(), endpoint: crate::abi::RangeEndpoint::Start, bound });
                    let mut last = crate::abi::ScalarParameter::plain(format!("{}_end", param.name), DType::I32);
                    last.range = Some(crate::abi::RangeScalar { parameter: param.name.clone(), endpoint: crate::abi::RangeEndpoint::End, bound });
                    crate::abi::ScalarLayout::words(&[first, last])?.encode(&[*start as f64, *end as f64])?;
                    Value::Range(*start, *end)
                }
                _ => {
                    return Err(format!(
                        "argument {i} does not match parameter type {}",
                        param.ty
                    ))
                }
            });
        }
        Ok(values)
    }

    /// Bind one definition to evaluated arguments (in parameter order). `Err` is the reason
    /// the definition is not applicable.
    fn instantiate(
        &self,
        def: &'a Definition,
        mut shapes: HashMap<String, i64>,
        caps: HashMap<String, Piece>,
        mut elems: HashMap<String, Elem>,
        values: Vec<Value>,
    ) -> Result<Frame<'a>, String> {
        let body = &def.body;
        if values.len() != def.params.len() {
            return Err(format!(
                "takes {} arguments, {} given",
                def.params.len(),
                values.len()
            ));
        }
        for (name, elem) in &def.elem_bindings {
            elems.entry(name.clone()).or_insert_with(|| elem.clone());
        }
        let mut slices = vec![None; body.slices.len()];
        for (param, value) in def.params.iter().zip(&values) {
            match (&param.ty, value) {
                (Ty::Tensor(t) | Ty::View(t) | Ty::Tile(t), Value::Tile(s) | Value::View(s)) => {
                    if t.axes.len() != s.shape.len() {
                        return Err(format!(
                            "`{}` has rank {}, argument has rank {}",
                            param.name,
                            t.axes.len(),
                            s.shape.len()
                        ));
                    }
                    let actual = self.elem_of(s);
                    match &t.elem {
                        Elem::Param(p) => {
                            if elems
                                .insert(p.clone(), actual.clone())
                                .is_some_and(|previous| previous != actual)
                            {
                                return Err(format!("inconsistent element parameter {p}"));
                            }
                        }
                        declared if declared != &actual => {
                            return Err(format!(
                                "`{}` requires element {declared}, argument has {actual}",
                                param.name
                            ))
                        }
                        _ => {}
                    }
                    for (axis, n) in t.axes.iter().zip(&s.shape) {
                        if let Extent::Semantic(sym) = axis {
                            if let Some(p) = def
                                .shape_params
                                .iter()
                                .find(|p| !shapes.contains_key(*p) && sym == &Sym::param(p))
                            {
                                shapes.insert(p.clone(), *n as i64);
                            }
                        }
                    }
                }
                (Ty::Scalar(_) | Ty::Index(_), Value::Scalar(..))
                | (Ty::Range(_), Value::Range(..))
                | (Ty::Tuple(_), Value::Tuple(_))
                | (Ty::Result(_), Value::Result(_))
                | (Ty::Native(_), Value::Native(_)) => {}
                (Ty::Slice(id), Value::Slice(piece)) => match slices.get_mut(id.0 as usize) {
                    Some(slot) => *slot = Some(*piece),
                    None => return Err(format!("`{}` names an unknown slice", param.name)),
                },
                (_, Value::Void) if param.mode == Mode::Out => {}
                (ty, v) => {
                    return Err(format!(
                        "`{}`: {} where {ty} is required",
                        param.name,
                        v.kind()
                    ))
                }
            }
        }
        let mut vars = vec![None; body.vars.len()];
        for (param, value) in def.params.iter().zip(values) {
            if !matches!(value, Value::Void) {
                *vars
                    .get_mut(param.var)
                    .ok_or("parameter variable outside the body")? = Some(value);
            }
        }
        let frame = Frame {
            def,
            body,
            vars,
            slices,
            shapes,
            caps,
            elems,
            dynamic: HashMap::new(),
        };
        for param in &def.params {
            let Some(value) = &frame.vars[param.var] else {
                continue;
            };
            match (&param.ty, value) {
                (Ty::Tensor(t) | Ty::View(t) | Ty::Tile(t), Value::Tile(s) | Value::View(s)) => {
                    for (k, (axis, n)) in t.axes.iter().zip(&s.shape).enumerate() {
                        if let Extent::Semantic(sym) = axis {
                            let expected = frame.sym(sym)?;
                            if expected != *n as i64 {
                                return Err(format!(
                                    "`{}` axis {k} has extent {n}, expected {sym} = {expected}",
                                    param.name
                                ));
                            }
                        }
                    }
                }
                (Ty::Index(bound), Value::Scalar(_, x)) => {
                    let bound = frame.sym(bound)?;
                    if *x < 0.0 || *x as i64 >= bound {
                        return Err(format!("`{}` = {x} is outside index[{bound}]", param.name));
                    }
                }
                _ => {}
            }
        }
        for p in &def.predicates {
            if !frame.holds(p)? {
                return Err(format!("predicate {p:?} does not hold"));
            }
        }
        Ok(frame)
    }

    /// Deterministic definition choice from the union of portable functions and matching
    /// target-specific functions and lowerings. `choice` may force any applicable body.
    fn select(
        &self,
        name: &str,
        families: &[&'a ContractFamily],
        make: &dyn Fn(&'a Definition) -> Result<Frame<'a>, String>,
    ) -> Result<Frame<'a>, String> {
        let program = self.program;
        let target = self.target.as_deref();
        let mut ids: Vec<DefId> = families
            .iter()
            .flat_map(|f| f.bodies.iter().chain(&f.lowerings))
            .copied()
            .collect();
        ids.sort();
        ids.dedup();
        let mut applicable: Vec<(DefId, Frame<'a>)> = Vec::new();
        let mut reasons = Vec::new();
        for id in ids {
            let d = program.definition(id);
            let available = match &d.kind {
                DefKind::Body { target: None } => true,
                DefKind::Body {
                    target: Some(required),
                }
                | DefKind::Lower { target: required } => Some(required.as_str()) == target,
            };
            if !available {
                continue;
            }
            match make(d) {
                Ok(frame) => applicable.push((id, frame)),
                Err(reason) => reasons.push(format!("definition {}: {reason}", id.0)),
            }
        }
        let forced = match &self.choice {
            Some(choose) => choose(
                name,
                &applicable.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            ),
            None => None,
        };
        let position = match forced {
            Some(id) => applicable
                .iter()
                .position(|(candidate, _)| *candidate == id)
                .ok_or_else(|| {
                    format!("forced definition {} of `{name}` is not applicable", id.0)
                })?,
            None => (!applicable.is_empty()).then_some(0).ok_or_else(|| {
                format!(
                    "no applicable definition of `{name}`{}{}",
                    if reasons.is_empty() { "" } else { ": " },
                    reasons.join("; ")
                )
            })?,
        };
        Ok(applicable.swap_remove(position).1)
    }

    pub(super) fn call(
        &mut self,
        call: CallId,
        args: &'a [Expr],
        f: &mut Frame<'a>,
    ) -> Result<Value, String> {
        let program = self.program;
        let body = f.body;
        let site = body.calls.get(call.0 as usize).ok_or("unknown call site")?;
        let family = program
            .families
            .get(site.family)
            .ok_or("call names an unknown family")?;
        let mut outs = vec![false; args.len()];
        if let Some(b) = site.bindings.first() {
            for (param, ordinal) in program
                .definition(b.definition)
                .params
                .iter()
                .zip(&b.arg_order)
            {
                if let Some(slot) = outs.get_mut(*ordinal) {
                    *slot = param.mode == Mode::Out;
                }
            }
        }
        let mut values = Vec::with_capacity(args.len());
        for (a, out) in args.iter().zip(&outs) {
            let unset = matches!(&a.kind, ExprKind::Var(id) if f.vars.get(*id).is_some_and(|v| v.is_none()));
            values.push(if *out && unset {
                Value::Void
            } else {
                self.expr(a, f)?
            });
        }
        let mut inner = {
            let caller: &Frame<'a> = f;
            let this: &Interpreter<'a> = self;
            let make = |d: &'a Definition| -> Result<Frame<'a>, String> {
                let b = site
                    .bindings
                    .iter()
                    .find(|b| b.definition == d.id)
                    .ok_or("the arguments do not bind its parameters")?;
                let mut shapes = HashMap::new();
                let mut caps = HashMap::new();
                for (name, extent) in &b.shape_args {
                    match extent {
                        Extent::Semantic(sym) => {
                            // An extent the caller cannot evaluate is solved from the runtime
                            // shapes of the arguments and checked for consistency there.
                            if let Ok(n) = caller.sym(sym) {
                                shapes.insert(name.clone(), n);
                            }
                            if let Some((_, piece)) =
                                caller.caps.iter().find(|(p, _)| sym == &Sym::param(p))
                            {
                                caps.insert(name.clone(), *piece);
                            }
                        }
                        Extent::Structural(slice) => {
                            let piece = this.piece(caller, *slice)?;
                            shapes.insert(name.clone(), piece.extent());
                            caps.insert(name.clone(), piece);
                        }
                    }
                }
                let mut elems = HashMap::new();
                for (name, elem) in &b.elem_args {
                    let resolved = caller.elem(elem);
                    if !matches!(resolved, Elem::Param(_)) {
                        elems.insert(name.clone(), resolved);
                    }
                }
                let ordered = b
                    .arg_order
                    .iter()
                    .map(|i| {
                        values
                            .get(*i)
                            .cloned()
                            .ok_or("argument order names a missing argument")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                this.instantiate(d, shapes, caps, elems, ordered)
            };
            this.select(&family.name, &[family], &make)?
        };
        let callee = inner.body;
        let result = match self.block(&callee.block, &mut inner)? {
            Flow::Return(v) | Flow::Yield(v) => v,
            Flow::Next => Value::Void,
        };
        let binding = site
            .bindings
            .iter()
            .find(|b| b.definition == inner.def.id)
            .ok_or("selected definition lost its binding")?;
        for (param, ordinal) in inner.def.params.iter().zip(&binding.arg_order) {
            if param.mode == Mode::In {
                continue;
            }
            let arg = args
                .get(*ordinal)
                .ok_or("argument order names a missing argument")?;
            match inner.vars[param.var].take() {
                Some(v @ (Value::Scalar(..) | Value::Tuple(_))) => {
                    self.assign(arg, AssignOp::Assign, v, f)?
                }
                Some(Value::Tile(s)) => {
                    if let ExprKind::Var(id) = &arg.kind {
                        let shared = matches!(&f.vars[*id], Some(Value::Tile(c) | Value::View(c)) if same_backing(c, &s));
                        if !shared {
                            f.vars[*id] = Some(Value::Tile(s));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(result)
    }

    // ----- statements ------------------------------------------------------------------

    fn locate(&self, def: &Definition, span: Span, message: String) -> String {
        if message.starts_with("in `") {
            return message;
        }
        let line = self
            .program
            .files
            .get(def.file)
            .map(|(_, text)| line_col(text, span.start).0)
            .unwrap_or(0);
        format!("in `{}` line {line}: {message}", def.name)
    }

    pub(super) fn block(&mut self, block: &'a Block, f: &mut Frame<'a>) -> Result<Flow, String> {
        for s in block {
            match self.stmt(s, f).map_err(|m| self.locate(f.def, s.span, m))? {
                Flow::Next => {}
                flow => return Ok(flow),
            }
        }
        Ok(Flow::Next)
    }

    fn bind(&mut self, pattern: &Pattern, value: Value, f: &mut Frame<'a>) -> Result<(), String> {
        match (pattern, value) {
            (Pattern::Var(id), value) => {
                *f.vars
                    .get_mut(*id)
                    .ok_or("binding outside the body's variables")? = Some(value);
                Ok(())
            }
            (Pattern::Tuple(patterns), Value::Tuple(items)) if patterns.len() == items.len() => {
                for (p, v) in patterns.iter().zip(items) {
                    self.bind(p, v, f)?;
                }
                Ok(())
            }
            (Pattern::Tuple(patterns), other) => Err(format!(
                "cannot destructure {} into {} names",
                other.kind(),
                patterns.len()
            )),
        }
    }

    fn values(&mut self, items: &'a [Expr], f: &mut Frame<'a>) -> Result<Value, String> {
        let value = match items {
            [] => Value::Void,
            [one] => self.expr(one, f)?,
            many => Value::Tuple(
                many.iter()
                    .map(|e| self.expr(e, f))
                    .collect::<Result<_, _>>()?,
            ),
        };
        self.snapshot(value)
    }

    pub(super) fn assign(
        &mut self,
        target: &'a Expr,
        op: AssignOp,
        value: Value,
        f: &mut Frame<'a>,
    ) -> Result<(), String> {
        match &target.kind {
            ExprKind::Tuple(targets) => match value {
                Value::Tuple(items) if items.len() == targets.len() => {
                    for (t, v) in targets.iter().zip(items) {
                        self.assign(t, op, v, f)?;
                    }
                    Ok(())
                }
                other => Err(format!(
                    "cannot assign {} to {} places",
                    other.kind(),
                    targets.len()
                )),
            },
            ExprKind::Var(id) => {
                let current = f
                    .vars
                    .get(*id)
                    .ok_or("assignment outside the body's variables")?
                    .clone();
                match (current, value) {
                    (Some(Value::Scalar(d, x)), Value::Scalar(vd, v)) if op != AssignOp::Assign => {
                        f.vars[*id] =
                            Some(Value::scalar(super::scalar::assign(op, (d, x), (vd, v))?));
                        Ok(())
                    }
                    (
                        Some(Value::View(s)),
                        v @ (Value::Scalar(..) | Value::Tile(_) | Value::View(_)),
                    ) => self.write(&s, op, &v),
                    (
                        Some(Value::Tile(s)),
                        v @ (Value::Scalar(..) | Value::Tile(_) | Value::View(_)),
                    ) if matches!(s.backing, Backing::Owned(_))
                        && (op != AssignOp::Assign
                            || v.shaped().is_some_and(|n| {
                                n.shape == s.shape && self.dtype_of(n) == self.dtype_of(&s)
                            })) =>
                    {
                        self.write(&s, op, &v)
                    }
                    (_, v) if op == AssignOp::Assign => {
                        f.vars[*id] = Some(self.snapshot(v)?);
                        Ok(())
                    }
                    (current, v) => Err(format!(
                        "`{}` of {} into {}",
                        op.text(),
                        v.kind(),
                        current.map_or("an unset variable", |c| c.kind())
                    )),
                }
            }
            ExprKind::Index { .. } | ExprKind::Transpose(_) => {
                let dst = self.place(target, f)?;
                if matches!(value, Value::Scalar(..))
                    && !dst.shape.is_empty()
                    && op == AssignOp::Assign
                {
                    return Err("scalar assigned to a place that is not one element".into());
                }
                self.write(&dst, op, &value)
            }
            _ => Err("unsupported assignment target".into()),
        }
    }

    fn counted(
        &mut self,
        vars: &[usize],
        bounds: &[(i64, i64)],
        body: &'a Block,
        f: &mut Frame<'a>,
    ) -> Result<Flow, String> {
        let mut flow = Flow::Next;
        if bounds.iter().all(|(lo, hi)| lo < hi) {
            let mut at: Vec<i64> = bounds.iter().map(|b| b.0).collect();
            'visits: loop {
                for (v, x) in vars.iter().zip(&at) {
                    f.vars[*v] = Some(Value::int(*x));
                }
                match self.block(body, f)? {
                    Flow::Next => {}
                    other => {
                        flow = other;
                        break;
                    }
                }
                let mut k = at.len();
                loop {
                    if k == 0 {
                        break 'visits;
                    }
                    k -= 1;
                    at[k] += 1;
                    if at[k] < bounds[k].1 {
                        break;
                    }
                    at[k] = bounds[k].0;
                }
            }
        }
        for v in vars {
            f.vars[*v] = None;
        }
        Ok(flow)
    }

    fn stmt(&mut self, s: &'a Stmt, f: &mut Frame<'a>) -> Result<Flow, String> {
        match &s.kind {
            StmtKind::Bind { pattern, value } => {
                let v = self.expr(value, f)?;
                let v = self.snapshot(v)?;
                self.bind(pattern, v, f)?;
                Ok(Flow::Next)
            }
            StmtKind::Assign { target, op, value } => {
                // The whole right-hand side reads old versions before anything is installed.
                let v = self.expr(value, f)?;
                let v = self.snapshot(v)?;
                self.assign(target, *op, v, f)?;
                Ok(Flow::Next)
            }
            StmtKind::Region(region) => {
                self.region(region, f)?;
                Ok(Flow::Next)
            }
            StmtKind::Stages(stages) => self.stages(stages, f),
            StmtKind::Range {
                var,
                lo,
                hi,
                value,
                body,
                ..
            } => {
                let bounds = if let Some(value) = value {
                    let Value::Range(lo, hi) = self.expr(value, f)? else {
                        return Err("range loop source is not a range".into());
                    };
                    [(lo, hi)]
                } else {
                    [(self.int(lo, f)?, self.int(hi, f)?)]
                };
                self.counted(&[*var], &bounds, body, f)
            }
            StmtKind::Coordinates {
                vars,
                of,
                axes,
                body,
            } => {
                let shaped = self.place(of, f)?;
                if vars.len() != axes.len() {
                    return Err(
                        "coordinate loop binds a different number of names than axes".into(),
                    );
                }
                let mut bounds = Vec::with_capacity(axes.len());
                for axis in axes {
                    let n = *shaped.shape.get(*axis).ok_or_else(|| {
                        format!("axis {axis} of a rank-{} value", shaped.shape.len())
                    })? as i64;
                    let base = match of.ty.shaped().and_then(|t| t.axes.get(*axis)) {
                        Some(Extent::Structural(slice)) => self.piece(f, *slice)?.lo,
                        _ => 0,
                    };
                    bounds.push((base, base + n));
                }
                self.counted(vars, &bounds, body, f)
            }
            StmtKind::Members { var, slice, body } => {
                let piece = self.piece(f, *slice)?;
                self.counted(&[*var], &[(piece.lo, piece.hi)], body, f)
            }
            StmtKind::If { cond, then, els } => match self.scalar(cond, f)? {
                (DType::Bool, x) => self.block(if x != 0.0 { then } else { els }, f),
                (d, _) => Err(format!("condition is {}, not bool", d.name())),
            },
            StmtKind::Publish { value, destination } => {
                let v = self.expr(value, f)?;
                let dst = self.place(destination, f)?;
                match &v {
                    Value::Scalar(..) if !dst.shape.is_empty() => {
                        return Err(format!(
                            "publish of a scalar to a destination of shape {:?}",
                            dst.shape
                        ))
                    }
                    Value::Scalar(..) | Value::Tile(_) | Value::View(_) => {}
                    other => return Err(format!("publish of {}", other.kind())),
                }
                self.write(&dst, AssignOp::Assign, &v)?;
                Ok(Flow::Next)
            }
            StmtKind::Yield(items) => Ok(Flow::Yield(self.values(items, f)?)),
            StmtKind::Return(items) => Ok(Flow::Return(self.values(items, f)?)),
            StmtKind::Expr(e) => {
                self.expr(e, f)?;
                Ok(Flow::Next)
            }
        }
    }

    /// A linear stage chain: each stage completes before the next; a yield binds the next
    /// stage's ports positionally; the terminal stage's yield is the chain's yield.
    fn stages(&mut self, stages: &'a [Stage], f: &mut Frame<'a>) -> Result<Flow, String> {
        let mut port: Option<Value> = None;
        for (i, stage) in stages.iter().enumerate() {
            match (stage.ports.as_slice(), port.take()) {
                ([], _) => {}
                ([one], Some(v)) => f.vars[*one] = Some(v),
                (many, Some(Value::Tuple(items))) if items.len() == many.len() => {
                    for (var, v) in many.iter().zip(items) {
                        f.vars[*var] = Some(v);
                    }
                }
                (many, _) => {
                    return Err(self.locate(
                        f.def,
                        stage.span,
                        format!(
                            "stage `{}` binds {} ports the previous stage did not yield",
                            stage.name,
                            many.len()
                        ),
                    ))
                }
            }
            match self.block(&stage.body, f)? {
                Flow::Next => {}
                Flow::Yield(v) if i + 1 == stages.len() => return Ok(Flow::Yield(v)),
                Flow::Yield(v) => port = Some(v),
                flow @ Flow::Return(_) => return Ok(flow),
            }
        }
        Ok(Flow::Next)
    }

    // ----- regions ---------------------------------------------------------------------

    /// Pieces of one binder: uniform-capacity pieces with a shorter tail, or for a merge the
    /// canonical near-equal partition into `clamp(ceil(extent / width), 1, extent)` parts.
    fn pieces(
        &self,
        f: &Frame<'a>,
        region: &Region,
        slice: SliceId,
        merge: bool,
    ) -> Result<Vec<Piece>, String> {
        let decl = f
            .body
            .slices
            .get(slice.0 as usize)
            .ok_or("unknown slice binder")?;
        let (lo, hi) = match &decl.parent {
            SliceParent::Domain { lo, hi } => (f.sym(lo)?, f.sym(hi)?),
            SliceParent::Refine(parent) | SliceParent::Rebind(parent) => {
                let p = self.piece(f, *parent)?;
                (p.lo, p.hi)
            }
        };
        let domain = (hi - lo).max(0);
        if domain == 0 {
            return Ok(Vec::new());
        }
        let width = self
            .partitioner
            .width(&f.def.name, region.id, slice, domain);
        if width < 1 || width > domain {
            return Err(format!(
                "partitioner chose width {width} for an extent of {domain}"
            ));
        }
        let mut out = Vec::new();
        if merge {
            let parts = ((domain + width - 1) / width).clamp(1, domain);
            let (base, extra) = (domain / parts, domain % parts);
            let capacity = base + i64::from(extra > 0);
            let mut at = lo;
            for p in 0..parts {
                let n = base + i64::from(p < extra);
                out.push(Piece {
                    lo: at,
                    hi: at + n,
                    width: capacity,
                    domain,
                });
                at += n;
            }
        } else {
            let mut at = lo;
            while at < hi {
                out.push(Piece {
                    lo: at,
                    hi: (at + width).min(hi),
                    width,
                    domain,
                });
                at += width;
            }
        }
        Ok(out)
    }

    /// Execute a region. A result-producing region returns `Flow::Yield` of its value.
    pub(super) fn region(&mut self, r: &'a Region, f: &mut Frame<'a>) -> Result<Flow, String> {
        let body = f.body;
        let binders = r
            .binders
            .iter()
            .map(|v| match body.vars.get(*v).map(|var| &var.kind) {
                Some(VarKind::Slice(id)) => Ok((*v, *id)),
                _ => Err("region binder is not a slice".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let visits: Vec<Vec<Piece>> = match &r.source {
            RegionSource::Results(source) => match self.expr(source, f)? {
                Value::Result(result) => result.pieces.clone(),
                other => {
                    return Err(format!(
                        "region traverses {}, not a region result",
                        other.kind()
                    ))
                }
            },
            RegionSource::Domains => {
                let axes = binders
                    .iter()
                    .map(|(_, id)| self.pieces(f, r, *id, r.merge.is_some()))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut visits = vec![Vec::new()];
                for axis in &axes {
                    visits = visits
                        .iter()
                        .flat_map(|prefix| {
                            axis.iter()
                                .map(move |p| prefix.iter().copied().chain([*p]).collect())
                        })
                        .collect();
                }
                visits
            }
        };
        // A rebinding also stands for the producer's binder in member types.
        let mut bound: Vec<(usize, SliceId)> = Vec::new();
        for (k, (_, id)) in binders.iter().enumerate() {
            bound.push((k, *id));
            if let Some(SliceParent::Rebind(origin)) =
                body.slices.get(id.0 as usize).map(|d| &d.parent)
            {
                if matches!(r.source, RegionSource::Results(_))
                    && f.slices.get(origin.0 as usize).is_some_and(|s| s.is_none())
                {
                    bound.push((k, *origin));
                }
            }
        }
        let saved: Vec<Option<Piece>> = bound
            .iter()
            .map(|(_, id)| f.slices.get(id.0 as usize).copied().flatten())
            .collect();
        let producing = r.result.is_some() || r.merge.is_some();
        let mut values = Vec::with_capacity(visits.len());
        for visit in &visits {
            if visit.len() != binders.len() {
                return Err(format!(
                    "region binds {} slices over a result of {} axes",
                    binders.len(),
                    visit.len()
                ));
            }
            for (k, id) in &bound {
                *f.slices
                    .get_mut(id.0 as usize)
                    .ok_or("unknown slice binder")? = Some(visit[*k]);
            }
            for ((var, _), piece) in binders.iter().zip(visit) {
                f.vars[*var] = Some(Value::Slice(*piece));
            }
            match self.block(&r.body, f)? {
                Flow::Yield(v) if producing => values.push(v),
                Flow::Next if !producing => {}
                Flow::Yield(_) => return Err("a statement region cannot yield".into()),
                Flow::Next => {
                    return Err("a visit of a result-producing region did not yield".into())
                }
                Flow::Return(_) => return Err("return inside a region".into()),
            }
        }
        for ((_, id), old) in bound.iter().zip(saved) {
            f.slices[id.0 as usize] = old;
        }
        for (var, _) in &binders {
            f.vars[*var] = None;
        }
        match &r.merge {
            Some(merge) => Ok(Flow::Yield(self.merge(merge, values, f)?)),
            None if producing => Ok(Flow::Yield(Value::Result(Rc::new(ResultVal::new(
                visits, values,
            ))))),
            None => Ok(Flow::Next),
        }
    }

    /// Adjacent pairs level by level, an odd last value forwarded; empty gives the identity
    /// and a single part its partial.
    fn merge(
        &mut self,
        merge: &'a Merge,
        mut level: Vec<Value>,
        f: &mut Frame<'a>,
    ) -> Result<Value, String> {
        if level.is_empty() {
            let identity = self.expr(&merge.identity, f)?;
            return self.snapshot(identity);
        }
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut items = level.into_iter();
            while let Some(left) = items.next() {
                let Some(right) = items.next() else {
                    next.push(left);
                    break;
                };
                self.bind(&merge.left, left, f)?;
                self.bind(&merge.right, right, f)?;
                match self.block(&merge.body, f)? {
                    Flow::Yield(v) => next.push(v),
                    _ => return Err("merge body did not yield".into()),
                }
            }
            level = next;
        }
        level
            .pop()
            .ok_or_else(|| "merge lost its value".to_string())
    }
}
