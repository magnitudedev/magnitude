//! Frames, reference-body selection, calls, and statement execution over the
//! checked representation. A function family is evaluated through its
//! reference body; lowerings are non-reference alternatives and are never
//! interpreted.
use super::value::{Backing, Flow, Shaped, Value};
use super::{Arg, Bindings, Interpreter};
use crate::sir::{
    BlockTerminator, CheckedBlock, CheckedCall, CheckedExpr, CheckedExprKind, CheckedPlace,
    CheckedStmt, ContractFamily, DefId, DefKind, Definition, Mode, Pattern, Predicate,
};
use crate::span::{line_col, Span};
use crate::sym::Sym;
use crate::syntax::ast::AssignOp;
use crate::types::{DType, Elem, ExtentExpr, ValueType};
use std::collections::HashMap;

pub(super) struct Frame<'a> {
    pub def: &'a Definition,
    pub body: &'a crate::sir::CheckedBody,
    pub locals: Vec<Option<Value>>,
    /// Shape parameters.
    pub shapes: HashMap<String, i64>,
    pub elems: HashMap<String, Elem>,
}

impl<'a> Frame<'a> {
    /// Symbols name shape parameters, or the runtime value of an integer body
    /// local as `name#LocalId` (the checker's variable atoms).
    fn lookup(&self, name: &str) -> Option<i64> {
        if let Some(v) = self.shapes.get(name) {
            return Some(*v);
        }
        let integer = |i: usize| match self.locals.get(i) {
            Some(Some(Value::Scalar(d, x))) if d.is_int() => Some(*x as i64),
            _ => None,
        };
        if let Some((base, id)) = name.rsplit_once('#') {
            if let Some(id) = id
                .parse::<usize>()
                .ok()
                .filter(|id| self.body.locals.get(*id).is_some_and(|v| v.name == base))
            {
                return integer(id);
            }
        }
        self.body
            .locals
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, local)| local.name == name)
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

    /// A `where` predicate over the bound shapes.
    fn holds(&self, p: &Predicate) -> Result<bool, String> {
        let eval = |s: &Sym| {
            s.eval(&|n| self.shapes.get(n).copied())
                .ok_or_else(|| format!("cannot evaluate predicate operand `{s}`"))
        };
        Ok(match p {
            Predicate::NonNegative(s) => eval(s)? >= 0,
            Predicate::Zero(s) => eval(s)? == 0,
            Predicate::NonZero(s) => eval(s)? != 0,
        })
    }
}

fn same_backing(a: &Shaped, b: &Shaped) -> bool {
    match (&a.backing, &b.backing) {
        (Backing::Owned(x), Backing::Owned(y)) => std::rc::Rc::ptr_eq(x, y),
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
        bindings: &Bindings,
    ) -> Result<Value, String> {
        let program = self.program;
        let family = program.resolve_family(name)?;
        let families = [family];
        let shapes: HashMap<String, i64> = bindings
            .shapes
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let elems: HashMap<String, Elem> = bindings
            .elems
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let this: &Interpreter<'a> = self;
        let make = |d: &'a Definition| -> Result<Frame<'a>, String> {
            if let Some(missing) = d.shape_params.iter().find(|p| !shapes.contains_key(*p)) {
                return Err(format!("bindings do not bind shape parameter {missing}"));
            }
            let values = this.entry_values(d, args, &shapes)?;
            this.instantiate(d, shapes.clone(), elems.clone(), values)
        };
        let mut frame = self.select_body(&family.name, &families, &make)?;
        let body = frame.body;
        match self.block(&body.root, &mut frame)? {
            Flow::Return(v) => Ok(v),
            Flow::Next => Ok(Value::Void),
        }
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
                        super::round_to(dtype, v)
                    } else {
                        v
                    },
                ))
            };
            values.push(match (arg, &param.ty) {
                (Arg::Tensor(id), ValueType::Tensor(_)) => {
                    let t = self.tensors.get(*id).ok_or_else(|| {
                        format!("argument {i} names tensor {id}, which does not exist")
                    })?;
                    Value::Tensor(Shaped::tensor(*id, t.shape()))
                }
                (Arg::Scalar(v), ValueType::Scalar(dtype)) => scalar(*dtype, None, *v)?,
                (Arg::Scalar(v), ValueType::Index { bound }) => {
                    let bound = bound
                        .sym()
                        .and_then(|s| s.eval(&|n| shapes.get(n).copied()))
                        .and_then(|n| u64::try_from(n).ok())
                        .ok_or_else(|| format!("unresolved index bound for `{}`", param.name))?;
                    scalar(DType::I32, Some(bound), *v)?
                }
                (Arg::Range(start, end), ValueType::Range { bound }) => {
                    let bound = bound
                        .sym()
                        .and_then(|s| s.eval(&|n| shapes.get(n).copied()))
                        .and_then(|n| u64::try_from(n).ok())
                        .ok_or_else(|| format!("unresolved range bound for `{}`", param.name))?;
                    let mut first = crate::abi::ScalarParameter::plain(
                        format!("{}_start", param.name),
                        DType::I32,
                    );
                    first.range = Some(crate::abi::RangeScalar {
                        parameter: param.name.clone(),
                        endpoint: crate::abi::RangeEndpoint::Start,
                        bound,
                    });
                    let mut last = crate::abi::ScalarParameter::plain(
                        format!("{}_end", param.name),
                        DType::I32,
                    );
                    last.range = Some(crate::abi::RangeScalar {
                        parameter: param.name.clone(),
                        endpoint: crate::abi::RangeEndpoint::End,
                        bound,
                    });
                    crate::abi::ScalarLayout::words(&[first, last])?
                        .encode(&[*start as f64, *end as f64])?;
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

    /// Bind one definition to evaluated arguments (in parameter order). `Err`
    /// is the reason the definition is not applicable.
    fn instantiate(
        &self,
        def: &'a Definition,
        mut shapes: HashMap<String, i64>,
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
        for (param, value) in def.params.iter().zip(&values) {
            match (&param.ty, value) {
                (ValueType::Tensor(t), Value::Tensor(s)) => {
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
                        if let ExtentExpr::Sym(sym) = axis {
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
                (ValueType::Scalar(_) | ValueType::Index { .. }, Value::Scalar(..))
                | (ValueType::Range { .. }, Value::Range(..))
                | (ValueType::Tuple(_), Value::Tuple(_)) => {}
                (ty, v) => {
                    return Err(format!(
                        "`{}`: {} where {ty} is required",
                        param.name,
                        v.kind()
                    ))
                }
            }
        }
        let mut locals = vec![None; body.locals.len()];
        for (param, value) in def.params.iter().zip(values) {
            *locals
                .get_mut(param.local)
                .ok_or("parameter local outside the body")? = Some(value);
        }
        let frame = Frame {
            def,
            body,
            locals,
            shapes,
            elems,
        };
        for param in &def.params {
            let Some(value) = &frame.locals[param.local] else {
                continue;
            };
            match (&param.ty, value) {
                (ValueType::Tensor(t), Value::Tensor(s)) => {
                    for (k, (axis, n)) in t.axes.iter().zip(&s.shape).enumerate() {
                        if let ExtentExpr::Sym(sym) = axis {
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
                (ValueType::Index { bound }, Value::Scalar(_, x)) => {
                    let bound = bound
                        .sym()
                        .and_then(|s| frame.sym(s).ok())
                        .unwrap_or(i64::MAX);
                    if *x < 0.0 || *x as i64 >= bound {
                        return Err(format!(
                            "`{}` = {x} is outside its declared index bound {bound}",
                            param.name
                        ));
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

    /// Deterministic reference-body choice: portable bodies first, then
    /// backend-specific bodies matching the interpreter target. Lowerings are
    /// non-reference alternatives and are never selected.
    fn select_body(
        &self,
        name: &str,
        families: &[&'a ContractFamily],
        make: &dyn Fn(&'a Definition) -> Result<Frame<'a>, String>,
    ) -> Result<Frame<'a>, String> {
        let program = self.program;
        let target = self.target.as_deref();
        let mut ids: Vec<DefId> = families
            .iter()
            .flat_map(|f| f.bodies.iter())
            .copied()
            .collect();
        ids.sort();
        ids.dedup();
        // Prefer a portable reference body; fall back to the target's helper.
        ids.sort_by_key(|id| {
            matches!(
                program.definition(*id).kind,
                DefKind::Body { target: Some(_) }
            ) as u8
        });
        let mut applicable: Vec<(DefId, Frame<'a>)> = Vec::new();
        let mut reasons = Vec::new();
        for id in ids {
            let d = program.definition(id);
            let available = match &d.kind {
                DefKind::Body { target: None } => true,
                DefKind::Body {
                    target: Some(required),
                } => Some(required.as_str()) == target,
                DefKind::Lower { .. } => false,
            };
            if !available {
                continue;
            }
            match make(d) {
                Ok(frame) => applicable.push((id, frame)),
                Err(reason) => reasons.push(format!("definition {}: {reason}", id.0)),
            }
        }
        applicable
            .into_iter()
            .map(|(_, frame)| frame)
            .next()
            .ok_or_else(|| {
                format!(
                    "no reference body of `{name}` is applicable{}{}",
                    if reasons.is_empty() { "" } else { ": " },
                    reasons.join("; ")
                )
            })
    }

    pub(super) fn call(
        &mut self,
        call: &CheckedCall,
        args: &[CheckedExpr],
        f: &mut Frame<'a>,
    ) -> Result<Value, String> {
        let program = self.program;
        let family = program
            .families
            .get(call.family)
            .ok_or("call names an unknown family")?;
        let mut values = Vec::with_capacity(args.len());
        for a in args {
            values.push(self.expr(a, f)?);
        }
        let mut inner = {
            let caller: &Frame<'a> = f;
            let this: &Interpreter<'a> = self;
            let make = |d: &'a Definition| -> Result<Frame<'a>, String> {
                let b = call
                    .bindings
                    .iter()
                    .find(|b| b.definition == d.id)
                    .ok_or("the arguments do not bind its parameters")?;
                let mut shapes = HashMap::new();
                for (name, sym) in &b.shape_args {
                    // An extent the caller cannot evaluate is solved from the
                    // runtime shapes of the arguments and checked there.
                    if let Ok(n) = caller.sym(sym) {
                        shapes.insert(name.clone(), n);
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
                this.instantiate(d, shapes, elems, ordered)
            };
            this.select_body(&family.name, &[family], &make)?
        };
        let callee = inner.body;
        let binding = call
            .bindings
            .iter()
            .find(|b| b.definition == inner.def.id)
            .ok_or("selected definition lost its binding")?;
        let result = match self.block(&callee.root, &mut inner)? {
            Flow::Return(v) => v,
            Flow::Next => Value::Void,
        };
        // `inout` effects: scalar and tuple state is installed back; tensor
        // storage was shared through the same backing during the call.
        for (param, ordinal) in inner.def.params.iter().zip(&binding.arg_order) {
            if param.mode == Mode::In {
                continue;
            }
            let arg = args
                .get(*ordinal)
                .ok_or("argument order names a missing argument")?;
            match inner.locals[param.local].take() {
                Some(v @ (Value::Scalar(..) | Value::Tuple(_))) => {
                    self.assign_expr(arg, AssignOp::Assign, v, f)?
                }
                Some(Value::Tensor(s)) => {
                    if let CheckedExprKind::Local(id) = &arg.kind {
                        let shared = matches!(
                            &f.locals[*id],
                            Some(Value::Tensor(c)) if same_backing(c, &s)
                        );
                        if !shared {
                            f.locals[*id] = Some(Value::Tensor(s));
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

    pub(super) fn block(
        &mut self,
        block: &'a CheckedBlock,
        f: &mut Frame<'a>,
    ) -> Result<Flow, String> {
        for s in &block.statements {
            match self
                .stmt(s, f)
                .map_err(|m| self.locate(f.def, s.span(), m))?
            {
                Flow::Next => {}
                flow => return Ok(flow),
            }
        }
        match &block.terminator {
            BlockTerminator::Continue => Ok(Flow::Next),
            BlockTerminator::Return(values) => {
                let value = self.values(values, f)?;
                Ok(Flow::Return(value))
            }
        }
    }

    fn bind(&mut self, pattern: &Pattern, value: Value, f: &mut Frame<'a>) -> Result<(), String> {
        match (pattern, value) {
            (Pattern::Local(id), value) => {
                *f.locals
                    .get_mut(*id)
                    .ok_or("binding outside the body's locals")? = Some(value);
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

    fn values(&mut self, items: &[CheckedExpr], f: &mut Frame<'a>) -> Result<Value, String> {
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

    /// Assign through a checked place.
    fn assign_place(
        &mut self,
        place: &'a CheckedPlace,
        op: AssignOp,
        value: Value,
        f: &mut Frame<'a>,
    ) -> Result<(), String> {
        match place {
            CheckedPlace::Local { root } => {
                let current = f
                    .locals
                    .get(*root)
                    .ok_or("assignment outside the body's locals")?
                    .clone();
                match (current, value) {
                    (Some(Value::Scalar(d, x)), Value::Scalar(vd, v)) if op != AssignOp::Assign => {
                        f.locals[*root] =
                            Some(Value::scalar(super::scalar::assign(op, (d, x), (vd, v))?));
                        Ok(())
                    }
                    (Some(Value::Tensor(s)), v @ (Value::Scalar(..) | Value::Tensor(_)))
                        if op != AssignOp::Assign
                            || v.shaped().is_some_and(|n| {
                                n.shape == s.shape && self.dtype_of(n) == self.dtype_of(&s)
                            }) =>
                    {
                        self.write(&s, op, &v)
                    }
                    (_, v) if op == AssignOp::Assign => {
                        f.locals[*root] = Some(self.snapshot(v)?);
                        Ok(())
                    }
                    (current, v) => Err(format!(
                        "`{}` of {} into {}",
                        op.text(),
                        v.kind(),
                        current.map_or("an unset local", |c| c.kind())
                    )),
                }
            }
            CheckedPlace::Element { root, indices } => {
                let current = f
                    .locals
                    .get(*root)
                    .ok_or("assignment outside the body's locals")?
                    .clone()
                    .ok_or_else(|| "`assignment target has no value".to_string())?;
                let Value::Tensor(base) = current else {
                    return Err("element assignment on a value that is not tensor storage".into());
                };
                let dst = self.select(base, indices, f)?;
                if matches!(value, Value::Scalar(..))
                    && !dst.shape.is_empty()
                    && op == AssignOp::Assign
                {
                    return Err("scalar assigned to a place that is not one element".into());
                }
                self.write(&dst, op, &value)
            }
            CheckedPlace::Tuple(places) => match value {
                Value::Tuple(items) if items.len() == places.len() => {
                    for (p, v) in places.iter().zip(items) {
                        self.assign_place(p, op, v, f)?;
                    }
                    Ok(())
                }
                other => Err(format!(
                    "cannot assign {} to {} places",
                    other.kind(),
                    places.len()
                )),
            },
        }
    }

    /// Assign back through a call-argument expression (a local or a selection).
    fn assign_expr(
        &mut self,
        target: &CheckedExpr,
        op: AssignOp,
        value: Value,
        f: &mut Frame<'a>,
    ) -> Result<(), String> {
        match &target.kind {
            CheckedExprKind::Local(id) => {
                let current = f.locals.get(*id).cloned().flatten();
                match (current, value) {
                    (Some(Value::Scalar(d, x)), Value::Scalar(vd, v)) if op != AssignOp::Assign => {
                        f.locals[*id] =
                            Some(Value::scalar(super::scalar::assign(op, (d, x), (vd, v))?));
                        Ok(())
                    }
                    (Some(Value::Tensor(s)), v @ (Value::Scalar(..) | Value::Tensor(_)))
                        if matches!(s.backing, Backing::Owned(_))
                            && (op != AssignOp::Assign
                                || v.shaped().is_some_and(|n| {
                                    n.shape == s.shape && self.dtype_of(n) == self.dtype_of(&s)
                                })) =>
                    {
                        self.write(&s, op, &v)
                    }
                    (_, v) if op == AssignOp::Assign => {
                        f.locals[*id] = Some(self.snapshot(v)?);
                        Ok(())
                    }
                    (current, v) => Err(format!(
                        "`{}` of {} into {}",
                        op.text(),
                        v.kind(),
                        current.map_or("an unset local", |c| c.kind())
                    )),
                }
            }
            CheckedExprKind::Primitive {
                id: crate::intrinsics::PrimitiveId::SliceView { .. },
                ..
            } => {
                let dst = self.place(target, f)?;
                self.write(&dst, op, &value)
            }
            _ => Err("unsupported assignment target".into()),
        }
    }

    fn stmt(&mut self, s: &'a CheckedStmt, f: &mut Frame<'a>) -> Result<Flow, String> {
        match s {
            CheckedStmt::Let { pattern, value, .. } => {
                let v = self.expr(value, f)?;
                let v = self.snapshot(v)?;
                self.bind(pattern, v, f)?;
                Ok(Flow::Next)
            }
            CheckedStmt::Assign { place, op, value } => {
                // The whole right-hand side reads old versions before anything is installed.
                let v = self.expr(value, f)?;
                let v = self.snapshot(v)?;
                self.assign_place(place, *op, v, f)?;
                Ok(Flow::Next)
            }
            CheckedStmt::Loop {
                binder,
                range,
                body,
                ..
            } => {
                let lo = self.int(&range.start, f)?;
                let hi = self.int(&range.end, f)?;
                // Ascending coordinate order: the deterministic reference order
                // for ordered and independent loops alike.
                for i in lo..hi {
                    *f.locals
                        .get_mut(*binder)
                        .ok_or("loop binder outside the body's locals")? = Some(Value::int(i));
                    match self.block(body, f)? {
                        Flow::Next => {}
                        Flow::Return(_) => {
                            return Err("`return` inside a loop is rejected while checking".into())
                        }
                    }
                }
                f.locals[*binder] = None;
                Ok(Flow::Next)
            }
            CheckedStmt::If {
                condition,
                then_body,
                else_body,
            } => match self.scalar(condition, f)? {
                (DType::Bool, x) => self.block(if x != 0.0 { then_body } else { else_body }, f),
                (d, _) => Err(format!("condition is {}, not bool", d.name())),
            },
            CheckedStmt::Evaluate(expr) => {
                self.expr(expr, f)?;
                Ok(Flow::Next)
            }
        }
    }
}

/// Span of a checked statement, for diagnostics.
trait StmtSpan {
    fn span(&self) -> Span;
}

impl StmtSpan for CheckedStmt {
    fn span(&self) -> Span {
        match self {
            CheckedStmt::Let { value, .. } => value.span,
            CheckedStmt::Assign { value, .. } => value.span,
            CheckedStmt::Loop { range, .. } => range.start.span,
            CheckedStmt::If { condition, .. } => condition.span,
            CheckedStmt::Evaluate(e) => e.span,
        }
    }
}
