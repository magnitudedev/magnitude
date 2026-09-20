//! Semantic invariants of the execution IR. Every load must be resolved before
//! realization. These checks validate retained facts; they do not infer
//! optimization permissions.
use super::{ir::*, lowered_ir::LoweredIr, types::Ty};
use crate::syntax::ast::AssignOp;
use crate::types::{DType, Elem};
use std::collections::HashSet;

pub fn lowered(function: &LoweredIr) -> Result<(), String> {
    check(function, None)
        .map_err(|error| format!("{}: invalid Executable IR: {error}", function.name))
}
/// Verify normalized launches with the index values supplied by their explicit
/// dispatch prologues. A binding is visible only within its owning phase.
pub fn executable_phases(function: &LoweredIr, indices: &[Vec<VarId>]) -> Result<(), String> {
    check(function, Some(indices))
        .map_err(|error| format!("{}: invalid Executable phase IR: {error}", function.name))
}
fn check(function: &LoweredIr, phase_indices: Option<&[Vec<VarId>]>) -> Result<(), String> {
    function.ownership.validate(function)?;
    if function.source_param_count > function.params.len() {
        return Err("source parameter prefix exceeds the entry ABI".into());
    }
    fn leaves(ty: &Ty, path: &mut Vec<u32>, out: &mut Vec<(Vec<u32>, Ty)>) -> Result<(), String> {
        match ty {
            Ty::Tensor(_) => out.push((path.clone(), ty.clone())),
            Ty::Tuple(items) => {
                for (ordinal, item) in items.iter().enumerate() {
                    path.push(ordinal as u32);
                    leaves(item, path, out)?;
                    path.pop();
                }
            }
            Ty::Void => {}
            _ => {
                return Err("entry result must be an owned tensor or tuple of owned tensors".into());
            }
        }
        Ok(())
    }
    let mut expected = Vec::new();
    leaves(&function.result, &mut Vec::new(), &mut expected)?;
    if expected.len() != function.result_bindings.len() {
        return Err("entry result leaf count disagrees with its hidden destinations".into());
    }
    if function.params.len() - function.source_param_count != expected.len() {
        return Err("entry ABI has a hidden parameter that is not a result destination".into());
    }
    let mut result_names = HashSet::new();
    for (ordinal, ((path, ty), binding)) in
        expected.iter().zip(&function.result_bindings).enumerate()
    {
        if path != &binding.path || binding.parameter != function.source_param_count + ordinal {
            return Err("entry result destination has an invalid path or ABI position".into());
        }
        let Some((name, parameter_ty)) = function.params.get(binding.parameter) else {
            return Err("entry result destination is outside the ABI".into());
        };
        if parameter_ty != ty || !function.ownership.results.contains(name) {
            return Err("entry result destination type or ownership metadata disagrees".into());
        }
        result_names.insert(name.clone());
    }
    if result_names != function.ownership.results.iter().cloned().collect() {
        return Err("entry result ownership metadata contains a non-result binding".into());
    }
    for requirement in &function.alias_requirements {
        if [requirement.left, requirement.right]
            .iter()
            .any(|&p| !matches!(function.params.get(p), Some((_, Ty::Tensor(_)))))
        {
            return Err("memory alias requirement does not refer to tensor parameters".into());
        }
    }
    let mut bound = HashSet::new();
    let mut parameters = HashSet::new();
    for (id, var) in function.vars.iter().enumerate() {
        let parameter = match &var.kind {
            VarKind::Param(parameter) => Some(*parameter),
            VarKind::Index(crate::sym::Atom::Param(name))
                if function.index_params.iter().any(|(n, _)| n == name) =>
            {
                function.params.iter().position(|(n, _)| n == name)
            }
            _ => None,
        };
        if let Some(parameter) = parameter {
            let Some((_, ty)) = function.params.get(parameter) else {
                return Err("parameter ordinal is outside the entry ABI".into());
            };
            if &var.ty != ty {
                return Err(format!(
                    "parameter `{}` has inconsistent type metadata",
                    var.name
                ));
            }
            bound.insert(id);
            parameters.insert(parameter);
        }
    }
    if parameters.len() != function.params.len() {
        return Err("entry parameter has no variable binding".into());
    }
    let verifier = Verifier { function };
    if let Some(indices) = phase_indices {
        if indices.len() != function.body.len() {
            return Err("launch bindings disagree with the phase count".into());
        }
        for (phase, indices) in function.body.iter().zip(indices) {
            if !matches!(phase.kind, StmtKind::Parallel { .. }) {
                return Err("executable phase has no parallel work domain".into());
            }
            let mut local = bound.clone();
            verifier.indices(indices, &mut local)?;
            verifier.body(std::slice::from_ref(phase), &mut local)?;
        }
    } else {
        verifier.body(&function.body, &mut bound)?;
    }
    verifier.snapshots(&function.body)?;
    Ok(())
}

struct Verifier<'a> {
    function: &'a LoweredIr,
}
impl Verifier<'_> {
    fn var(&self, id: VarId) -> Result<&Var, String> {
        self.function
            .vars
            .get(id)
            .ok_or_else(|| format!("invalid variable identity {id}"))
    }
    fn reference(&self, id: VarId, ty: &Ty) -> Result<(), String> {
        let var = self.var(id)?;
        if &var.ty != ty {
            return Err(format!(
                "reference to `{}` has type {ty}, binding has {}",
                var.name, var.ty
            ));
        }
        Ok(())
    }
    fn indices(&self, vars: &[VarId], bound: &mut HashSet<VarId>) -> Result<(), String> {
        for &id in vars {
            let var = self.var(id)?;
            if !matches!(var.kind, VarKind::Index(_)) || var.ty != Ty::Scalar(DType::I32) {
                return Err(format!(
                    "iteration binding `{}` is not an i32 index",
                    var.name
                ));
            }
            if !bound.insert(id) {
                return Err(format!(
                    "iteration binding `{}` shadows an active identity",
                    var.name
                ));
            }
        }
        Ok(())
    }
    fn body(&self, body: &[Stmt], bound: &mut HashSet<VarId>) -> Result<(), String> {
        for statement in body {
            match &statement.kind {
                StmtKind::Assign { target, op, value } => {
                    if matches!(&value.kind, ExprKind::Intrinsic { op, .. } if op.produces_owned_result())
                    {
                        let ExprKind::Var(id) = target.kind else {
                            return Err(
                                "owned intrinsic result requires a direct local destination".into(),
                            );
                        };
                        if *op != AssignOp::Assign || !matches!(value.ty, Ty::Tile(_)) {
                            return Err("owned intrinsic result requires plain assignment of an owned tile value".into());
                        }
                        if !matches!(self.var(id)?.kind, VarKind::Local) {
                            return Err(
                                "owned intrinsic result destination is not a compiler-owned local"
                                    .into(),
                            );
                        }
                        self.result_intrinsic(value, bound)?;
                    } else {
                        self.expr(value, bound)?;
                    }
                    if !assignable(&target.ty, &value.ty) {
                        return Err(format!(
                            "assignment changes type/shape from {} to {}",
                            value.ty, target.ty
                        ));
                    }
                    if let ExprKind::Var(id) = target.kind {
                        self.reference(id, &target.ty)?;
                        if *op != AssignOp::Assign {
                            self.expr(target, bound)?;
                        }
                        bound.insert(id);
                    } else {
                        self.expr(target, bound)?;
                        if !matches!(
                            target.kind,
                            ExprKind::Index { .. } | ExprKind::Accessor { .. }
                        ) {
                            return Err("assignment target has no storage identity".into());
                        }
                    }
                }
                StmtKind::Expr(expr) => self.expr(expr, bound)?,
                StmtKind::If { cond, then, els } => {
                    self.expr(cond, bound)?;
                    if cond.ty != Ty::Scalar(DType::Bool) {
                        return Err("conditional predicate is not scalar bool".into());
                    }
                    self.body(then, &mut bound.clone())?;
                    self.body(els, &mut bound.clone())?;
                }
                StmtKind::Parallel {
                    vars,
                    extents,
                    body,
                } => {
                    if vars.len() != extents.len()
                        || extents
                            .iter()
                            .any(|n| n.as_constant().is_some_and(|n| n < 0))
                    {
                        return Err("parallel indices and extents disagree".into());
                    }
                    let mut inner = bound.clone();
                    self.indices(vars, &mut inner)?;
                    self.body(body, &mut inner)?;
                }
                StmtKind::Owned { vars, tile, body } => {
                    self.expr(tile, bound)?;
                    if tile.ty.rank() != Some(vars.len()) {
                        return Err("owned indices and tile rank disagree".into());
                    }
                    let mut inner = bound.clone();
                    self.indices(vars, &mut inner)?;
                    self.body(body, &mut inner)?;
                }
                StmtKind::Range { var, body, .. } | StmtKind::Lanes { var, body, .. } => {
                    let mut inner = bound.clone();
                    self.indices(&[*var], &mut inner)?;
                    self.body(body, &mut inner)?;
                }
                StmtKind::LoadLoop {
                    domain,
                    offset,
                    modes,
                    vars,
                    views,
                    axes,
                    body,
                    capacity,
                    ..
                } => {
                    self.expr(&domain.view, bound)?;
                    let extent = domain
                        .view
                        .ty
                        .shaped()
                        .and_then(|s| s.shape.get(domain.axis))
                        .ok_or("stream domain axis is outside its view")?;
                    if vars.len() != views.len()
                        || vars.len() != axes.len()
                        || modes.as_ref().is_some_and(|m| m.len() != views.len())
                        || capacity.is_some_and(|n| n <= 0)
                    {
                        return Err("stream binding/transfer metadata disagree".into());
                    }
                    if modes.is_none() {
                        return Err("unresolved stream load choices".into());
                    }
                    let mut inner = bound.clone();
                    for ((&id, view), &axis) in vars.iter().zip(views).zip(axes) {
                        self.expr(view, bound)?;
                        let source = view.ty.shaped().ok_or("stream input is not shaped")?;
                        let active = source
                            .shape
                            .get(axis)
                            .ok_or("stream transfer axis is outside its view")?;
                        if extent
                            .as_constant()
                            .zip(active.as_constant())
                            .is_some_and(|(a, b)| a != b)
                        {
                            return Err("stream domain and transfer extents disagree".into());
                        }
                        let Ty::Tile(tile) = &self.var(id)?.ty else {
                            return Err("stream binding is not a tile".into());
                        };
                        if tile.shape.len() != source.shape.len() || tile.elem != source.elem {
                            return Err("stream snapshot loses input type/shape provenance".into());
                        }
                        inner.insert(id);
                    }
                    if let Some(offset) = offset {
                        self.indices(&[*offset], &mut inner)?;
                    }
                    self.body(body, &mut inner)?;
                }
            }
        }
        Ok(())
    }

    fn result_intrinsic(&self, expr: &Expr, bound: &HashSet<VarId>) -> Result<(), String> {
        let ExprKind::Intrinsic { op, args } = &expr.kind else {
            return Err("owned intrinsic result is not an intrinsic expression".into());
        };
        if !op.produces_owned_result() {
            return Err("value intrinsic was classified as an owned result incorrectly".into());
        }
        for argument in args {
            self.expr(argument, bound)?;
        }
        Ok(())
    }

    fn expr(&self, expr: &Expr, bound: &HashSet<VarId>) -> Result<(), String> {
        validate_type(&expr.ty)?;
        match &expr.kind {
            ExprKind::Var(id) => {
                self.reference(*id, &expr.ty)?;
                if !bound.contains(id) {
                    return Err(format!(
                        "`{}` is used outside its defining scope",
                        self.var(*id)?.name
                    ));
                }
            }
            ExprKind::TileAlloc { shape, dtype } => {
                let Ty::Tile(tile) = &expr.ty else {
                    return Err("allocation is not typed as a tile".into());
                };
                if &tile.shape != shape || &tile.elem != dtype {
                    return Err("allocation and tile type disagree".into());
                }
            }
            ExprKind::Load { view, .. } => {
                self.expr(view, bound)?;
                validate_load(&view.ty, &expr.ty, true)?;
            }
            ExprKind::Index { base, indices } => {
                self.expr(base, bound)?;
                let source = base.ty.shaped().ok_or("index base is not shaped")?;
                if indices.len() > source.shape.len() {
                    return Err("too many index coordinates".into());
                }
                for index in indices {
                    let values: Vec<_> = match index {
                        Index::Point(e) => vec![e],
                        Index::Slice { start, end } => start.iter().chain(end).collect(),
                    };
                    for value in values {
                        self.expr(value, bound)?;
                        if !matches!(value.ty, Ty::Scalar(d) if d.is_int()) {
                            return Err("index coordinate is not an integer scalar".into());
                        }
                    }
                }
                let rank = source.shape.len()
                    - indices
                        .iter()
                        .filter(|i| matches!(i, Index::Point(_)))
                        .count();
                if rank == 0 {
                    let dtype = source.elem.read_dtype().ok_or("unbound indexed element")?;
                    // Reduction slices retain a zero-rank tile view for their
                    // shaped callback ABI. A later index with no coordinates
                    // reads its scalar; the view itself is not a scalar value.
                    let scalar = expr.ty == Ty::Scalar(dtype);
                    let tile = matches!(&expr.ty, Ty::Tile(shape)
                        if shape.shape.is_empty() && shape.elem == Elem::Dtype(dtype)
                            && shape.packed_axis.is_none());
                    if !scalar && !tile {
                        return Err(format!(
                            "indexed scalar type {:?} differs from its storage {:?} at {:?}",
                            expr.ty, base.ty, expr.span,
                        ));
                    }
                } else if expr.ty.rank() != Some(rank)
                    || expr.ty.shaped().is_none_or(|s| s.elem != source.elem)
                {
                    return Err("indexed view loses storage rank or element type".into());
                }
            }
            ExprKind::Transpose(base) => {
                self.expr(base, bound)?;
                let source = base.ty.shaped().ok_or("transpose base is not shaped")?;
                let output = expr.ty.shaped().ok_or("transpose result is not shaped")?;
                if source.elem != output.elem || !source.shape.iter().rev().eq(output.shape.iter())
                {
                    return Err("transpose type does not reverse its source axes".into());
                }
            }
            ExprKind::Accessor { base, .. } | ExprKind::Lanes { base, .. } => {
                self.expr(base, bound)?
            }
            ExprKind::Unary { expr: inner, .. } => self.expr(inner, bound)?,
            ExprKind::Cast { dtype, expr: inner } => {
                self.expr(inner, bound)?;
                let actual = match &expr.ty {
                    Ty::Scalar(d) => Some(*d),
                    Ty::Tile(s) => s.elem.read_dtype(),
                    _ => None,
                };
                if actual != Some(*dtype) || inner.ty.rank() != expr.ty.rank() {
                    return Err("cast result has inconsistent type/shape".into());
                }
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs, bound)?;
                self.expr(rhs, bound)?;
            }
            ExprKind::Tuple(values) => {
                for value in values {
                    self.expr(value, bound)?;
                }
                if expr.ty != Ty::Tuple(values.iter().map(|e| e.ty.clone()).collect()) {
                    return Err("tuple type differs from its components".into());
                }
            }
            ExprKind::Call { .. } => {
                return Err("unexpanded construct call".into());
            }
            ExprKind::Intrinsic { op, args } => {
                if op.produces_owned_result() {
                    return Err(
                        "owned intrinsic result is nested without a direct destination".into(),
                    );
                }
                for argument in args {
                    self.expr(argument, bound)?;
                }
                if op
                    .writes_arguments()
                    .iter()
                    .any(|&i| args.get(i).is_none_or(|e| e.ty.shaped().is_none()))
                {
                    return Err("intrinsic write effect has no shaped storage operand".into());
                }
            }
            ExprKind::Builtin { name, args } => {
                // Callback calls retained by a coupled source reduction are
                // metadata; the selected ordinary bodies are verified above.
                for argument in args {
                    if !matches!(name, Builtin::Reduce)
                        || !matches!(argument.kind, ExprKind::Call { .. })
                    {
                        self.expr(argument, bound)?;
                    }
                }
                match name {
                    Builtin::Select => {
                        let [condition, yes, no] = args.as_slice() else {
                            return Err("eager value selection requires three arguments".into());
                        };
                        if condition.ty != Ty::Scalar(DType::Bool)
                            || !matches!(yes.ty, Ty::Scalar(_))
                            || yes.ty != no.ty
                            || expr.ty != yes.ty
                        {
                            return Err("eager value selection requires a bool and equal scalar value types".into());
                        }
                    }
                    Builtin::Load => {
                        return Err("unresolved snapshot load".into());
                    }
                    Builtin::Store => {
                        let [tile, view] = args.as_slice() else {
                            return Err("store arity differs from its contract".into());
                        };
                        let (Ty::Tile(source), Ty::Tensor(destination)) = (&tile.ty, &view.ty)
                        else {
                            return Err("store requires a tile and tensor view".into());
                        };
                        if source.shape != destination.shape || expr.ty != Ty::Void {
                            return Err(
                                "store shape or result type disagrees with its effect".into()
                            );
                        }
                    }
                    _ => {}
                }
            }
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::ShapeParam(_) => {
            }
        }
        Ok(())
    }
    fn snapshots(&self, body: &[Stmt]) -> Result<(), String> {
        for site in super::normalize::loads::sites(body) {
            if site.selected == Some(LoadMode::Borrow) && !site.can_borrow {
                return Err(format!(
                    "snapshot {} has lost its borrowing lifetime proof",
                    site.variable
                ));
            }
            if site.selected.is_none() {
                return Err("unresolved snapshot realization".into());
            }
        }
        Ok(())
    }
}

fn assignable(target: &Ty, value: &Ty) -> bool {
    match (target, value) {
        (Ty::Scalar(a), Ty::Scalar(b)) => a == b || (a.is_float() && b.is_float()),
        // Ordinary assignment is a value conversion; explicit representations
        // are handled by their typed storage operations at realization.
        (Ty::Tile(a), Ty::Tile(b)) => a.shape == b.shape,
        _ => target == value,
    }
}
fn validate_load(source: &Ty, result: &Ty, selected: bool) -> Result<(), String> {
    match (source, result) {
        (Ty::Tensor(a), Ty::Tile(b)) if a == b => Ok(()),
        // Expanded reduction operands may snapshot an existing private tile
        // view. Borrowing still requires the independent lifetime proof.
        (Ty::Tile(a), Ty::Tile(b)) if selected && a == b => Ok(()),
        (Ty::Tuple(a), Ty::Tuple(b)) if a.len() == b.len() => {
            for (a, b) in a.iter().zip(b) {
                validate_load(a, b, selected)?;
            }
            Ok(())
        }
        _ => Err("snapshot load loses its source storage type/shape".into()),
    }
}
fn validate_type(ty: &Ty) -> Result<(), String> {
    if let Some(s) = ty.shaped() {
        if s.shape
            .iter()
            .any(|n| n.as_constant().is_some_and(|n| n < 0))
        {
            return Err("negative shaped extent".into());
        }
        if matches!(s.elem, Elem::Param(_)) {
            return Err("unbound element specialization".into());
        }
        if s.packed_axis.is_some_and(|axis| axis >= s.shape.len()) {
            return Err("packed axis is outside the shaped rank".into());
        }
    }
    if let Ty::Tuple(items) = ty {
        for item in items {
            validate_type(item)?;
        }
    }
    Ok(())
}
