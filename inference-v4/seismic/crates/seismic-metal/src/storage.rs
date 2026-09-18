//! IR-only legality of materialized tile placements. No source emission, native
//! compilation, device query, timing or performance ranking is involved.
use seismic_lang::{
    ir::{Expr, ExprKind, Index, Stmt, StmtKind, Var, VarId, VarKind},
    sym::{Atom, Sym},
    types::{DType, Ty},
};
use seismic_realization::dispatch::{TileDeclaration, TilePlacement};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageDecision {
    pub variable: VarId,
    pub name: String,
    pub capacity: i64,
    pub dtype: DType,
    pub intrinsic_operand: bool,
    pub cross_lane_read: bool,
    /// Complete placements supported by these ownership requirements. Capacity
    /// feasibility and native model fidelity are separate constraints.
    pub alternatives: Vec<TilePlacement>,
}

impl StorageDecision {
    /// Existing explicit diagnostic policy; this does not rank performance.
    pub fn diagnostic(&self) -> TilePlacement {
        if self.intrinsic_operand
            || (self.cross_lane_read && self.capacity > crate::execution::SUBGROUP)
        {
            TilePlacement::GroupShared
        } else if self.capacity <= crate::execution::SUBGROUP {
            TilePlacement::Replicated
        } else {
            TilePlacement::Distributed
        }
    }

    /// Resolve this ownership domain into the declaration consumed by emission
    /// and accounting. Selection does not print or compile a candidate.
    pub fn select(&self, placement: TilePlacement) -> Result<TileDeclaration, String> {
        if !self.alternatives.contains(&placement) {
            return Err(format!(
                "storage choice {placement:?} is incompatible with `{}`",
                self.name
            ));
        }
        Ok(TileDeclaration {
            symbol: self.name.clone(),
            dtype: self.dtype,
            capacity: u64::try_from(self.capacity).map_err(|_| "negative tile capacity")?,
            placement,
        })
    }
}

/// Facts belong to the analyzed body and variable identities. Transformations
/// introducing or changing variables must derive a new analysis before emission.
pub struct StorageAnalysis {
    variables: Vec<VariableStorage>,
}
struct VariableStorage {
    name: String,
    dtype: Option<DType>,
    intrinsic_operand: bool,
    cross_lane_read: bool,
}
impl StorageAnalysis {
    pub fn new(vars: &[Var], body: &[Stmt]) -> Self {
        let cross = analyze_usage(vars, body);
        let intrinsic = intrinsic_operands(body);
        Self {
            variables: vars
                .iter()
                .enumerate()
                .map(|(id, v)| VariableStorage {
                    name: v.name.clone(),
                    dtype: match &v.ty {
                        Ty::Tile(tile) => tile.elem.read_dtype(),
                        _ => None,
                    },
                    intrinsic_operand: intrinsic.contains(&id),
                    cross_lane_read: cross.contains(&id),
                })
                .collect(),
        }
    }
    pub fn decision(
        &self,
        variable: VarId,
        capacity: i64,
        dtype: DType,
    ) -> Result<StorageDecision, String> {
        if capacity < 0 {
            return Err("negative tile capacity".into());
        }
        let VariableStorage {
            name,
            dtype: expected_dtype,
            intrinsic_operand,
            cross_lane_read,
        } = self
            .variables
            .get(variable)
            .ok_or("tile variable is absent from the analyzed IR")?;
        if *expected_dtype != Some(dtype) {
            return Err(format!(
                "tile storage dtype for `{name}` does not match its checked IR type"
            ));
        }
        let mut alternatives = vec![TilePlacement::GroupShared];
        if !intrinsic_operand {
            alternatives.insert(0, TilePlacement::Replicated);
            if !cross_lane_read {
                alternatives.push(TilePlacement::Distributed);
            }
        }
        Ok(StorageDecision {
            variable,
            name: name.clone(),
            capacity,
            dtype,
            intrinsic_operand: *intrinsic_operand,
            cross_lane_read: *cross_lane_read,
            alternatives,
        })
    }
}

fn intrinsic_operands(stmts: &[Stmt]) -> std::collections::HashSet<VarId> {
    fn visit(stmts: &[Stmt], operands: &mut std::collections::HashSet<VarId>) {
        for stmt in stmts {
            match &stmt.kind {
                StmtKind::Parallel { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => visit(body, operands),
                StmtKind::If { then, els, .. } => {
                    visit(then, operands);
                    visit(els, operands);
                }
                StmtKind::Expr(expr) | StmtKind::Assign { value: expr, .. } => {
                    if let ExprKind::Intrinsic { args, .. } = &expr.kind {
                        for arg in args {
                            if let ExprKind::Var(id) = arg.kind {
                                operands.insert(id);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut operands = std::collections::HashSet::new();
    visit(stmts, &mut operands);
    operands
}

/// Tiles read outside their owning lane coordinates.
fn analyze_usage(vars: &[Var], body: &[Stmt]) -> std::collections::HashSet<VarId> {
    use std::collections::HashSet;
    let mut cross = HashSet::new();
    // Stack of enclosing owned loops: (tile var, index atoms).
    fn atoms_of(vars: &[Var], ids: &[VarId]) -> Vec<String> {
        ids.iter()
            .map(|v| match &vars[*v].kind {
                VarKind::Index(Atom::Param(p)) => p.clone(),
                _ => String::new(),
            })
            .collect()
    }
    fn same_shape(vars: &[Var], a: VarId, b: VarId) -> bool {
        match (&vars[a].ty, &vars[b].ty) {
            (Ty::Tile(x), Ty::Tile(y)) => x.shape == y.shape,
            _ => false,
        }
    }
    fn visit_expr(
        e: &Expr,
        vars: &[Var],
        owned: &[(VarId, Vec<String>)],
        cross: &mut HashSet<VarId>,
    ) {
        match &e.kind {
            ExprKind::Accessor { base, .. } => {
                // Packet coordinates differ from decoded element ownership.
                // Preserve a proven read-only packed view when lowering uses them.
                if let ExprKind::Var(v) = base.kind {
                    cross.insert(v);
                }
                visit_expr(base, vars, owned, cross);
            }
            ExprKind::Index { base, indices } => {
                // Ownership is expressed in the underlying tile's coordinates.
                // Analyze the same transposed read that emission will perform;
                // otherwise materialization may incorrectly assign private lanes.
                if let ExprKind::Transpose(inner) = &base.kind {
                    if indices.len() == 2 {
                        let mut read = e.clone();
                        read.kind = ExprKind::Index {
                            base: inner.clone(),
                            indices: vec![indices[1].clone(), indices[0].clone()],
                        };
                        visit_expr(&read, vars, owned, cross);
                        return;
                    }
                }

                if let ExprKind::Var(t) = base.kind {
                    if matches!(vars[t].ty, Ty::Tile(_)) {
                        let own = owned.last().map(|(ov, atoms)| {
                            same_shape(vars, *ov, t)
                                && indices.len() == atoms.len()
                                && indices.iter().zip(atoms).all(|(i, a)| matches!(i, Index::Point(p) if p.sym.as_ref().map(|s| *s == Sym::param(a)).unwrap_or(false)))
                        }).unwrap_or(false);
                        if !own {
                            cross.insert(t);
                        }
                    }
                }
                for i in indices {
                    match i {
                        Index::Point(p) => visit_expr(p, vars, owned, cross),
                        Index::Slice { start, end } => {
                            if let Some(x) = start {
                                visit_expr(x, vars, owned, cross)
                            }
                            if let Some(x) = end {
                                visit_expr(x, vars, owned, cross)
                            }
                        }
                    }
                }
                visit_expr(base, vars, owned, cross);
            }
            ExprKind::Load { view: x, .. }
            | ExprKind::Transpose(x)
            | ExprKind::Lanes { base: x, .. }
            | ExprKind::Unary { expr: x, .. }
            | ExprKind::Cast { expr: x, .. } => visit_expr(x, vars, owned, cross),
            ExprKind::Binary { lhs, rhs, .. } => {
                visit_expr(lhs, vars, owned, cross);
                visit_expr(rhs, vars, owned, cross);
            }
            ExprKind::Builtin { args, .. }
            | ExprKind::Intrinsic { args, .. }
            | ExprKind::Call { args, .. }
            | ExprKind::Tuple(args) => {
                for a in args {
                    visit_expr(a, vars, owned, cross);
                }
            }
            _ => {}
        }
    }
    fn visit(
        stmts: &[Stmt],
        vars: &[Var],
        owned: &mut Vec<(VarId, Vec<String>)>,
        cross: &mut HashSet<VarId>,
    ) {
        for s in stmts {
            match &s.kind {
                StmtKind::Owned {
                    vars: ids,
                    tile,
                    body,
                } => {
                    if let ExprKind::Var(t) = tile.kind {
                        owned.push((t, atoms_of(vars, ids)));
                        visit(body, vars, owned, cross);
                        owned.pop();
                    }
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => visit(body, vars, owned, cross),
                StmtKind::If { cond, then, els } => {
                    visit_expr(cond, vars, owned, cross);
                    visit(then, vars, owned, cross);
                    visit(els, vars, owned, cross);
                }
                StmtKind::Assign { target, value, .. } => {
                    if let ExprKind::Index { indices, .. } = &target.kind {
                        for i in indices {
                            if let Index::Point(p) = i {
                                visit_expr(p, vars, owned, cross);
                            }
                        }
                    }
                    visit_expr(value, vars, owned, cross);
                }
                StmtKind::Expr(e) => {
                    visit_expr(e, vars, owned, cross);
                }
            }
        }
    }
    visit(body, vars, &mut Vec::new(), &mut cross);
    cross
}

/// Selected placement contracts for materialized IR values. These are not an
/// allocation/lifetime account: repeated definitions may allocate multiple arrays.
#[derive(Clone, Debug, Default)]
pub struct StoragePlan {
    declarations: std::collections::BTreeMap<VarId, TileDeclaration>,
}
impl StoragePlan {
    pub fn declarations(&self) -> &std::collections::BTreeMap<VarId, TileDeclaration> {
        &self.declarations
    }
    pub fn declaration(&self, variable: VarId) -> Result<&TileDeclaration, String> {
        self.declarations
            .get(&variable)
            .ok_or_else(|| format!("no selected storage for IR value {variable}"))
    }
}

/// Derive materialization requests from the selected IR, then resolve their
/// placement domains. Neither discovery nor selection invokes source emission.
pub fn plan(
    vars: &[Var],
    body: &[Stmt],
    extra: &[&[Stmt]],
    select: &mut dyn FnMut(&StorageDecision) -> Result<TilePlacement, String>,
) -> Result<StoragePlan, String> {
    use seismic_lang::ir::{Builtin, LoadMode};
    use std::collections::{BTreeSet, HashMap, HashSet};
    #[derive(Default)]
    struct Requests {
        values: BTreeSet<VarId>,
        borrowed: HashSet<VarId>,
        reductions: HashSet<VarId>,
        pieces: HashMap<Atom, Sym>,
    }
    fn collect(body: &[Stmt], vars: &[Var], requests: &mut Requests) {
        for statement in body {
            match &statement.kind {
                StmtKind::LoadLoop {
                    vars: bindings,
                    modes,
                    piece,
                    capacity,
                    body,
                    ..
                } => {
                    if let Some(capacity) = capacity {
                        requests
                            .pieces
                            .insert(piece.clone(), Sym::constant(*capacity));
                    }
                    if let Some(modes) = modes {
                        for (var, mode) in bindings.iter().zip(modes) {
                            if *mode == LoadMode::Materialize {
                                requests.values.insert(*var);
                            } else {
                                requests.borrowed.insert(*var);
                            }
                        }
                    }
                    collect(body, vars, requests);
                }
                StmtKind::Assign { target, value, .. } => {
                    if let ExprKind::Var(var) = target.kind {
                        match &value.kind {
                            ExprKind::TileAlloc { .. } => {
                                requests.values.insert(var);
                            }
                            ExprKind::Load { mode, .. } => {
                                if *mode == LoadMode::Materialize {
                                    requests.values.insert(var);
                                } else {
                                    requests.borrowed.insert(var);
                                }
                            }
                            ExprKind::Var(_) if matches!(vars[var].ty, Ty::Tile(_)) => {
                                requests.values.insert(var);
                            }
                            ExprKind::Builtin {
                                name: Builtin::Reduce,
                                args,
                            } => {
                                if !matches!(args.get(2).map(|e| &e.kind), Some(ExprKind::Int(3))) {
                                    if let Some(Expr {
                                        kind: ExprKind::Var(input),
                                        ..
                                    }) = args.first()
                                    {
                                        requests.reductions.insert(*input);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => collect(body, vars, requests),
                StmtKind::If { then, els, .. } => {
                    collect(then, vars, requests);
                    collect(els, vars, requests);
                }
                _ => {}
            }
        }
    }
    let mut requests = Requests::default();
    collect(body, vars, &mut requests);
    for body in extra {
        collect(body, vars, &mut requests);
    }
    // Current sum/min/max implementations consume owned tile storage. Argmax
    // has a direct view implementation and does not require this declaration.
    requests.values.extend(
        requests
            .reductions
            .intersection(&requests.borrowed)
            .copied(),
    );
    let analysis = StorageAnalysis::new(vars, body);
    let mut result = StoragePlan::default();
    for variable in requests.values {
        let Ty::Tile(tile) = &vars[variable].ty else {
            return Err("materialized value is not a tile".into());
        };
        let capacity = tile.shape.iter().try_fold(1i64, |capacity, extent| {
            let mut extent = extent.clone();
            for (atom, value) in &requests.pieces {
                extent = extent.subst(atom, value);
            }
            let extent = extent
                .as_constant()
                .ok_or_else(|| format!("storage extent `{extent}` has no static capacity"))?;
            if extent < 0 {
                return Err("negative storage extent".to_string());
            }
            capacity
                .checked_mul(extent)
                .ok_or_else(|| "storage capacity overflow".to_string())
        })?;
        let decision = analysis.decision(
            variable,
            capacity,
            tile.elem.read_dtype().ok_or("unresolved storage dtype")?,
        )?;
        let declaration = decision.select(select(&decision)?)?;
        result.declarations.insert(variable, declaration);
    }
    Ok(result)
}
