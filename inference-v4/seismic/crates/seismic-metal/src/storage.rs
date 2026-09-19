//! IR-only legality of materialized tile placements. No source emission, native
//! compilation, device query, timing or performance ranking is involved.
mod ownership;
pub mod family;
use seismic_lang::{
    ir::{Expr, ExprKind, Index, Stmt, StmtKind, Var, VarId, VarKind},
    sym::{Atom, Sym},
    types::{DType, Ty},
};
use seismic_realization::dispatch::{TileDeclaration, TilePlacement};

/// Storage identity is unchanged by an ordinary view of a tile.
pub(crate) fn tile_root(expr: &Expr) -> Option<VarId> {
    match &expr.kind {
        ExprKind::Var(v) => Some(*v),
        ExprKind::Index { base, .. } | ExprKind::Transpose(base) => tile_root(base),
        ExprKind::Builtin {
            name: seismic_lang::ir::Builtin::Reshape,
            args,
        } => args.first().and_then(tile_root),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageDecision {
    pub variable: VarId,
    pub name: String,
    /// Construction envelope for a retained family; a selected decision uses
    /// its exact assigned capacity. Accounting reads `PhysicalLayout` instead.
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
        let placement = if self.intrinsic_operand
            || (self.cross_lane_read && self.capacity > crate::execution::SUBGROUP)
        {
            TilePlacement::GroupShared
        } else if self.capacity <= crate::execution::SUBGROUP {
            TilePlacement::Replicated
        } else {
            TilePlacement::Distributed
        };
        if self.alternatives.contains(&placement) {
            placement
        } else {
            self.alternatives
                .first()
                .cloned()
                .expect("storage domain is nonempty")
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
#[derive(Clone)]
pub struct StorageAnalysis {
    variables: Vec<VariableStorage>,
    ownership: ownership::Analysis,
}
#[derive(Clone)]
struct VariableStorage {
    name: String,
    dtype: Option<DType>,
    intrinsic_operand: bool,
    cross_lane_read: bool,
    packed: bool,
}
impl StorageAnalysis {
    pub fn new(vars: &[Var], body: &[Stmt]) -> Self {
        let cross = analyze_usage(vars, body);
        let intrinsic = intrinsic_operands(body);
        Self {
            ownership: ownership::Analysis::new(vars, body, &intrinsic),
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
                    packed:matches!(&v.ty,Ty::Tile(t) if matches!(t.elem,seismic_lang::types::Elem::Repr(_))),
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
            packed,
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
            if !self.ownership.requires_cooperation(variable) {
                alternatives.insert(0, TilePlacement::Replicated);
            }
            if !cross_lane_read && !packed {
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
                StmtKind::Reduction(r) => {
                    for body in r.bodies() {
                        visit(body, operands);
                    }
                }
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
                    if let ExprKind::Intrinsic { op, args } = &expr.kind {
                        if matches!(
                            op,
                            seismic_lang::intrinsics::Operation::MatrixLoad
                                | seismic_lang::intrinsics::Operation::MatrixLoadTranspose
                                | seismic_lang::intrinsics::Operation::MatrixStore
                        ) {
                            if let Some(root) = args.get(1).and_then(tile_root) {
                                operands.insert(root);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut operands = std::collections::HashSet::new();
    visit(stmts, &mut operands);
    fn aliases(body: &[Stmt], out: &mut Vec<(VarId, VarId)>) {
        for s in body {
            match &s.kind {
                StmtKind::LoadLoop {
                    vars,
                    views,
                    modes,
                    body,
                    ..
                } => {
                    if let Some(modes) = modes {
                        for ((var, view), mode) in vars.iter().zip(views).zip(modes) {
                            if *mode == seismic_lang::ir::LoadMode::Borrow {
                                if let Some(root) = tile_root(view) {
                                    out.push((*var, root));
                                }
                            }
                        }
                    }
                    aliases(body, out);
                }
                StmtKind::Assign { target, value, .. } => {
                    if let (
                        ExprKind::Var(var),
                        ExprKind::Load {
                            view,
                            mode: seismic_lang::ir::LoadMode::Borrow,
                        },
                    ) = (&target.kind, &value.kind)
                    {
                        if let Some(root) = tile_root(view) {
                            out.push((*var, root));
                        }
                    }
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => aliases(body, out),
                StmtKind::If { then, els, .. } => {
                    aliases(then, out);
                    aliases(els, out);
                }
                _ => {}
            }
        }
    }
    let mut borrowed = Vec::new();
    aliases(stmts, &mut borrowed);
    loop {
        let count = operands.len();
        for (view, root) in &borrowed {
            if operands.contains(view) {
                operands.insert(*root);
            }
        }
        if operands.len() == count {
            break;
        }
    }
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
    // Querying a view's shape does not read its elements. Its address expressions
    // still run, and their reads retain ordinary participant ownership rules.
    fn visit_metadata(
        e: &Expr,
        vars: &[Var],
        owned: &[(VarId, Vec<String>)],
        cross: &mut HashSet<VarId>,
    ) {
        match &e.kind {
            ExprKind::Var(_) => {}
            ExprKind::Transpose(base) => visit_metadata(base, vars, owned, cross),
            ExprKind::Index { base, indices } => {
                visit_metadata(base, vars, owned, cross);
                for index in indices {
                    match index {
                        Index::Point(point) => visit_expr(point, vars, owned, cross),
                        Index::Slice { start, end } => {
                            for endpoint in start.iter().chain(end) {
                                visit_expr(endpoint, vars, owned, cross);
                            }
                        }
                    }
                }
            }
            ExprKind::Builtin {
                name: seismic_lang::ir::Builtin::Reshape,
                args,
            } => {
                if let Some(source) = args.first() {
                    visit_metadata(source, vars, owned, cross);
                }
                for dimension in args.iter().skip(1) {
                    visit_expr(dimension, vars, owned, cross);
                }
            }
            _ => visit_expr(e, vars, owned, cross),
        }
    }
    fn visit_expr(
        e: &Expr,
        vars: &[Var],
        owned: &[(VarId, Vec<String>)],
        cross: &mut HashSet<VarId>,
    ) {
        match &e.kind {
            ExprKind::Builtin {
                name: seismic_lang::ir::Builtin::Extent,
                args,
            } => {
                if let Some(view) = args.first() {
                    visit_metadata(view, vars, owned, cross);
                }
                for axis in args.iter().skip(1) {
                    visit_expr(axis, vars, owned, cross);
                }
            }
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
                StmtKind::Reduction(r) => {
                    for e in r.operands() {
                        visit_expr(e, vars, owned, cross);
                    }
                    for body in r.bodies() {
                        visit(body, vars, owned, cross);
                    }
                }
                StmtKind::Owned {
                    vars: ids,
                    tile,
                    body,
                } => {
                    if let ExprKind::Var(t) = tile.kind {
                        owned.push((t, atoms_of(vars, ids)));
                        visit(body, vars, owned, cross);
                        owned.pop();
                    } else {
                        // A sliced domain retains its root allocation. Until a
                        // distributed coordinate cover is selected, require an
                        // addressable shared or replicated root.
                        if let Some(t) = tile_root(tile) {
                            cross.insert(t);
                        }
                        visit_expr(tile, vars, owned, cross);
                        visit(body, vars, owned, cross);
                    }
                }
                StmtKind::LoadLoop { views, body, .. } => {
                    for view in views {
                        if let Some(root) = tile_root(view) {
                            cross.insert(root);
                        }
                        visit_expr(view, vars, owned, cross);
                    }
                    visit(body, vars, owned, cross);
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => visit(body, vars, owned, cross),
                StmtKind::If { cond, then, els } => {
                    visit_expr(cond, vars, owned, cross);
                    visit(then, vars, owned, cross);
                    visit(els, vars, owned, cross);
                }
                StmtKind::Assign { target, value, .. } => {
                    // A logical view copy may permute or slice coordinates. Its
                    // source needs addressable storage unless a matching lane
                    // transfer cover is selected by this analysis.
                    if matches!(target.ty, Ty::Tile(_)) && !matches!(value.kind, ExprKind::Var(_)) {
                        if let Some(root) = tile_root(value) {
                            cross.insert(root);
                        }
                    }
                    visit_expr(target, vars, owned, cross);
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalLayout {
    /// Allocation extents after bounding invocation indices, retaining the
    /// original compile-time numeric decisions exactly.
    pub shape: Vec<Sym>,
    pub strides: Vec<Sym>,
    pub capacity: Sym,
    pub packets: Option<PacketLayout>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PacketLayout {
    pub physical_width: Sym,
    pub strides: Vec<Sym>,
    pub planes: Vec<PacketPlane>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PacketPlane {
    pub plane: seismic_lang::repr::Plane,
    pub elements_per_row: Sym,
    pub elements: Sym,
}
fn row_strides(shape: &[Sym]) -> Vec<Sym> {
    let mut strides = vec![Sym::constant(1); shape.len()];
    let mut stride = Sym::constant(1);
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride.clone();
        stride = stride.mul(&shape[axis]);
    }
    strides
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StoragePlan {
    owned_cooperative: std::collections::BTreeMap<VarId, bool>,
    declarations: std::collections::BTreeMap<VarId, TileDeclaration>,
    data_variables: std::collections::HashSet<VarId>,
    bounds: seismic_lang::sym::Facts,
    packets: std::collections::BTreeMap<VarId, seismic_lang::repr::SnapshotLayout>,
    physical: std::collections::BTreeMap<VarId, PhysicalLayout>,
    parameters: std::collections::BTreeMap<String, (i64, i64)>,
}
impl StoragePlan {
    pub fn owned_cooperative(&self, variable: VarId) -> bool { self.owned_cooperative.get(&variable).copied().unwrap_or(false) }
    pub fn requires_data(&self, variable: VarId) -> bool {
        self.data_variables.contains(&variable)
    }
    pub fn packets(&self, variable: VarId) -> Option<&seismic_lang::repr::SnapshotLayout> {
        self.packets.get(&variable)
    }
    pub fn capacity(&self, extent: &Sym) -> Result<i64, String> {
        self.capacity_expression(extent)
            .eval_interval(&|name| self.parameters.get(name).copied())
            .map(|(_, hi)| hi)
            .filter(|n| *n >= 0)
            .ok_or_else(|| format!("storage extent `{extent}` has no static capacity"))
    }
    /// Bound only invocation-varying coordinates. Quotients and remainders of
    /// original compile-time decisions remain exact physical geometry.
    pub fn capacity_expression(&self, extent: &Sym) -> Sym {
        fn compile_time(atom: &Atom, parameters: &std::collections::BTreeMap<String, (i64, i64)>) -> bool {
            match atom {
                Atom::Param(name) => parameters.contains_key(name),
                Atom::Quot(n, d) | Atom::Rem(n, d) => n.atoms().iter().chain(d.atoms().iter())
                    .all(|atom| compile_time(atom, parameters)),
            }
        }
        seismic_lang::sym::Prover::new(&self.bounds)
            .interval_over(extent, &|atom| !compile_time(atom, &self.parameters)).hi
    }
    pub fn physical(&self, variable: VarId) -> Option<&PhysicalLayout> {
        self.physical.get(&variable)
    }
    pub fn with_declaration(mut self, variable: VarId, declaration: TileDeclaration) -> Self {
        self.declarations.insert(variable, declaration);
        self
    }
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
    StorageFamily::derive(vars, body, extra)?.select(select)
}

/// All materialization requests and ownership relations, before any placement
/// is selected. The same owner supplies concrete and shared-model realization.
#[derive(Clone)]
pub struct StorageFamily {
    base: StoragePlan,
    analysis: StorageAnalysis,
    decisions: Vec<StorageDecision>,
    parameters: std::collections::BTreeMap<String, seismic_accounting::algebra::Value>,
}
impl StorageFamily {
    pub fn decisions(&self) -> &[StorageDecision] { &self.decisions }
    pub fn physical(&self, variable: VarId) -> Option<&PhysicalLayout> { self.base.physical(variable) }
    /// Typed union metadata for retained emission. Placements and envelope
    /// capacities here are not a selected execution or an allocation account.
    pub(crate) fn layout_template(&self) -> Result<StoragePlan, String> {
        let mut result = self.base.clone();
        for decision in &self.decisions {
            let placement = decision.alternatives.iter().find(|placement| **placement == TilePlacement::GroupShared)
                .or_else(|| decision.alternatives.first()).ok_or("empty storage placement domain")?;
            result.declarations.insert(decision.variable, decision.select(placement.clone())?);
        }
        Ok(result)
    }
    pub(crate) fn force_replicated(&mut self, variables: &[VarId]) -> Result<(), String> {
        for decision in &mut self.decisions {
            if variables.contains(&decision.variable) {
                decision.alternatives.retain(|placement| *placement == TilePlacement::Replicated);
                if decision.alternatives.is_empty() { return Err("private fold storage has no replicated realization".into()); }
            }
        }
        Ok(())
    }
    pub fn select(&self, select: &mut dyn FnMut(&StorageDecision) -> Result<TilePlacement, String>) -> Result<StoragePlan, String> {
        if self.parameters.values().any(|value| { let (lo, hi) = value.bounds(); lo != hi }) {
            return Err("parameterized storage requires its original numeric assignment".into());
        }
        let parameters = self.parameters.iter().map(|(name, value)| {
            i64::try_from(value.bounds().0).map(|value| (name.clone(), value)).map_err(|_| "storage parameter exceeds i64".to_string())
        }).collect::<Result<std::collections::BTreeMap<_, _>, _>>()?;
        self.select_parameters(&parameters, select)
    }
    fn select_parameters(&self, parameters: &std::collections::BTreeMap<String, i64>, select: &mut dyn FnMut(&StorageDecision) -> Result<TilePlacement, String>) -> Result<StoragePlan, String> {
        let mut result = self.base.clone();
        for (name, &(lo, hi)) in &self.base.parameters {
            let value = *parameters.get(name).ok_or_else(|| format!("missing storage parameter `{name}`"))?;
            if !(lo..=hi).contains(&value) { return Err(format!("storage parameter `{name}` is outside its original domain")); }
            result.parameters.insert(name.clone(), (value, value));
        }
        let mut ownership = self.analysis.ownership.selection();
        for original in &self.decisions {
            let mut decision = original.clone();
            let physical = &result.physical[&decision.variable];
            let evaluate = |expression: &Sym| expression.eval(&|name| parameters.get(name).copied())
                .filter(|value| *value >= 0).ok_or_else(|| format!("unresolved or negative selected storage extent `{expression}`"));
            decision.capacity = evaluate(&physical.capacity)?;
            if let Some(packet) = &physical.packets {
                let mut planes = Vec::new();
                for part in &packet.planes {
                    planes.push(seismic_lang::repr::SnapshotPlane { plane: part.plane.clone(),
                        elements_per_row: evaluate(&part.elements_per_row)? as u64, elements: evaluate(&part.elements)? as u64 });
                }
                result.packets.insert(decision.variable, seismic_lang::repr::SnapshotLayout {
                    physical_width: evaluate(&packet.physical_width)? as u64,
                    strides: packet.strides.iter().map(|s| evaluate(s).map(|n| n as u64)).collect::<Result<_, _>>()?, planes });
            }
            ownership.restrict(&mut decision)?;
            let declaration = decision.select(select(&decision)?)?;
            ownership.select(decision.variable, &declaration.placement)?;
            result.declarations.insert(decision.variable, declaration);
        }
        result.owned_cooperative = ownership.owners();
        Ok(result)
    }
    pub fn derive(vars: &[Var], body: &[Stmt], extra: &[&[Stmt]]) -> Result<Self, String> {
        Self::derive_parameterized(vars, body, extra, &std::collections::BTreeMap::new())
    }
    pub fn derive_parameterized(vars: &[Var], body: &[Stmt], extra: &[&[Stmt]],
        parameters: &std::collections::BTreeMap<String, seismic_accounting::algebra::Value>) -> Result<Self, String> {
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
                            ExprKind::Var(_)
                            | ExprKind::Index { .. }
                            | ExprKind::Transpose(_)
                            | ExprKind::Builtin {
                                name: Builtin::Reshape,
                                ..
                            } if matches!(vars[var].ty, Ty::Tile(_)) => {
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
    let mut result = StoragePlan::default();
    result.parameters = parameters.iter().map(|(name, value)| {
        let (lo, hi) = value.bounds();
        Ok((name.clone(), (i64::try_from(lo).map_err(|_| "storage parameter range exceeds i64")?,
            i64::try_from(hi).map_err(|_| "storage parameter range exceeds i64")?)))
    }).collect::<Result<_, String>>()?;
    let demand_body = body.iter().chain(extra.iter().flat_map(|body| body.iter()))
        .cloned().collect::<Vec<_>>();
    result.data_variables = seismic_lang::demand::data_variables(&demand_body);
    requests.values.retain(|variable| result.requires_data(*variable));
    // A logical slice keeps its invocation extent; its backing axis supplies a
    // finite allocation bound. The same symbolic facts feed emission and
    // reduction geometry, rather than substituting guessed piece sizes.
    fn view_axis_bound(e: &Expr, axis: usize, parameters: &std::collections::BTreeMap<String, (i64, i64)>) -> Option<Sym> {
        let shaped = e.ty.shaped()?;
        let extent = shaped.shape.get(axis)?;
        if extent.eval_interval(&|name| parameters.get(name).copied()).is_some() { return Some(extent.clone()); }
        match &e.kind {
            ExprKind::Index { base, indices } => {
                let rank = base.ty.shaped()?.shape.len();
                let axis = (0..rank).filter(|axis| !matches!(indices.get(*axis), Some(Index::Point(_)))).nth(axis)?;
                view_axis_bound(base, axis, parameters)
            }
            ExprKind::Transpose(base) => view_axis_bound(base, shaped.shape.len().checked_sub(axis + 1)?, parameters),
            ExprKind::Load { view, .. } => view_axis_bound(view, axis, parameters),
            _ => None,
        }
    }
    fn expression_bounds(e: &Expr, facts: &mut seismic_lang::sym::Facts, parameters: &std::collections::BTreeMap<String, (i64, i64)>) {
        if let Ty::Tensor(shaped) | Ty::Tile(shaped) = &e.ty {
            for (axis, extent) in shaped.shape.iter().enumerate() {
                if let [atom] = extent.atoms().as_slice() {
                    if *extent == Sym::atom(atom.clone()) {
                        if let Some(bound) = view_axis_bound(e, axis, parameters) {
                            if bound != *extent {
                                let bound = match (facts.upper_of(atom).and_then(|b| b.as_constant()), bound.as_constant()) {
                                    (Some(previous), Some(current)) => Sym::constant(previous.max(current)),
                                    _ => bound,
                                };
                                facts.set_range(atom.clone(), Sym::constant(0), bound);
                            }
                        }
                    }
                }
            }
        }
        match &e.kind {
            ExprKind::Index { base, indices } => {
                expression_bounds(base, facts, parameters);
                for i in indices {
                    match i {
                        Index::Point(e) => expression_bounds(e, facts, parameters),
                        Index::Slice { start, end } => {
                            for e in start.iter().chain(end) {
                                expression_bounds(e, facts, parameters);
                            }
                        }
                    }
                }
            }
            ExprKind::Load { view: e, .. }
            | ExprKind::Transpose(e)
            | ExprKind::Unary { expr: e, .. }
            | ExprKind::Cast { expr: e, .. }
            | ExprKind::Accessor { base: e, .. }
            | ExprKind::Lanes { base: e, .. } => expression_bounds(e, facts, parameters),
            ExprKind::Binary { lhs, rhs, .. } => {
                expression_bounds(lhs, facts, parameters);
                expression_bounds(rhs, facts, parameters);
            }
            ExprKind::Builtin { args, .. }
            | ExprKind::Intrinsic { args, .. }
            | ExprKind::Call { args, .. }
            | ExprKind::Tuple(args) => {
                for e in args {
                    expression_bounds(e, facts, parameters);
                }
            }
            _ => {}
        }
    }
    fn body_bounds(body: &[Stmt], facts: &mut seismic_lang::sym::Facts, parameters: &std::collections::BTreeMap<String, (i64, i64)>) {
        for s in body {
            match &s.kind {
                StmtKind::Assign { target, value, .. } => {
                    expression_bounds(target, facts, parameters);
                    expression_bounds(value, facts, parameters);
                }
                StmtKind::Expr(e) => expression_bounds(e, facts, parameters),
                StmtKind::Owned { tile, body, .. } => {
                    expression_bounds(tile, facts, parameters);
                    body_bounds(body, facts, parameters);
                }
                StmtKind::LoadLoop {
                    domain,
                    views,
                    body,
                    ..
                } => {
                    expression_bounds(&domain.view, facts, parameters);
                    for e in views {
                        expression_bounds(e, facts, parameters);
                    }
                    body_bounds(body, facts, parameters);
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => body_bounds(body, facts, parameters),
                StmtKind::If { cond, then, els } => {
                    expression_bounds(cond, facts, parameters);
                    body_bounds(then, facts, parameters);
                    body_bounds(els, facts, parameters);
                }
                StmtKind::Reduction(_) => {
                    unreachable!("reductions materialize before storage planning")
                }
            }
        }
    }
    body_bounds(body, &mut result.bounds, &result.parameters);
    for body in extra {
        body_bounds(body, &mut result.bounds, &result.parameters);
    }
    for (atom, bound) in &requests.pieces {
        result
            .bounds
            .set_range(atom.clone(), Sym::constant(0), bound.clone());
    }
    let analysis = StorageAnalysis::new(vars, &demand_body);
    let mut decisions = Vec::new();
    for variable in requests.values {
        let Ty::Tile(tile) = &vars[variable].ty else {
            return Err("materialized value is not a tile".into());
        };
        let shape = tile.shape.iter().map(|extent| result.capacity_expression(extent)).collect::<Vec<_>>();
        let capacity_expression = shape.iter().fold(Sym::constant(1), |product, extent| product.mul(extent));
        let capacity = tile.shape.iter().try_fold(1i64, |capacity, extent| {
            let extent = result.capacity(extent)?;
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
        let mut physical = PhysicalLayout { strides: row_strides(&shape), shape,
            capacity: capacity_expression, packets: None };
        if let seismic_lang::types::Elem::Repr(name) = &tile.elem {
            let capacities = tile
                .shape
                .iter()
                .map(|e| {
                    result
                        .capacity(e)
                        .and_then(|n| u64::try_from(n).map_err(|_| "negative packet extent".into()))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let repr = seismic_lang::repr::lookup(name).ok_or("unknown packet representation")?;
            let layout = repr
                .snapshot_layout(&capacities)
                .ok_or("packet snapshot capacity overflow")?;
            result.packets.insert(variable, layout);
            let (width, outer) = physical.shape.split_last().ok_or("packet storage has no row axis")?;
            let group = i64::from(repr.storage_group());
            // Complete physical rows own every logical prefix. The symbolic
            // ceil equation also preserves the zero-width case (`0 -> 0`).
            let physical_width = width.add(&Sym::constant(group - 1)).quot(&Sym::constant(group)).scale(group);
            let rows = outer.iter().fold(Sym::constant(1), |product, extent| product.mul(extent));
            let mut packet_shape = outer.to_vec(); packet_shape.push(physical_width.clone());
            let planes = repr.planes().into_iter().map(|plane| {
                let elements_per_row = plane.extent(&physical_width);
                PacketPlane { elements: rows.mul(&elements_per_row), elements_per_row, plane }
            }).collect();
            physical.packets = Some(PacketLayout { physical_width, strides: row_strides(&packet_shape), planes });
        }
        result.physical.insert(variable, physical);
        decisions.push(decision);
    }
    Ok(Self { base: result, analysis, decisions, parameters: parameters.clone() })
    }
}
