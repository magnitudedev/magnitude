//! Relationships between selected tile storage and the loops that publish it.
//! An owned loop uses either all logical elements per lane or one cooperative
//! partition. Its element-write destinations must use the same replication
//! class. Distributed reads additionally require a cooperative owning loop.
use super::*;
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
struct Components(Vec<usize>);
impl Components {
    fn new(count: usize) -> Self {
        Self((0..count).collect())
    }
    fn root(&self, mut value: usize) -> usize {
        while self.0[value] != value {
            value = self.0[value];
        }
        value
    }
    fn join(&mut self, a: usize, b: usize) {
        let a = self.root(a);
        let b = self.root(b);
        self.0[a.max(b)] = a.min(b);
    }
}
#[derive(Clone)]
pub(super) struct Analysis {
    aliases: Components,
    groups: Components,
    owners: HashSet<VarId>,
    read_owners: HashMap<VarId, HashSet<VarId>>,
    cooperative: HashSet<usize>,
}
impl Analysis {
    pub fn new(vars: &[Var], body: &[Stmt], intrinsic: &HashSet<VarId>) -> Self {
        let mut aliases = Components::new(vars.len());
        fn alias(body: &[Stmt], aliases: &mut Components) {
            for s in body {
                match &s.kind {
                    StmtKind::Assign {
                        target:
                            Expr {
                                kind: ExprKind::Var(target),
                                ..
                            },
                        value:
                            Expr {
                                kind:
                                    ExprKind::Load {
                                        view,
                                        mode: seismic_lang::ir::LoadMode::Borrow,
                                    },
                                ..
                            },
                        ..
                    } => {
                        if let Some(root) = tile_root(view) {
                            aliases.join(*target, root);
                        }
                    }
                    StmtKind::LoadLoop {
                        vars,
                        views,
                        modes: Some(modes),
                        body,
                        ..
                    } => {
                        for ((target, view), mode) in vars.iter().zip(views).zip(modes) {
                            if *mode == seismic_lang::ir::LoadMode::Borrow {
                                if let Some(root) = tile_root(view) {
                                    aliases.join(*target, root);
                                }
                            }
                        }
                        alias(body, aliases);
                    }
                    StmtKind::Owned { body, .. }
                    | StmtKind::Range { body, .. }
                    | StmtKind::Parallel { body, .. }
                    | StmtKind::LoadLoop { body, .. }
                    | StmtKind::Lanes { body, .. } => alias(body, aliases),
                    StmtKind::If { then, els, .. } => {
                        alias(then, aliases);
                        alias(els, aliases);
                    }
                    _ => {}
                }
            }
        }
        alias(body, &mut aliases);
        let mut groups = Components::new(vars.len());
        for v in 0..vars.len() {
            let root = aliases.root(v);
            // Borrowed views of device storage share their backing, but not
            // their owned-loop partition. Tile aliases share a selected array.
            if matches!(vars[root].ty, Ty::Tile(_)) {
                groups.join(v, root);
            }
        }
        let mut result = Self {
            groups,
            aliases,
            owners: HashSet::new(),
            read_owners: HashMap::new(),
            cooperative: HashSet::new(),
        };
        fn expression(e: &Expr, owner: Option<VarId>, vars: &[Var], analysis: &mut Analysis) {
            match &e.kind {
                ExprKind::Index { base, indices } => {
                    if let (Some(owner), Some(source)) = (owner, tile_root(base)) {
                        if matches!(vars[source].ty, Ty::Tile(_)) {
                            analysis
                                .read_owners
                                .entry(analysis.aliases.root(source))
                                .or_default()
                                .insert(owner);
                        }
                    }
                    expression(base, owner, vars, analysis);
                    for index in indices {
                        match index {
                            Index::Point(e) => expression(e, owner, vars, analysis),
                            Index::Slice { start, end } => {
                                for e in start.iter().chain(end) {
                                    expression(e, owner, vars, analysis);
                                }
                            }
                        }
                    }
                }
                ExprKind::Load { view: e, .. }
                | ExprKind::Transpose(e)
                | ExprKind::Lanes { base: e, .. }
                | ExprKind::Unary { expr: e, .. }
                | ExprKind::Cast { expr: e, .. }
                | ExprKind::Accessor { base: e, .. } => expression(e, owner, vars, analysis),
                ExprKind::Binary { lhs, rhs, .. } => {
                    expression(lhs, owner, vars, analysis);
                    expression(rhs, owner, vars, analysis);
                }
                ExprKind::Call { args, .. }
                | ExprKind::Builtin { args, .. }
                | ExprKind::Intrinsic { args, .. }
                | ExprKind::Tuple(args) => {
                    for e in args {
                        expression(e, owner, vars, analysis);
                    }
                }
                _ => {}
            }
        }
        fn visit(body: &[Stmt], owner: Option<VarId>, vars: &[Var], analysis: &mut Analysis) {
            for s in body {
                match &s.kind {
                    StmtKind::Owned { tile, body, .. } => {
                        let inner = tile_root(tile);
                        if let Some(inner) = inner {
                            analysis.owners.insert(inner);
                        }
                        expression(tile, owner, vars, analysis);
                        visit(body, inner, vars, analysis);
                    }
                    StmtKind::Assign { target, value, .. } => {
                        if matches!(target.kind, ExprKind::Index { .. }) {
                            if let (Some(owner), Some(target)) = (owner, tile_root(target)) {
                                if matches!(vars[target].ty, Ty::Tile(_)) {
                                    analysis.groups.join(owner, target);
                                }
                            }
                        }
                        expression(target, owner, vars, analysis);
                        expression(value, owner, vars, analysis);
                    }
                    StmtKind::Expr(e) => expression(e, owner, vars, analysis),
                    StmtKind::If { cond, then, els } => {
                        expression(cond, owner, vars, analysis);
                        visit(then, owner, vars, analysis);
                        visit(els, owner, vars, analysis);
                    }
                    StmtKind::Range { body, .. }
                    | StmtKind::Parallel { body, .. }
                    | StmtKind::LoadLoop { body, .. }
                    | StmtKind::Lanes { body, .. } => visit(body, owner, vars, analysis),
                    StmtKind::Reduction(r) => {
                        for body in r.bodies() {
                            visit(body, owner, vars, analysis);
                        }
                    }
                }
            }
        }
        visit(body, None, vars, &mut result);
        result.cooperative = intrinsic.iter().map(|&v| result.groups.root(v)).collect();
        result
    }
    /// Semantic classes shared by concrete selection and symbolic constraints.
    pub fn group(&self, variable: VarId) -> usize { self.groups.root(variable) }
    pub fn groups(&self) -> Vec<usize> {
        let mut result = (0..self.groups.0.len()).map(|v| self.groups.root(v)).collect::<Vec<_>>();
        result.sort_unstable(); result.dedup(); result
    }
    pub fn owner_variables(&self) -> impl Iterator<Item = VarId> + '_ { self.owners.iter().copied() }
    pub fn read_owner_groups(&self, variable: VarId) -> Vec<usize> {
        let mut result = self.read_owners.get(&self.aliases.root(variable)).into_iter().flatten()
            .map(|&owner| self.groups.root(owner)).collect::<Vec<_>>();
        result.sort_unstable(); result.dedup(); result
    }
    pub fn forced_groups(&self) -> impl Iterator<Item = usize> + '_ { self.cooperative.iter().copied() }
    pub fn requires_cooperation(&self, variable: VarId) -> bool {
        self.cooperative.contains(&self.groups.root(variable))
    }
    pub fn selection(&self) -> Selection<'_> {
        Selection {
            analysis: self,
            cooperative: self.cooperative.iter().map(|&g| (g, true)).collect(),
        }
    }
}
pub(super) struct Selection<'a> {
    analysis: &'a Analysis,
    cooperative: HashMap<usize, bool>,
}
impl Selection<'_> {
    pub fn owners(&self) -> std::collections::BTreeMap<VarId, bool> {
        (0..self.analysis.aliases.0.len())
            .filter(|&v| self.analysis.owners.contains(&v))
            .map(|v| (v, self.class(v).unwrap_or(false)))
            .collect()
    }

    fn class(&self, variable: VarId) -> Option<bool> {
        self.cooperative
            .get(&self.analysis.groups.root(variable))
            .copied()
    }
    fn require(&mut self, variable: VarId, cooperative: bool) -> Result<(), String> {
        let group = self.analysis.groups.root(variable);
        if self
            .cooperative
            .get(&group)
            .is_some_and(|&old| old != cooperative)
        {
            return Err("tile publication requires incompatible participant ownership".into());
        }
        self.cooperative.insert(group, cooperative);
        Ok(())
    }
    pub fn restrict(&self, decision: &mut StorageDecision) -> Result<(), String> {
        let variable = decision.variable;
        let read_owners = self
            .analysis
            .read_owners
            .get(&self.analysis.aliases.root(variable));
        decision.alternatives.retain(|placement| {
            self.class(variable)
                .is_none_or(|cooperative| cooperative == (*placement != TilePlacement::Replicated))
                && (*placement != TilePlacement::Distributed
                    || read_owners.is_none_or(|owners| {
                        owners.iter().all(|&owner| self.class(owner) != Some(false))
                    }))
        });
        if decision.alternatives.is_empty() {
            return Err(format!(
                "no compatible participant ownership for `{}`",
                decision.name
            ));
        }
        Ok(())
    }
    pub fn select(&mut self, variable: VarId, placement: &TilePlacement) -> Result<(), String> {
        self.require(variable, *placement != TilePlacement::Replicated)?;
        if *placement == TilePlacement::Distributed {
            if let Some(owners) = self
                .analysis
                .read_owners
                .get(&self.analysis.aliases.root(variable))
            {
                for &owner in owners {
                    self.require(owner, true)?;
                }
            }
        }
        Ok(())
    }
}
