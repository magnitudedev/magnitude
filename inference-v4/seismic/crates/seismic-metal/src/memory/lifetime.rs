//! Conservative source lifetimes for concrete synchronous Metal allocations.
//! Loops remain structured. Views extend their backing value's lifetime; a
//! use of an outer value in a loop retains it through the loop's completion.
use seismic_lang::{ir::*, types::Ty};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    pub begin: usize,
    pub end: usize,
}
impl Interval {
    pub fn overlaps(self, other: Self) -> bool {
        self.begin <= other.end && other.begin <= self.end
    }
    fn include(&mut self, other: Self) {
        self.begin = self.begin.min(other.begin);
        self.end = self.end.max(other.end);
    }
}
#[derive(Default)]
pub(super) struct Analysis {
    sites: HashMap<OperationId, Interval>,
    uses: HashMap<VarId, Interval>,
    aliases: HashMap<VarId, HashSet<VarId>>,
    loops: Vec<Interval>,
    cursor: usize,
}
impl Analysis {
    pub fn new(body: &[Stmt]) -> Result<Self, String> {
        let mut a = Self::default();
        a.body(body)?;
        Ok(a)
    }
    fn use_at(&mut self, refs: HashSet<VarId>, interval: Interval) {
        for v in refs {
            self.uses
                .entry(v)
                .and_modify(|r| r.include(interval))
                .or_insert(interval);
        }
    }
    fn body(&mut self, body: &[Stmt]) -> Result<(), String> {
        for s in body {
            let begin = self.cursor;
            self.cursor = self.cursor.checked_add(1).ok_or("lifetime site overflow")?;
            let mut refs = HashSet::new();
            match &s.kind {
                StmtKind::Assign { target, value, .. } => {
                    expression(target, &mut refs);
                    expression(value, &mut refs);
                    // A view binding can retain an older allocation even after
                    // its original name stops appearing. Tile copies are included
                    // conservatively; this cannot invent a reuse opportunity.
                    if let ExprKind::Var(v) = target.kind {
                        if matches!(target.ty, Ty::Tile(_)) {
                            let mut source = HashSet::new();
                            expression(value, &mut source);
                            for other in source {
                                self.aliases.entry(v).or_default().insert(other);
                                self.aliases.entry(other).or_default().insert(v);
                            }
                        }
                    }
                }
                StmtKind::Expr(e) => expression(e, &mut refs),
                StmtKind::Owned { tile, body, .. } => {
                    expression(tile, &mut refs);
                    self.body(body)?;
                }
                StmtKind::LoadLoop {
                    domain,
                    views,
                    body,
                    ..
                } => {
                    expression(&domain.view, &mut refs);
                    for e in views {
                        expression(e, &mut refs);
                    }
                    self.body(body)?;
                }
                StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. }
                | StmtKind::Parallel { body, .. } => self.body(body)?,
                StmtKind::If { cond, then, els } => {
                    expression(cond, &mut refs);
                    self.body(then)?;
                    self.body(els)?;
                }
                StmtKind::Reduction(_) => {
                    return Err(
                        "structured reduction must be materialized before allocation liveness"
                            .into(),
                    );
                }
            }
            let end = self.cursor;
            self.cursor = self.cursor.checked_add(1).ok_or("lifetime site overflow")?;
            let interval = Interval { begin, end };
            let id =
                s.id.ok_or("allocation liveness requires operation identities")?;
            if self.sites.insert(id, interval).is_some() {
                return Err("duplicate lifetime operation identity".into());
            }
            self.use_at(refs, interval);
            if matches!(
                s.kind,
                StmtKind::Range { .. }
                    | StmtKind::LoadLoop { .. }
                    | StmtKind::Owned { .. }
                    | StmtKind::Lanes { .. }
            ) {
                self.loops.push(interval);
            }
        }
        Ok(())
    }
    pub fn interval(&self, operation: OperationId, variable: VarId) -> Result<Interval, String> {
        let definition = *self
            .sites
            .get(&operation)
            .ok_or("allocation site absent from lifetime source")?;
        let mut interval = definition;
        let mut visited = HashSet::new();
        let mut pending = vec![variable];
        while let Some(v) = pending.pop() {
            if !visited.insert(v) {
                continue;
            }
            if let Some(r) = self.uses.get(&v) {
                interval.include(*r);
            }
            pending.extend(self.aliases.get(&v).into_iter().flatten().copied());
        }
        // Local temporaries that are born and die in the same iteration can
        // reuse storage. A value crossing a loop boundary spans every iteration.
        loop {
            let before = interval;
            for loop_ in &self.loops {
                if interval.overlaps(*loop_)
                    && !(definition.begin > loop_.begin
                        && interval.begin > loop_.begin
                        && interval.end < loop_.end)
                {
                    interval.include(*loop_);
                }
            }
            if interval == before {
                break;
            }
        }
        Ok(interval)
    }
}
fn expression(e: &Expr, out: &mut HashSet<VarId>) {
    match &e.kind {
        ExprKind::Var(v) => {
            out.insert(*v);
        }
        ExprKind::Index { base, indices } => {
            expression(base, out);
            for i in indices {
                match i {
                    Index::Point(e) => expression(e, out),
                    Index::Slice { start, end } => {
                        for e in start.iter().chain(end) {
                            expression(e, out)
                        }
                    }
                }
            }
        }
        ExprKind::Transpose(e)
        | ExprKind::Accessor { base: e, .. }
        | ExprKind::Lanes { base: e, .. }
        | ExprKind::Unary { expr: e, .. }
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::Load { view: e, .. } => expression(e, out),
        ExprKind::Binary { lhs, rhs, .. } => {
            expression(lhs, out);
            expression(rhs, out);
        }
        ExprKind::Builtin { args, .. }
        | ExprKind::Intrinsic { args, .. }
        | ExprKind::Call { args, .. }
        | ExprKind::Tuple(args) => {
            for e in args {
                expression(e, out)
            }
        }
        _ => {}
    }
}
