//! Borrowing belongs to a load occurrence, not to a variable's total number of
//! definitions. Follow that occurrence through the structured control flow,
//! retaining enclosing branch facts until their condition variables change.
use super::{expressions, tensor_effect, tile_mutated, uses};
use crate::exec::{ir::*, types::Ty};
use crate::syntax::ast::{AssignOp, UnaryOp};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

struct Node<'a> {
    original: &'a Stmt,
    operation: Stmt,
    alias_sources: Vec<(VarId, VarId)>,
    next: Vec<usize>,
    assumptions: BTreeMap<VarId, bool>,
}

/// A shared graph for all lexical load sites in one normalized body. Loop
/// backedges describe repeated visits without unrolling or enumerating paths.
pub(crate) struct LoadLifetimes<'a> {
    nodes: Vec<Node<'a>>,
    sites: HashMap<*const Stmt, usize>,
    backing_sources: HashMap<VarId, BTreeSet<VarId>>,
    backing_targets: HashMap<VarId, BTreeSet<VarId>>,
}

// Each bit denotes a possible reaching value from the load being checked.
// VALID | INVALID at a join means that a later read must reject borrowing.
const VALID: u8 = 1;
const INVALID: u8 = 2;
#[derive(Clone, PartialEq, Eq)]
struct State {
    aliases: BTreeMap<VarId, u8>,
    assumptions: BTreeSet<VarId>,
}
impl State {
    fn join(&mut self, other: &Self) -> bool {
        let previous = self.clone();
        for (&variable, &value) in &other.aliases {
            *self.aliases.entry(variable).or_default() |= value;
        }
        self.assumptions
            .retain(|variable| other.assumptions.contains(variable));
        *self != previous
    }
}

fn condition(expression: &Expr) -> Option<(VarId, bool)> {
    match &expression.kind {
        ExprKind::Var(variable) => Some((*variable, true)),
        ExprKind::Unary {
            op: UnaryOp::Not,
            expr,
        } => condition(expr).map(|(variable, value)| (variable, !value)),
        _ => None,
    }
}
fn backing(expression: &Expr) -> Option<VarId> {
    match &expression.kind {
        ExprKind::Var(variable) => Some(*variable),
        ExprKind::Index { base, .. } | ExprKind::Transpose(base) => backing(base),
        ExprKind::Builtin {
            name: Builtin::Reshape,
            args,
        } => args.first().and_then(backing),
        _ => None,
    }
}
fn load(statement: &Stmt) -> Option<(VarId, &Expr, bool)> {
    let StmtKind::Assign {
        target: Expr {
            kind: ExprKind::Var(variable),
            ..
        },
        op: AssignOp::Assign,
        value,
    } = &statement.kind
    else {
        return None;
    };
    match &value.kind {
        ExprKind::Load { view, mode } => Some((*variable, view, *mode == LoadMode::Borrow)),
        // The upstream proof must remain valid for either downstream ownership
        // alternative, including when that load retains the same backing.
        ExprKind::Builtin {
            name: Builtin::Load,
            args,
        } if args.len() == 1 => Some((*variable, &args[0], true)),
        _ => None,
    }
}
fn rebindings(statement: &Stmt) -> Vec<VarId> {
    match &statement.kind {
        StmtKind::Assign {
            target:
                Expr {
                    kind: ExprKind::Var(variable),
                    ..
                },
            op: AssignOp::Assign,
            ..
        } => vec![*variable],
        StmtKind::LoadLoop { vars, offset, .. } => vars.iter().copied().chain(*offset).collect(),
        StmtKind::Owned { vars, .. } | StmtKind::Parallel { vars, .. } => vars.clone(),
        StmtKind::Range { var, .. } | StmtKind::Lanes { var, .. } => vec![*var],
        _ => Vec::new(),
    }
}
fn alias_sources(statement: &Stmt) -> Vec<(VarId, VarId)> {
    let mut aliases = Vec::new();
    if let Some((target, source, true)) = load(statement) {
        if let Some(source) = backing(source) {
            aliases.push((target, source));
        }
    }
    if let StmtKind::Assign {
        target: Expr {
            kind: ExprKind::Var(target),
            ..
        },
        op: AssignOp::Assign,
        value,
    } = &statement.kind
    {
        if matches!(value.ty, Ty::Tensor(_)) {
            if let Some(source) = backing(value) {
                aliases.push((*target, source));
            }
        }
    }
    if let StmtKind::LoadLoop {
        vars, views, modes, ..
    } = &statement.kind
    {
        for (ordinal, (&target, source)) in vars.iter().zip(views).enumerate() {
            if modes.as_ref().and_then(|modes| modes.get(ordinal)) != Some(&LoadMode::Materialize) {
                if let Some(source) = backing(source) {
                    aliases.push((target, source));
                }
            }
        }
    }
    aliases
}
fn shallow(statement: &Stmt) -> Stmt {
    let kind = match &statement.kind {
        StmtKind::If { cond, .. } => StmtKind::If {
            cond: cond.clone(),
            then: Vec::new(),
            els: Vec::new(),
        },
        StmtKind::Range {
            independent,
            var,
            lo,
            hi,
            ..
        } => StmtKind::Range {
            independent: *independent,
            var: *var,
            lo: lo.clone(),
            hi: hi.clone(),
            body: Vec::new(),
        },
        StmtKind::Parallel { vars, extents, .. } => StmtKind::Parallel {
            vars: vars.clone(),
            extents: extents.clone(),
            body: Vec::new(),
        },
        StmtKind::Owned { vars, tile, .. } => StmtKind::Owned {
            vars: vars.clone(),
            tile: tile.clone(),
            body: Vec::new(),
        },
        StmtKind::Lanes {
            var, extent, width, ..
        } => StmtKind::Lanes {
            var: *var,
            extent: extent.clone(),
            width: *width,
            body: Vec::new(),
        },
        StmtKind::LoadLoop {
            domain,
            offset,
            modes,
            vars,
            views,
            axes,
            piece,
            capacity,
            ..
        } => StmtKind::LoadLoop {
            domain: domain.clone(),
            offset: *offset,
            modes: modes.clone(),
            vars: vars.clone(),
            views: views.clone(),
            axes: axes.clone(),
            piece: piece.clone(),
            capacity: *capacity,
            body: Vec::new(),
        },
        _ => statement.kind.clone(),
    };
    Stmt {
        id: statement.id,
        span: statement.span,
        kind,
    }
}
fn backing_rebound(statement: &Stmt, variable: VarId) -> bool {
    match &statement.kind {
        // Rebinding a view changes the reference, not its previous backing.
        StmtKind::Assign {
            target:
                Expr {
                    kind: ExprKind::Var(target),
                    ty,
                    ..
                },
            op: AssignOp::Assign,
            value,
        } if *target == variable
            && (matches!(ty, Ty::Tensor(_))
                || matches!(
                    value.kind,
                    ExprKind::Load {
                        mode: LoadMode::Borrow,
                        ..
                    }
                )) =>
        {
            false
        }
        StmtKind::Parallel { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::Lanes { body, .. }
        | StmtKind::Owned { body, .. }
        | StmtKind::LoadLoop { body, .. } => {
            tile_mutated(&shallow(statement), variable)
                || body
                    .iter()
                    .any(|statement| backing_rebound(statement, variable))
        }
        StmtKind::If { then, els, .. } => {
            tile_mutated(&shallow(statement), variable)
                || then
                    .iter()
                    .chain(els)
                    .any(|statement| backing_rebound(statement, variable))
        }
        _ => tile_mutated(statement, variable),
    }
}
fn reads(statement: &Stmt, variable: VarId) -> bool {
    match &statement.kind {
        StmtKind::Assign {
            target: Expr {
                kind: ExprKind::Var(_),
                ..
            },
            op: AssignOp::Assign,
            value,
        } => expressions(
            value,
            &|expr| matches!(expr.kind, ExprKind::Var(id) if id == variable),
        ),
        _ => uses(statement, variable),
    }
}

impl<'a> LoadLifetimes<'a> {
    pub(crate) fn new(body: &'a [Stmt]) -> Self {
        let mut result = Self {
            nodes: Vec::new(),
            sites: HashMap::new(),
            backing_sources: HashMap::new(),
            backing_targets: HashMap::new(),
        };
        result.block(body, None, &BTreeMap::new());
        for node in &result.nodes {
            for &(target, source) in &node.alias_sources {
                result
                    .backing_sources
                    .entry(target)
                    .or_default()
                    .insert(source);
                result
                    .backing_targets
                    .entry(source)
                    .or_default()
                    .insert(target);
            }
        }
        result
    }
    fn block(
        &mut self,
        body: &'a [Stmt],
        after: Option<usize>,
        assumptions: &BTreeMap<VarId, bool>,
    ) -> Option<usize> {
        // A lexical branch fact survives only while its variable is unchanged.
        // Compute the entry facts before building reverse continuation edges.
        let mut current = assumptions.clone();
        let entries = body
            .iter()
            .map(|statement| {
                let entry = current.clone();
                current.retain(|variable, _| !tile_mutated(statement, *variable));
                entry
            })
            .collect::<Vec<_>>();
        let mut next = after;
        for (original, assumptions) in body.iter().zip(&entries).rev() {
            let id = self.nodes.len();
            let operation = shallow(original);
            let alias_sources = alias_sources(&operation);
            self.nodes.push(Node {
                original,
                operation,
                alias_sources,
                next: Vec::new(),
                assumptions: assumptions.clone(),
            });
            if load(original).is_some() {
                self.sites.insert(original as *const Stmt, id);
            }
            let successors = match &original.kind {
                StmtKind::If { cond, then, els } => {
                    let mut yes = assumptions.clone();
                    let mut no = assumptions.clone();
                    if let Some((variable, value)) = condition(cond) {
                        yes.insert(variable, value);
                        no.insert(variable, !value);
                    }
                    // Keep a slot for both branches, including an empty exit.
                    let then = self.block(then, next, &yes);
                    let els = self.block(els, next, &no);
                    vec![then.unwrap_or(usize::MAX), els.unwrap_or(usize::MAX)]
                }
                StmtKind::Parallel { body, .. } => {
                    // Parallel invocations have separate local bindings. The
                    // checked invocation contract separately establishes their
                    // memory independence; no local loan flows from one group
                    // into the next group's fresh compiler predicates/storage.
                    next.into_iter()
                        .chain(self.block(body, next, assumptions))
                        .collect()
                }
                StmtKind::Range { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Lanes { body, .. }
                | StmtKind::LoadLoop { body, .. } => {
                    let mut repeated = assumptions.clone();
                    repeated.retain(|variable, _| !tile_mutated(original, *variable));
                    next.into_iter()
                        .chain(self.block(body, Some(id), &repeated))
                        .collect()
                }
                _ => next.into_iter().collect(),
            };
            self.nodes[id].next = successors;
            next = Some(id);
        }
        next
    }
    fn backing_aliases(&self, root: VarId) -> BTreeSet<VarId> {
        // Include aliases established before this occurrence, as well as those
        // created later. Full ordinary tile assignment is a value copy and does
        // not add an alias edge. Retaining edges across redefinitions is a safe
        // overapproximation of the storage that a write can reach.
        let mut aliases = BTreeSet::from([root]);
        let mut pending = vec![root];
        while let Some(variable) = pending.pop() {
            for &alias in self.backing_sources.get(&variable).into_iter().flatten() {
                if aliases.insert(alias) {
                    pending.push(alias);
                }
            }
        }
        // Descendant bindings may borrow different sources on other visits.
        // Those redefinitions do not make the independent sources alias each
        // other. Find source ancestors first, then only follow aliases forward.
        let mut pending = aliases.iter().copied().collect::<Vec<_>>();
        while let Some(variable) = pending.pop() {
            for &alias in self.backing_targets.get(&variable).into_iter().flatten() {
                if aliases.insert(alias) {
                    pending.push(alias);
                }
            }
        }
        aliases
    }
    pub(crate) fn can_borrow(&self, statement: &Stmt) -> bool {
        let Some(&start) = self.sites.get(&(statement as *const Stmt)) else {
            return false;
        };
        let Some((variable, source, _)) = load(statement) else {
            return false;
        };
        let Some(root) = backing(source).filter(|root| *root != variable) else {
            return false;
        };
        let backing_aliases = self.backing_aliases(root);
        let assumptions = &self.nodes[start].assumptions;
        let initial = State {
            aliases: BTreeMap::from([(variable, VALID)]),
            assumptions: assumptions
                .keys()
                .copied()
                .filter(|&variable| !tile_mutated(statement, variable))
                .collect(),
        };
        let mut incoming = vec![None::<State>; self.nodes.len()];
        let mut pending = VecDeque::new();
        for &next in &self.nodes[start].next {
            if next < self.nodes.len() {
                incoming[next] = Some(initial.clone());
                pending.push_back(next);
            }
        }
        while let Some(id) = pending.pop_front() {
            let node = &self.nodes[id];
            let mut state = incoming[id].as_ref().unwrap().clone();
            let rebound = rebindings(&node.operation);
            // A loan captured from outside parallel work must survive every
            // invocation's writes, even when a use precedes the write in one
            // body. Loans created inside a body start after this boundary.
            let operation = if matches!(node.original.kind, StmtKind::Parallel { .. }) {
                node.original
            } else {
                &node.operation
            };
            let source_changed = tensor_effect(operation)
                || backing_aliases
                    .iter()
                    .any(|&root| backing_rebound(operation, root));
            if source_changed {
                for value in state.aliases.values_mut() {
                    *value = INVALID;
                }
            }
            for (&alias, &value) in &state.aliases {
                let reborrowed = matches!(&node.operation.kind, StmtKind::Assign {
                    target: Expr { kind: ExprKind::Var(target), .. }, op: AssignOp::Assign,
                    value: Expr { kind: ExprKind::Load { mode: LoadMode::Borrow, .. }, .. },
                } if *target == alias);
                if (value & INVALID != 0 && reads(&node.operation, alias))
                    // Ordinary tile reassignment writes value storage in the
                    // native realizations. Only an explicit borrowed load is
                    // a checked reference rebinding that can end this loan.
                    || (!reborrowed && tile_mutated(&node.operation, alias))
                {
                    return false;
                }
            }
            let derived = node
                .alias_sources
                .iter()
                .filter_map(|&(target, source)| {
                    state
                        .aliases
                        .get(&source)
                        .copied()
                        .map(|value| (target, value))
                })
                .collect::<Vec<_>>();
            for variable in rebound {
                state.aliases.remove(&variable);
                state.assumptions.remove(&variable);
            }
            // Unknown argument-writing operations may also invalidate a branch
            // fact. Ordinary tensor stores do not change scalar conditions.
            state
                .assumptions
                .retain(|variable| !tile_mutated(&node.operation, *variable));
            for (target, value) in derived {
                state.aliases.insert(target, value);
            }
            if state.aliases.is_empty() {
                continue;
            }
            let selected_branch = match &node.original.kind {
                StmtKind::If {
                    cond:
                        Expr {
                            kind: ExprKind::Bool(value),
                            ..
                        },
                    ..
                } => Some(usize::from(!*value)),
                StmtKind::If { cond, .. } => condition(cond).and_then(|(variable, positive)| {
                    state
                        .assumptions
                        .contains(&variable)
                        .then(|| usize::from(assumptions[&variable] != positive))
                }),
                _ => None,
            };
            for (ordinal, &next) in node.next.iter().enumerate() {
                if next >= self.nodes.len()
                    || selected_branch.is_some_and(|selected| selected != ordinal)
                {
                    continue;
                }
                let changed = match &mut incoming[next] {
                    Some(previous) => previous.join(&state),
                    entry @ None => {
                        *entry = Some(state.clone());
                        true
                    }
                };
                if changed {
                    pending.push_back(next);
                }
            }
        }
        true
    }
}
