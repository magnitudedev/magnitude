//! One source-owned expansion into ordinary parameterized computation. The
//! expansion contains every retained alternative; no numeric value is chosen.
use super::*;
use crate::{ast::{AssignOp, BinaryOp}, reduction::structured::{self, Tree}, types::{DType, Elem, Ty}};
use std::collections::{HashMap, HashSet};
mod step;
mod stream;

#[derive(Clone)]
struct Site { occurrence: OccurrenceId, guard: Guard, span: crate::span::Span }

pub(super) fn expand(source: &ExecutionFamily) -> Result<ExecutionFamily, String> {
    let mut expansion = Expander { family: source.clone() };
    let original = expansion.family.regions.len();
    for ordinal in 0..original {
        let region = expansion.family.regions[ordinal].clone();
        let site = Site { occurrence: region.occurrence, guard: region.guard, span: match &region.kind {
            RegionKind::Reduction(reduction) => reduction.operation.span,
            RegionKind::Stream { header, .. } => header.span,
            _ => continue,
        } };
        let body = match region.kind {
            RegionKind::Reduction(reduction) => expansion.reduction(&reduction, &site)?,
            RegionKind::Stream { header, body, geometry, .. } => expansion.stream(&header, body, &geometry, &site)?,
            _ => unreachable!(),
        };
        let effects = expansion.family.regions[body.0].effects.clone();
        expansion.family.regions[ordinal].kind = RegionKind::Sequence(vec![body]);
        expansion.family.regions[ordinal].effects = effects;
    }
    for snapshot in &mut expansion.family.snapshots {
        snapshot.consumers = expansion.family.regions.iter().enumerate().filter_map(|(ordinal, region)|
            (ordinal != snapshot.producer.0 && matches!(region.kind, RegionKind::Statement(_))
                && region.effects.reads.contains(&snapshot.variable)).then_some(RegionId(ordinal))).collect();
    }
    Ok(expansion.family)
}

struct Expander { family: ExecutionFamily }
impl Expander {
    fn ordinary(&mut self, site: &Site) -> structured::Builder<'_> {
        structured::Builder { vars: &mut self.family.template.vars, span: site.span }
    }
    fn push(&mut self, site: &Site, kind: RegionKind, effects: Effects) -> RegionId {
        let id = RegionId(self.family.regions.len());
        self.family.regions.push(Region { occurrence: site.occurrence.clone(), guard: site.guard.clone(), effects, kind }); id
    }
    fn statement(&mut self, statement: Stmt, site: &Site) -> RegionId {
        let mut writes = HashSet::new(); crate::rewrite::writes(&statement, &mut writes);
        let effects = Effects { reads: (0..self.family.template.vars.len()).filter(|&id| crate::effects::uses(&statement, id)).collect(),
            writes: writes.into_iter().collect(), tensor_effect: crate::effects::tensor_effect(&statement) };
        let snapshot = match &statement.kind {
            StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), .. }, value, .. } => match &value.kind {
                ExprKind::Builtin { name: Builtin::Load, args } => args.first().map(|source| (*variable, source.clone())),
                ExprKind::Load { view, .. } => Some((*variable, (**view).clone())),
                _ => None,
            },
            _ => None,
        };
        let id = self.push(site, RegionKind::Statement(statement), effects);
        if let Some((variable, source)) = snapshot {
            self.family.snapshots.push(Snapshot { producer: id, variable, source, guard: site.guard.clone(), consumers: Vec::new() });
        }
        id
    }
    fn sequence(&mut self, children: Vec<RegionId>, site: &Site) -> RegionId {
        let mut effects = Effects::default();
        for (position, &id) in children.iter().enumerate() {
            let next = &self.family.regions[id.0].effects;
            for &previous in &children[..position] {
                let before = &self.family.regions[previous.0].effects;
                for &variable in before.writes.intersection(&next.reads).chain(before.reads.intersection(&next.writes)).chain(before.writes.intersection(&next.writes)) {
                    self.family.dependencies.push(Dependency { from: previous, to: id, kind: EdgeKind::Value(variable) });
                }
                if before.tensor_effect || next.tensor_effect { self.family.dependencies.push(Dependency { from: previous, to: id, kind: EdgeKind::Effect }); }
            }
            effects.reads.extend(&next.reads); effects.writes.extend(&next.writes); effects.tensor_effect |= next.tensor_effect;
        }
        self.push(site, RegionKind::Sequence(children), effects)
    }
    /// Sequence a linear frontier walk without constructing all pairwise
    /// dependencies between its merge blocks. The explicit-tree encoding has
    /// one predecessor edge per merge, so retaining a quadratic dependency
    /// matrix would erase the compactness of the source representation.
    fn linear_sequence(&mut self, children: Vec<RegionId>, site: &Site) -> RegionId {
        let mut effects = Effects::default();
        for &child in &children {
            let child_effects = &self.family.regions[child.0].effects;
            effects.reads.extend(&child_effects.reads);
            effects.writes.extend(&child_effects.writes);
            effects.tensor_effect |= child_effects.tensor_effect;
        }
        for pair in children.windows(2) {
            self.family.dependencies.push(Dependency { from: pair[0], to: pair[1], kind: EdgeKind::Effect });
        }
        self.push(site, RegionKind::Sequence(children), effects)
    }
    fn statements(&mut self, statements: Vec<Stmt>, site: &Site) -> RegionId {
        let children = statements.into_iter().map(|statement| self.statement(statement, site)).collect(); self.sequence(children, site)
    }
    fn branch(&mut self, decision: &DecisionId, arms: Vec<RegionId>, site: &Site) -> RegionId {
        let mut effects = Effects::default();
        for id in &arms { let next = &self.family.regions[id.0].effects; effects.reads.extend(&next.reads); effects.writes.extend(&next.writes); effects.tensor_effect |= next.tensor_effect; }
        self.push(site, RegionKind::Choice { decision: decision.clone(), arms }, effects)
    }
    fn range(&mut self, index: &Expr, lo: Sym, hi: Sym, body: RegionId, site: &Site) -> RegionId {
        let ExprKind::Var(variable) = index.kind else { unreachable!() };
        let effects = self.family.regions[body.0].effects.clone();
        let carried = effects.reads.intersection(&effects.writes).copied().filter(|id| *id != variable).collect::<BTreeSet<_>>();
        for &variable in &carried { self.family.dependencies.push(Dependency { from: body, to: body, kind: EdgeKind::Recurrence(variable) }); }
        self.push(site, RegionKind::Repeated { header: stmt(StmtKind::Range { var: variable, lo: lo.clone(), hi: hi.clone(), body: Vec::new() }, site.span), body,
            repetition: Repetition { order: RepeatOrder::Serial, indices: vec![variable], counts: vec![hi.sub(&lo)], carried } }, effects)
    }
    fn conditional(&mut self, cond: Expr, then: RegionId, els: RegionId, site: &Site) -> RegionId {
        let mut effects = self.family.regions[then.0].effects.clone(); let other = &self.family.regions[els.0].effects;
        effects.reads.extend(&other.reads); effects.writes.extend(&other.writes); effects.tensor_effect |= other.tensor_effect;
        self.push(site, RegionKind::Conditional { header: stmt(StmtKind::If { cond, then: Vec::new(), els: Vec::new() }, site.span), then, els }, effects)
    }
    fn active(&self, site: &Site, decision: &DecisionId, ordinal: usize) -> Site {
        Site { guard: site.guard.with(decision.clone(), ordinal), ..site.clone() }
    }
    fn predicate(&self, site: &Site, nonnegative: Sym) -> Result<Site, String> {
        let mut parameters = Vec::new();
        for name in nonnegative.params() {
            let decision = self.family.decisions.iter().find(|decision| decision.numeric.as_ref().is_some_and(|parameter| parameter.atom == Atom::Param(name.clone())))
                .ok_or_else(|| format!("source presence depends on runtime symbol {name}"))?;
            parameters.push((name, decision.id.clone(), decision.domain.alternatives.clone()));
        }
        let mut active = site.clone(); active.guard.predicates.push(Predicate { nonnegative, parameters }); Ok(active)
    }
    fn numeric(&self, decision: &DecisionId) -> Result<Sym, String> {
        self.family.decisions.iter().find(|entry| entry.id == *decision).and_then(|entry| entry.numeric.as_ref())
            .map(|parameter| Sym::atom(parameter.atom.clone())).ok_or_else(|| "source numeric parameter is absent".into())
    }
    fn remap_region(&mut self, id: RegionId, rename: &HashMap<VarId, VarId>, atoms: &[(Atom, Sym)], site: &Site) -> Result<RegionId, String> {
        let region = self.family.regions[id.0].clone();
        let mut active = site.clone();
        for choice in region.guard.choices { if !active.guard.choices.contains(&choice) { active.guard.choices.push(choice); } }
        for predicate in region.guard.predicates { if !active.guard.predicates.contains(&predicate) { active.guard.predicates.push(predicate); } }
        for alternatives in region.guard.one_of { if !active.guard.one_of.contains(&alternatives) { active.guard.one_of.push(alternatives); } }
        match region.kind {
            RegionKind::Statement(mut statement) => { crate::composition::remap(&mut statement, rename, atoms); Ok(self.statement(statement, &active)) },
            RegionKind::Sequence(children) => { let mut body = Vec::new(); for child in children { body.push(self.remap_region(child, rename, atoms, &active)?); } Ok(self.sequence(body, &active)) },
            RegionKind::Choice { decision, arms } => { let mut body = Vec::new(); for arm in arms { body.push(self.remap_region(arm, rename, atoms, &active)?); } Ok(self.branch(&decision, body, &active)) },
            RegionKind::Conditional { mut header, then, els } => {
                crate::composition::remap(&mut header, rename, atoms);
                let StmtKind::If { cond, .. } = header.kind else { return Err("conditional region header changed kind".into()); };
                let then = self.remap_region(then, rename, atoms, &active)?; let els = self.remap_region(els, rename, atoms, &active)?;
                Ok(self.conditional(cond, then, els, &active))
            },
            RegionKind::Repeated { mut header, body, mut repetition } => {
                crate::composition::remap(&mut header, rename, atoms);
                let body = self.remap_region(body, rename, atoms, &active)?;
                for variable in &mut repetition.indices { *variable = rename.get(variable).copied().unwrap_or(*variable); }
                for count in &mut repetition.counts { for (atom, value) in atoms { *count = count.subst(atom, value); } }
                repetition.carried = repetition.carried.into_iter().map(|variable| rename.get(&variable).copied().unwrap_or(variable)).collect();
                let effects = self.family.regions[body.0].effects.clone();
                Ok(self.push(&active, RegionKind::Repeated { header, body, repetition }, effects))
            },
            RegionKind::Replicated { index, mut count, body } => {
                for (atom, value) in atoms { count = count.subst(atom, value); }
                let body = self.remap_region(body, rename, atoms, &active)?; let effects = self.family.regions[body.0].effects.clone();
                Ok(self.push(&active, RegionKind::Replicated { index: rename.get(&index).copied().unwrap_or(index), count, body }, effects))
            },
            RegionKind::Reduction(reduction) => { let expanded = self.reduction(&reduction, &active)?; self.remap_region(expanded, rename, atoms, &active) },
            RegionKind::Stream { header, body, geometry, .. } => { let expanded = self.stream(&header, body, &geometry, &active)?; self.remap_region(expanded, rename, atoms, &active) },
        }
    }
    fn bound(&self, expression: &Sym) -> Option<i64> {
        expression.eval_interval(&|name| self.family.decisions.iter().find_map(|decision| {
            if decision.numeric.as_ref()?.atom != Atom::Param(name.to_owned()) { return None; }
            decision.domain.alternatives.numeric()?.runs().try_fold(None, |range: Option<(i64, i64)>, run| {
                let last = run.get(run.ordinal.checked_add(run.count)?.checked_sub(1)?)?;
                let next = (run.first.min(last), run.first.max(last));
                Some(Some(range.map_or(next, |(lo, hi)| (lo.min(next.0), hi.max(next.1)))))
            }).flatten()
        })).map(|(_, hi)| hi)
    }
    fn callback(&mut self, reduction: &ReductionFamily, role: CallbackRole, site: &Site) -> Result<RegionId, String> {
        if let Some(callback) = reduction.callbacks.iter().find(|callback| callback.role == role) { return Ok(callback.body); }
        let implementation = match role { CallbackRole::Merge => reduction.operation.implementation.as_ref(),
            CallbackRole::Step => reduction.operation.step.as_ref().and_then(|step| step.implementation.as_ref()) };
        Ok(self.statements(implementation.ok_or("retained callback implementation is absent")?.body.clone(), site))
    }
    fn allocations(&mut self, values: &[Expr], site: &Site) -> RegionId {
        let mut allocated = HashSet::new(); let mut statements = Vec::new();
        for value in values {
            if let ExprKind::Var(id) = value.kind { if !allocated.insert(id) { continue; } }
            statements.push(self.ordinary(site).allocate(value));
        }
        self.statements(statements, site)
    }
    fn copies(&mut self, targets: &[Expr], values: &[Expr], site: &Site) -> RegionId {
        let mut statements = Vec::new();
        for (target, value) in targets.iter().zip(values) {
            if matches!((&target.kind, &value.kind), (ExprKind::Var(a), ExprKind::Var(b)) if a == b) { continue; }
            statements.push(self.ordinary(site).copy(target, value));
        }
        self.statements(statements, site)
    }
    fn reduction(&mut self, reduction: &ReductionFamily, site: &Site) -> Result<RegionId, String> {
        let alternatives = self.family.decisions.iter().find(|entry| entry.id == reduction.tree).ok_or("retained reduction tree domain is absent")?.domain.alternatives.clone();
        let mut arms = Vec::new();
        for ordinal in 0..alternatives.len() {
            let Some(Alternative::ReductionTree(tree)) = alternatives.get(ordinal) else { return Err("invalid retained reduction tree domain".into()); };
            let active = self.active(site, &reduction.tree, ordinal);
            arms.push(match tree {
                Tree::Ordered => self.ordered_decomposition(reduction, &active)?,
                Tree::Pairwise | Tree::SeedThenPairwise => {
                    if reduction.operation.step.is_some() { self.segmented(reduction, tree, &active)? }
                    else { self.pairwise(reduction, &reduction.operation.inputs, reduction.operation.axis, reduction.operation.extent().clone(), tree == Tree::SeedThenPairwise, &active)? }
                },
                Tree::Explicit => {
                    let explicit = reduction.explicit.as_ref().ok_or("explicit reduction tree metadata is absent")?;
                    self.explicit(reduction, explicit, &active)?
                },
            });
        }
        Ok(self.branch(&reduction.tree, arms, site))
    }

    fn explicit(&mut self, reduction: &ReductionFamily, family: &ExplicitReductionFamily, site: &Site) -> Result<RegionId, String> {
        if reduction.operation.step.is_some() { return self.segmented(reduction, Tree::Explicit, site); }
        self.explicit_values(reduction, family, &reduction.operation.inputs, reduction.operation.axis, site)
    }

    fn explicit_values(&mut self, reduction: &ReductionFamily, family: &ExplicitReductionFamily, inputs: &[Expr], axis: usize, site: &Site) -> Result<RegionId, String> {
        let operation = &reduction.operation;
        let merge = operation.implementation.as_ref().ok_or("explicit reduction has no merge bindings")?.clone();
        let allocated = merge.left.iter().chain(&merge.right).chain(&merge.output).cloned().collect::<Vec<_>>();
        let mut setup = vec![self.allocations(&allocated, site)];
        let mut buffers = Vec::new();
        for state in &operation.state {
            let mut shape = state.ty.shaped().ok_or("explicit reduction state is not shaped")?.clone();
            shape.shape.insert(0, family.leaves.clone());
            let mut declarations = Vec::new();
            let buffer = self.ordinary(site).alloc(&Ty::Tile(shape), &mut declarations);
            setup.push(self.statements(declarations, site));
            let target = structured::slice(&buffer, 0, &structured::integer(0, site.span), site.span);
            setup.push(self.copies(std::slice::from_ref(&target), std::slice::from_ref(state), site));
            buffers.push(buffer);
        }
        let mut blocks = setup;
        let mut previous = Sym::constant(1);
        for (merge_index, retained) in family.merges.iter().enumerate() {
            let active = Site { occurrence: site.occurrence.clone(), guard: retained.guard.clone(), span: site.span };
            let frontier = self.numeric(&retained.frontier)?;
            let index = self.ordinary(&active).index();
            let index_sym = index.sym.clone().ok_or("explicit frontier index has no symbol")?;
            let source_index = symbol(index_sym.sub(&Sym::constant(1)), site.span);
            let destination_index = symbol(index_sym.sub(&Sym::constant(merge_index as i64)), site.span);
            let mut copies = Vec::new();
            for (buffer, input) in buffers.iter().zip(inputs) {
                let target = structured::slice(buffer, 0, &destination_index, site.span);
                let value = structured::slice(input, axis, &source_index, site.span);
                copies.push(self.ordinary(&active).copy(&target, &value));
            }
            let leaf = self.statements(copies, &active);
            let leaves = self.range(&index, previous.clone(), frontier.clone(), leaf, &active);
            let left_index = symbol(frontier.clone().sub(&Sym::constant(merge_index as i64 + 2)), site.span);
            let right_index = symbol(frontier.clone().sub(&Sym::constant(merge_index as i64 + 1)), site.span);
            let left = buffers.iter().map(|buffer| structured::slice(buffer, 0, &left_index, site.span)).collect::<Vec<_>>();
            let right = buffers.iter().map(|buffer| structured::slice(buffer, 0, &right_index, site.span)).collect::<Vec<_>>();
            let targets = buffers.iter().map(|buffer| structured::slice(buffer, 0, &left_index, site.span)).collect::<Vec<_>>();
            let merge_left = self.copies(&merge.left, &left, &active);
            let merge_right = self.copies(&merge.right, &right, &active);
            let callback = self.callback(reduction, CallbackRole::Merge, &active)?;
            let publish = self.copies(&targets, &merge.output, &active);
            let done = self.predicate(&active, Sym::constant(merge_index as i64 + 2).sub(&family.leaves))?;
            let complete = self.copies(&operation.state, &merge.output, &done);
            blocks.push(self.sequence(vec![leaves, merge_left, merge_right, callback, publish, complete], &active));
            previous = frontier;
        }
        Ok(self.linear_sequence(blocks, site))
    }
    fn ordered_decomposition(&mut self, reduction: &ReductionFamily, site: &Site) -> Result<RegionId, String> {
        if let Some(decomposition) = &reduction.decomposition {
            let whole_site = self.predicate(site, decomposition.whole_condition.clone())?;
            let split_site = self.predicate(site, decomposition.geometry.extent.sub(&decomposition.geometry.capacity).sub(&Sym::constant(1)))?;
            let whole = self.ordered(reduction, &reduction.operation, &whole_site)?;
            let mut full = vec![self.statements(decomposition.full.setup.clone(), &split_site)];
            full.push(self.ordered(reduction, &decomposition.full.operation, &split_site)?);
            let full = self.sequence(full, &split_site);
            let index = variable(decomposition.index, &self.family.template.vars);
            let full = self.range(&index, Sym::constant(0), decomposition.geometry.complete_pieces.clone(), full, &split_site);
            let tail_site = self.predicate(&split_site, decomposition.geometry.tail_extent.sub(&Sym::constant(1)))?;
            let mut tail = vec![self.statements(decomposition.tail.setup.clone(), &tail_site)];
            tail.push(self.ordered(reduction, &decomposition.tail.operation, &tail_site)?);
            let tail = self.sequence(tail, &tail_site);
            return Ok(self.sequence(vec![whole, full, tail], site));
        }
        if let Some(decomposition) = &reduction.dynamic_decomposition {
            let body = self.ordered(reduction, &decomposition.operation, site)?;
            return self.stream(&decomposition.header, body, &decomposition.geometry, site);
        }
        self.ordered(reduction, &reduction.operation, site)
    }
    fn ordered(&mut self, reduction: &ReductionFamily, operation: &structured::Reduction, site: &Site) -> Result<RegionId, String> {
        if operation.step.is_some() {
            let setup = self.step_allocations(reduction, site)?;
            let index = self.ordinary(site).index();
            let inputs = operation.inputs.iter().map(|input| structured::slice(input, operation.axis, &index, site.span)).collect::<Vec<_>>();
            let (inputs, visit) = self.step_visit(reduction, &operation.state, &inputs, site)?;
            let body = self.range(&index, Sym::constant(0), operation.extent().clone(), visit, site);
            return Ok(self.sequence(vec![setup, inputs, body], site));
        }
        let merge = operation.implementation.as_ref().ok_or("reduction has no merge bindings")?.clone();
        let allocated = merge.left.iter().chain(&merge.right).chain(&merge.output).cloned().collect::<Vec<_>>();
        let setup = self.allocations(&allocated, site);
        let index = self.ordinary(site).index();
        let right = operation.inputs.iter().map(|input| structured::slice(input, operation.axis, &index, site.span)).collect::<Vec<_>>();
        let left = self.copies(&merge.left, &operation.state, site); let right = self.copies(&merge.right, &right, site);
        let callback = self.callback(reduction, CallbackRole::Merge, site)?;
        let publish = self.copies(&operation.state, &merge.output, site);
        let visit = self.sequence(vec![left, right, callback, publish], site);
        let body = self.range(&index, Sym::constant(0), operation.extent().clone(), visit, site);
        Ok(self.sequence(vec![setup, body], site))
    }
    fn pairwise(&mut self, reduction: &ReductionFamily, inputs: &[Expr], axis: usize, extent: Sym, root_seed: bool, site: &Site) -> Result<RegionId, String> {
        let merge = reduction.operation.implementation.as_ref().ok_or("reduction has no merge bindings")?.clone();
        let allocated = merge.left.iter().chain(&merge.right).chain(&merge.output).cloned().collect::<Vec<_>>();
        let mut body = vec![self.allocations(&allocated, site)];
        let seed = i64::from(!root_seed);
        let mut count = extent.add(&Sym::constant(seed));
        let mut maximum = self.bound(&count).ok_or("pairwise family needs a finite source extent bound")?;
        let mut level = Vec::new();
        for (state, input) in reduction.operation.state.iter().zip(inputs) {
            let mut shape = state.ty.shaped().ok_or("reduction state is not shaped")?.clone(); shape.shape.insert(0, count.clone());
            let mut setup = Vec::new(); let buffer = self.ordinary(site).alloc(&Ty::Tile(shape), &mut setup);
            if !root_seed { setup.push(self.ordinary(site).copy(&structured::slice(&buffer, 0, &structured::integer(0, site.span), site.span), state)); }
            body.push(self.statements(setup, site));
            let index = self.ordinary(site).index();
            let at = symbol(index.sym.as_ref().unwrap().add(&Sym::constant(seed)), site.span);
            let target = structured::slice(&buffer, 0, &at, site.span); let value = structured::slice(input, axis, &index, site.span);
            let copy = self.ordinary(site).copy(&target, &value); let copy = self.statement(copy, site);
            body.push(self.range(&index, Sym::constant(0), extent.clone(), copy, site)); level.push(buffer);
        }
        while maximum > 1 {
            let active = self.predicate(site, count.sub(&Sym::constant(2)))?;
            let next_count = count.add(&Sym::constant(1)).quot(&Sym::constant(2));
            let mut next = Vec::new(); let mut setup = Vec::new();
            for state in &reduction.operation.state {
                let mut shape = state.ty.shaped().ok_or("reduction state is not shaped")?.clone(); shape.shape.insert(0, next_count.clone());
                next.push(self.ordinary(&active).alloc(&Ty::Tile(shape), &mut setup));
            }
            body.push(self.statements(setup, &active));
            let index = self.ordinary(&active).index();
            let even = symbol(index.sym.as_ref().unwrap().scale(2), site.span); let odd = symbol(even.sym.as_ref().unwrap().add(&Sym::constant(1)), site.span);
            let left = level.iter().map(|buffer| structured::slice(buffer, 0, &even, site.span)).collect::<Vec<_>>();
            let right = level.iter().map(|buffer| structured::slice(buffer, 0, &odd, site.span)).collect::<Vec<_>>();
            let targets = next.iter().map(|buffer| structured::slice(buffer, 0, &index, site.span)).collect::<Vec<_>>();
            let a = self.copies(&merge.left, &left, &active); let b = self.copies(&merge.right, &right, &active);
            let callback = self.callback(reduction, CallbackRole::Merge, &active)?; let publish = self.copies(&targets, &merge.output, &active);
            let visit = self.sequence(vec![a, b, callback, publish], &active);
            body.push(self.range(&index, Sym::constant(0), count.quot(&Sym::constant(2)), visit, &active));
            let tail = self.predicate(&active, count.rem(&Sym::constant(2)).sub(&Sym::constant(1)))?;
            let target = next.iter().map(|buffer| structured::slice(buffer, 0, &symbol(next_count.sub(&Sym::constant(1)), site.span), site.span)).collect::<Vec<_>>();
            let value = level.iter().map(|buffer| structured::slice(buffer, 0, &symbol(count.sub(&Sym::constant(1)), site.span), site.span)).collect::<Vec<_>>();
            body.push(self.copies(&target, &value, &tail));
            // A completed level publishes its root at its own presence guard;
            // later levels remain absent rather than reading unallocated buffers.
            let done = self.predicate(&active, Sym::constant(1).sub(&next_count))?;
            body.push(self.finish_pairwise(reduction, &next, root_seed, &done)?);
            level = next; count = next_count; maximum = (maximum + 1) / 2;
        }
        let initial_count = extent.add(&Sym::constant(seed));
        let singleton = self.predicate(site, Sym::constant(1).sub(&initial_count))?;
        // For maximum one the initial level is still current. For larger
        // domains, publish singleton inputs directly without a level read.
        if self.bound(&initial_count) == Some(1) { body.push(self.finish_pairwise(reduction, &level, root_seed, &singleton)?); }
        else if root_seed {
            let values = inputs.iter().map(|input| structured::slice(input, axis, &structured::integer(0, site.span), site.span)).collect::<Vec<_>>();
            body.push(self.merge_values(reduction, &reduction.operation.state, &values, &singleton)?);
        }
        Ok(self.sequence(body, site))
    }
    fn finish_pairwise(&mut self, reduction: &ReductionFamily, buffers: &[Expr], root_seed: bool, site: &Site) -> Result<RegionId, String> {
        let values = buffers.iter().map(|buffer| structured::slice(buffer, 0, &structured::integer(0, site.span), site.span)).collect::<Vec<_>>();
        if root_seed { self.merge_values(reduction, &reduction.operation.state, &values, site) }
        else { Ok(self.copies(&reduction.operation.state, &values, site)) }
    }
    fn merge_values(&mut self, reduction: &ReductionFamily, left: &[Expr], right: &[Expr], site: &Site) -> Result<RegionId, String> {
        let merge = reduction.operation.implementation.as_ref().ok_or("reduction has no merge bindings")?.clone();
        let left = self.copies(&merge.left, left, site); let right = self.copies(&merge.right, right, site);
        let callback = self.callback(reduction, CallbackRole::Merge, site)?;
        let publish = self.copies(&reduction.operation.state, &merge.output, site);
        Ok(self.sequence(vec![left, right, callback, publish], site))
    }
}

fn stmt(kind: StmtKind, span: crate::span::Span) -> Stmt { Stmt { id: None, span, kind } }
fn symbol(value: Sym, span: crate::span::Span) -> Expr {
    Expr { kind: ExprKind::ShapeParam(value.to_string()), ty: Ty::Scalar(DType::I32), sym: Some(value), span }
}
fn variable(id: VarId, vars: &[Var]) -> Expr {
    Expr { kind: ExprKind::Var(id), ty: vars[id].ty.clone(), sym: match &vars[id].kind { VarKind::Index(atom) => Some(Sym::atom(atom.clone())), _ => None }, span: vars[id].span }
}
fn condition(left: Sym, op: BinaryOp, right: Sym, span: crate::span::Span) -> Expr {
    Expr { kind: ExprKind::Binary { op, lhs: Box::new(symbol(left, span)), rhs: Box::new(symbol(right, span)) }, ty: Ty::Scalar(DType::Bool), sym: None, span }
}
