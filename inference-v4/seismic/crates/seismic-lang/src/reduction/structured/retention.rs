//! Private step storage may alias its left parameter only when the ordinary
//! callback establishes elementwise independence. No reduction algebra changes.
use super::Merge;
use crate::{
    ast::AssignOp,
    ir::{Expr, ExprKind, Index, StmtKind, VarId},
    types::Ty,
};
use std::collections::{HashMap, HashSet};

/// Abstract state of the ordinary element-independence proof. Retained source
/// regions use the same transfer rules and intersect facts at local choices.
#[derive(Clone)]
pub(crate) struct StateRetention {
    left: Vec<VarId>,
    right: Vec<VarId>,
    output: Vec<VarId>,
    parameters: HashSet<VarId>,
    completed: HashSet<VarId>,
}

#[derive(Clone)]
pub(crate) struct ElementRetention {
    state: VarId,
    output: VarId,
    coordinates: Vec<VarId>,
    parameters: HashSet<VarId>,
    available: HashSet<VarId>,
    conditional: HashSet<VarId>,
    written: bool,
}

impl StateRetention {
    pub(crate) fn new(implementation: &Merge) -> Option<Self> {
        let ids = |values: &[Expr]| values.iter().map(|value| match value.kind {
            ExprKind::Var(variable) => Some(variable), _ => None,
        }).collect::<Option<Vec<_>>>();
        let left = ids(&implementation.left)?;
        let right = ids(&implementation.right)?;
        let output = ids(&implementation.output)?;
        if left.is_empty() || left.len() != output.len()
            || implementation.left.iter().zip(&implementation.output).any(|(left, output)| left.ty != output.ty) { return None; }
        let parameters = left.iter().chain(&right).chain(&output).copied().collect::<HashSet<_>>();
        if parameters.len() != left.len() + right.len() + output.len() { return None; }
        Some(Self { left, right, output, parameters, completed: HashSet::new() })
    }

    pub(crate) fn begin(&self, coordinates: &[VarId], tile: &Expr) -> Option<ElementRetention> {
        let ExprKind::Var(output) = tile.kind else { return None; };
        let field = self.output.iter().position(|variable| *variable == output)?;
        if self.completed.contains(&output) || coordinates.len() != tile.ty.shaped()?.shape.len()
            || coordinates.iter().any(|variable| self.parameters.contains(variable)) { return None; }
        Some(ElementRetention { state: self.left[field], output, coordinates: coordinates.to_vec(),
            parameters: self.parameters.clone(), available: self.right.iter().chain(coordinates).copied().collect(),
            conditional: HashSet::new(), written: false })
    }

    pub(crate) fn finish(&mut self, element: &ElementRetention) -> bool {
        element.written && self.completed.insert(element.output)
    }

    pub(crate) fn merge(&mut self, other: &Self) -> bool { self.completed == other.completed }

    pub(crate) fn complete(&self) -> bool { self.completed.len() == self.output.len() }
}

impl ElementRetention {
    pub(crate) fn statement(&mut self, statement: &crate::ir::Stmt) -> bool {
        let StmtKind::Assign { target, op: AssignOp::Assign, value } = &statement.kind else { return false; };
        if self.written || !self.expression(value) { return false; }
        if point(target, self.output, &self.coordinates) { self.written = true; return true; }
        let ExprKind::Var(variable) = target.kind else { return false; };
        if !matches!(target.ty, Ty::Scalar(_)) || self.parameters.contains(&variable) || self.coordinates.contains(&variable) { return false; }
        self.available.insert(variable); true
    }

    pub(crate) fn expression(&self, expression: &Expr) -> bool {
        crate::effects::expression_can_be_omitted(expression)
            && reads(expression, self.state, &self.coordinates, &self.available)
    }

    pub(crate) fn merge(&mut self, other: &Self) -> bool {
        if self.written != other.written { return false; }
        self.conditional.extend(&other.conditional);
        self.conditional.extend(self.available.symmetric_difference(&other.available));
        self.available.retain(|variable| other.available.contains(variable)); true
    }

    pub(crate) fn conditional_read(&self, statement: &crate::ir::Stmt) -> bool {
        self.conditional.iter().any(|variable| !self.available.contains(variable) && crate::effects::uses(statement, *variable))
    }
}

impl Merge {
    pub fn can_view_operand(&self, input: usize) -> bool {
        self.right.get(input).is_some_and(|parameter| {
            parameter
                .ty
                .shaped()
                .is_some_and(|s| matches!(s.elem, crate::types::Elem::Dtype(_)))
                && !self.body.iter().any(crate::effects::tensor_effect)
                && crate::composition::parameter_read_only(parameter, &self.body)
        })
    }

    pub fn can_retain_state(&self) -> bool {
        self.retained_parameters().is_some()
    }

    pub(super) fn retain_state(&self) -> Option<Self> {
        let rename = self.retained_parameters()?;
        let mut result = self.clone();
        for statement in &mut result.body {
            crate::composition::remap(statement, &rename, &[]);
        }
        result.output = result.left.clone();
        Some(result)
    }

    fn retained_parameters(&self) -> Option<HashMap<VarId, VarId>> {
        let mut proof = StateRetention::new(self)?;
        for statement in &self.body {
            let StmtKind::Owned { vars, tile, body } = &statement.kind else { return None; };
            let mut element = proof.begin(vars, tile)?;
            for statement in body { if !element.statement(statement) { return None; } }
            if !proof.finish(&element) { return None; }
        }
        proof.complete().then(|| proof.output.into_iter().zip(proof.left).collect())
    }
}

fn point(e: &Expr, parameter: VarId, coordinates: &[VarId]) -> bool {
    let ExprKind::Index { base, indices } = &e.kind else {
        return false;
    };
    matches!(base.kind, ExprKind::Var(v) if v == parameter)
        && indices.len() == coordinates.len()
        && indices.iter().zip(coordinates).all(|(index, coordinate)| {
            matches!(index, Index::Point(e) if matches!(e.kind, ExprKind::Var(v) if v == *coordinate))
        })
}

fn reads(e: &Expr, state: VarId, coordinates: &[VarId], available: &HashSet<VarId>) -> bool {
    if point(e, state, coordinates) {
        return true;
    }
    let read = |e: &Expr| reads(e, state, coordinates, available);
    match &e.kind {
        ExprKind::Var(v) => available.contains(v),
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::ShapeParam(_) => true,
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => read(expr),
        ExprKind::Binary { lhs, rhs, .. } => read(lhs) && read(rhs),
        ExprKind::Builtin { args, .. } => args.iter().all(read),
        ExprKind::Index { base, indices } => {
            read(base)
                && indices.iter().all(|i| match i {
                    Index::Point(e) => read(e),
                    Index::Slice { start, end } => start.iter().chain(end).all(read),
                })
        }
        ExprKind::Accessor { base, .. } | ExprKind::Transpose(base) => read(base),
        _ => false,
    }
}
