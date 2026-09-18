//! Private step storage may alias its left parameter only when the ordinary
//! callback establishes elementwise independence. No reduction algebra changes.
use super::Merge;
use crate::{
    ast::AssignOp,
    ir::{Expr, ExprKind, Index, StmtKind, VarId},
    types::Ty,
};
use std::collections::{HashMap, HashSet};

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
        let ids = |values: &[Expr]| {
            values
                .iter()
                .map(|e| match e.kind {
                    ExprKind::Var(v) => Some(v),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        };
        let left = ids(&self.left)?;
        let right = ids(&self.right)?;
        let output = ids(&self.output)?;
        if left.is_empty()
            || left.len() != output.len()
            || self
                .left
                .iter()
                .zip(&self.output)
                .any(|(a, b)| a.ty != b.ty)
        {
            return None;
        }
        let parameters: HashSet<_> = left.iter().chain(&right).chain(&output).copied().collect();
        if parameters.len() != left.len() + right.len() + output.len() {
            return None;
        }
        let mut completed = HashSet::new();
        for statement in &self.body {
            let StmtKind::Owned { vars, tile, body } = &statement.kind else {
                return None;
            };
            let ExprKind::Var(out) = tile.kind else {
                return None;
            };
            let field = output.iter().position(|v| *v == out)?;
            if !completed.insert(out)
                || vars.len() != tile.ty.shaped()?.shape.len()
                || vars.iter().any(|v| parameters.contains(v))
            {
                return None;
            }
            let mut available: HashSet<_> = right.iter().chain(vars).copied().collect();
            let mut written = false;
            for statement in body {
                let StmtKind::Assign {
                    target,
                    op: AssignOp::Assign,
                    value,
                } = &statement.kind
                else {
                    return None;
                };
                // A write is last, so no temporary can observe updated state.
                if written
                    || !crate::effects::expression_can_be_omitted(value)
                    || !reads(value, left[field], vars, &available)
                {
                    return None;
                }
                if point(target, out, vars) {
                    written = true;
                } else {
                    let ExprKind::Var(v) = target.kind else {
                        return None;
                    };
                    if !matches!(target.ty, Ty::Scalar(_))
                        || parameters.contains(&v)
                        || vars.contains(&v)
                    {
                        return None;
                    }
                    available.insert(v);
                }
            }
            if !written {
                return None;
            }
        }
        (completed.len() == output.len()).then(|| output.into_iter().zip(left).collect())
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
