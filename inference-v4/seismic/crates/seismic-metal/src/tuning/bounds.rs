//! Necessary work retained before choosing a Metal decomposition. These are
//! projections of ordinary store/dispatch lowering, never candidate scores.
use crate::{
    execution::SUBGROUP,
    model,
    terminal::{Primitive, Space, Type},
};
use seismic_accounting::schedule::Demand;
use seismic_accounting::selection::{Choices, Domain, IntegerRange};
use seismic_lang::{
    ir::{Builtin, Expr, ExprKind, Stmt, StmtKind, VarKind},
    lowered_ir::LoweredIr,
    types::{DType, Elem, Ty},
};
use seismic_realization::dispatch::WorkMapping;

const TYPES: [DType; 6] = [
    DType::F32,
    DType::BF16,
    DType::F16,
    DType::I32,
    DType::U32,
    DType::Bool,
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Remaining {
    Partition,
    Mapping { phase: usize, extent: u64 },
    Split { phases: Vec<usize> },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Dispatch {
    items: Vec<u64>,
    remaining: Remaining,
}
impl Dispatch {
    /// Previously selected axes retain their work-item counts. Each unresolved
    /// axis may put its full extent in one item, which is an optimistic envelope
    /// over the complete supported mapping domain rather than a chosen mapping.
    pub(super) fn derive(
        function: &LoweredIr,
        mappings: &[WorkMapping],
        steps: &[u64],
        remaining: Remaining,
    ) -> Result<Self, String> {
        let mut items = Vec::new();
        for (phase, statement) in function.body.iter().enumerate() {
            if let Some(mapping) = mappings.get(phase) {
                items.push(mapping.work_items());
                continue;
            }
            let StmtKind::Parallel { extents, .. } = &statement.kind else {
                return Err("dispatch bound requires normalized parallel phases".into());
            };
            let count = extents
                .iter()
                .enumerate()
                .try_fold(1u64, |count, (axis, extent)| {
                    let extent = extent
                        .as_constant()
                        .and_then(|n| u64::try_from(n).ok())
                        .ok_or("dispatch bound requires nonnegative specialized extents")?;
                    let step = if phase == mappings.len() {
                        steps.get(axis).copied().unwrap_or(extent.max(1))
                    } else {
                        extent.max(1)
                    };
                    count
                        .checked_mul(extent.div_ceil(step))
                        .ok_or("dispatch lower envelope overflow")
                })?;
            items.push(count);
        }
        Ok(Self { items, remaining })
    }
    pub(super) fn demand(
        &self,
        alternatives: &Domain,
        indices: std::ops::Range<usize>,
        maximum_threads: u64,
        hardware: &model::Hardware,
    ) -> Result<Option<Demand>, String> {
        let mut items = self.items.clone();
        match &self.remaining {
            Remaining::Partition => {}
            Remaining::Mapping { phase, extent } => {
                let maximum = alternatives
                    .owner::<IntegerRange<super::MappingDecision>>()
                    .and_then(|domain| domain.get(indices.end - 1))
                    .ok_or("mapping bound has a different numeric domain")?;
                items[*phase] = items[*phase]
                    .checked_mul(extent.div_ceil(maximum))
                    .ok_or("mapping lower envelope overflow")?;
            }
            Remaining::Split { phases } => {
                let minimum = alternatives
                    .owner::<IntegerRange<super::SplitDecision>>()
                    .and_then(|domain| domain.get(indices.start))
                    .ok_or("split bound has a different numeric domain")?;
                if minimum > 1 {
                    for &phase in phases {
                        let base = items[phase];
                        items[phase] = base
                            .checked_mul(minimum)
                            .ok_or("split lower envelope overflow")?;
                        items.push(base); // Required merge launch for this phase.
                    }
                }
            }
        }
        let maximum_items = maximum_threads / SUBGROUP as u64;
        model::dispatch_demand(
            hardware,
            items.into_iter().map(|count| (count, maximum_items)),
        )
    }
}

/// Successful external dense publications survive decomposition, placement,
/// packet preparation and traversal. Private intermediate stores may disappear
/// during composition and never enter this projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Publications([u64; 6]);
impl Publications {
    pub(super) fn derive(function: &LoweredIr) -> Self {
        fn root(expression: &Expr) -> Option<usize> {
            match &expression.kind {
                ExprKind::Var(variable) => Some(*variable),
                ExprKind::Index { base, .. } | ExprKind::Transpose(base) => root(base),
                ExprKind::Builtin {
                    name: Builtin::Reshape,
                    args,
                } => root(args.first()?),
                _ => None,
            }
        }
        fn extent(dimensions: &[seismic_lang::sym::Sym]) -> Option<u64> {
            dimensions.iter().try_fold(1u64, |count, dimension| {
                Some(count.saturating_mul(u64::try_from(dimension.as_constant()?).ok()?))
            })
        }
        fn block(function: &LoweredIr, statements: &[Stmt]) -> [u64; 6] {
            let mut total = [0u64; 6];
            for statement in statements {
                let term = match &statement.kind {
                    StmtKind::Parallel { extents, body, .. } => {
                        let count = extent(extents).unwrap_or(0);
                        block(function, body).map(|n| n.saturating_mul(count))
                    }
                    StmtKind::Range { lo, hi, body, .. } => {
                        let count = hi
                            .sub(lo)
                            .as_constant()
                            .map(|n| n.max(0) as u64)
                            .unwrap_or(0);
                        block(function, body).map(|n| n.saturating_mul(count))
                    }
                    StmtKind::If { cond, then, els } => match cond.kind {
                        ExprKind::Bool(true) => block(function, then),
                        ExprKind::Bool(false) => block(function, els),
                        _ => {
                            let (left, right) = (block(function, then), block(function, els));
                            std::array::from_fn(|i| left[i].min(right[i]))
                        }
                    },
                    StmtKind::Expr(Expr {
                        kind:
                            ExprKind::Builtin {
                                name: Builtin::Store,
                                args,
                            },
                        ..
                    }) if args.len() == 2 => {
                        let mut counts = [0u64; 6];
                        if let Some((dtype, count)) = (|| {
                            let variable = function.vars.get(root(&args[1])?)?;
                            let VarKind::Param(parameter) = variable.kind else {
                                return None;
                            };
                            if !matches!(variable.ty, Ty::Tensor(_))
                                || function
                                    .ownership
                                    .intermediates
                                    .contains(&function.params.get(parameter)?.0)
                            {
                                return None;
                            }
                            let target = args[1].ty.shaped()?;
                            let Elem::Dtype(dtype) = target.elem else {
                                return None;
                            };
                            // A successful store validates equal source and destination
                            // extents. Use the destination's exact scalar element count.
                            Some((dtype, extent(&target.shape)?))
                        })() {
                            counts[TYPES.iter().position(|&ty| ty == dtype).unwrap()] = count;
                        }
                        counts
                    }
                    // Unresolved loops, helper effects and collective bodies can
                    // contribute no bound until their own lowering is retained.
                    _ => [0; 6],
                };
                for (sum, n) in total.iter_mut().zip(term) {
                    *sum = sum.saturating_add(n);
                }
            }
            total
        }
        Self(block(function, &function.body))
    }

    pub(super) fn include(
        &self,
        demand: &mut Demand,
        hardware: &model::Hardware,
    ) -> Result<(), String> {
        let mut operations = Vec::new();
        for (dtype, count) in TYPES.into_iter().zip(self.0) {
            // Store lowering publishes at most one scalar per subgroup lane.
            // Maximally packing those writes relaxes instruction/transaction
            // service while preserving the total PerLane demand. Unknown
            // addresses use the existing conservative transaction expansion.
            for (lanes, instances) in [
                (SUBGROUP as u64, count / SUBGROUP as u64),
                (count % SUBGROUP as u64, 1),
            ] {
                if lanes == 0 || instances == 0 {
                    continue;
                }
                operations.push(model::OperationCount {
                    primitive: Primitive::Write {
                        space: Space::Device,
                        ty: Type::from(dtype),
                    },
                    lanes,
                    instances,
                    access: None,
                });
            }
        }
        model::include_preserved_demand(
            demand,
            &model::InvocationAccount {
                operations,
                unmapped: Vec::new(),
                exhausted: None,
                visits: 0,
            },
            hardware,
            |_| true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_floor_keeps_only_guaranteed_external_effects() {
        use seismic_lang::{Scope, program::{SourceFile, compile}};
        let program = compile(&[SourceFile {
            path: "publication-floor.seismic.portable".into(), scope: Scope::Portable,
            text: "fn write(flag: i32, scratch: tensor[4] f32, out: tensor[4] f32):\n  whole = tile[4] f32\n  part = tile[2] f32\n  for i in owned(whole): whole[i] = 1.0\n  for i in owned(part): part[i] = 1.0\n  store(whole, scratch)\n  for r in range(0, 2):\n    if flag == 0:\n      store(part, out[0:2])\n    else:\n      store(whole, out)\n".into(),
        }], &[]).unwrap();
        let mut function = seismic_lang::lower::lower(&program, "write", "metal", &Default::default()).unwrap();
        let before = Publications::derive(&function);
        function.ownership.intermediates.insert("scratch".into());
        let after = Publications::derive(&function);
        assert_eq!(before.0, [8, 0, 0, 0, 0, 0]);
        assert_eq!(after.0, [4, 0, 0, 0, 0, 0]);
    }
}
