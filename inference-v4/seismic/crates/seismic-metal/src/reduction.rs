//! Reduction geometry and legal execution forms, independent of source emission.
//! These contracts do not rank algorithms or predict native register allocation.
use seismic_lang::types::DType;
use seismic_realization::dispatch::{TileDeclaration, TilePlacement};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// Each lane folds a complete output in axis order; distributed inputs need shuffles.
    Ordered,
    /// Each lane folds only the outputs whose complete reduction axis it owns.
    LaneLocal,
    /// Lanes accumulate their input elements and combine through a collective.
    Collective,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionDomain {
    input_capacity: u64,
    output_capacity: u64,
    axis_capacity: u64,
    inner_capacity: u64,
    scalar_output: bool,
    dtype: DType,
    lanes: u64,
    algorithms: Vec<Algorithm>,
}

impl ReductionDomain {
    pub fn new(
        capacities: &[i64],
        axis: usize,
        dtype: DType,
        ordered: bool,
        source: TilePlacement,
        lanes: u64,
    ) -> Result<Self, String> {
        if axis >= capacities.len() || lanes == 0 {
            return Err("invalid reduction axis or lane count".into());
        }
        let capacities = capacities
            .iter()
            .map(|&n| u64::try_from(n).map_err(|_| "negative reduction extent"))
            .collect::<Result<Vec<_>, _>>()?;
        let product = |dimensions: &[u64]| -> Result<u64, String> {
            if dimensions.contains(&0) {
                return Ok(0);
            }
            dimensions.iter().try_fold(1u64, |n, &d| {
                n.checked_mul(d)
                    .ok_or_else(|| "reduction extent product overflow".into())
            })
        };
        let input_capacity = product(&capacities)?;
        let axis_capacity = capacities[axis];
        let inner_capacity = product(&capacities[axis + 1..])?;
        let mut output = capacities.clone();
        output.remove(axis);
        let output_capacity = product(&output)?;
        // Emitted element/index arithmetic uses signed 32-bit indices.
        if [
            input_capacity,
            output_capacity,
            axis_capacity,
            inner_capacity,
        ]
        .into_iter()
        .any(|n| n > i32::MAX as u64)
        {
            return Err("reduction geometry exceeds Metal's signed index width".into());
        }
        let mut algorithms = vec![Algorithm::Ordered];
        if source == TilePlacement::Distributed {
            if inner_capacity > 0 && inner_capacity.is_multiple_of(lanes) {
                algorithms.push(Algorithm::LaneLocal);
            }
            // Narrow collectives are not qualified to preserve publication rounding.
            if !ordered
                && matches!(dtype, DType::F32 | DType::I32 | DType::U32)
                && axis_capacity > 0
                && inner_capacity > 0
            {
                algorithms.push(Algorithm::Collective);
            }
        }
        Ok(Self {
            input_capacity,
            output_capacity,
            axis_capacity,
            inner_capacity,
            scalar_output: output.is_empty(),
            dtype,
            lanes,
            algorithms,
        })
    }
    /// Argmax produces integer indices, preserving first-maximum semantics.
    /// Borrowed inputs permit direct reads; distributed inputs need every lane
    /// even for ordered folding because the fold uses lane shuffles.
    pub fn argmax(
        capacities: &[i64],
        axis: usize,
        input_dtype: DType,
        source: Option<TilePlacement>,
        full_lanes: bool,
        lanes: u64,
    ) -> Result<Self, String> {
        let mut domain = Self::new(
            capacities,
            axis,
            DType::I32,
            true,
            TilePlacement::Replicated,
            lanes,
        )?;
        if domain.axis_capacity == 0 {
            return Err("argmax requires a nonempty axis".into());
        }
        domain.algorithms.clear();
        if full_lanes || source != Some(TilePlacement::Distributed) {
            domain.algorithms.push(Algorithm::Ordered);
        }
        if full_lanes
            && matches!(input_dtype, DType::F32 | DType::I32 | DType::U32)
            && (source.is_none() || source == Some(TilePlacement::Distributed))
        {
            domain.algorithms.push(Algorithm::Collective);
        }
        Ok(domain)
    }
    pub fn algorithms(&self) -> &[Algorithm] {
        &self.algorithms
    }
    pub fn input_capacity(&self) -> u64 {
        self.input_capacity
    }
    pub fn output_capacity(&self) -> u64 {
        self.output_capacity
    }
    pub fn axis_capacity(&self) -> u64 {
        self.axis_capacity
    }
    pub fn inner_capacity(&self) -> u64 {
        self.inner_capacity
    }
    pub fn scalar_output(&self) -> bool {
        self.scalar_output
    }
    pub fn output(
        &self,
        algorithm: Algorithm,
        symbol: String,
    ) -> Result<Option<TileDeclaration>, String> {
        if !self.algorithms.contains(&algorithm) {
            return Err(
                "reduction algorithm violates its input ownership or numerical contract".into(),
            );
        }
        if self.scalar_output {
            return Ok(None);
        }
        Ok(Some(TileDeclaration {
            symbol,
            dtype: self.dtype,
            capacity: self.output_capacity,
            placement: if algorithm == Algorithm::LaneLocal {
                TilePlacement::Distributed
            } else {
                TilePlacement::Replicated
            },
        }))
    }
    pub fn output_slots(&self, algorithm: Algorithm) -> Result<u64, String> {
        if !self.algorithms.contains(&algorithm) {
            return Err("unadmitted reduction algorithm".into());
        }
        Ok(if algorithm == Algorithm::LaneLocal {
            self.output_capacity.div_ceil(self.lanes)
        } else {
            self.output_capacity
        }
        .max(1))
    }
}

/// Identity of a normalized reduction operation and its output binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Site {
    pub output: seismic_lang::ir::VarId,
    pub operation: seismic_lang::ir::OperationId,
}
#[derive(Clone, Debug)]
pub struct Decision {
    pub site: Site,
    pub input: seismic_lang::ir::VarId,
    pub materialize_input: bool,
    pub input_placement: Option<TilePlacement>,
    pub contract: seismic_lang::reduction::Contract,
    pub full_lanes: bool,
    pub domain: ReductionDomain,
}
#[derive(Clone, Debug)]
pub struct Selected {
    pub decision: Decision,
    pub algorithm: Algorithm,
    pub output: Option<TileDeclaration>,
}
#[derive(Clone, Debug, Default)]
pub struct ReductionPlan {
    selections: std::collections::HashMap<Site, Selected>,
}
impl ReductionPlan {
    pub fn selections(&self) -> &std::collections::HashMap<Site, Selected> {
        &self.selections
    }
    pub fn get(&self, site: Site) -> Result<&Selected, String> {
        self.selections
            .get(&site)
            .ok_or_else(|| format!("unresolved reduction at {site:?}"))
    }
}

/// Resolve reduction forms against already selected materialized-value placements.
/// Output ownership feeds subsequent reductions; no native code is generated.
pub fn plan(
    vars: &[seismic_lang::ir::Var],
    body: &[seismic_lang::ir::Stmt],
    phases: &[crate::execution::Phase],
    storage: &crate::storage::StoragePlan,
    select: &mut dyn FnMut(&Decision) -> Result<Algorithm, String>,
) -> Result<ReductionPlan, String> {
    use seismic_lang::{
        ir::*,
        sym::{Atom, Sym},
        types::Ty,
    };
    use std::collections::HashMap;
    struct Planner<'a> {
        vars: &'a [Var],
        storage: &'a crate::storage::StoragePlan,
        select: &'a mut dyn FnMut(&Decision) -> Result<Algorithm, String>,
        bindings: HashMap<VarId, Option<TilePlacement>>,
        pieces: HashMap<Atom, Sym>,
        result: ReductionPlan,
        full_lanes: bool,
    }
    impl Planner<'_> {
        fn body(&mut self, body: &[Stmt]) -> Result<(), String> {
            for statement in body {
                match &statement.kind {
                    StmtKind::LoadLoop {
                        vars,
                        modes,
                        piece,
                        capacity,
                        body,
                        ..
                    } => {
                        if let Some(capacity) = capacity {
                            self.pieces.insert(piece.clone(), Sym::constant(*capacity));
                        }
                        let modes = modes
                            .as_ref()
                            .filter(|m| m.len() == vars.len())
                            .ok_or("reduction planning requires selected stream loads")?;
                        for (var, mode) in vars.iter().zip(modes) {
                            self.bindings.insert(
                                *var,
                                if *mode == LoadMode::Borrow {
                                    None
                                } else {
                                    Some(self.storage.declaration(*var)?.placement.clone())
                                },
                            );
                        }
                        self.body(body)?;
                    }
                    StmtKind::Assign { target, value, .. } => {
                        let ExprKind::Var(output) = target.kind else {
                            continue;
                        };
                        match &value.kind {
                            ExprKind::TileAlloc { .. } => {
                                self.bindings.insert(
                                    output,
                                    Some(self.storage.declaration(output)?.placement.clone()),
                                );
                            }
                            ExprKind::Load { mode, .. } => {
                                let placement = if *mode == LoadMode::Borrow {
                                    None
                                } else {
                                    Some(self.storage.declaration(output)?.placement.clone())
                                };
                                self.bindings.entry(output).or_insert(placement);
                            }
                            ExprKind::Var(_) if matches!(target.ty, Ty::Tile(_)) => {
                                if !self.bindings.contains_key(&output) {
                                    self.bindings.insert(
                                        output,
                                        Some(self.storage.declaration(output)?.placement.clone()),
                                    );
                                }
                            }
                            ExprKind::Builtin {
                                name: Builtin::Reduce,
                                args,
                            } => {
                                let [input, axis, operation, ..] = args.as_slice() else {
                                    return Err("invalid normalized reduction".into());
                                };
                                let (
                                    ExprKind::Var(input),
                                    ExprKind::Int(axis),
                                    ExprKind::Int(operation),
                                ) = (&input.kind, &axis.kind, &operation.kind)
                                else {
                                    return Err("reduction parameters are unresolved".into());
                                };
                                let argmax = *operation == 3;
                                let binding = self
                                    .bindings
                                    .get(input)
                                    .ok_or("reduction input has no selected ownership")?
                                    .clone();
                                let materialize_input = binding.is_none() && !argmax;
                                let placement = if materialize_input {
                                    Some(self.storage.declaration(*input)?.placement.clone())
                                } else {
                                    binding
                                };
                                let Ty::Tile(tile) = &self.vars[*input].ty else {
                                    return Err("reduction input must be a tile".into());
                                };
                                let capacities = tile
                                    .shape
                                    .iter()
                                    .map(|extent| {
                                        let mut extent = extent.clone();
                                        for (atom, value) in &self.pieces {
                                            extent = extent.subst(atom, value);
                                        }
                                        extent.as_constant().ok_or_else(|| {
                                            format!("reduction extent `{extent}` has no capacity")
                                        })
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                let dtype =
                                    tile.elem.read_dtype().ok_or("unresolved reduction dtype")?;
                                let contract = seismic_lang::reduction::Contract::new(
                                    ReduceOp::from_tag(*operation)
                                        .ok_or("unknown reduction operation")?,
                                    dtype,
                                    matches!(
                                        args.get(3).map(|e| &e.kind),
                                        Some(ExprKind::Bool(true))
                                    ),
                                );
                                let ordered = contract.ordered
                                    || matches!(dtype, DType::F16 | DType::BF16)
                                    || contract.combination()
                                        == seismic_lang::reduction::Combination::SaturatingAdd;
                                let domain = if argmax {
                                    ReductionDomain::argmax(
                                        &capacities,
                                        *axis as usize,
                                        dtype,
                                        placement.clone(),
                                        self.full_lanes,
                                        crate::execution::SUBGROUP as u64,
                                    )?
                                } else {
                                    ReductionDomain::new(
                                        &capacities,
                                        *axis as usize,
                                        dtype,
                                        ordered,
                                        placement
                                            .clone()
                                            .ok_or("fold requires materialized input")?,
                                        crate::execution::SUBGROUP as u64,
                                    )?
                                };
                                let site = Site {
                                    output,
                                    operation: statement
                                        .id
                                        .ok_or("reduction operation has no normalized identity")?,
                                };
                                let decision = Decision {
                                    site,
                                    input: *input,
                                    materialize_input,
                                    contract,
                                    input_placement: placement.clone(),
                                    full_lanes: self.full_lanes,
                                    domain,
                                };
                                let algorithm = (self.select)(&decision)?;
                                if !self.full_lanes
                                    && ((placement == Some(TilePlacement::Distributed)
                                        && algorithm != Algorithm::LaneLocal)
                                        || (materialize_input
                                            && placement == Some(TilePlacement::GroupShared)))
                                {
                                    return Err("selected reduction requires full-lane participation inside an owned domain".into());
                                }
                                let declaration = decision
                                    .domain
                                    .output(algorithm, self.vars[output].name.clone())?;
                                if self.result.selections.contains_key(&site) {
                                    return Err(
                                        "reduction identity is ambiguous after normalization"
                                            .into(),
                                    );
                                }
                                self.bindings.insert(*input, placement);
                                if let Some(declaration) = &declaration {
                                    self.bindings
                                        .entry(output)
                                        .or_insert(Some(declaration.placement.clone()));
                                }
                                self.result.selections.insert(
                                    site,
                                    Selected {
                                        decision,
                                        algorithm,
                                        output: declaration,
                                    },
                                );
                            }
                            _ => {}
                        }
                    }
                    StmtKind::Owned { tile, body, .. } => {
                        let previous = self.full_lanes;
                        let ExprKind::Var(var) = tile.kind else {
                            return Err("owned domain lacks a value binding".into());
                        };
                        self.full_lanes &=
                            self.bindings.get(&var) == Some(&Some(TilePlacement::Replicated));
                        self.body(body)?;
                        self.full_lanes = previous;
                    }
                    StmtKind::Parallel { body, .. }
                    | StmtKind::Range { body, .. }
                    | StmtKind::Lanes { body, .. } => self.body(body)?,
                    StmtKind::If { then, els, .. } => {
                        self.body(then)?;
                        self.body(els)?;
                    }
                    _ => {}
                }
            }
            Ok(())
        }
    }
    let mut planner = Planner {
        vars,
        storage,
        select,
        bindings: HashMap::new(),
        pieces: HashMap::new(),
        result: ReductionPlan::default(),
        full_lanes: true,
    };
    if phases.len() != body.len() {
        return Err("reduction plan phase/domain mismatch".into());
    }
    for (root, phase) in body.iter().zip(phases) {
        if let Some(split) = &phase.split {
            let StmtKind::Parallel { body, .. } = &root.kind else {
                return Err("split reduction phase has no parallel domain".into());
            };
            let (prefix, suffix) = body
                .split_at_checked(split.loop_at)
                .ok_or("split reduction position is invalid")?;
            planner.body(prefix)?;
            planner.body(&split.validation_bindings)?;
            planner.body(suffix)?;
        } else {
            planner.body(std::slice::from_ref(root))?;
        }
    }
    Ok(planner.result)
}
