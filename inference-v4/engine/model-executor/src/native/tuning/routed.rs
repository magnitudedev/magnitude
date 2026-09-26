//! Tuning cases of the routed (mixture-of-experts) entries. `routed_route`
//! serves every row class; the decode entries (`routed_expand`,
//! `routed_output`) are tuned at the decode row points, the grouped
//! entries (`routed_group`, `routed_experts`,
//! `routed_combine`) at the grouped ones. Routing tables are synthetic:
//! every row selects distinct experts with the uneven expert loads of real
//! routing (`routes`), and the grouped tables are formed from them exactly as
//! `routed_group` forms them.
//!
//! Route and group write their tables through `&mut` parameters; each
//! argument set owns those tables as case state.

use super::{
    row_points, served_row_points, CaseState, EntryTuning, PointShape, TuningInputs, TuningLimits,
};
use crate::programs::graph::routed::{grouped_blocks, DECODE_ROWS, TILE_ROWS};
use magnitude_model_contracts::{FeedForwardGeometry, WeightKind, WeightScope};
use magnitude_model_kernels::{
    routed_combine, routed_expand, routed_experts, routed_group, routed_output, routed_route,
};
use seismic::{Element, Tensor};

/// The routed geometry every case of one binding shares.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoutedShape {
    pub hidden: u64,
    pub experts: u64,
    pub selected: u64,
    pub features: u64,
    pub shared: u64,
}

/// The decode points: row counts the decode form serves.
fn decode_points(limits: TuningLimits) -> Vec<PointShape> {
    served_row_points(limits.max_rows, |rows| rows <= DECODE_ROWS)
}

/// The grouped points: row counts past the decode form.
fn grouped_points(limits: TuningLimits) -> Vec<PointShape> {
    served_row_points(limits.max_rows, |rows| rows > DECODE_ROWS)
}

/// Distinct experts per row with the uneven expert loads of real routing: the
/// expert of popularity rank r is (101 r) mod E (a permutation: 101 is prime
/// and no expert count is a multiple of it) with popularity 1 / sqrt(r + 1),
/// and each row draws its choices by popularity without replacement from a
/// fixed-seed generator. A grouped point then holds blocks of every live-row
/// count, from one row to full tiles, and experts with no rows.
fn routes(rows: u64, shape: RoutedShape) -> Vec<i32> {
    let experts = shape.experts as usize;
    let popularity = (0..experts)
        .map(|rank| 1.0 / ((rank + 1) as f64).sqrt())
        .collect::<Vec<_>>();
    let total = popularity.iter().sum::<f64>();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut routes = Vec::with_capacity(rows as usize * shape.selected as usize);
    for _ in 0..rows {
        let mut taken = vec![false; experts];
        let mut remaining = total;
        for _ in 0..shape.selected {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut target = (state >> 11) as f64 / (1_u64 << 53) as f64 * remaining;
            // The last untaken expert absorbs the rounding of `remaining`.
            let mut rank = 0;
            for candidate in (0..experts).filter(|&candidate| !taken[candidate]) {
                rank = candidate;
                target -= popularity[candidate];
                if target < 0.0 {
                    break;
                }
            }
            taken[rank] = true;
            remaining -= popularity[rank];
            routes.push(((rank * 101) % experts) as i32);
        }
    }
    routes
}

/// The tables `routed_group` forms from `routes`: (order [B, T],
/// inverse [M, K], blocks [B]).
fn group(
    routes: &[i32],
    rows: u64,
    shape: RoutedShape,
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>), String> {
    let blocks = grouped_blocks(rows, shape.experts, shape.selected)?;
    let tile = TILE_ROWS as usize;
    let mut order = vec![-1; blocks as usize * tile];
    let mut inverse = vec![0; routes.len()];
    let mut table = vec![-1; blocks as usize];
    let mut block = 0usize;
    for expert in 0..shape.experts as i32 {
        let mut lane = 0usize;
        for (flat, route) in routes.iter().enumerate() {
            if *route != expert {
                continue;
            }
            table[block] = expert;
            order[block * tile + lane] = (flat / shape.selected as usize) as i32;
            inverse[flat] = (block * tile + lane) as i32;
            lane += 1;
            if lane == tile {
                lane = 0;
                block += 1;
            }
        }
        if lane > 0 {
            block += 1;
        }
    }
    Ok((order, inverse, table))
}

fn scores(
    inputs: &TuningInputs<'_, '_>,
    rows: u64,
    shape: RoutedShape,
    seed: u64,
) -> Result<Tensor, String> {
    inputs.activation(Element::f32(), &[rows, shape.selected], seed)
}

/// `routed_route`: RMS prologue, router softmax, top-k selection and the
/// shared expert's coefficient. `SIMDGROUPS` reassociates the router's
/// reductions (arithmetic); its integer routes are still compared exactly,
/// so a configuration that flips a near-tie choice on the tuning rows is
/// not chosen.
pub(crate) struct RoutedRouteTuning {
    pub norm: Element,
    pub router: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct RoutedRouteCase {
    residual: Tensor,
    norm: Tensor,
    router: Tensor,
    shared_router: Tensor,
    routes: CaseState,
    scores: CaseState,
    epsilon: f32,
    normalize: i32,
}

impl RoutedRouteTuning {
    fn elements(&self) -> routed_route::Elements {
        routed_route::Elements {
            NW: self.norm,
            RW: self.router,
            A: self.activation,
        }
    }

    /// Whether the model divides the selected scores by their sum.
    fn normalize(&self, inputs: &TuningInputs<'_, '_>) -> Result<bool, String> {
        let scope = *self
            .scopes
            .first()
            .ok_or("a tuning case needs at least one layer")?;
        let WeightScope::TargetBlock(index) = scope else {
            return Err(format!("routed layers are target blocks, not {scope:?}"));
        };
        match inputs
            .definition
            .geometry
            .blocks
            .get(index as usize)
            .map(|block| &block.feedforward)
        {
            Some(FeedForwardGeometry::Routed(experts)) => Ok(experts.normalize_selected),
            _ => Err(format!("block {index} has no routed feed-forward")),
        }
    }
}

impl EntryTuning for RoutedRouteTuning {
    type Entry = routed_route::Entry;
    type Case = RoutedRouteCase;

    fn bindings(&self) -> String {
        format!(
            "NW={},RW={},A={}",
            self.norm.name(),
            self.router.name(),
            self.activation.name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        Ok(vec![
            ("H", shape.hidden),
            ("E", shape.experts),
            ("K", shape.selected),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let (shape, rows) = (self.shape, point.rows);
        let normalize = i32::from(self.normalize(inputs)?);
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let routes = inputs.scratch(Element::i32(), &[rows, shape.selected])?;
                let scores = inputs.scratch(Element::f32(), &[rows, shape.selected])?;
                Ok(RoutedRouteCase {
                    residual: inputs.activation(
                        Element::f32(),
                        &[rows, shape.hidden],
                        index as u64 + 1,
                    )?,
                    norm: inputs.weight(scope, WeightKind::FeedForwardNorm)?,
                    router: inputs.weight(scope, WeightKind::Router)?,
                    shared_router: inputs.weight(scope, WeightKind::SharedRouter)?,
                    routes: inputs.state(routes, 0..rows)?,
                    scores: inputs.state(scores, 0..rows)?,
                    epsilon: self.epsilon,
                    normalize,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_route::Args<'a> {
        routed_route::Args {
            residual: &case.residual,
            norm: &case.norm,
            router: &case.router,
            shared_router: &case.shared_router,
            routes: case.routes.tensor_mut(),
            scores: case.scores.tensor_mut(),
            eps: case.epsilon,
            normalize: case.normalize,
        }
    }

    fn state(case: &Self::Case) -> Vec<&CaseState> {
        vec![&case.routes, &case.scores]
    }

    generated_entry!(routed_route, this => this.elements());
}

/// `routed_group`: the grouped form's expert tiles, formed from the
/// routes. Its parameters are mappings.
pub(crate) struct RoutedGroupTuning {
    pub shape: RoutedShape,
}

pub(crate) struct RoutedGroupCase {
    routes: Tensor,
    counts: CaseState,
    order: CaseState,
    inverse: CaseState,
    blocks: CaseState,
}

impl EntryTuning for RoutedGroupTuning {
    type Entry = routed_group::Entry;
    type Case = RoutedGroupCase;

    fn bindings(&self) -> String {
        "fixed".into()
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(vec![("E", self.shape.experts), ("K", self.shape.selected)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    /// One argument set: the tables are small and stay cache resident in a
    /// real step as well.
    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let (shape, rows) = (self.shape, point.rows);
        let blocks = grouped_blocks(rows, shape.experts, shape.selected)?;
        let table = |extents: &[u64]| -> Result<CaseState, String> {
            let tensor = inputs.scratch(Element::i32(), extents)?;
            inputs.state(tensor, 0..extents[0])
        };
        Ok(vec![RoutedGroupCase {
            routes: inputs.i32s(&[rows, shape.selected], &routes(rows, shape))?,
            counts: table(&[shape.experts])?,
            order: table(&[blocks, TILE_ROWS])?,
            inverse: table(&[rows, shape.selected])?,
            blocks: table(&[blocks])?,
        }])
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_group::Args<'a> {
        routed_group::Args {
            routes: &case.routes,
            counts: case.counts.tensor_mut(),
            order: case.order.tensor_mut(),
            inverse: case.inverse.tensor_mut(),
            blocks: case.blocks.tensor_mut(),
        }
    }

    fn state(case: &Self::Case) -> Vec<&CaseState> {
        vec![&case.counts, &case.order, &case.inverse, &case.blocks]
    }

    generated_entry!(routed_group);
}

/// `routed_expand`: the selected experts' and the shared expert's
/// gate/up with SiLU·mul (decode rows).
pub(crate) struct RoutedExpandTuning {
    pub expert_gate: Element,
    pub expert_up: Element,
    pub shared_gate: Element,
    pub shared_up: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedExpandCase {
    normalized: Tensor,
    routes: Tensor,
    expert_gate: Tensor,
    expert_up: Tensor,
    shared_gate: Tensor,
    shared_up: Tensor,
}

impl RoutedExpandTuning {
    fn elements(&self) -> routed_expand::Elements {
        routed_expand::Elements {
            EGW: self.expert_gate,
            EUW: self.expert_up,
            SGW: self.shared_gate,
            SUW: self.shared_up,
            A: self.activation,
        }
    }
}

impl EntryTuning for RoutedExpandTuning {
    type Entry = routed_expand::Entry;
    type Case = RoutedExpandCase;

    fn bindings(&self) -> String {
        format!(
            "EGW={},EUW={},SGW={},SUW={},A={}",
            self.expert_gate.name(),
            self.expert_up.name(),
            self.shared_gate.name(),
            self.shared_up.name(),
            self.activation.name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        Ok(vec![
            ("H", shape.hidden),
            ("K", shape.selected),
            ("F", shape.features),
            ("S", shape.shared),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let shape = self.shape;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                Ok(RoutedExpandCase {
                    normalized: inputs.activation(
                        self.activation,
                        &[point.rows, shape.hidden],
                        index as u64 + 1,
                    )?,
                    routes: inputs
                        .i32s(&[point.rows, shape.selected], &routes(point.rows, shape))?,
                    expert_gate: inputs.weight(scope, WeightKind::ExpertGate)?,
                    expert_up: inputs.weight(scope, WeightKind::ExpertUp)?,
                    shared_gate: inputs.weight(scope, WeightKind::SharedGate)?,
                    shared_up: inputs.weight(scope, WeightKind::SharedUp)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_expand::Args<'a> {
        routed_expand::Args {
            normalized: &case.normalized,
            routes: &case.routes,
            expert_gate: &case.expert_gate,
            expert_up: &case.expert_up,
            shared_gate: &case.shared_gate,
            shared_up: &case.shared_up,
        }
    }

    generated_entry!(routed_expand, this => this.elements());
}

/// `routed_output`: the selected experts' down projections in slot
/// order, the shared expert's, and the residual (decode rows).
pub(crate) struct RoutedOutputTuning {
    pub expert_down: Element,
    pub shared_down: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedOutputCase {
    residual: Tensor,
    expert_product: Tensor,
    shared_product: Tensor,
    routes: Tensor,
    scores: Tensor,
    coefficient: Tensor,
    expert_down: Tensor,
    shared_down: Tensor,
}

impl RoutedOutputTuning {
    fn elements(&self) -> routed_output::Elements {
        routed_output::Elements {
            EDW: self.expert_down,
            SDW: self.shared_down,
            A: self.activation,
        }
    }
}

impl EntryTuning for RoutedOutputTuning {
    type Entry = routed_output::Entry;
    type Case = RoutedOutputCase;

    fn bindings(&self) -> String {
        format!(
            "EDW={},SDW={},A={}",
            self.expert_down.name(),
            self.shared_down.name(),
            self.activation.name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        Ok(vec![
            ("H", shape.hidden),
            ("K", shape.selected),
            ("F", shape.features),
            ("S", shape.shared),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let (shape, rows) = (self.shape, point.rows);
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 8 * index as u64;
                Ok(RoutedOutputCase {
                    residual: inputs.activation(Element::f32(), &[rows, shape.hidden], seed + 1)?,
                    expert_product: inputs.activation(
                        self.activation,
                        &[rows, shape.selected, shape.features],
                        seed + 2,
                    )?,
                    shared_product: inputs.activation(
                        self.activation,
                        &[rows, shape.shared],
                        seed + 3,
                    )?,
                    routes: inputs.i32s(&[rows, shape.selected], &routes(rows, shape))?,
                    scores: scores(inputs, rows, shape, seed + 4)?,
                    coefficient: inputs.activation(Element::f32(), &[rows], seed + 5)?,
                    expert_down: inputs.weight(scope, WeightKind::ExpertDown)?,
                    shared_down: inputs.weight(scope, WeightKind::SharedDown)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_output::Args<'a> {
        routed_output::Args {
            residual: &case.residual,
            expert_product: &case.expert_product,
            shared_product: &case.shared_product,
            routes: &case.routes,
            scores: &case.scores,
            coefficient: &case.coefficient,
            expert_down: &case.expert_down,
            shared_down: &case.shared_down,
        }
    }

    generated_entry!(routed_output, this => this.elements());
}

/// `routed_experts`: grouped expert tiles (grouped rows).
pub(crate) struct RoutedExpertsTuning {
    pub expert_gate: Element,
    pub expert_up: Element,
    pub expert_down: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedExpertsCase {
    normalized: Tensor,
    order: Tensor,
    blocks: Tensor,
    expert_gate: Tensor,
    expert_up: Tensor,
    expert_down: Tensor,
}

impl RoutedExpertsTuning {
    fn elements(&self) -> routed_experts::Elements {
        routed_experts::Elements {
            EGW: self.expert_gate,
            EUW: self.expert_up,
            EDW: self.expert_down,
            A: self.activation,
        }
    }
}

impl EntryTuning for RoutedExpertsTuning {
    type Entry = routed_experts::Entry;
    type Case = RoutedExpertsCase;

    fn bindings(&self) -> String {
        format!(
            "EGW={},EUW={},EDW={},A={}",
            self.expert_gate.name(),
            self.expert_up.name(),
            self.expert_down.name(),
            self.activation.name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(vec![("H", self.shape.hidden), ("F", self.shape.features)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let (shape, rows) = (self.shape, point.rows);
        let (order, _, table) = group(&routes(rows, shape), rows, shape)?;
        let blocks = table.len() as u64;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                Ok(RoutedExpertsCase {
                    normalized: inputs.activation(
                        self.activation,
                        &[rows, shape.hidden],
                        index as u64 + 1,
                    )?,
                    order: inputs.i32s(&[blocks, TILE_ROWS], &order)?,
                    blocks: inputs.i32s(&[blocks], &table)?,
                    expert_gate: inputs.weight(scope, WeightKind::ExpertGate)?,
                    expert_up: inputs.weight(scope, WeightKind::ExpertUp)?,
                    expert_down: inputs.weight(scope, WeightKind::ExpertDown)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_experts::Args<'a> {
        routed_experts::Args {
            normalized: &case.normalized,
            order: &case.order,
            blocks: &case.blocks,
            expert_gate: &case.expert_gate,
            expert_up: &case.expert_up,
            expert_down: &case.expert_down,
        }
    }

    generated_entry!(routed_experts, this => this.elements());
}

/// `routed_combine`: the shared expert over every row with the grouped
/// unpermute and the residual (grouped rows).
pub(crate) struct RoutedCombineTuning {
    pub shared_gate: Element,
    pub shared_up: Element,
    pub shared_down: Element,
    pub activation: Element,
    pub shape: RoutedShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedCombineCase {
    residual: Tensor,
    expert_output: Tensor,
    inverse: Tensor,
    scores: Tensor,
    normalized: Tensor,
    coefficient: Tensor,
    shared_gate: Tensor,
    shared_up: Tensor,
    shared_down: Tensor,
}

impl RoutedCombineTuning {
    fn elements(&self) -> routed_combine::Elements {
        routed_combine::Elements {
            SGW: self.shared_gate,
            SUW: self.shared_up,
            SDW: self.shared_down,
            A: self.activation,
        }
    }
}

impl EntryTuning for RoutedCombineTuning {
    type Entry = routed_combine::Entry;
    type Case = RoutedCombineCase;

    fn bindings(&self) -> String {
        format!(
            "SGW={},SUW={},SDW={},A={}",
            self.shared_gate.name(),
            self.shared_up.name(),
            self.shared_down.name(),
            self.activation.name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        Ok(vec![
            ("H", shape.hidden),
            ("K", shape.selected),
            ("S", shape.shared),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let (shape, rows) = (self.shape, point.rows);
        let (_, inverse, table) = group(&routes(rows, shape), rows, shape)?;
        let blocks = table.len() as u64;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 8 * index as u64;
                Ok(RoutedCombineCase {
                    residual: inputs.activation(Element::f32(), &[rows, shape.hidden], seed + 1)?,
                    expert_output: inputs.activation(
                        self.activation,
                        &[blocks, TILE_ROWS, shape.hidden],
                        seed + 2,
                    )?,
                    inverse: inputs.i32s(&[rows, shape.selected], &inverse)?,
                    scores: scores(inputs, rows, shape, seed + 3)?,
                    normalized: inputs.activation(
                        self.activation,
                        &[rows, shape.hidden],
                        seed + 4,
                    )?,
                    coefficient: inputs.activation(Element::f32(), &[rows], seed + 5)?,
                    shared_gate: inputs.weight(scope, WeightKind::SharedGate)?,
                    shared_up: inputs.weight(scope, WeightKind::SharedUp)?,
                    shared_down: inputs.weight(scope, WeightKind::SharedDown)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_combine::Args<'a> {
        routed_combine::Args {
            residual: &case.residual,
            expert_output: &case.expert_output,
            inverse: &case.inverse,
            scores: &case.scores,
            normalized: &case.normalized,
            coefficient: &case.coefficient,
            shared_gate: &case.shared_gate,
            shared_up: &case.shared_up,
            shared_down: &case.shared_down,
        }
    }

    generated_entry!(routed_combine, this => this.elements());
}
