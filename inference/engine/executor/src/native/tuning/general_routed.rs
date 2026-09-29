//! Tuning cases of the general routed entries (`operators::routed`):
//! `routed_select` at every row point; `routed_gate_up` / `routed_up` and
//! `routed_down` at the decode points; `routed_experts` /
//! `routed_experts_up` and `routed_scatter` at the grouped points
//! (`routed_group` is tuned by the fused form's case). Routing tables are
//! the fused cases' synthetic tables. The shared expert and latent
//! projections are the dense cases over their own weight kinds.

use super::routed::{decode_points, group, grouped_points, routes, RoutedShape};
use super::{row_points, CaseState, EntryTuning, PointShape, TuningInputs, TuningLimits};
use crate::operators::routed::fused_graph::TILE_ROWS;
use crate::GeneralRoutedBinding;
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::{
    routed_down, routed_experts, routed_experts_up, routed_gate_up, routed_scatter, routed_select,
    routed_up,
};
use seismic::{Element, Tensor};

/// The fused cases' routing shape for the synthetic tables.
pub(crate) fn routing_shape(binding: &GeneralRoutedBinding) -> RoutedShape {
    let shape = binding.shape;
    RoutedShape {
        hidden: shape.expert_hidden,
        experts: shape.experts,
        selected: shape.selected,
        features: shape.intermediate,
        shared: 0,
    }
}

/// The element the routed sum publishes in (`general_routed::sum_element`).
fn sum_element(binding: &GeneralRoutedBinding) -> Element {
    crate::operators::routed::graph::sum_element(binding)
}

/// `routed_select`: RMS prologues, router scores, biased top-k and the
/// normalized, scaled weights. Integer routes are compared exactly, so a
/// configuration that flips a near-tie choice on the tuning rows is not
/// chosen.
pub(crate) struct RoutedSelectTuning {
    pub binding: GeneralRoutedBinding,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct RoutedSelectCase {
    residual: Tensor,
    norm: Tensor,
    router_norm: Tensor,
    router: Tensor,
    bias: Tensor,
    expert_scale: Tensor,
    routes: CaseState,
    weights: CaseState,
    epsilon: f32,
    shape: crate::GeneralRoutedShape,
}

impl EntryTuning for RoutedSelectTuning {
    type Entry = routed_select::Entry;
    type Case = RoutedSelectCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn weight_scopes(&self) -> &[WeightScope] {
        &self.scopes
    }

    fn bindings(&self) -> String {
        let b = self.binding;
        format!(
            "NW={},RNW={},RW={},A={}",
            b.norm.name(),
            b.router_norm.name(),
            b.router.name(),
            b.activation.name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let [_, statics @ ..] = self.binding.shape.select_dimensions(0);
        Ok(statics.to_vec())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let (shape, rows) = (self.binding.shape, point.rows);
        let experts = usize::try_from(shape.experts).map_err(|_| "expert count exceeds host")?;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let routes = inputs.scratch(Element::i32(), &[rows, shape.selected])?;
                let weights = inputs.scratch(Element::f32(), &[rows, shape.selected])?;
                let norm = inputs.weight(scope, WeightKind::InputNorm)?;
                Ok(RoutedSelectCase {
                    residual: inputs.activation(
                        Element::f32(),
                        &[rows, shape.hidden],
                        index as u64 + 1,
                    )?,
                    router_norm: if shape.router_norm {
                        inputs.weight(scope, WeightKind::RouterInputNorm)?
                    } else {
                        norm.clone()
                    },
                    norm,
                    router: inputs.weight(scope, WeightKind::Router)?,
                    bias: if shape.bias {
                        inputs.weight(scope, WeightKind::RouterSelectionBias)?
                    } else {
                        inputs.f32s(&[shape.experts], &vec![0.0; experts])?
                    },
                    expert_scale: if shape.expert_scale {
                        inputs.weight(scope, WeightKind::ExpertScale)?
                    } else {
                        inputs.f32s(&[shape.experts], &vec![1.0; experts])?
                    },
                    routes: inputs.state(routes, 0..rows)?,
                    weights: inputs.state(weights, 0..rows)?,
                    epsilon: self.epsilon,
                    shape,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_select::Args<'a> {
        routed_select::Args {
            residual: &case.residual,
            norm: &case.norm,
            router_norm: &case.router_norm,
            router: &case.router,
            bias: &case.bias,
            expert_scale: &case.expert_scale,
            routes: case.routes.tensor_mut(),
            weights: case.weights.tensor_mut(),
            epsilon: case.epsilon,
            score: case.shape.score,
            normalization: case.shape.normalization,
            normalization_epsilon: f32::from_bits(case.shape.normalization_epsilon),
            scale: f32::from_bits(case.shape.scale),
        }
    }

    fn state(case: &Self::Case) -> Vec<&CaseState> {
        vec![&case.routes, &case.weights]
    }

    generated_entry!(routed_select, this => routed_select::Elements {
        NW: this.binding.norm,
        RNW: this.binding.router_norm,
        RW: this.binding.router,
        A: this.binding.activation,
    });
}

/// The decode expansion: `routed_gate_up` (gated experts) or `routed_up`.
pub(crate) struct RoutedExpandDecodeTuning {
    pub binding: GeneralRoutedBinding,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedExpandDecodeCase {
    normalized: Tensor,
    routes: Tensor,
    expert_gate: Option<Tensor>,
    expert_up: Tensor,
    up_scale: Tensor,
    activation: i32,
}

/// Unit accumulator scales for up-only experts (`routed_up` /
/// `routed_experts_up`; NVFP4's stored scales are refused at plan time).
fn unit_up_scales(inputs: &mut TuningInputs<'_, '_>, experts: u64) -> Result<Tensor, String> {
    let count = usize::try_from(experts).map_err(|_| "expert count exceeds host")?;
    inputs.f32s(&[experts], &vec![1.0; count])
}

impl RoutedExpandDecodeTuning {
    fn cases(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<RoutedExpandDecodeCase>, String> {
        let (shape, rows) = (self.binding.shape, point.rows);
        let table = routes(rows, routing_shape(&self.binding));
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                Ok(RoutedExpandDecodeCase {
                    normalized: inputs.activation(
                        self.binding.activation,
                        &[rows, shape.expert_hidden],
                        index as u64 + 1,
                    )?,
                    routes: inputs.i32s(&[rows, shape.selected], &table)?,
                    expert_gate: if shape.experts_expansion.gated {
                        Some(inputs.weight(scope, WeightKind::ExpertGate)?)
                    } else {
                        None
                    },
                    expert_up: inputs.weight(scope, WeightKind::ExpertUp)?,
                    up_scale: unit_up_scales(inputs, shape.experts)?,
                    activation: shape.experts_expansion.activation,
                })
            })
            .collect()
    }

    fn statics(&self) -> Vec<(&'static str, u64)> {
        let [_, statics @ ..] = self.binding.shape.expert_dimensions(0);
        statics.to_vec()
    }
}

/// `routed_gate_up` over [`RoutedExpandDecodeTuning`].
pub(crate) struct RoutedGateUpTuning(pub RoutedExpandDecodeTuning);

/// `routed_up` over [`RoutedExpandDecodeTuning`].
pub(crate) struct RoutedUpTuning(pub RoutedExpandDecodeTuning);

impl EntryTuning for RoutedGateUpTuning {
    type Entry = routed_gate_up::Entry;
    type Case = RoutedExpandDecodeCase;

    fn launches(&self) -> usize {
        self.0.scopes.len()
    }

    fn weight_scopes(&self) -> &[WeightScope] {
        &self.0.scopes
    }

    fn bindings(&self) -> String {
        let b = self.0.binding;
        format!(
            "EGW={},EUW={},A={}",
            b.expert_gate.unwrap_or(b.expert_up).name(),
            b.expert_up.name(),
            b.activation.name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(self.0.statics())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        self.0.cases(inputs, point)
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_gate_up::Args<'a> {
        routed_gate_up::Args {
            normalized: &case.normalized,
            routes: &case.routes,
            expert_gate: case
                .expert_gate
                .as_ref()
                .expect("a gated expansion case binds the gate weight"),
            expert_up: &case.expert_up,
            activation: case.activation,
        }
    }

    generated_entry!(routed_gate_up, this => routed_gate_up::Elements {
        EGW: this.0.binding.expert_gate.unwrap_or(this.0.binding.expert_up),
        EUW: this.0.binding.expert_up,
        A: this.0.binding.activation,
    });
}

impl EntryTuning for RoutedUpTuning {
    type Entry = routed_up::Entry;
    type Case = RoutedExpandDecodeCase;

    fn launches(&self) -> usize {
        self.0.scopes.len()
    }

    fn weight_scopes(&self) -> &[WeightScope] {
        &self.0.scopes
    }

    fn bindings(&self) -> String {
        let b = self.0.binding;
        format!("EUW={},A={}", b.expert_up.name(), b.activation.name())
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(self.0.statics())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        self.0.cases(inputs, point)
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_up::Args<'a> {
        routed_up::Args {
            normalized: &case.normalized,
            routes: &case.routes,
            expert_up: &case.expert_up,
            up_scale: &case.up_scale,
            activation: case.activation,
        }
    }

    generated_entry!(routed_up, this => routed_up::Elements {
        EUW: this.0.binding.expert_up,
        A: this.0.binding.activation,
    });
}

/// `routed_down`: the selected experts' down projections, weighted in slot
/// order, onto the base (decode rows).
pub(crate) struct RoutedDownTuning {
    pub binding: GeneralRoutedBinding,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedDownCase {
    base: Tensor,
    product: Tensor,
    routes: Tensor,
    weights: Tensor,
    expert_down: Tensor,
}

impl EntryTuning for RoutedDownTuning {
    type Entry = routed_down::Entry;
    type Case = RoutedDownCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn weight_scopes(&self) -> &[WeightScope] {
        &self.scopes
    }

    fn bindings(&self) -> String {
        let b = self.binding;
        format!(
            "EDW={},A={},R={}",
            b.expert_down.name(),
            b.activation.name(),
            sum_element(&b).name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let [_, statics @ ..] = self.binding.shape.expert_dimensions(0);
        Ok(statics.to_vec())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let (shape, rows) = (self.binding.shape, point.rows);
        let table = routes(rows, routing_shape(&self.binding));
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 3 * index as u64;
                Ok(RoutedDownCase {
                    base: inputs.activation(
                        Element::f32(),
                        &[rows, shape.expert_hidden],
                        seed + 1,
                    )?,
                    product: inputs.activation(
                        self.binding.activation,
                        &[rows, shape.selected, shape.intermediate],
                        seed + 2,
                    )?,
                    routes: inputs.i32s(&[rows, shape.selected], &table)?,
                    weights: inputs.activation(
                        Element::f32(),
                        &[rows, shape.selected],
                        seed + 3,
                    )?,
                    expert_down: inputs.weight(scope, WeightKind::ExpertDown)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_down::Args<'a> {
        routed_down::Args {
            base: &case.base,
            product: &case.product,
            routes: &case.routes,
            weights: &case.weights,
            expert_down: &case.expert_down,
        }
    }

    generated_entry!(routed_down, this => routed_down::Elements {
        EDW: this.binding.expert_down,
        A: this.binding.activation,
        R: sum_element(&this.binding),
    });
}

/// The grouped expert tiles: `routed_experts` (gated) or
/// `routed_experts_up`.
pub(crate) struct RoutedExpertTilesTuning {
    pub binding: GeneralRoutedBinding,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedExpertTilesCase {
    normalized: Tensor,
    order: Tensor,
    blocks: Tensor,
    expert_gate: Option<Tensor>,
    expert_up: Tensor,
    expert_down: Tensor,
    up_scale: Tensor,
    activation: i32,
}

impl RoutedExpertTilesTuning {
    fn cases(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<RoutedExpertTilesCase>, String> {
        let (shape, rows) = (self.binding.shape, point.rows);
        let routing = routing_shape(&self.binding);
        let (order, _, table) = group(&routes(rows, routing), rows, routing)?;
        let blocks = table.len() as u64;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                Ok(RoutedExpertTilesCase {
                    normalized: inputs.activation(
                        self.binding.activation,
                        &[rows, shape.expert_hidden],
                        index as u64 + 1,
                    )?,
                    order: inputs.i32s(&[blocks, TILE_ROWS], &order)?,
                    blocks: inputs.i32s(&[blocks], &table)?,
                    expert_gate: if shape.experts_expansion.gated {
                        Some(inputs.weight(scope, WeightKind::ExpertGate)?)
                    } else {
                        None
                    },
                    expert_up: inputs.weight(scope, WeightKind::ExpertUp)?,
                    expert_down: inputs.weight(scope, WeightKind::ExpertDown)?,
                    up_scale: unit_up_scales(inputs, shape.experts)?,
                    activation: shape.experts_expansion.activation,
                })
            })
            .collect()
    }

    fn statics(&self) -> Vec<(&'static str, u64)> {
        let shape = self.binding.shape;
        vec![
            ("H", shape.expert_hidden),
            ("E", shape.experts),
            ("F", shape.intermediate),
        ]
    }
}

/// `routed_experts` over [`RoutedExpertTilesTuning`].
pub(crate) struct RoutedGatedTilesTuning(pub RoutedExpertTilesTuning);

/// `routed_experts_up` over [`RoutedExpertTilesTuning`].
pub(crate) struct RoutedUpTilesTuning(pub RoutedExpertTilesTuning);

impl EntryTuning for RoutedGatedTilesTuning {
    type Entry = routed_experts::Entry;
    type Case = RoutedExpertTilesCase;

    fn launches(&self) -> usize {
        self.0.scopes.len()
    }

    fn weight_scopes(&self) -> &[WeightScope] {
        &self.0.scopes
    }

    fn bindings(&self) -> String {
        let b = self.0.binding;
        format!(
            "EGW={},EUW={},EDW={},A={}",
            b.expert_gate.unwrap_or(b.expert_up).name(),
            b.expert_up.name(),
            b.expert_down.name(),
            b.activation.name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(self.0.statics())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        self.0.cases(inputs, point)
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_experts::Args<'a> {
        routed_experts::Args {
            normalized: &case.normalized,
            order: &case.order,
            blocks: &case.blocks,
            expert_gate: case
                .expert_gate
                .as_ref()
                .expect("a gated tile case binds the gate weight"),
            expert_up: &case.expert_up,
            expert_down: &case.expert_down,
            activation: case.activation,
        }
    }

    generated_entry!(routed_experts, this => routed_experts::Elements {
        EGW: this.0.binding.expert_gate.unwrap_or(this.0.binding.expert_up),
        EUW: this.0.binding.expert_up,
        EDW: this.0.binding.expert_down,
        A: this.0.binding.activation,
    });
}

impl EntryTuning for RoutedUpTilesTuning {
    type Entry = routed_experts_up::Entry;
    type Case = RoutedExpertTilesCase;

    fn launches(&self) -> usize {
        self.0.scopes.len()
    }

    fn weight_scopes(&self) -> &[WeightScope] {
        &self.0.scopes
    }

    fn bindings(&self) -> String {
        let b = self.0.binding;
        format!(
            "EUW={},EDW={},A={}",
            b.expert_up.name(),
            b.expert_down.name(),
            b.activation.name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(self.0.statics())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        self.0.cases(inputs, point)
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_experts_up::Args<'a> {
        routed_experts_up::Args {
            normalized: &case.normalized,
            order: &case.order,
            blocks: &case.blocks,
            expert_up: &case.expert_up,
            expert_down: &case.expert_down,
            up_scale: &case.up_scale,
            activation: case.activation,
        }
    }

    generated_entry!(routed_experts_up, this => routed_experts_up::Elements {
        EUW: this.0.binding.expert_up,
        EDW: this.0.binding.expert_down,
        A: this.0.binding.activation,
    });
}

/// `routed_scatter`: the grouped outputs unpermuted and weighted onto the
/// base (grouped rows).
pub(crate) struct RoutedScatterTuning {
    pub binding: GeneralRoutedBinding,
    /// Layers prepared with this specialization.
    pub layers: usize,
}

pub(crate) struct RoutedScatterCase {
    base: Tensor,
    expert_output: Tensor,
    inverse: Tensor,
    weights: Tensor,
}

impl EntryTuning for RoutedScatterTuning {
    type Entry = routed_scatter::Entry;
    type Case = RoutedScatterCase;

    fn launches(&self) -> usize {
        self.layers
    }

    fn bindings(&self) -> String {
        format!(
            "A={},R={}",
            self.binding.activation.name(),
            sum_element(&self.binding).name()
        )
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        Ok(vec![("H", shape.expert_hidden), ("K", shape.selected)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    /// One argument set: the rows it reads are the step's own outputs.
    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let (shape, rows) = (self.binding.shape, point.rows);
        let routing = routing_shape(&self.binding);
        let (_, inverse, table) = group(&routes(rows, routing), rows, routing)?;
        let blocks = table.len() as u64;
        Ok(vec![RoutedScatterCase {
            base: inputs.activation(Element::f32(), &[rows, shape.expert_hidden], 1)?,
            expert_output: inputs.activation(
                self.binding.activation,
                &[blocks, TILE_ROWS, shape.expert_hidden],
                2,
            )?,
            inverse: inputs.i32s(&[rows, shape.selected], &inverse)?,
            weights: inputs.activation(Element::f32(), &[rows, shape.selected], 3)?,
        }])
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_scatter::Args<'a> {
        routed_scatter::Args {
            base: &case.base,
            expert_output: &case.expert_output,
            inverse: &case.inverse,
            weights: &case.weights,
        }
    }

    generated_entry!(routed_scatter, this => routed_scatter::Elements {
        A: this.binding.activation,
        R: sum_element(&this.binding),
    });
}
