//! Tuning cases of the general routed entries (`operators::routed`):
//! `routed_select` at every row point; `routed_gate_up` / `routed_up` and
//! `routed_down` at the decode points; `routed_experts` /
//! `routed_experts_up` and `routed_scatter` at the grouped points
//! (`routed_group` is tuned by the fused form's case). Routing tables are
//! the fused cases' synthetic tables. The shared expert and latent
//! projections are the dense cases over their own weight kinds.

use super::cases::scale_extent;
use super::routed::{
    BankOperands, ExpansionScales, GroupedTables, RoutedShape, bank_sources, bank_statics,
    decode_points, group, grouped_points, lead_ports, no_sources, no_tier, routes, streamed_hint,
};
use super::{
    CaseState, EntryTuning, ModelInputs, PointShape, TuningInputs, TuningLimits, row_points,
};
use crate::GeneralRoutedBinding;
use crate::operators::routed::fused_graph::TILE_ROWS;
use crate::planning::BankBinding;
use magnitude_family_contracts::{GluLimit, WeightKind, WeightScope};
use magnitude_kernels::{
    routed_down, routed_down_project, routed_experts, routed_experts_up, routed_gate_up, routed_scatter, routed_select,
    routed_select_streamed, routed_up,
};
use seismic::{BuildError, Element, Tensor};

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
#[derive(Clone)]
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

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        let [_, statics @ ..] = shape.select_dimensions(0);
        let down = scale_extent(inputs, &self.scopes, WeightKind::ExpertDown)?;
        Ok([statics.as_slice(), &[("ES", shape.expert_scale_extent(down))]].concat())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
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
                        inputs.unit_scale(inputs.scale_extent(scope, WeightKind::ExpertDown)?)?
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

    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        vec![("routes", &case.routes), ("weights", &case.weights)]
    }

    generated_entry!(routed_select, this => routed_select::Elements {
        NW: this.binding.norm,
        RNW: this.binding.router_norm,
        RW: this.binding.router,
        A: this.binding.activation,
    });
}

/// `routed_select_streamed`: a streamed layer's decode selection, its
/// resident selection's (`routed_select`) decode launches and parameters
/// with the resolve and, at `AHEAD` 1, the next layer's hints. It follows
/// the resident selection's unit (`Specializer::tuned_following`): SPLIT
/// (CUDA), which sets the order the logits' partial planes are summed in,
/// is the resident choice's, so its bits are the resident's; SIMDGROUPS,
/// which changes no bit, is searched for its own launch, which does more
/// than the resident's. Its points are the decode points over the layers
/// that launch it, its operands the resident's over them with the next
/// layer's hinted weights and a bank's ([`BankOperands`]).
#[derive(Clone)]
pub(crate) struct RoutedSelectStreamedTuning {
    /// The resident selection it follows, over every layer of the binding:
    /// its static values are the streamed selection's.
    pub resident: RoutedSelectTuning,
    /// The streamed layers that launch it.
    pub layers: Vec<WeightScope>,
    pub bank: BankBinding,
    pub ahead: u64,
}

pub(crate) struct RoutedSelectStreamedCase {
    select: RoutedSelectCase,
    /// Each lead's router norm, router and selection bias.
    leads: [[Tensor; 3]; crate::planning::HINT_LEADS as usize],
    bank: BankOperands,
}

impl RoutedSelectStreamedTuning {
    fn elements(&self) -> routed_select_streamed::Elements {
        let b = self.resident.binding;
        routed_select_streamed::Elements {
            NW: b.norm,
            RNW: b.router_norm,
            RW: b.router,
            A: b.activation,
        }
    }
}

impl EntryTuning for RoutedSelectStreamedTuning {
    type Entry = routed_select_streamed::Entry;
    type Case = RoutedSelectStreamedCase;

    fn launches(&self) -> usize {
        self.layers.len()
    }

    fn bindings(&self) -> String {
        self.resident.bindings()
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let mut statics = self.resident.statics(inputs)?;
        statics.extend(bank_statics(self.bank, self.ahead));
        Ok(statics)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let mut over = self.resident.clone();
        over.scopes = self.layers.clone();
        // The router's input norm the hints read, as the selection's
        // (`operators::routed::hint_kinds`).
        let router_norm = if self.resident.binding.shape.router_norm {
            WeightKind::RouterInputNorm
        } else {
            WeightKind::InputNorm
        };
        let scopes = TuningInputs::rotation_scopes(&self.layers, point);
        over.rotation(inputs, point)?
            .into_iter()
            .zip(scopes)
            .map(|(select, scope)| {
                let hinted = streamed_hint(inputs, scope)?;
                Ok(RoutedSelectStreamedCase {
                    leads: lead_ports(
                        inputs,
                        &hinted,
                        [router_norm, WeightKind::Router, WeightKind::RouterSelectionBias],
                        [&select.router_norm, &select.router, &select.bias],
                    )?,
                    bank: BankOperands::new(inputs, self.bank, self.ahead, point.rows)?,
                    select,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_select_streamed::Args<'a> {
        let RoutedSelectStreamedCase {
            select,
            leads: [[next_router_norm, next_router, next_bias], [after_router_norm, after_router, after_bias]],
            bank,
        } = case;
        routed_select_streamed::Args {
            residual: &select.residual,
            norm: &select.norm,
            router_norm: &select.router_norm,
            router: &select.router,
            bias: &select.bias,
            expert_scale: &select.expert_scale,
            routes: select.routes.tensor_mut(),
            weights: select.weights.tensor_mut(),
            epsilon: select.epsilon,
            score: select.shape.score,
            normalization: select.shape.normalization,
            normalization_epsilon: f32::from_bits(select.shape.normalization_epsilon),
            scale: f32::from_bits(select.shape.scale),
            map: &bank.map,
            tier: &bank.tier,
            policy: &bank.policy,
            sited: bank.sited,
            step: &bank.step,
            file: bank.file.tensor_mut(),
            grids: bank.grids.tensor_mut(),
            sources: bank.sources.tensor_mut(),
            next_router_norm,
            next_router,
            next_bias,
            after_router_norm,
            after_router,
            after_bias,
            next_policy: &bank.next_policy,
            rings: bank.rings.tensor_mut(),
            routing: bank.routing.tensor_mut(),
        }
    }

    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        let mut states = RoutedSelectTuning::state(&case.select);
        states.extend(case.bank.states());
        states
    }

    generated_entry!(routed_select_streamed, this => this.elements());
}

/// The decode expansion: `routed_gate_up` (gated experts) or `routed_up`.
#[derive(Clone)]
pub(crate) struct RoutedExpandDecodeTuning {
    pub binding: GeneralRoutedBinding,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedExpandDecodeCase {
    normalized: Tensor,
    routes: Tensor,
    /// No sources: every expert read from the stacks by index.
    sources: Tensor,
    expert_gate: Option<Tensor>,
    expert_up: Tensor,
    /// Views of no records of the experts' gate (gated experts) and up.
    tier_gate: Option<Tensor>,
    tier_up: Tensor,
    scales: ExpansionScales,
    activation: i32,
    /// The gated experts' clamp as `routed_gate_up`'s host port `[0, LB]`.
    glu_limit: Tensor,
}


impl RoutedExpandDecodeTuning {
    fn cases(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<RoutedExpandDecodeCase>, BuildError> {
        let (shape, rows) = (self.binding.shape, point.rows);
        let table = routes(rows, routing_shape(&self.binding));
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let expert_gate = if shape.experts_expansion.gated {
                    Some(inputs.weight(scope, WeightKind::ExpertGate)?)
                } else {
                    None
                };
                let expert_up = inputs.weight(scope, WeightKind::ExpertUp)?;
                Ok(RoutedExpandDecodeCase {
                    normalized: inputs.activation(
                        self.binding.activation,
                        &[rows, shape.expert_hidden],
                        index as u64 + 1,
                    )?,
                    routes: inputs.i32s(&[rows, shape.selected], &table)?,
                    sources: no_sources(inputs, rows, shape.selected)?,
                    tier_gate: expert_gate.as_ref().map(no_tier).transpose()?,
                    tier_up: no_tier(&expert_up)?,
                    expert_gate,
                    expert_up,
                    scales: ExpansionScales::at(inputs, scope, shape.experts_expansion.gated)?,
                    activation: shape.experts_expansion.activation,
                    glu_limit: inputs.i32s(&[0, GluLimit::word(shape.experts_expansion.limit)], &[])?,
                })
            })
            .collect()
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        let [_, statics @ ..] = shape.expert_dimensions(0);
        let scales = ExpansionScales::statics(inputs, &self.scopes, shape.experts_expansion.gated)?;
        Ok([statics.as_slice(), &scales].concat())
    }
}

/// `routed_gate_up` over [`RoutedExpandDecodeTuning`].
#[derive(Clone)]
pub(crate) struct RoutedGateUpTuning(pub RoutedExpandDecodeTuning);

/// `routed_up` over [`RoutedExpandDecodeTuning`].
#[derive(Clone)]
pub(crate) struct RoutedUpTuning(pub RoutedExpandDecodeTuning);

impl EntryTuning for RoutedGateUpTuning {
    type Entry = routed_gate_up::Entry;
    type Case = RoutedExpandDecodeCase;

    fn launches(&self) -> usize {
        self.0.scopes.len()
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

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let limit = GluLimit::word(self.0.binding.shape.experts_expansion.limit);
        Ok([self.0.statics(inputs)?, vec![("LB", limit)]].concat())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        self.0.cases(inputs, point)
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_gate_up::Args<'a> {
        routed_gate_up::Args {
            normalized: &case.normalized,
            routes: &case.routes,
            sources: &case.sources,
            expert_gate: case
                .expert_gate
                .as_ref()
                .expect("a gated expansion case binds the gate weight"),
            expert_up: &case.expert_up,
            tier_gate: case
                .tier_gate
                .as_ref()
                .expect("a gated expansion case binds the gate weight"),
            tier_up: &case.tier_up,
            activation: case.activation,
            gate_scale: &case.scales.gate,
            up_scale: &case.scales.up,
            glu_limit: &case.glu_limit,
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

    fn bindings(&self) -> String {
        let b = self.0.binding;
        format!("EUW={},A={}", b.expert_up.name(), b.activation.name())
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        self.0.statics(inputs)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        self.0.cases(inputs, point)
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_up::Args<'a> {
        routed_up::Args {
            normalized: &case.normalized,
            routes: &case.routes,
            sources: &case.sources,
            expert_up: &case.expert_up,
            tier_up: &case.tier_up,
            up_scale: &case.scales.up,
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
#[derive(Clone)]
pub(crate) struct RoutedDownTuning {
    pub binding: GeneralRoutedBinding,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedDownCase {
    base: Tensor,
    product: Tensor,
    routes: Tensor,
    /// No sources: every expert read from the stack by index.
    sources: Tensor,
    weights: Tensor,
    expert_down: Tensor,
    tier_down: Tensor,
    policy: Tensor,
}

impl EntryTuning for RoutedDownTuning {
    type Entry = routed_down::Entry;
    type Case = RoutedDownCase;

    fn launches(&self) -> usize {
        self.scopes.len()
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

    fn statics(&self, _: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
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
    ) -> Result<Vec<Self::Case>, BuildError> {
        let (shape, rows) = (self.binding.shape, point.rows);
        let table = routes(rows, routing_shape(&self.binding));
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 3 * index as u64;
                let expert_down = inputs.weight(scope, WeightKind::ExpertDown)?;
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
                    sources: no_sources(inputs, rows, shape.selected)?,
                    weights: inputs.activation(
                        Element::f32(),
                        &[rows, shape.selected],
                        seed + 3,
                    )?,
                    tier_down: no_tier(&expert_down)?,
                    expert_down,
                    policy: inputs.i32s(
                        &[magnitude_kernels::constants::POLICY_WORDS],
                        &[0; magnitude_kernels::constants::POLICY_WORDS as usize],
                    )?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_down::Args<'a> {
        routed_down::Args {
            base: &case.base,
            product: &case.product,
            routes: &case.routes,
            sources: &case.sources,
            weights: &case.weights,
            expert_down: &case.expert_down,
            tier_down: &case.tier_down,
            policy: &case.policy,
        }
    }

    generated_entry!(routed_down, this => routed_down::Elements {
        EDW: this.binding.expert_down,
        A: this.binding.activation,
        R: sum_element(&this.binding),
    });
}

/// `routed_down_project`: the device's projections of a streamed layer's
/// general combine, its resident down's (`routed_down`) launches and
/// parameters publishing the projections for `routed_down_sited`, following
/// the resident down's unit as `RoutedOutputProjectTuning` follows its
/// output.
#[derive(Clone)]
pub(crate) struct RoutedDownProjectTuning {
    /// The resident down it follows, over every layer of the binding.
    pub resident: RoutedDownTuning,
    /// The streamed layers that launch it.
    pub layers: Vec<WeightScope>,
}

impl EntryTuning for RoutedDownProjectTuning {
    type Entry = routed_down_project::Entry;
    type Case = RoutedDownCase;

    fn launches(&self) -> usize {
        self.layers.len()
    }

    fn bindings(&self) -> String {
        self.resident.bindings()
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        self.resident.statics(inputs)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        decode_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        let mut over = self.resident.clone();
        over.scopes = self.layers.clone();
        let shape = self.resident.binding.shape;
        let table = routes(point.rows, routing_shape(&self.resident.binding));
        over.rotation(inputs, point)?
            .into_iter()
            .map(|mut down| {
                down.sources = bank_sources(inputs, point.rows, shape.selected, &table)?;
                Ok(down)
            })
            .collect()
    }

    fn args<'a>(down: &'a mut Self::Case) -> routed_down_project::Args<'a> {
        routed_down_project::Args {
            product: &down.product,
            routes: &down.routes,
            sources: &down.sources,
            expert_down: &down.expert_down,
            tier_down: &down.tier_down,
        }
    }

    generated_entry!(routed_down_project, this => routed_down_project::Elements {
        EDW: this.resident.binding.expert_down,
        A: this.resident.binding.activation,
    });
}

/// The grouped expert tiles: `routed_experts` (gated) or
/// `routed_experts_up`.
#[derive(Clone)]
pub(crate) struct RoutedExpertTilesTuning {
    pub binding: GeneralRoutedBinding,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct RoutedExpertTilesCase {
    tables: GroupedTables,
    normalized: Tensor,
    order: Tensor,
    blocks: Tensor,
    expert_gate: Option<Tensor>,
    expert_up: Tensor,
    expert_down: Tensor,
    scales: ExpansionScales,
    activation: i32,
    /// The gated experts' clamp as `routed_experts`' host port `[0, LB]`.
    glu_limit: Tensor,
}

impl RoutedExpertTilesTuning {
    fn cases(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<RoutedExpertTilesCase>, BuildError> {
        let (shape, rows) = (self.binding.shape, point.rows);
        let routing = routing_shape(&self.binding);
        let (order, _, table) = group(&routes(rows, routing), rows, routing)?;
        let blocks = table.len() as u64;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                Ok(RoutedExpertTilesCase {
                    tables: GroupedTables::of(inputs, rows, &table)?,
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
                    scales: ExpansionScales::at(inputs, scope, shape.experts_expansion.gated)?,
                    activation: shape.experts_expansion.activation,
                    glu_limit: inputs.i32s(&[0, GluLimit::word(shape.experts_expansion.limit)], &[])?,
                })
            })
            .collect()
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        let mut statics = vec![
            ("H", shape.expert_hidden),
            ("E", shape.experts),
            ("F", shape.intermediate),
        ];
        statics.extend(ExpansionScales::statics(
            inputs,
            &self.scopes,
            shape.experts_expansion.gated,
        )?);
        Ok(statics)
    }
}

/// `routed_experts` over [`RoutedExpertTilesTuning`].
#[derive(Clone)]
pub(crate) struct RoutedGatedTilesTuning(pub RoutedExpertTilesTuning);

/// `routed_experts_up` over [`RoutedExpertTilesTuning`].
#[derive(Clone)]
pub(crate) struct RoutedUpTilesTuning(pub RoutedExpertTilesTuning);

impl EntryTuning for RoutedGatedTilesTuning {
    type Entry = routed_experts::Entry;
    type Case = RoutedExpertTilesCase;

    fn launches(&self) -> usize {
        self.0.scopes.len()
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

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let limit = GluLimit::word(self.0.binding.shape.experts_expansion.limit);
        Ok([self.0.statics(inputs)?, vec![("LB", limit)]].concat())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        self.0.cases(inputs, point)
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_experts::Args<'a> {
        let gate = case
            .expert_gate
            .as_ref()
            .expect("a gated tile case binds the gate weight");
        routed_experts::Args {
            first: &case.tables.first,
            rows: &case.tables.rows,
            normalized: &case.normalized,
            order: &case.order,
            blocks: &case.blocks,
            used: &case.tables.used,
            expert_gate: gate,
            expert_up: &case.expert_up,
            expert_down: &case.expert_down,
            map: &case.tables.map,
            step: &case.tables.step,
            records_gate: gate,
            records_up: &case.expert_up,
            records_down: &case.expert_down,
            activation: case.activation,
            gate_scale: &case.scales.gate,
            up_scale: &case.scales.up,
            glu_limit: &case.glu_limit,
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

    fn bindings(&self) -> String {
        let b = self.0.binding;
        format!(
            "EUW={},EDW={},A={}",
            b.expert_up.name(),
            b.expert_down.name(),
            b.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        self.0.statics(inputs)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        grouped_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
        self.0.cases(inputs, point)
    }

    fn args<'a>(case: &'a mut Self::Case) -> routed_experts_up::Args<'a> {
        routed_experts_up::Args {
            first: &case.tables.first,
            rows: &case.tables.rows,
            normalized: &case.normalized,
            order: &case.order,
            blocks: &case.blocks,
            used: &case.tables.used,
            expert_up: &case.expert_up,
            expert_down: &case.expert_down,
            map: &case.tables.map,
            step: &case.tables.step,
            records_up: &case.expert_up,
            records_down: &case.expert_down,
            up_scale: &case.scales.up,
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
#[derive(Clone)]
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

    fn statics(&self, _: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
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
    ) -> Result<Vec<Self::Case>, BuildError> {
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
