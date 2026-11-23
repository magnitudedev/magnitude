//! Routed (mixture-of-experts) feed-forward graph construction (program spec
//! K6, E8).
//!
//! Row classes up to the kernels' `VERIFY_ROWS` run the decode form ([`DecodeForm`]):
//! route, expand, output; or, on a backend that declares
//! `routed_route_shared`, the route with the shared expert's gate/up, the
//! choices' gate/up, output. Larger classes run the grouped form: route,
//! group, experts, combine. The grouping tables and grouped intermediates
//! are graph locals sized from the class, the selected-expert count and
//! [`TILE_ROWS`], so the graph plan charges them to the workspace; nothing is
//! uploaded per step.

use seismic::LaunchEncoding;
use crate::operators::{Join, SublayerInput};
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use super::grouped::{ExpertsPass, Grouped};
use super::split::{BackCrossing, Crossing, PassPorts, SharedInput, Split};
use super::streamed::{
    BankCombineEntry, BankForm, BankNumbers, BankRoute, LayerBank, DecodeReads, ResidentGateUp, RouteLocals, RoutedStep, StepChoices, StepInputs,
    StreamedLayer,
};
use crate::planning::RecordTensors;
use crate::programs::graph::facts::Numbers;
use crate::programs::native_target_graph::{
    PlannedWeights, WeightPart, WeightPort, WeightScale, weight, weight_port,
};
use magnitude_kernels::{routed_hint_route, routed_output_project, routed_output_sited, routed_route_streamed};
use crate::{
    ModelLoadPlan, RoutedBinding,
    planning::BankPart,
    native::{RoutedDecodeKernels, RoutedKernels},
};
use magnitude_family_contracts::{
    RouteNormalization, RoutedFfn, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    routed_combine, routed_expand, routed_experts, routed_gate_up, routed_group, routed_output,
    routed_route, routed_route_shared,
};
#[cfg(test)]
use seismic::NativeGraphMetadata;
use seismic::{
    Argument, BackendName, Dim, Element, NativeGraph, NativePort, OperandCarry, WorkflowTensor,
};

/// The choices' gate/up arguments (`routed_gate_up`: SiLU and unclamped,
/// the fused form's experts, `is_fused`) of `normalized` over `routes`,
/// each choice read where `reads` names (`tier_gate` its gate tier), from
/// the gate and up `stacks` at their `scales`; `glu_limit` the unclamped
/// host port.
fn choices_args<'t>(
    normalized: &'t WorkflowTensor,
    routes: &'t WorkflowTensor,
    reads: &'t DecodeReads,
    tier_gate: &'t seismic::WorkflowTensorView,
    [gate, up]: [&'t WorkflowTensor; 2],
    [gate_scale, up_scale]: [&'t WeightScale; 2],
    glu_limit: &'t seismic::WorkflowTensorView,
) -> routed_gate_up::WorkflowArgs<'t> {
    routed_gate_up::WorkflowArgs {
        normalized: normalized.into(),
        routes: routes.into(),
        sources: (&reads.sources).into(),
        expert_gate: gate.into(),
        expert_up: up.into(),
        tier_gate: tier_gate.into(),
        tier_up: (&reads.tier.up).into(),
        activation: (super::FUSED_ACTIVATION).into(),
        gate_scale: (&gate_scale.tensor).into(),
        up_scale: (&up_scale.tensor).into(),
        glu_limit: glu_limit.into(),
    }
}

/// A fused layer's gate tier records, which its reads hold.
fn tier_gate(reads: &DecodeReads) -> Result<&seismic::WorkflowTensorView, GraphError> {
    Ok(reads.tier.gate.as_ref().ok_or("a fused layer's reads lack its gate")?)
}

/// How a backend runs the decode rows' expansions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeForm {
    /// `routed_route`, then `routed_expand` over the choices and the shared
    /// expert.
    Expand,
    /// `routed_route_shared`, whose launch also expands the shared expert
    /// (it does not depend on the routes), then `routed_gate_up` (SiLU) over
    /// the choices.
    SharedRoute,
}

impl DecodeForm {
    /// The shared-route form wherever the checked bundle declares
    /// `routed_route_shared` for `backend`, else the expand form.
    pub(crate) fn of(backend: BackendName) -> Result<Self, String> {
        Ok(
            if <routed_route_shared::Entry as seismic::EntryContract>::contract_node_shape(backend)
                .is_ok()
            {
                Self::SharedRoute
            } else {
                Self::Expand
            },
        )
    }
}

/// A fused routed layer's entries; each projection's by row range.
pub(crate) struct RoutedGraphEntries<'a, G: GraphDraft + 'a> {
    pub route: G::Binding<'a, routed_route::Entry>,
    pub decode: RoutedDecodeEntries<'a, G>,
    pub output: G::Ranged<'a, routed_output::Entry>,
    /// The element `output` binds its operand norm at (`NW`): the layer's
    /// own norm's (`RoutedBinding::norm`), the absent norm's where the layer
    /// forms no operand for another graph.
    pub operand_norm: Element,
    pub group: G::Binding<'a, routed_group::Entry>,
    pub experts: G::Ranged<'a, routed_experts::Entry>,
    pub combine: G::Ranged<'a, routed_combine::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for RoutedGraphEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for RoutedGraphEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The decode rows' routing and expansion entries of a [`DecodeForm`].
pub(crate) enum RoutedDecodeEntries<'a, G: GraphDraft + 'a> {
    Expand(G::Ranged<'a, routed_expand::Entry>),
    SharedRoute {
        route: G::Ranged<'a, routed_route_shared::Entry>,
        choices: G::Ranged<'a, routed_gate_up::Entry>,
    },
}

impl<'a, G: GraphDraft + 'a> Copy for RoutedDecodeEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for RoutedDecodeEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a> From<&'a RoutedKernels> for RoutedGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a RoutedKernels) -> Self {
        Self {
            route: &kernels.route,
            decode: match &kernels.decode {
                RoutedDecodeKernels::Expand(expand) => RoutedDecodeEntries::Expand(expand),
                RoutedDecodeKernels::SharedRoute { route, choices } => {
                    RoutedDecodeEntries::SharedRoute { route, choices }
                }
            },
            output: &kernels.output,
            operand_norm: kernels.operand_norm,
            group: &kernels.group,
            experts: &kernels.experts,
            combine: &kernels.combine,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedRoutedEntries {
    route: [(&'static str, Element); 4],
    decode: CheckedDecodeEntries,
    output: [(&'static str, Element); 4],
    group: [(&'static str, Element); 0],
    experts: [(&'static str, Element); 4],
    combine: [(&'static str, Element); 4],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CheckedDecodeEntries {
    Expand([(&'static str, Element); 5]),
    SharedRoute {
        route: [(&'static str, Element); 6],
        choices: [(&'static str, Element); 3],
    },
}

impl CheckedRoutedEntries {
    /// The entries of `binding` on `backend` (its [`DecodeForm`]).
    pub(crate) fn new(binding: RoutedBinding, backend: BackendName) -> Result<Self, String> {
        Ok(Self {
            route: [
                ("NW", binding.norm),
                ("RW", binding.router),
                ("SRW", Element::f32()),
                ("A", binding.activation),
            ],
            decode: match DecodeForm::of(backend)? {
                DecodeForm::Expand => CheckedDecodeEntries::Expand([
                    ("EGW", binding.expert_gate),
                    ("EUW", binding.expert_up),
                    ("SGW", binding.shared_gate),
                    ("SUW", binding.shared_up),
                    ("A", binding.activation),
                ]),
                DecodeForm::SharedRoute => CheckedDecodeEntries::SharedRoute {
                    route: [
                        ("NW", binding.norm),
                        ("RW", binding.router),
                        ("SRW", Element::f32()),
                        ("SGW", binding.shared_gate),
                        ("SUW", binding.shared_up),
                        ("A", binding.activation),
                    ],
                    choices: [
                        ("EGW", binding.expert_gate),
                        ("EUW", binding.expert_up),
                        ("A", binding.activation),
                    ],
                },
            },
            output: [
                ("EDW", binding.expert_down),
                ("SDW", binding.shared_down),
                ("A", binding.activation),
                ("NW", binding.norm),
            ],
            group: [],
            experts: [
                ("EGW", binding.expert_gate),
                ("EUW", binding.expert_up),
                ("EDW", binding.expert_down),
                ("A", binding.activation),
            ],
            combine: [
                ("SGW", binding.shared_gate),
                ("SUW", binding.shared_up),
                ("SDW", binding.shared_down),
                ("A", binding.activation),
            ],
        })
    }

    pub(crate) fn entries<G: crate::programs::graph::draft::ElementBoundDraft>(
        &self,
    ) -> RoutedGraphEntries<'_, G> {
        RoutedGraphEntries {
            route: G::bind(&self.route),
            decode: match &self.decode {
                CheckedDecodeEntries::Expand(expand) => RoutedDecodeEntries::Expand(G::bind_ranged(&expand[..])),
                CheckedDecodeEntries::SharedRoute { route, choices } => {
                    RoutedDecodeEntries::SharedRoute {
                        route: G::bind_ranged(&route[..]),
                        choices: G::bind_ranged(&choices[..]),
                    }
                }
            },
            output: G::bind_ranged(&self.output),
            operand_norm: self.output[3].1,
            group: G::bind(&self.group),
            experts: G::bind_ranged(&self.experts),
            combine: G::bind_ranged(&self.combine),
        }
    }
}

/// The operand a fused layer's output forms for the next block's input
/// projection, in another graph (`routed_output`'s `operand_norm`): the
/// carry it writes (`GraphDraft::carry_out`), in the forms `F` its graph
/// names it in (`GraphDraft::Forms`), and the norm and epsilon that
/// projection normalizes under. Only a decode layer combining resident
/// forms one.
pub(crate) struct OperandOut<'c, F> {
    pub carry: &'c OperandCarry,
    pub forms: F,
    pub norm: &'c WorkflowTensor,
    pub epsilon: f32,
}

/// What decides a fused routed layer's graph beside its numbers: how it
/// reads its experts.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RoutedTopology {
    pub bank: BankForm,
}

/// The numbers the routed entries are specialized to: the residual's
/// width, the experts, the selected experts, their features and the shared
/// expert's, whether the route normalizes the selected weights (`1`, else
/// `0`: `routed_route`'s `normalize`), and a streamed layer's bank's.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ExpertShape<D = Dim> {
    pub hidden: D,
    pub count: D,
    pub selected: D,
    pub intermediate: D,
    pub shared_intermediate: D,
    pub normalize: D,
    pub bank: Option<BankNumbers<D>>,
}

impl<D> Numbers<D> for ExpertShape<D> {
    type Of<E> = ExpertShape<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> ExpertShape<E> {
        ExpertShape {
            hidden: f(self.hidden),
            count: f(self.count),
            selected: f(self.selected),
            intermediate: f(self.intermediate),
            shared_intermediate: f(self.shared_intermediate),
            normalize: f(self.normalize),
            bank: self.bank.map(|bank| bank.map(&mut *f)),
        }
    }
}

impl ExpertShape {
    /// `routed_route`'s `normalize` argument.
    fn normalize(&self) -> Result<Argument<i32>, GraphError> {
        crate::operators::scalar(&self.normalize, "a route's normalize code")
    }
}

/// What a fused routed layer's numbers are read from: a program's binding,
/// or an admitted operator over a `hidden`-wide residual (`operators::admit`
/// guarantees its shared expert).
pub(crate) enum RoutedSource<'r> {
    Binding(&'r RoutedBinding),
    Operator { routed: &'r RoutedFfn, hidden: u64 },
}

/// The topology and numbers of the fused routed layer of `scope` read from
/// `source`, whose graph reads its experts as `bank` does.
pub(crate) fn split<G: GraphDraft>(
    load: &ModelLoadPlan,
    scope: WeightScope,
    source: RoutedSource<'_>,
    bank: &LayerBank<'_, '_, G>,
) -> Result<(RoutedTopology, ExpertShape<u64>), GraphError> {
    let (bank, numbers) = super::streamed::split(load, scope, bank)?;
    let shape = match source {
        RoutedSource::Binding(binding) => ExpertShape {
            hidden: binding.hidden,
            count: binding.experts,
            selected: binding.selected,
            intermediate: binding.features,
            shared_intermediate: binding.shared,
            normalize: u64::from(binding.normalize_selected),
            bank: numbers,
        },
        RoutedSource::Operator { routed, hidden } => ExpertShape {
            hidden,
            count: routed.experts,
            selected: routed.selected,
            intermediate: routed.intermediate,
            shared_intermediate: routed
                .shared
                .as_ref()
                .ok_or("routed feed-forward without shared expert")?
                .intermediate,
            normalize: u64::from(routed.router.normalization == RouteNormalization::Sum),
            bank: numbers,
        },
    };
    Ok((RoutedTopology { bank }, shape))
}

/// Whether a `rows`-row class runs the decode form.
pub(crate) fn decodes(rows: impl Into<Dim>) -> bool {
    rows.into().at_most(magnitude_kernels::constants::VERIFY_ROWS)
}

/// Rows of one expert tile of the grouped form.
pub(crate) const TILE_ROWS: u64 = 32;

/// Expert tiles that hold every choice of a `rows`-row class: `rows *
/// selected` choices, the rows of each chosen expert padded to whole tiles.
/// At most `min(experts, rows * selected)` experts are chosen, so
/// `ceil((rows * selected + min(experts, rows * selected) * (TILE_ROWS - 1))
/// / TILE_ROWS)`.
pub(crate) fn grouped_blocks(
    rows: impl Into<Dim>,
    experts: impl Into<Dim>,
    selected: impl Into<Dim>,
) -> Result<Dim, String> {
    let rows = rows.into();
    let overflow = || format!("grouped tiles of {rows:?} rows overflow");
    let choices = rows.checked_mul(selected.into()).ok_or_else(overflow)?;
    experts
        .into()
        .min(&choices)
        .checked_mul(TILE_ROWS - 1)
        .and_then(|padding| choices.checked_add(padding))
        .map(|slots| slots.ceil_div(TILE_ROWS))
        .ok_or_else(overflow)
}

/// The first row and row count `routed_group` groups in a `rows`-row class
/// of a graph that groups its own rows: every row, from row 0.
pub(crate) fn class_rows<G: GraphDraft>(
    graph: &mut G,
    rows: &Dim,
    constants: &mut Vec<GraphConstant>,
) -> Result<[WorkflowTensor; 2], GraphError> {
    let [first, count] = [GraphConstant::i32(graph, &[0])?, GraphConstant::row_count(graph, rows)?];
    let tensors = [first.port().tensor().clone(), count.port().tensor().clone()];
    constants.extend([first, count]);
    Ok(tensors)
}

/// The map the grouped expert entries bind: none (X = 0), so every
/// expert is read from the stacks by index, which also bind as the records
/// (R = E) they never read.
pub(crate) fn absent_map<G: GraphDraft>(
    graph: &mut G,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    GraphConstant::empty_of(graph, Element::i32(), &[0, magnitude_kernels::constants::MAP_WORDS], constants)
}

/// A fused layer's route weights: its input norm (empty for a mix site's
/// partials), router and shared router, and the experts' down scale the
/// route folds into the scores (absent where they have none).
struct RouteWeights {
    norm: WorkflowTensor,
    router: WorkflowTensor,
    shared_router: WorkflowTensor,
    expert_scale: WeightScale,
}

impl RouteWeights {
    fn of<G: GraphDraft>(
        graph: &mut G,
        planned: &(impl PlannedWeights + ?Sized),
        scope: WeightScope,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
        input: SublayerInput<'_>,
    ) -> Result<Self, GraphError> {
        let norm = match input.form.normalized {
            true => weight(graph, planned, scope, WeightKind::InputNorm, weights)?,
            false => GraphConstant::absent(graph, constants)?.tensor().clone(),
        };
        Ok(Self {
            norm,
            router: weight(graph, planned, scope, WeightKind::Router, weights)?,
            shared_router: weight(graph, planned, scope, WeightKind::SharedRouter, weights)?,
            expert_scale: WeightScale::per_expert(
                graph,
                planned,
                WeightRole { scope, kind: WeightKind::ExpertDown },
                weights,
                constants,
            )?,
        })
    }

    /// The route's dimensions over `rows` rows of `input`.
    fn choices(&self, input: SublayerInput<'_>, rows: &Dim, shape: &ExpertShape) -> [(&'static str, Dim); 7] {
        let [parts, norm] = input.form.dimensions(shape.hidden.clone());
        [
            ("M", rows.clone()),
            ("H", shape.hidden.clone()),
            ("E", shape.count.clone()),
            ("K", shape.selected.clone()),
            parts,
            norm,
            ("ES", self.expert_scale.extent.clone()),
        ]
    }

    /// The grouped form's route of `input` into `routes` and `scores`: the
    /// normalized rows and the shared expert's coefficient.
    #[allow(clippy::too_many_arguments)]
    fn route<'a, G: GraphDraft + 'a>(
        &self,
        graph: &mut G,
        entry: G::Binding<'a, routed_route::Entry>,
        input: SublayerInput<'_>,
        routes: &mut NativePort,
        scores: &mut NativePort,
        rows: &Dim,
        shape: &ExpertShape,
        epsilon: Argument<f32>,
    ) -> Result<(WorkflowTensor, WorkflowTensor), GraphError> {
        let routed = graph.enqueue(
            entry,
            &self.choices(input, rows, shape),
            routed_route::WorkflowArgs {
                residual: input.rows.into(),
                norm: (&self.norm).into(),
                router: (&self.router).into(),
                shared_router: (&self.shared_router).into(),
                routes: routes.tensor_mut().into(),
                scores: scores.tensor_mut().into(),
                eps: epsilon.into(),
                normalize: shape.normalize()?,
                expert_scale: (&self.expert_scale.tensor).into(),
            },
        )?;
        Ok((routed.r0, routed.r1))
    }
}

/// A fused layer's routed expert weights: its expert stacks (a streamed
/// layer's its bank group's records, which its fills write, or in its
/// resident form its home records among them) and the experts'
/// gate and up per-expert scales on the expansions' accumulators (absent
/// where they have none). Its shared expert's are `shared_weights`.
struct ExpertWeights {
    stacks: [NativePort; 3],
    gate_scale: WeightScale,
    up_scale: WeightScale,
}

impl ExpertWeights {
    /// The layer's stacks, each its `part` of the planned stack
    /// (`LayerBank::stacks`), and their scales.
    fn of<G: GraphDraft>(
        graph: &mut G,
        planned: &(impl PlannedWeights + ?Sized),
        scope: WeightScope,
        weights: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
        part: WeightPart,
    ) -> Result<Self, GraphError> {
        let role = |kind| WeightRole { scope, kind };
        let mut stack = |kind| weight_port(graph, planned, role(kind), part, weights);
        let stacks = [
            stack(WeightKind::ExpertGate)?,
            stack(WeightKind::ExpertUp)?,
            stack(WeightKind::ExpertDown)?,
        ];
        Ok(Self {
            stacks,
            gate_scale: WeightScale::per_expert(graph, planned, role(WeightKind::ExpertGate), weights, constants)?,
            up_scale: WeightScale::per_expert(graph, planned, role(WeightKind::ExpertUp), weights, constants)?,
        })
    }

    /// The expert stacks' tensors.
    fn stacks(&self) -> [WorkflowTensor; 3] {
        self.stacks.each_ref().map(|stack| stack.tensor().clone())
    }
}

/// The routed feed-forward of one block over `rows` rows of `residual`, and
/// the sequence input a run of a streamed layer writes. A layer the load
/// streams takes its `bank` entries (`streamed`); a scratch-served one is
/// built as its front or its back (`split`), a front's output the residual
/// it read; its resident form (`LayerBank::Home`) decodes as a resident
/// layer does, over its whole bank's home records.
/// M7 split decode front for a fused streamed layer: route + hints +
/// file-miss landing (`routed_land`, when `needs_front_land`) + inline site
/// publish. Extracted out of [`routed`] to keep its stack frame bounded.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn decode_front<'a, G: GraphDraft + 'a, W: PlannedWeights + ?Sized>(
    graph: &mut G,
    layer: &mut StreamedLayer<'_, 'a, G, W>,
    part: WeightPart,
    shape: &ExpertShape,
    planned: &W,
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    input: SublayerInput<'_>,
    residual: &WorkflowTensor,
    rows: &Dim,
    epsilon: Argument<f32>,
    crossing: &mut super::split::DecodeCrossing<NativePort>,
) -> Result<(WorkflowTensor, Option<StepInputs>), GraphError> {
    let route_weights = RouteWeights::of(graph, planned, scope, weights, constants, input)?;
    let [parts, norm_extent] = input.form.dimensions(shape.hidden.clone());
    let RouteWeights { norm, router, shared_router, expert_scale } = &route_weights;
    let shared_gate = weight_port(
        graph,
        planned,
        WeightRole { scope, kind: WeightKind::SharedGate },
        WeightPart::Values,
        weights,
    )?
    .tensor()
    .clone();
    let shared_up = weight_port(
        graph,
        planned,
        WeightRole { scope, kind: WeightKind::SharedUp },
        WeightPart::Values,
        weights,
    )?
    .tensor()
    .clone();
    let normalize = shape.normalize()?;
    let bank = layer.numbers();
    let mut ports = layer.decode_front_ports(graph, rows, constants)?;
    let mut routing = layer.part(graph, BankPart::Routing)?;
    let mut front = layer.part(graph, BankPart::Front)?;
    let tier = layer.front_route_tier(graph, &ports, constants)?;
    let mut keys = layer.route_keys(graph)?;
    let (sf, sr) = match &ports.site {
        Some(_) => (bank.site_slots.clone(), bank.site_width.clone()),
        None => (Dim::from(0), Dim::from(0)),
    };
    let mut empty_site = ports
        .site
        .is_none()
        .then(|| GraphConstant::empty_port(graph, Element::i32(), &[Dim::from(0), Dim::from(0)], constants))
        .transpose()?;
    let mut empty_site_rows = ports
        .site
        .is_none()
        .then(|| {
            GraphConstant::empty_port(
                graph,
                Element::f32(),
                &[Dim::from(0), Dim::from(magnitude_kernels::constants::VERIFY_ROWS), shape.hidden.clone()],
                constants,
            )
        })
        .transpose()?;
    let route_dims = [
        ("M", rows.clone()),
        ("H", shape.hidden.clone()),
        ("E", shape.count.clone()),
        ("K", shape.selected.clone()),
        ("S", shape.shared_intermediate.clone()),
        parts.clone(),
        norm_extent.clone(),
        ("ES", expert_scale.extent.clone()),
        ("FW", bank.file_width.clone()),
        ("FLIGHT", bank.flight.clone()),
        ("RING", bank.ring_words.clone()),
        ("P", bank.choices.clone()),
        ("GM", bank.map_words.clone()),
        ("TM", tier.experts.clone()),
        ("SF", sf),
        ("SR", sr),
    ];
    let BankRoute::Fused { activation, route, hints: hint_kernels } = layer.route() else {
        return Err("a fused layer's bank holds another form's route".into());
    };
    let route = G::pick(route, rows);
    let sited_mode = if ports.site.is_some() {
        if layer.needs_front_land() { 1 } else { 2 }
    } else {
        0
    };
    let (site_ref, site_rows_ref) = match ports.site.as_mut() {
        Some(site) => (site.table.tensor_mut().into(), site.rows.tensor_mut().into()),
        None => (
            empty_site.as_mut().unwrap().tensor_mut().into(),
            empty_site_rows.as_mut().unwrap().tensor_mut().into(),
        ),
    };
    let routed = graph.enqueue(
        route,
        &route_dims,
        routed_route_streamed::WorkflowArgs {
            residual: input.rows.into(),
            norm: norm.into(),
            router: router.into(),
            shared_router: shared_router.into(),
            shared_gate: (&shared_gate).into(),
            shared_up: (&shared_up).into(),
            routes: crossing.routes.tensor_mut().into(),
            scores: crossing.weights.tensor_mut().into(),
            eps: epsilon.into(),
            normalize,
            expert_scale: (&expert_scale.tensor).into(),
            map: ports.map.tensor().into(),
            tier: (&tier.map).into(),
            policy: ports.policy.tensor().into(),
            sited: sited_mode.into(),
            step: ports.step.tensor().into(),
            file: ports.file.tensor_mut().into(),
            sources: crossing.sources.tensor_mut().into(),
            routing: routing.tensor_mut().into(),
            keys: keys.tensor_mut().into(),
            front: front.tensor_mut().into(),
            land_grid: ports.land_grid.tensor_mut().into(),
            land_groups: layer.land_groups().into(),
            site: site_ref,
            site_rows: site_rows_ref,
        },
    )?;
    if let Some(mut hints) = layer.hints(graph, weights, [norm, router], ports.policy.tensor())? {
        let hint = hint_kernels[hints.ahead as usize - 1]
            .ok_or("a streamed layer's lead was not prepared with its binding")?;
        let empty = GraphConstant::empty_of(graph, activation, &[Dim::from(0u64), shape.selected.clone()], constants)?;
        let hint_dims = [
            ("M", rows.clone()),
            ("H", shape.hidden.clone()),
            parts.clone(),
            norm_extent.clone(),
            ("K", shape.selected.clone()),
            ("AHEAD", Dim::from(hints.ahead)),
            ("RING", bank.ring_words.clone()),
            ("EN", bank.next_experts.clone()),
        ];
        graph.enqueue_under(
            hint,
            &hint_dims,
            routed_hint_route::WorkflowArgs {
                residual: input.rows.into(),
                activation: (&empty).into(),
                next_norm: (&hints.weights[0][0]).into(),
                next_router: (&hints.weights[0][1]).into(),
                after_norm: (&hints.weights[1][0]).into(),
                after_router: (&hints.weights[1][1]).into(),
                eps: epsilon.into(),
                step: ports.step.tensor().into(),
                next_policy: (&hints.policy).into(),
                rings: hints.rings.tensor_mut().into(),
            },
            &ports.policy,
            magnitude_kernels::constants::POLICY_HINTS,
            LaunchEncoding::WhileSet,
        )?;
    }
    let mut sources_tensor = crossing.sources.tensor().clone();
    if layer.needs_front_land() {
        let mut gate_records = weight_port(
            graph,
            planned,
            WeightRole { scope, kind: WeightKind::ExpertGate },
            part,
            weights,
        )?;
        let mut up_records = weight_port(
            graph,
            planned,
            WeightRole { scope, kind: WeightKind::ExpertUp },
            part,
            weights,
        )?;
        let mut down_records = weight_port(
            graph,
            planned,
            WeightRole { scope, kind: WeightKind::ExpertDown },
            part,
            weights,
        )?;
        let site_choices = StepChoices {
            routes: crossing.routes.tensor(),
            weights: crossing.weights.tensor(),
            input: &routed.r0,
            rows,
        };
        layer.land_front(
            graph,
            &mut ports,
            &mut sources_tensor,
            RecordTensors {
                gate: Some(&mut gate_records),
                up: &mut up_records,
                down: &mut down_records,
            },
            Some(&site_choices),
            &super::streamed::DecodeShape { rows },
            constants,
        )?;
    }
    graph.store(&routed.r0, &mut crossing.normalized)?;
    let Some(into_coeff) = crossing.coefficient.as_mut() else {
        return Err("a fused decode front crosses no coefficient".into());
    };
    graph.store(&routed.r1, into_coeff)?;
    let Some(into_shared) = crossing.shared_product.as_mut() else {
        return Err("a fused decode front crosses no shared product".into());
    };
    graph.store(&routed.r2, into_shared)?;
    layer.publish_site(graph);
    Ok((residual.clone(), ports.inputs.take()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn routed<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    handle: RoutedGraphEntries<'a, G>,
    bank: LayerBank<'_, 'a, G>,
    topology: &RoutedTopology,
    shape: &ExpertShape,
    planned: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    input: SublayerInput<'_>,
    residual: &WorkflowTensor,
    join: &Join,
    rows: &Dim,
    epsilon: Argument<f32>,
    split: Option<Split<'_>>,
    operand: Option<OperandOut<'_, G::Forms>>,
) -> Result<(WorkflowTensor, Option<StepInputs>), GraphError> {
    let part = bank.stacks();
    let mut streamed = StreamedLayer::of(planned, &topology.bank, shape.bank.as_ref(), scope, bank)?;
    let decode = decodes(rows);
    if operand.is_some()
        && (!decode || matches!(split, Some(Split::Front(_) | Split::Combine(..) | Split::DecodeFront(_))))
    {
        return Err("a routed layer forms an operand only at a whole or back decode".into());
    }
    let scratch = streamed.is_some() && !decode;
    let decode_back = match (split, &mut streamed) {
        (None, _) if !scratch => None,
        // A front routes into its crossing; a scratch-served one requests
        // its experts (none while its layer is whole).
        (Some(Split::Front(crossing)), layer) if !decode => {
            let route = RouteWeights::of(graph, planned, scope, weights, constants, input)?;
            let Crossing { routes, weights: scores, normalized: into, coefficient: Some(into_coefficient), .. } = crossing
            else {
                return Err("a fused layer's front crosses no coefficient".into());
            };
            let (normalized, coefficient) =
                route.route(graph, handle.route, input, routes, scores, rows, shape, epsilon)?;
            graph.store(&normalized, into)?;
            graph.store(&coefficient, into_coefficient)?;
            let sequence = match layer {
                Some(layer) => layer.request(graph, routes.tensor(), rows, constants)?,
                None => None,
            };
            return Ok((residual.clone(), sequence));
        }
        // A back after its merge set's pass: its rows combined from the
        // pass's.
        (Some(Split::Combine(back, blocks)), _) if !decode => {
            let shared = shared_weights(graph, planned, scope, weights)?;
            let combined = Combined::of(&back.crossing)?;
            let output = combine(graph, &handle, &shared, &back.pass(blocks), combined, residual, join, rows, shape, constants)?;
            return Ok((output, None));
        }
        // M7 split decode front: route + hints + file-miss landing (`routed_land`)
        // writing `sources` to the host-visible `DecodeSources` port.
        // Declares ONLY ports consumed by `DecodeFront` (no `SharedDown`,
        // no `gate_scale`/`up_scale`, no `Tier { kind }`, no `Site*`).
        (Some(Split::DecodeFront(crossing)), Some(layer)) if decode => {
            return decode_front(
                graph, layer, part, shape, planned, scope, weights, constants, input, residual, rows, epsilon, crossing,
            );
        }
        (Some(Split::DecodeBack(crossing)), Some(_)) if decode => Some(crossing),
        _ => return Err("a routed layer's graph disagrees with how its experts are served".into()),
    };
    let ExpertWeights {
        stacks: [mut gate_records, mut up_records, mut down_records],
        gate_scale,
        up_scale,
    } = ExpertWeights::of(graph, planned, scope, weights, constants, part)?;
    let [streams, injected] = join.dimensions();
    let output_dims = [
        ("M", rows.clone()),
        ("H", shape.hidden.clone()),
        ("E", shape.count.clone()),
        ("K", shape.selected.clone()),
        ("F", shape.intermediate.clone()),
        ("S", shape.shared_intermediate.clone()),
        streams,
        injected,
        ("XS", Dim::from(0)),
        ("TR", Dim::from(0)),
        ("NO", match operand {
            Some(_) => shape.hidden.clone(),
            None => Dim::from(0),
        }),
    ];

    if decode {
        let (routes_tensor, scores_tensor, shared_down, expert_product, shared_product, coefficient, reads, experts, sequence, sited) =
            if let Some(crossing) = decode_back {
                let shared_down = weight_port(
                    graph,
                    planned,
                    WeightRole { scope, kind: WeightKind::SharedDown },
                    WeightPart::Values,
                    weights,
                )?
                .tensor()
                .clone();
                let RoutedDecodeEntries::SharedRoute { choices: gate_up, .. } = handle.decode else {
                    return Err("M7 decode back requires SharedRoute decode entries".into());
                };
                let layer = streamed.as_mut().expect("DecodeBack requires a streamed layer");
                let ports = layer.decode_back_ports(graph, false)?;
                let sources_view = crossing
                    .sources
                    .tensor()
                    .slice_leading(0, magnitude_kernels::constants::SOURCE_ROWS);
                let reads = layer.reads_back(graph, &ports, sources_view, constants)?;
                let experts = layer.numbers().records.clone();
                let dimensions = [
                    [
                        ("M", rows.clone()),
                        ("H", shape.hidden.clone()),
                        ("E", experts.clone()),
                        ("K", shape.selected.clone()),
                        ("F", shape.intermediate.clone()),
                        ("GS", gate_scale.extent.clone()),
                        ("US", up_scale.extent.clone()),
                        ("LB", Dim::from(magnitude_family_contracts::GluLimit::word(None))),
                    ]
                    .as_slice(),
                    &reads.dimensions(),
                ]
                .concat();
                let glu_limit = crate::operators::glu_limit(crossing.routes.tensor(), None);
                let product = graph
                    .enqueue(
                        G::pick(gate_up, rows),
                        &dimensions,
                        choices_args(
                            crossing.normalized.tensor(),
                            crossing.routes.tensor(),
                            &reads,
                            tier_gate(&reads)?,
                            [gate_records.tensor(), up_records.tensor()],
                            [&gate_scale, &up_scale],
                            &glu_limit,
                        ),
                    )?
                    .value;
                let coefficient = crossing
                    .coefficient
                    .as_ref()
                    .ok_or("a fused decode back crosses no coefficient")?
                    .tensor()
                    .clone();
                let shared_product = crossing
                    .shared_product
                    .as_ref()
                    .ok_or("a fused decode back crosses no shared product")?
                    .tensor()
                    .clone();
                let sited = ports.site.map(|site| {
                    (
                        site,
                        ports.policy,
                        ports.step.expect("sited DecodeBackPorts binds Step"),
                    )
                });
                (
                    crossing.routes.tensor().clone(),
                    crossing.weights.tensor().clone(),
                    shared_down,
                    product,
                    shared_product,
                    coefficient,
                    reads,
                    experts,
                    None,
                    sited,
                )
            } else {
                let route = RouteWeights::of(graph, planned, scope, weights, constants, input)?;
                let choices = route.choices(input, rows, shape);
                let [parts, norm_extent] = input.form.dimensions(shape.hidden.clone());
                let RouteWeights {
                    norm,
                    router,
                    shared_router,
                    expert_scale,
                } = &route;
                let [shared_gate, shared_up, shared_down] = shared_weights(graph, planned, scope, weights)?;
                let experts_dims = [
                    ("M", rows.clone()),
                    ("H", shape.hidden.clone()),
                    ("E", shape.count.clone()),
                    ("K", shape.selected.clone()),
                    ("F", shape.intermediate.clone()),
                    ("S", shape.shared_intermediate.clone()),
                    ("GS", gate_scale.extent.clone()),
                    ("US", up_scale.extent.clone()),
                ];
                let mut routes = graph.local_for(handle.route, "routes", &choices)?;
                let mut scores = graph.local_for(handle.route, "scores", &choices)?;
                let normalize = shape.normalize()?;
                // The expansion's and output's inputs: the experts' product, the
                // shared product, the coefficient, where they read each choice's
                // record (`DecodeReads`), `E` of what they read, and the sequence
                // of a layer that sets a bank-served launch's step.
                let resident = |graph: &mut G, constants: &mut Vec<GraphConstant>| {
                    let stacks = RecordTensors {
                        gate: Some(gate_records.tensor()),
                        up: up_records.tensor(),
                        down: down_records.tensor(),
                    };
                    DecodeReads::resident(graph, rows, &shape.selected, stacks, constants)
                };
                let (expert_product, shared_product, coefficient, reads, experts, sequence, sited) = match handle
                    .decode
                {
            RoutedDecodeEntries::Expand(expand) => {
                super::streamed::expand_form(streamed.is_some())?;
                let (expert_gate, expert_up) =
                    (gate_records.tensor().clone(), up_records.tensor().clone());
                let (normalized, coefficient) = route.route(
                    graph,
                    handle.route,
                    input,
                    &mut routes,
                    &mut scores,
                    rows,
                    shape,
                    epsilon,
                )?;
                let expanded = graph.enqueue(
                    G::pick(expand, rows),
                    &experts_dims,
                    routed_expand::WorkflowArgs {
                        normalized: (&normalized).into(),
                        routes: routes.tensor().into(),
                        expert_gate: (&expert_gate).into(),
                        expert_up: (&expert_up).into(),
                        shared_gate: (&shared_gate).into(),
                        shared_up: (&shared_up).into(),
                        gate_scale: (&gate_scale.tensor).into(),
                        up_scale: (&up_scale.tensor).into(),
                    },
                )?;
                let reads = resident(graph, constants)?;
                (expanded.r0, expanded.r1, coefficient, reads, shape.count.clone(), None, None)
            }
            RoutedDecodeEntries::SharedRoute {
                route,
                choices: gate_up,
            } => {
                // The choices' gate/up dimensions, reading `E` experts where
                // `reads` names each choice.
                let gate_up_dimensions = |reads: &DecodeReads, experts: &Dim| {
                    [
                        [
                            ("M", rows.clone()),
                            ("H", shape.hidden.clone()),
                            ("E", experts.clone()),
                            ("K", shape.selected.clone()),
                            ("F", shape.intermediate.clone()),
                            ("GS", gate_scale.extent.clone()),
                            ("US", up_scale.extent.clone()),
                            ("LB", Dim::from(magnitude_family_contracts::GluLimit::word(None))),
                        ]
                        .as_slice(),
                        &reads.dimensions(),
                    ]
                    .concat()
                };
                // The fused form's experts are unclamped (`is_fused`).
                let glu_limit = crate::operators::glu_limit(routes.tensor(), None);
                // The choices' product, the route's coefficient and shared
                // product, and where the gate/up and output read each
                // choice: a resident layer's stacks by expert; a
                // bank-served one's sources (`E` = `BANK`): its group's
                // records, which its fill and promotion commit, or its
                // tier's.
                let (expert_product, coefficient, shared, reads, experts, sequence, sited) =
                    match &mut streamed {
                        None => {
                            let routed = graph.enqueue(
                                G::pick(route, rows),
                                &[
                                    ("M", rows.clone()),
                                    ("H", shape.hidden.clone()),
                                    ("E", shape.count.clone()),
                                    ("K", shape.selected.clone()),
                                    ("S", shape.shared_intermediate.clone()),
                                    parts.clone(),
                                    norm_extent.clone(),
                                    ("ES", expert_scale.extent.clone()),
                                ],
                                routed_route_shared::WorkflowArgs {
                                    residual: input.rows.into(),
                                    norm: norm.into(),
                                    router: router.into(),
                                    shared_router: shared_router.into(),
                                    shared_gate: (&shared_gate).into(),
                                    shared_up: (&shared_up).into(),
                                    routes: routes.tensor_mut().into(),
                                    scores: scores.tensor_mut().into(),
                                    eps: epsilon.into(),
                                    normalize,
                                    expert_scale: (&expert_scale.tensor).into(),
                                },
                            )?;
                            let reads = resident(graph, constants)?;
                            let experts = shape.count.clone();
                            let product = graph
                                .enqueue(
                                    G::pick(gate_up, rows),
                                    &gate_up_dimensions(&reads, &experts),
                                    choices_args(
                                        &routed.r0,
                                        routes.tensor(),
                                        &reads,
                                        tier_gate(&reads)?,
                                        [gate_records.tensor(), up_records.tensor()],
                                        [&gate_scale, &up_scale],
                                        &glu_limit,
                                    ),
                                )?
                                .value;
                            (product, routed.r1, routed.r2, reads, experts, None, None)
                        }
                        Some(layer) => {
                            let bank = layer.numbers();
                            let mut ports = layer.decode_ports(graph, rows, constants)?;
                            let mut routing = layer.part(graph, BankPart::Routing)?;
                            let mut front = layer.part(graph, BankPart::Front)?;
                            let tier = layer.route_tier(graph, &ports, constants)?;
                            let mut keys = layer.route_keys(graph)?;
                            let (sf, sr) = match &ports.site {
                                Some(_) => (bank.site_slots.clone(), bank.site_width.clone()),
                                None => (Dim::from(0), Dim::from(0)),
                            };
                            let mut empty_site = ports
                                .site
                                .is_none()
                                .then(|| GraphConstant::empty_port(graph, Element::i32(), &[Dim::from(0), Dim::from(0)], constants))
                                .transpose()?;
                            let mut empty_site_rows = ports
                                .site
                                .is_none()
                                .then(|| {
                                    GraphConstant::empty_port(
                                        graph,
                                        Element::f32(),
                                        &[Dim::from(0), Dim::from(magnitude_kernels::constants::VERIFY_ROWS), shape.hidden.clone()],
                                        constants,
                                    )
                                })
                                .transpose()?;
                            let route_dims = [
                                ("M", rows.clone()),
                                ("H", shape.hidden.clone()),
                                ("E", shape.count.clone()),
                                ("K", shape.selected.clone()),
                                ("S", shape.shared_intermediate.clone()),
                                parts.clone(),
                                norm_extent.clone(),
                                ("ES", expert_scale.extent.clone()),
                                ("FW", bank.file_width.clone()),
                                ("FLIGHT", bank.flight.clone()),
                                ("RING", bank.ring_words.clone()),
                                ("P", bank.choices.clone()),
                                ("GM", bank.map_words.clone()),
                                ("TM", tier.experts.clone()),
                                ("SF", sf),
                                ("SR", sr),
                            ];
                            let BankRoute::Fused { activation, route, hints: hint_kernels } = layer.route() else {
                                return Err("a fused layer's bank holds another form's route".into());
                            };
                            let route = G::pick(route, rows);
                            // The step's sources, which the route writes and its
                            // later entries read.
                            let mut locals = RouteLocals { sources: graph.local_for(route, "sources", &route_dims)? };
                            let sited_mode = ports.sited();
                            let (site_ref, site_rows_ref) = match ports.site.as_mut() {
                                Some(site) => (site.table.tensor_mut().into(), site.rows.tensor_mut().into()),
                                None => (
                                    empty_site.as_mut().unwrap().tensor_mut().into(),
                                    empty_site_rows.as_mut().unwrap().tensor_mut().into(),
                                ),
                            };
                            let routed = graph.enqueue(
                                route,
                                &route_dims,
                                routed_route_streamed::WorkflowArgs {
                                    residual: input.rows.into(),
                                    norm: norm.into(),
                                    router: router.into(),
                                    shared_router: shared_router.into(),
                                    shared_gate: (&shared_gate).into(),
                                    shared_up: (&shared_up).into(),
                                    routes: routes.tensor_mut().into(),
                                    scores: scores.tensor_mut().into(),
                                    eps: epsilon.into(),
                                    normalize,
                                    expert_scale: (&expert_scale.tensor).into(),
                                    map: ports.map.tensor().into(),
                                    tier: (&tier.map).into(),
                                    policy: ports.policy.tensor().into(),
                                    sited: sited_mode.into(),
                                    step: ports.step.tensor().into(),
                                    file: ports.file.tensor_mut().into(),
                                    sources: locals.sources.tensor_mut().into(),
                                    routing: routing.tensor_mut().into(),
                                    keys: keys.tensor_mut().into(),
                                    front: front.tensor_mut().into(),
                                    land_grid: ports.land_grid.tensor_mut().into(),
                                    land_groups: layer.land_groups().into(),
                                    site: site_ref,
                                    site_rows: site_rows_ref,
                                },
                            )?;
                            if let Some(mut hints) = layer.hints(
                                graph,
                                weights,
                                [norm, router],
                                ports.policy.tensor(),
                            )? {
                                let hint = hint_kernels[hints.ahead as usize - 1]
                                    .ok_or("a streamed layer's lead was not prepared with its binding")?;
                                let empty = GraphConstant::empty_of(graph, activation, &[Dim::from(0u64), shape.selected.clone()], constants)?;
                                let hint_dims = [
                                    ("M", rows.clone()),
                                    ("H", shape.hidden.clone()),
                                    parts.clone(),
                                    norm_extent.clone(),
                                    ("K", shape.selected.clone()),
                                    ("AHEAD", Dim::from(hints.ahead)),
                                    ("RING", bank.ring_words.clone()),
                                    ("EN", bank.next_experts.clone()),
                                ];
                                // Encoded while the layer's hints pay
                                // (POLICY_HINTS).
                                graph.enqueue_under(
                                    hint,
                                    &hint_dims,
                                    routed_hint_route::WorkflowArgs {
                                        residual: input.rows.into(),
                                        activation: (&empty).into(),
                                        next_norm: (&hints.weights[0][0]).into(),
                                        next_router: (&hints.weights[0][1]).into(),
                                        after_norm: (&hints.weights[1][0]).into(),
                                        after_router: (&hints.weights[1][1]).into(),
                                        eps: epsilon.into(),
                                        step: ports.step.tensor().into(),
                                        next_policy: (&hints.policy).into(),
                                        rings: hints.rings.tensor_mut().into(),
                                    },
                                    &ports.policy,
                                    magnitude_kernels::constants::POLICY_HINTS,
                                    LaunchEncoding::WhileSet,
                                )?;
                            }
                            // The gate/up in two passes around the wait, the
                            // site request between them: the step's CPU
                            // choices of the snapshot, weighed by their scores
                            // (`StreamedLayer::gate_up`).
                            let reads = layer.reads(graph, &ports, &locals, constants)?;
                            let experts = bank.records.clone();
                            let dimensions = gate_up_dimensions(&reads, &experts);
                            let (expert_gate, expert_up) = (gate_records.tensor().clone(), up_records.tensor().clone());
                            let gate_tier = tier_gate(&reads)?;
                            let args = || {
                                choices_args(
                                    &routed.r0,
                                    routes.tensor(),
                                    &reads,
                                    gate_tier,
                                    [&expert_gate, &expert_up],
                                    [&gate_scale, &up_scale],
                                    &glu_limit,
                                )
                            };
                            let mut step = RoutedStep { ports, locals, routing };
                            let product = layer.gate_up(
                                graph,
                                &mut step,
                                ResidentGateUp { args, dimensions: &dimensions },
                                RecordTensors { gate: Some(&mut gate_records), up: &mut up_records, down: &mut down_records },
                                &StepChoices { routes: routes.tensor(), weights: scores.tensor(), input: &routed.r0, rows },
                                constants,
                            )?;
                            let mut ports = step.ports;
                            let sequence = ports.inputs.take();
                            let sited = ports.site.map(|site| (site.into_combine(), Some(ports.policy), ports.step));
                            (product, routed.r1, routed.r2, reads, experts, sequence, sited)
                        }
                    };
                (expert_product, shared, coefficient, reads, experts, sequence, sited)
            }
        };
                (
                    routes.tensor().clone(),
                    scores.tensor().clone(),
                    shared_down,
                    expert_product,
                    shared_product,
                    coefficient,
                    reads,
                    experts,
                    sequence,
                    sited,
                )
            };
        let inject = join.inject(graph, rows, constants)?;
        let mut dimensions = output_dims;
        dimensions[2] = ("E", experts);
        [dimensions[8], dimensions[9]] = reads.dimensions();
        // A layer with a site combines, on backends that park in kernels
        // (CUDA), in one launch (`routed_output_streamed`), or else while its
        // group is not whole (`policy[POLICY_WHOLE]` clear) in two launches
        // (`routed_output_project` then `routed_output_sited`) beside the
        // whole-group `routed_output`. Only the arms that enqueue
        // `routed_output` declare the empty `operand_norm` constant port.
        let combine = match (streamed.as_mut(), sited) {
            (Some(layer), Some((site, policy, step))) => {
                layer.site_combine().map(|combine| (layer, site, combine, policy, step))
            }
            _ => None,
        };
        let output = match combine {
            // A sited join forms no operand.
            Some(_) if operand.is_some() => {
                return Err("a routed layer with a site forms no operand for another graph".into());
            }
            Some((layer, mut site, (BankCombineEntry::FusedInline(inline), site_dimensions), _, step)) => {
                // Single-launch streamed fused combine: projections into
                // scratch + lock-free `arrive::last` block that awaits the CPU
                // site only when `streaming_site_any` is true (zero waiting
                // when 0 experts are CPU-sited, including when whole).
                // On CUDA (`parks_in_kernels`), `park_site` is a zero-cost
                // graph marker satisfying Seismic's `&await` contract.
                layer.park_site(graph);
                let inline_dims = [
                    &dimensions[0..8],
                    &[dimensions[9].clone()],
                    site_dimensions.as_slice(),
                ]
                .concat();
                graph
                    .enqueue(
                        G::pick(inline, rows),
                        &inline_dims,
                        magnitude_kernels::routed_output_streamed::WorkflowArgs {
                            residual: residual.into(),
                            inject: (&inject).into(),
                            expert_product: (&expert_product).into(),
                            shared_product: (&shared_product).into(),
                            routes: (&routes_tensor).into(),
                            sources: (&reads.sources).into(),
                            scores: (&scores_tensor).into(),
                            coefficient: (&coefficient).into(),
                            expert_down: down_records.tensor().into(),
                            tier_down: (&reads.tier.down).into(),
                            shared_down: (&shared_down).into(),
                            step: step.tensor().into(),
                            site: site.table.tensor_mut().into(),
                            site_output: site.output.tensor().into(),
                        },
                    )?
                    .value
            }
            Some((layer, mut site, (BankCombineEntry::Fused { project, sited }, site_dimensions), policy, step)) => {
                let operand_norm = GraphConstant::empty(graph, handle.operand_norm, constants)?;
                let resident_args = routed_output::WorkflowArgs {
                    residual: residual.into(),
                    inject: (&inject).into(),
                    expert_product: (&expert_product).into(),
                    shared_product: (&shared_product).into(),
                    routes: (&routes_tensor).into(),
                    sources: (&reads.sources).into(),
                    scores: (&scores_tensor).into(),
                    coefficient: (&coefficient).into(),
                    expert_down: down_records.tensor().into(),
                    tier_down: (&reads.tier.down).into(),
                    shared_down: (&shared_down).into(),
                    operand_norm: (&operand_norm).into(),
                    operand_epsilon: 0.0.into(),
                };
                let policy = policy.expect("split sited combine binds Policy");
                let whole = magnitude_kernels::constants::POLICY_WHOLE;
                let resident = graph
                    .enqueue_under(G::pick(handle.output, rows), &dimensions, resident_args, &policy, whole, LaunchEncoding::WhileSet)?
                    .value;
                let published = graph
                    .enqueue_under(
                        G::pick(project, rows),
                        &dimensions,
                        routed_output_project::WorkflowArgs {
                            expert_product: (&expert_product).into(),
                            shared_product: (&shared_product).into(),
                            routes: (&routes_tensor).into(),
                            sources: (&reads.sources).into(),
                            expert_down: down_records.tensor().into(),
                            tier_down: (&reads.tier.down).into(),
                            shared_down: (&shared_down).into(),
                        },
                        &policy,
                        whole,
                        LaunchEncoding::WhileClear,
                    )?
                    .value;
                layer.park_site(graph);
                let joined = graph
                    .enqueue_under(
                        sited,
                        &[dimensions.as_slice(), &site_dimensions].concat(),
                        routed_output_sited::WorkflowArgs {
                            residual: residual.into(),
                            inject: (&inject).into(),
                            published: (&published).into(),
                            sources: (&reads.sources).into(),
                            scores: (&scores_tensor).into(),
                            coefficient: (&coefficient).into(),
                            step: step.tensor().into(),
                            site: site.table.tensor_mut().into(),
                            site_output: site.output.tensor().into(),
                        },
                        &policy,
                        whole,
                        LaunchEncoding::WhileClear,
                    )?
                    .value;
                graph.share(&joined, &resident)?;
                resident
            }
            Some((_, _, (BankCombineEntry::General { .. }, _), _, _)) => {
                return Err("a fused layer's bank holds another form's combine".into());
            }
            None => {
                let (operand_norm, operand_epsilon) = match &operand {
                    Some(operand) => (operand.norm.clone(), operand.epsilon),
                    None => (GraphConstant::empty(graph, handle.operand_norm, constants)?, 0.0),
                };
                graph
                    .enqueue(
                        G::pick(handle.output, rows),
                        &dimensions,
                        routed_output::WorkflowArgs {
                            residual: residual.into(),
                            inject: (&inject).into(),
                            expert_product: (&expert_product).into(),
                            shared_product: (&shared_product).into(),
                            routes: (&routes_tensor).into(),
                            sources: (&reads.sources).into(),
                            scores: (&scores_tensor).into(),
                            coefficient: (&coefficient).into(),
                            expert_down: down_records.tensor().into(),
                            tier_down: (&reads.tier.down).into(),
                            shared_down: (&shared_down).into(),
                            operand_norm: (&operand_norm).into(),
                            operand_epsilon: operand_epsilon.into(),
                        },
                    )?
                    .value
            }
        };
        if let Some(operand) = &operand {
            graph.carry_out(&output, operand.carry, operand.forms)?;
        }
        return Ok((output, sequence));
    }

    let route = RouteWeights::of(graph, planned, scope, weights, constants, input)?;
    let choices = route.choices(input, rows, shape);
    let [shared_gate, shared_up, shared_down] = shared_weights(graph, planned, scope, weights)?;
    let mut routes = graph.local_for(handle.route, "routes", &choices)?;
    let mut scores = graph.local_for(handle.route, "scores", &choices)?;
    let (normalized, coefficient) =
        route.route(graph, handle.route, input, &mut routes, &mut scores, rows, shape, epsilon)?;
    let experts = ExpertWeights {
        stacks: [gate_records, up_records, down_records],
        gate_scale,
        up_scale,
    };
    let shared = [shared_gate, shared_up, shared_down];
    let pass = super::grouped::class(
        graph,
        handle.group,
        [routes.tensor(), &normalized],
        experts.stacks(),
        rows,
        [&shape.count, &shape.selected],
        constants,
        |graph, grouped| expand(graph, handle.experts, &experts, grouped, shape),
    )?;
    let combined = Combined { scores: scores.tensor(), normalized: &normalized, coefficient: &coefficient };
    let output = combine(graph, &handle, &shared, &pass, combined, residual, join, rows, shape, constants)?;
    Ok((output, None))
}

/// A merge set's pass over a fused layer of `scope` (`OnceStage::Pass`): the
/// experts of the crossed rows the merged count names from the merged first
/// row, of the pass's `rows`, read where they live (`grouped::pass`), into
/// the merged ports.
#[allow(clippy::too_many_arguments)]
pub(crate) fn group_pass<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    handle: RoutedGraphEntries<'a, G>,
    bank: LayerBank<'_, 'a, G>,
    topology: &RoutedTopology,
    shape: &ExpertShape,
    planned: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    ports: &mut PassPorts,
    blocks: &Dim,
    rows: &Dim,
) -> Result<WorkflowTensor, GraphError> {
    let part = bank.stacks();
    let mut layer = StreamedLayer::of(planned, &topology.bank, shape.bank.as_ref(), scope, bank)?;
    let experts = ExpertWeights::of(graph, planned, scope, weights, constants, part)?;
    super::grouped::pass(
        graph,
        handle.group,
        layer.as_mut(),
        ports,
        blocks,
        experts.stacks(),
        |tier| match tier {
            RecordTensors { gate: Some(gate), up, down } => Ok([gate, up, down].map(|port| port.tensor().clone())),
            _ => Err("a fused layer's records are gated".into()),
        },
        rows,
        [&shape.count, &shape.selected],
        constants,
        |graph, grouped| expand(graph, handle.experts, &experts, grouped, shape),
    )
}

/// What the combine reads of a route beside the experts: the choices'
/// scores, the normalized rows and the shared expert's coefficient.
struct Combined<'t> {
    scores: &'t WorkflowTensor,
    normalized: &'t WorkflowTensor,
    coefficient: &'t WorkflowTensor,
}

impl<'t> Combined<'t> {
    fn of(crossing: &'t BackCrossing<NativePort>) -> Result<Self, GraphError> {
        let SharedInput { normalized, coefficient } =
            crossing.shared.as_ref().ok_or("a fused layer's back crosses no shared expert input")?;
        Ok(Self {
            scores: crossing.weights.tensor(),
            normalized: normalized.tensor(),
            coefficient: coefficient.tensor(),
        })
    }
}

/// A fused layer's shared expert's gate, up and down.
fn shared_weights<G: GraphDraft>(
    graph: &mut G,
    planned: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
) -> Result<[WorkflowTensor; 3], GraphError> {
    let mut shared = |kind| weight(graph, planned, scope, kind, weights);
    Ok([shared(WeightKind::SharedGate)?, shared(WeightKind::SharedUp)?, shared(WeightKind::SharedDown)?])
}

/// A fused layer's experts (`routed_experts`, gated) over a grouping
/// (`grouped`), with `experts`' scales.
fn expand<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entry: G::Ranged<'a, routed_experts::Entry>,
    experts: &ExpertWeights,
    grouped: Grouped<'_, [WorkflowTensor; 3]>,
    shape: &ExpertShape,
) -> Result<WorkflowTensor, GraphError> {
    let ExpertWeights { gate_scale, up_scale, .. } = experts;
    let Grouped { rows, blocks, first, rows_read, input, order, block_experts, used, reads } = grouped;
    let [expert_gate, expert_up, expert_down] = &reads.stacks;
    let [records_gate, records_up, records_down] = &reads.records;
    let [mapped, step_words] = reads.map_dimensions();
    let dimensions = [
        ("M", rows.clone()),
        ("H", shape.hidden.clone()),
        ("E", reads.stacks_rows.clone()),
        ("F", shape.intermediate.clone()),
        ("B", blocks),
        ("T", Dim::from(TILE_ROWS)),
        ("GS", gate_scale.extent.clone()),
        ("US", up_scale.extent.clone()),
        ("LB", Dim::from(magnitude_family_contracts::GluLimit::word(None))),
        mapped,
        step_words,
        ("R", reads.count.clone()),
    ];
    Ok(graph
        .enqueue(
            G::pick(entry, rows),
            &dimensions,
            routed_experts::WorkflowArgs {
                first: first.into(),
                rows: rows_read.into(),
                normalized: input.into(),
                order: order.into(),
                blocks: block_experts.into(),
                used: used.into(),
                expert_gate: expert_gate.into(),
                expert_up: expert_up.into(),
                expert_down: expert_down.into(),
                map: (&reads.map).into(),
                step: (&reads.step).into(),
                records_gate: records_gate.into(),
                records_up: records_up.into(),
                records_down: records_down.into(),
                activation: (super::FUSED_ACTIVATION).into(),
                gate_scale: (&gate_scale.tensor).into(),
                up_scale: (&up_scale.tensor).into(),
                // The fused form's experts are unclamped (`is_fused`).
                glu_limit: (&crate::operators::glu_limit(first, None)).into(),
            },
        )?
        .value)
}

/// The combine of `rows` rows onto `residual` with the shared expert's
/// `shared` weights, reading their choices' experts from `pass` at the
/// positions it names.
#[allow(clippy::too_many_arguments)]
fn combine<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    handle: &RoutedGraphEntries<'a, G>,
    shared: &[WorkflowTensor; 3],
    pass: &ExpertsPass,
    combined: Combined<'_>,
    residual: &WorkflowTensor,
    join: &Join,
    rows: &Dim,
    shape: &ExpertShape,
    constants: &mut Vec<GraphConstant>,
) -> Result<WorkflowTensor, GraphError> {
    let [shared_gate, shared_up, shared_down] = shared;
    let [streams, injected] = join.dimensions();
    let dimensions = [
        ("M", rows.clone()),
        ("H", shape.hidden.clone()),
        ("K", shape.selected.clone()),
        ("B", pass.blocks.clone()),
        ("T", Dim::from(TILE_ROWS)),
        ("S", shape.shared_intermediate.clone()),
        streams,
        injected,
    ];
    let inject = join.inject(graph, rows, constants)?;
    Ok(graph
        .enqueue(
            G::pick(handle.combine, rows),
            &dimensions,
            routed_combine::WorkflowArgs {
                residual: residual.into(),
                inject: (&inject).into(),
                expert_output: (&pass.output).into(),
                inverse: (&pass.inverse).into(),
                scores: combined.scores.into(),
                normalized: combined.normalized.into(),
                coefficient: combined.coefficient.into(),
                shared_gate: shared_gate.into(),
                shared_up: shared_up.into(),
                shared_down: shared_down.into(),
            },
        )?
        .value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, Layout};

    /// A down projection the Metal kernel's defaults cannot hold within its
    /// register budget (`routed_down`'s `where` in `routed.seismic`: at K 2,
    /// F 16,384, ROWS 2 LANES 16 hold 16 weight packets a lane, past 8) is
    /// outside the kernel's domain, and graph construction names the call.
    #[test]
    fn a_call_outside_its_kernel_domain_fails_graph_construction() {
        let (m, h, e, k, f, s) = (1, 256, 8, 2, 16_384, 256);
        let weight = Element::stored("q4k", Layout::Rows16).unwrap();
        let activation = Element::bf16();
        let elements = [("EDW", weight), ("SDW", weight), ("A", activation), ("NW", Element::f32())];
        let mut graph = NativeGraphMetadata::new(BackendName::Metal);
        let mut port =
            |element, extents: &[u64]| graph.port(element, extents).unwrap().tensor().clone();
        let residual = port(Element::f32(), &[m, h]);
        let inject = port(Element::f32(), &[m, 0]);
        let expert_product = port(activation, &[m, k, f]);
        let shared_product = port(activation, &[m, s]);
        let routes = port(Element::i32(), &[m, k]);
        let sources = port(Element::i32(), &[0, m, k]);
        let scores = port(Element::f32(), &[m, k]);
        let coefficient = port(Element::f32(), &[m]);
        let expert_down = port(weight, &[e, h, f]);
        let shared_down = port(weight, &[h, s]);
        let tier_down = port(weight, &[0, h, f]);
        let operand_norm = port(Element::f32(), &[0]);
        let Err(error) = GraphDraft::enqueue::<routed_output::Entry>(
            &mut graph,
            &elements,
            &[
                ("M", m),
                ("H", h),
                ("E", e),
                ("K", k),
                ("F", f),
                ("S", s),
                ("SN", 1),
                ("IN", 0),
                ("XS", 0),
                ("TR", 0),
                ("NO", 0),
            ],
            routed_output::WorkflowArgs {
                residual: (&residual).into(),
                inject: (&inject).into(),
                expert_product: (&expert_product).into(),
                shared_product: (&shared_product).into(),
                routes: (&routes).into(),
                sources: (&sources).into(),
                scores: (&scores).into(),
                coefficient: (&coefficient).into(),
                expert_down: (&expert_down).into(),
                tier_down: (&tier_down).into(),
                shared_down: (&shared_down).into(),
                operand_norm: (&operand_norm).into(),
                operand_epsilon: 0.0f32.into(),
            },
        ) else {
            panic!("an out-of-domain call enqueued");
        };
        let GraphError::KernelDomain(violation) = error else {
            panic!("expected a kernel domain violation, got {error}");
        };
        assert_eq!(violation.entry, "routed_output");
        assert_eq!(violation.backend, BackendName::Metal);
        assert_eq!(
            violation.statics,
            [("F", f), ("H", h), ("IN", 0), ("K", k), ("S", s), ("SN", 1)]
                .map(|(name, value)| (name.to_owned(), value))
                .to_vec()
        );
    }
}
