//! Routed (mixture-of-experts) feed-forward graph construction (program spec
//! K6, E8).
//!
//! Row classes up to [`DECODE_ROWS`] run the decode form ([`DecodeForm`]):
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
    BankCombineEntry, BankForm, BankNumbers, BankRoute, LayerBank, DecodeReads, DecodeShape, RouteLocals,
    StreamedLayer, WaitPorts,
};
use crate::planning::RecordTensors;
use crate::programs::graph::facts::Numbers;
use crate::programs::native_target_graph::{
    PlannedWeights, WeightPart, WeightPort, WeightScale, weight, weight_port,
};
use magnitude_kernels::{routed_output_project, routed_output_sited, routed_route_streamed};
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
use seismic::{Argument, BackendName, Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

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
            match seismic::generated::native_implementation_for_backend::<
                routed_route_shared::Entry,
            >(backend)
            .map_err(|error| error.to_string())?
            {
                Some(_) => Self::SharedRoute,
                None => Self::Expand,
            },
        )
    }
}

pub(crate) struct RoutedGraphEntries<'a, G: GraphDraft + 'a> {
    pub route: G::Binding<'a, routed_route::Entry>,
    pub decode: RoutedDecodeEntries<'a, G>,
    pub output: G::Binding<'a, routed_output::Entry>,
    pub group: G::Binding<'a, routed_group::Entry>,
    pub experts: G::Binding<'a, routed_experts::Entry>,
    pub combine: G::Binding<'a, routed_combine::Entry>,
}

/// The decode rows' routing and expansion entries of a [`DecodeForm`].
pub(crate) enum RoutedDecodeEntries<'a, G: GraphDraft + 'a> {
    Expand(G::Binding<'a, routed_expand::Entry>),
    SharedRoute {
        route: G::Binding<'a, routed_route_shared::Entry>,
        choices: G::Binding<'a, routed_gate_up::Entry>,
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
            group: &kernels.group,
            experts: &kernels.experts,
            combine: &kernels.combine,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedRoutedEntries {
    route: [(&'static str, Element); 3],
    decode: CheckedDecodeEntries,
    output: [(&'static str, Element); 3],
    group: [(&'static str, Element); 0],
    experts: [(&'static str, Element); 4],
    combine: [(&'static str, Element); 4],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CheckedDecodeEntries {
    Expand([(&'static str, Element); 5]),
    SharedRoute {
        route: [(&'static str, Element); 5],
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

    pub(crate) fn entries(&self) -> RoutedGraphEntries<'_, NativeGraphMetadata> {
        RoutedGraphEntries {
            route: &self.route,
            decode: match &self.decode {
                CheckedDecodeEntries::Expand(expand) => RoutedDecodeEntries::Expand(&expand[..]),
                CheckedDecodeEntries::SharedRoute { route, choices } => {
                    RoutedDecodeEntries::SharedRoute {
                        route: &route[..],
                        choices: &choices[..],
                    }
                }
            },
            output: &self.output,
            group: &self.group,
            experts: &self.experts,
            combine: &self.combine,
        }
    }
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

/// The largest row class the decode form serves: the K1 GEMV row bound,
/// `routed.seismic`'s `DECODE_ROWS`.
pub(crate) const DECODE_ROWS: u64 = magnitude_kernels::constants::DECODE_ROWS;

/// Whether a `rows`-row class runs the decode form.
pub(crate) fn decodes(rows: impl Into<Dim>) -> bool {
    rows.into().at_most(DECODE_ROWS)
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
) -> Result<(WorkflowTensor, Option<NativePort>), GraphError> {
    let part = bank.stacks();
    let mut streamed = StreamedLayer::of(planned, &topology.bank, shape.bank.as_ref(), scope, bank)?;
    let decode = decodes(rows);
    let scratch = streamed.is_some() && !decode;
    match (split, &mut streamed) {
        (None, _) if !scratch => {}
        // A front routes into its crossing; a scratch-served one requests
        // its experts (none while its layer is whole).
        (Some(Split::Front(crossing)), layer) if !decode => {
            let route = RouteWeights::of(graph, planned, scope, weights, constants, input)?;
            let Crossing { routes, weights: scores, normalized: into, coefficient: Some(into_coefficient) } = crossing
            else {
                return Err("a fused layer's front crosses no coefficient".into());
            };
            let (normalized, coefficient) =
                route.route(graph, handle.route, input, routes, scores, rows, shape, epsilon)?;
            graph.store(&normalized, into)?;
            graph.store(&coefficient, into_coefficient)?;
            let sequence = match layer {
                Some(layer) => layer.request(graph, routes.tensor(), rows)?,
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
        _ => return Err("a routed layer's graph disagrees with how its experts are served".into()),
    }
    let route = RouteWeights::of(graph, planned, scope, weights, constants, input)?;
    let choices = route.choices(input, rows, shape);
    let [parts, norm_extent] = input.form.dimensions(shape.hidden.clone());
    let RouteWeights {
        norm,
        router,
        shared_router,
        expert_scale,
    } = &route;
    let ExpertWeights {
        stacks: [mut gate_records, mut up_records, mut down_records],
        gate_scale,
        up_scale,
    } = ExpertWeights::of(graph, planned, scope, weights, constants, part)?;
    let [shared_gate, shared_up, shared_down] = shared_weights(graph, planned, scope, weights)?;
    let [streams, injected] = join.dimensions();
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
    ];
    let mut routes = graph.local_for(handle.route, "routes", &choices)?;
    let mut scores = graph.local_for(handle.route, "scores", &choices)?;
    let normalize = shape.normalize()?;

    if decode {
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
                    expand,
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
                // The route's normalized rows, coefficient and shared
                // product, and where the gate/up and output read each
                // choice: a resident layer's stacks by expert; a
                // bank-served one's sources (`E` = `BANK`): its group's
                // records, which its fill and promotion commit, or its
                // tier's.
                let (normalized, coefficient, shared, reads, experts, sequence, sited) =
                    match &mut streamed {
                        None => {
                            let routed = graph.enqueue(
                                route,
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
                            (routed.r0, routed.r1, routed.r2, reads, shape.count.clone(), None, None)
                        }
                        Some(layer) => {
                            let bank = layer.numbers();
                            let mut ports = layer.decode_ports(graph)?;
                            let mut hints = layer.hints(
                                graph,
                                weights,
                                [norm, router],
                                ports.policy.tensor(),
                            )?;
                            let mut routing = layer.part(graph, BankPart::Routing)?;
                            let tier = layer.route_tier(graph, &ports, constants)?;
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
                                ("AHEAD", Dim::from(hints.ahead)),
                                ("FLIGHT", bank.flight.clone()),
                                ("RING", bank.ring_words.clone()),
                                ("P", bank.choices.clone()),
                                ("EN", bank.next_experts.clone()),
                                ("GM", bank.map_words.clone()),
                                ("TM", tier.experts.clone()),
                            ];
                            let BankRoute::Fused(route) = layer.route() else {
                                return Err("a fused layer's bank holds another form's route".into());
                            };
                            let route = route[hints.ahead as usize].ok_or("a streamed layer's lead was not prepared with its binding")?;
                            // The step's sources and the tier wait's grid, which
                            // the route writes and its later entries read.
                            let mut locals = RouteLocals {
                                sources: graph.local_for(route, "sources", &route_dims)?,
                                grids: graph.local_for(route, "grids", &route_dims)?,
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
                                    sited: (ports.sited()).into(),
                                    step: ports.step.tensor().into(),
                                    file: ports.file.tensor_mut().into(),
                                    grids: locals.grids.tensor_mut().into(),
                                    sources: locals.sources.tensor_mut().into(),
                                    next_norm: (&hints.weights[0][0]).into(),
                                    next_router: (&hints.weights[0][1]).into(),
                                    after_norm: (&hints.weights[1][0]).into(),
                                    after_router: (&hints.weights[1][1]).into(),
                                    next_policy: (&hints.policy).into(),
                                    rings: hints.rings.tensor_mut().into(),
                                    routing: routing.tensor_mut().into(),
                                },
                            )?;
                            let decode_shape = DecodeShape { rows };
                            let wait_grid = locals.grid(magnitude_kernels::constants::WAIT_GRID);
                            let mut sources = locals.sources.tensor().clone();
                            layer.tier_wait(
                                graph,
                                WaitPorts {
                                    decode: &mut ports,
                                    sources: &mut sources,
                                    grid: &wait_grid,
                                    records: RecordTensors { gate: Some(&mut gate_records), up: &mut up_records, down: &mut down_records },
                                },
                                &decode_shape,
                            )?;
                            // The site request, before the gate/up: the step's
                            // CPU choices of the snapshot, weighed by their scores.
                            layer.site_request(
                                graph,
                                &mut ports,
                                routes.tensor(),
                                scores.tensor(),
                                &locals.snapshot(),
                                &routed.r0,
                                &decode_shape,
                            )?;
                            let reads = layer.reads(graph, &ports, &locals, constants)?;
                            let sequence = ports.sequence.take();
                            let sited = ports.site.is_some().then_some(ports);
                            (routed.r0, routed.r1, routed.r2, reads, bank.records.clone(), sequence, sited)
                        }
                    };
                let expert_product = graph
                    .enqueue(
                        gate_up,
                        &[
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
                        .concat(),
                        routed_gate_up::WorkflowArgs {
                            normalized: (&normalized).into(),
                            routes: routes.tensor().into(),
                            sources: (&reads.sources).into(),
                            expert_gate: gate_records.tensor().into(),
                            expert_up: up_records.tensor().into(),
                            tier_gate: (reads.tier.gate.as_ref().ok_or("a fused layer's reads lack its gate")?).into(),
                            tier_up: (&reads.tier.up).into(),
                            activation: (super::FUSED_ACTIVATION).into(),
                            gate_scale: (&gate_scale.tensor).into(),
                            up_scale: (&up_scale.tensor).into(),
                            // The fused form's experts are unclamped (`is_fused`).
                            glu_limit: (&crate::operators::glu_limit(routes.tensor(), None)).into(),
                        },
                    )?
                    .value;
                (expert_product, shared, coefficient, reads, experts, sequence, sited)
            }
        };
        let inject = join.inject(graph, rows, constants)?;
        let mut dimensions = output_dims;
        dimensions[2] = ("E", experts);
        [dimensions[8], dimensions[9]] = reads.dimensions();
        let resident_args = || routed_output::WorkflowArgs {
            residual: residual.into(),
            inject: (&inject).into(),
            expert_product: (&expert_product).into(),
            shared_product: (&shared_product).into(),
            routes: routes.tensor().into(),
            sources: (&reads.sources).into(),
            scores: scores.tensor().into(),
            coefficient: (&coefficient).into(),
            expert_down: down_records.tensor().into(),
            tier_down: (&reads.tier.down).into(),
            shared_down: (&shared_down).into(),
        };
        // A layer with a site combines, while its group is not whole
        // (`policy[POLICY_WHOLE]` clear), with the CPU's answer: its
        // projections (`routed_output_project`), beside the CPU, then their
        // join (`routed_output_sited`), which waits for the answer when a
        // choice of the step is the CPU's; while its group is whole, as a
        // resident layer does (`routed_output`), at no cost and with its
        // numerics. The two are exclusive by their conditions, so one
        // storage holds the output (`GraphDraft::share`). Any other layer
        // combines resident, unconditioned.
        let combine = match (streamed.as_mut(), sited) {
            (Some(layer), Some(mut ports)) => {
                ports.site.take().zip(layer.site_combine()).map(|(site, combine)| (layer, site, combine, ports))
            }
            _ => None,
        };
        let output = match combine {
            Some((layer, mut site, (BankCombineEntry::Fused { project, sited }, site_dimensions), ports)) => {
                let (policy, whole) = (ports.policy, magnitude_kernels::constants::POLICY_WHOLE);
                let resident = graph
                    .enqueue_under(handle.output, &dimensions, resident_args(), &policy, whole, LaunchEncoding::WhileSet)?
                    .value;
                let published = graph
                    .enqueue_under(
                        project,
                        &dimensions,
                        routed_output_project::WorkflowArgs {
                            expert_product: (&expert_product).into(),
                            shared_product: (&shared_product).into(),
                            routes: routes.tensor().into(),
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
                            scores: scores.tensor().into(),
                            coefficient: (&coefficient).into(),
                            step: ports.step.tensor().into(),
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
            Some((_, _, (BankCombineEntry::General { .. }, _), _)) => {
                return Err("a fused layer's bank holds another form's combine".into());
            }
            None => graph.enqueue(handle.output, &dimensions, resident_args())?.value,
        };
        return Ok((output, sequence));
    }

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
    entry: G::Binding<'a, routed_experts::Entry>,
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
        ("E", shape.count.clone()),
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
            entry,
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
            handle.combine,
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
        let elements = [("EDW", weight), ("SDW", weight), ("A", activation)];
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
