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

use crate::operators::{Join, SublayerInput};
use crate::programs::graph::{GraphError, draft::GraphDraft};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{BankPart, WeightPart, WeightPort, weight, weight_port};
use crate::{
    ModelLoadPlan, RoutedBinding,
    native::{RoutedDecodeKernels, RoutedKernels, StreamedRoutedKernels},
};
use magnitude_family_contracts::{
    RouteNormalization, RoutedFfn, SublayerIndex, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    routed_bank_fill, routed_combine, routed_convert_wide, routed_copy, routed_expand,
    routed_experts, routed_gate_up, routed_group, routed_output, routed_route, routed_route_shared,
    routed_route_streamed, routed_scratch_request, routed_scratch_wait, routed_seed_commit,
    routed_step,
};
use seismic::{BackendName, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

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
    /// A streamed layer's bank-served decode entries.
    pub streamed: Option<StreamedGraphEntries<'a, G>>,
}

/// A streamed layer's decode entries (`StreamedRoutedKernels`).
pub(crate) struct StreamedGraphEntries<'a, G: GraphDraft + 'a> {
    pub step: G::Binding<'a, routed_step::Entry>,
    pub route: G::Binding<'a, routed_route_streamed::Entry>,
    pub fill: G::Binding<'a, routed_bank_fill::Entry>,
    pub scratch_request: G::Binding<'a, routed_scratch_request::Entry>,
    pub scratch_wait: G::Binding<'a, routed_scratch_wait::Entry>,
    pub copy: [G::Binding<'a, routed_copy::Entry>; 3],
    pub seed_commit: G::Binding<'a, routed_seed_commit::Entry>,
    pub convert_wide: [G::Binding<'a, routed_convert_wide::Entry>; 3],
}

impl<'a> From<&'a StreamedRoutedKernels> for StreamedGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a StreamedRoutedKernels) -> Self {
        let [copy_gate, copy_up, copy_down] = &kernels.copy;
        let [wide_gate, wide_up, wide_down] = &kernels.convert_wide;
        Self {
            step: &kernels.step,
            route: &kernels.route,
            fill: &kernels.fill,
            scratch_request: &kernels.scratch_request,
            scratch_wait: &kernels.scratch_wait,
            copy: [copy_gate, copy_up, copy_down],
            seed_commit: &kernels.seed_commit,
            convert_wide: [wide_gate, wide_up, wide_down],
        }
    }
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
            streamed: kernels.streamed.as_ref().map(StreamedGraphEntries::from),
        }
    }
}

pub(crate) struct CheckedRoutedEntries {
    route: [(&'static str, Element); 3],
    decode: CheckedDecodeEntries,
    output: [(&'static str, Element); 3],
    group: [(&'static str, Element); 0],
    experts: [(&'static str, Element); 4],
    combine: [(&'static str, Element); 4],
    streamed: Option<CheckedStreamedEntries>,
}

struct CheckedStreamedEntries {
    route: [(&'static str, Element); 5],
    convert: [[(&'static str, Element); 2]; 3],
    copy: [[(&'static str, Element); 1]; 3],
    fill: [(&'static str, Element); 6],
}

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
            streamed: binding.bank.map(|bank| {
                let residents = [binding.expert_gate, binding.expert_up, binding.expert_down];
                CheckedStreamedEntries {
                    route: [
                        ("NW", binding.norm),
                        ("RW", binding.router),
                        ("SGW", binding.shared_gate),
                        ("SUW", binding.shared_up),
                        ("A", binding.activation),
                    ],
                    convert: [0, 1, 2].map(|kind| [("E", bank.sources[kind]), ("U", residents[kind])]),
                    copy: residents.map(|resident| [("U", resident)]),
                    fill: [
                        ("SG", bank.sources[0]),
                        ("SU", bank.sources[1]),
                        ("SD", bank.sources[2]),
                        ("EGW", binding.expert_gate),
                        ("EUW", binding.expert_up),
                        ("EDW", binding.expert_down),
                    ],
                }
            }),
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
            streamed: self.streamed.as_ref().map(|streamed| {
                let [gate, up, down] = &streamed.convert;
                let [copy_gate, copy_up, copy_down] = &streamed.copy;
                StreamedGraphEntries {
                    step: &[][..],
                    route: &streamed.route[..],
                    fill: &streamed.fill[..],
                    scratch_request: &[][..],
                    scratch_wait: &[][..],
                    copy: [&copy_gate[..], &copy_up[..], &copy_down[..]],
                    seed_commit: &[][..],
                    convert_wide: [&gate[..], &up[..], &down[..]],
                }
            }),
        }
    }
}

/// The expert dimensions the routed entries are specialized to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpertShape {
    pub count: u64,
    pub selected: u64,
    pub intermediate: u64,
    pub shared_intermediate: u64,
    pub normalize_selected: bool,
}

impl ExpertShape {
    pub(crate) fn of_binding(binding: &RoutedBinding) -> Self {
        Self {
            count: binding.experts,
            selected: binding.selected,
            intermediate: binding.features,
            shared_intermediate: binding.shared,
            normalize_selected: binding.normalize_selected,
        }
    }

    /// The shape of an admitted routed operator (`operators::admit`
    /// guarantees its shared expert).
    pub(crate) fn of_operator(routed: &RoutedFfn) -> Result<Self, String> {
        Ok(Self {
            count: routed.experts,
            selected: routed.selected,
            intermediate: routed.intermediate,
            shared_intermediate: routed
                .shared
                .as_ref()
                .ok_or("routed feed-forward without shared expert")?
                .intermediate,
            normalize_selected: routed.router.normalization == RouteNormalization::Sum,
        })
    }
}

/// The largest row class the decode form serves: the K1 GEMV row bound.
pub(crate) const DECODE_ROWS: u64 = 8;

/// Whether a `rows`-row class runs the decode form.
pub(crate) fn decodes(rows: u64) -> bool {
    rows <= DECODE_ROWS
}

/// Rows of one expert tile of the grouped form.
pub(crate) const TILE_ROWS: u64 = 32;

/// Expert tiles that hold every choice of a `rows`-row class: `rows *
/// selected` choices, the rows of each chosen expert padded to whole tiles.
/// At most `min(experts, rows * selected)` experts are chosen, so
/// `ceil((rows * selected + min(experts, rows * selected) * (TILE_ROWS - 1))
/// / TILE_ROWS)`.
pub(crate) fn grouped_blocks(rows: u64, experts: u64, selected: u64) -> Result<u64, String> {
    let choices = rows
        .checked_mul(selected)
        .ok_or_else(|| format!("grouped tiles of {rows} rows overflow"))?;
    experts
        .min(choices)
        .checked_mul(TILE_ROWS - 1)
        .and_then(|padding| choices.checked_add(padding))
        .map(|slots| slots.div_ceil(TILE_ROWS))
        .ok_or_else(|| format!("grouped tiles of {rows} rows overflow"))
}

/// The routed feed-forward of one block over `rows` rows of `residual`, and
/// the step-sequence input of a streamed layer that sets the step
/// (`routed_step`), which each run binds to its launch's sequence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn routed<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    handle: RoutedGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    input: SublayerInput<'_>,
    residual: &WorkflowTensor,
    join: &Join,
    rows: u64,
    hidden: u64,
    shape: &ExpertShape,
    epsilon: f32,
) -> Result<(WorkflowTensor, Option<NativePort>), GraphError> {
    // A mix site's partials have no input norm: the port is empty.
    let norm = if input.form.normalized {
        weight(graph, load, scope, WeightKind::InputNorm, weights)?
    } else {
        GraphConstant::absent(graph, constants)?.tensor().clone()
    };
    let router = weight(graph, load, scope, WeightKind::Router, weights)?;
    let shared_router = weight(graph, load, scope, WeightKind::SharedRouter, weights)?;
    // A streamed layer's converts write its expert weights (its banks), so
    // the expert ports stay at hand.
    let mut expert_port = |kind| {
        weight_port(graph, load, WeightRole { scope, kind }, WeightPart::Values, 0, weights)
    };
    let expert_ports = [
        expert_port(WeightKind::ExpertGate)?,
        expert_port(WeightKind::ExpertUp)?,
        expert_port(WeightKind::ExpertDown)?,
    ];
    let [expert_gate, expert_up, expert_down] =
        expert_ports.each_ref().map(|port| port.tensor().clone());
    let shared_gate = weight(graph, load, scope, WeightKind::SharedGate, weights)?;
    let shared_up = weight(graph, load, scope, WeightKind::SharedUp, weights)?;
    let shared_down = weight(graph, load, scope, WeightKind::SharedDown, weights)?;

    let form = input.form.dimensions(hidden);
    let [streams, injected] = join.dimensions();
    let choices = [
        ("M", rows),
        ("H", hidden),
        ("E", shape.count),
        ("K", shape.selected),
        form[0],
        form[1],
    ];
    let experts_dims = [
        ("M", rows),
        ("H", hidden),
        ("E", shape.count),
        ("K", shape.selected),
        ("F", shape.intermediate),
        ("S", shape.shared_intermediate),
    ];
    let output_dims = [
        ("M", rows),
        ("H", hidden),
        ("E", shape.count),
        ("K", shape.selected),
        ("F", shape.intermediate),
        ("S", shape.shared_intermediate),
        streams,
        injected,
    ];
    let mut routes = graph.local_for(handle.route, "routes", &choices)?;
    let mut scores = graph.local_for(handle.route, "scores", &choices)?;
    let normalize = i32::from(shape.normalize_selected);

    if decodes(rows) {
        if let Some(bank) = load.expert_bank().filter(|bank| bank.holds(WeightRole {
            scope,
            kind: WeightKind::ExpertGate,
        })) {
            let streamed = handle
                .streamed
                .as_ref()
                .ok_or("a streamed layer without its bank-served entries")?;
            let RoutedDecodeEntries::SharedRoute { choices, .. } = handle.decode else {
                return Err("a streamed layer without the shared-route decode form".into());
            };
            let inject = join.inject(graph, rows, constants)?;
            return streamed_decode(
                graph,
                streamed,
                (choices, handle.output),
                (&inject, &output_dims),
                load,
                bank,
                scope,
                weights,
                residual,
                [&mut routes, &mut scores],
                rows,
                hidden,
                shape,
                epsilon,
                LayerWeights {
                    norm,
                    router,
                    shared_router,
                    experts: expert_ports,
                    shared_gate,
                    shared_up,
                    shared_down,
                },
            );
        }
        let (expert_product, shared_product, coefficient) = match handle.decode {
            RoutedDecodeEntries::Expand(expand) => {
                let routed = graph.enqueue(
                    handle.route,
                    &choices,
                    routed_route::WorkflowArgs {
                        residual: input.rows.into(),
                        norm: (&norm).into(),
                        router: (&router).into(),
                        shared_router: (&shared_router).into(),
                        routes: routes.tensor_mut().into(),
                        scores: scores.tensor_mut().into(),
                        eps: epsilon,
                        normalize,
                    },
                )?;
                let expanded = graph.enqueue(
                    expand,
                    &experts_dims,
                    routed_expand::WorkflowArgs {
                        normalized: (&routed.r0).into(),
                        routes: routes.tensor().into(),
                        expert_gate: (&expert_gate).into(),
                        expert_up: (&expert_up).into(),
                        shared_gate: (&shared_gate).into(),
                        shared_up: (&shared_up).into(),
                    },
                )?;
                (expanded.r0, expanded.r1, routed.r1)
            }
            RoutedDecodeEntries::SharedRoute {
                route,
                choices: gate_up,
            } => {
                let routed = graph.enqueue(
                    route,
                    &[
                        ("M", rows),
                        ("H", hidden),
                        ("E", shape.count),
                        ("K", shape.selected),
                        ("S", shape.shared_intermediate),
                        form[0],
                        form[1],
                    ],
                    routed_route_shared::WorkflowArgs {
                        residual: input.rows.into(),
                        norm: (&norm).into(),
                        router: (&router).into(),
                        shared_router: (&shared_router).into(),
                        shared_gate: (&shared_gate).into(),
                        shared_up: (&shared_up).into(),
                        routes: routes.tensor_mut().into(),
                        scores: scores.tensor_mut().into(),
                        eps: epsilon,
                        normalize,
                    },
                )?;
                let expert_product = graph
                    .enqueue(
                        gate_up,
                        &[
                            ("M", rows),
                            ("H", hidden),
                            ("E", shape.count),
                            ("K", shape.selected),
                            ("F", shape.intermediate),
                        ],
                        routed_gate_up::WorkflowArgs {
                            normalized: (&routed.r0).into(),
                            routes: routes.tensor().into(),
                            expert_gate: (&expert_gate).into(),
                            expert_up: (&expert_up).into(),
                            // SiLU: the fused Qwen form's experts.
                            activation: 0,
                        },
                    )?
                    .value;
                (expert_product, routed.r2, routed.r1)
            }
        };
        let inject = join.inject(graph, rows, constants)?;
        let value = graph
            .enqueue(
                handle.output,
                &output_dims,
                routed_output::WorkflowArgs {
                    residual: residual.into(),
                    inject: (&inject).into(),
                    expert_product: (&expert_product).into(),
                    shared_product: (&shared_product).into(),
                    routes: routes.tensor().into(),
                    scores: scores.tensor().into(),
                    coefficient: (&coefficient).into(),
                    expert_down: (&expert_down).into(),
                    shared_down: (&shared_down).into(),
                },
            )?
            .value;
        return Ok((value, None));
    }

    let routed = graph.enqueue(
        handle.route,
        &choices,
        routed_route::WorkflowArgs {
            residual: input.rows.into(),
            norm: (&norm).into(),
            router: (&router).into(),
            shared_router: (&shared_router).into(),
            routes: routes.tensor_mut().into(),
            scores: scores.tensor_mut().into(),
            eps: epsilon,
            normalize,
        },
    )?;
    let (normalized, coefficient) = (routed.r0, routed.r1);
    // A streamed layer's products read its scratch banks, which the graph
    // fills in parked rounds (spec 4.9); every value is the resident
    // load's.
    let ([expert_gate, expert_up, expert_down], sequence) =
        match load.expert_bank().filter(|bank| {
            bank.holds(WeightRole {
                scope,
                kind: WeightKind::ExpertGate,
            })
        }) {
            None => ([expert_gate, expert_up, expert_down], None),
            Some(bank) => fill_scratch(
                graph,
                handle
                    .streamed
                    .as_ref()
                    .ok_or("a streamed layer without its scratch fill entries")?,
                load,
                bank,
                scope,
                weights,
                routes.tensor(),
                rows,
                shape,
                expert_ports,
            )
            .map(|(banks, sequence)| (banks, Some(sequence)))?,
        };
    let blocks = grouped_blocks(rows, shape.count, shape.selected)?;
    let tables = [
        ("M", rows),
        ("E", shape.count),
        ("K", shape.selected),
        ("B", blocks),
        ("T", TILE_ROWS),
    ];
    let grouped_experts_dims = [
        ("M", rows),
        ("H", hidden),
        ("E", shape.count),
        ("F", shape.intermediate),
        ("B", blocks),
        ("T", TILE_ROWS),
    ];
    let combine_dims = [
        ("M", rows),
        ("H", hidden),
        ("K", shape.selected),
        ("B", blocks),
        ("T", TILE_ROWS),
        ("S", shape.shared_intermediate),
        streams,
        injected,
    ];
    let mut counts = graph.local_for(handle.group, "counts", &tables)?;
    let mut order = graph.local_for(handle.group, "order", &tables)?;
    let mut inverse = graph.local_for(handle.group, "inverse", &tables)?;
    let mut block_experts = graph.local_for(handle.group, "blocks", &tables)?;
    graph.enqueue(
        handle.group,
        &tables,
        routed_group::WorkflowArgs {
            routes: routes.tensor().into(),
            counts: counts.tensor_mut().into(),
            order: order.tensor_mut().into(),
            inverse: inverse.tensor_mut().into(),
            blocks: block_experts.tensor_mut().into(),
        },
    )?;
    let experts = graph
        .enqueue(
            handle.experts,
            &grouped_experts_dims,
            routed_experts::WorkflowArgs {
                normalized: (&normalized).into(),
                order: order.tensor().into(),
                blocks: block_experts.tensor().into(),
                expert_gate: (&expert_gate).into(),
                expert_up: (&expert_up).into(),
                expert_down: (&expert_down).into(),
                // SiLU: the fused Qwen form's experts.
                activation: 0,
            },
        )?
        .value;
    let inject = join.inject(graph, rows, constants)?;
    let value = graph
        .enqueue(
            handle.combine,
            &combine_dims,
            routed_combine::WorkflowArgs {
                residual: residual.into(),
                inject: (&inject).into(),
                expert_output: (&experts).into(),
                inverse: inverse.tensor().into(),
                scores: scores.tensor().into(),
                normalized: (&normalized).into(),
                coefficient: (&coefficient).into(),
                shared_gate: (&shared_gate).into(),
                shared_up: (&shared_up).into(),
                shared_down: (&shared_down).into(),
            },
        )?
        .value;
    Ok((value, sequence))
}

/// The scratch fill of a scratch-served launch (spec 4.9): the request of
/// the launch's experts, then `rounds` parks, the first followed by the copy
/// of the experts in the layer's bank, each followed by the convert of the
/// round's staged records; then the seeding: the converted experts copied
/// from the scratch banks into the bank slots the host placed them in, and
/// committed. Returns the scratch banks, slot `e` holding expert `e`, and
/// the step-sequence input.
#[allow(clippy::too_many_arguments)]
fn fill_scratch<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: &StreamedGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    bank: &crate::ExpertBankPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    routes: &WorkflowTensor,
    rows: u64,
    shape: &ExpertShape,
    mut banks: [NativePort; 3],
) -> Result<([WorkflowTensor; 3], NativePort), GraphError> {
    let position = bank
        .layers
        .iter()
        .position(|layer| layer.scope == scope)
        .ok_or("a streamed layer outside its bank")?;
    let gate = WeightRole {
        scope,
        kind: WeightKind::ExpertGate,
    };
    let mut part = |graph: &mut G, part| {
        weight_port(graph, load, gate, WeightPart::Bank(part), 0, weights)
    };
    let mut step = part(graph, BankPart::Step)?;
    let heartbeat = part(graph, BankPart::Heartbeat)?;
    let mut control = part(graph, BankPart::ScratchControl)?;
    let mut flags = part(graph, BankPart::ScratchFlags)?;
    let mut scratch = [
        part(graph, BankPart::Scratch { kind: 0 })?,
        part(graph, BankPart::Scratch { kind: 1 })?,
        part(graph, BankPart::Scratch { kind: 2 })?,
    ];
    let staging = [
        part(graph, BankPart::Staging { kind: 0 })?,
        part(graph, BankPart::Staging { kind: 1 })?,
        part(graph, BankPart::Staging { kind: 2 })?,
    ];
    let copy_lists = [
        part(graph, BankPart::CopyList { kind: 0 })?,
        part(graph, BankPart::CopyList { kind: 1 })?,
        part(graph, BankPart::CopyList { kind: 2 })?,
    ];
    let rounds = bank.rounds();
    let mut lists = Vec::with_capacity(rounds as usize);
    for round in 0..rounds {
        let round = u8::try_from(round).map_err(|_| "scratch fill rounds exceed u8")?;
        lists.push([
            part(graph, BankPart::ScratchList { round, kind: 0 })?,
            part(graph, BankPart::ScratchList { round, kind: 1 })?,
            part(graph, BankPart::ScratchList { round, kind: 2 })?,
        ]);
    }
    // A layer-major prompt group interleaves its launches' layers on the
    // device, so every streamed layer of a scratch-served launch sets the
    // step to its own.
    let sequence = graph.input_for(entries.step, "sequence", &[])?;
    graph.enqueue(
        entries.step,
        &[],
        routed_step::WorkflowArgs {
            step: step.tensor_mut().into(),
            sequence: sequence.tensor().into(),
        },
    )?;
    let words = bank.scratch_control_words();
    graph.enqueue(
        entries.scratch_request,
        &[
            ("M", rows),
            ("K", shape.selected),
            ("E", shape.count),
            ("C", words),
        ],
        routed_scratch_request::WorkflowArgs {
            routes: routes.into(),
            step: step.tensor().into(),
            flags: flags.tensor_mut().into(),
            control: control.tensor_mut().into(),
            layer: i32::try_from(position).map_err(|_| "streamed layer index exceeds i32")?,
        },
    )?;
    let word = |index: u64| i32::try_from(index).map_err(|_| "scratch control word exceeds i32");
    for (round, lists) in lists.iter().enumerate() {
        let round = round as u64;
        graph.enqueue(
            entries.scratch_wait,
            &[("C", words)],
            routed_scratch_wait::WorkflowArgs {
                step: step.tensor().into(),
                control: control.tensor_mut().into(),
                heartbeat: heartbeat.tensor().into(),
                acknowledgement: word(4 + round)?,
                arrival: word(4 + rounds + round)?,
            },
        )?;
        for kind in 0..3 {
            let region = &bank.scratch[bank.layers[position].tensors[kind].scratch];
            let matrix = [("N", region.matrix[0]), ("K", region.matrix[1])];
            if round == 0 {
                graph.enqueue(
                    entries.copy[kind],
                    &[
                        ("S", bank.records(scope)?),
                        ("D", shape.count),
                        ("E", shape.count),
                        matrix[0],
                        matrix[1],
                    ],
                    routed_copy::WorkflowArgs {
                        source: banks[kind].tensor().into(),
                        list: copy_lists[kind].tensor().into(),
                        step: step.tensor().into(),
                        destination: scratch[kind].tensor_mut().into(),
                    },
                )?;
            }
            graph.enqueue(
                entries.convert_wide[kind],
                &[
                    ("R", bank.staging_records),
                    ("L", bank.round_records()),
                    ("S", shape.count),
                    matrix[0],
                    matrix[1],
                ],
                routed_convert_wide::WorkflowArgs {
                    source: staging[kind].tensor().into(),
                    list: lists[kind].tensor().into(),
                    step: step.tensor().into(),
                    destination: scratch[kind].tensor_mut().into(),
                },
            )?;
        }
    }
    // The seeding: the bank keeps what the launch read.
    let seed_list = part(graph, BankPart::SeedList)?;
    let mut slot_map = part(graph, BankPart::SlotMap)?;
    let mut stamps = part(graph, BankPart::Stamps)?;
    let mut commits = part(graph, BankPart::Commits)?;
    let records = bank.records(scope)?;
    for kind in 0..3 {
        let region = &bank.scratch[bank.layers[position].tensors[kind].scratch];
        graph.enqueue(
            entries.copy[kind],
            &[
                ("S", shape.count),
                ("D", records),
                ("E", shape.count),
                ("N", region.matrix[0]),
                ("K", region.matrix[1]),
            ],
            routed_copy::WorkflowArgs {
                source: scratch[kind].tensor().into(),
                list: seed_list.tensor().into(),
                step: step.tensor().into(),
                destination: banks[kind].tensor_mut().into(),
            },
        )?;
    }
    graph.enqueue(
        entries.seed_commit,
        &[
            ("E", shape.count),
            ("X", bank.experts),
            ("BANK", records),
            ("C", words),
        ],
        routed_seed_commit::WorkflowArgs {
            list: seed_list.tensor().into(),
            step: step.tensor().into(),
            control: control.tensor().into(),
            slot_map: slot_map.tensor_mut().into(),
            stamps: stamps.tensor_mut().into(),
            commits: commits.tensor_mut().into(),
        },
    )?;
    Ok((scratch.map(|port| port.tensor().clone()), sequence))
}

/// A routed layer's weight ports; the experts are its banks.
struct LayerWeights {
    norm: WorkflowTensor,
    router: WorkflowTensor,
    shared_router: WorkflowTensor,
    experts: [NativePort; 3],
    shared_gate: WorkflowTensor,
    shared_up: WorkflowTensor,
    shared_down: WorkflowTensor,
}

/// The decode rows of a streamed layer (spec 4.6–4.8):
///
/// ```text
/// [step] → route with the shared expert, resolve, hints for the layers ahead
///        → fill: absent choices park, convert into their placed slots, bind
///        → routed_gate_up → routed_output, the resident entries, over the
///          bank and the bound slots
/// ```
///
/// The first routed layer of the target sets the step sequence. The
/// products and the combine are the resident entries themselves, so the
/// value is the resident load's; a step that requested nothing adds one
/// small launch to the resident form.
#[allow(clippy::too_many_arguments)]
fn streamed_decode<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: &StreamedGraphEntries<'a, G>,
    (choices, output): (G::Binding<'a, routed_gate_up::Entry>, G::Binding<'a, routed_output::Entry>),
    (inject, output_dims): (&WorkflowTensor, &[(&str, u64)]),
    load: &ModelLoadPlan,
    bank: &crate::ExpertBankPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    residual: &WorkflowTensor,
    [routes, scores]: [&mut NativePort; 2],
    rows: u64,
    hidden: u64,
    shape: &ExpertShape,
    epsilon: f32,
    layer: LayerWeights,
) -> Result<(WorkflowTensor, Option<NativePort>), GraphError> {
    bank.shared
        .as_ref()
        .ok_or("slab decode requires an admitted shared plan")?;
    let LayerWeights {
        norm,
        router,
        shared_router,
        experts: [mut expert_gate, mut expert_up, mut expert_down],
        shared_gate,
        shared_up,
        shared_down,
    } = layer;
    let position = bank
        .layers
        .iter()
        .position(|layer| layer.scope == scope)
        .ok_or("a streamed layer outside its bank")?;
    let gate = WeightRole {
        scope,
        kind: WeightKind::ExpertGate,
    };
    let part = |graph: &mut G, weights: &mut Vec<(WeightPort, NativePort)>, part| {
        weight_port(graph, load, gate, WeightPart::Bank(part), 0, weights)
    };
    let mut step = part(graph, weights, BankPart::Step)?;
    let mut slot_map = part(graph, weights, BankPart::SlotMap)?;
    let mut stamps = part(graph, weights, BankPart::Stamps)?;
    let mut control = part(graph, weights, BankPart::Control)?;
    let mut near_ring = part(graph, weights, BankPart::Ring { distance: 1 })?;
    let mut far_ring = part(graph, weights, BankPart::Ring { distance: 2 })?;

    // A bank-served launch's layers run in order, so its first streamed
    // layer sets the step for all of them.
    let sequence = match position {
        0 => {
            let sequence = graph.input_for(entries.step, "sequence", &[])?;
            graph.enqueue(
                entries.step,
                &[],
                routed_step::WorkflowArgs {
                    step: step.tensor_mut().into(),
                    sequence: sequence.tensor().into(),
                },
            )?;
            Some(sequence)
        }
        _ => None,
    };

    // The layers ahead (spec 4.7), predicted from this layer's residual
    // through their own norm and router into this layer's rings. Near the
    // end of the target fewer layers lie ahead; the route reads none of a
    // missing distance's tensors, which take this layer's.
    let WeightScope::TargetSublayer(SublayerIndex { block, .. }) = scope else {
        return Err("a streamed layer outside the target".into());
    };
    let mut ahead = Vec::with_capacity(crate::streaming::layout::DISTANCES.len());
    for distance in crate::streaming::layout::DISTANCES {
        let Some(later_layer) = bank.layers.get(position + usize::from(distance)) else {
            break;
        };
        let WeightScope::TargetSublayer(SublayerIndex { block: later, .. }) = later_layer.scope else {
            return Err("a streamed layer outside the target".into());
        };
        let offset = later
            .checked_sub(block)
            .ok_or("streamed layers out of decoder order")?;
        let mut ahead_port = |graph: &mut G, kind, part| {
            weight_port(
                graph,
                load,
                WeightRole {
                    scope: later_layer.scope,
                    kind,
                },
                part,
                offset,
                weights,
            )
        };
        ahead.push([
            ahead_port(graph, WeightKind::InputNorm, WeightPart::Values)?.tensor().clone(),
            ahead_port(graph, WeightKind::Router, WeightPart::Values)?.tensor().clone(),
            ahead_port(graph, WeightKind::ExpertGate, WeightPart::Bank(BankPart::SlotMap))?
                .tensor()
                .clone(),
        ]);
    }
    let predicted = ahead.len();
    let own = [norm.clone(), router.clone(), slot_map.tensor().clone()];
    let [near_norm, near_router, near_map] = ahead.first().cloned().unwrap_or_else(|| own.clone());
    let [far_norm, far_router, far_map] = ahead.get(1).cloned().unwrap_or(own);
    let routed = graph.enqueue(
        entries.route,
        &[
            ("M", rows),
            ("H", hidden),
            // A streamed layer's router input is one normalized stream
            // (preparation refuses mixed inputs).
            ("PT", 1),
            ("NM", hidden),
            ("E", shape.count),
            ("K", shape.selected),
            ("S", shape.shared_intermediate),
            ("BANK", bank.records(scope)?),
            ("C", bank.control_words()),
            ("RING", bank.ring_words()),
        ],
        routed_route_streamed::WorkflowArgs {
            residual: residual.into(),
            norm: (&norm).into(),
            router: (&router).into(),
            shared_router: (&shared_router).into(),
            shared_gate: (&shared_gate).into(),
            shared_up: (&shared_up).into(),
            routes: routes.tensor_mut().into(),
            scores: scores.tensor_mut().into(),
            slot_map: slot_map.tensor().into(),
            stamps: stamps.tensor_mut().into(),
            step: step.tensor().into(),
            control: control.tensor_mut().into(),
            near_norm: (&near_norm).into(),
            near_router: (&near_router).into(),
            near_map: (&near_map).into(),
            far_norm: (&far_norm).into(),
            far_router: (&far_router).into(),
            far_map: (&far_map).into(),
            near_ring: near_ring.tensor_mut().into(),
            far_ring: far_ring.tensor_mut().into(),
            eps: epsilon,
            normalize: i32::from(shape.normalize_selected),
            ahead: i32::try_from(predicted).map_err(|_| "too many layers ahead")?,
        },
    )?;
    let (normalized, coefficient, shared_product, slots) =
        (routed.r0, routed.r1, routed.r2, routed.r3);
    // The resident choices' products while the host serves the absent ones
    // (4.8); late threadgroups park until acknowledgement, convert their
    // expert from staging into its placed slot, and compute it in-launch.
    // The absent choices: parked until the host acknowledges, converted
    // from staging into the slots it placed them in (4.6).
    let placement = part(graph, weights, BankPart::Placement)?;
    let heartbeat = part(graph, weights, BankPart::Heartbeat)?;
    let staged = [
        part(graph, weights, BankPart::Staging { kind: 0 })?,
        part(graph, weights, BankPart::Staging { kind: 1 })?,
        part(graph, weights, BankPart::Staging { kind: 2 })?,
    ];
    let mut commits = part(graph, weights, BankPart::Commits)?;
    let bound = graph
        .enqueue(
            entries.fill,
            &[
                ("M", rows),
                ("H", hidden),
                ("K", shape.selected),
                ("F", shape.intermediate),
                ("BANK", bank.records(scope)?),
                ("T", bank.staging_records),
                ("X", bank.experts),
                ("C", bank.control_words()),
                ("P", bank.placement_choices()),
            ],
            routed_bank_fill::WorkflowArgs {
                slots: (&slots).into(),
                stamps: stamps.tensor_mut().into(),
                step: step.tensor().into(),
                control: control.tensor_mut().into(),
                placement: placement.tensor().into(),
                heartbeat: heartbeat.tensor().into(),
                staged_gate: staged[0].tensor().into(),
                staged_up: staged[1].tensor().into(),
                staged_down: staged[2].tensor().into(),
                expert_gate: expert_gate.tensor_mut().into(),
                expert_up: expert_up.tensor_mut().into(),
                expert_down: expert_down.tensor_mut().into(),
                slot_map: slot_map.tensor_mut().into(),
                commits: commits.tensor_mut().into(),
            },
        )?
        .value;
    // The resident choices and output over the bank and the bound slots:
    // the value is the resident load's, bit for bit.
    let records = bank.records(scope)?;
    let expert_product = graph
        .enqueue(
            choices,
            &[
                ("M", rows),
                ("H", hidden),
                ("E", records),
                ("K", shape.selected),
                ("F", shape.intermediate),
            ],
            routed_gate_up::WorkflowArgs {
                normalized: (&normalized).into(),
                routes: (&bound).into(),
                expert_gate: expert_gate.tensor().into(),
                expert_up: expert_up.tensor().into(),
                // SiLU: the fused Qwen form's experts.
                activation: 0,
            },
        )?
        .value;
    let output_dims = output_dims
        .iter()
        .map(|&(name, value)| (name, if name == "E" { records } else { value }))
        .collect::<Vec<_>>();
    let value = graph
        .enqueue(
            output,
            &output_dims,
            routed_output::WorkflowArgs {
                residual: residual.into(),
                inject: inject.into(),
                expert_product: (&expert_product).into(),
                shared_product: (&shared_product).into(),
                routes: (&bound).into(),
                scores: scores.tensor().into(),
                coefficient: (&coefficient).into(),
                expert_down: expert_down.tensor().into(),
                shared_down: (&shared_down).into(),
            },
        )?
        .value;
    Ok((value, sequence))
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, Layout};

    /// A down projection wider than the Metal kernel's register budget
    /// admits (`ceil_div(F, 32 * LANES) * ROWS <= 8`: at most 8,192) is
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
        let scores = port(Element::f32(), &[m, k]);
        let coefficient = port(Element::f32(), &[m]);
        let expert_down = port(weight, &[e, h, f]);
        let shared_down = port(weight, &[h, s]);
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
            ],
            routed_output::WorkflowArgs {
                residual: (&residual).into(),
                inject: (&inject).into(),
                expert_product: (&expert_product).into(),
                shared_product: (&shared_product).into(),
                routes: (&routes).into(),
                scores: (&scores).into(),
                coefficient: (&coefficient).into(),
                expert_down: (&expert_down).into(),
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
            [("F", f), ("H", h), ("IN", 0), ("K", k), ("S", s), ("SN", 1),]
                .map(|(name, value)| (name.to_owned(), value))
                .to_vec()
        );
    }
}
