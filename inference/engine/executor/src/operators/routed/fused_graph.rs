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

use crate::programs::graph::{draft::GraphDraft, GraphError};
use crate::programs::native_target_graph::{weight, WeightPort};
use crate::{
    native::{RoutedDecodeKernels, RoutedKernels},
    ModelLoadPlan, RoutedBinding,
};
use magnitude_family_contracts::{
    RouteNormalization, RoutedFfn, WeightKind, WeightScope,
};
use magnitude_kernels::{
    routed_combine, routed_expand, routed_experts, routed_gate_up, routed_group, routed_output,
    routed_route, routed_route_shared,
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

pub(crate) struct CheckedRoutedEntries {
    route: [(&'static str, Element); 3],
    decode: CheckedDecodeEntries,
    output: [(&'static str, Element); 3],
    group: [(&'static str, Element); 0],
    experts: [(&'static str, Element); 4],
    combine: [(&'static str, Element); 4],
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

/// The routed feed-forward of one block over `rows` rows of `residual`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn routed<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    handle: RoutedGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    residual: &WorkflowTensor,
    rows: u64,
    hidden: u64,
    shape: &ExpertShape,
    epsilon: f32,
) -> Result<WorkflowTensor, GraphError> {
    let norm = weight(graph, load, scope, WeightKind::InputNorm, weights)?;
    let router = weight(graph, load, scope, WeightKind::Router, weights)?;
    let shared_router = weight(graph, load, scope, WeightKind::SharedRouter, weights)?;
    let expert_gate = weight(graph, load, scope, WeightKind::ExpertGate, weights)?;
    let expert_up = weight(graph, load, scope, WeightKind::ExpertUp, weights)?;
    let expert_down = weight(graph, load, scope, WeightKind::ExpertDown, weights)?;
    let shared_gate = weight(graph, load, scope, WeightKind::SharedGate, weights)?;
    let shared_up = weight(graph, load, scope, WeightKind::SharedUp, weights)?;
    let shared_down = weight(graph, load, scope, WeightKind::SharedDown, weights)?;

    let choices = [
        ("M", rows),
        ("H", hidden),
        ("E", shape.count),
        ("K", shape.selected),
    ];
    let experts_dims = [
        ("M", rows),
        ("H", hidden),
        ("E", shape.count),
        ("K", shape.selected),
        ("F", shape.intermediate),
        ("S", shape.shared_intermediate),
    ];
    let mut routes = graph.local_for(handle.route, "routes", &choices)?;
    let mut scores = graph.local_for(handle.route, "scores", &choices)?;
    let normalize = i32::from(shape.normalize_selected);

    if decodes(rows) {
        let (expert_product, shared_product, coefficient) = match handle.decode {
            RoutedDecodeEntries::Expand(expand) => {
                let routed = graph.enqueue(
                    handle.route,
                    &choices,
                    routed_route::WorkflowArgs {
                        residual: residual.into(),
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
                    ],
                    routed_route_shared::WorkflowArgs {
                        residual: residual.into(),
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
        return Ok(graph
            .enqueue(
                handle.output,
                &experts_dims,
                routed_output::WorkflowArgs {
                    residual: residual.into(),
                    expert_product: (&expert_product).into(),
                    shared_product: (&shared_product).into(),
                    routes: routes.tensor().into(),
                    scores: scores.tensor().into(),
                    coefficient: (&coefficient).into(),
                    expert_down: (&expert_down).into(),
                    shared_down: (&shared_down).into(),
                },
            )?
            .value);
    }

    let routed = graph.enqueue(
        handle.route,
        &choices,
        routed_route::WorkflowArgs {
            residual: residual.into(),
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
    Ok(graph
        .enqueue(
            handle.combine,
            &combine_dims,
            routed_combine::WorkflowArgs {
                residual: residual.into(),
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
        .value)
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
            &[("M", m), ("H", h), ("E", e), ("K", k), ("F", f), ("S", s)],
            routed_output::WorkflowArgs {
                residual: (&residual).into(),
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
            [("F", f), ("H", h), ("K", k), ("S", s)]
                .map(|(name, value)| (name.to_owned(), value))
                .to_vec()
        );
    }
}
