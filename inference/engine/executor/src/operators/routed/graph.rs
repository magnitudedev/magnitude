//! The general routed feed-forward graph (`operators::routed`): the shared
//! expert's dense feed-forward onto the residual as the `base`, selection,
//! an optional latent down projection, the experts (decode form up to
//! [`DECODE_ROWS`](super::fused_graph::DECODE_ROWS) rows, grouped form beyond) summed onto the base, and an
//! optional latent up projection. Row tables, the selection bias and scales
//! a model lacks and a latent sum's zero base are graph constants.

use super::fused_graph::{decodes, grouped_blocks, TILE_ROWS};
use crate::programs::graph::{draft::GraphDraft, GraphError};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{
    resident_scale, scaled_weight, weight, WeightPort,
};
use crate::native::{DenseExpansionKernel, ExpertKernels, GeneralRoutedKernels};
use crate::{GeneralRoutedBinding, GeneralRoutedShape, ModelLoadPlan};
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::{
    dense_expand, dense_output, dense_up, project_rows, routed_down, routed_experts,
    routed_experts_up, routed_gate_up, routed_group, routed_scatter, routed_select, routed_up,
};
use seismic::{Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

/// The experts' expansion entries.
pub(crate) enum ExpertEntries<'a, G: GraphDraft + 'a> {
    Gated {
        decode: G::Binding<'a, routed_gate_up::Entry>,
        grouped: G::Binding<'a, routed_experts::Entry>,
    },
    Plain {
        decode: G::Binding<'a, routed_up::Entry>,
        grouped: G::Binding<'a, routed_experts_up::Entry>,
    },
}

/// What the routed sum accumulates onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RoutedSum {
    /// The residual: the sublayer's output is the next residual.
    Residual,
    /// Zeros: the output is a branch's own, which a tail combines
    /// (`moe_tail`).
    Branch,
}

/// A dense expansion entry.
pub(crate) enum ExpansionEntry<'a, G: GraphDraft + 'a> {
    Gated(G::Binding<'a, dense_expand::Entry>),
    Plain(G::Binding<'a, dense_up::Entry>),
}

pub(crate) struct GeneralRoutedGraphEntries<'a, G: GraphDraft + 'a> {
    pub select: G::Binding<'a, routed_select::Entry>,
    pub experts: ExpertEntries<'a, G>,
    pub down: G::Binding<'a, routed_down::Entry>,
    pub group: G::Binding<'a, routed_group::Entry>,
    pub scatter: G::Binding<'a, routed_scatter::Entry>,
    pub shared: Option<(ExpansionEntry<'a, G>, G::Binding<'a, dense_output::Entry>)>,
    pub latent: Option<(
        G::Binding<'a, project_rows::Entry>,
        G::Binding<'a, dense_output::Entry>,
    )>,
}

impl<'a> From<&'a GeneralRoutedKernels> for GeneralRoutedGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a GeneralRoutedKernels) -> Self {
        Self {
            select: &kernels.select,
            experts: match &kernels.experts {
                ExpertKernels::Gated { decode, grouped } => ExpertEntries::Gated { decode, grouped },
                ExpertKernels::Plain { decode, grouped } => ExpertEntries::Plain { decode, grouped },
            },
            down: &kernels.down,
            group: &kernels.group,
            scatter: &kernels.scatter,
            shared: kernels.shared.as_ref().map(|(expansion, output)| {
                (
                    match expansion {
                        DenseExpansionKernel::Gated(kernel) => ExpansionEntry::Gated(kernel),
                        DenseExpansionKernel::Plain(kernel) => ExpansionEntry::Plain(kernel),
                    },
                    output,
                )
            }),
            latent: kernels.latent.as_ref().map(|(down, up)| (down, up)),
        }
    }
}

/// Entry element assignments from the program binding.
pub(crate) struct CheckedGeneralRoutedEntries {
    select: [(&'static str, Element); 4],
    gate_up: [(&'static str, Element); 3],
    up: [(&'static str, Element); 2],
    experts: [(&'static str, Element); 4],
    experts_up: [(&'static str, Element); 3],
    down: [(&'static str, Element); 3],
    group: [(&'static str, Element); 0],
    scatter: [(&'static str, Element); 2],
    shared_expand: [(&'static str, Element); 4],
    shared_up: [(&'static str, Element); 3],
    shared_output: [(&'static str, Element); 2],
    latent_down: [(&'static str, Element); 3],
    latent_up: [(&'static str, Element); 2],
    binding: GeneralRoutedBinding,
}

/// The element the routed sum publishes in: activations for a latent sum
/// that feeds the latent up projection, else F32 onto the residual.
pub(crate) fn sum_element(binding: &GeneralRoutedBinding) -> Element {
    if binding.shape.latent {
        binding.activation
    } else {
        Element::f32()
    }
}

impl CheckedGeneralRoutedEntries {
    pub(crate) fn new(binding: GeneralRoutedBinding) -> Self {
        let a = binding.activation;
        // An absent weight's element is never read: its entry is not built.
        let gate = binding.expert_gate.unwrap_or(binding.expert_up);
        let (shared_gate, shared_up, shared_down) = binding
            .shared
            .map(|(gate, up, down)| (gate.unwrap_or(up), up, down))
            .unwrap_or((a, a, a));
        let (latent_down, latent_up) = binding.latent.unwrap_or((a, a));
        Self {
            select: [
                ("NW", binding.norm),
                ("RNW", binding.router_norm),
                ("RW", binding.router),
                ("A", a),
            ],
            gate_up: [("EGW", gate), ("EUW", binding.expert_up), ("A", a)],
            up: [("EUW", binding.expert_up), ("A", a)],
            experts: [
                ("EGW", gate),
                ("EUW", binding.expert_up),
                ("EDW", binding.expert_down),
                ("A", a),
            ],
            experts_up: [
                ("EUW", binding.expert_up),
                ("EDW", binding.expert_down),
                ("A", a),
            ],
            down: [
                ("EDW", binding.expert_down),
                ("A", a),
                ("R", sum_element(&binding)),
            ],
            group: [],
            scatter: [("A", a), ("R", sum_element(&binding))],
            shared_expand: [
                ("NW", binding.norm),
                ("GW", shared_gate),
                ("UW", shared_up),
                ("A", a),
            ],
            shared_up: [("NW", binding.norm), ("UW", shared_up), ("A", a)],
            shared_output: [("DW", shared_down), ("A", a)],
            latent_down: [("A", a), ("W", latent_down), ("Y", a)],
            latent_up: [("DW", latent_up), ("A", a)],
            binding,
        }
    }

    pub(crate) fn entries(&self) -> GeneralRoutedGraphEntries<'_, NativeGraphMetadata> {
        let shape = self.binding.shape;
        GeneralRoutedGraphEntries {
            select: &self.select,
            experts: if shape.experts_expansion.gated {
                ExpertEntries::Gated {
                    decode: &self.gate_up,
                    grouped: &self.experts,
                }
            } else {
                ExpertEntries::Plain {
                    decode: &self.up,
                    grouped: &self.experts_up,
                }
            },
            down: &self.down,
            group: &self.group,
            scatter: &self.scatter,
            shared: shape.shared.map(|(_, expansion)| {
                (
                    if expansion.gated {
                        ExpansionEntry::Gated(&self.shared_expand[..])
                    } else {
                        ExpansionEntry::Plain(&self.shared_up[..])
                    },
                    &self.shared_output[..],
                )
            }),
            latent: shape
                .latent
                .then_some((&self.latent_down[..], &self.latent_up[..])),
        }
    }
}

/// The general routed feed-forward of one block over `rows` rows of
/// `residual`, returning the new residual rows (F32).
#[allow(clippy::too_many_arguments)]
pub(crate) fn general_routed<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: GeneralRoutedGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    residual: &WorkflowTensor,
    sum: RoutedSum,
    rows: u64,
    shape: &GeneralRoutedShape,
    epsilon: f32,
) -> Result<WorkflowTensor, GraphError> {
    let mut weight = |graph: &mut G, kind| weight(graph, load, scope, kind, weights);
    let norm = weight(graph, WeightKind::InputNorm)?;
    let router_norm = if shape.router_norm {
        weight(graph, WeightKind::RouterInputNorm)?
    } else {
        norm.clone()
    };
    let router = weight(graph, WeightKind::Router)?;
    let bias = if shape.bias {
        Some(weight(graph, WeightKind::RouterSelectionBias)?)
    } else {
        None
    };
    let expert_scale = if shape.expert_scale {
        Some(weight(graph, WeightKind::ExpertScale)?)
    } else {
        None
    };
    let expert_gate = if shape.experts_expansion.gated {
        Some(weight(graph, WeightKind::ExpertGate)?)
    } else {
        None
    };
    let expert_up = weight(graph, WeightKind::ExpertUp)?;
    let expert_down = weight(graph, WeightKind::ExpertDown)?;
    // Per-expert scales of the stacked expert weights, when they have them.
    let expert_up_scale = resident_scale(graph, load, scope, WeightKind::ExpertUp, weights)?;
    let expert_down_scale = resident_scale(graph, load, scope, WeightKind::ExpertDown, weights)?;
    let mut scaled =
        |graph: &mut G, kind| scaled_weight(graph, load, scope, kind, weights, constants);
    let shared_weights = match shape.shared {
        Some((_, expansion)) => Some((
            if expansion.gated {
                Some(scaled(graph, WeightKind::SharedGate)?)
            } else {
                None
            },
            scaled(graph, WeightKind::SharedUp)?,
            scaled(graph, WeightKind::SharedDown)?,
        )),
        None => None,
    };
    let latent_weights = if shape.latent {
        Some((
            scaled(graph, WeightKind::LatentDown)?,
            scaled(graph, WeightKind::LatentUp)?,
        ))
    } else {
        None
    };
    // The selection bias a model lacks is zeros, absent expert scales ones.
    // The down weight's per-expert second-level scales are the combine's
    // expert scales (plan admission refuses both at once).
    let experts = usize::try_from(shape.experts).map_err(|_| "expert count exceeds host")?;
    let bias = match bias {
        Some(bias) => bias,
        None => {
            let constant = GraphConstant::f32(graph, &vec![0.0; experts])?;
            let tensor = constant.port().tensor().clone();
            constants.push(constant);
            tensor
        }
    };
    let expert_scale = match (expert_scale, expert_down_scale) {
        (Some(scale), None) | (None, Some(scale)) => scale,
        (Some(_), Some(_)) => {
            return Err("stored expert scales beside a scaled expert down weight".into())
        }
        (None, None) => {
            let constant = GraphConstant::f32(graph, &vec![1.0; experts])?;
            let tensor = constant.port().tensor().clone();
            constants.push(constant);
            tensor
        }
    };
    // Up-only experts' accumulator scales: the up weight's per-expert
    // second-level scales, else ones (gated experts bind none; plan
    // admission refuses a scaled gated expert).
    let up_scale = match (shape.experts_expansion.gated, expert_up_scale) {
        (true, None) => None,
        (true, Some(_)) => return Err("gated experts have no up scale port".into()),
        (false, Some(scale)) => Some(scale),
        (false, None) => {
            let constant = GraphConstant::f32(graph, &vec![1.0; experts])?;
            let tensor = constant.port().tensor().clone();
            constants.push(constant);
            Some(tensor)
        }
    };
    // The row table of the dense projections (shared expert, latent up): a
    // graph port only when one reads it (an unread port is unbound).
    let out_rows = (shape.shared.is_some() || shape.latent)
        .then(|| GraphConstant::identity_for_class(graph, rows, Some("M")))
        .transpose()?;
    let row_table = || {
        out_rows
            .as_ref()
            .map(|table| table.port().tensor().clone())
            .ok_or("general routed row table is absent")
    };

    let root = match sum {
        RoutedSum::Residual => residual.clone(),
        RoutedSum::Branch => {
            let zeros = GraphConstant::zeros_for_class(graph, rows, shape.hidden, "M")?;
            let tensor = zeros.port().tensor().clone();
            constants.push(zeros);
            tensor
        }
    };
    // The shared expert onto the root: the base of the routed sum.
    let base = match (entries.shared, &shared_weights) {
        (Some((expansion, output)), Some((gate, up, down))) => {
            let (features, _) = shape.shared.ok_or("shared expert shape is absent")?;
            let dimensions = [("M", rows), ("O", rows), ("H", shape.hidden), ("F", features)];
            let activation = shape
                .shared
                .map(|(_, expansion)| expansion.activation)
                .ok_or("shared expert shape is absent")?;
            let product = match (expansion, gate) {
                (ExpansionEntry::Gated(entry), Some(gate)) => {
                    graph
                        .enqueue(
                            entry,
                            &[dimensions.as_slice(), &[("GS", gate.extent), ("US", up.extent)]]
                                .concat(),
                            dense_expand::WorkflowArgs {
                                residual: residual.into(),
                                norm: (&norm).into(),
                                gate_weight: (&gate.weight).into(),
                                up_weight: (&up.weight).into(),
                                out_rows: (&row_table()?).into(),
                                eps: epsilon,
                                activation,
                                gate_scale: (&gate.scale).into(),
                                up_scale: (&up.scale).into(),
                            },
                        )?
                        .value
                }
                (ExpansionEntry::Plain(entry), None) => {
                    graph
                        .enqueue(
                            entry,
                            &[dimensions.as_slice(), &[("US", up.extent)]].concat(),
                            dense_up::WorkflowArgs {
                                residual: residual.into(),
                                norm: (&norm).into(),
                                up_weight: (&up.weight).into(),
                                out_rows: (&row_table()?).into(),
                                eps: epsilon,
                                activation,
                                up_scale: (&up.scale).into(),
                            },
                        )?
                        .value
                }
                _ => return Err("shared expert entries disagree with its weights".into()),
            };
            graph
                .enqueue(
                    output,
                    &[dimensions.as_slice(), &[("DS", down.extent)]].concat(),
                    dense_output::WorkflowArgs {
                        residual: (&root).into(),
                        product: (&product).into(),
                        down_weight: (&down.weight).into(),
                        out_rows: (&row_table()?).into(),
                        down_scale: (&down.scale).into(),
                    },
                )?
                .value
        }
        (None, None) => root,
        _ => return Err("shared expert entries disagree with its weights".into()),
    };

    let choices = shape.select_dimensions(rows);
    let mut routes = graph.local_for(entries.select, "routes", &choices)?;
    let mut route_weights = graph.local_for(entries.select, "weights", &choices)?;
    let normalized = graph
        .enqueue(
            entries.select,
            &choices,
            routed_select::WorkflowArgs {
                residual: residual.into(),
                norm: (&norm).into(),
                router_norm: (&router_norm).into(),
                router: (&router).into(),
                bias: (&bias).into(),
                expert_scale: (&expert_scale).into(),
                routes: routes.tensor_mut().into(),
                weights: route_weights.tensor_mut().into(),
                epsilon,
                score: shape.score,
                normalization: shape.normalization,
                normalization_epsilon: f32::from_bits(shape.normalization_epsilon),
                scale: f32::from_bits(shape.scale),
            },
        )?
        .value;

    // A latent operator's experts act on the projected-down input and sum
    // onto zeros in the latent width.
    let latent = match (entries.latent, &latent_weights) {
        (Some((down_entry, up_entry)), Some((down, up))) => {
            let projected = graph
                .enqueue(
                    down_entry,
                    &[
                        ("M", rows),
                        ("K", shape.hidden),
                        ("N", shape.expert_hidden),
                        ("WS", down.extent),
                    ],
                    project_rows::WorkflowArgs {
                        source: (&normalized).into(),
                        weight: (&down.weight).into(),
                        weight_scale: (&down.scale).into(),
                    },
                )?
                .value;
            let zeros = GraphConstant::zeros_for_class(graph, rows, shape.expert_hidden, "M")?;
            Some((projected, zeros, up_entry, up))
        }
        (None, None) => None,
        _ => return Err("latent entries disagree with its weights".into()),
    };
    let (expert_input, sum_base) = match &latent {
        Some((projected, zeros, _, _)) => (projected.clone(), zeros.port().tensor().clone()),
        None => (normalized.clone(), base.clone()),
    };
    let activation = shape.experts_expansion.activation;
    let routed = if decodes(rows) {
        let dimensions = shape.expert_dimensions(rows);
        let product = match (entries.experts, &expert_gate) {
            (ExpertEntries::Gated { decode, .. }, Some(gate)) => {
                graph
                    .enqueue(
                        decode,
                        &dimensions,
                        routed_gate_up::WorkflowArgs {
                            normalized: (&expert_input).into(),
                            routes: routes.tensor().into(),
                            expert_gate: gate.into(),
                            expert_up: (&expert_up).into(),
                            activation,
                        },
                    )?
                    .value
            }
            (ExpertEntries::Plain { decode, .. }, None) => {
                graph
                    .enqueue(
                        decode,
                        &dimensions,
                        routed_up::WorkflowArgs {
                            normalized: (&expert_input).into(),
                            routes: routes.tensor().into(),
                            expert_up: (&expert_up).into(),
                            up_scale: up_scale
                                .as_ref()
                                .ok_or("up-only experts without up scales")?
                                .into(),
                            activation,
                        },
                    )?
                    .value
            }
            _ => return Err("expert entries disagree with its weights".into()),
        };
        graph
            .enqueue(
                entries.down,
                &dimensions,
                routed_down::WorkflowArgs {
                    base: (&sum_base).into(),
                    product: (&product).into(),
                    routes: routes.tensor().into(),
                    weights: route_weights.tensor().into(),
                    expert_down: (&expert_down).into(),
                },
            )?
            .value
    } else {
        let blocks = grouped_blocks(rows, shape.experts, shape.selected)?;
        let tables = [
            ("M", rows),
            ("E", shape.experts),
            ("K", shape.selected),
            ("B", blocks),
            ("T", TILE_ROWS),
        ];
        let mut counts = graph.local_for(entries.group, "counts", &tables)?;
        let mut order = graph.local_for(entries.group, "order", &tables)?;
        let mut inverse = graph.local_for(entries.group, "inverse", &tables)?;
        let mut block_experts = graph.local_for(entries.group, "blocks", &tables)?;
        graph.enqueue(
            entries.group,
            &tables,
            routed_group::WorkflowArgs {
                routes: routes.tensor().into(),
                counts: counts.tensor_mut().into(),
                order: order.tensor_mut().into(),
                inverse: inverse.tensor_mut().into(),
                blocks: block_experts.tensor_mut().into(),
            },
        )?;
        let grouped = [
            ("M", rows),
            ("H", shape.expert_hidden),
            ("E", shape.experts),
            ("F", shape.intermediate),
            ("B", blocks),
            ("T", TILE_ROWS),
        ];
        let expert_output = match (entries.experts, &expert_gate) {
            (ExpertEntries::Gated { grouped: entry, .. }, Some(gate)) => {
                graph
                    .enqueue(
                        entry,
                        &grouped,
                        routed_experts::WorkflowArgs {
                            normalized: (&expert_input).into(),
                            order: order.tensor().into(),
                            blocks: block_experts.tensor().into(),
                            expert_gate: gate.into(),
                            expert_up: (&expert_up).into(),
                            expert_down: (&expert_down).into(),
                            activation,
                        },
                    )?
                    .value
            }
            (ExpertEntries::Plain { grouped: entry, .. }, None) => {
                graph
                    .enqueue(
                        entry,
                        &grouped,
                        routed_experts_up::WorkflowArgs {
                            normalized: (&expert_input).into(),
                            order: order.tensor().into(),
                            blocks: block_experts.tensor().into(),
                            expert_up: (&expert_up).into(),
                            expert_down: (&expert_down).into(),
                            up_scale: up_scale
                                .as_ref()
                                .ok_or("up-only experts without up scales")?
                                .into(),
                            activation,
                        },
                    )?
                    .value
            }
            _ => return Err("expert entries disagree with its weights".into()),
        };
        graph
            .enqueue(
                entries.scatter,
                &[
                    ("M", rows),
                    ("H", shape.expert_hidden),
                    ("K", shape.selected),
                    ("B", blocks),
                    ("T", TILE_ROWS),
                ],
                routed_scatter::WorkflowArgs {
                    base: (&sum_base).into(),
                    expert_output: (&expert_output).into(),
                    inverse: inverse.tensor().into(),
                    weights: route_weights.tensor().into(),
                },
            )?
            .value
    };
    let output = match latent {
        Some((_, zeros, up_entry, up)) => {
            let output = graph
                .enqueue(
                    up_entry,
                    &[
                        ("M", rows),
                        ("O", rows),
                        ("H", shape.hidden),
                        ("F", shape.expert_hidden),
                        ("DS", up.extent),
                    ],
                    dense_output::WorkflowArgs {
                        residual: (&base).into(),
                        product: (&routed).into(),
                        down_weight: (&up.weight).into(),
                        out_rows: (&row_table()?).into(),
                        down_scale: (&up.scale).into(),
                    },
                )?
                .value;
            constants.push(zeros);
            output
        }
        None => routed,
    };
    constants.extend(out_rows);
    Ok(output)
}

/// The row-class constants a general routed sublayer binds at `rows` rows
/// (for resource accounting): the dense projections' row table, a branch's
/// zero root and a latent sum's zero base.
pub(crate) fn class_constants(
    shape: &GeneralRoutedShape,
    sum: RoutedSum,
    rows: u64,
) -> Result<Vec<GraphConstant>, String> {
    let mut constants = Vec::new();
    if sum == RoutedSum::Branch {
        constants.push(GraphConstant::zeros_value(rows, shape.hidden)?);
    }
    if shape.shared.is_some() || shape.latent {
        constants.push(GraphConstant::identity_value(rows)?);
    }
    if shape.latent {
        constants.push(GraphConstant::zeros_value(rows, shape.expert_hidden)?);
    }
    Ok(constants)
}
