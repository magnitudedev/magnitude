//! Routed (mixture-of-experts) feed-forward graph construction (program spec
//! K6, E8).
//!
//! Row classes up to [`DECODE_ROWS`] run the decode form: route, expand,
//! output. Larger classes run the grouped form: route, group, experts,
//! combine. The grouping tables and grouped intermediates are graph locals
//! sized from the class, the selected-expert count and [`TILE_ROWS`], so the
//! graph plan charges them to the workspace; nothing is uploaded per step.

use super::super::native_target_graph::weight;
use crate::{native::RoutedKernels, ModelLoadPlan};
use magnitude_model_contracts::{ExpertGeometry, WeightKind, WeightRole, WeightScope};
use magnitude_model_kernels::{
    qwen_routed_combine, qwen_routed_expand, qwen_routed_experts, qwen_routed_group,
    qwen_routed_output, qwen_routed_route,
};
use seismic::{NativeGraph, NativePort, WorkflowTensor};

/// The largest row class the decode form serves: the K1 GEMV row bound.
pub(crate) const DECODE_ROWS: u64 = 8;

/// Rows of one expert tile of the grouped form.
pub(crate) const TILE_ROWS: u64 = 32;

/// Expert tiles that hold every choice of a `rows`-row class: `rows *
/// selected` choices, each expert's rows padded to whole tiles,
/// `ceil((rows * selected + experts * (TILE_ROWS - 1)) / TILE_ROWS)`.
pub(crate) fn grouped_blocks(rows: u64, experts: u64, selected: u64) -> Result<u64, String> {
    rows.checked_mul(selected)
        .zip(experts.checked_mul(TILE_ROWS - 1))
        .and_then(|(choices, padding)| choices.checked_add(padding))
        .map(|slots| slots.div_ceil(TILE_ROWS))
        .ok_or_else(|| format!("grouped tiles of {rows} rows overflow"))
}

fn failed(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// The routed feed-forward of one block over `rows` rows of `residual`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn routed(
    graph: &mut NativeGraph,
    handle: &RoutedKernels,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, NativePort)>,
    residual: &WorkflowTensor,
    rows: u64,
    hidden: u64,
    shape: &ExpertGeometry,
    epsilon: f32,
) -> Result<WorkflowTensor, String> {
    let norm = weight(graph, load, scope, WeightKind::FeedForwardNorm, weights)?;
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
    let mut routes = graph.local_for(&handle.route, "routes", &choices).map_err(failed)?;
    let mut scores = graph.local_for(&handle.route, "scores", &choices).map_err(failed)?;
    let routed = graph
        .enqueue(
            &handle.route,
            qwen_routed_route::WorkflowArgs {
                residual: residual.into(),
                norm: (&norm).into(),
                router: (&router).into(),
                shared_router: (&shared_router).into(),
                routes: routes.tensor_mut().into(),
                scores: scores.tensor_mut().into(),
                eps: epsilon,
                normalize: i32::from(shape.normalize_selected),
            },
        )
        .map_err(failed)?;
    let (normalized, coefficient) = (routed.r0, routed.r1);

    if rows <= DECODE_ROWS {
        let expanded = graph
            .enqueue(
                &handle.expand,
                qwen_routed_expand::WorkflowArgs {
                    normalized: (&normalized).into(),
                    routes: routes.tensor().into(),
                    expert_gate: (&expert_gate).into(),
                    expert_up: (&expert_up).into(),
                    shared_gate: (&shared_gate).into(),
                    shared_up: (&shared_up).into(),
                },
            )
            .map_err(failed)?;
        return Ok(graph
            .enqueue(
                &handle.output,
                qwen_routed_output::WorkflowArgs {
                    residual: residual.into(),
                    expert_product: (&expanded.r0).into(),
                    shared_product: (&expanded.r1).into(),
                    routes: routes.tensor().into(),
                    scores: scores.tensor().into(),
                    coefficient: (&coefficient).into(),
                    expert_down: (&expert_down).into(),
                    shared_down: (&shared_down).into(),
                },
            )
            .map_err(failed)?
            .value);
    }

    let blocks = grouped_blocks(rows, shape.count, shape.selected)?;
    let tables = [
        ("M", rows),
        ("E", shape.count),
        ("K", shape.selected),
        ("B", blocks),
        ("T", TILE_ROWS),
    ];
    let mut counts = graph.local_for(&handle.group, "counts", &tables).map_err(failed)?;
    let mut order = graph.local_for(&handle.group, "order", &tables).map_err(failed)?;
    let mut inverse = graph.local_for(&handle.group, "inverse", &tables).map_err(failed)?;
    let mut block_experts = graph.local_for(&handle.group, "blocks", &tables).map_err(failed)?;
    graph
        .enqueue(
            &handle.group,
            qwen_routed_group::WorkflowArgs {
                routes: routes.tensor().into(),
                counts: counts.tensor_mut().into(),
                order: order.tensor_mut().into(),
                inverse: inverse.tensor_mut().into(),
                blocks: block_experts.tensor_mut().into(),
            },
        )
        .map_err(failed)?;
    let experts = graph
        .enqueue(
            &handle.experts,
            qwen_routed_experts::WorkflowArgs {
                normalized: (&normalized).into(),
                order: order.tensor().into(),
                blocks: block_experts.tensor().into(),
                expert_gate: (&expert_gate).into(),
                expert_up: (&expert_up).into(),
                expert_down: (&expert_down).into(),
            },
        )
        .map_err(failed)?
        .value;
    Ok(graph
        .enqueue(
            &handle.combine,
            qwen_routed_combine::WorkflowArgs {
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
        )
        .map_err(failed)?
        .value)
}
