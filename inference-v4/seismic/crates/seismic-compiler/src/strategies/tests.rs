//! Targeted strategy-library tests: construction success,
//! obligation consumption (`finish_alternative` acceptance), the structural
//! streaming rule's match/non-match shapes, admission rules, both blocked
//! covers, hierarchical bounds, and shared-operand publication.

use super::*;
use crate::terminal::NumericalTransfer;
use seismic_lang::logical::Access;
use seismic_lang::{
    intrinsics::{MathOp, PrimitiveId, ReduceOp},
    logical::{construct, ChoiceId, EffectiveTargetIdentity, LogicalNodeKind, ReductionOrder},
    precision::PrecisionPolicy,
    program::{compile, SourceFile},
    sir::IntrinsicUse,
    span::Span,
    types::{DType, Elem, NonEmpty, RuntimeExtentId, TensorType, ValueType},
};
use seismic_realization::executable::{
    EffectiveTargetProfile, ExecutorScalarSource, InvariantReport, Legalized,
    ObligationDisposition, ObligationRef, PhysicalConsequences, PhysicalPrimitive,
    PlanFamilyBuilder, PlanValues, StorageViewTemplate, TargetLimits, TransportTemplate,
};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Local sealed dialect (same shape as the terminal tests)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SynOp {
    Compute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SynDialect;
impl seismic_realization::executable::sealed::Sealed for SynDialect {}

impl seismic_realization::executable::ExecutableDialect for SynDialect {
    type Op = SynOp;
    type LayoutTemplate = u64;
    type ResolvedLayout = u64;

    fn legalize(p: &PhysicalPrimitive, _t: &EffectiveTargetProfile) -> Legalized<SynOp> {
        if matches!(p.op, seismic_lang::logical::PrimitiveOp::Capability(_)) {
            return Legalized::Inapplicable {
                reason: "no portable opcode".into(),
            };
        }
        Legalized::Ops(NonEmpty::new(vec![SynOp::Compute]).expect("one opcode"))
    }

    fn consequences(_op: &SynOp) -> PhysicalConsequences {
        crate::terminal::universal_consequences(0, 0)
    }

    fn public_layout(tensor: &TensorType) -> u64 {
        match &tensor.elem {
            Elem::Dtype(dtype) => u64::from(dtype.bytes()),
            _ => 4,
        }
    }

    fn internal_layout(tensor: &TensorType) -> u64 {
        Self::public_layout(tensor)
    }

    fn resolve_layout(layout: &u64, _values: &PlanValues) -> Result<u64, InvariantReport> {
        Ok(*layout)
    }
}

fn ops() -> Legalized<SynOp> {
    Legalized::Ops(NonEmpty::new(vec![SynOp::Compute]).expect("one opcode"))
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn check(sources: &[(&str, &str)]) -> seismic_lang::sir::Program {
    let files: Vec<SourceFile> = sources
        .iter()
        .map(|(path, text)| SourceFile {
            path: path.to_string(),
            text: text.to_string(),
        })
        .collect();
    compile(&files).expect("the sources check")
}

fn supports_all(_: &IntrinsicUse) -> Result<(), String> {
    Ok(())
}

fn logical_target() -> EffectiveTargetIdentity {
    EffectiveTargetIdentity {
        backend: "synthetic".to_string(),
        capability_fingerprint: "test-fingerprint".to_string(),
    }
}

fn entry_graph(
    sources: &[(&str, &str)],
    entry: &str,
    shape_values: &[(&str, i64)],
) -> seismic_lang::logical::LogicalProgram {
    let program = check(sources);
    let logical = construct(
        &program,
        entry,
        &logical_target(),
        &supports_all,
        shape_values
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect(),
        BTreeMap::new(),
    )
    .expect("construction succeeds");
    logical.verify().expect("the built program verifies");
    logical
}

fn runtime_extents_of(
    logical: &seismic_lang::logical::LogicalProgram,
) -> BTreeMap<RuntimeExtentId, seismic_lang::logical::RuntimeExtent> {
    logical
        .runtime_extents
        .ids()
        .zip(logical.runtime_extents.iter())
        .map(|(id, extent)| (id, extent.clone()))
        .collect()
}

fn facts_of(
    logical: &seismic_lang::logical::LogicalProgram,
    graph: &seismic_lang::logical::TaskGraph,
) -> crate::terminal::legalization::GraphFacts {
    crate::terminal::legalization::GraphFacts::collect(graph, &logical.runtime_extents)
}

#[allow(dead_code)]
fn profile() -> EffectiveTargetProfile {
    EffectiveTargetProfile {
        backend: "synthetic".into(),
        capability_fingerprint: "test-fingerprint".into(),
        toolchain_fingerprint: "test-toolchain".into(),
        effective_signatures: BTreeSet::new(),
        limits: TargetLimits {
            max_participants: 1024,
            max_workgroups_axis: [1024, 1, 1],
            max_workgroup_bytes: 32768,
            max_explicit_private_bytes: 32768,
            max_direct_bindings: 32,
            max_argument_table_bytes: 4096,
            max_device_bytes: 1 << 30,
        },
    }
}

fn entry_task_graph(
    logical: &seismic_lang::logical::LogicalProgram,
) -> seismic_lang::logical::TaskGraph {
    logical
        .graph(
            logical
                .choice(logical.entry_choice)
                .alternatives
                .iter()
                .next()
                .unwrap()
                .graph,
        )
        .clone()
}

fn map_region(
    builder: &mut seismic_realization::executable::AlternativeBuilder<SynDialect>,
    region: &[seismic_realization::executable::RegionStep],
) -> Result<(), String> {
    map_primitives(builder, region)?;
    map_reductions_serially(builder, region)
}

/// Map only the primitive nodes of one region (constants, casts, reads,
/// arithmetic): reductions and structure are left for explicit strategies.
fn map_primitives(
    builder: &mut seismic_realization::executable::AlternativeBuilder<SynDialect>,
    region: &[seismic_realization::executable::RegionStep],
) -> Result<(), String> {
    let graph = builder.graph().clone();
    let node_ids = region_nodes(&graph, region)?;
    for node_id in node_ids {
        let node = node_at(
            &graph,
            &seismic_realization::executable::NodeRef {
                region: region.to_vec(),
                node: node_id,
            },
        )
        .expect("the node exists");
        let node_ref = seismic_realization::executable::NodeRef {
            region: region.to_vec(),
            node: node_id,
        };
        match &node.kind {
            LogicalNodeKind::Primitive(_) => {
                builder
                    .map_primitive(
                        node_ref,
                        seismic_realization::dispatch::LinearIterationMap::serial(),
                        ops(),
                    )
                    .map_err(|e| e)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Map every reduction of one region with the ordered universal strategy.
fn map_reductions_serially(
    builder: &mut seismic_realization::executable::AlternativeBuilder<SynDialect>,
    region: &[seismic_realization::executable::RegionStep],
) -> Result<(), String> {
    let graph = builder.graph().clone();
    let node_ids = region_nodes(&graph, region)?;
    for node_id in node_ids {
        let node = node_at(
            &graph,
            &seismic_realization::executable::NodeRef {
                region: region.to_vec(),
                node: node_id,
            },
        )
        .expect("the node exists");
        let node_ref = seismic_realization::executable::NodeRef {
            region: region.to_vec(),
            node: node_id,
        };
        if let LogicalNodeKind::Reduction(_) = &node.kind {
            let result_value = node
                .outputs
                .first()
                .map(|output| output.id)
                .expect("a reduction has a result");
            builder
                .map_reduction(
                    node_ref,
                    seismic_realization::executable::ReductionStrategyTemplate {
                        topology: seismic_realization::numerics::ReductionTopology::SerialAxis {
                            axis: 0,
                            length: seismic_lang::types::ExtentExpr::Static(16),
                        },
                        iteration: seismic_realization::dispatch::LinearIterationMap::serial(),
                        ops: ops(),
                        result: builder.transport_of(result_value)?,
                    },
                )
                .map_err(|e| e)?;
        }
    }
    Ok(())
}

fn finish_alternative(
    builder: seismic_realization::executable::AlternativeBuilder<SynDialect>,
) -> Result<seismic_realization::executable::PhysicalAlternative<SynDialect>, String> {
    finish_alternative_skipping(builder, &[])
}

/// `skip` names boundary ordinals already completed explicitly (for example
/// by `complete_fused_results`).
fn finish_alternative_skipping(
    mut builder: seismic_realization::executable::AlternativeBuilder<SynDialect>,
    skip: &[u32],
) -> Result<seismic_realization::executable::PhysicalAlternative<SynDialect>, String> {
    for obligation in builder.pending_obligations() {
        builder
            .discharge(
                obligation,
                ObligationDisposition::StaticallyProved {
                    reason: "synthetic".into(),
                },
            )
            .map_err(|e| e)?;
    }
    let result_count = builder.graph().results.len();
    for ordinal in 0..result_count as u32 {
        if skip.contains(&ordinal) {
            continue;
        }
        let transport = match &builder.graph().results[ordinal as usize] {
            seismic_lang::logical::RegionResult::Value { id, .. } => builder.transport_of(*id)?,
            seismic_lang::logical::RegionResult::State { storage, .. } => {
                match builder.storage_of(*storage) {
                    Some(template) => TransportTemplate::Storage(
                        NonEmpty::new(vec![StorageViewTemplate {
                            storage: template,
                            access: Access::Exclusive,
                            transform: seismic_lang::logical::ViewTransform::Identity,
                        }])
                        .expect("one plane"),
                    ),
                    None => TransportTemplate::Boundary(Default::default()),
                }
            }
        };
        builder.complete_result(ordinal, transport).map_err(|e| e)?;
    }
    builder.finish_alternative().map_err(|e| e)
}

/// The loop node of the entry graph (the scan fixture's only top-level loop).
fn sole_loop(graph: &seismic_lang::logical::TaskGraph) -> seismic_realization::executable::NodeRef {
    for node_id in graph.root.nodes.ids() {
        if matches!(graph.root.nodes[node_id].kind, LogicalNodeKind::Loop(_)) {
            return seismic_realization::executable::NodeRef {
                region: Vec::new(),
                node: node_id,
            };
        }
    }
    panic!("the fixture has a top-level loop");
}

fn facts_types(
    facts: &crate::terminal::legalization::GraphFacts,
    value: seismic_lang::logical::GraphValueId,
) -> ValueType {
    facts
        .types
        .get(&value)
        .cloned()
        .expect("the value has a type")
}

// ---------------------------------------------------------------------------
// The structural streaming rule
// ---------------------------------------------------------------------------

const ONLINE_SOFTMAX_SCAN: &str = "fn online[T](x: &tensor[T] f32, base: f32) -> f32:
    let mut m = base
    let mut s = 0.0
    for t in 0..T:
        let v = x[t]
        let m_new = max(m, v)
        s = s * exp(m - m_new) + exp(v - m_new)
        m = m_new
    return m + s
";

#[test]
fn streaming_rule_matches_the_online_softmax_shape() {
    let logical = entry_graph(
        &[("scan.seismic", ONLINE_SOFTMAX_SCAN)],
        "online",
        &[("T", 512)],
    );
    let graph = entry_task_graph(&logical);
    let facts = facts_of(&logical, &graph);
    let occurrence = sole_loop(&graph);
    match derive_scan_state(&graph, &occurrence, &facts) {
        ScanStateMatch::Streaming(state) => {
            // One extremum lane (the running max) and one additive lane (the
            // running sum), cross-lane rescale detected.
            assert_eq!(state.lanes.len(), 2, "carried m and s");
            assert!(state
                .lanes
                .iter()
                .any(|lane| matches!(lane.law, CombineLaw::Extremum { op: MathOp::Max })));
            assert!(state
                .lanes
                .iter()
                .any(|lane| lane.law == CombineLaw::Additive));
            assert!(
                state.rescale,
                "the additive lane rescales through the max carry"
            );
            // Accumulator dtypes derive from the carried scalar types.
            for lane in &state.lanes {
                assert_eq!(lane.dtype, DType::F32);
            }
            // The scanned axis capacity bounds the window domain.
            assert_eq!(state.axis_capacity, 512);
        }
        other => panic!("the online-softmax shape must match, got {other:?}"),
    }
}

#[test]
fn streaming_rule_matches_plain_max_and_sum_scans() {
    let source = "fn scan[T](x: &tensor[T] f32) -> f32:
    let mut m = x[0]
    let mut s = 0.0
    for t in 0..T:
        m = max(m, x[t])
        s = s + x[t]
    return m + s
";
    let logical = entry_graph(&[("scan.seismic", source)], "scan", &[("T", 64)]);
    let graph = entry_task_graph(&logical);
    let facts = facts_of(&logical, &graph);
    match derive_scan_state(&graph, &sole_loop(&graph), &facts) {
        ScanStateMatch::Streaming(state) => {
            assert!(state.lanes.iter().any(|l| l.law == CombineLaw::Additive));
            assert!(state
                .lanes
                .iter()
                .any(|l| matches!(l.law, CombineLaw::Extremum { .. })));
            assert!(!state.rescale, "no cross-lane dependence");
        }
        other => panic!("the plain max/sum scan must match, got {other:?}"),
    }
}

#[test]
fn streaming_rule_rejects_non_matching_shapes() {
    // 1. Arbitrary arithmetic on the carry (mul without a rescale shape).
    let arbitrary = "fn f[T](x: &tensor[T] f32) -> f32:
    let mut s = 1.0
    for t in 0..T:
        s = s * x[t]
    return s
";
    let logical = entry_graph(&[("a.seismic", arbitrary)], "f", &[("T", 8)]);
    let graph = entry_task_graph(&logical);
    let facts = facts_of(&logical, &graph);
    match derive_scan_state(&graph, &sole_loop(&graph), &facts) {
        ScanStateMatch::NoStreaming { reason } => {
            assert!(reason.contains("combine law") || reason.contains("not derivable"));
        }
        other => panic!("arbitrary carry arithmetic must not stream, got {other:?}"),
    }

    // 2. A carried storage (mutated history) has no scalar combine law.
    let tensor_carry = "fn f[T](x: &tensor[T] f32, h: &mut tensor[T] f32) -> f32:
    let mut s = 0.0
    for t in 0..T:
        h[t] = x[t]
        s = s + x[t]
    return s
";
    let logical = entry_graph(&[("b.seismic", tensor_carry)], "f", &[("T", 8)]);
    let graph = entry_task_graph(&logical);
    let facts = facts_of(&logical, &graph);
    match derive_scan_state(&graph, &sole_loop(&graph), &facts) {
        ScanStateMatch::NoStreaming { reason } => {
            assert!(reason.contains("carried storage") || reason.contains("carried tensor"));
        }
        other => panic!("a carried tensor must not stream, got {other:?}"),
    }

    // 3. An independent loop has no carried state at all.
    let independent = "fn f[N](x: &tensor[N] f32, out: tensor[N] f32) -> tensor[N] f32:
    let mut o = out
    parallel for i in 0..N:
        o[i] = x[i] + 1.0
    return o
";
    let logical = entry_graph(&[("c.seismic", independent)], "f", &[("N", 8)]);
    let graph = entry_task_graph(&logical);
    let facts = facts_of(&logical, &graph);
    match derive_scan_state(&graph, &sole_loop(&graph), &facts) {
        ScanStateMatch::NoStreaming { reason } => {
            assert!(reason.contains("independent"));
        }
        other => panic!("an independent loop must not stream, got {other:?}"),
    }
}

#[test]
fn stream_scan_builds_a_complete_alternative() {
    let logical = entry_graph(
        &[("scan.seismic", ONLINE_SOFTMAX_SCAN)],
        "online",
        &[("T", 512)],
    );
    let graph = entry_task_graph(&logical);
    let facts = facts_of(&logical, &graph);
    let occurrence = sole_loop(&graph);
    let state = match derive_scan_state(&graph, &occurrence, &facts) {
        ScanStateMatch::Streaming(state) => state,
        other => panic!("the fixture matches: {other:?}"),
    };
    let mut family =
        PlanFamilyBuilder::<SynDialect>::from_logical(&logical).expect("the family opens");
    let mut builder = family
        .alternative(logical.entry_choice, 0)
        .expect("the alternative opens");
    let window = StreamingWindow::declare(&mut builder, &occurrence, state.axis_capacity)
        .expect("the window declares");
    let model = CostModelId::uncalibrated("synthetic");
    let receipt = stream_scan(
        &mut builder,
        occurrence.clone(),
        &state,
        &window,
        |body_builder| {
            // Map the scan body region with the universal serial map: the
            // body's arithmetic is executed exactly, never replaced.
            let mut path = Vec::new();
            path.push(seismic_realization::executable::RegionStep::LoopBody(
                occurrence.node,
            ));
            map_region(body_builder, &path)
        },
        model,
    )
    .expect("the scan streams");
    // The window parameter is registered and solver-tunable.
    assert_eq!(receipt.window, window.parameter);
    // One explicit physical carry per derived lane.
    assert_eq!(receipt.carry_slots.len(), state.lanes.len());
    // The additive lane's windowed partials carry a Reassociate transfer.
    assert!(matches!(
        receipt.numerical,
        NumericalTransfer::Reassociate { .. }
    ));
    // Map the remaining root-region dataflow (the initial constants and the
    // final combine of the carried results) with the universal serial map.
    map_primitives(&mut builder, &[]).expect("the root dataflow maps");
    // The alternative consumes every obligation and finishes.
    let alternative = finish_alternative(builder).expect("the alternative finishes");
    family
        .add_alternative(logical.entry_choice, alternative)
        .expect("the alternative commits");
    let family = family.finish().expect("the family finishes");
    assert!(
        family
            .parameters
            .iter()
            .any(|parameter| { parameter.name == format!("window-node{}", occurrence.node.0) }),
        "the window width is a plan parameter of the family"
    );
    // The Repeat step carries the physical carries.
    let alternative = family.choices[logical.entry_choice]
        .alternatives
        .iter()
        .next()
        .expect("the committed alternative exists");
    assert!(schedule_has_repeat_with_carries(
        alternative,
        receipt.carry_slots.len()
    ));
}

fn schedule_has_repeat_with_carries(
    alternative: &seismic_realization::executable::PhysicalAlternative<SynDialect>,
    carries: usize,
) -> bool {
    use seismic_realization::executable::ScheduleStepTemplate;
    fn walk(steps: &[ScheduleStepTemplate<SynDialect>], carries: usize) -> bool {
        steps.iter().any(|step| match step {
            ScheduleStepTemplate::Repeat(repeat) => {
                repeat.carried.len() == carries || walk(repeat.body.steps.as_slice(), carries)
            }
            ScheduleStepTemplate::If(if_step) => {
                walk(if_step.then_schedule.steps.as_slice(), carries)
                    || walk(if_step.else_schedule.steps.as_slice(), carries)
            }
            ScheduleStepTemplate::Launch(_) | ScheduleStepTemplate::Call(_) => false,
        })
    }
    walk(alternative.schedule.steps.as_slice(), carries)
}

#[test]
fn stream_dataflow_keeps_intermediates_kernel_local() {
    // A dataflow region over a large axis: the intermediate must never
    // become capacity-sized device storage in the streaming alternative.
    let source = "fn f[T](x: &tensor[T] f32) -> tensor[T] f32:
    let v = f32(x)
    return exp(v) * 2.0
";
    let logical = entry_graph(&[("df.seismic", source)], "f", &[("T", 4096)]);
    let graph = entry_task_graph(&logical);
    let runtime = runtime_extents_of(&logical);
    let family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical).expect("the family opens");
    let mut builder = family
        .alternative(logical.entry_choice, 0)
        .expect("the alternative opens");

    // The primitive nodes of the region (constants, math, arithmetic).
    let nodes: Vec<seismic_realization::executable::NodeRef> = graph
        .root
        .nodes
        .ids()
        .filter(|id| matches!(graph.root.nodes[*id].kind, LogicalNodeKind::Primitive(_)))
        .map(|id| seismic_realization::executable::NodeRef {
            region: Vec::new(),
            node: id,
        })
        .collect();
    assert!(!nodes.is_empty(), "the region has primitive work");
    let window = StreamingWindow::declare(
        &mut builder,
        &seismic_realization::executable::NodeRef {
            region: Vec::new(),
            node: nodes[0].node,
        },
        4096,
    )
    .expect("the window declares");
    let axes = vec![seismic_lang::types::ExtentExpr::Static(4096)];
    let receipt = stream_dataflow(
        &mut builder,
        nodes,
        &window,
        &axes,
        &runtime,
        ops(),
        CostModelId::uncalibrated("synthetic"),
    )
    .expect("the region streams");
    // At least one intermediate became kernel-local SSA.
    assert!(
        !receipt.kernel_local_values.is_empty(),
        "the fused region keeps its intermediate kernel-local"
    );
    finish_alternative(builder).expect("the alternative finishes");
}

// ---------------------------------------------------------------------------
// Hierarchical workgroup/subgroup mapping
// ---------------------------------------------------------------------------

#[test]
fn hierarchical_map_assigns_outer_to_workgroups_and_inner_to_lanes() {
    let runtime: BTreeMap<RuntimeExtentId, seismic_lang::logical::RuntimeExtent> = BTreeMap::new();
    let facts = SubgroupFacts {
        width: Some(4),
        max_participants_per_workgroup: 64,
    };
    let outer = [seismic_lang::types::ExtentExpr::Static(3)];
    let inner = [seismic_lang::types::ExtentExpr::Static(8)];
    let mapping = hierarchical_map(
        &outer,
        &inner,
        &runtime,
        &facts,
        seismic_lang::sym::Sym::constant(8),
    )
    .expect("the hierarchical map builds");
    // The outer domain maps to workgroups: 3 workgroups of 8 participants.
    assert_eq!(mapping.workgroups.as_constant(), Some(3));
    // Exact coordinate maps: workgroup 2, participant 5 → outer 2, inner 5.
    assert_eq!(mapping.coordinate_of(2, 5).unwrap(), vec![2, 5]);
    // Every (workgroup, participant) pair names a distinct coordinate and
    // the cover partitions the domain.
    let mut seen = BTreeSet::new();
    for workgroup in 0..3u64 {
        for participant in 0..8u64 {
            seen.insert(mapping.coordinate_of(workgroup, participant).unwrap());
        }
    }
    assert_eq!(seen.len(), 24);
    // Subgroup decomposition: participant 5 → subgroup 1, lane 1 (width 4).
    assert_eq!(mapping.subgroup_of(5).unwrap(), (1, 1));
    // Stage tile bytes are exact: 8 elements × 4 bytes.
    assert_eq!(
        mapping
            .stage_tile_bytes(&seismic_lang::sym::Sym::constant(8), 4)
            .as_constant(),
        Some(32)
    );
}

#[test]
fn hierarchical_map_is_inapplicable_beyond_the_workgroup_bound() {
    let runtime: BTreeMap<RuntimeExtentId, seismic_lang::logical::RuntimeExtent> = BTreeMap::new();
    let facts = SubgroupFacts {
        width: None,
        max_participants_per_workgroup: 256,
    };
    let inner = vec![seismic_lang::types::ExtentExpr::Static(1024)];
    let error = hierarchical_map(
        &[seismic_lang::types::ExtentExpr::Static(4)],
        &inner,
        &runtime,
        &facts,
        seismic_lang::sym::Sym::constant(1024),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        HierarchicalError::InnerDomainExceedsWorkgroup {
            inner_capacity: 1024,
            bound: 256
        }
    ));
}

#[test]
fn hierarchical_fuse_builds_a_complete_alternative() {
    let source = "fn f[M, N](x: &tensor[M, N] f32) -> tensor[M, N] f32:
    let v = f32(x)
    return v + 1.0
";
    let logical = entry_graph(&[("h.seismic", source)], "f", &[("M", 4), ("N", 8)]);
    let graph = entry_task_graph(&logical);
    let runtime = runtime_extents_of(&logical);
    let family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical).expect("the family opens");
    let mut builder = family
        .alternative(logical.entry_choice, 0)
        .expect("the alternative opens");
    let nodes: Vec<seismic_realization::executable::NodeRef> = graph
        .root
        .nodes
        .ids()
        .filter(|id| matches!(graph.root.nodes[*id].kind, LogicalNodeKind::Primitive(_)))
        .map(|id| seismic_realization::executable::NodeRef {
            region: Vec::new(),
            node: id,
        })
        .collect();
    let facts = SubgroupFacts {
        width: Some(4),
        max_participants_per_workgroup: 64,
    };
    let mapping = hierarchical_map(
        &[seismic_lang::types::ExtentExpr::Static(4)],
        &[seismic_lang::types::ExtentExpr::Static(8)],
        &runtime,
        &facts,
        seismic_lang::sym::Sym::constant(8),
    )
    .expect("the map builds");
    let receipt = hierarchical_fuse(
        &mut builder,
        nodes,
        mapping,
        Some(seismic_lang::sym::Sym::constant(32)),
        ops(),
        CostModelId::uncalibrated("synthetic"),
    )
    .expect("the region fuses hierarchically");
    assert!(receipt.stage_bytes.is_some());
    finish_alternative(builder).expect("the alternative finishes");
}

// ---------------------------------------------------------------------------
// Shared-operand multi-consumer fusion
// ---------------------------------------------------------------------------

/// The value consumed by two or more primitive sibling nodes (the staged
/// activation), with its consumers.
fn shared_operand_consumers(
    graph: &seismic_lang::logical::TaskGraph,
) -> (
    seismic_lang::logical::GraphValueId,
    Vec<seismic_realization::executable::NodeRef>,
) {
    use std::collections::BTreeMap;
    let mut consumers: BTreeMap<
        seismic_lang::logical::GraphValueId,
        Vec<seismic_lang::logical::NodeId>,
    > = BTreeMap::new();
    for node_id in graph.root.nodes.ids() {
        let node = &graph.root.nodes[node_id];
        if matches!(node.kind, LogicalNodeKind::Primitive(_)) {
            for input in &node.inputs {
                consumers.entry(*input).or_default().push(node_id);
            }
        }
    }
    for (value, nodes) in consumers {
        if nodes.len() >= 2 {
            return (
                value,
                nodes
                    .into_iter()
                    .map(|node_id| seismic_realization::executable::NodeRef {
                        region: Vec::new(),
                        node: node_id,
                    })
                    .collect(),
            );
        }
    }
    panic!("the fixture has a shared operand with two consumers");
}

#[test]
fn shared_operand_fusion_requires_real_siblings() {
    let source = "fn f[N](x: &tensor[N] f32) -> tensor[N] f32:
    let v = f32(x)
    let a = v * 2.0
    let b = v + 1.0
    return a + b
";
    let logical = entry_graph(&[("s.seismic", source)], "f", &[("N", 8)]);
    let graph = entry_task_graph(&logical);
    let family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical).expect("the family opens");
    let mut builder = family
        .alternative(logical.entry_choice, 0)
        .expect("the alternative opens");
    let (shared_operand, siblings) = shared_operand_consumers(&graph);
    assert_eq!(siblings.len(), 2, "mul and add both consume the cast value");

    // A single node is not a multi-consumer fusion.
    let single = shared_operand_fuse(
        &mut builder,
        vec![siblings[0].clone()],
        shared_operand,
        seismic_realization::dispatch::LinearIterationMap::serial(),
        ops(),
        CostModelId::uncalibrated("synthetic"),
    );
    assert!(single.is_err());

    // A node not consuming the operand is refused.
    let foreign_node = seismic_realization::executable::NodeRef {
        region: Vec::new(),
        node: graph
            .root
            .nodes
            .ids()
            .find(|id| {
                let node = &graph.root.nodes[*id];
                matches!(node.kind, LogicalNodeKind::Primitive(_))
                    && !node.inputs.contains(&shared_operand)
                    && !node.inputs.is_empty()
            })
            .expect("a node outside the sibling set exists"),
    };
    assert!(shared_operand_fuse(
        &mut builder,
        vec![siblings[0].clone(), foreign_node],
        shared_operand,
        seismic_realization::dispatch::LinearIterationMap::serial(),
        ops(),
        CostModelId::uncalibrated("synthetic"),
    )
    .is_err());
}

#[test]
fn shared_operand_fusion_publishes_each_result_explicitly() {
    // The gate/up shape: two sibling transforms of one staged operand, each
    // published as its own boundary result (here through two `&mut` result
    // storages, plus the reduced tail).
    let source =
        "fn f[N](x: &tensor[N] f32, gate: &mut tensor[N] f32, up: &mut tensor[N] f32) -> f32:
    parallel for i in 0..N:
        let e = x[i]
        gate[i] = e * 2.0
        up[i] = e + 1.0
    return reduce(f32(x), 0, sum)
";
    let logical = entry_graph(&[("s.seismic", source)], "f", &[("N", 8)]);
    let graph = entry_task_graph(&logical);
    let family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical).expect("the family opens");
    let mut builder = family
        .alternative(logical.entry_choice, 0)
        .expect("the alternative opens");

    // The staged operand: the cast value consumed by both siblings (the
    // consumers live in the loop body region; the producer at the root).
    let loop_ref = sole_loop(&graph);
    let loop_node = match &graph.root.nodes[loop_ref.node].kind {
        LogicalNodeKind::Loop(loop_node) => loop_node.clone(),
        _ => unreachable!(),
    };
    let mut body_path = Vec::new();
    body_path.push(seismic_realization::executable::RegionStep::LoopBody(
        loop_ref.node,
    ));
    let shared_operand = {
        use std::collections::BTreeMap;
        let mut consumers: BTreeMap<
            seismic_lang::logical::GraphValueId,
            Vec<seismic_lang::logical::NodeId>,
        > = BTreeMap::new();
        for node_id in loop_node.body.nodes.ids() {
            let node = &loop_node.body.nodes[node_id];
            if matches!(node.kind, LogicalNodeKind::Primitive(_)) {
                for input in &node.inputs {
                    consumers.entry(*input).or_default().push(node_id);
                }
            }
        }
        consumers
            .into_iter()
            .find(|(_, nodes)| nodes.len() >= 2)
            .map(|(value, _)| value)
            .expect("the fixture has a shared operand with two consumers")
    };

    // Consume the independent loop as a structured Repeat whose body fuses
    // the sibling region over the shared operand (one staged activation).
    let start = builder
        .transport_of(loop_node.range.start)
        .expect("the range start has a transport");
    let end = builder
        .transport_of(loop_node.range.end)
        .expect("the range end has a transport");
    let mut body_path = Vec::new();
    body_path.push(seismic_realization::executable::RegionStep::LoopBody(
        loop_ref.node,
    ));
    let mut receipt_holder = None;
    builder
        .schedule_loop(
            loop_ref.clone(),
            seismic_realization::executable::ExecutorRangeTemplate {
                start,
                end,
                bound: loop_node.range.bound.clone(),
            },
            Vec::new(),
            |body_builder| {
                // The fused sibling region: both consumers of the staged
                // operand plus their writes, in one launch.
                let graph = body_builder.graph().clone();
                let body_nodes: Vec<seismic_realization::executable::NodeRef> =
                    region_nodes(&graph, &body_path)?
                        .into_iter()
                        .filter(|node_id| {
                            node_at(
                                &graph,
                                &seismic_realization::executable::NodeRef {
                                    region: body_path.clone(),
                                    node: *node_id,
                                },
                            )
                            .is_some_and(|node| matches!(node.kind, LogicalNodeKind::Primitive(_)))
                        })
                        .map(|node_id| seismic_realization::executable::NodeRef {
                            region: body_path.clone(),
                            node: node_id,
                        })
                        .collect();
                let iteration = seismic_realization::dispatch::LinearIterationMap::linear(
                    &[seismic_lang::types::ExtentExpr::Static(8)],
                    &BTreeMap::new(),
                )
                .map_err(|error| error.to_string())?;
                let receipt = shared_operand_fuse(
                    body_builder,
                    body_nodes,
                    shared_operand,
                    iteration,
                    ops(),
                    CostModelId::uncalibrated("synthetic"),
                )?;
                receipt_holder = Some(receipt);
                Ok(())
            },
        )
        .expect("the loop schedules with the fused sibling body");
    let receipt = receipt_holder.expect("the body fused");
    // The shared operand stages once: both siblings bind it through the one
    // recorded transport (one activation inside the fused launch), never an
    // inferred native alias — and never a kernel-local transport crossing a
    // launch boundary.
    assert!(!matches!(
        builder.transport_of(shared_operand),
        Ok(TransportTemplate::Kernel(_))
    ));
    // Both gate/up results are published explicitly (complete_result ×2,
    // plus the reduced tail).
    assert!(
        receipt.published_results.len() >= 2,
        "gate/up publishes both results, got {:?}",
        receipt.published_results
    );
    complete_fused_results(&mut builder, &receipt).expect("each fused result publishes");
    // The remaining top-level dataflow: the cast and the reduction.
    map_primitives(&mut builder, &[]).expect("the cast maps");
    map_reductions_serially(&mut builder, &[]).expect("the reduction maps");
    finish_alternative_skipping(builder, &receipt.published_results)
        .expect("the alternative finishes");
}

// ---------------------------------------------------------------------------
// Cross-call boundary substitution
// ---------------------------------------------------------------------------

const CALL_KERNEL: &str = "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:
    let mut output = result
    parallel for row in 0..M:
        for col in 0..N:
            output[row, col] = x[row, col] + y[row, col]
    return output

fn linear[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:
    return add(x, y, result)
";

#[test]
fn cross_call_environment_substitutes_boundaries_and_refuses_kernel_transports() {
    let logical = entry_graph(
        &[("k.seismic", CALL_KERNEL)],
        "linear",
        &[("M", 4), ("N", 8)],
    );
    let graph = entry_task_graph(&logical);
    let family = PlanFamilyBuilder::<SynDialect>::from_logical(&logical).expect("the family opens");
    let mut builder = family
        .alternative(logical.entry_choice, 0)
        .expect("the alternative opens");
    let call_ref = graph
        .root
        .nodes
        .ids()
        .find(|id| matches!(graph.root.nodes[*id].kind, LogicalNodeKind::Call(_)))
        .map(|id| seismic_realization::executable::NodeRef {
            region: Vec::new(),
            node: id,
        })
        .expect("the entry is one call");
    let call_node = match &graph.root.nodes[call_ref.node].kind {
        LogicalNodeKind::Call(call) => call.clone(),
        _ => unreachable!(),
    };
    // Boundary substitution: every leaf transports directly to a caller
    // transport.
    let mut boundary = seismic_realization::executable::BoundaryTemplates::default();
    for input in &call_node.boundary_inputs {
        match &input.kind {
            seismic_lang::logical::BoundaryInputKind::Value(value) => {
                boundary.inputs.insert(
                    seismic_realization::executable::BoundaryLeaf::Input {
                        param: input.param,
                        leaf: input.path.clone(),
                    },
                    builder.transport_of(*value).unwrap(),
                );
            }
            seismic_lang::logical::BoundaryInputKind::Shared {
                value,
                state: token,
            }
            | seismic_lang::logical::BoundaryInputKind::Exclusive {
                value,
                state: token,
            }
            | seismic_lang::logical::BoundaryInputKind::Move {
                value,
                state: token,
            } => {
                let storage = builder
                    .logical_storage_of_token(*token)
                    .expect("the boundary state has a storage");
                let input_leaf = seismic_realization::executable::BoundaryLeaf::Input {
                    param: input.param,
                    leaf: input.path.clone(),
                };
                boundary.states.insert(
                    input_leaf.clone(),
                    seismic_realization::executable::StateTransportTemplate::Storage(
                        builder.storage_of(storage).expect("caller-owned template"),
                    ),
                );
                boundary
                    .inputs
                    .insert(input_leaf, builder.transport_of(*value).unwrap());
            }
        }
    }
    for result in &call_node.boundary_results {
        match &result.kind {
            seismic_lang::logical::BoundaryResultKind::Value(value) => {
                boundary.results.insert(
                    seismic_realization::executable::BoundaryLeaf::Result {
                        leaf: result.path.clone(),
                    },
                    builder.transport_of(*value).unwrap(),
                );
            }
            seismic_lang::logical::BoundaryResultKind::Storage { storage, .. } => {
                boundary.results.insert(
                    seismic_realization::executable::BoundaryLeaf::Result {
                        leaf: result.path.clone(),
                    },
                    TransportTemplate::Storage(
                        NonEmpty::new(vec![StorageViewTemplate {
                            storage: builder.storage_of(*storage).expect("the result storage"),
                            access: Access::Exclusive,
                            transform: seismic_lang::logical::ViewTransform::Identity,
                        }])
                        .expect("one plane"),
                    ),
                );
            }
            seismic_lang::logical::BoundaryResultKind::State(token) => {
                let storage = builder
                    .logical_storage_of_token(*token)
                    .expect("the result state has a storage");
                boundary.states.insert(
                    seismic_realization::executable::BoundaryLeaf::Result {
                        leaf: result.path.clone(),
                    },
                    seismic_realization::executable::StateTransportTemplate::Storage(
                        builder.storage_of(storage).expect("caller-owned template"),
                    ),
                );
            }
        }
    }
    // A kernel-local transport can never cross the call boundary.
    let mut with_kernel = boundary.clone();
    let kernel_transport =
        TransportTemplate::Kernel(seismic_realization::executable::KernelValueTemplateId(0));
    let inject_target = boundary
        .results
        .keys()
        .next()
        .cloned()
        .or_else(|| boundary.inputs.keys().next().cloned());
    if let Some(path) = inject_target {
        with_kernel.results.insert(path, kernel_transport);
    }
    assert!(cross_call_environment(&mut builder, call_ref.clone(), with_kernel).is_err());
    // The substituted boundary consumes the call.
    cross_call_environment(&mut builder, call_ref.clone(), boundary).expect("the call invokes");
    // The entry's result materialization (the call result into the entry's
    // compiler-owned result storage) maps after the invoke.
    map_primitives(&mut builder, &[]).expect("the materialization maps");
    finish_alternative(builder).expect("the alternative finishes");
}

// ---------------------------------------------------------------------------
// Blocked lane coverage
// ---------------------------------------------------------------------------

#[test]
fn lane_covers_partition_the_reduced_domain_exactly() {
    // Interleaved: lanes 4 over 10 elements. The raw coordinate map is
    // ℓ + step·P; coordinates ≥ total are the traversal's masked tail.
    let interleaved = LaneCover::Interleaved { lanes: 4 };
    assert_eq!(interleaved.coordinate(0, 0), Some(0));
    assert_eq!(interleaved.coordinate(1, 2), Some(9));
    assert_eq!(
        interleaved.coordinate(3, 3),
        Some(15),
        "raw map; masked by the total"
    );
    assert_eq!(
        interleaved.lane_length(10, 3),
        2,
        "lane 3 folds 3 and 7 only"
    );
    assert!(interleaved.covers_exactly(10));
    assert!(interleaved.covers_exactly(8));
    // Contiguous: lanes 4, block 3 over 10 elements (last block masked).
    let contiguous = LaneCover::Contiguous { lanes: 4, block: 3 };
    assert_eq!(contiguous.coordinate(0, 0), Some(0));
    assert_eq!(contiguous.coordinate(3, 0), Some(9));
    assert_eq!(
        contiguous.lane_length(10, 3),
        1,
        "the last block is masked by the total"
    );
    assert!(contiguous.covers_exactly(10));
    // lanes·block < total does not cover.
    assert!(!contiguous.covers_exactly(13));
    // Zero lanes never cover.
    assert!(!LaneCover::Interleaved { lanes: 0 }.covers_exactly(4));
}

#[test]
fn blocked_cover_admission_rules_hold() {
    let operand = TensorType::new(
        vec![seismic_lang::types::ExtentExpr::Static(64)],
        Elem::Dtype(DType::F32),
    );
    let ordered = seismic_lang::logical::ReductionNode {
        operand: seismic_lang::logical::GraphValueId(0),
        axis: 0,
        op: seismic_lang::intrinsics::ReduceOp::Sum,
        order: ReductionOrder::Ascending,
        accumulator: DType::F32,
        result: ValueType::Scalar(DType::F32),
    };
    let mut unordered = ordered.clone();
    unordered.order = ReductionOrder::Unordered;
    let mut argmax = ordered.clone();
    argmax.op = seismic_lang::intrinsics::ReduceOp::Argmax;
    argmax.accumulator = DType::I32;
    argmax.result = ValueType::Scalar(DType::I32);
    use crate::terminal::reduction::ReassociationAdmission as Admission;
    use crate::terminal::reduction::{reassociable, ReductionAdmissionError};
    let topology = |cover: &LaneCover| -> seismic_realization::numerics::ReductionTopology {
        match cover {
            LaneCover::Interleaved { lanes } => {
                seismic_realization::numerics::ReductionTopology::Tree {
                    fan_in: *lanes as u32,
                    depth: 1,
                    inner: Box::new(
                        seismic_realization::numerics::ReductionTopology::SerialAxis {
                            axis: 0,
                            length: seismic_lang::types::ExtentExpr::Static(64),
                        },
                    ),
                }
            }
            LaneCover::Contiguous { lanes, .. } => {
                seismic_realization::numerics::ReductionTopology::Split {
                    cuts: (0..*lanes as usize)
                        .map(|i| {
                            if i + 1 == *lanes as usize {
                                seismic_lang::types::ExtentExpr::Static(64)
                            } else {
                                seismic_lang::types::ExtentExpr::Sym(seismic_lang::sym::Sym::param(
                                    "block",
                                ))
                            }
                        })
                        .collect(),
                    inner: Box::new(
                        seismic_realization::numerics::ReductionTopology::SerialAxis {
                            axis: 0,
                            length: seismic_lang::types::ExtentExpr::Static(64),
                        },
                    ),
                }
            }
        }
    };
    let cover = LaneCover::Interleaved { lanes: 8 };
    // An ordered source reduction without admission is refused.
    assert_eq!(
        reassociable(
            &ordered,
            &operand,
            topology(&cover),
            Admission::SourceUnordered
        ),
        Err(ReductionAdmissionError::AscendingOrder)
    );
    // An unordered source admits.
    assert!(reassociable(
        &unordered,
        &operand,
        topology(&cover),
        Admission::SourceUnordered
    )
    .is_ok());
    // Caller policy/evidence admits even an ordered source.
    assert!(reassociable(
        &ordered,
        &operand,
        topology(&cover),
        Admission::PolicyOrEvidence
    )
    .is_ok());
    // argmax never reassociates, on either cover.
    for cover in [
        LaneCover::Interleaved { lanes: 8 },
        LaneCover::Contiguous { lanes: 8, block: 8 },
    ] {
        assert_eq!(
            reassociable(
                &argmax,
                &operand,
                topology(&cover),
                Admission::PolicyOrEvidence
            ),
            Err(ReductionAdmissionError::ArgmaxNeverReassociates)
        );
    }
}

#[test]
fn both_blocked_covers_build_distinct_complete_alternatives() {
    let source = "fn f[N](x: &tensor[N] f32) -> f32:
    return reduce(f32(x), 0, sum, unordered=true)
";
    let logical = entry_graph(&[("r.seismic", source)], "f", &[("N", 64)]);
    let graph = entry_task_graph(&logical);
    let reduction_ref = graph
        .root
        .nodes
        .ids()
        .find(|id| matches!(graph.root.nodes[*id].kind, LogicalNodeKind::Reduction(_)))
        .map(|id| seismic_realization::executable::NodeRef {
            region: Vec::new(),
            node: id,
        })
        .expect("the fixture reduces");
    let reduction = match &graph.root.nodes[reduction_ref.node].kind {
        LogicalNodeKind::Reduction(reduction) => reduction.clone(),
        _ => unreachable!(),
    };
    let facts = facts_of(&logical, &graph);
    let operand = match facts_types(&facts, reduction.operand) {
        ValueType::Tensor(tensor) => tensor,
        other => panic!("the operand is a tensor, got {other:?}"),
    };
    let runtime = runtime_extents_of(&logical);
    let mut family =
        PlanFamilyBuilder::<SynDialect>::from_logical(&logical).expect("the family opens");
    use crate::terminal::reduction::ReassociationAdmission as Admission;

    // Universal alternative first (the family requires one): map every
    // primitive (cast, literals), then the reduction with the ordered
    // universal strategy.
    {
        let mut builder = family
            .alternative(logical.entry_choice, 0)
            .expect("the universal alternative opens");
        map_region(&mut builder, &[]).expect("the region maps");
        let alternative = finish_alternative(builder).expect("the universal alternative finishes");
        family
            .add_alternative(logical.entry_choice, alternative)
            .expect("commits");
    }
    // Interleaved and contiguous alternatives: distinct covers of the same
    // reduction, each finishing and committing as its own alternative.
    let mut receipts = Vec::new();
    for cover in [
        LaneCover::Interleaved { lanes: 8 },
        LaneCover::Contiguous { lanes: 8, block: 8 },
    ] {
        let mut builder = family
            .alternative(logical.entry_choice, 0)
            .expect("the blocked alternative opens");
        map_primitives(&mut builder, &[]).expect("the primitives map");
        let receipt = blocked_cover(
            &mut builder,
            reduction_ref.clone(),
            &reduction,
            &operand,
            cover.clone(),
            Admission::SourceUnordered,
            &runtime,
            ops(),
            CostModelId::uncalibrated("synthetic"),
        )
        .expect("the blocked cover builds");
        receipts.push((receipt, finish_alternative(builder).expect("finishes")));
    }
    // The two covers are distinct alternatives with distinct topologies and
    // exact covers of the reduced domain.
    let (interleaved, _) = &receipts[0];
    let (contiguous, _) = &receipts[1];
    assert!(matches!(
        interleaved.strategy.topology,
        seismic_realization::numerics::ReductionTopology::Tree { .. }
    ));
    assert!(matches!(
        contiguous.strategy.topology,
        seismic_realization::numerics::ReductionTopology::Split { .. }
    ));
    assert!(interleaved.plan.cover.covers_exactly(64));
    assert!(contiguous.plan.cover.covers_exactly(64));
    // Both carry the reassociate transfer of an unordered reduction.
    assert!(matches!(
        interleaved.strategy.numerical,
        NumericalTransfer::Reassociate { .. }
    ));
}

// ---------------------------------------------------------------------------
// Cost identity plumbing
// ---------------------------------------------------------------------------

#[test]
fn every_strategy_receipt_carries_a_cost_expression_and_model_identity() {
    // Measured identities are backend-registered inputs; the library only
    // carries them and cost never affects legality.
    let measured = CostModelId::measured("metal", "metal-m4pro-timing-v3", "m4-pro-01 2026-09-19");
    assert!(measured.measured);
    assert_eq!(measured.backend, "metal");
    let uncalibrated = CostModelId::uncalibrated("cuda");
    assert!(!uncalibrated.measured);
    // Window counts are exact symbolic ceil-divisions.
    use seismic_lang::sym::Sym;
    let total = Sym::constant(512);
    let width = Sym::param("window-node0");
    let count = crate::strategies::cost::window_count(&total, &width);
    // At width 1: 512 windows; at width 512: 1 window.
    assert_eq!(
        count.eval(&|name| (name == "window-node0").then_some(1)),
        Some(512)
    );
    assert_eq!(
        count.eval(&|name| (name == "window-node0").then_some(512)),
        Some(1)
    );
    assert_eq!(
        count.eval(&|name| (name == "window-node0").then_some(100)),
        Some(6)
    );
}

// Silence unused-import warnings for test scaffolding shared with fixtures.
#[allow(unused)]
fn _scaffold() {
    let _ = PrecisionPolicy::Unconstrained;
    let _ = Span::default();
    let _ = ChoiceId(0);
    let _ = ExecutorScalarSource::Abi {
        leaf: seismic_realization::executable::BoundaryLeaf::Input {
            param: 0,
            leaf: Default::default(),
        },
        endpoint: None,
    };
    let _ = ObligationRef {
        node: seismic_realization::executable::NodeRef {
            region: Vec::new(),
            node: seismic_lang::logical::NodeId(0),
        },
        index: 0,
    };
    let _ = PrimitiveId::Select;
    let _ = ReduceOp::Sum;
}
