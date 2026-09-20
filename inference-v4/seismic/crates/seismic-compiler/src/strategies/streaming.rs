//! Runtime-axis streaming and the
//! structural online-softmax/scan admission rule.
//!
//! Two constructors:
//!
//! - [`stream_dataflow`]: a tensor dataflow region (chain of primitive nodes)
//!   over a runtime extent is fused into one launch whose physical linear
//!   participant count is the *window* `PlanParameter`. Every intermediate
//!   value of the region becomes kernel-local SSA inside the launch; the
//!   runtime axis is traversed grid-stride with a tail mask, so the last
//!   window is masked by the runtime value. The region's intermediates never
//!   become capacity-sized device storage, and never an unmodeled native
//!   thread array: all storage is `ConditionalStorageTemplate` (the
//!   device-materialized universal strategy remains a peer alternative when
//!   memory fits).
//! - [`stream_scan`]: an ordered scan loop over a runtime extent is consumed
//!   by `schedule_loop`, and its reducer/scan state becomes an explicit
//!   physical carry (`PhysicalCarryTemplate` over planned executor-scalar
//!   slots).
//!
//! ## The online streaming rule
//!
//! Streaming a scan is legal ONLY when the exact carried state and its
//! combination law can be *derived from the logical graph structure* — the
//! reduction semantics plus the carried state types — using registry
//! semantics. [`derive_scan_state`] pattern-matches the logical shape
//! structurally; it never recognizes function names and never silently
//! replaces arbitrary user arithmetic with a hand-coded kernel. When the rule
//! does not match, no streaming alternative exists: the explicit
//! device-materialized universal strategy still compiles the program subject
//! to real device memory.
//!
//! Matched shapes and their derived laws:
//!
//! - an **extremum lane** — a carried scalar folded by registry `max`/`min`
//!   over element reads — combines by the extremum law. Any visiting order
//!   yields the registry result, so the lane's transfer is `Exact`.
//! - an **additive lane** — a carried scalar folded by registry `add` —
//!   combines window partials with the carry. Windowed partial sums
//!   reassociate the reference ascending fold, so the lane carries a
//!   `Reassociate` transfer (source `unordered=true`, caller policy, or
//!   evidence must admit it — the same admission as any reassociating
//!   reduction, enforced by the reduction vocabulary this strategy builds
//!   on).
//! - a **rescale pair** — an additive lane whose chain rescales through
//!   another lane's extremum carry via `exp` (the online-softmax `m`/`l`
//!   shape) — carries the additive `Reassociate` transfer; the countable
//!   rescale roundings are subsumed by it under numerical-transfer
//!   composition.
//!
//! Anything else — carried tensors, capability operations, `argmax` folds,
//! arithmetic the registry does not admit as a combine law, independent
//! loops (which have no data carries) — is [`ScanStateMatch::NoStreaming`].

use crate::strategies::cost::{window_count, CostModelId};
use crate::strategies::node_at;
use crate::terminal::reduction::{reduction_identity, ReductionIdentity};
use seismic_lang::{
    intrinsics::{MathOp, PrimitiveId, ReduceOp},
    logical::{
        GraphRegion, GraphValueId, LogicalNode, LogicalNodeKind, LoopNode, PrimitiveOp,
        ReductionNode, RegionInput, RuntimeExtent,
    },
    sym::Sym,
    types::{DType, ExtentExpr, RuntimeExtentId, ValueType},
};
use seismic_realization::dispatch::LinearIterationMap;
use seismic_realization::executable::{
    AlternativeBuilder, BuilderError, ExecutableDialect, ExecutorScalarSlotId,
    ExecutorScalarSource, ExecutorScalarTemplate, FusedStrategyTemplate, Legalized, NodeRef,
    PhysicalCarryTemplate, PlanParamId, TransportTemplate,
};
use seismic_realization::numerics::{NumericalTransfer, ReductionTopology};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// The window parameter
// ---------------------------------------------------------------------------

/// One declared fixed-capacity window over a runtime extent: the window
/// width is a solver-tunable `PlanParameter` in `1..=capacity`. The retained
/// runtime value masks the last window (the traversal's tail mask); the
/// capacity is only the resource/tuning bound.
#[derive(Clone, Debug)]
pub struct StreamingWindow {
    pub parameter: PlanParamId,
    /// The planning symbol of the window width.
    pub symbol: Sym,
    /// The checked capacity bound of the streamed axis.
    pub capacity: u64,
}

impl StreamingWindow {
    /// Declare a fresh window parameter for one occurrence. The name embeds
    /// the node id so two occurrences never collide (family-wide uniqueness
    /// is enforced by the builder).
    pub fn declare<D: ExecutableDialect>(
        builder: &mut AlternativeBuilder<D>,
        occurrence: &NodeRef,
        capacity: u64,
    ) -> Result<Self, BuilderError> {
        let name = format!("window-node{}", occurrence.node.0);
        let upper = i64::try_from(capacity).map_err(|_| {
            "the runtime extent capacity exceeds the plan parameter domain".to_string()
        })?;
        let parameter = builder.plan_parameter(&name, 1, upper)?;
        Ok(StreamingWindow {
            parameter,
            symbol: Sym::param(&name),
            capacity,
        })
    }

    /// Symbolic window count covering this window's capacity bound:
    /// `ceil(capacity / width)`.
    pub fn count_symbol(&self) -> Sym {
        window_count(&Sym::constant(self.capacity as i64), &self.symbol)
    }
}

// ---------------------------------------------------------------------------
// The structural scan-state rule
// ---------------------------------------------------------------------------

/// The combination law of one derived carried accumulator, taken from
/// registry semantics only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineLaw {
    /// Registry `max`/`min`: order-insensitive under the registry NaN rule.
    Extremum { op: MathOp },
    /// Registry `add` over window partials: reassociates the ascending fold.
    Additive,
}

/// One derived carried accumulator lane.
#[derive(Clone, Debug, PartialEq)]
pub struct CarriedLane {
    /// Ordinal of the carried slot in the loop's carried order.
    pub slot: usize,
    /// The accumulator dtype (the carried value's scalar dtype).
    pub dtype: DType,
    pub law: CombineLaw,
    /// The registry identity the lane's fold starts from.
    pub identity: ReductionIdentity,
}

/// The derived scan state of a matched loop.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanState {
    pub lanes: Vec<CarriedLane>,
    /// An additive lane rescales through an extremum lane's carry (the
    /// online-softmax `m`/`l` shape).
    pub rescale: bool,
    /// The checked capacity bound of the scanned axis (the runtime extent's
    /// capacity for runtime axes, the exact length for static axes). The
    /// window parameter domain is `1..=axis_capacity`; the retained runtime
    /// value masks the last window.
    pub axis_capacity: u64,
}

/// Result of the structural online-streaming rule.
#[derive(Clone, Debug, PartialEq)]
pub enum ScanStateMatch {
    /// The rule matched; `stream_scan` may construct the streaming
    /// alternative with exactly this carried state.
    Streaming(ScanState),
    /// The rule did not match. No streaming alternative exists for this
    /// occurrence; the device-materialized universal strategy remains.
    NoStreaming { reason: String },
}

/// Apply the structural online-streaming rule to one loop occurrence. This
/// is pure logical-graph analysis: no names, no target facts, no arithmetic
/// replacement. See the module documentation for the matched shapes.
pub fn derive_scan_state(
    graph: &seismic_lang::logical::TaskGraph,
    occurrence: &NodeRef,
    facts: &crate::terminal::legalization::GraphFacts,
) -> ScanStateMatch {
    let Some(node) = node_at(graph, occurrence) else {
        return no("the region path names no node");
    };
    let LogicalNodeKind::Loop(loop_node) = &node.kind else {
        return no("streaming applies to loop occurrences");
    };
    derive_from_loop(loop_node, facts)
}

fn no(reason: impl Into<String>) -> ScanStateMatch {
    ScanStateMatch::NoStreaming {
        reason: reason.into(),
    }
}

fn derive_from_loop(
    loop_node: &LoopNode,
    facts: &crate::terminal::legalization::GraphFacts,
) -> ScanStateMatch {
    use seismic_lang::sir::LoopKind;
    // Independent loops carry no data at all: there is no scan state to
    // stream, and their joins are structural (disjoint/atomic), not
    // accumulator combines.
    if loop_node.kind != LoopKind::Ordered {
        return no("an independent loop has no carried scan state");
    }
    // The scanned axis must carry a checked bound: a runtime extent's
    // capacity (the retained runtime value masks the last window) or an
    // exact static length. Streaming exists to bound that axis's history.
    let axis_capacity = match &loop_node.range.bound {
        ExtentExpr::Static(n) => Some(*n),
        ExtentExpr::Runtime(id) => facts.runtime_extents.get(id).map(|extent| extent.capacity),
        ExtentExpr::Sym(sym) => sym
            .as_constant()
            .and_then(|value| u64::try_from(value).ok()),
    };
    let Some(axis_capacity) = axis_capacity else {
        return no("the scan range has no checked bound");
    };
    if axis_capacity == 0 {
        return no("an empty scan range has no scan state");
    }
    if loop_node.carried.is_empty() {
        return no("the loop carries no scan state");
    }
    let body = &loop_node.body;
    // The carried body parameter values: each carry's own parameter, by slot.
    let mut carry_params: Vec<Option<GraphValueId>> = Vec::with_capacity(loop_node.carried.len());
    for slot in &loop_node.carried {
        match slot.initial {
            RegionInput::Value(_) => {
                let parameter = body.parameters.get(slot.body_parameter.index()).cloned();
                carry_params.push(parameter.and_then(|p| match p {
                    seismic_lang::logical::RegionParameter::Value { id, .. } => Some(id),
                    _ => None,
                }));
            }
            RegionInput::State(_) => {
                // A carried storage (tensor accumulator) has no registry
                // scalar combine law: refuse.
                return no("a carried storage has no derived combine law");
            }
        }
    }
    let all_carry_params: BTreeSet<GraphValueId> = carry_params.iter().flatten().copied().collect();

    let mut lanes = Vec::new();
    let mut rescale = false;
    for (ordinal, slot) in loop_node.carried.iter().enumerate() {
        let RegionInput::Value(initial) = slot.initial else {
            return no("a carried state has no derived combine law");
        };
        let ty = facts
            .types
            .get(&initial)
            .cloned()
            .or_else(|| body_parameter_type(body, slot.body_parameter.index()));
        let Some(ValueType::Scalar(dtype)) = ty else {
            return no("a carried tensor or aggregate has no derived scalar combine law");
        };
        let Some(result_value) = body_result_value(body, slot.body_result.index()) else {
            return no("a carried slot has no body result value");
        };
        let own_param = carry_params[ordinal];
        let analysis = match analyze_lane(body, result_value, own_param, &all_carry_params) {
            Ok(analysis) => analysis,
            Err(match_failure) => return match_failure,
        };
        match analysis {
            LaneAnalysis::Extremum { op } => {
                lanes.push(CarriedLane {
                    slot: ordinal,
                    dtype,
                    law: CombineLaw::Extremum { op },
                    identity: ReductionIdentity::FirstElement,
                });
            }
            LaneAnalysis::Additive {
                cross_lane,
                exp_present,
            } => {
                if cross_lane && !exp_present {
                    return no(
                        "an additive lane depends on another carry without a registry rescale \
                         (exp) shape; the arithmetic is not derivable",
                    );
                }
                if cross_lane {
                    rescale = true;
                }
                lanes.push(CarriedLane {
                    slot: ordinal,
                    dtype,
                    law: CombineLaw::Additive,
                    identity: ReductionIdentity::Zero,
                });
            }
            LaneAnalysis::Foreign => {
                return no(
                    "the carried lane combines operations the registry does not admit as a \
                     combine law; arbitrary user arithmetic is never silently streamed",
                );
            }
        }
    }
    if rescale {
        // The rescale pair requires exactly one extremum lane providing the
        // running scale and additive lanes consuming it.
        if !lanes
            .iter()
            .any(|lane| matches!(lane.law, CombineLaw::Extremum { .. }))
        {
            return no("a rescaling additive lane has no extremum carry to rescale through");
        }
    }
    ScanStateMatch::Streaming(ScanState {
        lanes,
        rescale,
        axis_capacity,
    })
}

fn body_parameter_type(body: &GraphRegion, index: usize) -> Option<ValueType> {
    match body.parameters.get(index) {
        Some(seismic_lang::logical::RegionParameter::Value { ty, .. }) => Some(ty.clone()),
        _ => None,
    }
}

fn body_result_value(body: &GraphRegion, ordinal: usize) -> Option<GraphValueId> {
    match body.results.get(ordinal) {
        Some(seismic_lang::logical::RegionResult::Value { id, .. }) => Some(*id),
        _ => None,
    }
}

/// How one carried lane's body result is produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LaneAnalysis {
    /// Folded by registry `max`/`min` over admitted leaves.
    Extremum { op: MathOp },
    /// Folded by registry `add`; `cross_lane` marks dependence on another
    /// carry's parameter (rescale shape, legal only with `exp`).
    Additive { cross_lane: bool, exp_present: bool },
    /// Anything else: not derivable.
    Foreign,
}

/// The terminal combine of one lane: the first node on the carry path from
/// the lane's result down to its carry parameter. Everything off that path
/// is operand/context arithmetic (executed exactly by the mapped body —
/// never replaced), constrained to shapes whose combination law the
/// registry derives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Combine {
    Extremum(MathOp),
    Additive,
}

/// Walk the body region's value graph backwards from one lane's result
/// producer, classifying the fold. The **carry path** is the chain of nodes
/// from the result producer down to the lane's carry parameter; its first
/// node is the terminal combine, and every node strictly between it and the
/// carry must be a `mul`/`sub` rescale step consuming the carry (the
/// `l · exp(m_old − m_new)` shape). See the module documentation for the
/// matched shapes and their derived laws.
fn analyze_lane(
    body: &GraphRegion,
    result: GraphValueId,
    own_param: Option<GraphValueId>,
    all_carry_params: &BTreeSet<GraphValueId>,
) -> Result<LaneAnalysis, ScanStateMatch> {
    use std::collections::VecDeque;
    // Producer of each value in this region (parameters excluded).
    let mut producer: BTreeMap<GraphValueId, &LogicalNode> = BTreeMap::new();
    for node in body.nodes.iter() {
        for output in &node.outputs {
            producer.insert(output.id, node);
        }
    }
    let Some(own) = own_param else {
        return Ok(LaneAnalysis::Foreign);
    };

    // Backward breadth-first walk from the result, with parent tracking so
    // the carry path can be reconstructed on first discovery of the carry.
    // Every visited node is also classified as operand/context.
    let mut prev: BTreeMap<GraphValueId, GraphValueId> = BTreeMap::new();
    let mut visited: BTreeSet<GraphValueId> = BTreeSet::from([result]);
    let mut queue: VecDeque<GraphValueId> = VecDeque::from([result]);
    let mut cross_lane = false;
    let mut exp_present = false;
    let mut foreign = false;
    while let Some(value) = queue.pop_front() {
        if value == own {
            continue;
        }
        if all_carry_params.contains(&value) {
            cross_lane = true;
            continue;
        }
        let Some(node) = producer.get(&value).cloned() else {
            // A region parameter that is not a carry (an invariant read):
            // an admitted leaf.
            continue;
        };
        match &node.kind {
            LogicalNodeKind::Primitive(application) => match &application.op {
                PrimitiveOp::Constant(_) | PrimitiveOp::RuntimeExtent(_) => {}
                PrimitiveOp::Capability(_) => foreign = true,
                PrimitiveOp::Primitive(id) => match id {
                    PrimitiveId::ElementRead { .. } | PrimitiveId::Cast(_) => {}
                    PrimitiveId::Math(op) => {
                        if matches!(op, MathOp::Exp) {
                            exp_present = true;
                        }
                    }
                    PrimitiveId::Binary(_) => {}
                    // Tensor-producing/aggregate primitives are not scalar
                    // fold operands.
                    _ => foreign = true,
                },
            },
            LogicalNodeKind::Reduction(reduction) => match reduction.op {
                ReduceOp::Sum | ReduceOp::Max | ReduceOp::Min => {}
                ReduceOp::Argmax => {
                    return Err(no(
                        "an argmax fold never streams; smaller-coordinate ties must be \
                         preserved in the reference order",
                    ));
                }
            },
            // Control flow or calls inside a carried fold: not derivable.
            _ => foreign = true,
        }
        for input in node.inputs.iter().copied() {
            if visited.insert(input) {
                prev.insert(input, value);
                queue.push_back(input);
            }
        }
    }
    if foreign {
        return Ok(LaneAnalysis::Foreign);
    }
    if !visited.contains(&own) {
        // The result does not depend on the carry: a pass-through, not a fold.
        return Ok(LaneAnalysis::Foreign);
    }
    // Reconstruct the carry path: values from the carry up to the result;
    // the producing node of each upper value lies on the path.
    let mut chain = vec![own];
    while *chain.last().expect("the chain is nonempty") != result {
        let current = *chain.last().expect("the chain is nonempty");
        let Some(parent) = prev.get(&current).cloned() else {
            return Ok(LaneAnalysis::Foreign);
        };
        chain.push(parent);
    }
    chain.reverse();
    if chain.len() < 2 {
        return Ok(LaneAnalysis::Foreign);
    }
    let mut path_nodes: Vec<&LogicalNode> = Vec::with_capacity(chain.len() - 1);
    for window in chain.windows(2) {
        let [upper, _lower] = window else {
            unreachable!("windows of two")
        };
        match producer.get(upper) {
            Some(node) => path_nodes.push(node),
            None => return Ok(LaneAnalysis::Foreign),
        }
    }
    // The terminal combine must be an admitted registry fold.
    let combine = match path_nodes.first() {
        Some(node) => match &node.kind {
            LogicalNodeKind::Primitive(application) => match &application.op {
                PrimitiveOp::Primitive(PrimitiveId::Math(op @ (MathOp::Max | MathOp::Min))) => {
                    Combine::Extremum(*op)
                }
                PrimitiveOp::Primitive(PrimitiveId::Binary(
                    seismic_lang::syntax::ast::BinaryOp::Add,
                )) => Combine::Additive,
                _ => return Ok(LaneAnalysis::Foreign),
            },
            LogicalNodeKind::Reduction(reduction) => match reduction.op {
                ReduceOp::Sum => Combine::Additive,
                ReduceOp::Max => Combine::Extremum(MathOp::Max),
                ReduceOp::Min => Combine::Extremum(MathOp::Min),
                ReduceOp::Argmax => return Ok(LaneAnalysis::Foreign),
            },
            _ => return Ok(LaneAnalysis::Foreign),
        },
        None => return Ok(LaneAnalysis::Foreign),
    };
    // Every node strictly between the combine and the carry must be a
    // mul/sub rescale step that itself consumes the carry; an extremum
    // combine takes the carry directly.
    let mut rescale_step_with_own = false;
    for node in &path_nodes[1..] {
        let admitted = matches!(
            &node.kind,
            LogicalNodeKind::Primitive(application)
                if matches!(
                    &application.op,
                    PrimitiveOp::Primitive(PrimitiveId::Binary(
                        seismic_lang::syntax::ast::BinaryOp::Mul
                            | seismic_lang::syntax::ast::BinaryOp::Sub
                    ))
                ) && node.inputs.contains(&own)
        );
        if !admitted {
            return Ok(LaneAnalysis::Foreign);
        }
        rescale_step_with_own = true;
    }
    match combine {
        Combine::Extremum(op) => {
            if rescale_step_with_own {
                return Ok(LaneAnalysis::Foreign);
            }
            Ok(LaneAnalysis::Extremum { op })
        }
        Combine::Additive => {
            if rescale_step_with_own && !(cross_lane && exp_present) {
                return Ok(LaneAnalysis::Foreign);
            }
            Ok(LaneAnalysis::Additive {
                cross_lane,
                exp_present,
            })
        }
    }
}

/// Receipt of one streamed dataflow region.
#[derive(Clone, Debug)]
pub struct StreamingReceipt {
    pub window: PlanParamId,
    pub window_symbol: Sym,
    /// Values of the region whose transports became kernel-local SSA (the
    /// capacity-sized intermediates that never became device storage).
    pub kernel_local_values: Vec<GraphValueId>,
    pub cost: Sym,
    pub model: CostModelId,
}

/// Stream one tensor dataflow region over a runtime axis: fuse the region's
/// primitive nodes into a single launch whose participant count is the
/// window parameter, traversing the runtime axis grid-stride with the tail
/// mask (the last window is masked by the runtime value). All purely internal
/// intermediates become kernel-local SSA; nothing realizes the runtime-axis
/// history as unmodeled native storage.
///
/// Inapplicable when: a node is not a pending primitive, the region's total
/// overflows, or the window domain exceeds the axis capacity (the caller
/// then simply does not add this alternative; the universal strategy
/// remains).
pub fn stream_dataflow<D: ExecutableDialect>(
    builder: &mut AlternativeBuilder<D>,
    nodes: Vec<NodeRef>,
    window: &StreamingWindow,
    axes: &[ExtentExpr],
    runtime_extents: &BTreeMap<RuntimeExtentId, RuntimeExtent>,
    ops: Legalized<D::Op>,
    model: CostModelId,
) -> Result<StreamingReceipt, BuilderError> {
    if nodes.is_empty() {
        return Err("stream_dataflow requires a nonempty region".into());
    }
    let iteration = LinearIterationMap::linear(axes, runtime_extents)
        .map_err(|error| format!("the streamed domain is infeasible: {error}"))?
        .with_participants(window.symbol.clone());
    let region_outputs: Vec<GraphValueId> = nodes
        .iter()
        .filter_map(|node| crate::strategies::node_at(builder.graph(), node))
        .flat_map(|logical| {
            logical
                .outputs
                .into_iter()
                .map(|output| output.id)
                .collect::<Vec<_>>()
        })
        .collect();
    builder.fuse(nodes, FusedStrategyTemplate { iteration, ops })?;
    // After the fuse, every purely internal edge was rewritten to a kernel
    // transport; report which values actually became kernel-local.
    let kernel_local = region_outputs
        .into_iter()
        .filter(|value| {
            matches!(
                builder.transport_of(*value),
                Ok(TransportTemplate::Kernel(_))
            )
        })
        .collect();
    let total = Sym::constant(
        i64::try_from(window.capacity).map_err(|_| "capacity exceeds i64".to_string())?,
    );
    Ok(StreamingReceipt {
        window: window.parameter,
        window_symbol: window.symbol.clone(),
        kernel_local_values: kernel_local,
        // Elementwise work is conserved; the window enters through launch
        // pressure and occupancy, expressed by the family's cost model.
        cost: total.scale(1),
        model,
    })
}

// ---------------------------------------------------------------------------
// Constructor: stream a scan loop with explicit physical carries
// ---------------------------------------------------------------------------

/// Receipt of one streamed scan.
#[derive(Clone, Debug)]
pub struct StreamingScanReceipt {
    pub window: PlanParamId,
    pub window_symbol: Sym,
    /// The declared executor-scalar slots carrying the scan state, one per
    /// derived lane, in carried order.
    pub carry_slots: Vec<ExecutorScalarSlotId>,
    /// The composed numerical transfer of the windowed scan (see the module
    /// documentation; additive lanes carry `Reassociate`, extremum-only
    /// scans are `Exact`, rescale roundings are subsumed).
    pub numerical: NumericalTransfer,
    pub cost: Sym,
    pub model: CostModelId,
}

/// Consume one ordered scan loop over a runtime extent as a structured
/// executor `Repeat` whose carried state is explicit physical carries
/// (planned executor-scalar slots rebound each visit), with the window
/// width as a solver-tunable `PlanParameter` bounding each visit's
/// participant domain.
///
/// `state` must come from [`derive_scan_state`] matching this occurrence;
/// `body` maps the loop's body region (built with this same builder inside
/// the closure, exactly like `schedule_loop`).
///
/// Numerical policy: an additive lane's windowed partials reassociate the
/// reference ascending fold — the composed transfer is
/// `Reassociate { op: Sum, topology: Split }`, requiring source
/// `unordered=true`, caller policy, or evidence, exactly like every other
/// reassociating strategy. Extremum-only scans stay `Exact`.
pub fn stream_scan<D: ExecutableDialect>(
    builder: &mut AlternativeBuilder<D>,
    occurrence: NodeRef,
    state: &ScanState,
    window: &StreamingWindow,
    body: impl FnOnce(&mut AlternativeBuilder<D>) -> Result<(), BuilderError>,
    model: CostModelId,
) -> Result<StreamingScanReceipt, BuilderError> {
    let node = crate::strategies::node_at(builder.graph(), &occurrence)
        .ok_or_else(|| "the region path names no node".to_string())?;
    let LogicalNodeKind::Loop(loop_node) = &node.kind else {
        return Err("stream_scan requires a loop occurrence".into());
    };
    if loop_node.carried.len() != state.lanes.len() {
        return Err(
            "the derived scan state disagrees with the loop's carried slots (compiler bug)".into(),
        );
    }
    // The ascending retained range of the scan: exact start/end transports.
    let start = builder.transport_of(loop_node.range.start)?;
    let end = builder.transport_of(loop_node.range.end)?;
    // Explicit physical carries: one planned executor-scalar slot per lane.
    let mut carry_slots = Vec::with_capacity(state.lanes.len());
    let mut carries = Vec::with_capacity(state.lanes.len());
    for (lane, slot) in state.lanes.iter().zip(&loop_node.carried) {
        let RegionInput::Value(_) = slot.initial else {
            return Err("a derived scan lane must be a value carry".into());
        };
        let name = format!("scan-carry-node{}-slot{}", occurrence.node.0, lane.slot);
        let slot_id = builder.declare_executor_scalar_slot(lane.dtype, name);
        carry_slots.push(slot_id);
        carries.push(PhysicalCarryTemplate {
            transport: TransportTemplate::ExecutorScalar(ExecutorScalarTemplate {
                source: ExecutorScalarSource::Slot(slot_id),
                dtype: lane.dtype,
            }),
        });
    }
    builder.schedule_loop(
        occurrence.clone(),
        seismic_realization::executable::ExecutorRangeTemplate {
            start,
            end,
            bound: loop_node.range.bound.clone(),
        },
        carries,
        body,
    )?;
    // The composed transfer of the windowed scan.
    let mut numerical = NumericalTransfer::Exact;
    if state
        .lanes
        .iter()
        .any(|lane| lane.law == CombineLaw::Additive)
    {
        let topology = ReductionTopology::Split {
            cuts: vec![ExtentExpr::Sym(window.symbol.clone())],
            inner: Box::new(ReductionTopology::SerialAxis {
                axis: 0,
                length: ExtentExpr::Static(window.capacity.max(1)),
            }),
        };
        numerical = seismic_realization::numerics::compose(
            &NumericalTransfer::Reassociate {
                op: ReduceOp::Sum,
                topology,
            },
            &numerical,
        );
    }
    Ok(StreamingScanReceipt {
        window: window.parameter,
        window_symbol: window.symbol.clone(),
        carry_slots,
        numerical,
        cost: window
            .count_symbol()
            .scale(i64::try_from(state.lanes.len()).unwrap_or(1)),
        model,
    })
}

/// The registry identity of a reduce operator (re-exported for strategy
/// consumers assembling receipts).
pub fn lane_identity(op: ReduceOp) -> ReductionIdentity {
    reduction_identity(op)
}

/// The reduction node of a reduction occurrence, for strategy inspection.
pub fn reduction_of(
    graph: &seismic_lang::logical::TaskGraph,
    occurrence: &NodeRef,
) -> Option<ReductionNode> {
    node_at(graph, occurrence).and_then(|node| match node.kind {
        LogicalNodeKind::Reduction(reduction) => Some(reduction),
        _ => None,
    })
}
