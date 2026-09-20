//! Fusion strategies: shared-operand multi-consumer fusion and cross-call
//! fusion.
//!
//! ## Shared-operand multi-consumer fusion (family 3)
//!
//! Sibling consumers of one operand (the gate/up shape: two contractions
//! reading one staged activation) share that operand ONLY through a fused
//! alternative that consumes both consumers' graphs in one launch
//! ([`shared_operand_fuse`]) and explicitly publishes every result
//! ([`complete_fused_results`]). Sharing is never inferred by aliasing
//! native buffers after selection: within the fused launch the shared
//! operand is bound once per launch through its recorded transport, and the
//! siblings' purely internal edges become kernel-local SSA.
//!
//! ## Cross-call fusion (family 4)
//!
//! A caller/callee interval is fused only through boundary substitution and
//! consumption of the child graph obligations. The boundary-substitution
//! form constructible over the published alternative-builder surface is
//! [`cross_call_environment`]: every canonical boundary leaf transports
//! directly to a caller transport (caller storage views, planned
//! executor-scalar slots, or boundary placeholders) — the child allocates
//! nothing for its boundary — and `TransportTemplate::Kernel` can never
//! cross a schedule step or call boundary (enforced structurally here, in
//! addition to the realization layer's own rules). Fully mapping the child
//! interval into the caller's launch requires a cross-graph fuse surface on
//! the realization layer; until it exists, calls
//! remain nested plans and launch-count reduction stays explicit.

use crate::strategies::cost::CostModelId;
use crate::strategies::node_at;
use seismic_lang::logical::GraphValueId;
use seismic_realization::dispatch::LinearIterationMap;
use seismic_realization::executable::{
    AlternativeBuilder, BoundaryTemplates, BuilderError, ExecutableDialect, FusedStrategyTemplate,
    Legalized, NodeRef, StateTransportTemplate, TransportTemplate,
};

/// Receipt of one shared-operand fusion.
#[derive(Clone, Debug)]
pub struct FusionReceipt {
    /// The one shared operand, bound once per launch.
    pub shared_operand: GraphValueId,
    /// The fused sibling consumers, in the order supplied to `fuse`.
    pub fused_nodes: Vec<NodeRef>,
    /// Values whose transports became kernel-local SSA inside the launch
    /// (the shared intermediate activation).
    pub kernel_local_values: Vec<GraphValueId>,
    /// Boundary result ordinals whose values originate in the fused set:
    /// the assembler MUST `complete_result` each of them explicitly (two
    /// for the gate/up shape) — publication is never inferred.
    pub published_results: Vec<u32>,
    pub cost: seismic_lang::sym::Sym,
    pub model: CostModelId,
}

/// Fuse the sibling consumers of one shared operand into a single launch.
///
/// Validates structurally, before any transition: at least two nodes; every
/// node a pending primitive of this alternative; every node consuming the
/// shared operand. Then consumes them through the public `fuse` transition:
/// the shared operand's transport is the caller-recorded transport (one
/// staged activation inside the launch), and every purely internal edge of
/// the region becomes kernel-local SSA.
///
/// After the fuse, complete every boundary result explicitly with
/// [`complete_fused_results`].
pub fn shared_operand_fuse<D: ExecutableDialect>(
    builder: &mut AlternativeBuilder<D>,
    nodes: Vec<NodeRef>,
    shared_operand: GraphValueId,
    iteration: LinearIterationMap,
    ops: Legalized<D::Op>,
    model: CostModelId,
) -> Result<FusionReceipt, BuilderError> {
    if nodes.len() < 2 {
        return Err(
            "shared-operand fusion requires at least two sibling consumers of the operand".into(),
        );
    }
    let pending: std::collections::BTreeSet<NodeRef> =
        builder.pending_nodes().into_iter().collect();
    let mut region_outputs = Vec::new();
    let mut region_state_storages = Vec::new();
    let mut operand_consumers = 0usize;
    for node in &nodes {
        if !pending.contains(node) {
            return Err(format!(
                "node#{} is absent or already consumed",
                node.node.0
            ));
        }
        let logical = node_at(builder.graph(), node)
            .ok_or_else(|| "the region path names no node".to_string())?;
        if !matches!(
            logical.kind,
            seismic_lang::logical::LogicalNodeKind::Primitive(_)
        ) {
            return Err(format!(
                "node#{} is not a primitive of the fused region",
                node.node.0
            ));
        }
        if logical.inputs.contains(&shared_operand) {
            operand_consumers += 1;
        }
        region_outputs.extend(logical.outputs.iter().map(|output| output.id));
        region_state_storages.extend(logical.state_outputs.iter().map(|token| token.storage));
    }
    if operand_consumers < 2 {
        return Err(format!(
            "shared-operand fusion requires at least two consumers of value#{} in the fused \
             region (found {operand_consumers})",
            shared_operand.0
        ));
    }
    // The shared operand must already have a recorded transport (its producer
    // mapped or a boundary/storage value): fuse binds it per node through
    // exactly this transport — one staged activation, never an inferred
    // native alias.
    let shared_transport = builder.transport_of(shared_operand)?;
    if crosses_launch_boundary(&shared_transport) {
        return Err(
            "the shared operand's transport must be retained storage or an executor scalar".into(),
        );
    }
    builder.fuse(nodes.clone(), FusedStrategyTemplate { iteration, ops })?;
    let kernel_local = region_outputs
        .iter()
        .copied()
        .filter(|value| {
            matches!(
                builder.transport_of(*value),
                Ok(TransportTemplate::Kernel(_))
            )
        })
        .collect();
    // Every boundary result originating in the fused set must be published
    // explicitly by the assembler: value results whose producer is fused,
    // and state results of storages the fused region writes (the gate/up
    // shape publishes both).
    let mut published_results = Vec::new();
    for (ordinal, result) in builder.graph().results.iter().enumerate() {
        match result {
            seismic_lang::logical::RegionResult::Value { id, .. } => {
                if region_outputs.contains(id) {
                    published_results.push(ordinal as u32);
                }
            }
            seismic_lang::logical::RegionResult::State { storage, .. } => {
                if region_state_storages.contains(storage) {
                    published_results.push(ordinal as u32);
                }
            }
        }
    }
    let cost = seismic_lang::sym::Sym::constant(0).add(&seismic_lang::sym::Sym::constant(1));
    Ok(FusionReceipt {
        shared_operand,
        fused_nodes: nodes,
        kernel_local_values: kernel_local,
        published_results,
        cost,
        model,
    })
}

/// Explicitly publish every boundary result of one fusion receipt. For the
/// gate/up shape this is exactly two `complete_result` transitions with
/// distinct result storages — never aliasing after selection.
pub fn complete_fused_results<D: ExecutableDialect>(
    builder: &mut AlternativeBuilder<D>,
    receipt: &FusionReceipt,
) -> Result<(), BuilderError> {
    for ordinal in &receipt.published_results {
        let result = builder
            .graph()
            .results
            .get(*ordinal as usize)
            .cloned()
            .ok_or_else(|| format!("boundary result#{ordinal} is absent"))?;
        let transport = match &result {
            seismic_lang::logical::RegionResult::Value { id, .. } => builder.transport_of(*id)?,
            seismic_lang::logical::RegionResult::State { storage, .. } => {
                match builder.storage_of(*storage) {
                    Some(template) => TransportTemplate::Storage(
                        seismic_lang::types::NonEmpty::new(vec![
                            seismic_realization::executable::StorageViewTemplate {
                                storage: template,
                                access: seismic_lang::logical::Access::Exclusive,
                                transform: seismic_lang::logical::ViewTransform::Identity,
                            },
                        ])
                        .expect("one plane"),
                    ),
                    None => TransportTemplate::Boundary(
                        seismic_realization::executable::BoundaryLeaf::Result {
                            leaf: seismic_lang::types::ValuePath::default(),
                        },
                    ),
                }
            }
        };
        builder.complete_result(*ordinal, transport)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Cross-call boundary substitution (family 4)
// ---------------------------------------------------------------------------

/// Whether one transport (recursively through tuples) is legal to cross a
/// call boundary: retained storage, a planned executor-scalar slot, a
/// boundary placeholder, or void. `Kernel` transports are legal only
/// between steps fused into the same native launch — they can never cross a
/// schedule step or call boundary.
fn crosses_launch_boundary(transport: &TransportTemplate) -> bool {
    match transport {
        TransportTemplate::Kernel(_) => true,
        TransportTemplate::Tuple(items) => items.as_slice().iter().any(crosses_launch_boundary),
        _ => false,
    }
}

/// Consume one call occurrence through boundary substitution: every boundary
/// leaf transports directly to a caller transport, the child allocates
/// nothing for its boundary, and no `Kernel` transport crosses the call.
/// Capability values cannot cross an unfused call boundary (the logical
/// layer already rejects them; `invoke` refuses them again).
pub fn cross_call_environment<D: ExecutableDialect>(
    builder: &mut AlternativeBuilder<D>,
    call: NodeRef,
    boundary: BoundaryTemplates,
) -> Result<(), BuilderError> {
    for (path, transport) in boundary.inputs.iter().chain(boundary.results.iter()) {
        if crosses_launch_boundary(transport) {
            return Err(format!(
                "a kernel-local transport cannot cross the call boundary at {path:?}"
            ));
        }
    }
    for state in boundary.states.values() {
        match state {
            // A state transport always names resolved storage or a boundary
            // placeholder; `invoke` re-validates both.
            StateTransportTemplate::Storage(_) | StateTransportTemplate::Boundary(_) => {}
        }
    }
    builder.invoke(call, boundary)
}
