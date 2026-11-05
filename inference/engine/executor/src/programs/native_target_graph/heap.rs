use super::{
    block::{BlockGraphs, PreparedTargetBlockGraph},
    bound::BlockBindings,
    certify::CertifiedBlockGraph,
    classes::StagedClass,
    weights::{BankPort, GraphTimeline, WeightPort},
};
use crate::{
    operators::routed::split::{OnceStage, PassPorts, SplitPorts, Stage},
    programs::native_constants::GraphConstant,
};
use seismic::{
    BoundNativeGraphPlan, HostHeap, NativeGraphLayout, NativeGraphPlan, NativePort,
    NativeTimelinePort,
};
use std::sync::Arc;

/// A count of sealed target graphs and the host heap that many hold. Each
/// term is exact for the graphs counted: a plan's heap follows from its
/// layout's topology (`NativeGraphLayout::plan_heap`), and so do the
/// lengths of the vectors its prepared graph owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SealedGraphsHeap {
    pub graphs: usize,
    /// The sealed plans (`NativeGraphPlan::host_heap`).
    pub plans: HostHeap,
    /// What stores the prepared block graphs (`block_graph_storage`): one
    /// shared allocation for each sealed block graph, and each block's
    /// vector of its classes.
    pub storage: HostHeap,
    /// The vectors the prepared block graphs own: each one's weight ports,
    /// constants and state ports (`PreparedTargetBlockGraph::owned_heap`).
    pub owned: HostHeap,
}

impl SealedGraphsHeap {
    /// The three terms together.
    pub fn heap(&self) -> Option<HostHeap> {
        self.plans.plus(self.storage)?.plus(self.owned)
    }
}

/// The target graphs a load seals and the host heap they hold, from the
/// plan alone (`certify_target_family`), bounded from both sides. Where
/// every block's sharing follows from the plan the two are one count, the
/// sealed one. Where blocks of one shape read layer scales, which the plan
/// does not know, they are apart: what holds a load to a budget reads
/// `at_most`, and what is held to the load's own holding reads `at_least`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlannedSealedGraphs {
    /// No load seals more: every block that reads a layer scale counted
    /// with plans of its own (`UnreadScales::Distinct`).
    pub at_most: SealedGraphsHeap,
    /// No load seals fewer: blocks of one shape counted with one block's
    /// plans (`UnreadScales::Equal`).
    pub at_least: SealedGraphsHeap,
    /// The host heap of the graphs' static bindings, which a load makes
    /// when it binds its resident weights (`BoundTargetGraphs::host_heap`):
    /// one binding for each entry graph and for every block's every class,
    /// shared plan or not, and the vectors that hold the blocks' bindings.
    /// Exact whether or not the count is: blocks that share a plan each
    /// bind it.
    pub bound: HostHeap,
    /// The host heap of the family's one copy of each distinct constant
    /// value its block graphs bind (`FamilyConstants::host_heap`). Exact
    /// whether or not the count is: blocks of one shape bind the same
    /// values.
    pub constants: HostHeap,
}

impl PlannedSealedGraphs {
    /// Whether the plan fixes the sealed count: `at_most` is `at_least`.
    pub fn exact(&self) -> bool {
        self.at_most == self.at_least
    }

    /// The most host bytes a load's sealed target graphs hold: their
    /// plans, the prepared graphs that hold them and the family's constant
    /// values.
    pub fn at_most_bytes(&self) -> Option<u64> {
        self.at_most.heap()?.bytes.checked_add(self.constants.bytes)
    }

    /// The least host bytes a load's sealed target graphs hold.
    pub fn at_least_bytes(&self) -> Option<u64> {
        self.at_least
            .heap()?
            .bytes
            .checked_add(self.constants.bytes)
    }
}

/// The host heap of the vectors a prepared block graph owns, from their
/// lengths: `weights` weight ports, `banks` bank ports, `parks` park and
/// publication nodes, `constants` constants and `state` state ports. What
/// the planner counts from a block's template and what
/// `PreparedTargetBlockGraph::prepare` leaves a graph holding.
pub(super) fn owned_vectors_heap(weights: usize, banks: usize, parks: usize, constants: usize, state: usize) -> HostHeap {
    use std::mem::size_of;
    let mut heap = HostHeap::default();
    heap.hold(weights, size_of::<(WeightPort, NativePort)>());
    heap.hold(banks, size_of::<(BankPort, NativePort)>());
    heap.hold(parks, size_of::<(GraphTimeline, NativeTimelinePort)>());
    heap.hold(constants, size_of::<GraphConstant>());
    heap.hold(state, size_of::<NativePort>());
    heap
}

/// A prepared block graph of ports `P` in the shared allocation the blocks
/// that seal one plan hold it in: the graph and the allocation's two
/// reference counts.
const fn shared_block_graph_bytes<P>() -> usize {
    std::mem::size_of::<PreparedTargetBlockGraph<P>>() + 2 * std::mem::size_of::<usize>()
}

/// The host heap that stores a target's prepared block graphs: the vector
/// of the blocks' `BlockGraphs`, each block's vectors of its `classes[block]`
/// run graphs and passes, and one shared allocation for each of the
/// `sealed` graphs. What the planner counts and what
/// `PreparedTargetGraphs::prepare` allocates.
pub(super) fn block_graph_storage(classes: &[GraphCounts], sealed: GraphCounts) -> Result<HostHeap, String> {
    use std::mem::size_of;
    let shared = |count: usize, bytes: usize| HostHeap::allocation(1, bytes).times(count as u64);
    classes
        .iter()
        .try_fold(HostHeap::allocation(classes.len(), size_of::<BlockGraphs>()), |heap, counts| {
            heap.plus(HostHeap::allocation(
                counts.runs,
                size_of::<(StagedClass, Arc<PreparedTargetBlockGraph>)>(),
            ))?
            .plus(HostHeap::allocation(
                counts.passes,
                size_of::<(u64, Arc<PreparedTargetBlockGraph<PassPorts>>)>(),
            ))
        })
        .and_then(|heap| heap.plus(shared(sealed.runs, shared_block_graph_bytes::<Option<SplitPorts>>())?))
        .and_then(|heap| heap.plus(shared(sealed.passes, shared_block_graph_bytes::<PassPorts>())?))
        .ok_or_else(|| "prepared block graph storage overflows".to_owned())
}

/// The host heap of the vectors that hold a target's block bindings: one of
/// the blocks' `BlockBindings`, and each block's of its run graphs' and
/// passes' bindings, `bindings[block]` of them. What the planner counts and
/// what `BoundTargetGraphs` allocates.
pub(super) fn block_binding_storage(bindings: &[GraphCounts]) -> Result<HostHeap, String> {
    use std::mem::size_of;
    bindings
        .iter()
        .try_fold(HostHeap::allocation(bindings.len(), size_of::<BlockBindings>()), |heap, counts| {
            heap.plus(HostHeap::allocation(counts.runs, size_of::<(StagedClass, BoundNativeGraphPlan)>()))?
                .plus(HostHeap::allocation(counts.passes, size_of::<(u64, BoundNativeGraphPlan)>()))
        })
        .ok_or_else(|| "target binding storage overflows".to_owned())
}

/// The planned graphs accumulated: each adds its layout's plan heap, and a
/// block graph the vectors its prepared graph owns.
#[derive(Default)]
pub(super) struct PlannedHeap {
    pub(super) graphs: usize,
    pub(super) plans: HostHeap,
    pub(super) owned: HostHeap,
}

impl PlannedHeap {
    pub(super) fn seal(&mut self, layout: &NativeGraphLayout) -> Result<(), String> {
        self.graphs += 1;
        self.plans = self
            .plans
            .plus(layout.plan_heap())
            .ok_or("planned sealed graph heap overflows")?;
        Ok(())
    }

    pub(super) fn seal_block(&mut self, certified: &CertifiedBlockGraph) -> Result<(), String> {
        self.seal(&certified.layout)?;
        self.owned = self
            .owned
            .plus(certified.owned)
            .ok_or("planned prepared graph heap overflows")?;
        Ok(())
    }
}

/// The sealed graphs' host heap against their layouts' planned heap.
#[derive(Default)]
pub(super) struct SealedHeap {
    pub(super) heap: HostHeap,
    /// The vectors the prepared block graphs own.
    pub(super) owned: HostHeap,
}

impl SealedHeap {
    /// Count the vectors `block` owns, prepared into `certified`. Vectors
    /// of other lengths than its template's are a defect of the plan's
    /// accounting, as a plan of another heap is (`seal`).
    pub(super) fn own<P>(
        &mut self,
        block: &PreparedTargetBlockGraph<P>,
        certified: &CertifiedBlockGraph,
        graph: impl Fn() -> String,
    ) -> Result<(), String> {
        let (held, planned) = (block.owned_heap(), certified.owned);
        self.owned = self
            .owned
            .plus(held)
            .ok_or("prepared graph heap overflows")?;
        if held != planned {
            return Err(format!(
                "{} owns {} host bytes in {} allocations; its template planned {} in {}",
                graph(),
                held.bytes,
                held.allocations,
                planned.bytes,
                planned.allocations
            ));
        }
        Ok(())
    }

    /// Count `plan`, sealed into `layout`. A plan that holds another heap
    /// than its layout planned is a defect of the plan's accounting, which
    /// refuses the load in every build.
    pub(super) fn seal(
        &mut self,
        plan: &NativeGraphPlan,
        layout: &NativeGraphLayout,
        graph: impl Fn() -> String,
    ) -> Result<(), String> {
        let (held, planned) = (plan.host_heap(), layout.plan_heap());
        self.heap = self.heap.plus(held).ok_or("sealed graph heap overflows")?;
        if held != planned {
            return Err(format!(
                "{} holds {} host bytes in {} allocations; its layout planned {} in {}",
                graph(),
                held.bytes,
                held.allocations,
                planned.bytes,
                planned.allocations
            ));
        }
        Ok(())
    }
}

/// How many graphs of each kind: a run's stages' and a pass's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct GraphCounts {
    pub(super) runs: usize,
    pub(super) passes: usize,
}

impl GraphCounts {
    /// Count one graph of `stage`.
    pub(super) fn count(&mut self, stage: Stage) {
        match stage {
            Stage::Once(OnceStage::Pass) => self.passes += 1,
            Stage::Run(_) | Stage::Once(OnceStage::Fill) => self.runs += 1,
        }
    }
}
