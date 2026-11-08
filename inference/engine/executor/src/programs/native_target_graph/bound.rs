use super::{
    block::PreparedTargetBlockGraph,
    classes::{Passes, StagedClass, block_class, once_class},
    entry::{EntryTokens, PreparedPerLayerEntryGraph, PreparedTargetEntryGraph},
    heap::{GraphCounts, block_binding_storage},
    prepared::{PreparedTargetGraphs, block_scope},
    weights::{BankPort, GraphPark, GraphTimeline},
};
use crate::{
    AllocationError, ResidentTarget,
    device_table::ResidentTable,
    operators::routed::split::{
        Extents, Merged, OnceStage, PassPorts, RunStage, SplitValues, Stage,
    },
    programs::native_constants::ConstantTensors,
    stored_table::{SelectionNotice, StoredTable},
    streamed_table::StreamedTable,
};
use magnitude_family_contracts::{WeightRole, WeightScope};
use seismic::{BoundNativeGraphPlan, Device, HostHeap, NativePort};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::Arc};

pub(crate) struct BoundTargetGraphs {
    /// The prepared graphs these bind: the load's one copy, shared with
    /// its attested programs. They are immutable once sealed.
    pub prepared: Arc<PreparedTargetGraphs>,
    pub entry_bound: BTreeMap<(u64, EntryTokens), BoundNativeGraphPlan>,
    /// Per block, its bound graphs (`BlockBindings`).
    pub(super) bound: Vec<BlockBindings>,
    pub(super) constants: Vec<seismic::Tensor>,
    pub per_layer: Option<BoundPerLayerEntry>,
    /// The pass's buffers of a load with split layers.
    pub(super) pass: Option<PassBuffers>,
    /// The floor group's buffers, of one place: never released.
    pub(super) floor: Rc<GroupBuffers>,
    /// The buffers of the group of more places held now, if any: kept while
    /// memory is Normal, released at Reclaim's surplus rung.
    pub(super) held: RefCell<Option<Rc<GroupBuffers>>>,
    pub(super) device: Device,
    /// The token embedding as the entries gather it.
    pub embedding: ResidentTable,
    /// The stored tables the blocks read (`stored_table_rows`), by block.
    pub stored: BTreeMap<usize, Arc<StoredTable>>,
    /// Where steps reading selected tokens copy them for the host: present
    /// with a stored table (`EntryNotice`).
    pub notice: Option<SelectionNotice>,
    /// The expert bank of a load that streams its routed experts: every
    /// step is one of its steps.
    pub bank: Option<Arc<crate::experts::ExpertBank>>,
}

/// A run of a step of its load's expert bank: the bank, the step's
/// sequence in it (`ExpertStep::sequence`), whether the step times its
/// blocks (`ExpertStep::samples`), and whether its chain began resident
/// (`ChainServing::resident`), so its whole decode blocks take their
/// resident forms (`PreparedTargetGraphs::run_stage`). Only a load with a
/// bank holds graphs of bank parts and parks
/// (`PreparedTargetGraphs::bind_weights`).
#[derive(Clone)]
pub(crate) struct BankRun {
    pub(crate) bank: Arc<crate::experts::ExpertBank>,
    pub(crate) sequence: u64,
    pub(crate) sampled: bool,
    pub(crate) resident: bool,
}

impl BankRun {
    /// Bind `graph`'s bank parts for this run as block `index` at chain
    /// place `place` (`placed_banks`), and its park and publication nodes
    /// to the bank's park timeline (`crate::experts::Parks`); the run takes
    /// their values when queued ([`Self::register_parks`]).
    pub(crate) fn bind<P>(
        &self,
        graph: &PreparedTargetBlockGraph<P>,
        index: usize,
        place: usize,
        bindings: &mut seismic::NativeGraphBindings,
    ) -> Result<(), String> {
        for (port, words) in placed_banks(&self.bank, graph, index, place)? {
            bindings.set(port, &words).map_err(|error| error.to_string())?;
        }
        for &(_, port) in &graph.parks {
            bindings.set_park(port, self.bank.parks().timeline()).map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// Whether `graph`'s run in this step is sampled for its block's
    /// device time: a graph whose ends are marked
    /// (`PreparedTargetBlockGraph::span`), of a sampled step.
    pub(crate) fn samples<P>(&self, graph: &PreparedTargetBlockGraph<P>) -> bool {
        self.sampled && graph.span.is_some()
    }

    /// `graph`'s run as block `index` of this step was sampled (`spans`):
    /// its streamed layer's block line observes its device time once the
    /// step completes.
    pub(crate) fn block_sampled<P>(&self, graph: &PreparedTargetBlockGraph<P>, index: usize, spans: seismic::NativeSpans) -> Result<(), String> {
        // Only a graph whose ends are marked is sampled (`Self::samples`).
        let marks = graph.span.expect("a sampled run's graph is marked at its ends");
        self.bank.block_sampled(self.sequence, block_layer(self.bank.plan(), index)?, spans, marks);
        Ok(())
    }

    /// This step's fill, `graph`'s run as block `index`, is `followed`
    /// directly by a sibling fill on its queue
    /// (`ExpertBank::fill_followed`), at block `index`'s streamed layer.
    pub(crate) fn fill_followed(&self, index: usize, followed: bool) -> Result<(), String> {
        self.bank.fill_followed(self.sequence, block_layer(self.bank.plan(), index)?, followed);
        Ok(())
    }

    /// Register the parks and publications of `graph`'s run as block
    /// `index` of this step that took a value when it was queued
    /// (`outputs`, its outputs), each at block `index`'s streamed layer,
    /// whichever block the graph was prepared for.
    pub(crate) fn register_parks<P>(
        &self,
        graph: &PreparedTargetBlockGraph<P>,
        index: usize,
        outputs: &seismic::NativeGraphOutputs,
    ) -> Result<(), String> {
        let layer = |scope: WeightScope| self.bank.streamed_layer(block_scope(scope, graph.block, index)?);
        for &(node, port) in &graph.parks {
            let Some(value) = outputs.park(port) else {
                continue;
            };
            match node {
                GraphTimeline::Park(GraphPark { scope, kind }) => {
                    let site = crate::experts::ParkSite { layer: layer(scope)?, kind };
                    self.bank.parks().register(value, self.sequence, site);
                }
                GraphTimeline::Publish(scope) => self.bank.parks().publish(value, layer(scope)?, self.sequence),
            }
        }
        Ok(())
    }
}

/// The bank's index of block `index`'s streamed layer: its feed-forward,
/// the sublayer's own or a parallel sublayer's routed branch
/// (`ExpertBankPlan::feed_forward`).
pub(crate) fn block_layer(bank: &crate::ExpertBankPlan, index: usize) -> Result<usize, String> {
    let [_, feed_forward] = super::prepared::block_scopes(index)?;
    bank.feed_forward(feed_forward)
        .map(|(layer, _)| layer)
        .ok_or_else(|| format!("{feed_forward:?} has no streamed routed layer"))
}

/// The words of `bank` a run of block `index` at chain place `place` binds
/// to `graph`'s placed bank ports (`BankPart::placed`).
pub(crate) fn placed_banks<'g, P>(
    bank: &crate::experts::ExpertBank,
    graph: &'g PreparedTargetBlockGraph<P>,
    index: usize,
    place: usize,
) -> Result<Vec<(&'g NativePort, seismic::Tensor)>, String> {
    graph
        .banks
        .iter()
        .filter(|(bank, _)| bank.part.placed())
        .map(|(port, native)| {
            let scope = block_scope(port.scope, graph.block, index)?;
            Ok((native, bank.placed(scope, port.part, place)?))
        })
        .collect()
}

/// The merged experts and row table of a group's pass, which every group
/// shares: a pass's backs read the merged rows before the next pass writes
/// them; beside the split layers whose values each group's buffers hold at
/// its rows.
pub(crate) struct PassBuffers {
    pub(super) experts: seismic::Tensor,
    pub(super) table: seismic::Tensor,
    pub(super) layers: super::SplitLayers,
}

pub(super) fn zeros(device: &Device, (element, extents): &Extents) -> Result<seismic::Tensor, AllocationError> {
    seismic::Tensor::zeros(device, *element, extents).map_err(AllocationError::Tensor)
}

/// The buffers a prompt group of `places` places holds (`GroupExtents`),
/// with each routed block's views of its crossing and merged positions and
/// of the pass's merged experts and row table (`SplitValues::view`), which
/// its stages bind. A launch alone takes the floor's, of one place; a chain
/// the held group's (`BoundTargetGraphs::group`).
pub(crate) struct GroupBuffers {
    places: usize,
    /// The draft input rows of a tapped target: tapped blocks write their
    /// column blocks of the leading rows of a step's region, and the
    /// readout fuses them.
    pub taps: Option<seismic::Tensor>,
    /// The per-layer rows of a per-layer model: the entry writes the leading
    /// rows of a step's region, and every per-layer input sublayer reads
    /// them.
    pub per_layer_rows: Option<seismic::Tensor>,
    /// The step's token rows of a model whose blocks read them: a step of
    /// host tokens copies them into its region in its entry
    /// (`kept_tokens`), and its blocks read them; a step of selected tokens
    /// reads its selection.
    pub token_rows: Option<seismic::Tensor>,
    /// The group's crossing and merged positions beside the load's pass
    /// buffers, of a load whose blocks split.
    split: Option<SplitValues<seismic::Tensor>>,
    blocks: Vec<Option<SplitValues<seismic::Tensor>>>,
}

impl GroupBuffers {
    /// A group of `places` places' buffers, beside the load's `pass`
    /// buffers, which a load holds exactly where its blocks split.
    pub(super) fn new(prepared: &PreparedTargetGraphs, device: &Device, places: usize, pass: Option<&PassBuffers>) -> Result<Self, AllocationError> {
        let invariant = |detail: String| AllocationError::Plan(crate::InvariantError { context: "group buffers", detail });
        let extents = prepared.group_extents(places);
        let split = pass
            .map(|pass| {
                let values = pass.layers.at_rows(prepared.region_start(places));
                Ok::<_, AllocationError>(SplitValues {
                    crossing: values.crossing.try_map(|value| zeros(device, value))?,
                    merged: Merged {
                        experts: pass.experts.clone(),
                        inverse: zeros(device, &values.merged.inverse)?,
                        table: pass.table.clone(),
                    },
                })
            })
            .transpose()?;
        let blocks = match &split {
            Some(buffers) => {
                let rows = prepared.region_start(places);
                prepared
                    .served
                    .iter()
                    .map(|block| block.split.as_ref().map(|layer| buffers.view(&layer.at_rows(rows))).transpose())
                    .collect::<Result<Vec<_>, String>>()
                    .map_err(invariant)?
            }
            // No block splits.
            None => Vec::new(),
        };
        let allocate = |value: &Option<Extents>| value.as_ref().map(|value| zeros(device, value)).transpose();
        Ok(Self {
            places,
            taps: allocate(&extents.taps)?,
            per_layer_rows: allocate(&extents.per_layer)?,
            token_rows: allocate(&extents.tokens)?,
            split,
            blocks,
        })
    }

    /// The group's places: one, the floor's, or those `hold_group` granted.
    pub(crate) fn places(&self) -> usize {
        self.places
    }

    /// Block `index`'s routed layer's views of the crossing and merged
    /// buffers.
    pub(crate) fn split(&self, index: usize) -> Result<&SplitValues<seismic::Tensor>, String> {
        self.blocks
            .get(index)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("block {index} splits without a routed layer"))
    }

    /// The buffers the group owns.
    fn storage(&self) -> impl Iterator<Item = &seismic::Tensor> {
        let split = self.split.iter().flat_map(|split| split.crossing.iter().chain([&split.merged.inverse]));
        self.taps.iter().chain(&self.per_layer_rows).chain(&self.token_rows).chain(split)
    }
}

impl BoundTargetGraphs {
    /// The buffers of a chain of `launches` launches: the floor's alone,
    /// the held group's otherwise, held with the class whose places bound
    /// a chain (`AllocatedResources::group_places`).
    pub(crate) fn group(&self, launches: usize) -> Result<Rc<GroupBuffers>, String> {
        match launches {
            1 => Ok(self.floor.clone()),
            _ => self
                .held
                .borrow()
                .clone()
                .ok_or_else(|| format!("a chain of {launches} launches without a group held")),
        }
    }

    /// The bytes a group of `places` places' buffers hold beside the floor's
    /// (`hold_group`): none of one place, which the floor serves.
    pub(crate) fn group_bytes(&self, places: usize) -> Result<u64, String> {
        self.prepared.group_bytes(places)
    }

    /// Hold a group of `places` places' buffers in place of any held now
    /// (`group_bytes`); the floor serves a group of one. A failure keeps
    /// the buffers held before.
    pub(crate) fn hold_group(&self, places: usize) -> Result<(), AllocationError> {
        let held = (places > 1)
            .then(|| GroupBuffers::new(&self.prepared, &self.device, places, self.pass.as_ref()).map(Rc::new))
            .transpose()?;
        *self.held.borrow_mut() = held;
        Ok(())
    }

    /// Visit the graph constants and buffers the bindings hold.
    pub(crate) fn visit_constants(&self, visit: &mut dyn FnMut(&seismic::Tensor)) {
        let held = self.held.borrow();
        self.constants
            .iter()
            .chain(self.pass.iter().flat_map(|pass| [&pass.experts, &pass.table]))
            .chain(self.floor.storage())
            .chain(held.iter().flat_map(|held| held.storage()))
            .chain(self.notice.iter().map(SelectionNotice::copies))
            .for_each(visit);
    }

    /// The host heap the bindings hold beyond the prepared graphs they
    /// share: each binding's own (`BoundNativeGraphPlan::host_heap`) and the
    /// vectors that hold the blocks' bindings, at their capacities. The
    /// nodes of the two small maps of entry bindings are the allocator's
    /// and are not in it.
    pub(super) fn host_heap(&self) -> Result<HostHeap, String> {
        let entries = self.entry_bound.values().chain(
            self.per_layer
                .iter()
                .flat_map(|per_layer| per_layer.bound.values()),
        );
        let blocks = self.bound.iter().flat_map(|block| {
            let runs = block.runs.iter().chain(&block.untapped).map(|(_, bound)| bound);
            runs.chain(block.passes.iter().map(|(_, bound)| bound))
        });
        let bindings = entries
            .chain(blocks)
            .try_fold(HostHeap::default(), |heap, bound| {
                heap.plus(bound.host_heap())
            })
            .ok_or("bound graph heap overflows")?;
        // Each block's vectors were made with exactly its graphs' capacity,
        // and the outer one with the blocks'.
        let held = self
            .bound
            .iter()
            .map(|block| GraphCounts {
                runs: block.runs.capacity(),
                passes: block.passes.capacity(),
                untapped: block.untapped.capacity(),
            })
            .collect::<Vec<_>>();
        if self.bound.capacity() != held.len() {
            return Err("the blocks' bindings are held in more than their vector".into());
        }
        bindings
            .plus(block_binding_storage(&held)?)
            .ok_or_else(|| "bound graph heap overflows".to_owned())
    }
}

/// Block `index`'s `graphs` of one kind, each bound to the resident weights,
/// bank parts and constants it reads statically, in their order.
pub(super) fn bind_block_graphs<K: Copy + std::fmt::Debug, P>(
    graphs: &[(K, Arc<PreparedTargetBlockGraph<P>>)],
    index: usize,
    resident: &ResidentTarget,
    constants: &mut ConstantTensors,
) -> Result<Vec<(K, BoundNativeGraphPlan)>, String> {
    let mut bound = Vec::with_capacity(graphs.len());
    for (class, graph) in graphs {
        let constant_tensors = graph
            .constants
            .iter()
            .map(|constant| Ok((constant.port(), constants.tensor(constant)?)))
            .collect::<Result<Vec<_>, String>>()?;
        let fixed = graph
            .weights
            .iter()
            .map(|(weight, port)| {
                // A graph shared by equal-shape blocks names the roles of
                // the block it was prepared for; each block binds its own.
                let role = WeightRole {
                    scope: block_scope(weight.role.scope, graph.block, index)?,
                    kind: weight.role.kind,
                };
                Ok((port, resident.bound(role, weight.part)?))
            })
            // A placed bank part is a run's own place's.
            .chain(graph.banks.iter().filter(|(bank, _)| !bank.part.placed()).map(|(bank, port)| {
                let scope = block_scope(bank.scope, graph.block, index)?;
                Ok((port, resident.bank_part(BankPort { scope, ..*bank })?))
            }))
            .chain(constant_tensors.iter().map(|(port, tensor)| Ok((*port, tensor))))
            .collect::<Result<Vec<_>, String>>()?;
        bound.push((
            *class,
            graph.plan.bind_static(&fixed).map_err(|error| {
                format!("target graph static binding class {class:?} block {index}: {error}")
            })?,
        ));
    }
    Ok(bound)
}

/// A block's bound graphs, as its prepared ones (`BlockGraphs`): a vector
/// of exactly each kind, what the plan counted (`block_binding_storage`).
pub(super) struct BlockBindings {
    pub(super) runs: Vec<(StagedClass, BoundNativeGraphPlan)>,
    pub(super) passes: Vec<(u64, BoundNativeGraphPlan)>,
    pub(super) untapped: Vec<(StagedClass, BoundNativeGraphPlan)>,
}

/// The bound per-layer entry of a per-layer model.
pub(crate) struct BoundPerLayerEntry {
    /// The graph per row class, bound to its weights.
    pub(super) bound: BTreeMap<u64, BoundNativeGraphPlan>,
    /// The table whose rows each step gathers, streamed.
    pub table: StreamedTable,
    /// The per-layer rows the entry computed for single tokens, kept in the
    /// floor's rows a single-row step leaves unwritten
    /// (`GroupBuffers::per_layer_rows`).
    pub token_rows: TokenRows,
}

/// Where a single-row step of a text token finds its per-layer row. A text
/// token's per-layer row is its own alone, so the row a step computed for a
/// token stays where the step wrote it, and a later single-row step of the
/// token reads it there instead of running the entry. The rows held are the
/// per-layer rows after the first, the last row the first slot; a step of
/// more rows writes over the ones its rows cover.
pub(crate) struct TokenRows(std::sync::Mutex<crate::slot_directory::SlotDirectory>);

/// Where a single-row step of a token finds its per-layer row.
pub(crate) enum TokenRow {
    /// An earlier step wrote it into this slot.
    Held(usize),
    /// No slot holds it, and this one is free: the step that writes it
    /// there holds it (`TokenRows::hold`).
    Free(usize),
    /// No slot holds it, and none is free.
    None,
}

impl TokenRows {
    /// Rows of `tokens` tokens, in `slots` slots.
    pub(super) fn new(tokens: u64, slots: usize) -> Self {
        Self(std::sync::Mutex::new(
            crate::slot_directory::SlotDirectory::new(
                tokens,
                slots as u64,
                1,
                crate::planning::DirectoryForm::Dense,
            ),
        ))
    }

    /// Where a single-row step of `token` finds its per-layer row.
    pub(crate) fn find(&self, token: u64) -> Result<TokenRow, String> {
        let mut slots = self.0.lock().expect("token rows lock");
        if !slots.has_row(token) {
            return Err(format!("token {token} has no per-layer row"));
        }
        Ok(match slots.use_slot(token) {
            Some(slot) => TokenRow::Held(slot as usize),
            None => slots
                .free(|_| false)
                .map_or(TokenRow::None, |(slot, _)| TokenRow::Free(slot)),
        })
    }

    /// The step that writes `token`'s row into the free slot `slot` is
    /// queued: steps after it find the row there.
    pub(crate) fn hold(&self, token: u64, slot: usize) {
        self.0
            .lock()
            .expect("token rows lock")
            .hold(token, slot, true);
    }

    /// A step writes other rows over the slots `slots`.
    pub(crate) fn overwrite(&self, slots: std::ops::Range<usize>) {
        let mut held = self.0.lock().expect("token rows lock");
        for slot in slots {
            held.release(slot);
        }
    }
}

impl BoundTargetGraphs {
    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        let mut bytes = Some(0u64);
        self.visit_constants(&mut |tensor| bytes = bytes.and_then(|bytes| bytes.checked_add(tensor.storage_bytes())));
        bytes.ok_or("target graph constant charge overflows")
    }

    pub(crate) fn entry(
        &self,
        rows: u64,
        source: EntryTokens,
    ) -> Result<(&PreparedTargetEntryGraph, &BoundNativeGraphPlan), String> {
        let key = (rows, source);
        let graph = self
            .prepared
            .entries
            .get(&key)
            .ok_or_else(|| format!("target embedding class {key:?} was not sealed"))?;
        let bound = self
            .entry_bound
            .get(&key)
            .ok_or_else(|| format!("target embedding class {key:?} was not bound"))?;
        Ok((graph, bound))
    }

    /// The per-layer entry graph of row class `rows`, of a per-layer model.
    #[allow(clippy::type_complexity)]
    pub(crate) fn per_layer_entry(
        &self,
        rows: u64,
    ) -> Option<
        Result<
            (
                &PreparedPerLayerEntryGraph,
                &BoundNativeGraphPlan,
                &BoundPerLayerEntry,
            ),
            String,
        >,
    > {
        let per_layer = self.per_layer.as_ref()?;
        Some(
            self.prepared
                .per_layer_entries
                .get(&rows)
                .zip(per_layer.bound.get(&rows))
                .map(|(graph, bound)| (graph, bound, per_layer))
                .ok_or_else(|| format!("per-layer entry class {rows} was not sealed and bound")),
        )
    }

    /// Block `index`'s graph of a run's `stage` of its class, in a launch
    /// some of whose rows demand features (`tapped`) or none: a tapped
    /// block's untapped graph in the latter (`BlockGraphs::untapped`), as a
    /// load without the drafter runs it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn block(
        &self,
        rows: u64,
        segments: u64,
        slots: u64,
        listed: u64,
        scored: bool,
        index: usize,
        stage: RunStage,
        tapped: bool,
    ) -> Result<(&PreparedTargetBlockGraph, &BoundNativeGraphPlan), String> {
        let mixer = *self
            .prepared
            .mixers
            .get(index)
            .ok_or_else(|| format!("target block {index} does not exist"))?;
        let key = (block_class(mixer, rows, segments, slots, listed, scored), Stage::Run(stage));
        match tapped || !self.prepared.block_taps(index).positions().any() {
            true => self.staged(index, key),
            false => sealed_and_bound(&self.prepared.blocks_by_class[index].untapped, &self.bound[index].untapped, index, key),
        }
    }

    /// Block `index`'s fill, sealed once (`once_class`).
    pub(crate) fn fill(&self, index: usize) -> Result<(&PreparedTargetBlockGraph, &BoundNativeGraphPlan), String> {
        self.staged(index, once_class(self.prepared.passes.rows(), OnceStage::Fill))
    }

    /// Block `index`'s pass in a chain of `launches` launches and its rows
    /// (`Passes::of_chain`).
    pub(crate) fn pass(
        &self,
        index: usize,
        launches: usize,
    ) -> Result<(&PreparedTargetBlockGraph<PassPorts>, &BoundNativeGraphPlan, u64), String> {
        let rows = self.prepared.passes.of_chain(launches);
        let (graph, bound) = sealed_and_bound(
            &self.prepared.blocks_by_class[index].passes,
            &self.bound[index].passes,
            index,
            rows,
        )?;
        Ok((graph, bound, rows))
    }

    /// The passes a group runs (`group_pass_rows`).
    pub(crate) fn passes(&self) -> Passes {
        self.prepared.passes
    }

    /// Block `index`'s graph of a run's stage of class `key`.
    fn staged(
        &self,
        index: usize,
        key: StagedClass,
    ) -> Result<(&PreparedTargetBlockGraph, &BoundNativeGraphPlan), String> {
        sealed_and_bound(&self.prepared.blocks_by_class[index].runs, &self.bound[index].runs, index, key)
    }
}

/// Block `index`'s graph of `key` among its sealed `graphs` of one kind, and
/// its binding among their `bound`.
fn sealed_and_bound<'g, K: Ord + std::fmt::Debug, P>(
    graphs: &'g [(K, Arc<PreparedTargetBlockGraph<P>>)],
    bound: &'g [(K, BoundNativeGraphPlan)],
    index: usize,
    key: K,
) -> Result<(&'g PreparedTargetBlockGraph<P>, &'g BoundNativeGraphPlan), String> {
    let graph = graphs
        .binary_search_by(|(class, _)| class.cmp(&key))
        .map(|at| graphs[at].1.as_ref())
        .map_err(|_| format!("target graph {key:?} of block {index} was not sealed"))?;
    let bound = bound
        .binary_search_by(|(class, _)| class.cmp(&key))
        .map(|at| &bound[at].1)
        .map_err(|_| format!("target graph {key:?} of block {index} was not bound"))?;
    Ok((graph, bound))
}
