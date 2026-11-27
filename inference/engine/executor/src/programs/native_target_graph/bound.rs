use super::{
    block::{PassGraph, PreparedTargetBlockGraph},
    classes::{BindingKey, Passes, StagedClass, binding_key, block_class, once_class},
    entry::{EntryTokens, PreparedPerLayerEntryGraph, PreparedTargetEntryGraph},
    heap::{GraphCounts, block_binding_storage},
    prepared::{PreparedTargetGraphs, block_scope},
    weights::{BankPort, GraphPark, GraphTimeline},
};
use crate::{
    AllocationError, ResidentTarget,
    device_table::ResidentTable,
    operators::routed::split::{
        Extents, Merged, OnceStage, RunStage, SplitValues, Stage,
    },
    programs::native_constants::ConstantTensors,
    stored_table::{SelectionNotice, StoredTable},
    streamed_table::StreamedTable,
};
use magnitude_family_contracts::{WeightRole, WeightScope};
use seismic::{BoundNativeGraphPlan, HostHeap, NativePort};
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
    /// Bound `ResidualForm::BlockAttention` executor when the target uses
    /// Block Attention Residuals (Inkling 2); `None` for standard sealed graphs.
    pub(crate) block_attn_res: Option<Arc<super::block_attn_res::BoundBlockAttnRes>>,
}

/// A run of a step of its load's expert bank: the bank, the step's
/// sequence in it (`ExpertStep::sequence`), whether the step times its
/// blocks (`ExpertStep::samples`), and how its chain is served
/// (`ChainServing`), so each whole decode block takes its resident form
/// (`PreparedTargetGraphs::run_stage`) and each whole prompt block skips
/// scratch fills. Only a load with a bank holds graphs of bank parts and parks
/// (`PreparedTargetGraphs::bind_weights`).
#[derive(Clone)]
pub(crate) struct BankRun {
    pub(crate) bank: Arc<crate::experts::ExpertBank>,
    pub(crate) sequence: u64,
    /// The step's front rows (STEP_FRONT's bits), its step words carry.
    pub(crate) front: u32,
    pub(crate) sampled: bool,
    pub(crate) serving: crate::experts::ChainServing,
}

impl BankRun {
    /// Whether block `index`'s streamed layer runs its resident decode form
    /// (`RunStage::Resident`) in this chain.
    pub(crate) fn block_decode_resident(&self, index: usize) -> bool {
        let Some(layer) = self.bank.plan().block_layer(index) else {
            return false;
        };
        self.serving.decode_resident(layer, self.bank.plan().sets_step(layer))
    }

    /// Whether block `index`'s streamed layer is whole (`POLICY_WHOLE == 1`)
    /// in this chain, so a prompt group skips `routed_scratch_request` and
    /// `run_fill` on it.
    pub(crate) fn block_whole(&self, index: usize) -> bool {
        let Some(layer) = self.bank.plan().block_layer(index) else {
            return false;
        };
        self.serving.layer_whole(layer)
    }

    /// Prepare one unified zero-switch 3-tier streamed decode flight for block
    /// `index` over `rows` rows: acquires the next ring slot lock-free via
    /// `completed_gen`, populates the host-mapped `ready_slot` descriptor
    /// header, notifies `DecodeReactor`, and returns the bound tensors and
    /// timeline handles (`land_gate`, `completed_gen`, `gen`).
    pub(crate) fn prepare_unified_decode_flight(
        &self,
        index: usize,
        rows: u64,
        mask_hits_back: bool,
    ) -> Result<crate::experts::UnifiedDecodeFlight, String> {
        let layer = self
            .bank
            .plan()
            .block_layer(index)
            .ok_or_else(|| format!("target block {index} has no streamed routed layer"))?;
        self.bank
            .prepare_unified_decode_flight(layer, self.sequence, rows, mask_hits_back)
    }

    /// If prefill copies host-tier experts into VRAM scratch via hardware DMA
    /// (`ExpertBank::tier_copy_dma`), returns the streamed layer index for
    /// block `index`; otherwise `None`.
    pub(crate) fn tier_copy_dma_layer(&self, index: usize) -> Result<Option<usize>, String> {
        if !self.bank.tier_copy_dma() {
            return Ok(None);
        }
        self.bank.plan().block_layer(index).ok_or_else(|| format!("target block {index} has no streamed routed layer")).map(Some)
    }

    /// Bind `graph`'s bank parts for this run as block `index` at chain
    /// place `place` (`placed_banks`), and its park and publication nodes
    /// to the bank's park timeline (`crate::experts::Parks`); the run takes
    /// their values when queued ([`Self::register_parks`]). The parts bind
    /// through `charges` (`seismic::NativeGraphBindings::set`).
    pub(crate) fn bind<P>(
        &self,
        graph: &PreparedTargetBlockGraph<P>,
        index: usize,
        place: usize,
        charges: &seismic::Charges,
        bindings: &mut seismic::NativeGraphBindings,
    ) -> Result<(), String> {
        for (port, words) in placed_banks(&self.bank, graph, index, place)? {
            bindings.set(charges, port, &words).map_err(|error| error.to_string())?;
        }
        for &(_, port) in &graph.parks {
            bindings.set_park(port, self.bank.parks().timeline()).map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    pub(crate) fn bind_window<P>(
        &self,
        graph: &PreparedTargetBlockGraph<P>,
        index: usize,
        window: u64,
        charges: &seismic::Charges,
        bindings: &mut seismic::NativeGraphBindings,
    ) -> Result<(), String> {
        for (port, words) in windowed_banks(&self.bank, graph, index, window)? {
            bindings.set(charges, port, &words).map_err(|error| error.to_string())?;
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
        self.bank.block_sampled(self.sequence, self.bank.plan().block_layer(index).ok_or_else(|| format!("target block {index} has no streamed routed layer"))?, spans, marks);
        Ok(())
    }

    /// This step's fill, `graph`'s run as block `index`, is `followed`
    /// directly by a sibling fill on its queue
    /// (`ExpertBank::fill_followed`), at block `index`'s streamed layer.
    pub(crate) fn fill_followed(&self, index: usize, window: u64, followed: bool) -> Result<(), String> {
        self.bank.fill_followed(self.sequence, self.bank.plan().block_layer(index).ok_or_else(|| format!("target block {index} has no streamed routed layer"))?, window, followed);
        Ok(())
    }

    /// Register the parks and publications of `graph`'s run as block
    /// `index` of this step that took a value when it was queued
    /// (`outputs`, its outputs), each at block `index`'s streamed layer
    /// (`block_scope`).
    pub(crate) fn register_parks<P>(
        &self,
        graph: &PreparedTargetBlockGraph<P>,
        index: usize,
        outputs: &seismic::NativeGraphOutputs,
    ) -> Result<(), String> {
        self.register_parks_for(graph, index, None, outputs)
    }

    pub(crate) fn register_fill_parks<P>(
        &self,
        graph: &PreparedTargetBlockGraph<P>,
        index: usize,
        window: u64,
        outputs: &seismic::NativeGraphOutputs,
    ) -> Result<(), String> {
        self.register_parks_for(graph, index, Some(window), outputs)
    }

    fn register_parks_for<P>(
        &self,
        graph: &PreparedTargetBlockGraph<P>,
        index: usize,
        window: Option<u64>,
        outputs: &seismic::NativeGraphOutputs,
    ) -> Result<(), String> {
        let layer = |scope: WeightScope| self.bank.streamed_layer(block_scope(scope, index)?);
        for &(node, port) in &graph.parks {
            let Some(value) = outputs.park(port) else {
                continue;
            };
            match node {
                GraphTimeline::Park(GraphPark { scope, kind }) => {
                    let kind = match (window, kind) {
                        (Some(w), crate::experts::ParkKind::Round(round)) => {
                            crate::experts::ParkKind::Round(w * self.bank.plan().window_rounds() + round)
                        }
                        (_, kind) => kind,
                    };
                    let site = crate::experts::ParkSite { layer: layer(scope)?, kind };
                    self.bank.parks().register(value, self.sequence, site);
                }
                GraphTimeline::Publish(scope) => self.bank.parks().publish(value, layer(scope)?, self.sequence),
            }
        }
        Ok(())
    }
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
            let scope = block_scope(port.scope, index)?;
            Ok((native, bank.placed(scope, port.part, place)?))
        })
        .collect()
}

pub(crate) fn windowed_banks<'g, P>(
    bank: &crate::experts::ExpertBank,
    graph: &'g PreparedTargetBlockGraph<P>,
    index: usize,
    window: u64,
) -> Result<Vec<(&'g NativePort, seismic::Tensor)>, String> {
    graph
        .banks
        .iter()
        .filter(|(bank, _)| bank.part.windowed())
        .map(|(port, native)| {
            let scope = block_scope(port.scope, index)?;
            Ok((native, bank.windowed(scope, port.part, window)?))
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

pub(super) fn zeros(charges: &seismic::Charges, (element, extents): &Extents) -> Result<seismic::Tensor, AllocationError> {
    seismic::Tensor::zeros(charges, *element, extents).map_err(AllocationError::Tensor)
}

/// One double-buffered slot of per-block host-visible `[rows, K]` `i32` route
/// tensors (`Tensor::shared`) for grouped-prefill per-row route export.
struct PromptRoutesSlot {
    in_use: Arc<std::sync::atomic::AtomicBool>,
    blocks: Vec<Option<seismic::Tensor>>,
}

/// Exclusive lease on one [`PromptRoutesSlot`]: guarantees no pipelined chain
/// overwrites the host-visible route buffers while the GPU writes them or
/// while [`TargetOutput::complete`] reads them on the host.
pub(crate) struct PromptRoutesLease {
    in_use: Arc<std::sync::atomic::AtomicBool>,
    blocks: Vec<Option<seismic::Tensor>>,
}

impl PromptRoutesLease {
    /// Host-visible `[rows, K]` `i32` route tensor for block `index`, if
    /// `index` is a streamed split routed block.
    pub(crate) fn routes(&self, index: usize) -> Option<&seismic::Tensor> {
        self.blocks.get(index).and_then(Option::as_ref)
    }

    /// Iterate `(block_index, tensor)` over all captured streamed blocks.
    pub(crate) fn iter_blocks(&self) -> impl Iterator<Item = (usize, &seismic::Tensor)> {
        self.blocks
            .iter()
            .enumerate()
            .filter_map(|(index, tensor)| tensor.as_ref().map(|tensor| (index, tensor)))
    }
}

impl Drop for PromptRoutesLease {
    fn drop(&mut self) {
        self.in_use.store(false, std::sync::atomic::Ordering::Release);
    }
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
    /// The host's guesses kept for a step of host tokens, of a load that
    /// fronts: `[rows, 2]` (guess, kind) rows its front rows run on, which
    /// its entry copies into its region as it keeps its tokens
    /// (`PreparedTargetEntryGraph::kept_guesses`) and its step words read.
    pub kept_guesses: Option<seismic::Tensor>,
    /// The group's crossing and merged positions beside the load's pass
    /// buffers, of a load whose blocks split.
    split: Option<SplitValues<seismic::Tensor>>,
    blocks: Vec<Option<SplitValues<seismic::Tensor>>>,
    /// Double-buffered host-visible per-layer `[rows, K]` `i32` route buffers
    /// (`Tensor::shared`) for grouped-prefill per-row route export into
    /// `TokenRoutes` (empty when no block streams).
    prompt_routes: Vec<PromptRoutesSlot>,
    /// Per place, the two operand carries its run alternates between, of a
    /// load whose decode graphs carry (`GroupExtents::operand_carry`):
    /// a block's graph reads the one the block before wrote and writes the
    /// other, which the next block's reads. A place's graphs run in queue
    /// order, so a carry is rewritten only after its reader.
    operands: Vec<[[seismic::Tensor; 2]; 2]>,
    /// Shared VRAM picks tables (`[places * region_rows, max_top]`, I32),
    /// one per non-overlapping live-interval slot.
    picks_slots: Vec<seismic::Tensor>,
    /// Per history read, `(slot, top)` into `picks_slots`.
    picks_by_read: Vec<Option<(usize, u64)>>,
    /// Shared VRAM candidate pool tables
    /// (`[places * region_rows, max_pool_blocks + max_segments, 2]`, I32),
    /// one per non-overlapping live-interval slot.
    pool_slots: Vec<seismic::Tensor>,
    /// Per history read, `(slot, pool_blocks)` into `pool_slots`.
    pool_by_read: Vec<Option<(usize, u64)>>,
}

impl GroupBuffers {
    /// A group of `places` places' buffers, beside the load's `pass`
    /// buffers, which a load holds exactly where its blocks split.
    pub(super) fn new(prepared: &PreparedTargetGraphs, charges: &seismic::Charges, places: usize, pass: Option<&PassBuffers>) -> Result<Self, AllocationError> {
        let invariant = |detail: String| AllocationError::Plan(crate::InvariantError { context: "group buffers", detail });
        let extents = prepared.group_extents(places);
        let split = pass
            .map(|pass| {
                let values = pass.layers.at_rows(prepared.region_start(places));
                Ok::<_, AllocationError>(SplitValues {
                    crossing: values.crossing.try_map(|value| zeros(charges, value))?,
                    merged: Merged {
                        experts: pass.experts.clone(),
                        inverse: zeros(charges, &values.merged.inverse)?,
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
        let prompt_routes = if split.is_some() && prepared.served.iter().any(|block| block.form.streams && block.split.is_some()) {
            let rows = prepared.region_start(places);
            (0..super::PROMPT_ROUTE_SLOTS)
                .map(|_| {
                    let blocks = prepared
                        .served
                        .iter()
                        .map(|block| {
                            if block.form.streams {
                                block
                                    .split
                                    .as_ref()
                                    .map(|layer| {
                                        let (element, extents) = layer.at_rows(rows).crossing.routes;
                                        seismic::Tensor::shared(charges, element, &extents)
                                            .map_err(AllocationError::Tensor)
                                    })
                                    .transpose()
                            } else {
                                Ok(None)
                            }
                        })
                        .collect::<Result<Vec<_>, AllocationError>>()?;
                    Ok(PromptRoutesSlot {
                        in_use: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        blocks,
                    })
                })
                .collect::<Result<Vec<_>, AllocationError>>()?
        } else {
            Vec::new()
        };
        let allocate = |value: &Option<Extents>| value.as_ref().map(|value| zeros(charges, value)).transpose();
        let operands = match &extents.operand_carry {
            None => Vec::new(),
            Some(([rows, quantized], carry_places)) => (0..*carry_places)
                .map(|_| {
                    let pair = || Ok::<_, AllocationError>([zeros(charges, rows)?, zeros(charges, quantized)?]);
                    Ok::<_, AllocationError>([pair()?, pair()?])
                })
                .collect::<Result<_, _>>()?,
        };
        let picks_slots = extents
            .picks
            .iter()
            .map(|ext| zeros(charges, ext))
            .collect::<Result<Vec<_>, _>>()?;
        let pool_slots = extents
            .pool
            .iter()
            .map(|ext| zeros(charges, ext))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            places,
            taps: allocate(&extents.taps)?,
            per_layer_rows: allocate(&extents.per_layer)?,
            token_rows: allocate(&extents.tokens)?,
            kept_guesses: allocate(&extents.guesses)?,
            split,
            blocks,
            prompt_routes,
            operands,
            picks_slots,
            picks_by_read: extents.picks_by_read,
            pool_slots,
            pool_by_read: extents.pool_by_read,
        })
    }

    /// The shared VRAM picks buffer and `top` width of history read `read`.
    pub(crate) fn picks(&self, read: usize) -> Result<(&seismic::Tensor, u64), String> {
        let (slot, top) = self
            .picks_by_read
            .get(read)
            .and_then(|entry| *entry)
            .ok_or_else(|| format!("history read {read} has no VRAM picks table"))?;
        let tensor = self
            .picks_slots
            .get(slot)
            .ok_or_else(|| format!("history read {read} picks slot {slot} out of bounds"))?;
        Ok((tensor, top))
    }

    /// The shared VRAM candidate pool buffer and `pool_blocks` of history read `read`.
    pub(crate) fn pool(&self, read: usize) -> Result<(&seismic::Tensor, u64), String> {
        let (slot, pool_blocks) = self
            .pool_by_read
            .get(read)
            .and_then(|entry| *entry)
            .ok_or_else(|| format!("history read {read} has no VRAM candidate pool table"))?;
        let tensor = self
            .pool_slots
            .get(slot)
            .ok_or_else(|| format!("history read {read} pool slot {slot} out of bounds"))?;
        Ok((tensor, pool_blocks))
    }

    /// The two operand carries of place `place`'s run (`Self::operands`).
    pub(crate) fn operands(&self, place: usize) -> Result<&[[seismic::Tensor; 2]; 2], String> {
        self.operands.get(place).ok_or_else(|| format!("place {place} of a group holds no operand carries"))
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

    /// Acquire an exclusive host-visible per-layer route buffer slot for a
    /// grouped prefill chain, if one is configured and free.
    pub(crate) fn acquire_prompt_routes(&self) -> Option<PromptRoutesLease> {
        for slot in &self.prompt_routes {
            if slot
                .in_use
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_ok()
            {
                return Some(PromptRoutesLease {
                    in_use: Arc::clone(&slot.in_use),
                    blocks: slot.blocks.clone(),
                });
            }
        }
        None
    }

    /// The buffers the group owns.
    fn storage(&self) -> impl Iterator<Item = &seismic::Tensor> {
        let split = self.split.iter().flat_map(|split| split.crossing.iter().chain([&split.merged.inverse]));
        let prompt_routes = self.prompt_routes.iter().flat_map(|slot| slot.blocks.iter().flatten());
        let operands = self.operands.iter().flatten().flatten();
        let picks = self.picks_slots.iter();
        let pool = self.pool_slots.iter();
        self.taps
            .iter()
            .chain(&self.per_layer_rows)
            .chain(&self.token_rows)
            .chain(&self.kept_guesses)
            .chain(split)
            .chain(prompt_routes)
            .chain(operands)
            .chain(picks)
            .chain(pool)
    }
}

impl BoundTargetGraphs {
    /// The buffers of a chain of `launches` launches: the floor's alone,
    /// the held group's otherwise, held with the class whose places bound
    /// a chain (`ExecutorDomain::group_places`).
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
    pub(crate) fn hold_group(&self, charges: &seismic::Charges, places: std::num::NonZeroUsize) -> Result<(), AllocationError> {
        let held = (places.get() > 1)
            .then(|| GroupBuffers::new(&self.prepared, charges, places.get(), self.pass.as_ref()).map(Rc::new))
            .transpose()?;
        *self.held.borrow_mut() = held;
        Ok(())
    }

    /// The floor's buffers alone held again: a group's released.
    pub(crate) fn release_group(&self) {
        *self.held.borrow_mut() = None;
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
            .chain(self.block_attn_res.iter().flat_map(|bar| bar.constant_tensors()))
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

/// Block `index`'s `graphs` of one kind, bound once per distinct `key(class)`
/// to the resident weights, bank parts and constants it reads statically, in
/// ascending key order with exact vector capacity. Classes sharing a binding
/// key are verified compatible with the shared static binding.
pub(super) fn bind_block_graphs<K: Copy + std::fmt::Debug, B: Ord + Copy + std::fmt::Debug, P>(
    graphs: &[(K, Arc<PreparedTargetBlockGraph<P>>)],
    index: usize,
    resident: &ResidentTarget,
    charges: &seismic::Charges,
    constants: &mut ConstantTensors,
    key: impl Fn(K) -> B,
) -> Result<Vec<(B, BoundNativeGraphPlan)>, String> {
    let mut by_key: BTreeMap<B, BoundNativeGraphPlan> = BTreeMap::new();
    for (class, graph) in graphs {
        let bkey = key(*class);
        if let Some(existing) = by_key.get(&bkey) {
            existing.check_compatible(&graph.plan).map_err(|error| {
                format!(
                    "target graph static binding compatibility class {class:?} key {bkey:?} block {index}: {error}"
                )
            })?;
            continue;
        }
        let constant_tensors = graph
            .constants
            .iter()
            .map(|constant| Ok((constant.port(), constants.tensor(constant)?)))
            .collect::<Result<Vec<_>, String>>()?;
        let fixed = graph
            .weights
            .iter()
            .map(|(weight, port)| {
                // A graph shared by equal-shape blocks names its roles
                // relative to its block; each block binds its own.
                let role = WeightRole {
                    scope: block_scope(weight.role.scope, index)?,
                    kind: weight.role.kind,
                };
                Ok((port, resident.bound(role, weight.part)?))
            })
            // Placed and windowed bank parts are bound per run or per window.
            .chain(graph.banks.iter().filter(|(bank, _)| !bank.part.placed() && !bank.part.windowed()).map(|(bank, port)| {
                let scope = block_scope(bank.scope, index)?;
                Ok((port, resident.bank_part(BankPort { scope, ..*bank })?))
            }))
            .chain(constant_tensors.iter().map(|(port, tensor)| Ok((*port, tensor))))
            .collect::<Result<Vec<_>, String>>()?;
        let bound_plan = graph.plan.bind_static(charges, &fixed).map_err(|error| {
            format!("target graph static binding class {class:?} block {index}: {error}")
        })?;
        by_key.insert(bkey, bound_plan);
    }
    let mut bound = Vec::with_capacity(by_key.len());
    bound.extend(by_key);
    Ok(bound)
}

/// A block's bound graphs, deduplicated by `(rows, Topology)` for run and
/// untapped graphs and keyed by `rows` for passes: a vector of exact capacity
/// for each kind, matching what the plan counted (`block_binding_storage`).
pub(super) struct BlockBindings {
    pub(super) runs: Vec<(BindingKey, BoundNativeGraphPlan)>,
    pub(super) passes: Vec<(u64, BoundNativeGraphPlan)>,
    pub(super) untapped: Vec<(BindingKey, BoundNativeGraphPlan)>,
}

/// The bound per-layer entry of a per-layer model.
pub(crate) struct BoundPerLayerEntry {
    /// The graph per row class, bound to its weights.
    pub(super) bound: BTreeMap<u64, BoundNativeGraphPlan>,
    /// The table whose rows each step gathers, streamed: the resident
    /// target's.
    pub table: std::rc::Rc<StreamedTable>,
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
        if let Some(bar) = &self.prepared.block_attn_res {
            let scratch = super::block_attn_res_graph::block_attn_res_runtime_scratch_bytes(
                self.prepared.passes.region,
                bar.hidden,
                bar.schedule.max_sources,
                bar.schedule.max_bank,
            );
            bytes = bytes.and_then(|b| b.checked_add(scratch));
        }
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
        let keeps = *self
            .prepared
            .keeps
            .get(index)
            .ok_or_else(|| format!("target block {index} does not exist"))?;
        let axes = self
            .prepared
            .stage_axes
            .of(index, crate::programs::graph::RowForm::of(rows), stage);
        let key = (
            axes.project(block_class(keeps, rows, segments, slots, listed, scored)),
            Stage::Run(stage),
        );
        match super::classes::uses_untapped(
            self.prepared.block_taps(index).positions().any(),
            rows,
            self.prepared.untapped_rows,
            tapped,
        ) {
            false => self.staged(index, key),
            true => sealed_and_bound(
                &self.prepared.blocks_by_class[index].untapped,
                &self.bound[index].untapped,
                index,
                key,
                binding_key(key),
            ),
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
    ) -> Result<(&PreparedTargetBlockGraph<PassGraph>, &BoundNativeGraphPlan, u64), String> {
        let rows = self.prepared.passes.of_chain(launches);
        let (graph, bound) = sealed_and_bound(
            &self.prepared.blocks_by_class[index].passes,
            &self.bound[index].passes,
            index,
            rows,
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
        sealed_and_bound(
            &self.prepared.blocks_by_class[index].runs,
            &self.bound[index].runs,
            index,
            key,
            binding_key(key),
        )
    }
}

/// Block `index`'s graph of `key` among its sealed `graphs` of one kind, and
/// its static binding of `bkey` among their deduplicated `bound`.
fn sealed_and_bound<'g, K: Ord + std::fmt::Debug, B: Ord + std::fmt::Debug, P>(
    graphs: &'g [(K, Arc<PreparedTargetBlockGraph<P>>)],
    bound: &'g [(B, BoundNativeGraphPlan)],
    index: usize,
    key: K,
    bkey: B,
) -> Result<(&'g PreparedTargetBlockGraph<P>, &'g BoundNativeGraphPlan), String> {
    let graph = graphs
        .binary_search_by(|(class, _)| class.cmp(&key))
        .map(|at| graphs[at].1.as_ref())
        .map_err(|_| format!("target graph {key:?} of block {index} was not sealed"))?;
    let bound = bound
        .binary_search_by(|(class, _)| class.cmp(&bkey))
        .map(|at| &bound[at].1)
        .map_err(|_| format!("target graph binding {bkey:?} of block {index} was not bound"))?;
    Ok((graph, bound))
}
