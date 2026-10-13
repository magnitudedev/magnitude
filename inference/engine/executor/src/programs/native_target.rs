//! Ordered target execution through sealed Seismic native graphs. A step
//! queues one run per graph without waiting; the device queue orders them,
//! and the returned submission owns the launch until completion is observed.

use super::{
    DeviceSubmission, TargetProgram,
    graph::readout::{
        BoundTargetReadoutGraphs, ReadoutClass, ReadoutKind, certifies, shapes, write_selection,
    },
    native_target_graph::{BoundTargetGraphs, EntryTokens, TokenRow},
};
use crate::device_table::SelectedRead;
use crate::host_tables::HostTableError;
use crate::stored_table::{NoticeStep, StepRows, StoredTable};
use crate::operators::block::{BlockControlPorts, BlockStatePorts};
use crate::{
    ConditioningRef, ConditioningSlice, DeviceError, GraphOutputTensor, InvariantError,
    NativeGraphOutputLease, NativeGraphWorkspaceLease, SubmitError, TargetGraphOutputLease,
    TargetGraphWorkspaceLease, TargetLaunchCore, TargetLaunchWorkspace, TargetTokens,
    ValidatedTargetLaunch, completion::CompletionWaiter, native::AttestedState,
};
use magnitude_batching::{Demand, TargetBatchUpload};
use magnitude_family_contracts::Decoder;
use magnitude_kernels::{conditioning_overlay, feature_copy};
use magnitude_state::LayerRef;
use seismic::{
    Device, Element, NativeGraphCompletion, NativeGraphOutputs, NativeGraphPlan,
    NativeGraphSequence, NativePort, Tensor,
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::Arc, time::Instant};

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native target program",
        detail: detail.into(),
    })
}

fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}

/// Host tokens as entry input rows: the `sample_rows` result layout
/// (token, status), status 0.
fn token_rows(tokens: &[i32]) -> Vec<u8> {
    tokens
        .iter()
        .flat_map(|token| [*token, 0])
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A launch's encoded visible spans, `segments` a row, with each row's
/// padded to `padded` spans by empty ones.
fn padded_spans(visible: &[u8], segments: u64, padded: u64) -> Vec<u8> {
    let (row, padded) = (segments as usize * 8, padded as usize * 8);
    visible
        .chunks_exact(row)
        .flat_map(|spans| {
            spans
                .iter()
                .copied()
                .chain(std::iter::repeat_n(0, padded - row))
        })
        .collect()
}

/// Whether a selected read scores in a launch whose rows read the block-key
/// history `blocks` over `segments` spans a row: whether some row has more
/// than `selected` complete blocks, the block rows it sees and its own when
/// it completes one (`indexer_select`). A launch where none does reads every
/// row's whole history through the class that does not score.
fn selection_scores(blocks: &HistoryControls, segments: u64, selected: u64) -> bool {
    let word = |bytes: &[u8], at: usize| {
        i64::from(i32::from_le_bytes([
            bytes[at],
            bytes[at + 1],
            bytes[at + 2],
            bytes[at + 3],
        ]))
    };
    let row = (segments as usize * 8).max(1);
    blocks
        .visible
        .chunks_exact(row)
        .zip(blocks.destinations.chunks_exact(4))
        .any(|(spans, destination)| {
            let seen = spans
                .chunks_exact(8)
                .map(|span| word(span, 4) - word(span, 0))
                .sum::<i64>();
            let own = i64::from(word(destination, 0) >= 0);
            seen + own > selected as i64
        })
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Observe every run, even after a failure, so none is still executing when
/// the caller regains the storage the runs use. The first failure is reported.
fn wait_all(completions: Vec<NativeGraphCompletion>) -> Result<(), SubmitError> {
    completions.into_iter().fold(Ok(()), |outcome, completion| {
        outcome.and(completion.wait().map_err(device))
    })
}

/// Submits a step's runs in sequences of doubling length (1, 2, 4, … runs):
/// the device starts on the first run as soon as it is queued, and each
/// sequence is queued while the device executes the previous ones, so a step
/// of N runs has about log2(N) submission boundaries instead of N.
struct StepSubmitter {
    device: Device,
    sequence: NativeGraphSequence,
    queued: usize,
    limit: usize,
    completions: Vec<NativeGraphCompletion>,
    /// The stored-table reads of the runs queued since the last
    /// submission, then of the submitted runs.
    pending_reads: Vec<StoredRead>,
    reads: Vec<StoredRead>,
}

/// A run's read of a stored table: the table and the step it queued there.
#[derive(Clone)]
pub struct StoredRead {
    table: Arc<StoredTable>,
    value: u64,
}

impl StoredRead {
    /// A read of `table`'s step `value`.
    pub fn new(table: Arc<StoredTable>, value: u64) -> Self {
        Self { table, value }
    }
}

impl StepSubmitter {
    fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            sequence: device.native_sequence(),
            queued: 0,
            limit: 1,
            completions: Vec::new(),
            pending_reads: Vec::new(),
            reads: Vec::new(),
        }
    }

    /// The next run queued reads a stored table at its step `value`.
    fn reads(&mut self, table: Arc<StoredTable>, value: u64) {
        self.pending_reads.push(StoredRead { table, value });
    }

    /// After a failure, once the submitted runs completed: the stored-table
    /// steps of runs never submitted read nothing.
    fn abandon_unsubmitted(self) {
        for read in self.pending_reads {
            let _ = read.table.abandon(read.value);
        }
    }

    fn queue(
        &mut self,
        ready: seismic::ReadyNativeGraphRun<'_>,
    ) -> Result<NativeGraphOutputs, SubmitError> {
        let outputs = ready.queue(&mut self.sequence).map_err(device)?;
        self.queued += 1;
        if self.queued == self.limit {
            self.flush()?;
            self.limit *= 2;
        }
        Ok(outputs)
    }

    /// Submit what is queued.
    fn flush(&mut self) -> Result<(), SubmitError> {
        self.queued = 0;
        let sequence = std::mem::replace(&mut self.sequence, self.device.native_sequence());
        if !sequence.is_empty() {
            self.completions.push(sequence.submit().map_err(device)?);
            self.reads.append(&mut self.pending_reads);
        }
        Ok(())
    }
}

/// Diagnostic output, chosen once when the program is bound.
#[derive(Clone, Copy)]
struct TargetTrace {
    batch: bool,
    blocks: bool,
}

impl TargetTrace {
    fn from_environment() -> Self {
        Self {
            batch: std::env::var_os("MAGNITUDE_TRACE_TARGET").is_some(),
            blocks: std::env::var_os("MAGNITUDE_TRACE_TARGET_BLOCKS").is_some(),
        }
    }
}

/// A sealed single-range conditioning overlay for one row count. The source
/// and the destination hidden rows are late bindings, so a plan serves every
/// placement of that many rows.
#[derive(Clone)]
struct OverlayGraph {
    plan: NativeGraphPlan,
    source: NativePort,
    destination: NativePort,
}

#[derive(Clone)]
pub struct NativeTargetProgram {
    device: Device,
    state: AttestedState,
    geometry: Decoder,
    graphs: Rc<BoundTargetGraphs>,
    readout_graphs: Rc<BoundTargetReadoutGraphs>,
    /// Overlay plans by row count, sealed on first use.
    overlays: Rc<RefCell<BTreeMap<u64, OverlayGraph>>>,
    boundary_copy: Rc<RefCell<Option<OverlayGraph>>>,
    waiter: CompletionWaiter,
    trace: TargetTrace,
}

/// The launch whose rows a step runs: its advances' state and conditioning.
struct RowState<'a>(&'a TargetLaunchCore);

pub struct TargetReadoutGraphResult {
    pub features: GraphOutputTensor,
    pub logits: Option<GraphOutputTensor>,
    pub selected: Option<GraphOutputTensor>,
    /// Indices into the feature output rows, in projected logits order.
    pub projected_output_rows: Vec<usize>,
}

/// Host instants at which a step's first and last command buffers were
/// committed to the device queue.
#[derive(Clone, Copy, Debug)]
pub struct CommitSpan {
    pub first: Instant,
    pub last: Instant,
}

/// A target step's readout and the commit span of its device work.
pub struct TargetOutput {
    pub boundary: Option<crate::FeatureRef>,
    pub readout: Option<TargetReadoutGraphResult>,
    pub commits: CommitSpan,
    /// The step's read of the table rows the device selected its tokens
    /// for; `None` of a step of the host's tokens.
    pub selected_rows: Option<SelectedRead>,
    /// The step's reads of stored tables (`StoredTable`).
    pub stored_reads: Vec<StoredRead>,
}

impl TargetOutput {
    /// Whether a step queued ahead of the device's selection found every
    /// device table row it read for `token`, the token then selected: a
    /// token's embedding row is the token. (A stored table's rows are those
    /// the host wrote before the step's gate opened, the selected token's;
    /// a slot whose rows could not be read fails, `table_read_failure`.)
    pub fn found_selected(&self, token: u32) -> bool {
        self.selected_rows
            .as_ref()
            .is_none_or(|rows| rows.found(u64::from(token)))
    }

    /// Why a stored table's rows batch slot `slot` named could not be read,
    /// once the step completed: that slot's outputs are not to be used.
    pub fn table_read_failure(&self, slot: usize) -> Option<crate::TableReadError> {
        self.stored_reads.iter().find_map(|read| {
            read.table
                .failure(read.value)
                .filter(|failure| failure.fails(slot))
                .map(|failure| crate::TableReadError {
                    detail: failure.error,
                })
        })
    }

    /// The step's outcome was taken: its stored tables forget its failures.
    pub fn retire_stored_reads(&self) {
        for read in &self.stored_reads {
            read.table.retire(read.value);
        }
    }
}

/// Runs of one step queued on the device, with the outputs they will fill.
struct QueuedStep {
    readout: Option<TargetReadoutGraphResult>,
    commits: CommitSpan,
    completions: Vec<NativeGraphCompletion>,
    selected_rows: Option<SelectedRead>,
    stored_reads: Vec<StoredRead>,
}

impl RowState<'_> {
    fn slots(&self) -> usize {
        self.0.advances().len()
    }
    fn tokens(&self) -> &TargetTokens {
        self.0.tokens()
    }
    fn preceding(&self) -> &[Vec<u64>] {
        self.0.preceding()
    }
    /// The state store's recurrent arenas, one per component. Every slot's
    /// advance belongs to the same store; the batch's bank columns select
    /// the rows each slot reads and publishes.
    fn recurrent_arenas(&self) -> Result<&[Tensor], SubmitError> {
        self.0
            .advances()
            .first()
            .map(|advance| advance.bindings().recurrent)
            .ok_or_else(|| invalid("recurrent batch has no state advance"))
    }
    /// `layer`'s history read (`StateStore::history_read`, the index of its
    /// controls) and planes in the codec's plane-descriptor order, as the
    /// attention graph's state ports take them. A Shared layer binds its
    /// source layer's planes and its source domain's shared read.
    fn history(&self, layer: LayerRef) -> Result<(usize, Vec<&Tensor>), SubmitError> {
        let store = self.0.store();
        let missing = || invalid("attention layer has no history in its store");
        let source = store.history_source(layer).ok_or_else(missing)?;
        let read = store.history_read(layer).ok_or_else(missing)?;
        let planes = self
            .0
            .advances()
            .first()
            .map(|advance| advance.bindings().history)
            .ok_or_else(|| invalid("attention batch has no state advance"))?;
        Ok((
            read,
            planes
                .iter()
                .filter(|plane| plane.domain == source.domain && plane.layer == source.layer)
                .map(|plane| &plane.buffer)
                .collect(),
        ))
    }
    /// Target block `block`'s indexer histories: its raw keys' and its
    /// block keys' reads, each with its planes in descriptor order.
    fn indexer(&self, block: u32) -> Result<[(usize, Vec<&Tensor>); 2], SubmitError> {
        let store = self.0.store();
        let (window, blocks) = store
            .indexer_domains(block)
            .ok_or_else(|| invalid("a selected attention layer has no indexer history"))?;
        let planes = self
            .0
            .advances()
            .first()
            .map(|advance| advance.bindings().history)
            .ok_or_else(|| invalid("attention batch has no state advance"))?;
        let layer = LayerRef::Indexer(block);
        let of = |domain: magnitude_state::HistoryDomainId| {
            (
                domain.0,
                planes
                    .iter()
                    .filter(|plane| plane.domain == domain && plane.layer == layer)
                    .map(|plane| &plane.buffer)
                    .collect(),
            )
        };
        Ok([of(window), of(blocks)])
    }
    fn conditioning(&self, slot: usize) -> Option<&ConditioningRef> {
        self.0.conditioning().get(slot).and_then(Option::as_ref)
    }
    fn conditioning_slices(&self, slot: usize) -> &[ConditioningSlice] {
        self.0
            .conditioning_slices()
            .get(slot)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

impl NativeTargetProgram {
    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    pub(crate) fn new(
        device: Device,
        state: AttestedState,
        geometry: Decoder,
        graphs: BoundTargetGraphs,
        readout_graphs: BoundTargetReadoutGraphs,
    ) -> Result<Self, SubmitError> {
        let waiter = CompletionWaiter::spawn()
            .map_err(|error| SubmitError::Device(DeviceError::Execution(error)))?;
        Ok(Self {
            device,
            state,
            geometry,
            graphs: Rc::new(graphs),
            readout_graphs: Rc::new(readout_graphs),
            overlays: Rc::new(RefCell::new(BTreeMap::new())),
            boundary_copy: Rc::default(),
            waiter,
            trace: TargetTrace::from_environment(),
        })
    }

    /// The readout over `hidden`, the step's final rows from batch row
    /// `window.first` on (`RowWindow`).
    fn graph_readout(
        &self,
        batch: &TargetBatchUpload<'_>,
        window: RowWindow,
        hidden: &Tensor,
        residual: Option<&Tensor>,
        workspace: &mut NativeGraphWorkspaceLease,
        mut output: NativeGraphOutputLease,
        submitter: &mut StepSubmitter,
    ) -> Result<Option<TargetReadoutGraphResult>, SubmitError> {
        let actual_outputs = batch.out_rows.len();
        if actual_outputs == 0 {
            return Ok(None);
        }
        let output_class = magnitude_batching::row_class(actual_outputs)
            .ok_or_else(|| invalid("readout output rows have no class"))?;
        let actual_selected = batch.select_rows.len();
        let selected_class = if actual_selected == 0 {
            0
        } else {
            magnitude_batching::row_class(actual_selected)
                .ok_or_else(|| invalid("readout selected rows have no class"))?
        };
        // Projected outputs, the selected ones first and in selection order:
        // shaping and sampling read the leading selected logits rows.
        let mut projected_output_rows = Vec::new();
        // Whether a selected row also reads its logits on the host.
        let mut selection_reads_logits = false;
        for &output_index in batch.select_rows {
            projected_output_rows.push(
                usize::try_from(output_index)
                    .map_err(|_| invalid("negative selection output row"))?,
            );
        }
        for (output_index, &row) in batch.out_rows.iter().enumerate() {
            let row = usize::try_from(row).map_err(|_| invalid("negative readout row"))?;
            let demand = batch
                .demand
                .get(row)
                .and_then(|bits| Demand::from_bits(*bits))
                .ok_or_else(|| invalid("readout demand is absent or invalid"))?;
            let selected = projected_output_rows[..actual_selected].contains(&output_index);
            selection_reads_logits |= selected && demand.contains(Demand::LOGITS);
            if demand.computes_logits() && !selected {
                projected_output_rows.push(output_index);
            } else if selected && !demand.computes_logits() {
                return Err(invalid("selection row has no projected logits"));
            }
        }
        let actual_projected = projected_output_rows.len();
        if actual_projected > self.readout_graphs.prepared.max_projected_rows() {
            return Err(invalid(format!(
                "readout requests {actual_projected} logits rows; admitted maximum is {}",
                self.readout_graphs.prepared.max_projected_rows()
            )));
        }
        let projected_class = if actual_projected == 0 {
            0
        } else {
            magnitude_batching::row_class(actual_projected)
                .ok_or_else(|| invalid("readout projected rows have no class"))?
        };
        // A selection the certified readout serves: every selected row
        // samples or takes its largest admitted logit at a temperature at
        // most, none reads its logits, and nothing else projects.
        let certified = actual_projected == actual_selected
            && self
                .readout_graphs
                .prepared
                .certifies(projected_class as u64)
            && !selection_reads_logits
            && batch.shaping[..actual_selected].iter().all(certifies);
        let kind = if actual_selected > 0 {
            ReadoutKind::Selection {
                shaped: batch.shaping[..actual_selected].iter().any(shapes),
                certified,
            }
        } else if actual_projected > 0 {
            ReadoutKind::Logits
        } else {
            ReadoutKind::Features
        };
        let class = ReadoutClass {
            rows: window.rows,
            outputs: output_class as u64,
            projected: projected_class as u64,
            selected: selected_class as u64,
            kind,
        };
        let (graph, bound) = self.readout_graphs.class(class).map_err(invalid)?;
        let mut active = workspace.slot_mut().activate(&graph.plan).map_err(device)?;
        let tap_rows = graph
            .taps
            .as_ref()
            .map(|_| self.tap_rows(class.rows))
            .transpose()?;
        let mut bindings = bound.bindings();
        if let Some(port) = &graph.residual {
            bindings.set(port, residual.ok_or_else(|| invalid("residual conditioning was not exported"))?).map_err(device)?;
        }
        if let Some(rows) = &graph.final_rows {
            bindings.set(&rows.hidden, hidden).map_err(device)?;
        }
        if let (Some(ports), Some(tap_rows)) = (&graph.taps, &tap_rows) {
            bindings.set(&ports.taps, tap_rows).map_err(device)?;
        }
        let mut out_rows = vec![0_i32; output_class];
        for (index, &row) in batch.out_rows.iter().enumerate() {
            out_rows[index] = window.row(row)?;
        }
        active
            .write_input(&graph.out_rows, &i32_bytes(&out_rows))
            .map_err(device)?;
        if let Some(port) = &graph.logit_rows {
            // The head gathers the hidden rows of the projected outputs.
            let mut rows = vec![0_i32; projected_class];
            for (index, &output_index) in projected_output_rows.iter().enumerate() {
                rows[index] = window.row(
                    *batch
                        .out_rows
                        .get(output_index)
                        .ok_or_else(|| invalid("projected output row exceeds output rows"))?,
                )?;
            }
            active
                .write_input(port, &i32_bytes(&rows))
                .map_err(device)?;
        }
        if let Some(ports) = &graph.selection {
            write_selection(batch, &mut active, ports, selected_class, batch.mask_words)?;
        }
        let ready = active
            .attach(
                bindings,
                output
                    .activate(&graph.plan)
                    .map_err(SubmitError::Invariant)?,
            )
            .map_err(device)?;
        let outputs = submitter.queue(ready)?;
        let owner = output.publish(outputs);
        let features = owner
            .tensor(&graph.features)
            .ok_or_else(|| invalid("readout features were not exported"))?;
        let logits = graph
            .logits
            .as_ref()
            .map(|edge| {
                owner
                    .tensor(edge)
                    .ok_or_else(|| invalid("readout logits were not exported"))
            })
            .transpose()?;
        let selected = graph
            .selected
            .as_ref()
            .map(|edge| {
                owner
                    .tensor(edge)
                    .ok_or_else(|| invalid("readout selection was not exported"))
            })
            .transpose()?;
        Ok(Some(TargetReadoutGraphResult {
            features,
            logits,
            selected,
            projected_output_rows,
        }))
    }

    /// The leading `rows` draft input rows of a tapped target.
    fn tap_rows(&self, rows: u64) -> Result<Tensor, SubmitError> {
        self.graphs
            .taps
            .as_ref()
            .ok_or_else(|| invalid("a tapped graph has no bound draft input rows"))?
            .slice_leading(0, rows)
            .map_err(device)
    }

    fn boundary_copy(&self) -> Result<OverlayGraph, SubmitError> {
        if let Some(graph) = self.boundary_copy.borrow().as_ref() {
            return Ok(graph.clone());
        }
        let (element, width) = self.readout_graphs.prepared.residual_width.map_or((Element::dense(crate::planning::activation_dtype(self.geometry.activation_dtype)), self.geometry.hidden), |width| (Element::f32(), width));
        let mut graph = self.device.native_graph();
        let source = graph
            .port(element, &[1, width])
            .map_err(device)?;
        let mut destination = graph
            .port(element, &[1, width])
            .map_err(device)?;
        graph
            .enqueue(
                &self.state.feature_copy,
                feature_copy::WorkflowArgs {
                    input: source.tensor().into(),
                    out: destination.tensor_mut().into(),
                },
            )
            .map_err(device)?;
        let plan = graph.seal().map_err(device)?;
        if plan.workspace_bytes() != 0 || plan.output_bytes() != 0 || plan.upload_bytes() != 0 {
            return Err(invalid(
                "boundary capture unexpectedly requires graph storage",
            ));
        }
        let graph = OverlayGraph {
            plan,
            source,
            destination,
        };
        *self.boundary_copy.borrow_mut() = Some(graph.clone());
        Ok(graph)
    }

    fn overlay(&self, rows: u64) -> Result<OverlayGraph, SubmitError> {
        if let Some(overlay) = self.overlays.borrow().get(&rows) {
            return Ok(overlay.clone());
        }
        let kernel = self
            .state
            .conditioning
            .as_ref()
            .ok_or_else(|| invalid("conditioning program slot is absent"))?;
        let mut graph = self.device.native_graph();
        let source = graph
            .port(Element::f32(), &[rows, self.geometry.hidden])
            .map_err(device)?;
        let mut destination = graph
            .port(Element::f32(), &[rows, self.geometry.hidden])
            .map_err(device)?;
        graph
            .enqueue(
                kernel,
                conditioning_overlay::WorkflowArgs {
                    input: source.tensor().into(),
                    out: destination.tensor_mut().into(),
                },
            )
            .map_err(device)?;
        let plan = graph.seal().map_err(device)?;
        if plan.workspace_bytes() != 0 || plan.output_bytes() != 0 || plan.upload_bytes() != 0 {
            return Err(invalid(
                "conditioning overlay unexpectedly requires graph storage",
            ));
        }
        let overlay = OverlayGraph {
            plan,
            source,
            destination,
        };
        self.overlays.borrow_mut().insert(rows, overlay.clone());
        Ok(overlay)
    }

    /// Queue a per-layer model's per-layer entry over the step's embedded
    /// and overlaid `hidden` rows, and return the per-layer rows its blocks
    /// read. Each batch row gathers its token's table row, a media row the
    /// entry's media row.
    fn queue_per_layer_entry(
        &self,
        batch: &TargetBatchUpload<'_>,
        tokens: &TargetTokens,
        rows: u64,
        hidden: &Tensor,
        media_rows: &[std::ops::Range<usize>],
        graph_workspace: &mut TargetGraphWorkspaceLease,
        submitter: &mut StepSubmitter,
    ) -> Result<Option<Tensor>, SubmitError> {
        let Some(entry) = self.graphs.per_layer_entry(rows) else {
            return Ok(None);
        };
        let (graph, bound, per_layer) = entry.map_err(invalid)?;
        let TargetTokens::Host = tokens else {
            return Err(invalid(
                "a per-layer entry gathers table rows by host tokens",
            ));
        };
        let media_row = self
            .geometry
            .entry
            .per_layer
            .as_ref()
            .ok_or_else(|| invalid("a per-layer entry graph without its geometry"))?
            .media_row;
        let media_row =
            i64::try_from(media_row).map_err(|_| invalid("per-layer media row exceeds i64"))?;
        let table_rows = batch
            .tokens
            .iter()
            .enumerate()
            .map(|(row, token)| {
                if media_rows.iter().any(|range| range.contains(&row)) {
                    media_row
                } else {
                    i64::from(*token)
                }
            })
            .collect::<Vec<_>>();
        let mut inputs = per_layer.rows.slice_leading(0, rows).map_err(device)?;
        // The token whose per-layer row this step writes into a token
        // row's slot, and the slot.
        let mut written = None;
        {
            // The token rows' slots are the per-layer rows after the first,
            // the last row the first slot. This step's entry writes its
            // leading rows.
            let last = per_layer.rows.extents()[0] - 1;
            per_layer
                .token_rows
                .overwrite((last + 1 - rows) as usize..last as usize);
            if let ([token], true) = (batch.tokens, media_rows.is_empty()) {
                let slot = |slot: usize| {
                    let row = last - slot as u64;
                    per_layer.rows.slice_leading(row, row + 1).map_err(device)
                };
                let token = u64::try_from(*token)
                    .map_err(|_| invalid(format!("token {token} has no per-layer row")))?;
                match per_layer.token_rows.find(token).map_err(invalid)? {
                    TokenRow::Held(held) => return Ok(Some(slot(held)?)),
                    TokenRow::Free(free) => {
                        inputs = slot(free)?;
                        written = Some((token, free));
                    }
                    TokenRow::None => {}
                }
            }
        }
        let mut active = graph_workspace
            .slot_mut()
            .activate(&graph.plan)
            .map_err(device)?;
        per_layer
            .table
            .gather(&table_rows, |gathered| {
                active
                    .write_input(&graph.ports.gathered, gathered)
                    .map_err(device)
            })
            .map_err(|error: HostTableError| invalid(error.to_string()))??;
        let mut bindings = bound.bindings();
        bindings.set(&graph.ports.hidden, hidden).map_err(device)?;
        bindings.set(&graph.ports.rows, &inputs).map_err(device)?;
        let ready = active
            .attach(bindings, graph.plan.new_outputs().map_err(device)?)
            .map_err(device)?;
        submitter.queue(ready)?;
        if let Some((token, slot)) = written {
            per_layer.token_rows.hold(token, slot);
        }
        Ok(Some(inputs))
    }

    /// The rows block `index`'s stored table is read at in a step: its row
    /// rule over each slot's tokens and the tokens its sequence holds before
    /// them (`NGramHash::rows`), the rule the device evaluates
    /// (`ngram_rows`). Of host tokens they are known now; of selected
    /// tokens, the table's filler learns them from the step's notice.
    fn stored_step_rows(
        &self,
        index: usize,
        batch: &TargetBatchUpload<'_>,
        state: &RowState<'_>,
        notice: Option<&Arc<NoticeStep>>,
    ) -> Result<StepRows, SubmitError> {
        let hash = crate::operators::row_rules(&self.geometry)
            .into_iter()
            .find_map(|(block, hash)| (block == index).then(|| hash.clone()))
            .ok_or_else(|| invalid("a stored table's block has no row rule"))?;
        let preceding = state.preceding();
        if preceding.len() != state.slots() {
            return Err(invalid("the step's preceding tokens differ from its slots"));
        }
        match state.tokens() {
            TargetTokens::Host => {
                let mut rows = Vec::new();
                for (slot, preceding) in preceding.iter().enumerate() {
                    let [start, end] = batch.segments[slot];
                    let tokens = usize::try_from(start)
                        .ok()
                        .zip(usize::try_from(end).ok())
                        .and_then(|(start, end)| batch.tokens.get(start..end))
                        .ok_or_else(|| invalid("a slot's segment lies outside the batch"))?
                        .iter()
                        .map(|token| u64::try_from(*token))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| invalid("a host token is negative"))?;
                    rows.push(hash.rows(preceding, &tokens));
                }
                Ok(StepRows::Known(rows))
            }
            TargetTokens::Selected(_) => {
                let notice = notice
                    .cloned()
                    .ok_or_else(|| invalid("selected tokens without their notice"))?;
                let preceding = preceding.to_vec();
                Ok(StepRows::Selected(Box::new(move || {
                    // A slot's one row is its selection row; a failed
                    // selection's slot names none, as its step is never
                    // used.
                    let selected = notice.read()?;
                    Ok(preceding
                        .iter()
                        .zip(selected)
                        .map(|(preceding, [token, status])| match (u64::try_from(token), status) {
                            (Ok(token), 0) => hash.rows(preceding, &[token]),
                            _ => Vec::new(),
                        })
                        .collect())
                })))
            }
        }
    }

    /// The leading `rows` of the token rows a model whose blocks read the
    /// step's token rows keeps (`BoundTargetGraphs::token_rows`).
    fn kept_token_rows(&self, rows: u64) -> Result<Tensor, SubmitError> {
        self.graphs
            .token_rows
            .as_ref()
            .ok_or_else(|| invalid("the model keeps no token rows"))?
            .slice_leading(0, rows)
            .map_err(device)
    }

    /// Conditioning rows that replace embedding rows, as (source, first
    /// batch row, end batch row). Most steps have none.
    fn conditioning_sources(
        &self,
        state: &RowState<'_>,
        batch: &TargetBatchUpload<'_>,
    ) -> Result<Vec<(Tensor, usize, usize)>, SubmitError> {
        let mut sources = Vec::new();
        for slot in 0..state.slots() {
            let conditioning = state.conditioning(slot);
            let slices = state.conditioning_slices(slot);
            if conditioning.is_none() && slices.is_empty() {
                continue;
            }
            let [start, end] = batch.segments[slot];
            let start = usize::try_from(start)
                .map_err(|_| invalid("conditioning segment start is negative"))?;
            let end = usize::try_from(end)
                .map_err(|_| invalid("conditioning segment end is negative"))?;
            if let Some(conditioning) = conditioning {
                if end - start != conditioning.allocation().rows() {
                    return Err(invalid("conditioning rows differ from validated segment"));
                }
                for range in conditioning.allocation().ranges() {
                    let source = conditioning
                        .allocation()
                        .tensor()
                        .map_err(device)?
                        .slice_leading(range.destination.start as u64, range.destination.end as u64)
                        .map_err(device)?;
                    sources.push((
                        source,
                        start + range.destination.start,
                        start + range.destination.end,
                    ));
                }
            }
            for slice in slices {
                let source_end = slice
                    .source
                    .start
                    .checked_add(slice.source.count)
                    .ok_or_else(|| invalid("conditioning source span overflows"))?;
                let destination_start = start
                    .checked_add(slice.destination)
                    .ok_or_else(|| invalid("conditioning destination overflows"))?;
                let destination_end = destination_start
                    .checked_add(slice.source.count)
                    .ok_or_else(|| invalid("conditioning destination span overflows"))?;
                if destination_end > end {
                    return Err(invalid("conditioning slice exceeds validated slot"));
                }
                let source = slice
                    .source
                    .features
                    .allocation()
                    .tensor()
                    .map_err(device)?
                    .slice_leading(slice.source.start as u64, source_end as u64)
                    .map_err(device)?;
                sources.push((source, destination_start, destination_end));
            }
        }
        Ok(sources)
    }

    /// Queue every run of one step. On failure, runs already queued are
    /// observed before the caller regains the storage they use.
    fn queue_rows(
        &self,
        batch: &TargetBatchUpload<'_>,
        state: &RowState<'_>,
        readout: (&mut NativeGraphWorkspaceLease, NativeGraphOutputLease),
        graph_workspace: &mut TargetGraphWorkspaceLease,
        graph_outputs: &mut [TargetGraphOutputLease; 2],
    ) -> Result<QueuedStep, SubmitError> {
        // Device-selected tokens name rows of the embedding table, which
        // the device may not hold of a streamed one: the step's claim asks
        // (`TargetOutput::found_selected`).
        let selected_rows = matches!(state.tokens(), TargetTokens::Selected(_))
            .then(|| self.graphs.embedding.read_selected());
        let mut submitter = StepSubmitter::new(&self.device);
        let queued = self
            .queue_graphs(
                batch,
                state,
                readout,
                graph_workspace,
                graph_outputs,
                &mut submitter,
            )
            .and_then(|step| submitter.flush().map(|()| step));
        match queued {
            Ok((readout, commits)) => Ok(QueuedStep {
                readout,
                commits,
                completions: submitter.completions,
                selected_rows,
                stored_reads: submitter.reads,
            }),
            Err(error) => {
                // The failure is reported; outcomes of runs submitted before
                // it matter only for draining the device. Runs still queued
                // are dropped unsubmitted, and their stored-table steps
                // read nothing.
                let _ = wait_all(std::mem::take(&mut submitter.completions));
                submitter.abandon_unsubmitted();
                Err(error)
            }
        }
    }

    fn queue_graphs(
        &self,
        batch: &TargetBatchUpload<'_>,
        state: &RowState<'_>,
        readout: (&mut NativeGraphWorkspaceLease, NativeGraphOutputLease),
        graph_workspace: &mut TargetGraphWorkspaceLease,
        graph_outputs: &mut [TargetGraphOutputLease; 2],
        submitter: &mut StepSubmitter,
    ) -> Result<(Option<TargetReadoutGraphResult>, CommitSpan), SubmitError> {
        let rows =
            u64::try_from(batch.class.rows()).map_err(|_| invalid("row class exceeds u64"))?;
        let mut pending: [Option<NativeGraphOutputs>; 2] = [None, None];
        let mut residual = None;
        let tokens = state.tokens();
        let source = match tokens {
            TargetTokens::Host => EntryTokens::Uploaded,
            TargetTokens::Selected(_) => EntryTokens::Selected,
        };
        let (entry, entry_bound) = self.graphs.entry(rows, source).map_err(invalid)?;
        // A step reading selected tokens in a model with a stored table
        // copies them for the host first (`EntryNotice`).
        let mut notice = None;
        let entry_outputs = {
            let mut active = graph_workspace
                .slot_mut()
                .activate(&entry.plan)
                .map_err(device)?;
            let mut bindings = entry_bound.bindings();
            match tokens {
                TargetTokens::Host => {
                    let rows = self
                        .graphs
                        .embedding
                        .indices(batch.tokens)
                        .map_err(invalid)?;
                    active
                        .write_input(&entry.tokens, &token_rows(&rows))
                        .map_err(device)?;
                    // A model whose blocks read the step's token rows keeps
                    // the host's tokens in the bound token rows.
                    if let Some(kept) = &entry.kept_tokens {
                        active
                            .write_input(&kept.uploaded, &token_rows(batch.tokens))
                            .map_err(device)?;
                        bindings
                            .set(&kept.rows, &self.kept_token_rows(batch.tokens.len() as u64)?)
                            .map_err(device)?;
                    }
                }
                TargetTokens::Selected(selected) => bindings
                    .set(&entry.tokens, selected.tensor())
                    .map_err(device)?,
            }
            // Selected tokens are copied for the host, which learns from
            // them the stored tables' rows the step names.
            if let Some(ports) = &entry.notice {
                let step = self
                    .graphs
                    .notice
                    .as_ref()
                    .ok_or_else(|| invalid("an entry notice without a selection notice"))?
                    .next(rows)
                    .map_err(invalid)?;
                bindings.set(&ports.tokens, &step.tokens).map_err(device)?;
                bindings
                    .set_timeline(ports.signal, &step.timeline, step.value)
                    .map_err(device)?;
                notice = Some(Arc::new(step));
            }
            let ready = active
                .attach(
                    bindings,
                    graph_outputs[0]
                        .activate(&entry.plan)
                        .map_err(SubmitError::Invariant)?,
                )
                .map_err(device)?;
            submitter.queue(ready)?
        };
        // The device embeds while the host prepares the remaining runs.
        let first_commit = Instant::now();
        let segments = u64::try_from(batch.class.segments())
            .map_err(|_| invalid("segment class exceeds u64"))?;
        let slots = u64::try_from(batch.actual_slots)
            .map_err(|_| invalid("request slot class exceeds u64"))?;
        let mut controls = GraphControls::new(batch)?;
        let mut window = RowWindow::whole(rows);
        let mut hidden = entry_outputs
            .exported(&entry.hidden)
            .ok_or_else(|| invalid("target embedding hidden was not exported"))?;
        let conditioning = self.conditioning_sources(state, batch)?;
        let media_rows = conditioning
            .iter()
            .map(|(_, start, end)| *start..*end)
            .collect::<Vec<_>>();
        for (source, start, end) in conditioning {
            let overlay = self.overlay((end - start) as u64)?;
            let destination = hidden
                .slice_leading(start as u64, end as u64)
                .map_err(device)?;
            let mut bindings = overlay.plan.bindings();
            bindings.set(&overlay.source, &source).map_err(device)?;
            bindings
                .set(&overlay.destination, &destination)
                .map_err(device)?;
            let mut slot = overlay.plan.new_slot().map_err(device)?;
            let ready = slot
                .attach(bindings, overlay.plan.new_outputs().map_err(device)?)
                .map_err(device)?;
            submitter.queue(ready)?;
        }
        let mut per_layer_rows = self.queue_per_layer_entry(
            batch,
            tokens,
            rows,
            &hidden,
            &media_rows,
            graph_workspace,
            submitter,
        )?;
        pending[0] = Some(entry_outputs);
        let mut recurrent_component = 0usize;
        if self.trace.batch {
            eprintln!(
                "target batch actual_rows={} class_rows={} actual_slots={} outputs={} selections={}",
                batch.actual_rows,
                batch.class.rows(),
                batch.actual_slots,
                batch.out_rows.len(),
                batch.select_rows.len()
            );
            for (domain, history) in controls.histories.iter().enumerate() {
                let spans = history
                    .visible
                    .chunks_exact(8)
                    .map(|span| {
                        let bound = |at: usize| {
                            i32::from_le_bytes([span[at], span[at + 1], span[at + 2], span[at + 3]])
                        };
                        (bound(0), bound(4))
                    })
                    .collect::<Vec<_>>();
                // Every row of a decode launch; a prefill launch's first
                // and last.
                let per_row = spans.chunks((segments as usize).max(1)).collect::<Vec<_>>();
                let shown = match per_row.len() {
                    0..=16 => per_row,
                    rows => vec![per_row[0], per_row[rows - 1]],
                };
                eprintln!(
                    "target batch history domain={domain} segments={segments} spans={shown:?}"
                );
            }
        }
        // Blocks whose outputs the step reads; the last one's output feeds
        // the readout.
        let mut executed = self.geometry.blocks.len();
        for index in 0..self.geometry.blocks.len() {
            if index == self.graphs.prepared.stateless_tail() {
                // From here no block publishes state: run the rest over the
                // rows the readout consumes, or not at all without any.
                match RowWindow::consumed(batch.out_rows, rows)? {
                    None => {
                        executed = index;
                        break;
                    }
                    Some(tail) if tail.rows < rows => {
                        hidden = tail.slice(&hidden)?;
                        per_layer_rows = per_layer_rows
                            .map(|inputs| tail.slice(&inputs))
                            .transpose()?;
                        controls = controls.window(rows, tail);
                        window = tail;
                    }
                    Some(_) => {}
                }
            }
            let rows = window.rows;
            let block_started = self.trace.blocks.then(Instant::now);
            // A block with listing classes lists the history row tiles the
            // launch's rows see in the smallest class that holds them; a
            // launch that sees more than the largest (many requests with
            // long histories) takes the class that lists none.
            let listed_tiles = self
                .graphs
                .prepared
                .listed_tiles(index, rows)
                .iter()
                .find_map(|&tiles| {
                    let (read, _) = state.history(LayerRef::Target(index as u32)).ok()?;
                    let visible = &controls.histories.get(read)?.visible;
                    crate::operators::attention::graph::history_tile_bytes(
                        visible,
                        usize::try_from(tiles).ok()?,
                    )
                    .map(|bytes| (tiles, bytes))
                });
            // A listing class may be sealed for more segments than the
            // launch has: its rows' spans are then padded with empty ones.
            let block_segments = match &listed_tiles {
                Some(_) => self.graphs.prepared.listed_segments(index, segments),
                None => segments,
            };
            let scored = match self.graphs.prepared.selection(index) {
                None => false,
                Some(selection) => {
                    let [_, (block_read, _)] = state.indexer(index as u32)?;
                    let blocks = controls
                        .histories
                        .get(block_read)
                        .ok_or_else(|| invalid("the batch lacks an indexer's history read"))?;
                    selection_scores(blocks, segments, selection.blocks)
                }
            };
            let (graph, bound) = self
                .graphs
                .block(
                    rows,
                    block_segments,
                    slots,
                    listed_tiles.as_ref().map_or(0, |(tiles, _)| *tiles),
                    scored,
                    index,
                )
                .map_err(invalid)?;
            let parity = (index + 1) % 2;
            let mut active = graph_workspace
                .slot_mut()
                .activate(&graph.plan)
                .map_err(device)?;
            let tap_rows = graph
                .taps
                .as_ref()
                .map(|_| self.tap_rows(rows))
                .transpose()?;
            let mut bindings = bound.bindings();
            bindings.set(&graph.hidden, &hidden).map_err(device)?;
            match (&graph.per_layer, &per_layer_rows) {
                (None, _) => {}
                (Some(port), Some(inputs)) => bindings.set(port, inputs).map_err(device)?,
                (Some(_), None) => {
                    return Err(invalid("a per-layer block graph has no per-layer rows"));
                }
            }
            // An n-gram input reads the step's token rows: the host's, as
            // the entry kept them, or the previous step's selection.
            if let Some(ports) = &graph.ngram {
                match state.tokens() {
                    TargetTokens::Host => bindings
                        .set(&ports.tokens, &self.kept_token_rows(rows)?)
                        .map_err(device)?,
                    TargetTokens::Selected(selected) => bindings
                        .set(&ports.tokens, selected.tensor())
                        .map_err(device)?,
                }
                // Its table's rows: written by the host before the step's
                // gate opens, from the step's tokens and the ones before.
                let table = self
                    .graphs
                    .stored
                    .get(&index)
                    .ok_or_else(|| invalid("an n-gram block has no stored table"))?;
                let rows = self.stored_step_rows(index, batch, state, notice.as_ref())?;
                let step = table.queue(rows).map_err(invalid)?;
                submitter.reads(table.clone(), step.value);
                bindings
                    .set_timeline(ports.table.gate, &step.gate, step.value)
                    .map_err(device)?;
                bindings
                    .set_timeline(ports.table.read, &step.read, step.value)
                    .map_err(device)?;
            }
            if let (Some(ports), Some(tap_rows)) = (&graph.taps, &tap_rows) {
                let taps = self.graphs.prepared.block_taps(index);
                bindings.set(&ports.taps, tap_rows).map_err(device)?;
                for (port, tap) in [
                    (&ports.entry, taps.entry),
                    (&ports.middle, taps.middle),
                    (&ports.output, taps.output),
                ] {
                    match (port, tap) {
                        (Some(port), Some(tap)) => active
                            .write_input(port, &i32_bytes(&[tap as i32]))
                            .map_err(device)?,
                        (None, None) => {}
                        _ => return Err(invalid("a block's tap ports differ from its taps")),
                    }
                }
            }
            match (&graph.state, &graph.controls) {
                (
                    BlockStatePorts::Attention(ports, indexer_planes),
                    BlockControlPorts::Attention {
                        coordinates,
                        visible,
                        fresh,
                        destinations,
                        history_tiles,
                        indexer,
                    },
                ) => {
                    // A selected read's indexer: its planes, and its rows'
                    // spans and destinations in its two histories, written
                    // as the block's own read is.
                    match (indexer_planes, indexer) {
                        (Some(planes), Some(tables)) => {
                            let [(window_read, window), (block_read, blocks)] =
                                state.indexer(index as u32)?;
                            let [window_keys, window_coordinates] = window.as_slice() else {
                                return Err(invalid("an indexer's raw-key planes are not two"));
                            };
                            let [block_keys] = blocks.as_slice() else {
                                return Err(invalid("an indexer's block-key planes are not one"));
                            };
                            for (port, plane) in [
                                (&planes.window_keys, window_keys),
                                (&planes.window_coordinates, window_coordinates),
                                (&planes.block_keys, block_keys),
                            ] {
                                bindings.set(port, plane).map_err(device)?;
                            }
                            for (read, visible, destinations) in [
                                (
                                    window_read,
                                    &tables.window_visible,
                                    &tables.window_destinations,
                                ),
                                (block_read, &tables.block_visible, &tables.block_destinations),
                            ] {
                                let history = controls.histories.get(read).ok_or_else(|| {
                                    invalid("the batch lacks an indexer's history read")
                                })?;
                                active
                                    .write_input(visible, &history.visible)
                                    .map_err(device)?;
                                active
                                    .write_input(destinations, &history.destinations)
                                    .map_err(device)?;
                            }
                        }
                        (None, None) => {}
                        _ => return Err(invalid("an indexer's planes and tables disagree")),
                    }
                    let (read, history) = state.history(LayerRef::Target(index as u32))?;
                    if history.len() != ports.len() {
                        return Err(invalid(
                            "the layer's history planes disagree with its attention entry",
                        ));
                    }
                    for (port, plane) in ports.iter().zip(history) {
                        bindings.set(port, plane).map_err(device)?;
                    }
                    let history_controls = controls
                        .histories
                        .get(read)
                        .ok_or_else(|| invalid("the batch lacks the layer's history read"))?;
                    active
                        .write_input(coordinates, &controls.coordinates)
                        .map_err(device)?;
                    match block_segments == segments {
                        true => active.write_input(visible, &history_controls.visible),
                        false => active.write_input(
                            visible,
                            &padded_spans(&history_controls.visible, segments, block_segments),
                        ),
                    }
                    .map_err(device)?;
                    active
                        .write_input(fresh, &history_controls.fresh)
                        .map_err(device)?;
                    active
                        .write_input(destinations, &history_controls.destinations)
                        .map_err(device)?;
                    match (history_tiles, &listed_tiles) {
                        (Some(tiles), Some((_, bytes))) => {
                            active.write_input(&tiles.port, bytes).map_err(device)?
                        }
                        (None, None) => {}
                        _ => {
                            return Err(invalid(
                                "the block's graph class and its history tile list disagree",
                            ));
                        }
                    }
                }
                (BlockStatePorts::Recurrent(ports), BlockControlPorts::Recurrent(recurrent)) => {
                    let arenas = state.recurrent_arenas()?;
                    for (offset, port) in ports.iter().enumerate() {
                        bindings
                            .set(
                                port,
                                arenas
                                    .get(recurrent_component + offset)
                                    .ok_or_else(|| invalid("recurrent bank arena is absent"))?,
                            )
                            .map_err(device)?;
                    }
                    active
                        .write_input(&recurrent.segments, &controls.segments)
                        .map_err(device)?;
                    active
                        .write_input(&recurrent.stop, &controls.stop)
                        .map_err(device)?;
                    active
                        .write_input(&recurrent.previous_bank, &controls.previous_bank)
                        .map_err(device)?;
                    active
                        .write_input(&recurrent.previous_tape, &controls.previous_tape)
                        .map_err(device)?;
                    active
                        .write_input(&recurrent.following_bank, &controls.following_bank)
                        .map_err(device)?;
                }
                _ => return Err(invalid("sealed graph block state and controls disagree")),
            }
            let ready = active
                .attach(
                    bindings,
                    graph_outputs[parity]
                        .activate(&graph.plan)
                        .map_err(SubmitError::Invariant)?,
                )
                .map_err(device)?;
            let outputs = submitter.queue(ready)?;
            if let BlockStatePorts::Recurrent(ports) = &graph.state {
                recurrent_component += ports.len();
            }
            residual = graph.residual.as_ref().map(|edge| outputs.exported(edge).ok_or_else(|| invalid("target residual was not exported"))).transpose()?;
            hidden = outputs
                .exported(&graph.output)
                .ok_or_else(|| invalid("target block output was not exported"))?;
            let previous = index % 2;
            graph_outputs[previous]
                .recycle(
                    pending[previous]
                        .take()
                        .ok_or_else(|| invalid("prior target graph output is absent"))?,
                )
                .map_err(SubmitError::Invariant)?;
            pending[parity] = Some(outputs);
            if let Some(started) = block_started {
                let kind = match &graph.state {
                    BlockStatePorts::Attention(..) => "attention",
                    BlockStatePorts::Recurrent(_) => "recurrent",
                };
                eprintln!(
                    "target block {index} {kind} rows {rows} lists {} history tiles {:.3}s",
                    listed_tiles.as_ref().map_or(0, |(tiles, _)| *tiles),
                    started.elapsed().as_secs_f64()
                );
            }
        }
        let (readout_workspace, readout_output) = readout;
        let result = self.graph_readout(
            batch,
            window,
            &hidden,
            residual.as_ref(),
            readout_workspace,
            readout_output,
            submitter,
        )?;
        if let Some(boundary) = state.0.boundary() {
            let features = &result
                .as_ref()
                .ok_or_else(|| invalid("boundary capture has no features"))?
                .features;
            let row = batch.actual_rows - 1;
            let source = features
                .tensor()
                .slice_leading(row as u64, row as u64 + 1)
                .map_err(device)?;
            let destination = boundary.allocation().tensor().map_err(device)?;
            let copy = self.boundary_copy()?;
            let mut bindings = copy.plan.bindings();
            bindings.set(&copy.source, &source).map_err(device)?;
            bindings
                .set(&copy.destination, destination)
                .map_err(device)?;
            let mut slot = copy.plan.new_slot().map_err(device)?;
            let ready = slot
                .attach(bindings, copy.plan.new_outputs().map_err(device)?)
                .map_err(device)?;
            submitter.queue(ready)?;
        }
        let last_commit = Instant::now();
        // Every consumer of the final hidden rows is queued, so its output
        // slot returns to the lease; the device queue orders any reuse.
        drop(hidden);
        drop(residual);
        let last = executed % 2;
        graph_outputs[last]
            .recycle(
                pending[last]
                    .take()
                    .ok_or_else(|| invalid("final target graph output is absent"))?,
            )
            .map_err(SubmitError::Invariant)?;
        Ok((
            result,
            CommitSpan {
                first: first_commit,
                last: last_commit,
            },
        ))
    }
}

/// The batch rows a step's blocks run over: `rows` rows (a row class) from
/// batch row `first`. Every block before the stateless tail runs the whole
/// batch; the tail runs the smallest window that holds the rows the readout
/// consumes. A row's computation in a block reads only its own hidden,
/// per-layer and control rows and the history, so a window's rows compute
/// what they would in the whole batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RowWindow {
    first: u64,
    rows: u64,
}

impl RowWindow {
    fn whole(rows: u64) -> Self {
        Self { first: 0, rows }
    }

    /// The window of the row class covering `out_rows` within a batch of
    /// `rows` rows, placed at their first row unless that would pass the
    /// batch's end. `None` when no row is consumed.
    fn consumed(out_rows: &[i32], rows: u64) -> Result<Option<Self>, SubmitError> {
        let row = |row: &i32| {
            u64::try_from(*row)
                .ok()
                .filter(|row| *row < rows)
                .ok_or_else(|| invalid("readout row lies outside the batch"))
        };
        let rows_read = out_rows.iter().map(row).collect::<Result<Vec<_>, _>>()?;
        let (Some(&first), Some(&last)) = (rows_read.iter().min(), rows_read.iter().max()) else {
            return Ok(None);
        };
        let class = magnitude_batching::row_class((last - first + 1) as usize)
            .ok_or_else(|| invalid("consumed rows have no row class"))? as u64;
        Ok(Some(Self {
            first: first.min(rows - class),
            rows: class,
        }))
    }

    /// The window's rows of a tensor over the batch rows.
    fn slice(self, rows: &Tensor) -> Result<Tensor, SubmitError> {
        rows.slice_leading(self.first, self.first + self.rows)
            .map_err(device)
    }

    /// Batch row `row` as a row of the window.
    fn row(self, row: i32) -> Result<i32, SubmitError> {
        u64::try_from(row)
            .ok()
            .and_then(|row| row.checked_sub(self.first))
            .filter(|row| *row < self.rows)
            .map(|row| row as i32)
            .ok_or_else(|| invalid("readout row lies outside the step's row window"))
    }

    /// The window's rows of a per-row host table over `batch_rows` rows.
    fn rows_of(self, table: &[u8], batch_rows: u64) -> Vec<u8> {
        let stride = table.len() / batch_rows as usize;
        table[self.first as usize * stride..(self.first + self.rows) as usize * stride].to_vec()
    }
}

struct GraphControls {
    coordinates: Vec<u8>,
    /// One control set per history domain, in domain order: an attention
    /// block reads its own domain's.
    histories: Vec<HistoryControls>,
    segments: Vec<u8>,
    /// Rows after which each slot publishes its recurrent state; later rows
    /// are recorded on the successor bank's tape.
    stop: Vec<u8>,
    /// Each slot's read version: bank, and the tape rows that complete it.
    previous_bank: Vec<u8>,
    previous_tape: Vec<u8>,
    following_bank: Vec<u8>,
}

/// One history domain's attention controls.
struct HistoryControls {
    visible: Vec<u8>,
    fresh: Vec<u8>,
    destinations: Vec<u8>,
}

impl GraphControls {
    /// The per-row controls of `window`'s rows, of controls over
    /// `batch_rows` rows. The per-slot recurrent controls stay whole: only
    /// attention blocks run a window (`stateless_tail`).
    fn window(self, batch_rows: u64, window: RowWindow) -> Self {
        Self {
            coordinates: window.rows_of(&self.coordinates, batch_rows),
            histories: self
                .histories
                .iter()
                .map(|history| HistoryControls {
                    visible: window.rows_of(&history.visible, batch_rows),
                    fresh: window.rows_of(&history.fresh, batch_rows),
                    destinations: window.rows_of(&history.destinations, batch_rows),
                })
                .collect(),
            ..self
        }
    }

    fn new(batch: &TargetBatchUpload<'_>) -> Result<Self, SubmitError> {
        let rows = batch.class.rows();
        let segments = batch.class.segments();
        let slots = batch
            .segments
            .len()
            .checked_sub(1)
            .ok_or_else(|| invalid("batch has no terminal segment row"))?;
        if !slots.is_power_of_two()
            || batch.actual_slots > slots
            || batch.actual_rows > rows
            || batch.tokens.len() != rows
            || batch.coordinates.len() != rows
            || batch.histories.iter().any(|history| {
                history.visible.len() != rows
                    || history.fresh.len() != rows
                    || history.destinations.len() != rows
                    || history
                        .visible
                        .iter()
                        .any(|ranges| ranges.len() != segments)
            })
        {
            return Err(invalid(
                "validated batch does not match exact graph row, segment, and slot classes",
            ));
        }
        Ok(Self {
            coordinates: batch
                .coordinates
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            histories: batch
                .histories
                .iter()
                .map(|history| HistoryControls {
                    visible: history
                        .visible
                        .iter()
                        .flatten()
                        .flatten()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                    fresh: history
                        .fresh
                        .iter()
                        .flatten()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                    destinations: i32_bytes(&history.destinations),
                })
                .collect(),
            segments: batch.segments[..=batch.actual_slots]
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            stop: i32_bytes(&batch.stop[..batch.actual_slots]),
            previous_bank: i32_bytes(&batch.bank[..batch.actual_slots]),
            previous_tape: i32_bytes(&batch.previous_tape[..batch.actual_slots]),
            following_bank: i32_bytes(&batch.following_bank[..batch.actual_slots]),
        })
    }
}

impl TargetProgram for NativeTargetProgram {
    type Submission = DeviceSubmission<TargetLaunchCore, TargetLaunchWorkspace, TargetOutput>;

    fn submit(
        &mut self,
        mut launch: ValidatedTargetLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedTargetLaunch)> {
        let queued = {
            let (core, graph_workspace, graph_outputs, readout_workspace, readout_output) =
                launch.submission_parts_mut();
            let batch = core.batch().upload();
            self.queue_rows(
                &batch,
                &RowState(core),
                (
                    readout_workspace,
                    readout_output
                        .take()
                        .expect("readout output was reserved before submit"),
                ),
                graph_workspace,
                graph_outputs,
            )
        };
        let queued = match queued {
            Ok(queued) => queued,
            Err(error) => return Err((error, launch)),
        };
        let (core, workspace) = launch.into_submission_parts();
        let boundary = core.boundary().cloned();
        Ok(DeviceSubmission::new(
            self.waiter.completion(queued.completions),
            core,
            workspace,
            TargetOutput {
                boundary,
                readout: queued.readout,
                commits: queued.commits,
                selected_rows: queued.selected_rows,
                stored_reads: queued.stored_reads,
            },
        ))
    }
}

#[cfg(test)]
mod row_window_tests {
    use super::RowWindow;

    #[test]
    fn an_unread_step_runs_no_tail() {
        assert_eq!(RowWindow::consumed(&[], 512).unwrap(), None);
    }

    #[test]
    fn the_window_is_the_row_class_covering_the_consumed_rows() {
        // A prompt chunk's last row.
        let last = RowWindow::consumed(&[511], 512).unwrap().unwrap();
        assert_eq!(
            last,
            RowWindow {
                first: 511,
                rows: 1
            }
        );
        assert_eq!(last.row(511).unwrap(), 0);
        assert!(last.row(510).is_err());
        // Rows spanning 11 rows take the 16-row class, kept inside the batch.
        let spread = RowWindow::consumed(&[500, 510, 505], 512).unwrap().unwrap();
        assert_eq!(
            spread,
            RowWindow {
                first: 496,
                rows: 16
            }
        );
        assert_eq!(spread.row(510).unwrap(), 14);
        // Rows across the batch keep the whole batch.
        assert_eq!(
            RowWindow::consumed(&[0, 511], 512).unwrap().unwrap(),
            RowWindow::whole(512)
        );
        assert!(RowWindow::consumed(&[512], 512).is_err());
    }

    #[test]
    fn a_window_takes_its_rows_of_a_host_table() {
        let table = (0..8u8).flat_map(|row| [row, row]).collect::<Vec<_>>();
        let window = RowWindow { first: 4, rows: 2 };
        assert_eq!(window.rows_of(&table, 8), vec![4, 4, 5, 5]);
    }
}
