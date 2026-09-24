//! Ordered target execution through sealed Seismic native graphs. A step
//! queues one run per graph without waiting; the device queue orders them,
//! and the returned submission owns the launch until completion is observed.

use super::{
    DeviceSubmission, TargetProgram,
    native_target_graph::{BlockControlPorts, BlockStatePorts, BoundTargetGraphs},
    graph::readout::{BoundTargetReadoutGraphs, ReadoutClass, ReadoutKind, SelectionPorts, shapes},
};
use crate::{
    ConditioningRef, ConditioningSlice, DeviceError, GraphOutputTensor, InvariantError,
    NativeGraphOutputLease, NativeGraphWorkspaceLease, SubmitError, TargetGraphOutputLease,
    TargetGraphWorkspaceLease, TargetLaunchCore, TargetLaunchWorkspace, ValidatedTargetLaunch,
    completion::CompletionWaiter, native::AttestedState,
};
use magnitude_model_batching::{Demand, TargetBatchUpload};
use magnitude_model_contracts::{DecoderGeometry, MixerGeometry};
use magnitude_model_kernels::qwen_conditioning_overlay;
use magnitude_model_state::{LayerRef, OwnedRepairAdvance, PlaneBuffer, PlaneName, VectorKind};
use seismic::{
    Device, Element, NativeGraphCompletion, NativeGraphOutputs, NativeGraphPlan,
    NativeGraphSequence, NativePort, Tensor,
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc, time::Instant};

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native target program",
        detail: detail.into(),
    })
}

fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
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
    completions
        .into_iter()
        .fold(Ok(()), |outcome, completion| {
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
}

impl StepSubmitter {
    fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            sequence: device.native_sequence(),
            queued: 0,
            limit: 1,
            completions: Vec::new(),
        }
    }

    fn queue(&mut self, ready: seismic::ReadyNativeGraphRun<'_>) -> Result<NativeGraphOutputs, SubmitError> {
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
    geometry: DecoderGeometry,
    graphs: Rc<BoundTargetGraphs>,
    readout_graphs: Rc<BoundTargetReadoutGraphs>,
    /// Overlay plans by row count, sealed on first use.
    overlays: Rc<RefCell<BTreeMap<u64, OverlayGraph>>>,
    waiter: CompletionWaiter,
    trace: TargetTrace,
}

enum RowState<'a> {
    Forward(&'a TargetLaunchCore),
    Repair {
        advance: &'a OwnedRepairAdvance,
        history: &'a [PlaneBuffer],
        conditioning: Option<&'a ConditioningRef>,
        conditioning_slices: &'a [ConditioningSlice],
    },
}

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
    pub readout: Option<TargetReadoutGraphResult>,
    pub commits: CommitSpan,
}

/// Runs of one step queued on the device, with the outputs they will fill.
struct QueuedStep {
    readout: Option<TargetReadoutGraphResult>,
    commits: CommitSpan,
    completions: Vec<NativeGraphCompletion>,
}

impl RowState<'_> {
    fn slots(&self) -> usize {
        match self {
            Self::Forward(core) => core.advances().len(),
            Self::Repair { .. } => 1,
        }
    }
    /// The state store's recurrent arenas, one per component. Every slot's
    /// advance belongs to the same store; the batch's bank columns select
    /// the rows each slot reads and publishes.
    fn recurrent_arenas(&self) -> Result<&[Tensor], SubmitError> {
        match self {
            Self::Forward(core) => core
                .advances()
                .first()
                .map(|advance| advance.bindings().recurrent)
                .ok_or_else(|| invalid("recurrent batch has no state advance")),
            Self::Repair { advance, .. } => Ok(advance.recurrent()),
        }
    }
    fn history(&self, layer: LayerRef, vector: VectorKind) -> Result<Tensor, SubmitError> {
        let plane = match self {
            Self::Forward(core) => core.advances().iter().find_map(|advance| {
                let binding = advance.bindings();
                binding
                    .history
                    .iter()
                    .find(|plane| {
                        plane.layer == layer
                            && plane.vector == vector
                            && plane.name == PlaneName::Dense
                    })
                    .cloned()
            }),
            Self::Repair { history, .. } => history
                .iter()
                .find(|plane| {
                    plane.layer == layer && plane.vector == vector && plane.name == PlaneName::Dense
                })
                .cloned(),
        };
        plane
            .map(|plane| plane.buffer)
            .ok_or_else(|| invalid("attested history plane is absent"))
    }
    fn conditioning(&self, slot: usize) -> Option<&ConditioningRef> {
        match self {
            Self::Forward(core) => core.conditioning().get(slot).and_then(Option::as_ref),
            Self::Repair { conditioning, .. } if slot == 0 => *conditioning,
            _ => None,
        }
    }
    fn conditioning_slices(&self, slot: usize) -> &[ConditioningSlice] {
        match self {
            Self::Forward(core) => core
                .conditioning_slices()
                .get(slot)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            Self::Repair {
                conditioning_slices,
                ..
            } => conditioning_slices,
        }
    }
}

impl NativeTargetProgram {
    pub(crate) fn new(
        device: Device,
        state: AttestedState,
        geometry: DecoderGeometry,
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
            waiter,
            trace: TargetTrace::from_environment(),
        })
    }

    fn graph_readout(
        &self,
        batch: &TargetBatchUpload<'_>,
        hidden: &Tensor,
        workspace: &mut NativeGraphWorkspaceLease,
        mut output: NativeGraphOutputLease,
        submitter: &mut StepSubmitter,
    ) -> Result<Option<TargetReadoutGraphResult>, SubmitError> {
        let actual_outputs = batch.out_rows.len();
        if actual_outputs == 0 {
            return Ok(None);
        }
        let output_class = magnitude_model_batching::row_class(actual_outputs)
            .ok_or_else(|| invalid("readout output rows have no class"))?;
        let actual_selected = batch.select_rows.len();
        let selected_class = if actual_selected == 0 {
            0
        } else {
            magnitude_model_batching::row_class(actual_selected)
                .ok_or_else(|| invalid("readout selected rows have no class"))?
        };
        // Projected outputs, the selected ones first and in selection order:
        // shaping and sampling read the leading selected logits rows.
        let mut projected_output_rows = Vec::new();
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
            magnitude_model_batching::row_class(actual_projected)
                .ok_or_else(|| invalid("readout projected rows have no class"))?
        };
        let kind = if actual_selected > 0 {
            ReadoutKind::Selection {
                shaped: batch.shaping[..actual_selected].iter().any(shapes),
            }
        } else if actual_projected > 0 {
            ReadoutKind::Logits
        } else {
            ReadoutKind::Features
        };
        let class = ReadoutClass {
            rows: batch.class.rows() as u64,
            outputs: output_class as u64,
            projected: projected_class as u64,
            selected: selected_class as u64,
            kind,
        };
        let (graph, bound) = self.readout_graphs.class(class).map_err(invalid)?;
        let mut active = workspace.slot_mut().activate(&graph.plan).map_err(device)?;
        let mut bindings = bound.bindings();
        bindings.set(&graph.hidden, hidden).map_err(device)?;
        let mut out_rows = vec![0_i32; output_class];
        out_rows[..actual_outputs].copy_from_slice(batch.out_rows);
        active
            .write_input(&graph.out_rows, &i32_bytes(&out_rows))
            .map_err(device)?;
        if let Some(port) = &graph.logit_rows {
            // The head gathers the hidden rows of the projected outputs.
            let mut rows = vec![0_i32; projected_class];
            for (index, &output_index) in projected_output_rows.iter().enumerate() {
                rows[index] = *batch
                    .out_rows
                    .get(output_index)
                    .ok_or_else(|| invalid("projected output row exceeds output rows"))?;
            }
            active
                .write_input(port, &i32_bytes(&rows))
                .map_err(device)?;
        }
        if let Some(ports) = &graph.selection {
            self.write_selection(batch, &mut active, ports, selected_class)?;
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

    /// Selection controls of the padded selection class. Padding rows repeat
    /// the first selected row. An unconstrained row carries only its flag:
    /// the mask input is written only when some row is constrained, and then
    /// holds zeros in the rows that are not.
    fn write_selection(
        &self,
        batch: &TargetBatchUpload<'_>,
        active: &mut seismic::NativeGraphFamilyActive<'_>,
        ports: &SelectionPorts,
        selected_class: usize,
    ) -> Result<(), SubmitError> {
        let actual_selected = batch.select_rows.len();
        let source = |index: usize| if index < actual_selected { index } else { 0 };
        if let Some(shaping) = &ports.shaping {
            let parameters = (0..selected_class)
                .flat_map(|index| batch.shaping[source(index)])
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            active
                .write_input(&shaping.parameters, &parameters)
                .map_err(device)?;
            let history = (0..selected_class)
                .flat_map(|index| batch.history[source(index)])
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>();
            active
                .write_input(&shaping.history, &history)
                .map_err(device)?;
        }
        let draws = (0..selected_class)
            .flat_map(|index| batch.draws[source(index)])
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        active.write_input(&ports.draws, &draws).map_err(device)?;
        let mask_rows = (0..selected_class)
            .map(|index| batch.mask_rows[source(index)])
            .collect::<Vec<_>>();
        let constrained = mask_rows
            .iter()
            .map(|&mask_row| i32::from(mask_row >= 0))
            .collect::<Vec<_>>();
        active
            .write_input(&ports.constrained, &i32_bytes(&constrained))
            .map_err(device)?;
        if mask_rows.iter().all(|&mask_row| mask_row < 0) {
            return Ok(());
        }
        let row_words = batch.mask_words;
        let mut masks = vec![0_u32; selected_class * row_words];
        for (index, &mask_row) in mask_rows.iter().enumerate() {
            let Ok(mask_row) = usize::try_from(mask_row) else {
                continue;
            };
            let mask = batch
                .masks
                .get(mask_row)
                .ok_or_else(|| invalid("selection mask is absent"))?;
            if mask.len() != row_words {
                return Err(invalid("selection mask width differs from the vocabulary"));
            }
            masks[index * row_words..(index + 1) * row_words].copy_from_slice(mask);
        }
        let bytes = masks
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        active.write_input(&ports.mask, &bytes).map_err(device)
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
                qwen_conditioning_overlay::WorkflowArgs {
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
        readout: Option<(&mut NativeGraphWorkspaceLease, NativeGraphOutputLease)>,
        graph_workspace: &mut TargetGraphWorkspaceLease,
        graph_outputs: &mut [TargetGraphOutputLease; 2],
    ) -> Result<QueuedStep, SubmitError> {
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
            }),
            Err(error) => {
                // The failure is reported; outcomes of runs submitted before
                // it matter only for draining the device. Runs still queued
                // are dropped unsubmitted.
                let _ = wait_all(submitter.completions);
                Err(error)
            }
        }
    }

    fn queue_graphs(
        &self,
        batch: &TargetBatchUpload<'_>,
        state: &RowState<'_>,
        readout: Option<(&mut NativeGraphWorkspaceLease, NativeGraphOutputLease)>,
        graph_workspace: &mut TargetGraphWorkspaceLease,
        graph_outputs: &mut [TargetGraphOutputLease; 2],
        submitter: &mut StepSubmitter,
    ) -> Result<(Option<TargetReadoutGraphResult>, CommitSpan), SubmitError> {
        let rows =
            u64::try_from(batch.class.rows()).map_err(|_| invalid("row class exceeds u64"))?;
        let mut pending: [Option<NativeGraphOutputs>; 2] = [None, None];
        let (entry, entry_bound) = self.graphs.entry(rows).map_err(invalid)?;
        let entry_outputs = {
            let mut active = graph_workspace
                .slot_mut()
                .activate(&entry.plan)
                .map_err(device)?;
            active
                .write_input(&entry.tokens, &i32_bytes(batch.tokens))
                .map_err(device)?;
            let ready = active
                .attach(
                    entry_bound.bindings(),
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
        let controls = GraphControls::new(batch)?;
        let mut hidden = entry_outputs
            .exported(&entry.hidden)
            .ok_or_else(|| invalid("target embedding hidden was not exported"))?;
        for (source, start, end) in self.conditioning_sources(state, batch)? {
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
        }
        for (index, geometry) in self.geometry.blocks.iter().enumerate() {
            let block_started = self.trace.blocks.then(Instant::now);
            let (graph, bound) = self
                .graphs
                .block(rows, segments, slots, index)
                .map_err(invalid)?;
            let parity = (index + 1) % 2;
            let mut active = graph_workspace
                .slot_mut()
                .activate(&graph.plan)
                .map_err(device)?;
            let mut bindings = bound.bindings();
            bindings.set(&graph.hidden, &hidden).map_err(device)?;
            match (&graph.state, &graph.controls, &geometry.mixer) {
                (
                    BlockStatePorts::Attention { key, value },
                    BlockControlPorts::Attention {
                        coordinates,
                        visible,
                        fresh,
                        destinations,
                    },
                    MixerGeometry::Attention(_),
                ) => {
                    let history_key =
                        state.history(LayerRef::Target(index as u32), VectorKind::Key)?;
                    let history_value =
                        state.history(LayerRef::Target(index as u32), VectorKind::Value)?;
                    bindings.set(key, &history_key).map_err(device)?;
                    bindings.set(value, &history_value).map_err(device)?;
                    active
                        .write_input(coordinates, &controls.coordinates)
                        .map_err(device)?;
                    active
                        .write_input(visible, &controls.visible)
                        .map_err(device)?;
                    active.write_input(fresh, &controls.fresh).map_err(device)?;
                    active
                        .write_input(destinations, &controls.destinations)
                        .map_err(device)?;
                }
                (
                    BlockStatePorts::Recurrent(ports),
                    BlockControlPorts::Recurrent(recurrent),
                    MixerGeometry::Recurrent(_),
                ) => {
                    let arenas = state.recurrent_arenas()?;
                    bindings
                        .set(
                            &ports.window,
                            arenas
                                .get(recurrent_component)
                                .ok_or_else(|| invalid("recurrent window arena is absent"))?,
                        )
                        .map_err(device)?;
                    bindings
                        .set(
                            &ports.delta,
                            arenas
                                .get(recurrent_component + 1)
                                .ok_or_else(|| invalid("recurrent delta arena is absent"))?,
                        )
                        .map_err(device)?;
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
                        .write_input(&recurrent.following_bank, &controls.following_bank)
                        .map_err(device)?;
                }
                _ => return Err(invalid("sealed graph block differs from decoder geometry")),
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
            if matches!(&graph.state, BlockStatePorts::Recurrent(_)) {
                recurrent_component += 2;
            }
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
                let kind = match &geometry.mixer {
                    MixerGeometry::Attention(_) => "attention",
                    MixerGeometry::Recurrent(_) => "recurrent",
                };
                eprintln!(
                    "target block {index} {kind} {:.3}s",
                    started.elapsed().as_secs_f64()
                );
            }
        }
        let result = match readout {
            Some((readout_workspace, readout_output)) => self.graph_readout(
                batch,
                &hidden,
                readout_workspace,
                readout_output,
                submitter,
            )?,
            None => None,
        };
        let last_commit = Instant::now();
        // Every consumer of the final hidden rows is queued, so its output
        // slot returns to the lease; the device queue orders any reuse.
        drop(hidden);
        let last = self.geometry.blocks.len() % 2;
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

    pub(crate) fn execute_repair(
        &self,
        replay: &magnitude_model_batching::ValidatedTargetBatch,
        advance: &OwnedRepairAdvance,
        history: &[PlaneBuffer],
        conditioning: Option<&ConditioningRef>,
        conditioning_slices: &[ConditioningSlice],
        graph_workspace: &mut TargetGraphWorkspaceLease,
        graph_outputs: &mut [TargetGraphOutputLease; 2],
    ) -> Result<(), SubmitError> {
        let batch = replay.upload();
        let queued = self.queue_rows(
            &batch,
            &RowState::Repair {
                advance,
                history,
                conditioning,
                conditioning_slices,
            },
            None,
            graph_workspace,
            graph_outputs,
        )?;
        wait_all(queued.completions)
    }
}

struct GraphControls {
    coordinates: Vec<u8>,
    visible: Vec<u8>,
    fresh: Vec<u8>,
    destinations: Vec<u8>,
    segments: Vec<u8>,
    /// Rows after which each slot publishes its recurrent state: all of them.
    stop: Vec<u8>,
    previous_bank: Vec<u8>,
    following_bank: Vec<u8>,
}

impl GraphControls {
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
            || batch.visible.len() != rows
            || batch.fresh.len() != rows
            || batch.destinations.len() != rows
            || batch.visible.iter().any(|ranges| ranges.len() != segments)
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
            visible: batch
                .visible
                .iter()
                .flatten()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            fresh: batch
                .fresh
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            destinations: i32_bytes(batch.destinations),
            segments: batch.segments[..=batch.actual_slots]
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            stop: batch.segments[..batch.actual_slots]
                .iter()
                .flat_map(|[start, end]| (end - start).to_le_bytes())
                .collect(),
            previous_bank: i32_bytes(&batch.bank[..batch.actual_slots]),
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
                &RowState::Forward(core),
                Some((
                    readout_workspace,
                    readout_output
                        .take()
                        .expect("readout output was reserved before submit"),
                )),
                graph_workspace,
                graph_outputs,
            )
        };
        let queued = match queued {
            Ok(queued) => queued,
            Err(error) => return Err((error, launch)),
        };
        let (core, workspace) = launch.into_submission_parts();
        Ok(DeviceSubmission::new(
            self.waiter.completion(queued.completions),
            core,
            workspace,
            TargetOutput {
                readout: queued.readout,
                commits: queued.commits,
            },
        ))
    }
}
