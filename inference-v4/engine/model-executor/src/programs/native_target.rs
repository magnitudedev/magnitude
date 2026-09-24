//! Ordered target execution through sealed Seismic native graphs.

use super::{
    ReadySubmission, TargetProgram,
    native_target_graph::{BlockControlPorts, BlockStatePorts, BoundTargetGraphs},
    native_target_readout_graph::{BoundTargetReadoutGraphs, ReadoutClass, ReadoutKind},
};
use crate::{
    ConditioningRef, ConditioningSlice, DeviceError, GraphOutputTensor, InvariantError,
    NativeGraphOutputLease, NativeGraphWorkspaceLease, SubmitError, TargetGraphOutputLease,
    TargetGraphWorkspaceLease, TargetLaunchCore, ValidatedTargetLaunch, native::AttestedState,
};
use magnitude_model_batching::{Demand, TargetBatchUpload};
use magnitude_model_contracts::{DecoderGeometry, MixerGeometry, RotarySemantics};
use magnitude_model_kernels::qwen_conditioning_overlay;
use magnitude_model_state::{LayerRef, OwnedRepairAdvance, PlaneBuffer, PlaneName, VectorKind};
use seismic::{Device, Element, NativeGraphOutputs, NativeGraphPlan, NativePort, Tensor};
use std::rc::Rc;

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

#[derive(Clone)]
pub struct NativeTargetProgram {
    device: Device,
    state: AttestedState,
    geometry: DecoderGeometry,
    graphs: Rc<BoundTargetGraphs>,
    readout_graphs: Rc<BoundTargetReadoutGraphs>,
    rotary_controls: Vec<Option<Vec<u8>>>,
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

struct PreparedConditioningOverlay {
    plan: NativeGraphPlan,
    destination: NativePort,
    sources: Vec<(NativePort, Tensor)>,
}

impl RowState<'_> {
    fn slots(&self) -> usize {
        match self {
            Self::Forward(core) => core.advances().len(),
            Self::Repair { .. } => 1,
        }
    }
    fn recurrent(&self, slot: usize) -> Result<(&[Tensor], &[Tensor]), SubmitError> {
        match self {
            Self::Forward(core) => {
                let advance = core
                    .advances()
                    .get(slot)
                    .ok_or_else(|| invalid("recurrent slot is absent"))?;
                let binding = advance.bindings();
                Ok((binding.previous, binding.following))
            }
            Self::Repair { advance, .. } if slot == 0 => {
                Ok((advance.previous(), advance.following()))
            }
            _ => Err(invalid("repair has only one recurrent slot")),
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
        let rotary_controls = geometry
            .blocks
            .iter()
            .map(|block| match &block.mixer {
                MixerGeometry::Attention(attention) => rotary_components(&attention.rotary)
                    .map(|(components, _)| Some(i32_bytes(&components))),
                MixerGeometry::Recurrent(_) => Ok(None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            device,
            state,
            geometry,
            graphs: Rc::new(graphs),
            readout_graphs: Rc::new(readout_graphs),
            rotary_controls,
        })
    }

    fn graph_readout(
        &self,
        batch: &TargetBatchUpload<'_>,
        hidden: &Tensor,
        workspace: &mut NativeGraphWorkspaceLease,
        mut output: NativeGraphOutputLease,
    ) -> Result<Option<TargetReadoutGraphResult>, SubmitError> {
        let actual_outputs = batch.out_rows.len();
        if actual_outputs == 0 {
            return Ok(None);
        }
        let output_class = actual_outputs
            .checked_next_power_of_two()
            .ok_or_else(|| invalid("readout output class overflows"))?;
        let actual_selected = batch.select_rows.len();
        let selected_class = if actual_selected == 0 {
            0
        } else {
            actual_selected
                .checked_next_power_of_two()
                .ok_or_else(|| invalid("readout selection class overflows"))?
        };
        let mut projected_output_rows = Vec::new();
        for (output_index, &row) in batch.out_rows.iter().enumerate() {
            let row = usize::try_from(row).map_err(|_| invalid("negative readout row"))?;
            let demand = batch
                .demand
                .get(row)
                .and_then(|bits| Demand::from_bits(*bits))
                .ok_or_else(|| invalid("readout demand is absent or invalid"))?;
            if demand.computes_logits() {
                projected_output_rows.push(output_index);
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
            actual_projected
                .checked_next_power_of_two()
                .ok_or_else(|| invalid("readout projected class overflows"))?
        };
        let kind = if actual_selected > 0 {
            ReadoutKind::Selection
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
            let mut rows = vec![0_i32; projected_class];
            for (index, &output_index) in projected_output_rows.iter().enumerate() {
                rows[index] = i32::try_from(output_index)
                    .map_err(|_| invalid("projected readout row exceeds i32"))?;
            }
            active
                .write_input(port, &i32_bytes(&rows))
                .map_err(device)?;
            let to_rows = (0..projected_class)
                .map(|index| {
                    i32::try_from(index).map_err(|_| invalid("projected destination exceeds i32"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            active
                .write_input(
                    graph
                        .logit_to_rows
                        .as_ref()
                        .ok_or_else(|| invalid("readout destination port is absent"))?,
                    &i32_bytes(&to_rows),
                )
                .map_err(device)?;
        }
        if kind == ReadoutKind::Selection {
            let mut selected_rows = vec![0_i32; selected_class];
            for (index, &output_index) in batch.select_rows.iter().enumerate() {
                let output_index = usize::try_from(output_index)
                    .map_err(|_| invalid("negative selection output row"))?;
                let projected_index = projected_output_rows
                    .iter()
                    .position(|&row| row == output_index)
                    .ok_or_else(|| invalid("selection row has no projected logits"))?;
                selected_rows[index] = i32::try_from(projected_index)
                    .map_err(|_| invalid("selection projected index exceeds i32"))?;
            }
            active
                .write_input(
                    graph
                        .select_rows
                        .as_ref()
                        .ok_or_else(|| invalid("readout selection port is absent"))?,
                    &i32_bytes(&selected_rows),
                )
                .map_err(device)?;
            let first_shape = *batch
                .shaping
                .first()
                .ok_or_else(|| invalid("selection shaping is absent"))?;
            let first_history = *batch
                .history
                .first()
                .ok_or_else(|| invalid("selection history is absent"))?;
            let first_draw = *batch
                .draws
                .first()
                .ok_or_else(|| invalid("selection draws are absent"))?;
            let mut shaping = vec![first_shape; selected_class];
            shaping[..actual_selected].copy_from_slice(&batch.shaping[..actual_selected]);
            let mut history = vec![first_history; selected_class];
            history[..actual_selected].copy_from_slice(&batch.history[..actual_selected]);
            let mut draws = vec![first_draw; selected_class];
            draws[..actual_selected].copy_from_slice(&batch.draws[..actual_selected]);
            let shaping_bytes = shaping
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            active
                .write_input(
                    graph
                        .shaping
                        .as_ref()
                        .ok_or_else(|| invalid("readout shaping port is absent"))?,
                    &shaping_bytes,
                )
                .map_err(device)?;
            active
                .write_input(
                    graph
                        .history
                        .as_ref()
                        .ok_or_else(|| invalid("readout history port is absent"))?,
                    &i32_bytes(&history.into_iter().flatten().collect::<Vec<_>>()),
                )
                .map_err(device)?;
            active
                .write_input(
                    graph
                        .draws
                        .as_ref()
                        .ok_or_else(|| invalid("readout draws port is absent"))?,
                    &draws
                        .iter()
                        .flatten()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .map_err(device)?;
            let mut masks = Vec::with_capacity(selected_class * batch.mask_words);
            for &mask_row in batch.mask_rows.iter().take(actual_selected) {
                if mask_row < 0 {
                    let mut mask = vec![u32::MAX; batch.mask_words];
                    let tail = self.geometry.vocabulary as usize % 32;
                    if tail != 0 {
                        mask[batch.mask_words - 1] = (1u32 << tail) - 1;
                    }
                    masks.extend(mask);
                } else {
                    masks.extend(
                        batch
                            .masks
                            .get(mask_row as usize)
                            .ok_or_else(|| invalid("selection mask is absent"))?,
                    );
                }
            }
            let first_mask = masks[..batch.mask_words].to_vec();
            while masks.len() < selected_class * batch.mask_words {
                masks.extend(&first_mask);
            }
            active
                .write_input(
                    graph
                        .mask
                        .as_ref()
                        .ok_or_else(|| invalid("readout mask port is absent"))?,
                    &masks
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .map_err(device)?;
        }
        let outputs = active
            .attach(
                bindings,
                output
                    .activate(&graph.plan)
                    .map_err(SubmitError::Invariant)?,
            )
            .and_then(super::run_graph)
            .map_err(device)?;
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

    fn prepare_conditioning_overlay(
        &self,
        state: &RowState<'_>,
        batch: &TargetBatchUpload<'_>,
    ) -> Result<Option<PreparedConditioningOverlay>, SubmitError> {
        let mut graph = self.device.native_graph();
        let mut destination = graph
            .port(
                Element::f32(),
                &[batch.class.rows() as u64, self.geometry.hidden],
            )
            .map_err(device)?;
        let mut sources = Vec::new();
        let mut append = |source: Tensor, start: usize, end: usize| -> Result<(), SubmitError> {
            let source_port = graph
                .port(
                    Element::f32(),
                    &[end as u64 - start as u64, self.geometry.hidden],
                )
                .map_err(device)?;
            let mut destination_view = destination.tensor().slice_leading(start as u64, end as u64);
            graph
                .enqueue(
                    self.state
                        .conditioning
                        .as_ref()
                        .ok_or_else(|| invalid("conditioning program slot is absent"))?,
                    qwen_conditioning_overlay::WorkflowArgs {
                        input: source_port.tensor().into(),
                        out: (&mut destination_view).into(),
                    },
                )
                .map_err(device)?;
            sources.push((source_port, source));
            Ok(())
        };
        for slot in 0..state.slots() {
            let [start, end] = batch.segments[slot];
            let start = usize::try_from(start)
                .map_err(|_| invalid("conditioning segment start is negative"))?;
            let end = usize::try_from(end)
                .map_err(|_| invalid("conditioning segment end is negative"))?;
            if let Some(conditioning) = state.conditioning(slot) {
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
                    append(
                        source,
                        start + range.destination.start,
                        start + range.destination.end,
                    )?;
                }
            }
            for slice in state.conditioning_slices(slot) {
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
                append(source, destination_start, destination_end)?;
            }
        }
        if sources.is_empty() {
            return Ok(None);
        }
        let plan = graph.seal().map_err(device)?;
        if plan.workspace_bytes() != 0 || plan.output_bytes() != 0 {
            return Err(invalid(
                "external conditioning overlay unexpectedly requires graph storage",
            ));
        }
        Ok(Some(PreparedConditioningOverlay {
            plan,
            destination,
            sources,
        }))
    }

    fn execute_graph_blocks(
        &self,
        batch: &TargetBatchUpload<'_>,
        state: &RowState<'_>,
        readout: Option<(&mut NativeGraphWorkspaceLease, NativeGraphOutputLease)>,
        graph_workspace: &mut TargetGraphWorkspaceLease,
        graph_outputs: &mut [TargetGraphOutputLease; 2],
    ) -> Result<Option<TargetReadoutGraphResult>, SubmitError> {
        let rows =
            u64::try_from(batch.class.rows()).map_err(|_| invalid("row class exceeds u64"))?;
        let segments = u64::try_from(batch.class.segments())
            .map_err(|_| invalid("segment class exceeds u64"))?;
        let controls = GraphControls::new(batch)?;
        let slots = u64::try_from(batch.actual_slots)
            .map_err(|_| invalid("request slot class exceeds u64"))?;
        let mut pending: [Option<NativeGraphOutputs>; 2] = [None, None];
        // Every run of the step is queued without waiting; the device queue
        // orders them, and their outcomes are checked once all are queued.
        let mut completions = Vec::new();
        let (entry, entry_bound) = self.graphs.entry(rows).map_err(invalid)?;
        let overlay = self.prepare_conditioning_overlay(state, batch)?;
        let mut overlay_slot = overlay
            .as_ref()
            .map(|prepared| prepared.plan.new_slot().map_err(device))
            .transpose()?;
        let entry_lease = graph_outputs[0]
            .activate(&entry.plan)
            .map_err(SubmitError::Invariant)?;
        let overlay_ready = if let Some(prepared) = &overlay {
            let mut bindings = prepared.plan.bindings();
            bindings
                .set_reserved_export(&prepared.destination, &entry_lease, &entry.hidden)
                .map_err(device)?;
            for (port, source) in &prepared.sources {
                bindings.set(port, source).map_err(device)?;
            }
            Some(
                overlay_slot
                    .as_mut()
                    .expect("overlay slot follows prepared overlay")
                    .attach(bindings, prepared.plan.new_outputs().map_err(device)?)
                    .map_err(device)?,
            )
        } else {
            None
        };
        let entry_outputs = {
            let mut active = graph_workspace
                .slot_mut()
                .activate(&entry.plan)
                .map_err(device)?;
            active
                .write_input(&entry.tokens, &i32_bytes(batch.tokens))
                .map_err(device)?;
            let (outputs, completion) = active
                .attach(entry_bound.bindings(), entry_lease)
                .and_then(|ready| ready.submit())
                .map_err(device)?;
            completions.push(completion);
            outputs
        };
        if let Some(ready) = overlay_ready {
            let (_, completion) = ready.submit().map_err(device)?;
            completions.push(completion);
        }
        let mut hidden = entry_outputs
            .exported(&entry.hidden)
            .ok_or_else(|| invalid("target embedding hidden was not exported"))?;
        pending[0] = Some(entry_outputs);
        let mut recurrent_component = 0usize;
        let trace_target = std::env::var_os("MAGNITUDE_TRACE_TARGET").is_some();
        if trace_target {
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
            let block_started = std::time::Instant::now();
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
            if let Some(source_rows) = &graph.routed_rows {
                let identity = (0..batch.class.rows())
                    .map(|row| i32::try_from(row).map_err(|_| invalid("routed row exceeds i32")))
                    .collect::<Result<Vec<_>, _>>()?;
                active
                    .write_input(source_rows, &i32_bytes(&identity))
                    .map_err(device)?;
            }
            match (&graph.state, &graph.controls, &geometry.mixer) {
                (
                    BlockStatePorts::Attention { key, value },
                    BlockControlPorts::Attention {
                        coordinates,
                        rotary,
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
                    let rotary_bytes = self.rotary_controls[index]
                        .as_ref()
                        .expect("attention rotary was checked at program construction");
                    active.write_input(rotary, rotary_bytes).map_err(device)?;
                }
                (
                    BlockStatePorts::Recurrent { slots: state_ports },
                    BlockControlPorts::Recurrent { segments },
                    MixerGeometry::Recurrent(_),
                ) => {
                    if state_ports.len() != state.slots() {
                        return Err(invalid(
                            "recurrent graph active slots differ from state transaction",
                        ));
                    }
                    for (slot, ports) in state_ports.iter().enumerate() {
                        let (previous, following) = state.recurrent(slot)?;
                        bindings
                            .set(
                                &ports.previous_window,
                                previous.get(recurrent_component).ok_or_else(|| {
                                    invalid("recurrent previous window is absent")
                                })?,
                            )
                            .map_err(device)?;
                        bindings
                            .set(
                                &ports.previous_delta,
                                previous
                                    .get(recurrent_component + 1)
                                    .ok_or_else(|| invalid("recurrent previous delta is absent"))?,
                            )
                            .map_err(device)?;
                        bindings
                            .set(
                                &ports.following_window,
                                following.get(recurrent_component).ok_or_else(|| {
                                    invalid("recurrent successor window is absent")
                                })?,
                            )
                            .map_err(device)?;
                        bindings
                            .set(
                                &ports.following_delta,
                                following.get(recurrent_component + 1).ok_or_else(|| {
                                    invalid("recurrent successor delta is absent")
                                })?,
                            )
                            .map_err(device)?;
                        let zero = 0i32.to_le_bytes();
                        let slot = i32::try_from(slot)
                            .map_err(|_| invalid("recurrent slot exceeds i32"))?
                            .to_le_bytes();
                        active
                            .write_input(&ports.window_from, &zero)
                            .map_err(device)?;
                        active
                            .write_input(&ports.window_to, &slot)
                            .map_err(device)?;
                        active
                            .write_input(&ports.delta_from, &zero)
                            .map_err(device)?;
                        active.write_input(&ports.delta_to, &slot).map_err(device)?;
                        active
                            .write_input(&ports.window_scatter_from, &slot)
                            .map_err(device)?;
                        active
                            .write_input(&ports.window_scatter_to, &zero)
                            .map_err(device)?;
                        active
                            .write_input(&ports.delta_scatter_from, &slot)
                            .map_err(device)?;
                        active
                            .write_input(&ports.delta_scatter_to, &zero)
                            .map_err(device)?;
                    }
                    active
                        .write_input(segments, &controls.segments)
                        .map_err(device)?;
                }
                _ => return Err(invalid("sealed graph block differs from decoder geometry")),
            }
            let (outputs, completion) = active
                .attach(
                    bindings,
                    graph_outputs[parity]
                        .activate(&graph.plan)
                        .map_err(SubmitError::Invariant)?,
                )
                .and_then(|ready| ready.submit())
                .map_err(device)?;
            completions.push(completion);
            if matches!(&graph.state, BlockStatePorts::Recurrent { .. }) {
                recurrent_component += 2;
            }
            let next_hidden = outputs
                .exported(&graph.output)
                .ok_or_else(|| invalid("target block output was not exported"))?;
            let previous_hidden = std::mem::replace(&mut hidden, next_hidden);
            drop(previous_hidden);
            let previous = index % 2;
            graph_outputs[previous]
                .recycle(
                    pending[previous]
                        .take()
                        .ok_or_else(|| invalid("prior target graph output is absent"))?,
                )
                .map_err(SubmitError::Invariant)?;
            pending[parity] = Some(outputs);
            if std::env::var_os("MAGNITUDE_TRACE_TARGET_BLOCKS").is_some() {
                let kind = match &geometry.mixer {
                    MixerGeometry::Attention(_) => "attention",
                    MixerGeometry::Recurrent(_) => "recurrent",
                };
                eprintln!(
                    "target block {index} {kind} {:.3}s",
                    block_started.elapsed().as_secs_f64()
                );
            }
        }
        let result = if let Some((readout_workspace, readout_output)) = readout {
            self.graph_readout(batch, &hidden, readout_workspace, readout_output)?
        } else {
            None
        };
        for completion in completions {
            completion.wait().map_err(device)?;
        }
        drop(hidden);
        let last = self.geometry.blocks.len() % 2;
        graph_outputs[last]
            .recycle(
                pending[last]
                    .take()
                    .ok_or_else(|| invalid("final target graph output is absent"))?,
            )
            .map_err(SubmitError::Invariant)?;
        Ok(result)
    }

    fn execute_rows(
        &self,
        batch: &TargetBatchUpload<'_>,
        state: &RowState<'_>,
        readout: Option<(&mut NativeGraphWorkspaceLease, NativeGraphOutputLease)>,
        graph_workspace: &mut TargetGraphWorkspaceLease,
        graph_outputs: &mut [TargetGraphOutputLease; 2],
    ) -> Result<Option<TargetReadoutGraphResult>, SubmitError> {
        self.execute_graph_blocks(batch, state, readout, graph_workspace, graph_outputs)
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
        self.execute_rows(
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
        )
        .map(|_| ())
    }
}

struct GraphControls {
    coordinates: Vec<u8>,
    visible: Vec<u8>,
    fresh: Vec<u8>,
    destinations: Vec<u8>,
    segments: Vec<u8>,
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
            coordinates: i32_bytes(
                &batch
                    .coordinates
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
            ),
            visible: i32_bytes(
                &batch
                    .visible
                    .iter()
                    .flatten()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
            ),
            fresh: i32_bytes(&batch.fresh.iter().flatten().copied().collect::<Vec<_>>()),
            destinations: i32_bytes(batch.destinations),
            segments: i32_bytes(
                &batch.segments[..=batch.actual_slots]
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
            ),
        })
    }
}

fn rotary_components(rotary: &RotarySemantics) -> Result<(Vec<i32>, f32), SubmitError> {
    match rotary {
        RotarySemantics::Interleaved {
            width,
            base,
            sections,
            axis_pattern,
        } => {
            let pairs = usize::try_from(width / 2)
                .map_err(|_| invalid("rotary width exceeds host domain"))?;
            let first = *axis_pattern
                .first()
                .ok_or_else(|| invalid("rotary axis pattern is empty"))?;
            let components = if axis_pattern.len() == 1 {
                vec![i32::from(first); pairs]
            } else if axis_pattern.len() == 3
                && sections.len() >= 3
                && sections[3..].iter().all(|section| *section == 0)
            {
                let axis_one_end = sections[1]
                    .checked_mul(3)
                    .ok_or_else(|| invalid("rotary section exceeds host domain"))?;
                let axis_two_end = sections[2]
                    .checked_mul(3)
                    .ok_or_else(|| invalid("rotary section exceeds host domain"))?;
                (0..pairs)
                    .map(|index| {
                        let index = index as u64;
                        let axis = if index % 3 == 1 && index < axis_one_end {
                            1
                        } else if index % 3 == 2 && index < axis_two_end {
                            2
                        } else {
                            0
                        };
                        i32::from(axis_pattern[axis])
                    })
                    .collect()
            } else {
                return Err(invalid("unsupported rotary axis and section mapping"));
            };
            Ok((components, *base as f32))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotary_components_interleave_axes_with_section_cutoffs() {
        let rotary = RotarySemantics::Interleaved {
            width: 14,
            base: 10_000.0,
            sections: vec![4, 2, 1, 0],
            axis_pattern: vec![0, 1, 2],
        };
        let (components, base) = rotary_components(&rotary).unwrap();
        assert_eq!(components, [0, 1, 2, 0, 1, 0, 0]);
        assert_eq!(base, 10_000.0);
    }
}

impl TargetProgram for NativeTargetProgram {
    type Submission = ReadySubmission<TargetLaunchCore, (), Option<TargetReadoutGraphResult>>;

    fn submit(
        &mut self,
        mut launch: ValidatedTargetLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedTargetLaunch)> {
        let result = {
            let (core, graph_workspace, graph_outputs, readout_workspace, readout_output) =
                launch.submission_parts_mut();
            let batch = core.batch().upload();
            self.execute_rows(
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
        let result = match result {
            Ok(result) => result,
            Err(error) => return Err((error, launch)),
        };
        let core = launch.into_submission_parts();
        Ok(ReadySubmission::new(core, (), result))
    }
}
