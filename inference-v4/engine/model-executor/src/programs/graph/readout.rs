//! Target readout graphs. Features, logits and selection have distinct
//! sealed graphs, so feature-only work never touches the vocabulary
//! projection. Every entry gathers its rows from the final hidden rows
//! directly: `readout_features_rows` through `out_rows`, `readout_head_rows`
//! through `logit_rows` (the hidden rows of the projected outputs). The host
//! orders the projected rows with the selected ones first, so shaping and
//! sampling read the leading `selected` logits rows; no identity copy or
//! gather node precedes any readout entry.

use crate::{
    DeviceError, InvariantError, ModelLoadPlan, ResidentTarget, ResourceLimits, SubmitError,
    native::AttestedTarget,
};
use magnitude_model_batching::TargetBatchUpload;
use magnitude_model_contracts::{DecoderGeometry, WeightKind, WeightRole, WeightScope};
use magnitude_model_kernels::{readout_features_rows, readout_head_rows, sample_rows, shape_rows};
use seismic::{
    BoundNativeGraphPlan, Device, Element, NativeGraphFamily, NativeGraphPlan, NativeKernel,
    NativePort, WorkflowTensor, WorkflowTensorMut, WorkflowTensorRef,
};
use std::collections::BTreeMap;

/// Penalty history tokens per selected row.
const HISTORY_TOKENS: u64 = 64;

fn error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ReadoutKind {
    Features,
    Logits,
    /// Sampling of the leading selected logits rows; `shaped` graphs run
    /// `shape_rows` first, the others sample the logits as projected.
    Selection { shaped: bool },
}

/// Whether `shape_rows` changes a row with these shaping parameters
/// (`[temperature, top_k, top_p, min_p, repetition, presence, frequency,
/// flags]`, the `shape_rows` contract). Without penalties it is the identity
/// at temperature 0 (greedy) and at temperature 1 with no top-k, top-p or
/// min-p cut.
pub(crate) fn shapes(parameters: &[f32; 8]) -> bool {
    let [temperature, top_k, top_p, min_p, repetition, presence, frequency, _] = *parameters;
    let penalized = repetition != 1.0 || presence != 0.0 || frequency != 0.0;
    let cut = top_k > 0.0 || top_p < 1.0 || min_p > 0.0;
    penalized || (temperature != 0.0 && (temperature != 1.0 || cut))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ReadoutClass {
    pub rows: u64,
    pub outputs: u64,
    pub projected: u64,
    pub selected: u64,
    pub kind: ReadoutKind,
}

#[derive(Clone)]
pub(crate) struct PreparedTargetReadoutGraph {
    pub class: ReadoutClass,
    pub plan: NativeGraphPlan,
    pub hidden: NativePort,
    pub norm: NativePort,
    pub weight: Option<NativePort>,
    /// Hidden rows of the feature outputs.
    pub out_rows: NativePort,
    /// Hidden rows of the projected outputs, selected outputs first.
    pub logit_rows: Option<NativePort>,
    pub selection: Option<SelectionPorts>,
    pub features: WorkflowTensor,
    pub logits: Option<WorkflowTensor>,
    pub selected: Option<WorkflowTensor>,
}

#[derive(Clone)]
pub struct PreparedTargetReadoutGraphs {
    classes: BTreeMap<ReadoutClass, PreparedTargetReadoutGraph>,
    family: NativeGraphFamily,
    max_projected_rows: usize,
}

pub(crate) struct BoundTargetReadoutGraphs {
    pub prepared: PreparedTargetReadoutGraphs,
    bound: BTreeMap<ReadoutClass, BoundNativeGraphPlan>,
}

impl PreparedTargetReadoutGraphs {
    pub(crate) fn prepare(
        device: &Device,
        target: &AttestedTarget,
        load: &ModelLoadPlan,
        geometry: &DecoderGeometry,
        limits: ResourceLimits,
    ) -> Result<Self, String> {
        let row_classes = magnitude_model_batching::row_classes(limits.max_batch_rows);
        if row_classes.is_empty() {
            return Err(format!(
                "readout row bound {} has no row class",
                limits.max_batch_rows
            ));
        }
        // Outputs, projected and selected rows are subsets of a class's rows
        // and use the same ladder.
        let ladder = |bound: usize| -> Vec<u64> {
            magnitude_model_batching::row_classes(bound)
                .into_iter()
                .map(|rows| rows as u64)
                .collect()
        };
        let role = |kind| WeightRole {
            scope: WeightScope::Target,
            kind,
        };
        let norm = load
            .weights()
            .find(|weight| weight.role == role(WeightKind::OutputNorm))
            .ok_or("readout output norm weight is absent")?;
        let projection = load
            .weights()
            .find(|weight| weight.role == role(WeightKind::Output))
            .ok_or("readout output projection weight is absent")?;
        let mut classes = BTreeMap::new();
        let mut plans = Vec::new();
        let mut add = |class: ReadoutClass| -> Result<(), String> {
            let variant = PreparedTargetReadoutGraph::prepare(
                device, target, geometry, norm, projection, class,
            )?;
            plans.push(variant.plan.clone());
            classes.insert(class, variant);
            Ok(())
        };
        for rows in row_classes.iter().map(|&rows| rows as u64) {
            for outputs in ladder(rows as usize) {
                add(ReadoutClass {
                    rows,
                    outputs,
                    projected: 0,
                    selected: 0,
                    kind: ReadoutKind::Features,
                })?;
                for projected in ladder((outputs as usize).min(limits.max_projected_rows)) {
                    add(ReadoutClass {
                        rows,
                        outputs,
                        projected,
                        selected: 0,
                        kind: ReadoutKind::Logits,
                    })?;
                    for selected in ladder(projected as usize) {
                        for shaped in [false, true] {
                            add(ReadoutClass {
                                rows,
                                outputs,
                                projected,
                                selected,
                                kind: ReadoutKind::Selection { shaped },
                            })?;
                        }
                    }
                }
            }
        }
        let family = NativeGraphFamily::new(&plans).map_err(|error| error.to_string())?;
        Ok(Self {
            classes,
            family,
            max_projected_rows: limits.max_projected_rows,
        })
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.family.workspace_bytes()
    }
    pub fn output_bytes(&self) -> u64 {
        self.family.output_bytes()
    }
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    pub(crate) fn max_projected_rows(&self) -> usize {
        self.max_projected_rows
    }

    pub(crate) fn bind_weights(
        self,
        resident: &ResidentTarget,
    ) -> Result<BoundTargetReadoutGraphs, String> {
        let mut bound = BTreeMap::new();
        for (class, graph) in &self.classes {
            let mut fixed = vec![(&graph.norm, resident.output_norm.tensor())];
            if let Some(weight) = &graph.weight {
                fixed.push((weight, resident.output.tensor()));
            }
            bound.insert(
                *class,
                graph
                    .plan
                    .bind_static(&fixed)
                    .map_err(|error| format!("readout graph class {class:?}: {error}"))?,
            );
        }
        Ok(BoundTargetReadoutGraphs {
            prepared: self,
            bound,
        })
    }
}

impl BoundTargetReadoutGraphs {
    pub(crate) fn class(
        &self,
        class: ReadoutClass,
    ) -> Result<(&PreparedTargetReadoutGraph, &BoundNativeGraphPlan), String> {
        let graph = self
            .prepared
            .classes
            .get(&class)
            .ok_or_else(|| format!("readout graph class {class:?} was not sealed"))?;
        let bound = self
            .bound
            .get(&class)
            .ok_or_else(|| format!("readout graph class {class:?} was not bound"))?;
        Ok((graph, bound))
    }
}

/// The host-written inputs of a selection readout.
#[derive(Clone)]
pub(crate) struct SelectionPorts {
    /// Shaping parameters and penalty history; present when the class is
    /// shaped.
    pub shaping: Option<ShapingPorts>,
    /// Per-row flags: nonzero rows sample under their `mask` row.
    pub constrained: NativePort,
    /// Written only when some row is constrained.
    pub mask: NativePort,
    pub draws: NativePort,
}

#[derive(Clone)]
pub(crate) struct ShapingPorts {
    pub parameters: NativePort,
    pub history: NativePort,
}

impl PreparedTargetReadoutGraph {
    fn prepare(
        device: &Device,
        target: &AttestedTarget,
        geometry: &DecoderGeometry,
        norm_plan: &crate::WeightPlan,
        weight_plan: &crate::WeightPlan,
        class: ReadoutClass,
    ) -> Result<Self, String> {
        let mut graph = device.native_graph();
        let hidden = graph
            .port(Element::f32(), &[class.rows, geometry.hidden])
            .map_err(|error| error.to_string())?;
        let norm = graph
            .port(norm_plan.resident, &norm_plan.shape)
            .map_err(|error| error.to_string())?;
        let epsilon = geometry.epsilon as f32;
        let out_rows = graph
            .input_for(
                &target.readout.features,
                "out_rows",
                &[("M", class.rows), ("O", class.outputs), ("D", geometry.hidden)],
            )
            .map_err(error)?;
        let features = graph
            .enqueue(
                &target.readout.features,
                readout_features_rows::WorkflowArgs {
                    hidden: hidden.tensor().into(),
                    norm: norm.tensor().into(),
                    out_rows: out_rows.tensor().into(),
                    epsilon,
                },
            )
            .map_err(error)?
            .value;
        graph.export(&features).map_err(error)?;
        let mut weight = None;
        let mut logit_rows = None;
        let mut logits = None;
        let mut selection = None;
        let mut selected = None;
        if class.kind != ReadoutKind::Features {
            let projection = graph
                .port(weight_plan.resident, &weight_plan.shape)
                .map_err(|error| error.to_string())?;
            let rows = graph
                .input_for(
                    &target.readout.head,
                    "out_rows",
                    &[
                        ("M", class.rows),
                        ("O", class.projected),
                        ("V", geometry.vocabulary),
                        ("D", geometry.hidden),
                    ],
                )
                .map_err(error)?;
            let projected = graph
                .enqueue(
                    &target.readout.head,
                    readout_head_rows::WorkflowArgs {
                        hidden: hidden.tensor().into(),
                        norm: norm.tensor().into(),
                        weight: projection.tensor().into(),
                        out_rows: rows.tensor().into(),
                        epsilon,
                    },
                )
                .map_err(error)?
                .value;
            graph.export(&projected).map_err(error)?;
            if let ReadoutKind::Selection { shaped } = class.kind {
                let mut sampled = graph
                    .local_for(
                        &target.sample,
                        "result",
                        &[("M", class.selected), ("V", geometry.vocabulary)],
                    )
                    .map_err(error)?;
                let leading = projected.slice_leading(0, class.selected);
                let ports = sample(
                    &mut graph,
                    &target.shape,
                    &target.sample,
                    geometry.vocabulary,
                    (&leading).into(),
                    class.selected,
                    shaped,
                    sampled.tensor_mut().into(),
                )?;
                graph.export(sampled.tensor()).map_err(error)?;
                selection = Some(ports);
                selected = Some(sampled.tensor().clone());
            }
            weight = Some(projection);
            logit_rows = Some(rows);
            logits = Some(projected);
        }
        let plan = graph.seal().map_err(error)?;
        Ok(Self {
            class,
            plan,
            hidden,
            norm,
            weight,
            out_rows,
            logit_rows,
            selection,
            features,
            logits,
            selected,
        })
    }

}

/// Sampling of `rows` logits rows into `result`, after `shape_rows` when
/// `shaped`. The target readout and the draft head select through this one
/// node sequence and control layout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample(
    graph: &mut seismic::NativeGraph,
    shape: &NativeKernel<shape_rows::Entry>,
    sample: &NativeKernel<sample_rows::Entry>,
    vocabulary: u64,
    logits: WorkflowTensorRef<'_>,
    rows: u64,
    shaped: bool,
    result: WorkflowTensorMut<'_>,
) -> Result<SelectionPorts, String> {
    let sample_dims = [("M", rows), ("V", vocabulary)];
    let mask = graph.input_for(sample, "mask", &sample_dims).map_err(error)?;
    let constrained = graph
        .input_for(sample, "constrained", &sample_dims)
        .map_err(error)?;
    let draws = graph.input_for(sample, "draws", &sample_dims).map_err(error)?;
    let shaping = if shaped {
        let shape_dims = [("Sx", rows), ("V", vocabulary), ("Hn", HISTORY_TOKENS)];
        let parameters = graph.input_for(shape, "params", &shape_dims).map_err(error)?;
        let history = graph.input_for(shape, "history", &shape_dims).map_err(error)?;
        let mut out = graph.local_for(shape, "out", &shape_dims).map_err(error)?;
        graph
            .enqueue(
                shape,
                shape_rows::WorkflowArgs {
                    logits,
                    params: parameters.tensor().into(),
                    history: history.tensor().into(),
                    out: out.tensor_mut().into(),
                },
            )
            .map_err(error)?;
        graph
            .enqueue(
                sample,
                sample_rows::WorkflowArgs {
                    logits: out.tensor().into(),
                    mask: mask.tensor().into(),
                    constrained: constrained.tensor().into(),
                    draws: draws.tensor().into(),
                    result,
                },
            )
            .map_err(error)?;
        Some(ShapingPorts {
            parameters,
            history,
        })
    } else {
        graph
            .enqueue(
                sample,
                sample_rows::WorkflowArgs {
                    logits,
                    mask: mask.tensor().into(),
                    constrained: constrained.tensor().into(),
                    draws: draws.tensor().into(),
                    result,
                },
            )
            .map_err(error)?;
        None
    };
    Ok(SelectionPorts {
        shaping,
        constrained,
        mask,
        draws,
    })
}

/// Selection controls of a padded selection class from a pass's packed
/// selections. Padding rows repeat the first selected row. An unconstrained
/// row carries only its flag: the mask input is written only when some row
/// is constrained, and then holds zeros in the rows that are not. A graph
/// that selects over the leading `row_words * 32` tokens takes each mask's
/// leading `row_words` words.
pub(crate) fn write_selection(
    batch: &TargetBatchUpload<'_>,
    active: &mut seismic::NativeGraphFamilyActive<'_>,
    ports: &SelectionPorts,
    selected_class: usize,
    row_words: usize,
) -> Result<(), SubmitError> {
    fn device(error: impl std::fmt::Display) -> SubmitError {
        SubmitError::Device(DeviceError::Execution(error.to_string()))
    }
    fn invalid(detail: &str) -> SubmitError {
        SubmitError::Invariant(InvariantError {
            context: "selection controls",
            detail: detail.into(),
        })
    }
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
        .flat_map(|&mask_row| i32::from(mask_row >= 0).to_le_bytes())
        .collect::<Vec<_>>();
    active
        .write_input(&ports.constrained, &constrained)
        .map_err(device)?;
    if mask_rows.iter().all(|&mask_row| mask_row < 0) {
        return Ok(());
    }
    if row_words > batch.mask_words {
        return Err(invalid("selection graph is wider than the vocabulary"));
    }
    let mut masks = vec![0_u32; selected_class * row_words];
    for (index, &mask_row) in mask_rows.iter().enumerate() {
        let Ok(mask_row) = usize::try_from(mask_row) else {
            continue;
        };
        let mask = batch
            .masks
            .get(mask_row)
            .ok_or_else(|| invalid("selection mask is absent"))?;
        if mask.len() != batch.mask_words {
            return Err(invalid("selection mask width differs from the vocabulary"));
        }
        masks[index * row_words..(index + 1) * row_words].copy_from_slice(&mask[..row_words]);
    }
    let bytes = masks
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    active.write_input(&ports.mask, &bytes).map_err(device)
}
