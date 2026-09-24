//! Target readout graphs. Features, logits and selection have distinct
//! sealed graphs, so feature-only work never touches the vocabulary
//! projection. Every entry gathers its rows from the final hidden rows
//! directly: `qwen_features_rows` through `out_rows`, `qwen_head_rows`
//! through `logit_rows` (the hidden rows of the projected outputs). The host
//! orders the projected rows with the selected ones first, so shaping and
//! sampling read the leading `selected` logits rows; no identity copy or
//! gather node precedes any readout entry.

use crate::{ModelLoadPlan, ResidentTarget, ResourceLimits, native::AttestedTarget};
use magnitude_model_contracts::{DecoderGeometry, WeightKind, WeightRole, WeightScope};
use magnitude_model_kernels::{qwen_features_rows, qwen_head_rows, sample_rows, shape_rows};
use seismic::{
    BoundNativeGraphPlan, Device, Element, NativeGraphFamily, NativeGraphPlan, NativePort,
    WorkflowTensor,
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
                qwen_features_rows::WorkflowArgs {
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
                    qwen_head_rows::WorkflowArgs {
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
                let (ports, sampled) = Self::select(
                    &mut graph,
                    target,
                    geometry,
                    &projected.slice_leading(0, class.selected),
                    class.selected,
                    shaped,
                )?;
                selection = Some(ports);
                selected = Some(sampled);
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

    /// Sampling of the leading `selected` logits rows, after `shape_rows`
    /// when the class is `shaped`.
    fn select(
        graph: &mut seismic::NativeGraph,
        target: &AttestedTarget,
        geometry: &DecoderGeometry,
        logits: &seismic::WorkflowTensorView,
        selected: u64,
        shaped: bool,
    ) -> Result<(SelectionPorts, WorkflowTensor), String> {
        let sample_dims = [("M", selected), ("V", geometry.vocabulary)];
        let mask = graph.input_for(&target.sample, "mask", &sample_dims).map_err(error)?;
        let constrained = graph
            .input_for(&target.sample, "constrained", &sample_dims)
            .map_err(error)?;
        let draws = graph.input_for(&target.sample, "draws", &sample_dims).map_err(error)?;
        let mut sampled = graph.local_for(&target.sample, "result", &sample_dims).map_err(error)?;
        let shaping = if shaped {
            let shape_dims = [
                ("Sx", selected),
                ("V", geometry.vocabulary),
                ("Hn", HISTORY_TOKENS),
            ];
            let parameters = graph.input_for(&target.shape, "params", &shape_dims).map_err(error)?;
            let history = graph.input_for(&target.shape, "history", &shape_dims).map_err(error)?;
            let mut out = graph.local_for(&target.shape, "out", &shape_dims).map_err(error)?;
            graph
                .enqueue(
                    &target.shape,
                    shape_rows::WorkflowArgs {
                        logits: logits.into(),
                        params: parameters.tensor().into(),
                        history: history.tensor().into(),
                        out: out.tensor_mut().into(),
                    },
                )
                .map_err(error)?;
            graph
                .enqueue(
                    &target.sample,
                    sample_rows::WorkflowArgs {
                        logits: out.tensor().into(),
                        mask: mask.tensor().into(),
                        constrained: constrained.tensor().into(),
                        draws: draws.tensor().into(),
                        result: sampled.tensor_mut().into(),
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
                    &target.sample,
                    sample_rows::WorkflowArgs {
                        logits: logits.into(),
                        mask: mask.tensor().into(),
                        constrained: constrained.tensor().into(),
                        draws: draws.tensor().into(),
                        result: sampled.tensor_mut().into(),
                    },
                )
                .map_err(error)?;
            None
        };
        graph.export(sampled.tensor()).map_err(error)?;
        Ok((
            SelectionPorts {
                shaping,
                constrained,
                mask,
                draws,
            },
            sampled.tensor().clone(),
        ))
    }
}
