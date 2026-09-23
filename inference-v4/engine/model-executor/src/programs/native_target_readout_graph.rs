//! Exact Seismic readout graphs. Features, logits, and selection have distinct
//! checked graphs, so feature-only work never materializes a vocabulary row.

use crate::{
    ModelLoadPlan, ResidentTarget, ResourceLimits,
    native::{AttestedState, AttestedTarget},
};
use magnitude_model_contracts::{
    ActivationDType, DecoderGeometry, WeightKind, WeightRole, WeightScope,
};
use magnitude_model_kernels::{
    copy_rows, gather_rows, head_logits_rows, qwen_features_rows, sample_rows, shape_rows,
};
use seismic::{
    BoundNativeGraphPlan, Device, Element, NativeGraphFamily, NativeGraphPlan, NativePort,
    WorkflowTensor,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ReadoutKind {
    Features,
    Logits,
    Selection,
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
    pub out_rows: NativePort,
    pub logit_rows: Option<NativePort>,
    pub logit_to_rows: Option<NativePort>,
    pub select_rows: Option<NativePort>,
    pub shaping: Option<NativePort>,
    pub history: Option<NativePort>,
    pub mask: Option<NativePort>,
    pub draws: Option<NativePort>,
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
        state: &AttestedState,
        load: &ModelLoadPlan,
        geometry: &DecoderGeometry,
        limits: ResourceLimits,
    ) -> Result<Self, String> {
        let max_rows = u64::try_from(limits.max_batch_rows)
            .map_err(|_| "readout row bound exceeds u64")?
            .checked_next_power_of_two()
            .ok_or("readout row class overflows")?;
        let max_projected = u64::try_from(limits.max_projected_rows)
            .map_err(|_| "projected row bound exceeds u64")?
            .checked_next_power_of_two()
            .ok_or("projected row class overflows")?;
        let norm = load
            .weights()
            .find(|weight| {
                weight.role
                    == WeightRole {
                        scope: WeightScope::Target,
                        kind: WeightKind::OutputNorm,
                    }
            })
            .ok_or("readout output norm weight is absent")?;
        let projection = load
            .weights()
            .find(|weight| {
                weight.role
                    == WeightRole {
                        scope: WeightScope::Target,
                        kind: WeightKind::Output,
                    }
            })
            .ok_or("readout output projection weight is absent")?;
        let mut classes = BTreeMap::new();
        let mut plans = Vec::new();
        let mut rows = 1;
        while rows <= max_rows {
            let mut outputs = 1;
            while outputs <= rows {
                let class = ReadoutClass {
                    rows,
                    outputs,
                    projected: 0,
                    selected: 0,
                    kind: ReadoutKind::Features,
                };
                let variant = PreparedTargetReadoutGraph::prepare(
                    device, target, state, geometry, norm, projection, class,
                )?;
                plans.push(variant.plan.clone());
                classes.insert(class, variant);
                let mut projected = 1;
                while projected <= outputs.min(max_projected) {
                    let class = ReadoutClass {
                        rows,
                        outputs,
                        projected,
                        selected: 0,
                        kind: ReadoutKind::Logits,
                    };
                    let variant = PreparedTargetReadoutGraph::prepare(
                        device, target, state, geometry, norm, projection, class,
                    )?;
                    plans.push(variant.plan.clone());
                    classes.insert(class, variant);
                    let mut selected = 1;
                    while selected <= projected {
                        let class = ReadoutClass {
                            rows,
                            outputs,
                            projected,
                            selected,
                            kind: ReadoutKind::Selection,
                        };
                        let variant = PreparedTargetReadoutGraph::prepare(
                            device, target, state, geometry, norm, projection, class,
                        )?;
                        plans.push(variant.plan.clone());
                        classes.insert(class, variant);
                        selected *= 2;
                    }
                    projected *= 2;
                }
                outputs *= 2;
            }
            rows *= 2;
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

impl PreparedTargetReadoutGraph {
    fn prepare(
        device: &Device,
        target: &AttestedTarget,
        state: &AttestedState,
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
        let out_rows = graph
            .input_for(
                &target.readout.features,
                "out_rows",
                &[
                    ("M", class.rows),
                    ("O", class.outputs),
                    ("D", geometry.hidden),
                ],
            )
            .map_err(|error| error.to_string())?;
        let features = graph
            .enqueue(
                &target.readout.features,
                qwen_features_rows::WorkflowArgs {
                    hidden: hidden.tensor().into(),
                    norm: norm.tensor().into(),
                    out_rows: out_rows.tensor().into(),
                    epsilon: geometry.epsilon as f32,
                },
            )
            .map_err(|error| error.to_string())?
            .value;
        graph.export(&features).map_err(|error| error.to_string())?;
        let mut weight = None;
        let mut logit_rows = None;
        let mut logit_to_rows = None;
        let mut logits = None;
        let mut selected_value = None;
        let mut select_rows = None;
        let mut shaping = None;
        let mut history = None;
        let mut mask = None;
        let mut draws = None;
        if class.kind != ReadoutKind::Features {
            let activation = match geometry.activation_dtype {
                ActivationDType::F16 => Element::f16(),
                ActivationDType::BF16 => Element::bf16(),
            };
            let copy = &state
                .copies
                .iter()
                .find(|(element, _)| *element == activation)
                .ok_or("readout activation copy kernel is absent")?
                .1;
            let copy_dims = [
                ("N", class.projected),
                ("TS", class.outputs),
                ("TD", class.projected),
                ("KV", 1),
                ("W", geometry.hidden),
            ];
            let rows = graph
                .input_for(copy, "from", &copy_dims)
                .map_err(|error| error.to_string())?;
            let to_rows = graph
                .input_for(copy, "to", &copy_dims)
                .map_err(|error| error.to_string())?;
            let mut projected_features = graph
                .local_for(copy, "dst", &copy_dims)
                .map_err(|error| error.to_string())?;
            graph
                .enqueue(
                    copy,
                    copy_rows::WorkflowArgs {
                        src: (&features.reshape(&[class.outputs, 1, geometry.hidden])).into(),
                        dst: projected_features.tensor_mut().into(),
                        from: rows.tensor().into(),
                        to: to_rows.tensor().into(),
                    },
                )
                .map_err(|error| error.to_string())?;
            let projection = graph
                .port(weight_plan.resident, &weight_plan.shape)
                .map_err(|error| error.to_string())?;
            let projected = graph
                .enqueue(
                    &target.readout.logits,
                    head_logits_rows::WorkflowArgs {
                        features: (&projected_features
                            .tensor()
                            .reshape(&[class.projected, geometry.hidden]))
                            .into(),
                        weight: projection.tensor().into(),
                    },
                )
                .map_err(|error| error.to_string())?
                .value;
            graph
                .export(&projected)
                .map_err(|error| error.to_string())?;
            weight = Some(projection);
            logit_rows = Some(rows);
            logit_to_rows = Some(to_rows);
            if class.kind == ReadoutKind::Selection {
                let gather = state
                    .gather
                    .as_ref()
                    .ok_or("readout selection gather kernel is absent")?;
                let selected_rows = graph
                    .input_for(
                        gather,
                        "rows",
                        &[
                            ("M", class.projected),
                            ("O", class.selected),
                            ("D", geometry.vocabulary),
                        ],
                    )
                    .map_err(|error| error.to_string())?;
                let selected_logits = graph
                    .enqueue(
                        gather,
                        gather_rows::WorkflowArgs {
                            source: (&projected).into(),
                            rows: selected_rows.tensor().into(),
                        },
                    )
                    .map_err(|error| error.to_string())?
                    .value;
                let shape_dims = [
                    ("Sx", class.selected),
                    ("V", geometry.vocabulary),
                    ("Hn", 64),
                ];
                let params = graph
                    .input_for(&target.shape, "params", &shape_dims)
                    .map_err(|error| error.to_string())?;
                let history_rows = graph
                    .input_for(&target.shape, "history", &shape_dims)
                    .map_err(|error| error.to_string())?;
                let mut shaped = graph
                    .local_for(&target.shape, "out", &shape_dims)
                    .map_err(|error| error.to_string())?;
                graph
                    .enqueue(
                        &target.shape,
                        shape_rows::WorkflowArgs {
                            logits: (&selected_logits).into(),
                            params: params.tensor().into(),
                            history: history_rows.tensor().into(),
                            out: shaped.tensor_mut().into(),
                        },
                    )
                    .map_err(|error| error.to_string())?;
                let sample_dims = [("M", class.selected), ("V", geometry.vocabulary)];
                let mask_rows = graph
                    .input_for(&target.sample, "mask", &sample_dims)
                    .map_err(|error| error.to_string())?;
                let draw_rows = graph
                    .input_for(&target.sample, "draws", &sample_dims)
                    .map_err(|error| error.to_string())?;
                let mut sampled = graph
                    .local_for(&target.sample, "result", &sample_dims)
                    .map_err(|error| error.to_string())?;
                graph
                    .enqueue(
                        &target.sample,
                        sample_rows::WorkflowArgs {
                            logits: shaped.tensor().into(),
                            mask: mask_rows.tensor().into(),
                            draws: draw_rows.tensor().into(),
                            result: sampled.tensor_mut().into(),
                        },
                    )
                    .map_err(|error| error.to_string())?;
                graph
                    .export(sampled.tensor())
                    .map_err(|error| error.to_string())?;
                selected_value = Some(sampled.tensor().clone());
                select_rows = Some(selected_rows);
                shaping = Some(params);
                history = Some(history_rows);
                mask = Some(mask_rows);
                draws = Some(draw_rows);
            }
            logits = Some(projected);
        }
        let plan = graph.seal().map_err(|error| error.to_string())?;
        Ok(Self {
            class,
            plan,
            hidden,
            norm,
            weight,
            out_rows,
            logit_rows,
            logit_to_rows,
            select_rows,
            shaping,
            history,
            mask,
            draws,
            features,
            logits,
            selected: selected_value,
        })
    }
}
