//! Target readout graphs. Features, logits and selection have distinct
//! sealed graphs, so feature-only work never touches the vocabulary
//! projection. Every entry gathers its rows from the final hidden rows
//! directly: `readout_features_rows` through `out_rows`, `readout_head_rows`
//! through `logit_rows` (the hidden rows of the projected outputs). The host
//! orders the projected rows with the selected ones first, so shaping and
//! sampling read the leading `selected` logits rows; no identity copy or
//! gather node precedes any readout entry.
//!
//! When a separate draft drafts, the features are its conditioning instead:
//! the fusion of the target taps (`project_rows` over the draft input rows
//! the tapped blocks wrote) gathered through `out_rows` (`feature_rows`).

use crate::{
    native::AttestedTarget, programs::graph::draft::GraphDraft, DeviceError, InvariantError,
    ModelLoadPlan, ResidentTarget, ResourceLimits, SubmitError,
};
use magnitude_batching::TargetBatchUpload;
use magnitude_family_contracts::{Decoder, ExitNorm, WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{
    feature_rows, project_rows, readout_features_rows, readout_head_rows, sample_rows, shape_rows,
};
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, Element, Entry, NativeGraphClassSlice,
    NativeGraphFamily, NativeGraphLayout, NativeGraphMetadata, NativeGraphPlan,
    NativeGraphStorageBytes, NativePort, WorkflowTensor, WorkflowTensorMut, WorkflowTensorRef,
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
    Selection {
        shaped: bool,
    },
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

/// Where a readout's features come from.
#[derive(Clone, Copy)]
enum FeatureSource<'p> {
    /// The final norm of the hidden rows.
    Output,
    /// A separate draft's fused taps: `fusion` projects the draft input rows.
    Taps { fusion: &'p crate::WeightPlan },
}

impl<'p> FeatureSource<'p> {
    /// The draft's taps when its fusion is planned with the target.
    fn of(load: &'p ModelLoadPlan) -> Self {
        let role = WeightRole {
            scope: WeightScope::Draft,
            kind: WeightKind::DraftFusion,
        };
        match load.target().iter().find(|weight| weight.role == role) {
            Some(fusion) => Self::Taps { fusion },
            None => Self::Output,
        }
    }
}

/// The feature entries of a readout graph, by its feature source.
enum FeatureEntries<'a, G: GraphDraft + 'a> {
    Output(G::Binding<'a, readout_features_rows::Entry>),
    Taps {
        fusion: G::Binding<'a, project_rows::Entry>,
        features: G::Binding<'a, feature_rows::Entry>,
    },
}

/// The draft input rows (bound per run), the fusion weight and its
/// accumulator-scale port of a readout graph that publishes a separate
/// draft's conditioning.
#[derive(Clone)]
pub(crate) struct ReadoutTapPorts {
    pub taps: NativePort,
    pub fusion: NativePort,
    pub fusion_scale: NativePort,
}

#[derive(Clone)]
pub(crate) struct PreparedTargetReadoutGraph {
    pub plan: NativeGraphPlan,
    /// Absent for a tapped readout's feature-only classes.
    pub final_rows: Option<FinalRowPorts>,
    pub taps: Option<ReadoutTapPorts>,
    /// The vocabulary projection and its accumulator-scale port.
    pub weight: Option<(NativePort, NativePort)>,
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

fn readout_classes(limits: ResourceLimits) -> Result<Vec<ReadoutClass>, String> {
    let row_classes = magnitude_batching::row_classes(limits.max_launch_rows);
    if row_classes.is_empty() {
        return Err(format!(
            "readout row bound {} has no row class",
            limits.max_launch_rows
        ));
    }
    // Outputs, projected and selected rows are subsets of a class's rows
    // and use the same ladder.
    let ladder = |bound: usize| -> Vec<u64> {
        magnitude_batching::row_classes(bound)
            .into_iter()
            .map(|rows| rows as u64)
            .collect()
    };
    let mut classes = Vec::new();
    for rows in row_classes.into_iter().map(|rows| rows as u64) {
        for outputs in ladder(rows as usize) {
            classes.push(ReadoutClass {
                rows,
                outputs,
                projected: 0,
                selected: 0,
                kind: ReadoutKind::Features,
            });
            for projected in ladder((outputs as usize).min(limits.max_projected_rows)) {
                classes.push(ReadoutClass {
                    rows,
                    outputs,
                    projected,
                    selected: 0,
                    kind: ReadoutKind::Logits,
                });
                for selected in ladder(projected as usize) {
                    for shaped in [false, true] {
                        classes.push(ReadoutClass {
                            rows,
                            outputs,
                            projected,
                            selected,
                            kind: ReadoutKind::Selection { shaped },
                        });
                    }
                }
            }
        }
    }
    Ok(classes)
}

impl PreparedTargetReadoutGraphs {
    pub(crate) fn prepare(
        device: &Device,
        target: &AttestedTarget,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        limits: ResourceLimits,
    ) -> Result<Self, String> {
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
        let source = FeatureSource::of(load);
        let regimes =
            certify_readout_regimes(device.backend(), geometry, norm, projection, source, limits)?;
        let mut classes = BTreeMap::new();
        let mut plans = Vec::new();
        let mut add = |class: ReadoutClass| -> Result<(), String> {
            let layout = regimes
                .get(&readout_regime(class))
                .ok_or("readout class has no resource regime")?;
            let variant = PreparedTargetReadoutGraph::prepare(
                device, target, geometry, norm, projection, source, class, layout,
            )?;
            plans.push(variant.plan.clone());
            classes.insert(class, variant);
            Ok(())
        };
        for class in readout_classes(limits)? {
            add(class)?;
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
            let mut fixed = Vec::new();
            let absent_scale = seismic::Tensor::from_host(
                &resident.output.tensor().device(), Element::f32(), &[0], &[],
            ).map_err(|error| error.to_string())?;
            if let Some(rows) = &graph.final_rows {
                fixed.push((&rows.norm, resident.output_norm.tensor()));
            }
            if let Some((weight, scale)) = &graph.weight {
                fixed.push((weight, resident.output.tensor()));
                fixed.push((scale, resident.output.scale().unwrap_or(&absent_scale)));
            }
            if let Some(taps) = &graph.taps {
                let fusion = resident
                    .fusion
                    .as_ref()
                    .ok_or("a tapped readout has no resident draft fusion")?;
                fixed.push((&taps.fusion, fusion.projection.tensor()));
                fixed.push((
                    &taps.fusion_scale,
                    fusion.projection.scale().unwrap_or(&absent_scale),
                ));
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
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        device: &Device,
        target: &AttestedTarget,
        geometry: &Decoder,
        norm_plan: &crate::WeightPlan,
        weight_plan: &crate::WeightPlan,
        source: FeatureSource<'_>,
        class: ReadoutClass,
        layout: &NativeGraphLayout,
    ) -> Result<Self, String> {
        let entries = match (source, &target.taps) {
            (FeatureSource::Output, _) => FeatureEntries::Output(&target.readout.features),
            (FeatureSource::Taps { .. }, Some(taps)) => FeatureEntries::Taps {
                fusion: &taps.fusion,
                features: &taps.features,
            },
            (FeatureSource::Taps { .. }, None) => {
                return Err("a tapped readout has no tap entries".into())
            }
        };
        let (mut graph, final_rows, out_rows, features, taps) = feature_topology(
            device.native_graph_with_layout(layout),
            entries,
            geometry,
            norm_plan,
            source,
            class,
        )?;
        let mut weight = None;
        let mut logit_rows = None;
        let mut logits = None;
        let mut selection = None;
        let mut selected = None;
        if class.kind != ReadoutKind::Features {
            let rows = final_rows
                .as_ref()
                .ok_or("a projecting readout has no final rows")?;
            let (projected_graph, projection, rows, projected) = projected_topology(
                graph,
                &target.readout.head,
                geometry,
                weight_plan,
                class,
                &rows.hidden,
                &rows.norm,
            )?;
            graph = projected_graph;
            if let ReadoutKind::Selection { .. } = class.kind {
                let (selection_graph, ports, sampled) = selected_topology(
                    graph,
                    &target.shape,
                    &target.sample,
                    geometry,
                    class,
                    &projected,
                )?;
                graph = selection_graph;
                selection = Some(ports);
                selected = Some(sampled);
            }
            weight = Some(projection);
            logit_rows = Some(rows);
            logits = Some(projected);
        }
        let plan = graph.seal().map_err(error)?;
        Ok(Self {
            plan,
            final_rows,
            taps,
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

/// The ports and exported features of a readout's feature prefix. The final
/// hidden rows and norm are absent when the class reads neither (tapped
/// features without a projection).
type FeaturePrefix<G> = (
    G,
    Option<FinalRowPorts>,
    NativePort,
    WorkflowTensor,
    Option<ReadoutTapPorts>,
);

/// The decoder's final hidden rows (bound per run) and final norm.
#[derive(Clone)]
pub(crate) struct FinalRowPorts {
    pub hidden: NativePort,
    pub norm: NativePort,
}

fn final_row_ports<G: GraphDraft>(
    graph: &mut G,
    geometry: &Decoder,
    norm_plan: &crate::WeightPlan,
    class: ReadoutClass,
) -> Result<FinalRowPorts, String> {
    Ok(FinalRowPorts {
        hidden: graph.port_with_class_extent(
            Element::f32(),
            &[class.rows, geometry.hidden],
            0,
            "M",
        )?,
        norm: graph.port(norm_plan.resident, &norm_plan.shape)?,
    })
}

/// The same feature prefix is used by feature-only and projected readout
/// graphs. The checked route seals this prefix only for the feature class.
fn feature_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entries: FeatureEntries<'a, G>,
    geometry: &Decoder,
    norm_plan: &crate::WeightPlan,
    source: FeatureSource<'_>,
    class: ReadoutClass,
) -> Result<FeaturePrefix<G>, String> {
    let dimensions = [
        ("M", class.rows),
        ("O", class.outputs),
        ("D", geometry.hidden),
    ];
    let (final_rows, out_rows, features, taps) = match (entries, source) {
        (FeatureEntries::Output(entry), FeatureSource::Output) => {
            let final_rows = final_row_ports(&mut graph, geometry, norm_plan, class)?;
            let (hidden, norm) = (&final_rows.hidden, &final_rows.norm);
            let out_rows = graph.input_for(entry, "out_rows", &dimensions)?;
            let features = graph
                .enqueue::<readout_features_rows::Entry>(
                    entry,
                    &dimensions,
                    readout_features_rows::WorkflowArgs {
                        hidden: hidden.tensor().into(),
                        norm: norm.tensor().into(),
                        out_rows: out_rows.tensor().into(),
                        epsilon: readout_epsilon(geometry)?,
                    },
                )?
                .value;
            (Some(final_rows), out_rows, features, None)
        }
        (FeatureEntries::Taps { fusion, features }, FeatureSource::Taps { fusion: plan }) => {
            let [_, width] = plan.shape[..] else {
                return Err("the draft fusion is not a matrix".into());
            };
            let activation = match geometry.activation_dtype {
                magnitude_family_contracts::ActivationDType::F16 => Element::f16(),
                magnitude_family_contracts::ActivationDType::BF16 => Element::bf16(),
            };
            let taps = graph.port_with_class_extent(activation, &[class.rows, width], 0, "M")?;
            let weight = graph.port(plan.resident, &plan.shape)?;
            // The fusion's resident second-level scale, or an absent scale.
            let scale_extent = plan.scale_extent();
            let fusion_scale = graph.port(Element::f32(), &[scale_extent])?;
            let fused = graph
                .enqueue::<project_rows::Entry>(
                    fusion,
                    &[
                        ("M", class.rows),
                        ("K", width),
                        ("N", geometry.hidden),
                        ("WS", scale_extent),
                    ],
                    project_rows::WorkflowArgs {
                        source: taps.tensor().into(),
                        weight: weight.tensor().into(),
                        weight_scale: fusion_scale.tensor().into(),
                    },
                )?
                .value;
            let out_rows = graph.input_for(features, "out_rows", &dimensions)?;
            let conditioning = graph
                .enqueue::<feature_rows::Entry>(
                    features,
                    &dimensions,
                    feature_rows::WorkflowArgs {
                        fused: (&fused).into(),
                        out_rows: out_rows.tensor().into(),
                    },
                )?
                .value;
            // A projecting class reads the final rows after the features.
            let final_rows = (class.kind != ReadoutKind::Features)
                .then(|| final_row_ports(&mut graph, geometry, norm_plan, class))
                .transpose()?;
            (
                final_rows,
                out_rows,
                conditioning,
                Some(ReadoutTapPorts {
                    taps,
                    fusion: weight,
                    fusion_scale,
                }),
            )
        }
        _ => return Err("readout feature entries disagree with their source".into()),
    };
    graph.export(&features)?;
    Ok((graph, final_rows, out_rows, features, taps))
}

fn projected_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entry: G::Binding<'a, readout_head_rows::Entry>,
    geometry: &Decoder,
    weight_plan: &crate::WeightPlan,
    class: ReadoutClass,
    hidden: &NativePort,
    norm: &NativePort,
) -> Result<(G, (NativePort, NativePort), NativePort, WorkflowTensor), String> {
    let weight = graph.port(weight_plan.resident, &weight_plan.shape)?;
    // The weight's resident second-level scale, or an absent scale.
    let extent = weight_plan.scale_extent();
    let scale = graph.port(Element::f32(), &[extent])?;
    let dimensions = [
        ("M", class.rows),
        ("O", class.projected),
        ("V", geometry.vocabulary),
        ("D", geometry.hidden),
        ("WS", extent),
    ];
    let rows = graph.input_for(entry, "out_rows", &dimensions)?;
    let logits = graph
        .enqueue::<readout_head_rows::Entry>(
            entry,
            &dimensions,
            readout_head_rows::WorkflowArgs {
                hidden: hidden.tensor().into(),
                norm: norm.tensor().into(),
                weight: weight.tensor().into(),
                out_rows: rows.tensor().into(),
                epsilon: readout_epsilon(geometry)?,
                softcap: readout_softcap(geometry),
                weight_scale: scale.tensor().into(),
            },
        )?
        .value;
    graph.export(&logits)?;
    Ok((graph, (weight, scale), rows, logits))
}

/// The epsilon of the decoder's final normalization, which the readout
/// entries fuse (`operators::admit` admits only an RMS exit norm).
pub(crate) fn readout_epsilon(geometry: &Decoder) -> Result<f32, String> {
    match &geometry.exit.norm {
        ExitNorm::Rms(norm) => Ok(norm.epsilon as f32),
        _ => Err("readout requires an RMS final normalization".into()),
    }
}

/// The head entries' `softcap` scalar: the exit softcap, 0 for none (the
/// entries' contract).
pub(crate) fn readout_softcap(geometry: &Decoder) -> f32 {
    geometry.exit.softcap.map_or(0.0, |cap| cap as f32)
}

fn selected_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    shape: G::Binding<'a, shape_rows::Entry>,
    sampler: G::Binding<'a, sample_rows::Entry>,
    geometry: &Decoder,
    class: ReadoutClass,
    logits: &WorkflowTensor,
) -> Result<(G, SelectionPorts, WorkflowTensor), String> {
    let ReadoutKind::Selection { shaped } = class.kind else {
        return Err("selection topology requires a selection class".into());
    };
    let mut result = graph.local_for(
        sampler,
        "result",
        &[("M", class.selected), ("V", geometry.vocabulary)],
    )?;
    let leading = logits.slice_leading(0, class.selected);
    let ports = sample(
        &mut graph,
        shape,
        sampler,
        geometry.vocabulary,
        (&leading).into(),
        class.selected,
        shaped,
        result.tensor_mut().into(),
    )?;
    graph.export(result.tensor())?;
    Ok((graph, ports, result.tensor().clone()))
}

#[cfg(test)]
fn checked_readout_class_storage(
    backend: BackendName,
    geometry: &Decoder,
    norm_plan: &crate::WeightPlan,
    weight_plan: Option<&crate::WeightPlan>,
    class: ReadoutClass,
) -> Result<NativeGraphStorageBytes, String> {
    checked_readout_class_draft(
        NativeGraphMetadata::new(backend),
        geometry,
        norm_plan,
        weight_plan,
        FeatureSource::Output,
        class,
    )?
    .seal()
    .map_err(error)
}

fn checked_readout_class_draft(
    graph: NativeGraphMetadata,
    geometry: &Decoder,
    norm_plan: &crate::WeightPlan,
    weight_plan: Option<&crate::WeightPlan>,
    source: FeatureSource<'_>,
    class: ReadoutClass,
) -> Result<NativeGraphMetadata, String> {
    let activation = match geometry.activation_dtype {
        magnitude_family_contracts::ActivationDType::F16 => Element::f16(),
        magnitude_family_contracts::ActivationDType::BF16 => Element::bf16(),
    };
    let feature_elements = [("NW", norm_plan.resident), ("A", activation)];
    let fusion_elements = match source {
        FeatureSource::Output => Vec::new(),
        FeatureSource::Taps { fusion } => vec![
            ("A", activation),
            ("W", fusion.resident),
            ("Y", Element::f32()),
        ],
    };
    let tap_feature_elements = [("A", activation)];
    let entries = match source {
        FeatureSource::Output => FeatureEntries::Output(&feature_elements[..]),
        FeatureSource::Taps { .. } => FeatureEntries::Taps {
            fusion: &fusion_elements[..],
            features: &tap_feature_elements[..],
        },
    };
    let (graph, final_rows, _, _, _) =
        feature_topology(graph, entries, geometry, norm_plan, source, class)?;
    let graph = if class.kind == ReadoutKind::Features {
        graph
    } else {
        let FinalRowPorts { hidden, norm } =
            final_rows.ok_or("a projecting readout has no final rows")?;
        let weight_plan = weight_plan.ok_or("projected readout weight is absent")?;
        let head_elements = [
            ("NW", norm_plan.resident),
            ("OW", weight_plan.resident),
            ("A", activation),
        ];
        let (graph, _, _, logits) = projected_topology(
            graph,
            &head_elements,
            geometry,
            weight_plan,
            class,
            &hidden,
            &norm,
        )?;
        if matches!(class.kind, ReadoutKind::Selection { .. }) {
            let sample_elements: [(&str, Element); 0] = [];
            let (graph, _, _) = selected_topology(
                graph,
                &sample_elements,
                &sample_elements,
                geometry,
                class,
                &logits,
            )?;
            graph
        } else {
            graph
        }
    };
    Ok(graph)
}

#[cfg(test)]
pub(crate) fn checked_projected_graph_storage(
    backend: BackendName,
    geometry: &Decoder,
    norm_plan: &crate::WeightPlan,
    weight_plan: &crate::WeightPlan,
    rows: u64,
    outputs: u64,
    projected: u64,
) -> Result<NativeGraphStorageBytes, String> {
    checked_readout_class_storage(
        backend,
        geometry,
        norm_plan,
        Some(weight_plan),
        ReadoutClass {
            rows,
            outputs,
            projected,
            selected: 0,
            kind: ReadoutKind::Logits,
        },
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn checked_selection_graph_storage(
    backend: BackendName,
    geometry: &Decoder,
    norm_plan: &crate::WeightPlan,
    weight_plan: &crate::WeightPlan,
    rows: u64,
    outputs: u64,
    projected: u64,
    selected: u64,
    shaped: bool,
) -> Result<NativeGraphStorageBytes, String> {
    checked_readout_class_storage(
        backend,
        geometry,
        norm_plan,
        Some(weight_plan),
        ReadoutClass {
            rows,
            outputs,
            projected,
            selected,
            kind: ReadoutKind::Selection { shaped },
        },
    )
}

pub(crate) fn checked_readout_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    limits: ResourceLimits,
) -> Result<NativeGraphStorageBytes, String> {
    let weight = |kind| {
        load.weights()
            .find(|weight| {
                weight.role
                    == WeightRole {
                        scope: WeightScope::Target,
                        kind,
                    }
            })
            .ok_or_else(|| format!("readout {kind:?} weight is absent"))
    };
    let norm = weight(WeightKind::OutputNorm)?;
    let projection = weight(WeightKind::Output)?;
    let regimes = certify_readout_regimes(
        backend,
        geometry,
        norm,
        projection,
        FeatureSource::of(load),
        limits,
    )?;
    regimes
        .values()
        .map(NativeGraphLayout::storage_bytes)
        .reduce(|previous, bytes| NativeGraphStorageBytes {
            workspace: previous.workspace.max(bytes.workspace),
            output: previous.output.max(bytes.output),
            upload: previous.upload.max(bytes.upload),
        })
        .ok_or_else(|| "readout graph family has no classes".into())
}

/// The structure a readout class selects: its kind and, for a selection,
/// the selected-row count its sampling view is built from.
type ReadoutRegime = (ReadoutKind, u64);

fn readout_regime(class: ReadoutClass) -> ReadoutRegime {
    match class.kind {
        ReadoutKind::Selection { .. } => (class.kind, class.selected),
        kind => (kind, 0),
    }
}

fn certify_readout_regimes(
    backend: BackendName,
    geometry: &Decoder,
    norm: &crate::WeightPlan,
    projection: &crate::WeightPlan,
    source: FeatureSource<'_>,
    limits: ResourceLimits,
) -> Result<BTreeMap<ReadoutRegime, NativeGraphLayout>, String> {
    // The entry whose `O` is the class's feature outputs.
    let features_entry = match source {
        FeatureSource::Output => readout_features_rows::Entry::NAME,
        FeatureSource::Taps { .. } => feature_rows::Entry::NAME,
    };
    let mut regimes: BTreeMap<ReadoutRegime, Vec<ReadoutClass>> = BTreeMap::new();
    for class in readout_classes(limits)? {
        regimes
            .entry(readout_regime(class))
            .or_default()
            .push(class);
    }
    regimes
        .into_iter()
        .map(|((kind, selected), classes)| {
            let largest = |field: fn(&ReadoutClass) -> u64| {
                classes
                    .iter()
                    .map(field)
                    .max()
                    .expect("a regime holds a class")
            };
            let template = ReadoutClass {
                rows: largest(|class| class.rows),
                outputs: largest(|class| class.outputs),
                projected: largest(|class| class.projected),
                selected,
                kind,
            };
            let slices = classes
                .iter()
                .map(|class| {
                    let slice = NativeGraphClassSlice::new()
                        .dimension("M", [class.rows])
                        .scoped(features_entry, "O", [class.outputs]);
                    match class.kind {
                        ReadoutKind::Features => slice,
                        ReadoutKind::Logits => {
                            slice.scoped(readout_head_rows::Entry::NAME, "O", [class.projected])
                        }
                        ReadoutKind::Selection { .. } => slice
                            .scoped(readout_head_rows::Entry::NAME, "O", [class.projected])
                            .scoped(sample_rows::Entry::NAME, "M", [class.selected]),
                    }
                })
                .collect::<Vec<_>>();
            let layout = checked_readout_class_draft(
                NativeGraphMetadata::new_template(backend),
                geometry,
                norm,
                Some(projection),
                source,
                template,
            )?
            .seal_template()
            .and_then(|template| template.certify(&slices))
            .map_err(error)?;
            Ok(((kind, selected), layout))
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn checked_features_graph_storage(
    backend: BackendName,
    geometry: &Decoder,
    norm_plan: &crate::WeightPlan,
    rows: u64,
    outputs: u64,
) -> Result<NativeGraphStorageBytes, String> {
    checked_readout_class_storage(
        backend,
        geometry,
        norm_plan,
        None,
        ReadoutClass {
            rows,
            outputs,
            projected: 0,
            selected: 0,
            kind: ReadoutKind::Features,
        },
    )
}

/// Sampling of `rows` logits rows into `result`, after `shape_rows` when
/// `shaped`. The target readout and the draft head select through this one
/// node sequence and control layout. A constrained row's mask applies before
/// shaping cuts (shaping reads the same mask and flag inputs as sampling), so
/// top-k, min-p and top-p rank only admitted tokens.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    shape: G::Binding<'a, shape_rows::Entry>,
    sample: G::Binding<'a, sample_rows::Entry>,
    vocabulary: u64,
    logits: WorkflowTensorRef<'_>,
    rows: u64,
    shaped: bool,
    result: WorkflowTensorMut<'_>,
) -> Result<SelectionPorts, String> {
    let sample_dims = [("M", rows), ("V", vocabulary)];
    let mask = graph.input_for(sample, "mask", &sample_dims)?;
    let constrained = graph.input_for(sample, "constrained", &sample_dims)?;
    let draws = graph.input_for(sample, "draws", &sample_dims)?;
    let shaping = if shaped {
        let shape_dims = [("Sx", rows), ("V", vocabulary), ("Hn", HISTORY_TOKENS)];
        let parameters = graph.input_for(shape, "params", &shape_dims)?;
        let history = graph.input_for(shape, "history", &shape_dims)?;
        let mut out = graph.local_for(shape, "out", &shape_dims)?;
        graph.enqueue::<shape_rows::Entry>(
            shape,
            &shape_dims,
            shape_rows::WorkflowArgs {
                logits,
                mask: mask.tensor().into(),
                constrained: constrained.tensor().into(),
                params: parameters.tensor().into(),
                history: history.tensor().into(),
                out: out.tensor_mut().into(),
            },
        )?;
        graph.enqueue::<sample_rows::Entry>(
            sample,
            &sample_dims,
            sample_rows::WorkflowArgs {
                logits: out.tensor().into(),
                mask: mask.tensor().into(),
                constrained: constrained.tensor().into(),
                draws: draws.tensor().into(),
                result,
            },
        )?;
        Some(ShapingPorts {
            parameters,
            history,
        })
    } else {
        graph.enqueue::<sample_rows::Entry>(
            sample,
            &sample_dims,
            sample_rows::WorkflowArgs {
                logits,
                mask: mask.tensor().into(),
                constrained: constrained.tensor().into(),
                draws: draws.tensor().into(),
                result,
            },
        )?;
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
    let actual_selected = batch.select_rows.len();
    let sources = (0..selected_class)
        .map(|index| if index < actual_selected { index } else { 0 })
        .collect::<Vec<_>>();
    write_selection_rows(batch, active, ports, &sources, row_words)
}

/// Selection controls whose graph row `j` takes the pass's packed selection
/// `sources[j]`.
pub(crate) fn write_selection_rows(
    batch: &TargetBatchUpload<'_>,
    active: &mut seismic::NativeGraphFamilyActive<'_>,
    ports: &SelectionPorts,
    sources: &[usize],
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
    let selected_class = sources.len();
    if let Some(shaping) = &ports.shaping {
        let parameters = sources
            .iter()
            .flat_map(|&source| batch.shaping[source])
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        active
            .write_input(&shaping.parameters, &parameters)
            .map_err(device)?;
        let history = sources
            .iter()
            .flat_map(|&source| batch.history[source])
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        active
            .write_input(&shaping.history, &history)
            .map_err(device)?;
    }
    let draws = sources
        .iter()
        .flat_map(|&source| batch.draws[source])
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    active.write_input(&ports.draws, &draws).map_err(device)?;
    let mask_rows = sources
        .iter()
        .map(|&source| batch.mask_rows[source])
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

#[cfg(test)]
mod resource_regime_tests {
    use super::*;
    use crate::ComponentSelection;
    use seismic::Layout;

    #[test]
    fn every_readout_class_fits_its_shared_regime_layout() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let weight = |kind| {
            load.weights()
                .find(|weight| {
                    weight.role
                        == WeightRole {
                            scope: WeightScope::Target,
                            kind,
                        }
                })
                .unwrap()
        };
        let limits = ResourceLimits {
            max_launch_rows: 512,
            max_launch_slots: 32,
            max_projected_rows: 32,
            max_images_per_request: 0,
            lookahead: false,
        };
        let classes = readout_classes(limits).unwrap();
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Cuda,
            BackendName::Vulkan,
        ] {
            let regimes = certify_readout_regimes(
                backend,
                &definition.decoder,
                weight(WeightKind::OutputNorm),
                weight(WeightKind::Output),
                FeatureSource::Output,
                limits,
            )
            .unwrap();
            for &class in &classes {
                let layout = &regimes[&readout_regime(class)];
                let bytes = checked_readout_class_draft(
                    NativeGraphMetadata::new(backend),
                    &definition.decoder,
                    weight(WeightKind::OutputNorm),
                    Some(weight(WeightKind::Output)),
                    FeatureSource::Output,
                    class,
                )
                .unwrap()
                .seal_with_layout(layout)
                .unwrap();
                assert_eq!(bytes, layout.storage_bytes(), "{backend:?} {class:?}");
            }
        }
    }
}
