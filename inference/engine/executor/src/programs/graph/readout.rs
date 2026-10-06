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

use super::GraphError;
use crate::{
    native::{AttestedTarget, ReadoutHeadKernels},
    programs::graph::draft::GraphDraft,
    DeviceError, InvariantError, ModelLoadPlan, ResidentOutput, ResidentTarget, ResourceLimits,
    SubmitError,
};
use magnitude_batching::TargetBatchUpload;
use magnitude_family_contracts::{
    Decoder, ExitNorm, ProgressivePlane, WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    feature_rows, project_rows, readout_exact_rows, readout_features_rows, readout_head_rows,
    readout_planes_rows, readout_refine_rows, readout_top_rows, sample_rows, shape_rows,
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
    /// `certified` graphs project a progressive head through its certified
    /// levels (`readout_top_rows`, `readout_refine_rows`,
    /// `readout_exact_rows`): every row that can be selected keeps its exact
    /// logit and the others -inf, so the selection is the full readout's.
    /// Their rows are unpenalized and uncut (shaping is a temperature at
    /// most), and no row reads its logits on the host.
    Selection {
        shaped: bool,
        certified: bool,
    },
}

/// Whether `backend` declares every progressive head readout entry.
pub(crate) fn reads_progressive_heads(backend: BackendName) -> Result<bool, String> {
    use seismic::generated::native_implementation_for_backend as implementation;
    let declared = [
        implementation::<readout_top_rows::Entry>(backend).map(|native| native.is_some()),
        implementation::<readout_refine_rows::Entry>(backend).map(|native| native.is_some()),
        implementation::<readout_exact_rows::Entry>(backend).map(|native| native.is_some()),
        implementation::<readout_planes_rows::Entry>(backend).map(|native| native.is_some()),
    ];
    declared.into_iter().try_fold(true, |all, declared| {
        Ok(all && declared.map_err(|error| error.to_string())?)
    })
}

/// The most selected rows a certified selection class serves on `backend`:
/// the row counts whose levels take less time than the full pass. Metal's
/// batched projection is bound by its matrix arithmetic, not the bytes it
/// reads, so only a single row gains; CUDA's levels gain to four rows. Zero
/// on a backend without the levels.
pub(crate) fn certified_rows(backend: BackendName) -> u64 {
    match backend {
        BackendName::Metal => 1,
        BackendName::Cuda => 4,
        BackendName::Cpu | BackendName::Vulkan => 0,
    }
}

/// The planned vocabulary projection of a readout.
#[derive(Clone, Copy)]
pub(crate) enum HeadPlans<'p> {
    /// The output projection.
    Packed(&'p crate::WeightPlan),
    /// Its progressive planes, in `ProgressivePlane::ALL` order.
    Progressive([&'p crate::WeightPlan; 5]),
}

impl<'p> HeadPlans<'p> {
    pub(crate) fn of(load: &'p ModelLoadPlan) -> Result<Self, String> {
        let weight = |kind| {
            load.weights().find(|weight| {
                weight.role
                    == WeightRole {
                        scope: WeightScope::Target,
                        kind,
                    }
            })
        };
        if let Some(output) = weight(WeightKind::Output) {
            return Ok(Self::Packed(output));
        }
        let [top, bit3, rest, scales, radius] = ProgressivePlane::ALL.map(|plane| {
            weight(WeightKind::OutputPlane(plane)).ok_or("readout output projection weight is absent")
        });
        Ok(Self::Progressive([top?, bit3?, rest?, scales?, radius?]))
    }

    /// The most selected rows its certified classes serve on `backend`:
    /// none for a packed head.
    fn certified_rows(self, backend: BackendName) -> u64 {
        match self {
            Self::Packed(_) => 0,
            Self::Progressive(_) => certified_rows(backend),
        }
    }

    fn plane(planes: &[&'p crate::WeightPlan; 5], plane: ProgressivePlane) -> &'p crate::WeightPlan {
        planes[plane as usize]
    }
}

/// A readout graph's head weight ports, by its `HeadPlans`.
#[derive(Clone)]
pub(crate) enum HeadPorts {
    /// The output projection and its accumulator-scale port.
    Packed { weight: NativePort, scale: NativePort },
    /// The planes the graph reads.
    Progressive(Vec<(ProgressivePlane, NativePort)>),
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

/// Whether a certified class serves a row with these shaping parameters: no
/// penalty, and no top-k, top-p or min-p cut unless the row is greedy (a cut
/// always keeps the largest logit), so shaping is at most a temperature,
/// which the certified levels apply to their bounds.
pub(crate) fn certifies(parameters: &[f32; 8]) -> bool {
    let [temperature, top_k, top_p, min_p, repetition, presence, frequency, _] = *parameters;
    let penalized = repetition != 1.0 || presence != 0.0 || frequency != 0.0;
    let cut = top_k > 0.0 || top_p < 1.0 || min_p > 0.0;
    !penalized && (temperature == 0.0 || !cut)
}

/// A certified level's score divisor for a row's temperature: the row's
/// temperature, or 1 for a greedy row (temperature 0), whose score is its
/// logit.
fn score_divisor(temperature: f32) -> f32 {
    if temperature > 0.0 {
        temperature
    } else {
        1.0
    }
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
    /// The vocabulary projection's ports, in projecting classes.
    pub head: Option<HeadPorts>,
    /// Hidden rows of the feature outputs.
    pub out_rows: NativePort,
    /// Hidden rows of the projected outputs, selected outputs first.
    pub logit_rows: Option<NativePort>,
    pub selection: Option<SelectionPorts>,
    pub features: WorkflowTensor,
    /// The exported logits of a diagnostic load's projecting class.
    pub logits: Option<WorkflowTensor>,
    pub selected: Option<WorkflowTensor>,
}

#[derive(Clone)]
pub struct PreparedTargetReadoutGraphs {
    classes: BTreeMap<ReadoutClass, PreparedTargetReadoutGraph>,
    family: NativeGraphFamily,
    max_projected_rows: usize,
    certified_rows: u64,
}

pub(crate) struct BoundTargetReadoutGraphs {
    pub prepared: PreparedTargetReadoutGraphs,
    bound: BTreeMap<ReadoutClass, BoundNativeGraphPlan>,
}

/// Whether the load's projecting readout classes export their logits: only
/// a diagnostic load, which declares rows to export, does. A served load's
/// logits are graph locals its selection consumes.
pub(crate) fn exports_logits(limits: ResourceLimits) -> bool {
    limits.exported_logits_rows > 0
}

/// The most rows one readout projects: the exported rows of a diagnostic
/// load, otherwise the selection bound, since a served load projects exactly
/// the rows it selects.
pub(crate) fn max_projected_rows(limits: ResourceLimits) -> usize {
    if exports_logits(limits) {
        limits.exported_logits_rows
    } else {
        limits.max_selected_rows
    }
}

/// The most selected rows a load's certified selection classes serve: the
/// head's bound on `backend` for a served load, none for a diagnostic load,
/// whose selections read the logits it exports.
fn load_certified_rows(limits: ResourceLimits, head: HeadPlans<'_>, backend: BackendName) -> u64 {
    if exports_logits(limits) {
        0
    } else {
        head.certified_rows(backend)
    }
}

/// A served load's classes are its features and its selections, whose
/// projected rows are their selected rows. A diagnostic load's projecting
/// classes export their logits: a logits class per exported row count, and
/// unshaped selection of its leading rows, since shaping must not rewrite
/// the logits it exports. A served load also has certified selections of up
/// to `certified_rows` rows.
fn readout_classes(limits: ResourceLimits, certified_rows: u64) -> Result<Vec<ReadoutClass>, String> {
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
            if exports_logits(limits) {
                for projected in ladder((outputs as usize).min(limits.exported_logits_rows)) {
                    classes.push(ReadoutClass {
                        rows,
                        outputs,
                        projected,
                        selected: 0,
                        kind: ReadoutKind::Logits,
                    });
                    for selected in ladder((projected as usize).min(limits.max_selected_rows)) {
                        classes.push(ReadoutClass {
                            rows,
                            outputs,
                            projected,
                            selected,
                            kind: ReadoutKind::Selection {
                                shaped: false,
                                certified: false,
                            },
                        });
                    }
                }
            } else {
                for selected in ladder((outputs as usize).min(limits.max_selected_rows)) {
                    let certifying = selected <= certified_rows;
                    for certified in [false, true] {
                        if certified && !certifying {
                            continue;
                        }
                        for shaped in [false, true] {
                            classes.push(ReadoutClass {
                                rows,
                                outputs,
                                projected: selected,
                                selected,
                                kind: ReadoutKind::Selection { shaped, certified },
                            });
                        }
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
        let head = HeadPlans::of(load)?;
        let source = FeatureSource::of(load);
        let regimes =
            certify_readout_regimes(device.backend(), geometry, norm, head, source, limits)
                .map_err(error)?;
        let mut classes = BTreeMap::new();
        let mut plans = Vec::new();
        let mut add = |class: ReadoutClass| -> Result<(), String> {
            let layout = regimes
                .get(&readout_regime(class))
                .ok_or("readout class has no resource regime")?;
            let variant = PreparedTargetReadoutGraph::prepare(
                device,
                target,
                geometry,
                norm,
                head,
                source,
                class,
                exports_logits(limits),
                layout,
            )?;
            plans.push(variant.plan.clone());
            classes.insert(class, variant);
            Ok(())
        };
        let certified_rows = load_certified_rows(limits, head, device.backend());
        for class in readout_classes(limits, certified_rows)? {
            add(class)?;
        }
        let family = NativeGraphFamily::new(&plans).map_err(|error| error.to_string())?;
        Ok(Self {
            classes,
            family,
            max_projected_rows: max_projected_rows(limits),
            certified_rows,
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

    /// The most selected rows the family's certified selection classes
    /// serve (none without them).
    pub(crate) fn certified_rows(&self) -> u64 {
        self.certified_rows
    }

    pub(crate) fn bind_weights(
        self,
        resident: &ResidentTarget,
    ) -> Result<BoundTargetReadoutGraphs, String> {
        let mut bound = BTreeMap::new();
        for (class, graph) in &self.classes {
            let mut fixed = Vec::new();
            let absent_scale = seismic::Tensor::from_host(
                &resident.output_norm.tensor().device(), Element::f32(), &[0], &[],
            ).map_err(|error| error.to_string())?;
            if let Some(rows) = &graph.final_rows {
                fixed.push((&rows.norm, resident.output_norm.tensor()));
            }
            match (&graph.head, &resident.output) {
                (None, _) => {}
                (Some(HeadPorts::Packed { weight, scale }), ResidentOutput::Packed(output)) => {
                    fixed.push((weight, output.tensor()));
                    fixed.push((scale, output.scale().unwrap_or(&absent_scale)));
                }
                (Some(HeadPorts::Progressive(ports)), ResidentOutput::Progressive(planes)) => {
                    for (plane, port) in ports {
                        fixed.push((port, planes.plane(*plane).tensor()));
                    }
                }
                _ => return Err("the readout graph's head differs from the resident head".into()),
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
    /// Each row's score divisor (its temperature, 1 for a greedy row); in
    /// certified classes, whose levels score rows as the sampler does.
    pub temperature: Option<NativePort>,
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
        head_plans: HeadPlans<'_>,
        source: FeatureSource<'_>,
        class: ReadoutClass,
        export: bool,
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
        )
        .map_err(error)?;
        let mut head = None;
        let mut logit_rows = None;
        let mut logits = None;
        let mut selection = None;
        let mut selected = None;
        if class.kind != ReadoutKind::Features {
            let rows = final_rows
                .as_ref()
                .ok_or("a projecting readout has no final rows")?;
            let entries = match &target.readout.head {
                ReadoutHeadKernels::Packed { head, .. } => HeadEntries::Packed(head),
                ReadoutHeadKernels::Progressive(kernels) => HeadEntries::Progressive {
                    top: &kernels.top,
                    refine: &kernels.refine,
                    exact: &kernels.exact,
                    planes: &kernels.planes,
                },
            };
            let projected = head_topology(
                graph,
                entries,
                (&target.shape, &target.sample),
                geometry,
                head_plans,
                class,
                export,
                rows,
            )
            .map_err(error)?;
            graph = projected.graph;
            head = Some(projected.head);
            logit_rows = Some(projected.rows);
            logits = projected.logits;
            selection = projected.selection;
            selected = projected.selected;
        }
        let plan = graph.seal().map_err(error)?;
        Ok(Self {
            plan,
            final_rows,
            taps,
            head,
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
) -> Result<FinalRowPorts, GraphError> {
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
) -> Result<FeaturePrefix<G>, GraphError> {
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

/// The head entries of a readout graph, by its head plans.
enum HeadEntries<'a, G: GraphDraft + 'a> {
    Packed(G::Binding<'a, readout_head_rows::Entry>),
    Progressive {
        top: G::Binding<'a, readout_top_rows::Entry>,
        refine: G::Binding<'a, readout_refine_rows::Entry>,
        exact: G::Binding<'a, readout_exact_rows::Entry>,
        planes: G::Binding<'a, readout_planes_rows::Entry>,
    },
}

/// A progressive head's projection entries.
pub(crate) struct ProgressiveEntries<'a, G: GraphDraft + 'a> {
    pub top: G::Binding<'a, readout_top_rows::Entry>,
    pub refine: G::Binding<'a, readout_refine_rows::Entry>,
    pub exact: G::Binding<'a, readout_exact_rows::Entry>,
    pub planes: G::Binding<'a, readout_planes_rows::Entry>,
}

/// The planes a progressive projection reads: the radii only when certified.
pub(crate) struct ProgressivePorts<'p> {
    pub top: &'p WorkflowTensor,
    pub bit3: &'p WorkflowTensor,
    pub rest: &'p WorkflowTensor,
    pub scales: &'p WorkflowTensor,
    pub radius: Option<&'p WorkflowTensor>,
}

/// The rows a progressive projection projects: `out_rows` picks `projected`
/// of the `rows` hidden rows (normed by `norm`), the first `selected` of
/// which a certified projection selects.
pub(crate) struct ProgressiveRows<'p> {
    pub hidden: &'p WorkflowTensor,
    pub norm: &'p WorkflowTensor,
    pub out_rows: &'p WorkflowTensor,
    pub rows: u64,
    pub projected: u64,
    pub selected: u64,
}

/// The projection of `rows` onto a progressive head of `[vocabulary,
/// hidden]`: with the radii, a certified selection's three levels (logits
/// exact at its survivors, -inf elsewhere) and the sampler inputs they read,
/// score divisors included; otherwise the full exact pass.
pub(crate) fn progressive_projection<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: ProgressiveEntries<'a, G>,
    sampler: G::Binding<'a, sample_rows::Entry>,
    planes: ProgressivePorts<'_>,
    rows: ProgressiveRows<'_>,
    (vocabulary, hidden): (u64, u64),
    epsilon: f32,
) -> Result<(WorkflowTensor, Option<SelectionInputs>), GraphError> {
    let dimensions = [
        ("M", rows.rows),
        ("O", rows.projected),
        ("V", vocabulary),
        ("D", hidden),
    ];
    let Some(radius) = planes.radius else {
        let logits = graph
            .enqueue::<readout_planes_rows::Entry>(
                entries.planes,
                &dimensions,
                readout_planes_rows::WorkflowArgs {
                    hidden: rows.hidden.into(),
                    norm: rows.norm.into(),
                    top: planes.top.into(),
                    bit3: planes.bit3.into(),
                    rest: planes.rest.into(),
                    scales: planes.scales.into(),
                    out_rows: rows.out_rows.into(),
                    epsilon,
                },
            )?
            .value;
        return Ok((logits, None));
    };
    let selecting = SelectionInputs {
        temperature: Some(graph.input_for(entries.top, "temperature", &dimensions)?),
        ..selection_inputs(graph, sampler, rows.selected, vocabulary)?
    };
    let (draws, temperature, mask, constrained) = (
        selecting.draws.tensor(),
        selecting
            .temperature
            .as_ref()
            .expect("a certified projection scores by temperature")
            .tensor(),
        selecting.mask.tensor(),
        selecting.constrained.tensor(),
    );
    let first = graph.enqueue::<readout_top_rows::Entry>(
        entries.top,
        &dimensions,
        readout_top_rows::WorkflowArgs {
            hidden: rows.hidden.into(),
            norm: rows.norm.into(),
            top: planes.top.into(),
            bit3: planes.bit3.into(),
            rest: planes.rest.into(),
            scales: planes.scales.into(),
            radius: radius.into(),
            out_rows: rows.out_rows.into(),
            draws: draws.into(),
            temperature: temperature.into(),
            mask: mask.into(),
            constrained: constrained.into(),
            epsilon,
        },
    )?;
    let levels = [("O", rows.projected), ("V", vocabulary), ("D", hidden)];
    let second = graph.enqueue::<readout_refine_rows::Entry>(
        entries.refine,
        &levels,
        readout_refine_rows::WorkflowArgs {
            features: (&first.r2).into(),
            top: planes.top.into(),
            bit3: planes.bit3.into(),
            rest: planes.rest.into(),
            scales: planes.scales.into(),
            radius: radius.into(),
            coarse: (&first.r0).into(),
            floor: (&first.r1).into(),
            length: (&first.r3).into(),
            draws: draws.into(),
            temperature: temperature.into(),
            mask: mask.into(),
            constrained: constrained.into(),
        },
    )?;
    let logits = graph
        .enqueue::<readout_exact_rows::Entry>(
            entries.exact,
            &levels,
            readout_exact_rows::WorkflowArgs {
                features: (&first.r2).into(),
                top: planes.top.into(),
                bit3: planes.bit3.into(),
                rest: planes.rest.into(),
                scales: planes.scales.into(),
                radius: radius.into(),
                fine: (&second.r0).into(),
                floor: (&second.r1).into(),
                length: (&first.r3).into(),
                draws: draws.into(),
                temperature: temperature.into(),
                mask: mask.into(),
                constrained: constrained.into(),
            },
        )?
        .value;
    Ok((logits, Some(selecting)))
}

/// A projecting readout's head and selection suffix.
struct Projected<G> {
    graph: G,
    head: HeadPorts,
    /// Hidden rows of the projected outputs.
    rows: NativePort,
    /// The logits, exported by a diagnostic load (`export`).
    logits: Option<WorkflowTensor>,
    selection: Option<SelectionPorts>,
    selected: Option<WorkflowTensor>,
}

/// The projection of the class's projected rows (the head as placed, or a
/// certified class's levels over the progressive planes), then its
/// selection. The logits are exported only by a diagnostic load (`export`);
/// otherwise they are a graph local its selection consumes (and shaping
/// rewrites in place).
#[allow(clippy::too_many_arguments)]
fn head_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entries: HeadEntries<'a, G>,
    (shape, sampler): (
        G::Binding<'a, shape_rows::Entry>,
        G::Binding<'a, sample_rows::Entry>,
    ),
    geometry: &Decoder,
    plans: HeadPlans<'_>,
    class: ReadoutClass,
    export: bool,
    final_rows: &FinalRowPorts,
) -> Result<Projected<G>, GraphError> {
    let (hidden, norm) = (&final_rows.hidden, &final_rows.norm);
    let epsilon = readout_epsilon(geometry)?;
    let vocabulary = geometry.vocabulary;
    let dimensions = [
        ("M", class.rows),
        ("O", class.projected),
        ("V", vocabulary),
        ("D", geometry.hidden),
    ];
    let certified = matches!(
        class.kind,
        ReadoutKind::Selection {
            certified: true,
            ..
        }
    );
    let mut inputs = None;
    let (head, rows, logits) = match (entries, plans) {
        (HeadEntries::Packed(entry), HeadPlans::Packed(plan)) if !certified => {
            let weight = graph.port(plan.resident, &plan.shape)?;
            // The weight's resident second-level scale, or an absent scale.
            let extent = plan.scale_extent();
            let scale = graph.port(Element::f32(), &[extent])?;
            let dimensions = [
                ("M", class.rows),
                ("O", class.projected),
                ("V", vocabulary),
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
                        epsilon,
                        softcap: readout_softcap(geometry),
                        weight_scale: scale.tensor().into(),
                    },
                )?
                .value;
            (HeadPorts::Packed { weight, scale }, rows, logits)
        }
        (
            HeadEntries::Progressive {
                top,
                refine,
                exact,
                planes,
            },
            HeadPlans::Progressive(plans),
        ) => {
            let mut ports = Vec::new();
            let mut port = |graph: &mut G, plane| -> Result<NativePort, GraphError> {
                let plan = HeadPlans::plane(&plans, plane);
                let port = graph.port(plan.resident, &plan.shape)?;
                ports.push((plane, port.clone()));
                Ok(port)
            };
            let top_plane = port(&mut graph, ProgressivePlane::Top)?;
            let bit3 = port(&mut graph, ProgressivePlane::Bit3)?;
            let rest = port(&mut graph, ProgressivePlane::Rest)?;
            let scales = port(&mut graph, ProgressivePlane::Scales)?;
            let radius = certified
                .then(|| port(&mut graph, ProgressivePlane::Radius))
                .transpose()?;
            let rows = if certified {
                graph.input_for(top, "out_rows", &dimensions)?
            } else {
                graph.input_for(planes, "out_rows", &dimensions)?
            };
            let (logits, selecting) = progressive_projection(
                &mut graph,
                ProgressiveEntries {
                    top,
                    refine,
                    exact,
                    planes,
                },
                sampler,
                ProgressivePorts {
                    top: top_plane.tensor(),
                    bit3: bit3.tensor(),
                    rest: rest.tensor(),
                    scales: scales.tensor(),
                    radius: radius.as_ref().map(NativePort::tensor),
                },
                ProgressiveRows {
                    hidden: hidden.tensor(),
                    norm: norm.tensor(),
                    out_rows: rows.tensor(),
                    rows: class.rows,
                    projected: class.projected,
                    selected: class.selected,
                },
                (vocabulary, geometry.hidden),
                epsilon,
            )?;
            inputs = selecting;
            (HeadPorts::Progressive(ports), rows, logits)
        }
        _ => return Err("readout head entries disagree with the head plans".into()),
    };
    if export {
        graph.export(&logits)?;
    }
    let exported = export.then(|| logits.clone());
    let ReadoutKind::Selection { shaped, .. } = class.kind else {
        return Ok(Projected {
            graph,
            head,
            rows,
            logits: exported,
            selection: None,
            selected: None,
        });
    };
    let inputs = match inputs {
        Some(inputs) => inputs,
        None => selection_inputs(&mut graph, sampler, class.selected, vocabulary)?,
    };
    let mut result = graph.local_for(
        sampler,
        "result",
        &[("M", class.selected), ("V", vocabulary)],
    )?;
    let mut leading = logits.slice_leading(0, class.selected);
    let ports = sample(
        &mut graph,
        shape,
        sampler,
        vocabulary,
        &mut leading,
        class.selected,
        shaped,
        inputs,
        result.tensor_mut().into(),
    )?;
    graph.export(result.tensor())?;
    let selected = result.tensor().clone();
    Ok(Projected {
        graph,
        head,
        rows,
        logits: exported,
        selection: Some(ports),
        selected: Some(selected),
    })
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

#[cfg(test)]
fn checked_readout_class_storage(
    backend: BackendName,
    geometry: &Decoder,
    norm_plan: &crate::WeightPlan,
    head: Option<HeadPlans<'_>>,
    class: ReadoutClass,
    export: bool,
) -> Result<NativeGraphStorageBytes, GraphError> {
    Ok(checked_readout_class_draft(
        NativeGraphMetadata::new(backend),
        geometry,
        norm_plan,
        head,
        FeatureSource::Output,
        class,
        export,
    )?
    .seal()?)
}

fn checked_readout_class_draft(
    graph: NativeGraphMetadata,
    geometry: &Decoder,
    norm_plan: &crate::WeightPlan,
    head: Option<HeadPlans<'_>>,
    source: FeatureSource<'_>,
    class: ReadoutClass,
    export: bool,
) -> Result<NativeGraphMetadata, GraphError> {
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
    if class.kind == ReadoutKind::Features {
        return Ok(graph);
    }
    let rows = final_rows.ok_or("a projecting readout has no final rows")?;
    let head = head.ok_or("projected readout weight is absent")?;
    let packed_elements = match head {
        HeadPlans::Packed(weight) => vec![
            ("NW", norm_plan.resident),
            ("OW", weight.resident),
            ("A", activation),
        ],
        HeadPlans::Progressive(_) => Vec::new(),
    };
    let normed_elements = [("NW", norm_plan.resident), ("A", activation)];
    let level_elements = [("A", activation)];
    let entries = match head {
        HeadPlans::Packed(_) => HeadEntries::Packed(&packed_elements[..]),
        HeadPlans::Progressive(_) => HeadEntries::Progressive {
            top: &normed_elements[..],
            refine: &level_elements[..],
            exact: &level_elements[..],
            planes: &normed_elements[..],
        },
    };
    let sample_elements: [(&str, Element); 0] = [];
    Ok(head_topology(
        graph,
        entries,
        (&sample_elements[..], &sample_elements[..]),
        geometry,
        head,
        class,
        export,
        &rows,
    )?
    .graph)
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
) -> Result<NativeGraphStorageBytes, GraphError> {
    checked_readout_class_storage(
        backend,
        geometry,
        norm_plan,
        Some(HeadPlans::Packed(weight_plan)),
        ReadoutClass {
            rows,
            outputs,
            projected,
            selected: 0,
            kind: ReadoutKind::Logits,
        },
        true,
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
    selected: u64,
    shaped: bool,
) -> Result<NativeGraphStorageBytes, GraphError> {
    checked_readout_class_storage(
        backend,
        geometry,
        norm_plan,
        Some(HeadPlans::Packed(weight_plan)),
        ReadoutClass {
            rows,
            outputs,
            projected: selected,
            selected,
            kind: ReadoutKind::Selection {
                shaped,
                certified: false,
            },
        },
        false,
    )
}

pub(crate) fn checked_readout_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    limits: ResourceLimits,
) -> Result<NativeGraphStorageBytes, GraphError> {
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
    let head = HeadPlans::of(load)?;
    let regimes = certify_readout_regimes(
        backend,
        geometry,
        norm,
        head,
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
    head: HeadPlans<'_>,
    source: FeatureSource<'_>,
    limits: ResourceLimits,
) -> Result<BTreeMap<ReadoutRegime, NativeGraphLayout>, GraphError> {
    // The entry whose `O` is the class's feature outputs.
    let features_entry = match source {
        FeatureSource::Output => readout_features_rows::Entry::NAME,
        FeatureSource::Taps { .. } => feature_rows::Entry::NAME,
    };
    let mut regimes: BTreeMap<ReadoutRegime, Vec<ReadoutClass>> = BTreeMap::new();
    // The entries whose `O` is the class's projected rows.
    let projecting: &[&'static str] = match head {
        HeadPlans::Packed(_) => &[readout_head_rows::Entry::NAME],
        HeadPlans::Progressive(_) => &[readout_planes_rows::Entry::NAME],
    };
    let certifying = [
        readout_top_rows::Entry::NAME,
        readout_refine_rows::Entry::NAME,
        readout_exact_rows::Entry::NAME,
    ];
    for class in readout_classes(limits, load_certified_rows(limits, head, backend))? {
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
                    let scoped = |slice: NativeGraphClassSlice, entries: &[&'static str]| {
                        entries.iter().fold(slice, |slice, entry| {
                            slice.scoped(entry, "O", [class.projected])
                        })
                    };
                    match class.kind {
                        ReadoutKind::Features => slice,
                        ReadoutKind::Logits => scoped(slice, projecting),
                        ReadoutKind::Selection {
                            certified: true, ..
                        } => scoped(slice, &certifying)
                            .scoped(sample_rows::Entry::NAME, "M", [class.selected]),
                        ReadoutKind::Selection { .. } => scoped(slice, projecting)
                            .scoped(sample_rows::Entry::NAME, "M", [class.selected]),
                    }
                })
                .collect::<Vec<_>>();
            let layout = checked_readout_class_draft(
                NativeGraphMetadata::new_template(backend),
                geometry,
                norm,
                Some(head),
                source,
                template,
                exports_logits(limits),
            )?
            .seal_template()
            .and_then(|template| template.certify(&slices))?;
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
) -> Result<NativeGraphStorageBytes, GraphError> {
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
        false,
    )
}

/// The host-written selection inputs of `rows` selected rows: a sampler's
/// mask, flags and draws, and a certified class's score divisors.
pub(crate) struct SelectionInputs {
    pub mask: NativePort,
    pub constrained: NativePort,
    pub draws: NativePort,
    pub temperature: Option<NativePort>,
}

/// The sampler's inputs of `rows` selected rows.
pub(crate) fn selection_inputs<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    sample: G::Binding<'a, sample_rows::Entry>,
    rows: u64,
    vocabulary: u64,
) -> Result<SelectionInputs, GraphError> {
    let dimensions = [("M", rows), ("V", vocabulary)];
    Ok(SelectionInputs {
        mask: graph.input_for(sample, "mask", &dimensions)?,
        constrained: graph.input_for(sample, "constrained", &dimensions)?,
        draws: graph.input_for(sample, "draws", &dimensions)?,
        temperature: None,
    })
}

/// Sampling of `rows` logits rows into `result`, after `shape_rows` shapes
/// them in place when `shaped`, reading `inputs`. The target readout and the
/// draft head select through this one node sequence and control layout. A constrained row's
/// mask applies before shaping cuts (shaping reads the same mask and flag
/// inputs as sampling), so top-k, min-p and top-p rank only admitted tokens.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample<'a, G, L>(
    graph: &mut G,
    shape: G::Binding<'a, shape_rows::Entry>,
    sample: G::Binding<'a, sample_rows::Entry>,
    vocabulary: u64,
    logits: &mut L,
    rows: u64,
    shaped: bool,
    inputs: SelectionInputs,
    result: WorkflowTensorMut<'_>,
) -> Result<SelectionPorts, GraphError>
where
    G: GraphDraft + 'a,
    for<'x> &'x L: Into<WorkflowTensorRef<'x>>,
    for<'x> &'x mut L: Into<WorkflowTensorMut<'x>>,
{
    let sample_dims = [("M", rows), ("V", vocabulary)];
    let SelectionInputs {
        mask,
        constrained,
        draws,
        temperature,
    } = inputs;
    // Shaping rewrites the logits in place, so sampling reads the same rows.
    let shaping = if shaped {
        let shape_dims = [("Sx", rows), ("V", vocabulary), ("Hn", HISTORY_TOKENS)];
        let parameters = graph.input_for(shape, "params", &shape_dims)?;
        let history = graph.input_for(shape, "history", &shape_dims)?;
        graph.enqueue::<shape_rows::Entry>(
            shape,
            &shape_dims,
            shape_rows::WorkflowArgs {
                logits: (&mut *logits).into(),
                mask: mask.tensor().into(),
                constrained: constrained.tensor().into(),
                params: parameters.tensor().into(),
                history: history.tensor().into(),
            },
        )?;
        Some(ShapingPorts {
            parameters,
            history,
        })
    } else {
        None
    };
    graph.enqueue::<sample_rows::Entry>(
        sample,
        &sample_dims,
        sample_rows::WorkflowArgs {
            logits: (&*logits).into(),
            mask: mask.tensor().into(),
            constrained: constrained.tensor().into(),
            draws: draws.tensor().into(),
            result,
        },
    )?;
    Ok(SelectionPorts {
        shaping,
        constrained,
        mask,
        draws,
        temperature,
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
    if let Some(temperature) = &ports.temperature {
        let divisors = sources
            .iter()
            .flat_map(|&source| score_divisor(batch.shaping[source][0]).to_le_bytes())
            .collect::<Vec<_>>();
        active.write_input(temperature, &divisors).map_err(device)?;
    }
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
        // A Q8_0 head, so a backend that reads progressive heads places it
        // in planes and certifies selection classes.
        let definition = crate::planning::tests::fixture_definition();
        let mut manifest = crate::planning::tests::fixture_manifest(&definition);
        let output = manifest
            .target
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == definition.decoder.exit.output.name)
            .unwrap();
        output.encoding = magnitude_artifacts::gguf::Encoding::Q8_0;
        output.nbytes = output.shape.iter().product::<u64>() / 32 * 34;
        let served = ResourceLimits {
            max_launch_rows: 512,
            max_launch_slots: 512,
            max_selected_rows: 32,
            max_drafting_slots: 32,
            exported_logits_rows: 0,
            max_images_per_request: 0,
            max_image_cells: 0,
            lookahead: false,
        };
        let diagnostic = ResourceLimits {
            exported_logits_rows: 64,
            ..served
        };
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Cuda,
            BackendName::Vulkan,
        ] {
            let layout = crate::planning::resident_layout(crate::ExecutionPath::Native, backend);
            let load = ModelLoadPlan::derive(
                &manifest,
                &definition,
                ComponentSelection {
                    head: false,
                    vision: false,
                },
                layout,
            )
            .unwrap();
            let progressive = reads_progressive_heads(backend).unwrap();
            let load = if progressive {
                load.with_progressive_head(&definition).unwrap()
            } else {
                load
            };
            let head = HeadPlans::of(&load).unwrap();
            assert_eq!(head.certified_rows(backend) > 0, progressive, "{backend:?}");
            let norm = load
                .weights()
                .find(|weight| {
                    weight.role
                        == WeightRole {
                            scope: WeightScope::Target,
                            kind: WeightKind::OutputNorm,
                        }
                })
                .unwrap();
            for limits in [served, diagnostic] {
                let classes =
                    readout_classes(limits, load_certified_rows(limits, head, backend)).unwrap();
                // A diagnostic load's selections read the logits it exports.
                assert_eq!(
                    classes.iter().any(|class| matches!(
                        class.kind,
                        ReadoutKind::Selection {
                            certified: true,
                            ..
                        }
                    )),
                    progressive && !exports_logits(limits),
                    "{backend:?}"
                );
                let regimes = certify_readout_regimes(
                    backend,
                    &definition.decoder,
                    norm,
                    head,
                    FeatureSource::Output,
                    limits,
                )
                .unwrap();
                for &class in &classes {
                    let layout = &regimes[&readout_regime(class)];
                    let bytes = checked_readout_class_draft(
                        NativeGraphMetadata::new(backend),
                        &definition.decoder,
                        norm,
                        Some(head),
                        FeatureSource::Output,
                        class,
                        exports_logits(limits),
                    )
                    .unwrap()
                    .seal_with_layout(layout)
                    .unwrap();
                    assert_eq!(bytes, layout.storage_bytes(), "{backend:?} {class:?}");
                }
            }
        }
    }

    #[test]
    fn a_served_readout_selects_within_the_bound_and_exports_no_logits() {
        let limits = ResourceLimits {
            max_launch_rows: 512,
            max_launch_slots: 512,
            max_selected_rows: 32,
            max_drafting_slots: 32,
            exported_logits_rows: 0,
            max_images_per_request: 0,
            max_image_cells: 0,
            lookahead: false,
        };
        let classes = readout_classes(limits, 4).unwrap();
        assert!(classes
            .iter()
            .all(|class| class.kind != ReadoutKind::Logits));
        assert!(classes.iter().all(|class| class.selected <= 32));
        assert!(classes
            .iter()
            .all(|class| class.projected == class.selected));
        assert!(classes
            .iter()
            .any(|class| matches!(class.kind, ReadoutKind::Selection { shaped: true, .. })));
        // Certified selections serve at most the bound's rows.
        assert!(classes.iter().all(|class| {
            !matches!(class.kind, ReadoutKind::Selection { certified: true, .. })
                || class.selected <= 4
        }));
        let diagnostic = readout_classes(
            ResourceLimits {
                exported_logits_rows: 64,
                ..limits
            },
            4,
        )
        .unwrap();
        // A diagnostic load exports logits, so it neither shapes nor
        // certifies.
        assert!(diagnostic.iter().all(|class| !matches!(
            class.kind,
            ReadoutKind::Selection { shaped: true, .. } | ReadoutKind::Selection { certified: true, .. }
        )));
        assert!(diagnostic.iter().all(|class| class.projected <= 64));
    }
}
