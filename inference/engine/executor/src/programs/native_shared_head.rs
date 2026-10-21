//! The native program of a draft head that reads target history
//! (`operators::SharedHead`). One sealed graph per class runs a head
//! transaction in a single device submission: one pass per proposal, every
//! pass one row per slot at the anchor's position.
//!
//! A pass gathers the target's embedding rows of its tokens, writes them and
//! the conditioning hidden rows into the two halves of the joined rows
//! (`tap_rows`), projects those into the head's width (`project_rows`), runs
//! the head's decoder blocks (the target's block fragments at the head's
//! width, each attention bound to the history planes of the target layer it
//! reads), normalizes the result, projects it onto the whole vocabulary
//! through the head's own table and selects. A further pass embeds that
//! selection and conditions on the pass's normalized output projected back to
//! the target's width, so no proposal returns to the host before the chain
//! ends. No pass writes history: the head keeps none.

use super::native_draft::widen;
use super::{DeviceSubmission, HeadOutput, HeadProgram, SelectedRows};
use crate::operators::block::{
    BlockControlPorts, BlockLane, BlockStatePorts, BlockSublayers, CheckedFeedForwardEntries,
    CheckedMixerEntries, FeedForwardEntries, MixerEntries,
};
use crate::operators::{InputForm, SharedHead, paired_block};
use crate::{
    DeviceError, HeadLaunchCore, InvariantError, ModelLoadPlan,
    NativeGraphOutputLease, NativeGraphWorkspaceLease, ResidentHead, ResidentWeight,
    ResourceLimits, SharedHeadProgramPlan, StateResourcePlan, SubmitError, ValidatedHeadLaunch,
    completion::CompletionWaiter,
    native::{AttestedSharedHead, OutputScales},
    programs::{
        graph::readout::{self, SelectionPorts, shapes},
        graph::{GraphError, draft::GraphDraft},
        native_constants::{
            CheckedGraphFamilyResources, CheckedGraphResources, ConstantTensors, GraphConstant,
            distinct_storage_bytes,
        },
        native_target_graph::{
            GatheredTable, WeightPort, embedding_transform, packed_weight, projection_ports,
            scaled_weight_ports, weight,
        },
    },
};
use magnitude_batching::{TargetBatchUpload, row_class};
use magnitude_family_contracts::{Decoder, ModelDefinition, WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{
    cluster_token_rows, draft_top_k, embedding_rows, place_logits_rows, project_rows,
    readout_features_rows, readout_project_rows, sample_rows, shape_rows, tap_rows,
};
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, Element, NativeGraph, NativeGraphFamily,
    NativeGraphMetadata, NativeGraphPlan, NativePort, Tensor, WorkflowTensor,
};
use std::{collections::BTreeMap, rc::Rc};

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native shared head program",
        detail: detail.into(),
    })
}
fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}
fn i32_bytes(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// One graph of the head: its slot class (every pass has one row per slot),
/// its selections per slot and whether selection is shaped first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SharedHeadGraphClass {
    pub slots: u64,
    pub steps: u64,
    pub shaped: bool,
}

/// The admitted classes used by both prepared formation and metadata
/// assessment: every drafting slot class with every proposal count up to the
/// method's bound.
pub(crate) fn shared_head_graph_classes(
    limits: ResourceLimits,
    proposals: usize,
) -> Result<Vec<SharedHeadGraphClass>, String> {
    let slot_bound = limits
        .max_launch_slots
        .min(limits.max_launch_rows)
        .min(limits.max_drafting_slots);
    let slot_classes = magnitude_batching::row_classes(slot_bound);
    if slot_classes.is_empty() || proposals == 0 {
        return Err("a head reading target history has no drafting class".into());
    }
    let mut classes = Vec::new();
    for slots in slot_classes {
        for steps in 1..=proposals as u64 {
            for shaped in [false, true] {
                classes.push(SharedHeadGraphClass {
                    slots: slots as u64,
                    steps,
                    shaped,
                });
            }
        }
    }
    Ok(classes)
}

/// What the head's graphs are built over: the target decoder whose embedding
/// and history it reads, its own blocks as a decoder of its width, and the
/// state plan whose target store holds that history.
pub(crate) struct SharedHeadGeometry<'a> {
    target: &'a Decoder,
    /// The head's blocks at its width; its entry and exit are the target's
    /// and are not read.
    head: Decoder,
    /// Per head block, the target block whose history it reads.
    sources: Vec<u32>,
    state: &'a StateResourcePlan,
    plan: &'a SharedHeadProgramPlan,
    /// The span class of every pass: the target store's span bound.
    segments: u64,
    epsilon: f32,
}

impl<'a> SharedHeadGeometry<'a> {
    pub(crate) fn new(
        definition: &'a ModelDefinition,
        state: &'a StateResourcePlan,
        plan: &'a SharedHeadProgramPlan,
    ) -> Result<Self, String> {
        let head = definition
            .head
            .as_ref()
            .ok_or("a shared head program without a head")?;
        let crate::operators::HeadForm::Shared(shared) =
            crate::operators::head_form(head, &definition.decoder)
                .map_err(|error| error.to_string())?
        else {
            return Err("a shared head program for a head with its own history".into());
        };
        let SharedHead {
            width,
            blocks,
            output_norm,
            readout: _,
        } = shared;
        if width != plan.width || blocks.len() != plan.blocks.len() {
            return Err("the shared head program disagrees with its head".into());
        }
        let sources = (0..blocks.len())
            .map(|index| {
                shared
                    .source(index)
                    .map(|source| source.block)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let segments = u64::try_from(
            state
                .target_state()
                .span_limit()
                .checked_next_power_of_two()
                .ok_or("head segment class overflows")?,
        )
        .map_err(|_| "head segment class exceeds u64")?;
        Ok(Self {
            target: &definition.decoder,
            head: Decoder {
                hidden: width,
                blocks: blocks.to_vec(),
                ..definition.decoder.clone()
            },
            sources,
            state,
            plan,
            segments,
            epsilon: output_norm.epsilon as f32,
        })
    }
}

/// The kernel entries of a head graph: prepared kernels or checked bindings.
/// A block's entries are taken once per pass.
struct SharedHeadEntries<'a, 'e, G: GraphDraft + 'a> {
    embedding: G::Binding<'a, embedding_rows::Entry>,
    join: G::Binding<'a, tap_rows::Entry>,
    input: G::Binding<'a, project_rows::Entry>,
    mixer: &'e dyn Fn(usize) -> Result<MixerEntries<'a, G>, String>,
    feed_forward: &'e dyn Fn(usize) -> Result<Option<FeedForwardEntries<'a, G>>, String>,
    scales: &'e dyn Fn(usize) -> Result<OutputScales, String>,
    features: G::Binding<'a, readout_features_rows::Entry>,
    projection: readout::ProjectionEntries<'a, G>,
    clustered: Option<ClusterEntries<'a, G>>,
    chain: G::Binding<'a, project_rows::Entry>,
    shape: G::Binding<'a, shape_rows::Entry>,
    sample: G::Binding<'a, sample_rows::Entry>,
}

struct ClusterEntries<'a, G: GraphDraft + 'a> {
    centroids: G::Binding<'a, readout_project_rows::Entry>,
    selection: G::Binding<'a, draft_top_k::Entry>,
    rows: G::Binding<'a, cluster_token_rows::Entry>,
    placement: G::Binding<'a, place_logits_rows::Entry>,
}

/// One block's per-run ports in one pass.
struct BlockPorts {
    coordinates: NativePort,
    visible: NativePort,
    fresh: NativePort,
    destinations: NativePort,
    /// The source layer's history planes, in plane-descriptor order.
    planes: Vec<NativePort>,
}

struct PassPorts {
    blocks: Vec<BlockPorts>,
    selection: SelectionPorts,
}

struct SharedHeadGraphParts<P> {
    plan: P,
    /// The first pass's `[slots, 2]` (token, status) rows.
    tokens: NativePort,
    /// The first pass's `[slots, hidden]` F32 conditioning rows.
    conditioning: NativePort,
    passes: Vec<PassPorts>,
    constants: Vec<GraphConstant>,
    weights: Vec<(WeightPort, NativePort)>,
    /// Selections `[steps · slots, 2]`, step-major.
    output: WorkflowTensor,
}

type PreparedSharedHeadGraph = SharedHeadGraphParts<NativeGraphPlan>;

fn shared_head_graph<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entries: SharedHeadEntries<'a, '_, G>,
    load: &ModelLoadPlan,
    geometry: &SharedHeadGeometry<'_>,
    class: SharedHeadGraphClass,
    backend: BackendName,
) -> Result<SharedHeadGraphParts<G>, GraphError> {
    if class.slots == 0 || class.steps == 0 {
        return Err(format!("shared head graph class {class:?} is inconsistent").into());
    }
    let slots = class.slots;
    let hidden = geometry.target.hidden;
    let width = geometry.head.hidden;
    let vocabulary = geometry.target.vocabulary;
    let epsilon = geometry.epsilon;
    let head = WeightScope::HeadBlock(0);
    let mut weights = Vec::new();
    let mut constants = Vec::new();

    // The first pass embeds the host's tokens, a chained pass the tokens
    // the device selected.
    let mut table = GatheredTable::of(
        &mut graph,
        load,
        WeightScope::Target,
        WeightKind::Embedding,
        &mut weights,
    )?;
    let input = packed_weight(
        &mut graph,
        load,
        head,
        WeightKind::HeadInputProjection,
        &mut weights,
        &mut constants,
    )?;
    let output_norm = weight(&mut graph, load, head, WeightKind::OutputNorm, &mut weights)?;
    let certified = geometry.plan.clustered.is_none()
        && readout::HeadPlans::for_recipe(load, head, head, true)?.certifies(slots, backend);
    let own_table = projection_ports(
        &mut graph,
        load,
        head,
        certified,
        &mut weights,
        &mut constants,
    )?;
    let centroids = geometry
        .plan
        .clustered
        .as_ref()
        .map(|_| {
            let (weight, scale, scale_extent) = scaled_weight_ports(
                &mut graph,
                load,
                head,
                WeightKind::OutputCentroids,
                &mut weights,
                &mut constants,
            )?;
            Ok::<_, GraphError>(readout::ProjectionPorts::Packed {
                weight,
                scale,
                scale_extent,
            })
        })
        .transpose()?;
    let token_order = geometry
        .plan
        .clustered
        .as_ref()
        .map(|cluster| {
            let tokens = cluster
                .token_order
                .iter()
                .map(|token| {
                    i32::try_from(token.0)
                        .map_err(|_| GraphError::from("canonical proposal token exceeds i32"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            GraphConstant::i32(&mut graph, &tokens)
        })
        .transpose()?;

    // A one-step graph chains nothing, and a sealed graph reads every port.
    let chain = (class.steps > 1)
        .then(|| {
            packed_weight(
                &mut graph,
                load,
                head,
                WeightKind::HeadChainProjection,
                &mut weights,
                &mut constants,
            )
        })
        .transpose()?;

    let embedding_dims = [
        ("M", slots),
        ("V", table.rows),
        ("D", hidden),
        ("R", 0),
        ("F", 1),
    ];
    let join_dims = [("M", slots), ("D", hidden), ("T", 2)];
    let tokens = graph.input_for(entries.embedding, "tokens", &embedding_dims)?;
    let conditioning = graph.input_for(entries.join, "residual", &join_dims)?;
    // The embedding half then the hidden half of the joined rows.
    let halves = [
        GraphConstant::i32(&mut graph, &[0])?,
        GraphConstant::i32(&mut graph, &[1])?,
    ];
    let out_rows = GraphConstant::identity(&mut graph, slots)?;
    let (scale, normalize, embedding_epsilon) = embedding_transform(geometry.target);
    let mut selections = graph.local_for(
        entries.sample,
        "result",
        &[("M", class.steps * slots), ("V", vocabulary)],
    )?;
    let mut passes = Vec::with_capacity(class.steps as usize);
    // The hidden rows the next pass conditions on.
    let mut chained: Option<WorkflowTensor> = None;
    for step in 0..class.steps {
        // The first pass embeds the host's rows, a chained pass the rows
        // the device selected.
        let selected;
        let (tokens, directory): (seismic::WorkflowTensorRef<'_>, _) = match step {
            0 => (
                tokens.tensor().into(),
                table.by_rows(&mut graph, &mut constants)?,
            ),
            _ => {
                selected = selections
                    .tensor()
                    .slice_leading((step - 1) * slots, step * slots);
                (
                    (&selected).into(),
                    table.selected(&mut graph, load, &mut weights, &mut constants)?,
                )
            }
        };
        let embedded = graph
            .enqueue(
                entries.embedding,
                &[
                    ("M", slots),
                    ("V", table.rows),
                    ("D", hidden),
                    ("R", directory.shape[0]),
                    ("F", directory.shape[1]),
                ],
                embedding_rows::WorkflowArgs {
                    table: (&table.table).into(),
                    tokens,
                    directory: (&directory.tensor).into(),
                    scale,
                    normalize,
                    epsilon: embedding_epsilon,
                },
            )?
            .r1;
        let mut joined = graph.local_for(entries.join, "taps", &join_dims)?;
        graph.enqueue(
            entries.join,
            &join_dims,
            tap_rows::WorkflowArgs {
                residual: (&embedded).into(),
                index: halves[0].port().tensor().into(),
                taps: joined.tensor_mut().into(),
            },
        )?;
        graph.enqueue(
            entries.join,
            &join_dims,
            tap_rows::WorkflowArgs {
                residual: match &chained {
                    None => conditioning.tensor().into(),
                    Some(hidden) => hidden.into(),
                },
                index: halves[1].port().tensor().into(),
                taps: joined.tensor_mut().into(),
            },
        )?;
        let mut residual = graph
            .enqueue(
                entries.input,
                &[
                    ("M", slots),
                    ("K", 2 * hidden),
                    ("N", width),
                    ("WS", input.scaled.extent),
                    ("WB", input.bound.extent),
                ],
                project_rows::WorkflowArgs {
                    source: joined.tensor().into(),
                    weight: (&input.scaled.weight).into(),
                    weight_scale: (&input.scaled.scale).into(),
                    weight_bound: (&input.bound.tensor).into(),
                },
            )?
            .value;
        let mut blocks = Vec::with_capacity(geometry.sources.len());
        for (index, &source) in geometry.sources.iter().enumerate() {
            let block = geometry
                .head
                .blocks
                .get(index)
                .ok_or("head block geometry is absent")?;
            let paired = paired_block(block).map_err(|error| error.to_string())?;
            let sublayers = BlockSublayers {
                lane: BlockLane::Head { source },
                paired: &paired,
                load,
                geometry: &geometry.head,
                state: geometry.state,
                block_index: index,
                rows: slots,
                segments: geometry.segments,
                slots,
                // The head's classes list no history row tiles and score
                // no selected read.
                listed: 0,
                scored: false,
                output_scales: (entries.scales)(index)?,
            };
            let (mixed, state, controls) = sublayers.mixer(
                &mut graph,
                (entries.mixer)(index)?,
                None,
                &residual,
                &mut weights,
                &mut constants,
            )?;
            let (
                BlockStatePorts::Attention(planes, None),
                BlockControlPorts::Attention {
                    coordinates,
                    visible,
                    fresh,
                    destinations,
                    history_tiles: None,
                    indexer: None,
                },
            ) = (state, controls)
            else {
                return Err("a head block other than unlisted attention".into());
            };
            residual = sublayers.feed_forward(
                &mut graph,
                (entries.feed_forward)(index)?,
                None,
                mixed,
                &mut weights,
                &mut constants,
            )?;
            blocks.push(BlockPorts {
                coordinates,
                visible,
                fresh,
                destinations,
                planes,
            });
        }
        let features = graph
            .enqueue(
                entries.features,
                &[
                    ("M", slots),
                    ("O", slots),
                    ("D", width),
                    ("PT", 1),
                    ("NM", width),
                ],
                readout_features_rows::WorkflowArgs {
                    hidden: (&residual.reshape(&[slots, 1, width])).into(),
                    norm: (&output_norm).into(),
                    out_rows: out_rows.port().tensor().into(),
                    epsilon,
                },
            )?
            .value;
        let mut projection_selection = None;
        let mut logits = if let Some(policy) = &geometry.plan.clustered {
            let cluster = entries
                .clustered
                .as_ref()
                .ok_or("clustered proposal entries are absent")?;
            let centroids = centroids
                .as_ref()
                .ok_or("clustered proposal centroids are absent")?;
            let order = token_order
                .as_ref()
                .ok_or("clustered proposal ordering is absent")?;
            let (cluster_scores, _) = readout::project_readout(
                &mut graph,
                readout::ProjectionEntries::PreparedPacked(cluster.centroids),
                centroids,
                readout::ReadoutInput::PreparedFeatures {
                    features: &features,
                    canonical_rows: None,
                },
                readout::ProjectionGeometry {
                    rows: slots,
                    projected: slots,
                    selected: 0,
                    vocabulary: policy.clusters,
                    hidden: width,
                    columns: policy.clusters,
                },
                0.0,
                None,
                &mut constants,
            )?;
            let top_dims = [
                ("M", slots),
                ("V", policy.clusters),
                ("K", policy.selected_clusters),
            ];
            let candidates_port = graph.local_for(cluster.selection, "candidates", &top_dims)?;
            let unary_port = graph.local_for(cluster.selection, "unary", &top_dims)?;
            let mut candidates = candidates_port.tensor().clone();
            let mut unary = unary_port.tensor().clone();
            graph.enqueue(
                cluster.selection,
                &top_dims,
                draft_top_k::WorkflowArgs {
                    logits: (&cluster_scores).into(),
                    candidates: (&mut candidates).into(),
                    unary: (&mut unary).into(),
                },
            )?;
            let per_cluster = vocabulary / policy.clusters;
            let columns = policy.selected_clusters * per_cluster;
            let token_rows = graph
                .enqueue(
                    cluster.rows,
                    &[
                        ("M", slots),
                        ("C", policy.clusters),
                        ("B", per_cluster),
                        ("K", policy.selected_clusters),
                    ],
                    cluster_token_rows::WorkflowArgs {
                        selected: (&candidates.reshape(&[slots, policy.selected_clusters, 2]))
                            .into(),
                        token_order: (&order
                            .port()
                            .tensor()
                            .reshape(&[policy.clusters, per_cluster]))
                            .into(),
                    },
                )?
                .value;
            let (compact, _) = readout::project_readout(
                &mut graph,
                entries.projection.borrowed(),
                &own_table,
                readout::ReadoutInput::PreparedFeatures {
                    features: &features,
                    canonical_rows: Some(&token_rows),
                },
                readout::ProjectionGeometry {
                    rows: slots,
                    projected: slots,
                    selected: 0,
                    vocabulary,
                    hidden: width,
                    columns,
                },
                0.0,
                None,
                &mut constants,
            )?;
            let placement_dims = [("M", slots), ("V", vocabulary), ("S", columns)];
            let logits_port = graph.local_for(cluster.placement, "logits", &placement_dims)?;
            let mut logits = logits_port.tensor().clone();
            graph.enqueue(
                cluster.placement,
                &placement_dims,
                place_logits_rows::WorkflowArgs {
                    scores: (&compact).into(),
                    token_rows: (&token_rows).into(),
                    logits: (&mut logits).into(),
                },
            )?;
            logits
        } else {
            let (logits, inputs) = readout::project_readout(
                &mut graph,
                entries.projection.borrowed(),
                &own_table,
                match &own_table {
                    readout::ProjectionPorts::Packed { .. } => {
                        readout::ReadoutInput::PreparedFeatures {
                            features: &features,
                            canonical_rows: None,
                        }
                    }
                    readout::ProjectionPorts::Progressive { .. } => readout::ReadoutInput::RawRms {
                        hidden: &residual,
                        norm: &output_norm,
                        form: InputForm::SINGLE,
                        out_rows: out_rows.port().tensor(),
                        epsilon,
                    },
                },
                readout::ProjectionGeometry {
                    rows: slots,
                    projected: slots,
                    selected: slots,
                    vocabulary,
                    hidden: width,
                    columns: vocabulary,
                },
                0.0,
                Some(entries.sample),
                &mut constants,
            )?;
            projection_selection = inputs;
            logits
        };
        let mut result = selections
            .tensor()
            .slice_leading(step * slots, (step + 1) * slots);
        let inputs = match projection_selection {
            Some(inputs) => inputs,
            None => readout::selection_inputs(&mut graph, entries.sample, slots, vocabulary)?,
        };
        let selection = readout::sample(
            &mut graph,
            entries.shape,
            entries.sample,
            vocabulary,
            &mut logits,
            slots,
            class.shaped,
            inputs,
            (&mut result).into(),
        )?;
        passes.push(PassPorts { blocks, selection });
        chained = if step + 1 < class.steps {
            let chain = chain
                .as_ref()
                .ok_or("a chaining head graph has no chain projection")?;
            Some(
                graph
                    .enqueue(
                        entries.chain,
                        &[
                            ("M", slots),
                            ("K", width),
                            ("N", hidden),
                            ("WS", chain.scaled.extent),
                            ("WB", chain.bound.extent),
                        ],
                        project_rows::WorkflowArgs {
                            source: (&features).into(),
                            weight: (&chain.scaled.weight).into(),
                            weight_scale: (&chain.scaled.scale).into(),
                            weight_bound: (&chain.bound.tensor).into(),
                        },
                    )?
                    .value,
            )
        } else {
            None
        };
    }
    let output = selections.tensor_mut().clone();
    graph.export(&output)?;
    constants.extend(halves);
    constants.push(out_rows);
    if let Some(order) = token_order {
        constants.push(order);
    }
    Ok(SharedHeadGraphParts {
        plan: graph,
        tokens,
        conditioning,
        passes,
        constants,
        weights,
        output,
    })
}

/// The checked bindings of a head's entries, from its program plan.
struct CheckedSharedHeadEntries {
    embedding: [(&'static str, Element); 2],
    join: [(&'static str, Element); 1],
    input: [(&'static str, Element); 3],
    blocks: Vec<(CheckedMixerEntries, Option<CheckedFeedForwardEntries>)>,
    features: [(&'static str, Element); 2],
    projection: readout::CheckedProjectionEntries,
    centroids: Option<[(&'static str, Element); 2]>,
    chain: [(&'static str, Element); 3],
    selection: [(&'static str, Element); 0],
}

impl CheckedSharedHeadEntries {
    fn new(plan: &SharedHeadProgramPlan, backend: BackendName) -> Result<Self, String> {
        let activation = plan.activation;
        let projection =
            |weight: Element| [("A", activation), ("W", weight), ("Y", Element::f32())];
        Ok(Self {
            embedding: [
                ("EW", plan.embedding.table),
                ("A", plan.embedding.activation),
            ],
            join: [("A", activation)],
            input: projection(plan.input.element),
            blocks: plan
                .blocks
                .iter()
                .map(|slot| {
                    Ok((
                        CheckedMixerEntries::new(slot.mixer(), backend, false)?,
                        slot.feed_forward()
                            .map(|slot| CheckedFeedForwardEntries::new(slot, backend))
                            .transpose()?,
                    ))
                })
                .collect::<Result<_, String>>()?,
            features: [("NW", plan.output_norm), ("A", activation)],
            projection: readout::CheckedProjectionEntries::new(
                plan.projection,
                plan.output_norm,
                activation,
                true,
            ),
            centroids: plan
                .clustered
                .as_ref()
                .map(|cluster| [("OW", cluster.centroids.element), ("A", activation)]),
            chain: projection(plan.chain.element),
            selection: [],
        })
    }
}

/// The storage and bound constants of the head's graph family over
/// `classes`, from metadata alone: each class is sealed exactly, and the
/// family holds the largest of each arena.
pub(crate) fn checked_shared_head_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &SharedHeadGeometry<'_>,
    classes: impl IntoIterator<Item = SharedHeadGraphClass>,
) -> Result<CheckedGraphResources, GraphError> {
    let checked = CheckedSharedHeadEntries::new(geometry.plan, backend)?;
    let block = |index: usize| {
        checked
            .blocks
            .get(index)
            .ok_or_else(|| "checked head block is absent".to_owned())
    };
    let mut family = CheckedGraphFamilyResources::new();
    for class in classes {
        let context = || format!("shared head graph class {class:?}");
        let parts = shared_head_graph(
            NativeGraphMetadata::new(backend),
            SharedHeadEntries {
                embedding: &checked.embedding[..],
                join: &checked.join[..],
                input: &checked.input[..],
                mixer: &|index| block(index)?.0.entries(),
                feed_forward: &|index| {
                    Ok(block(index)?.1.as_ref().map(|checked| checked.entries()))
                },
                // Scalar arguments leave the checked shapes and storage alone.
                scales: &|_| {
                    Ok(OutputScales {
                        mixer: 1.0,
                        feed_forward: 1.0,
                        per_layer: 1.0,
                    })
                },
                features: &checked.features[..],
                projection: checked.projection.entries(),
                clustered: checked.centroids.as_ref().map(|centroids| ClusterEntries {
                    centroids: &centroids[..],
                    selection: &checked.selection[..],
                    rows: &checked.selection[..],
                    placement: &checked.selection[..],
                }),
                chain: &checked.chain[..],
                shape: &checked.selection[..],
                sample: &checked.selection[..],
            },
            load,
            geometry,
            class,
            backend,
        )
        .map_err(|error| error.context(context()))?;
        family.include_plan(parts.plan.plan_heap())?;
        let storage = GraphDraft::seal(parts.plan).map_err(|error| error.context(context()))?;
        family.include(storage, parts.constants);
    }
    Ok(family.finish()?)
}

pub struct PreparedSharedHeadGraphs {
    classes: BTreeMap<SharedHeadGraphClass, PreparedSharedHeadGraph>,
    family: NativeGraphFamily,
    segments: u64,
}

pub(crate) struct BoundSharedHeadGraphs {
    prepared: Rc<PreparedSharedHeadGraphs>,
    bound: BTreeMap<SharedHeadGraphClass, BoundNativeGraphPlan>,
    constants: Vec<Tensor>,
    /// The token embedding as the passes gather it: the first pass by the
    /// host's tokens, a chained pass by the device's selections.
    embedding: crate::device_table::ResidentTable,
}

impl PreparedSharedHeadGraphs {
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        distinct_storage_bytes(
            self.classes
                .values()
                .flat_map(|graph| graph.constants.iter()),
        )
    }

    pub(crate) fn prepare(
        target_device: &Device,
        kernels: &AttestedSharedHead,
        load: &ModelLoadPlan,
        geometry: &SharedHeadGeometry<'_>,
        classes: impl IntoIterator<Item = SharedHeadGraphClass>,
    ) -> Result<Self, SubmitError> {
        let block = |index: usize| {
            kernels
                .blocks
                .get(index)
                .ok_or_else(|| "attested head block is absent".to_owned())
        };
        let mut prepared = BTreeMap::new();
        for class in classes {
            if prepared.contains_key(&class) {
                return Err(invalid("shared head graph class is duplicated"));
            }
            let class_error =
                |error: GraphError| invalid(format!("shared head graph class {class:?}: {error}"));
            let parts = shared_head_graph::<NativeGraph>(
                target_device.native_graph(),
                SharedHeadEntries {
                    embedding: &kernels.embedding,
                    join: &kernels.join,
                    input: &kernels.input,
                    mixer: &|index| Ok((&block(index)?.mixer).into()),
                    feed_forward: &|index| Ok(block(index)?.feed_forward.as_ref().map(Into::into)),
                    scales: &|index| Ok(block(index)?.output_scales),
                    features: &kernels.features,
                    projection: readout::ProjectionEntries::prepared(&kernels.projection),
                    clustered: if geometry.plan.clustered.is_some() {
                        Some(ClusterEntries {
                            centroids: kernels
                                .centroids
                                .as_ref()
                                .ok_or_else(|| invalid("clustered centroid entry is absent"))?,
                            selection: kernels
                                .cluster_selection
                                .as_ref()
                                .ok_or_else(|| invalid("cluster selection entry is absent"))?,
                            rows: kernels
                                .cluster_rows
                                .as_ref()
                                .ok_or_else(|| invalid("cluster row entry is absent"))?,
                            placement: kernels
                                .placement
                                .as_ref()
                                .ok_or_else(|| invalid("cluster placement entry is absent"))?,
                        })
                    } else {
                        None
                    },
                    chain: &kernels.chain,
                    shape: &kernels.shape,
                    sample: &kernels.sample,
                },
                load,
                geometry,
                class,
                target_device.backend(),
            )
            .map_err(class_error)?;
            let plan = GraphDraft::seal(parts.plan).map_err(class_error)?;
            prepared.insert(
                class,
                SharedHeadGraphParts {
                    plan,
                    tokens: parts.tokens,
                    conditioning: parts.conditioning,
                    passes: parts.passes,
                    constants: parts.constants,
                    weights: parts.weights,
                    output: parts.output,
                },
            );
        }
        if prepared.is_empty() {
            return Err(invalid("shared head graph family has no classes"));
        }
        let plans = prepared
            .values()
            .map(|graph| graph.plan.clone())
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(device)?;
        Ok(Self {
            classes: prepared,
            family,
            segments: geometry.segments,
        })
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    pub(crate) fn bind_weights(
        self: &Rc<Self>,
        resident: &ResidentHead,
    ) -> Result<BoundSharedHeadGraphs, SubmitError> {
        let mut uploaded = ConstantTensors::new(resident.embedding.tensor().device());
        let mut bound = BTreeMap::new();
        for (class, graph) in &self.classes {
            let constants = graph
                .constants
                .iter()
                .map(|constant| Ok((constant.port(), uploaded.tensor(constant).map_err(device)?)))
                .collect::<Result<Vec<_>, SubmitError>>()?;
            let fixed = graph
                .weights
                .iter()
                .map(|(weight, port)| {
                    let resident = resident_weight(resident, weight.role)?;
                    Ok((port, weight.part.of(resident).map_err(invalid)?))
                })
                .chain(constants.iter().map(|(port, tensor)| Ok((*port, tensor))))
                .collect::<Result<Vec<_>, SubmitError>>()?;
            bound.insert(*class, graph.plan.bind_static(&fixed).map_err(device)?);
        }
        Ok(BoundSharedHeadGraphs {
            prepared: self.clone(),
            bound,
            constants: uploaded.into_tensors(),
            embedding: resident.embedding.table().clone(),
        })
    }
}

impl BoundSharedHeadGraphs {
    fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.constants.iter().try_fold(0u64, |bytes, tensor| {
            bytes
                .checked_add(tensor.storage_bytes())
                .ok_or("shared head graph constant charge overflows")
        })
    }

    fn class(
        &self,
        class: SharedHeadGraphClass,
    ) -> Result<(&PreparedSharedHeadGraph, &BoundNativeGraphPlan), SubmitError> {
        let graph = self.prepared.classes.get(&class).ok_or_else(|| {
            invalid(format!(
                "shared head graph class {class:?} was not prepared"
            ))
        })?;
        let bound = self
            .bound
            .get(&class)
            .ok_or_else(|| invalid(format!("shared head graph class {class:?} was not bound")))?;
        Ok((graph, bound))
    }
}

/// A head graph's weight: the target's token table, or one of the head's own.
fn resident_weight(
    resident: &ResidentHead,
    role: WeightRole,
) -> Result<&ResidentWeight, SubmitError> {
    match (role.scope, role.kind) {
        (WeightScope::Target, WeightKind::Embedding) => Ok(&resident.embedding),
        _ => resident.weights.get(role).map_err(invalid),
    }
}

/// One pass's attention controls in the target history domain `domain`,
/// padded to `rows` rows of `segments` spans: padding rows attend to nothing.
/// No row appends.
struct PassControls {
    coordinates: Vec<u8>,
    visible: Vec<u8>,
    fresh: Vec<u8>,
    destinations: Vec<u8>,
}

impl PassControls {
    fn new(
        pass: &TargetBatchUpload<'_>,
        domain: usize,
        rows: usize,
        segments: usize,
    ) -> Result<Self, SubmitError> {
        let actual = pass.actual_rows;
        if actual > rows {
            return Err(invalid("head pass has more rows than its graph class"));
        }
        let history = pass
            .histories
            .get(domain)
            .ok_or_else(|| invalid("head pass lacks the target history domain it reads"))?;
        let mut visible = vec![0_i32; rows * segments * 2];
        for (row, ranges) in history.visible[..actual].iter().enumerate() {
            let used = ranges
                .iter()
                .rposition(|range| range[1] > range[0])
                .map_or(0, |last| last + 1);
            if used > segments {
                return Err(invalid(format!(
                    "head row attends {used} history spans; the head admits {segments}"
                )));
            }
            for (span, [start, end]) in ranges[..used].iter().enumerate() {
                let at = (row * segments + span) * 2;
                visible[at] = *start;
                visible[at + 1] = *end;
            }
        }
        Ok(Self {
            coordinates: i32_bytes((0..rows).flat_map(|row| {
                pass.coordinates
                    .get(row)
                    .filter(|_| row < actual)
                    .copied()
                    .unwrap_or([0; 4])
            })),
            visible: i32_bytes(visible),
            fresh: i32_bytes((0..rows).flat_map(|row| {
                history
                    .fresh
                    .get(row)
                    .filter(|_| row < actual)
                    .copied()
                    .unwrap_or([0, 0])
            })),
            destinations: i32_bytes((0..rows).map(|_| -1)),
        })
    }
}

pub struct NativeSharedHeadProgram {
    geometry: Decoder,
    graphs: BoundSharedHeadGraphs,
    waiter: CompletionWaiter,
}

impl NativeSharedHeadProgram {
    /// Visit the graph constants the program binds.
    pub(crate) fn visit_constants(&self, visit: &mut dyn FnMut(&seismic::Tensor)) {
        self.graphs.constants.iter().for_each(visit);
    }

    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    /// `geometry` is the target decoder.
    pub(crate) fn new(
        geometry: Decoder,
        graphs: BoundSharedHeadGraphs,
    ) -> Result<Self, SubmitError> {
        let waiter = CompletionWaiter::spawn().map_err(device)?;
        Ok(Self {
            geometry,
            graphs,
            waiter,
        })
    }

    /// Queue one head transaction. Returns its completion and the
    /// selections the graph fills.
    fn queue(
        &self,
        core: &HeadLaunchCore,
        workspace: &mut NativeGraphWorkspaceLease,
        output: NativeGraphOutputLease,
    ) -> Result<(seismic::NativeGraphCompletion, HeadOutput), SubmitError> {
        let reads = core
            .target_reads()
            .ok_or_else(|| invalid("a head without a store reads target history"))?;
        let batch = core.batch();
        let entry = batch.upload();
        let chain = batch.chain().collect::<Vec<_>>();
        let steps = batch.steps();
        let actual_slots = batch.actual_slots();
        let slots = row_class(actual_slots).ok_or_else(|| invalid("head slots have no class"))?;
        let shaped = std::iter::once(&entry)
            .chain(&chain)
            .any(|pass| pass.shaping[..pass.select_rows.len()].iter().any(shapes));
        let class = SharedHeadGraphClass {
            slots: slots as u64,
            steps: steps as u64,
            shaped,
        };
        let (graph, bound) = self.graphs.class(class)?;
        if graph.passes.len() != chain.len() + 1 {
            return Err(invalid("head graph passes differ from the batch's steps"));
        }
        let segments = self.graphs.prepared.segments as usize;
        let mut bindings = bound.bindings();
        for pass in &graph.passes {
            if pass.blocks.len() != reads.history.len() {
                return Err(invalid("head blocks differ from the target history read"));
            }
            for (block, read) in pass.blocks.iter().zip(&reads.history) {
                if block.planes.len() != read.planes.len() {
                    return Err(invalid(
                        "target history planes differ from the attention entry",
                    ));
                }
                for (port, plane) in block.planes.iter().zip(&read.planes) {
                    bindings.set(port, plane).map_err(device)?;
                }
            }
        }
        // Conditioning rows in slot order, widened to F32 and padded.
        let crate::HeadConditioning::Rows(rows) = core.conditioning() else {
            return Err(invalid(
                "a head reading target history enters host conditioning rows",
            ));
        };
        let row_bytes = self.geometry.hidden as usize * 4;
        let mut conditioning = Vec::with_capacity(slots * row_bytes);
        for rows in rows {
            conditioning.extend(widen(rows.bytes(), self.geometry.activation_dtype));
        }
        if conditioning.len() != actual_slots * row_bytes {
            return Err(invalid("head conditioning bytes differ from its slots"));
        }
        conditioning.resize(slots * row_bytes, 0);
        let mut active = workspace.slot_mut().activate(&graph.plan).map_err(device)?;
        active
            .write_input(&graph.conditioning, &conditioning)
            .map_err(device)?;
        // The first pass's tokens gather by table row: of a streamed table,
        // by the slots their rows are brought to. A chained pass gathers
        // the device's selections through the table's directory.
        let tokens = (0..slots)
            .map(|slot| {
                entry
                    .tokens
                    .get(slot)
                    .copied()
                    .filter(|_| slot < actual_slots)
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        let rows = self.graphs.embedding.indices(&tokens).map_err(invalid)?;
        // A chained pass embeds the device's selection. One the device
        // does not hold embeds another row, and the target rejects the
        // proposal that follows; observing the selections brings it for the
        // rounds after.
        let selected_rows = (!chain.is_empty())
            .then(|| SelectedRows::selections(self.graphs.embedding.read_selected()));
        active
            .write_input(
                &graph.tokens,
                &i32_bytes(rows.iter().flat_map(|row| [*row, 0])),
            )
            .map_err(device)?;
        let words = self.geometry.vocabulary.div_ceil(32) as usize;
        for (ports, pass) in graph
            .passes
            .iter()
            .zip(std::iter::once(&entry).chain(&chain))
        {
            for (block, read) in ports.blocks.iter().zip(&reads.history) {
                let controls = PassControls::new(pass, read.domain, slots, segments)?;
                active
                    .write_input(&block.coordinates, &controls.coordinates)
                    .map_err(device)?;
                active
                    .write_input(&block.visible, &controls.visible)
                    .map_err(device)?;
                active
                    .write_input(&block.fresh, &controls.fresh)
                    .map_err(device)?;
                active
                    .write_input(&block.destinations, &controls.destinations)
                    .map_err(device)?;
            }
            readout::write_selection(pass, &mut active, &ports.selection, slots, words)?;
        }
        let mut output = output;
        let outputs = output
            .activate(&graph.plan)
            .map_err(SubmitError::Invariant)?;
        let (outputs, completion) = active
            .attach(bindings, outputs)
            .and_then(|ready| ready.submit())
            .map_err(device)?;
        let owner = output.publish(outputs);
        let selections = owner
            .tensor(&graph.output)
            .ok_or_else(|| invalid("head graph omitted its selections"))?;
        Ok((
            completion,
            HeadOutput {
                selections: Some(selections),
                selected_rows,
            },
        ))
    }
}

impl HeadProgram for NativeSharedHeadProgram {
    type Submission =
        DeviceSubmission<HeadLaunchCore, NativeGraphWorkspaceLease, HeadOutput>;

    fn submit(
        &mut self,
        mut launch: ValidatedHeadLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedHeadLaunch)> {
        let queued = {
            let (core, workspace, output) = launch.execution_parts_mut();
            match output.take() {
                Some(output) => self.queue(core, workspace, output),
                None => Err(invalid("head graph output lease is absent")),
            }
        };
        let (completion, output) = match queued {
            Ok(queued) => queued,
            Err(error) => return Err((error, launch)),
        };
        let (core, workspace, _) = launch.into_submission_parts();
        Ok(DeviceSubmission::new(
            self.waiter.completion(vec![completion]),
            core,
            workspace,
            output,
        ))
    }
}
