use super::weights::{
    CheckedLandingEntries, LandedReadPorts, LandingEntries, PlannedWeights, WeightView, activation, landed_read,
};
use crate::programs::graph::{
    draft::Certified,
    facts::{FactIndex, FactSink, Facts, RealIndex},
};
use crate::{
    ModelLoadPlan, TargetProgramPlan,
    native::AttestedTarget,
    operators::per_layer::graph::{PerLayerEntryEntries, PerLayerEntryPorts, per_layer_entry, split_entry},
    programs::graph::{GraphError, draft::GraphDraft},
    stored_table::SelectionNotice,
};
use magnitude_family_contracts::{Decoder, EmbeddingScale, WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{embedding_rows, selection_notice};
#[cfg(test)]
use seismic::NativeGraphStorageBytes;
use seismic::{Argument, BackendName, ClassFact, NativeGraphClassSlice, NativeGraphMetadata, NativeGraphResourceTemplate, NativeGraphTraces};
use seismic::{
    Device, Dim, Element, NativeGraphLayout, NativeGraphPlan, NativePort, NativeTimelinePort,
    WorkflowTensor,
};

/// The per-layer entry graph of one row class.
#[derive(Clone)]
pub(crate) struct PreparedPerLayerEntryGraph {
    pub plan: NativeGraphPlan,
    pub ports: PerLayerEntryPorts,
}

/// The per-layer entry graph's inputs apart from its row class.
pub(super) struct PerLayerEntryGraphSource<'a> {
    kernels: &'a crate::native::PerLayerEntryKernels,
    binding: crate::PerLayerEntryBinding,
    entry: &'a magnitude_family_contracts::PerLayerEntry,
    load: &'a ModelLoadPlan,
    pub(super) layout: &'a NativeGraphLayout,
}

impl<'a> PerLayerEntryGraphSource<'a> {
    pub(super) fn of(
        handles: &'a AttestedTarget,
        load: &'a ModelLoadPlan,
        geometry: &'a Decoder,
        plan: &TargetProgramPlan,
        layout: Option<&'a NativeGraphLayout>,
    ) -> Result<Option<Self>, String> {
        match (
            &handles.per_layer,
            plan.per_layer(),
            &geometry.entry.per_layer,
            layout,
        ) {
            (None, None, None, None) => Ok(None),
            (Some(kernels), Some(binding), Some(entry), Some(layout)) => Ok(Some(Self {
                kernels,
                binding,
                entry,
                load,
                layout,
            })),
            _ => Err("the per-layer entry's kernels, binding and geometry disagree".into()),
        }
    }

    pub(super) fn prepare(&self, device: &Device, rows: u64) -> Result<PreparedPerLayerEntryGraph, String> {
        let mut graph = device.native_graph_with_layout(self.layout);
        let (topology, numbers) = split_entry(&self.binding, self.entry);
        let topology = crate::programs::graph::facts::constant_reals(topology);
        let ports = per_layer_entry(
            &mut graph,
            PerLayerEntryEntries::from(self.kernels),
            &topology,
            &crate::programs::graph::facts::constant(numbers),
            &target_view(self.load, WeightKind::PerLayerModelProjection)?,
            &target_view(self.load, WeightKind::PerLayerProjectionNorm)?,
            &seismic::Dim::from(rows),
        )
        .map_err(|error| format!("per-layer entry graph rows {rows}: {error}"))?;
        let plan = GraphDraft::seal(graph).map_err(|error| error.to_string())?;
        // It writes the bound per-layer rows and exports nothing.
        if plan.output_bytes() != 0 {
            return Err(format!("per-layer entry graph rows {rows} exports outputs"));
        }
        Ok(PreparedPerLayerEntryGraph { plan, ports })
    }
}

/// How an entry graph receives its `[rows, 2]` (token, status) input rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum EntryTokens {
    /// Written by the host into the step's upload region.
    Uploaded,
    /// Bound to the previous step's selection tensor on the device.
    Selected,
}

#[derive(Clone)]
pub(crate) struct PreparedTargetEntryGraph {
    pub plan: NativeGraphPlan,
    pub table: NativePort,
    /// The directory the embedding addresses the table through: a streamed
    /// table's for selected tokens, else empty (`RowDirectory`).
    pub directory: NativePort,
    pub tokens: NativePort,
    pub hidden: WorkflowTensor,
    /// Of a step reading selected tokens in a model with a stored table:
    /// where the step copies its tokens for the host, and the signal after.
    pub notice: Option<EntryNotice>,
    /// Of a step uploading the host's tokens in a model whose blocks read
    /// the step's token rows: where it keeps them.
    pub kept_tokens: Option<KeptTokens>,
    /// Of a step reading selected tokens of a table with landing slots: its
    /// landed read's ports.
    pub landed: Option<LandedReadPorts>,
}

/// An entry graph's kept token rows (`keeps_token_rows`): the host's token
/// rows uploaded in the `sample_rows` layout, and the bound token rows
/// (`GroupBuffers::token_rows`) the entry copies them to for the blocks.
#[derive(Clone)]
pub(crate) struct KeptTokens {
    pub uploaded: NativePort,
    pub rows: NativePort,
}

/// An entry graph's selection notice (`selection_notice`): the step's
/// `[rows, 2]` selected tokens copied into a host-shared tensor the run
/// binds, then a signal the run binds, which the stored table's filler
/// waits for to learn the rows the step names.
#[derive(Clone)]
pub(crate) struct EntryNotice {
    pub tokens: NativePort,
    pub signal: NativeTimelinePort<seismic::timeline::Reached>,
}

impl PreparedTargetEntryGraph {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare(
        device: &Device,
        embedding: &seismic::NativeKernel<embedding_rows::Entry>,
        notice: Option<&seismic::NativeKernel<selection_notice::Entry>>,
        landing: Option<&crate::native::AttestedLanding>,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        rows: u64,
        source: EntryTokens,
        layout: &NativeGraphLayout,
    ) -> Result<Self, String> {
        let draft = entry_graph_topology(
            device.native_graph_with_layout(layout),
            embedding,
            notice,
            landing.map(LandingEntries::native),
            keeps_token_rows(load),
            &load.weight(TARGET_TABLE).map_err(|error| error.to_string())?,
            EntryWidth { hidden: &Dim::from(geometry.hidden), transform: transform_arguments(embedding_transform(geometry)) },
            &Dim::from(rows),
            source,
        )
        .map_err(|error| error.to_string())?;
        Ok(Self {
            plan: GraphDraft::seal(draft.graph).map_err(|error| error.to_string())?,
            table: draft.table,
            directory: draft.directory,
            tokens: draft.tokens,
            hidden: draft.hidden,
            notice: draft.notice,
            kept_tokens: draft.kept_tokens,
            landed: draft.landed,
        })
    }
}

/// The target's token table, which the target and its draft heads embed.
pub(crate) const TARGET_TABLE: WeightRole = WeightRole {
    scope: WeightScope::Target,
    kind: WeightKind::Embedding,
};

/// Whether the steps of `load` that read selected tokens notify the host of
/// them (`EntryNotice`): its target has a stored table.
pub(super) fn notifies_selection(load: &ModelLoadPlan) -> bool {
    load.weights().any(crate::WeightPlan::stored_in_slots)
}

/// The rows of each copy of the selection notice (`SelectionNotice`) a load
/// whose steps read selected tokens of a stored table binds: the largest row
/// class; none without a stored table.
pub(super) fn notice_rows(load: &ModelLoadPlan, largest_rows: u64) -> Option<u64> {
    notifies_selection(load).then_some(largest_rows)
}

/// What the selection notice of `rows` rows (`notice_rows`) holds, its
/// binding charge; none without one.
pub(super) fn notice_bytes(rows: Option<u64>) -> Result<u64, String> {
    rows.map_or(Ok(0), |rows| {
        Element::i32()
            .canonical_byte_len(&SelectionNotice::extents(rows))
            .map_err(|error| format!("the selection notice: {error}"))
    })
}

/// Whether blocks of `load` read the step's token rows past the entry: its
/// target has an n-gram input. Its steps of host tokens then keep them in
/// the bound token rows.
pub(super) fn keeps_token_rows(load: &ModelLoadPlan) -> bool {
    load.weights()
        .any(|weight| weight.role.kind == WeightKind::NGramTable)
}

/// The `[rows, 2]` I32 extents of the token rows a model that keeps them
/// binds, over `rows` rows.
pub(super) fn token_rows_extents(load: &ModelLoadPlan, rows: u64) -> Option<[u64; 2]> {
    keeps_token_rows(load).then_some([rows, 2])
}

/// The view of the target's planned weight of `kind`.
pub(super) fn target_view(load: &ModelLoadPlan, kind: WeightKind) -> Result<WeightView, String> {
    load.weight(WeightRole { scope: WeightScope::Target, kind }).map_err(|error| error.to_string())
}

/// An entry graph before it is sealed, and its external ports.
pub(super) struct EntryGraphDraft<G> {
    pub(super) graph: G,
    table: NativePort,
    directory: NativePort,
    tokens: NativePort,
    hidden: WorkflowTensor,
    notice: Option<EntryNotice>,
    kept_tokens: Option<KeptTokens>,
    landed: Option<LandedReadPorts>,
}

/// The entry graph: the step's token rows embedded. `notice` (of a model
/// with a stored table) copies selected tokens for the host first. With
/// `keeps_tokens` (a model whose blocks read the step's token rows) the
/// host's uploaded tokens are copied into the bound token rows
/// (`kept_tokens`), so later graphs read a device tensor of the step's token
/// rows whatever their source, as they read a selection. Selected tokens of
/// a table with landing slots are a landed read (`landed_read`, `landing`).
#[allow(clippy::too_many_arguments)]
pub(super) fn entry_graph_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    embedding: G::Binding<'a, embedding_rows::Entry>,
    notice: Option<G::Binding<'a, selection_notice::Entry>>,
    landing: Option<LandingEntries<'a, G>>,
    keeps_tokens: bool,
    weight: &WeightView,
    width: EntryWidth<'_>,
    rows: &Dim,
    source: EntryTokens,
) -> Result<EntryGraphDraft<G>, GraphError> {
    let table = graph.port(weight.resident, &weight.shape)?;
    let two = Dim::from(2);
    // Uploaded tokens are the host's: of a streamed table it writes their
    // slots. Selected tokens are the device's, which finds their slots in
    // the directory.
    let [entries, fields] = match source {
        EntryTokens::Uploaded => [Dim::from(0), Dim::from(1)],
        EntryTokens::Selected => weight.directory.clone(),
    };
    let table_rows = weight.shape.first().ok_or("the token table has no rows")?;
    let dimensions = [("M", rows), ("V", table_rows), ("D", width.hidden), ("R", &entries), ("F", &fields)];
    let tokens = match source {
        EntryTokens::Uploaded => graph.input_for(embedding, "tokens", &dimensions)?,
        EntryTokens::Selected => graph.port_with_class_extent(Element::i32(), &[rows, &two], 0, "M")?,
    };
    // Selected tokens reach the host first, as early in the step as they
    // can: the filler of a stored table learns from them the rows the step
    // names while the device runs the blocks before the table's gate.
    // The token row copy entry (`selection_notice`), which the kept token
    // rows use too.
    let copy_entry = notice;
    let notice = match (notice, source) {
        (Some(entry), EntryTokens::Selected) => {
            let mut copied = graph.port_with_class_extent(Element::i32(), &[rows, &two], 0, "M")?;
            graph.enqueue::<selection_notice::Entry>(
                entry,
                &[("M", rows)],
                selection_notice::WorkflowArgs {
                    selection: tokens.tensor().into(),
                    notice: copied.tensor_mut().into(),
                },
            )?;
            Some(EntryNotice {
                tokens: copied,
                signal: graph.timeline(seismic::timeline::Signal),
            })
        }
        _ => None,
    };
    // The host's tokens themselves (the embedding's uploaded rows are its
    // table's indices, a streamed table's slots), copied into the bound
    // token rows.
    let kept_tokens = match (keeps_tokens, source) {
        (true, EntryTokens::Uploaded) => {
            let copy = copy_entry.ok_or("kept token rows need the token row copy entry")?;
            let uploaded = graph.input_for(copy, "selection", &[("M", rows)])?;
            let mut rows_port = graph.port_with_class_extent(Element::i32(), &[rows, &two], 0, "M")?;
            graph.enqueue::<selection_notice::Entry>(
                copy,
                &[("M", rows)],
                selection_notice::WorkflowArgs {
                    selection: uploaded.tensor().into(),
                    notice: rows_port.tensor_mut().into(),
                },
            )?;
            Some(KeptTokens {
                uploaded,
                rows: rows_port,
            })
        }
        _ => None,
    };
    let directory = graph.port(Element::i32(), &[&entries, &fields])?;
    let (scale, normalize, epsilon) = width.transform;
    let embed = |graph: &mut G| {
        graph.enqueue::<embedding_rows::Entry>(
            embedding,
            &dimensions,
            embedding_rows::WorkflowArgs {
                table: table.tensor().into(),
                tokens: tokens.tensor().into(),
                directory: directory.tensor().into(),
                scale,
                normalize: normalize.into(),
                epsilon,
            },
        )
    };
    let (result, landed) = match source {
        EntryTokens::Selected if weight.reads_landed()? => {
            let landing = landing
                .ok_or_else(|| format!("a landed read of {:?} without its entries", weight.role))?;
            let (result, ports) = landed_read(
                &mut graph,
                weight,
                landing,
                tokens.tensor().into(),
                rows,
                Some("M"),
                embed,
            )?;
            (result, Some(ports))
        }
        _ => (embed(&mut graph)?, None),
    };
    graph.export(&result.r1)?;
    Ok(EntryGraphDraft {
        graph,
        table,
        directory,
        tokens,
        hidden: result.r1,
        notice,
        kept_tokens,
        landed,
    })
}

/// What decides a target entry graph's topology: everything it is built
/// from but its numbers (its token table's extents, its rows and width),
/// which are facts of its trace (`certify_entry`).
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct EntryTopology {
    backend: BackendName,
    source: EntryTokens,
    activation: Element,
    notifies: bool,
    landing: Option<CheckedLandingEntries>,
    keeps_tokens: bool,
    transform: (RealIndex, i32, RealIndex),
    table: WeightView<FactIndex>,
    rows: FactIndex,
    hidden: FactIndex,
}

/// The target entry graphs' traces, one per topology.
pub(super) static ENTRY_TRACES: NativeGraphTraces<EntryTopology, ()> = NativeGraphTraces::new();

/// The template of a target entry graph of `topology` at `facts`.
fn record_entry(
    topology: &EntryTopology,
    facts: &[Dim],
    reals: &seismic::Reals,
) -> Result<(NativeGraphResourceTemplate, ()), GraphError> {
    let facts = Facts::new(facts, reals);
    let weight = facts.dims(topology.table.clone());
    let elements = [("EW", weight.resident), ("A", topology.activation)];
    let (scale, normalize, epsilon) = topology.transform;
    let draft = entry_graph_topology(
        NativeGraphMetadata::new_template(topology.backend),
        &elements,
        topology.notifies.then_some(&[][..]),
        topology.landing.as_ref().map(CheckedLandingEntries::entries),
        topology.keeps_tokens,
        &weight,
        EntryWidth { hidden: &facts.dim(topology.hidden), transform: (facts.real(scale).f32(), normalize, facts.real(epsilon).f32()) },
        &facts.dim(topology.rows),
        topology.source,
    )?;
    Ok((draft.graph.seal_template()?, ()))
}

/// The target entry graph of `source`, certified over the row classes
/// `row_classes`, the largest `largest_rows`: from the entry graphs' traces
/// (`Certified::traced_each`).
pub(super) fn certify_entry<L: Certified>(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    source: EntryTokens,
    row_classes: &[u64],
    largest_rows: u64,
) -> Result<L, GraphError> {
    let (scale, normalize, epsilon) = embedding_transform(geometry);
    let mut facts = FactSink::new();
    let topology = EntryTopology {
        backend,
        source,
        activation: activation(geometry),
        notifies: notifies_selection(load),
        landing: CheckedLandingEntries::of(load, TARGET_TABLE)?,
        keeps_tokens: keeps_token_rows(load),
        transform: (facts.real(f64::from(scale)), normalize, facts.real(f64::from(epsilon))),
        table: facts.index(WeightView::planned(load, TARGET_TABLE)?),
        rows: facts.fact(largest_rows),
        hidden: facts.fact(geometry.hidden),
    };
    let rows = ClassFact { fact: topology.rows.position(), scope: None, dimension: "M" };
    let slices = [vec![NativeGraphClassSlice::new().dimension("M", row_classes.iter().copied())]];
    let [(layout, ())] = <[_; 1]>::try_from(
        L::traced_each(&ENTRY_TRACES, &topology, &facts, &[rows], &slices, record_entry)
            .map_err(|error| error.context("target entry graph"))?,
    )
    .unwrap_or_else(|_| unreachable!("one slice set certifies to one layout"));
    Ok(layout)
}

/// What an entry graph embeds into: the rows' width and `embedding_rows`'
/// transform (`embedding_transform`).
#[derive(Clone, Copy)]
pub(super) struct EntryWidth<'d> {
    pub(super) hidden: &'d Dim,
    pub(super) transform: (Argument<f32>, i32, Argument<f32>),
}

/// A constant transform (`embedding_transform`) as `embedding_rows` takes
/// its arguments.
pub(super) fn transform_arguments((scale, normalize, epsilon): (f32, i32, f32)) -> (Argument<f32>, i32, Argument<f32>) {
    (scale.into(), normalize, epsilon.into())
}

/// `embedding_rows`' (`scale`, `normalize`, `epsilon`) for the decoder's
/// entry form: the text-row scale, and the unweighted RMS when present (its
/// epsilon is not read otherwise).
pub(crate) fn embedding_transform(geometry: &Decoder) -> (f32, i32, f32) {
    let scale = match geometry.entry.scale {
        EmbeddingScale::Unit => 1.0,
        EmbeddingScale::SqrtHidden => (geometry.hidden as f32).sqrt(),
    };
    match geometry.entry.norm {
        Some(norm) => (scale, 1, norm.epsilon as f32),
        None => (scale, 0, 0.0),
    }
}

#[cfg(test)]
pub(crate) fn checked_entry_graph_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    rows: u64,
    uploaded: bool,
) -> Result<NativeGraphStorageBytes, GraphError> {
    let weight = load.weight(TARGET_TABLE)?;
    let elements = [("EW", weight.resident), ("A", activation(geometry))];
    let landing = CheckedLandingEntries::of(load, TARGET_TABLE)?;
    let draft = entry_graph_topology(
        NativeGraphMetadata::new(backend),
        &elements,
        notifies_selection(load).then_some(&[][..]),
        landing.as_ref().map(CheckedLandingEntries::entries),
        keeps_token_rows(load),
        &weight,
        EntryWidth { hidden: &Dim::from(geometry.hidden), transform: transform_arguments(embedding_transform(geometry)) },
        &Dim::from(rows),
        if uploaded {
            EntryTokens::Uploaded
        } else {
            EntryTokens::Selected
        },
    )?;
    GraphDraft::seal(draft.graph)
}
