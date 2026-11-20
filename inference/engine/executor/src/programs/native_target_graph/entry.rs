use super::weights::{
    CheckedLandingEntries, LandedReadPorts, LandingEntries, PlannedWeights, WeightView, activation, landed_read,
};
use crate::programs::graph::draft::{Certified, ElementBoundDraft, GraphDraft};
use crate::{
    ModelLoadPlan, TargetProgramPlan,
    native::AttestedTarget,
    operators::per_layer::graph::{PerLayerEntryEntries, PerLayerEntryPorts, per_layer_entry, split_entry},
    programs::graph::GraphError,
    stored_table::SelectionNotice,
};
use magnitude_family_contracts::{Decoder, EmbeddingScale, WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{embedding_rows, selection_notice};
#[cfg(test)]
use seismic::{NativeGraphMetadata, NativeGraphStorageBytes};
use seismic::{
    Argument, BackendName, Device, Dim, Element, NativeGraphLayout,
    NativeGraphPlan, NativePort, NativeTimelinePort, WorkflowTensor,
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
    /// Of a step uploading the host's tokens in a load that fronts
    /// (`ModelLoadPlan::fronts`): the host's guesses its front rows run on
    /// (`[rows, 2]` guess, kind rows, `TargetLaunchCore::host_guesses`),
    /// kept as its tokens are (`GroupBuffers::kept_guesses`), which its step
    /// words read (`routed_step`).
    pub kept_guesses: Option<KeptTokens>,
    /// Of a step reading selected tokens of a table with landing slots: its
    /// landed read's ports.
    pub landed: Option<LandedReadPorts>,
}

/// An entry graph's kept token rows (`ModelLoadPlan::keeps_token_rows`):
/// the host's token rows uploaded in the `sample_rows` layout, and the bound
/// token rows (`GroupBuffers::token_rows`) the entry copies them to for the
/// graphs after it.
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
        copy: Option<&seismic::NativeKernel<selection_notice::Entry>>,
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
            copy,
            load.notifies_selection(),
            landing.map(LandingEntries::native),
            load.keeps_token_rows(),
            load.fronts(),
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
            kept_guesses: draft.kept_guesses,
            landed: draft.landed,
        })
    }
}

/// The target's token table, which the target and its draft heads embed.
pub(crate) const TARGET_TABLE: WeightRole = WeightRole {
    scope: WeightScope::Target,
    kind: WeightKind::Embedding,
};

/// The rows of each copy of the selection notice (`SelectionNotice`) a load
/// whose steps read selected tokens of a stored table binds: the largest row
/// class; none without a stored table.
pub(super) fn notice_rows(load: &ModelLoadPlan, largest_rows: u64) -> Option<u64> {
    load.notifies_selection().then_some(largest_rows)
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

/// The `[rows, 2]` I32 extents of the token rows a model that keeps them
/// binds, over `rows` rows.
pub(super) fn token_rows_extents(load: &ModelLoadPlan, rows: u64) -> Option<[u64; 2]> {
    load.keeps_token_rows().then_some([rows, 2])
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
    kept_guesses: Option<KeptTokens>,
    landed: Option<LandedReadPorts>,
}

/// The entry graph: the step's token rows embedded, `copy` the token row
/// copy entry of a model that copies them
/// (`ModelLoadPlan::copies_token_rows`). Where it
/// `notifies` (a model with a stored table) it copies selected tokens for
/// the host first. With
/// `keeps_tokens` (a model whose blocks read the step's token rows) the
/// host's uploaded tokens are copied into the bound token rows
/// (`keep_tokens`), so later graphs read a device tensor of the step's token
/// rows whatever their source, as they read a selection. With
/// `keeps_guesses` (a load that fronts, which keeps its token rows) the
/// host's guesses of a step of its tokens are kept alike, for its step words.
/// Selected tokens of a table with landing slots are a landed read
/// (`landed_read`, `landing`).
#[allow(clippy::too_many_arguments)]
pub(super) fn entry_graph_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    embedding: G::Binding<'a, embedding_rows::Entry>,
    copy: Option<G::Binding<'a, selection_notice::Entry>>,
    notifies: bool,
    landing: Option<LandingEntries<'a, G>>,
    keeps_tokens: bool,
    keeps_guesses: bool,
    weight: &WeightView,
    width: EntryWidth<'_>,
    rows: &Dim,
    source: EntryTokens,
) -> Result<EntryGraphDraft<G>, GraphError> {
    if copy.is_some() != (notifies || keeps_tokens) {
        return Err("the token row copy entry is given exactly where token rows are copied".into());
    }
    if keeps_guesses && !keeps_tokens {
        return Err("a load that fronts keeps its token rows".into());
    }
    let table = graph.weight_port(weight.resident, &weight.shape)?;
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
        EntryTokens::Selected => graph.port(Element::i32(), &[rows, &two])?,
    };
    // Selected tokens reach the host first, as early in the step as they
    // can: the filler of a stored table learns from them the rows the step
    // names while the device runs the blocks before the table's gate.
    let notice = match (copy, notifies, source) {
        (Some(entry), true, EntryTokens::Selected) => {
            let mut copied = graph.port(Element::i32(), &[rows, &two])?;
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
    let kept_tokens = match (keeps_tokens, source) {
        (true, EntryTokens::Uploaded) => {
            let copy = copy.ok_or("kept token rows need the token row copy entry")?;
            Some(keep_tokens(&mut graph, copy, rows)?)
        }
        _ => None,
    };
    let kept_guesses = match (keeps_guesses, source) {
        (true, EntryTokens::Uploaded) => {
            let copy = copy.ok_or("kept guesses need the token row copy entry")?;
            Some(keep_tokens(&mut graph, copy, rows)?)
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
        kept_guesses,
        landed,
    })
}

/// The host's `[rows, 2]` rows of a step of `rows` rows (its tokens, the
/// embedding's uploaded rows being its table's indices, a streamed table's
/// slots; or its guesses), copied into the bound rows by `copy`.
pub(super) fn keep_tokens<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    copy: G::Binding<'a, selection_notice::Entry>,
    rows: &Dim,
) -> Result<KeptTokens, GraphError> {
    let uploaded = graph.input_for(copy, "selection", &[("M", rows)])?;
    let mut copied = graph.port(Element::i32(), &[rows, &Dim::from(2)])?;
    let dimensions = [("M", rows)];
    let args = selection_notice::WorkflowArgs {
        selection: uploaded.tensor().into(),
        notice: copied.tensor_mut().into(),
    };
    graph.enqueue::<selection_notice::Entry>(copy, &dimensions, args)?;
    Ok(KeptTokens { uploaded, rows: copied })
}

/// The target entry graph of `source`, certified over the row classes
/// `row_classes`.
pub(super) fn certify_entry<L: Certified>(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    source: EntryTokens,
    row_classes: &[u64],
) -> Result<L, GraphError> {
    let weight = load.weight(TARGET_TABLE)?;
    let elements = [("EW", weight.resident), ("A", activation(geometry))];
    let landing = CheckedLandingEntries::of(load, TARGET_TABLE)?;
    let (layout, _) = L::certify_group(backend, row_classes, |draft, rows| {
        let built = entry_graph_topology(
            draft,
            L::Draft::bind(&elements),
            load.copies_token_rows().then(|| L::Draft::bind(&[][..])),
            load.notifies_selection(),
            landing.as_ref().map(CheckedLandingEntries::entries),
            load.keeps_token_rows(),
            load.fronts(),
            &weight,
            EntryWidth {
                hidden: &Dim::from(geometry.hidden),
                transform: transform_arguments(embedding_transform(geometry)),
            },
            &Dim::from(rows),
            source,
        )
        .map_err(|error| error.context("target entry graph"))?;
        Ok((built.graph, ()))
    })
    .map_err(|error| error.context("target entry graph"))?;
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
        load.copies_token_rows().then_some(&[][..]),
        load.notifies_selection(),
        landing.as_ref().map(CheckedLandingEntries::entries),
        load.keeps_token_rows(),
        load.fronts(),
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
