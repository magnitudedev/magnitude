use super::prepared::{
    BlockGraphShape, BlockTaps, block_graph_shapes, block_scopes, block_taps, chain_rows,
};
use crate::{
    ModelLoadPlan, ResourceLimits, StateResourcePlan, TargetBlockProgramSlot, TargetProgramPlan,
    native::OutputScales,
    operators::{
        MixerKind, PairedBlock, block, paired_block,
        routed::{
            split::{Extents, OnceStage, RunStage, SplitValues, Stage},
            streamed,
        },
    },
    programs::graph::RowForm,
};
use magnitude_family_contracts::{Decoder, RoutedFfn, WeightScope};
use magnitude_state::LayerRef;
use seismic::{BackendName, Element, NativeGraphClassSlice};

/// A block graph's class: (rows, history segments, request slots, the
/// history row tiles of its launch's rows it lists, 0 for a class that lists
/// none; whether a selected read scores, 1, or every row of the launch reads
/// its whole history, 0 and every other block's).
pub(super) type BlockClass = (u64, u64, u64, u64, u64);

/// A block graph's class and the part of the block it holds (`Stage`).
pub(super) type StagedClass = (BlockClass, Stage);

/// The topology a block graph certifies in: its row form, whether it lists
/// history row tiles, whether it scores, its stage, and a pass's rows
/// (`Passes::lengths`), whose ports and merged extents are of its rows.
pub(super) type Topology = (RowForm, bool, bool, Stage, u64);

pub(super) fn class_topology((class, stage): StagedClass) -> Topology {
    let (rows, _, _, listed, scored) = class;
    let pass = if stage == Stage::Once(OnceStage::Pass) { rows } else { 0 };
    (RowForm::of(rows), listed > 0, scored > 0, stage, pass)
}

/// The routed feed-forward of block `index`'s feed-forward sublayer (a
/// parallel sublayer's routed branch's), if it has one.
pub(super) fn block_routed(geometry: &Decoder, index: usize) -> Result<Option<&RoutedFfn>, String> {
    let scope = block_scopes(index)?[1];
    Ok(geometry
        .sublayers()
        .find(|&(at, _)| WeightScope::TargetSublayer(at) == scope)
        .and_then(|(_, sublayer)| crate::operators::routed::routed_ffn(&sublayer.op)))
}

/// The routed feed-forward block `index`'s prompt rows split at: a layer
/// whose experts stream, or any routed layer of a chain, whose launches
/// merge their experts in a pass. A lone launch of resident experts has
/// nothing to merge, so its block stays whole; so does every block of a
/// load whose launches all decode, which run blocks whole
/// (`RunStage::served`).
pub(super) fn block_split<'g>(
    load: &ModelLoadPlan,
    geometry: &'g Decoder,
    index: usize,
    limits: ResourceLimits,
) -> Result<Option<&'g RoutedFfn>, String> {
    let merges = limits.chain_capacity.get() > 1 || streamed::streams(load, block_scopes(index)?[1]);
    let prompts = magnitude_batching::row_classes(limits.max_launch_rows)
        .last()
        .is_some_and(|&rows| !crate::operators::routed::fused_graph::decodes(rows as u64));
    Ok(block_routed(geometry, index)?.filter(|_| merges && prompts))
}

/// How a block is served, which decides its stages (`RunStage::served`):
/// whether the load streams its routed layer's experts, and the values a
/// split layer (`block_split`) crosses and merges over the chain's rows.
#[derive(Clone, Debug)]
pub(super) struct BlockServed {
    pub(super) streams: bool,
    pub(super) split: Option<SplitValues<Extents>>,
}

/// Each block's `BlockServed`, its rows crossed over a chain of
/// `chain_rows` rows and merged by the group's `passes`.
pub(super) fn block_served(
    load: &ModelLoadPlan,
    geometry: &Decoder,
    limits: ResourceLimits,
    chain_rows: u64,
    passes: Passes,
) -> Result<Vec<BlockServed>, String> {
    (0..geometry.blocks.len())
        .map(|index| {
            Ok(BlockServed {
                streams: streamed::streams(load, block_scopes(index)?[1]),
                split: block_split(load, geometry, index, limits)?
                    .map(|routed| SplitValues::of(routed, geometry, chain_rows, passes.rows()))
                    .transpose()?,
            })
        })
        .collect()
}

/// The stages of block `index`'s graphs of `rows` rows, of the blocks'
/// `served` (`block_served`).
pub(super) fn block_stages(served: &[BlockServed], index: usize, rows: u64) -> &'static [RunStage] {
    RunStage::served(served[index].split.is_some(), rows)
}

/// Whether a launch of `rows` rows splits at block `index`
/// (`RunStage::Front`): only such a launch runs the block's graphs sealed
/// once (`once_graphs`), its fill and its passes.
fn block_splits(served: &[BlockServed], index: usize, rows: u64) -> bool {
    block_stages(served, index, rows).contains(&RunStage::Front)
}

/// Whether a launch of `class` runs block `index`'s graph of `staged`: a
/// run (`Stage::Run`) only where its class is the launch's, and a graph the
/// block runs once a group (`Stage::Once`, its fill and its passes) only
/// where the launch splits at the block: a class served whole, as decode
/// is, runs none. The plan's census (`certify_target_family`) and the
/// sealed family (`PreparedTargetGraphs`) both count a launch's graphs by it.
pub(super) fn launch_runs(served: &[BlockServed], index: usize, class: BlockClass, (sealed, stage): StagedClass) -> bool {
    match stage {
        Stage::Run(_) => sealed == class,
        Stage::Once(_) => block_splits(served, index, class.0),
    }
}

/// The graphs sealed once of a block whose prompt rows `split` and whose
/// experts are `streamed` (`OnceStage::sealed`), each its stage and rows: its pass
/// at each length a group's `passes` take (`Passes::lengths`), its fill
/// once at the longest pass's rows. The one enumeration of a block's once
/// classes (`once_class`).
pub(super) fn once_graphs(streamed: bool, split: bool, passes: Passes) -> Vec<(OnceStage, u64)> {
    OnceStage::sealed(streamed, split)
        .iter()
        .flat_map(|&stage| match stage {
            OnceStage::Pass => passes.lengths().map(|rows| (stage, rows)).collect::<Vec<_>>(),
            OnceStage::Fill => vec![(stage, passes.rows())],
        })
        .collect()
}

/// One control of a launch's one upload of its per-step controls
/// (`LaunchControlLayout`), which every block graph of the launch binds
/// (`ControlSource::Bound`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LaunchControl {
    /// Each row's `[4]` coordinates.
    Coordinates,
    /// A history read's visible spans, `segments` a row.
    Visible { read: usize },
    /// A history read's visible spans padded to the most segments, for a
    /// block whose listing classes are sealed for that alone
    /// (`BlockListings::every_segment_count`).
    Padded { read: usize },
    Fresh { read: usize },
    Destinations { read: usize },
    /// The slots' `[B + 1, 2]` row segments and `[B]` recurrent words.
    Segments,
    Stop,
    PreviousBank,
    PreviousTape,
    FollowingBank,
}

impl LaunchControl {
    /// Whether the control has a row of each launch row (a launch's
    /// stateless tail takes the rows it reads), not one of each slot.
    pub(crate) fn per_row(self) -> bool {
        matches!(
            self,
            Self::Coordinates
                | Self::Visible { .. }
                | Self::Padded { .. }
                | Self::Fresh { .. }
                | Self::Destinations { .. }
        )
    }
}

/// The I32 controls of a target's launches, in the order a launch uploads
/// them (`seismic::NativeGraphFamilySlot::upload`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LaunchControlLayout {
    /// Per history read of a launch row, whether a block reading it lists
    /// its prefill rows' history tiles in classes sealed for the most
    /// segments alone (`BlockListings::segments`).
    pub(super) padded: Vec<bool>,
    pub(super) max_segments: u64,
    /// The upload region that holds any launch's controls.
    pub(super) region_bytes: u64,
}

impl LaunchControlLayout {
    /// The layout of a target's launches from its store's history `reads`
    /// and its blocks' prefill `listings`, of at most `rows` rows,
    /// `max_segments` segments and `slots` slots.
    pub(super) fn of(
        reads: &magnitude_state::HistoryReads,
        listings: &[BlockListings],
        rows: u64,
        max_segments: u64,
        slots: u64,
    ) -> Result<Self, String> {
        let mut padded = vec![false; reads.len()];
        for (index, listings) in listings.iter().enumerate() {
            if listings.tiles.is_empty() || listings.every_segment_count {
                continue;
            }
            let read = reads
                .read(magnitude_state::LayerRef::Target(index as u32))
                .ok_or("a listing block has no history read")?;
            padded[read] = true;
        }
        let mut layout = Self {
            padded,
            max_segments,
            region_bytes: 0,
        };
        // The most over the segment classes: fewer segments may pad.
        for segments in segment_classes(max_segments) {
            let controls = layout.controls(rows, segments, slots);
            let tensors = controls
                .iter()
                .map(|(_, extents)| (Element::i32(), extents.as_slice()))
                .collect::<Vec<_>>();
            let bytes =
                seismic::upload_region_bytes(&tensors).map_err(|error| error.to_string())?;
            layout.region_bytes = layout.region_bytes.max(bytes);
        }
        Ok(layout)
    }

    /// The upload region that holds any launch's controls.
    pub(crate) fn region_bytes(&self) -> u64 {
        self.region_bytes
    }

    /// A launch's controls of `rows` rows over `segments` segments and
    /// `slots` slots, in upload order, each with its extents. A read's
    /// spans are padded where a prefill launch has fewer segments than the
    /// most (`PreparedTargetGraphs::listed_segments`).
    pub(crate) fn controls(
        &self,
        rows: u64,
        segments: u64,
        slots: u64,
    ) -> Vec<(LaunchControl, Vec<u64>)> {
        let pads =
            segments < self.max_segments && !crate::operators::attention::graph::decodes(rows);
        let per_read = self.padded.iter().enumerate().flat_map(|(read, &padded)| {
            [
                Some((LaunchControl::Visible { read }, vec![rows, segments, 2])),
                (padded && pads)
                    .then(|| (LaunchControl::Padded { read }, vec![rows, self.max_segments, 2])),
                Some((LaunchControl::Fresh { read }, vec![rows, 2])),
                Some((LaunchControl::Destinations { read }, vec![rows])),
            ]
            .into_iter()
            .flatten()
        });
        std::iter::once((LaunchControl::Coordinates, vec![rows, 4]))
            .chain(per_read)
            .chain([
                (LaunchControl::Segments, vec![slots + 1, 2]),
                (LaunchControl::Stop, vec![slots]),
                (LaunchControl::PreviousBank, vec![slots]),
                (LaunchControl::PreviousTape, vec![slots]),
                (LaunchControl::FollowingBank, vec![slots]),
            ])
            .collect()
    }
}

/// The history segment classes of a launch of at most `max_segments`
/// segments, ascending: powers of two.
fn segment_classes(max_segments: u64) -> impl Iterator<Item = u64> {
    std::iter::successors(Some(1u64), move |segments| {
        segments.checked_mul(2).filter(|segments| *segments <= max_segments)
    })
}

/// The upload regions a step's runs of one block take, largest first: at
/// each of the step's row classes `rows`, its first stage's run and one
/// back's, whichever of its backs it runs, each writing at most the input
/// `upload(rows, stage)` of that stage's largest graph there; per region,
/// the most over row classes. A stage whose graphs write none takes none.
/// So whatever class a step runs the block at, each of its runs has a
/// region of its own that holds its input.
pub(super) fn block_upload_regions(
    rows: impl IntoIterator<Item = u64>,
    stages: impl Fn(u64) -> &'static [RunStage],
    upload: impl Fn(u64, RunStage) -> u64,
) -> Vec<u64> {
    let mut regions = Vec::new();
    for rows in rows {
        let stages = stages(rows);
        let back = stages[1..].iter().map(|&stage| upload(rows, stage)).max().unwrap_or(0);
        let mut runs = [upload(rows, stages[0]), back];
        runs.sort_unstable_by(|a, b| b.cmp(a));
        for (at, bytes) in runs.into_iter().filter(|&bytes| bytes > 0).enumerate() {
            match regions.get_mut(at) {
                Some(region) => *region = bytes.max(*region),
                None => regions.push(bytes),
            }
        }
    }
    regions
}

/// The passes a group runs over its routed layers: each over the regions,
/// of `region` rows, of at most `launches` of its chain's launches
/// (`group_pass_rows`). A chain of `n` launches passes over `min(n,
/// launches)` regions, so a pass reads no more of its group's crossing and
/// merged positions than its group's rows; a longer merge set is several
/// passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Passes {
    pub(super) region: u64,
    pub(super) launches: usize,
}

impl Passes {
    /// The most rows a pass covers.
    pub(crate) fn rows(self) -> u64 {
        self.of_chain(self.launches)
    }

    /// The most launches a pass covers.
    pub(crate) fn launches(self) -> usize {
        self.launches
    }

    /// The rows of the passes of a chain of `launches` launches.
    pub(crate) fn of_chain(self, launches: usize) -> u64 {
        self.region * launches.min(self.launches) as u64
    }

    /// The rows of each pass a group can run, a graph sealed for each.
    fn lengths(self) -> impl Iterator<Item = u64> {
        (1..=self.launches).map(move |launches| self.of_chain(launches))
    }
}

/// The class of block `index`'s graph of `stage` sealed once at a pass's
/// `capacity` rows (`Passes`): no mixer, so no segments or slots of its
/// own.
pub(super) fn once_class(capacity: u64, stage: OnceStage) -> StagedClass {
    ((capacity, 1, 1, 0, 0), Stage::Once(stage))
}

/// The shape of `slot`'s selected read, when its block reads one
/// (`AttentionBinding::selection`).
pub(super) fn slot_selection(slot: TargetBlockProgramSlot) -> Option<crate::SelectionShape> {
    match slot.mixer() {
        crate::MixerProgramSlot::Attention(binding) => {
            binding.selection.map(|selection| selection.shape)
        }
        _ => None,
    }
}

/// Whether `slot`'s block has classes that score a selected read.
pub(super) fn slot_scores(slot: TargetBlockProgramSlot) -> bool {
    slot_selection(slot).is_some()
}

/// The classes of a block's graphs of one row class that list their
/// launch's history row tiles, beside the class that lists none.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct BlockListings {
    /// Their tile counts, ascending
    /// (`HistoryStorePlan::listed_tile_classes`); empty when the row class
    /// has none.
    pub(super) tiles: Vec<u64>,
    /// Whether each is sealed for every history segment count, or for the
    /// most alone (`DecodedHistoryForm::lists_every_segment_count`).
    every_segment_count: bool,
}

impl BlockListings {
    /// The segment count of the listing class a launch over `segments`
    /// history segments takes, of at most `max_segments`.
    pub(super) fn segments(&self, segments: u64, max_segments: u64) -> u64 {
        match self.every_segment_count {
            true => segments,
            false => max_segments,
        }
    }
}

/// The listing classes of block `index`'s graphs of `rows` rows: those of
/// an attention block's prefill rows, on a plan whose device of `backend`
/// has a form that reads its heads' history decoded
/// (`StateResourcePlan::lists_history_tiles`).
pub(super) fn block_listings(
    state: &StateResourcePlan,
    backend: BackendName,
    slot: TargetBlockProgramSlot,
    index: usize,
    rows: u64,
) -> Result<BlockListings, String> {
    let form = match slot.mixer() {
        crate::MixerProgramSlot::Attention(binding)
            if crate::operators::attention::graph::lists_history_tiles(binding) =>
        {
            state.decoded_history_form(backend, binding.shape.width)
        }
        _ => None,
    };
    let Some(form) = form.filter(|_| !crate::operators::attention::graph::decodes(rows)) else {
        return Ok(BlockListings::default());
    };
    Ok(BlockListings {
        tiles: state
            .target_state()
            .layer_history(LayerRef::Target(index as u32))
            .ok_or("attention block has no history domain")?
            .store
            .listed_tile_classes(form)?,
        every_segment_count: form.lists_every_segment_count(),
    })
}

/// The classes a block's sealed graph distinguishes for one row class. An
/// attention block reads history segments and is independent of request
/// slots; each segment count has a class per `listed` tile count and one
/// that lists none. A recurrent block
/// binds exact per-slot bank tables (every slot count a launch of `rows`
/// rows can serve, up to the launch slot bound) and is independent of
/// segments. Everything else is per row. A block that `scores` a selected
/// read has, per segment count, a class that scores (and lists none).
pub(super) fn block_classes(
    mixer: MixerKind,
    rows: u64,
    max_slots: u64,
    max_segments: u64,
    listings: &BlockListings,
    scores: bool,
) -> Vec<BlockClass> {
    match mixer {
        MixerKind::Attention => segment_classes(max_segments).flat_map(|segments| {
            let listed = listings
                .tiles
                .iter()
                .copied()
                .filter(move |_| listings.segments(segments, max_segments) == segments);
            std::iter::once(0)
                .chain(listed)
                .map(move |listed| (rows, segments, 1, listed, 0))
                .chain(scores.then_some((rows, segments, 1, 0, 1)))
        })
        .collect(),
        MixerKind::Recurrent => (1..=rows.min(max_slots))
            .map(|slots| (rows, 1, slots, 0, 0))
            .collect(),
    }
}

/// The class of a block's graph serving a launch of `rows` rows over
/// `segments` history segments and `slots` requests, listing `listed` of the
/// launch's history row tiles, scoring a selected read when `scored`.
pub(super) fn block_class(
    mixer: MixerKind,
    rows: u64,
    segments: u64,
    slots: u64,
    listed: u64,
    scored: bool,
) -> BlockClass {
    match mixer {
        MixerKind::Attention => (rows, segments, 1, listed, u64::from(scored)),
        MixerKind::Recurrent => (rows, 1, slots, 0, 0),
    }
}

/// The admitted classes of one block for a `rows`-row launch, as one slice:
/// the row count, and every history segment count (attention) or request
/// slot count (recurrent) that launch can serve. A routed feed-forward's
/// grouped entries size their tile table from the rows they group,
/// `grouped_rows` (`block::feed_forward_class_slice`).
pub(super) fn block_class_slice(
    block: &PairedBlock,
    rows: u64,
    grouped_rows: u64,
    max_slots: u64,
    max_segments: u64,
    listed: &[u64],
    scored: bool,
) -> Result<NativeGraphClassSlice, String> {
    let mixer = block.mixer.kind();
    // The segment and slot counts are those of every listing class.
    let classes = block_classes(
        mixer,
        rows,
        max_slots,
        max_segments,
        &BlockListings::default(),
        false,
    );
    let slice = NativeGraphClassSlice::new()
        .dimension("M", [rows])
        .dimension("O", [rows]);
    let slice = match mixer {
        MixerKind::Attention if !listed.is_empty() => slice
            .dimension("R", classes.iter().map(|&(_, segments, ..)| segments))
            .dimension("HT", listed.iter().copied()),
        MixerKind::Attention => {
            slice.dimension("R", classes.iter().map(|&(_, segments, ..)| segments))
        }
        MixerKind::Recurrent => slice.dimension("B", classes.iter().map(|&(_, _, slots, ..)| slots)),
    };
    let slice = match block.mixer {
        crate::operators::Mixer::Attention(operator) if scored => {
            crate::operators::attention::graph::decode_class_segments(slice, &operator.reads)
        }
        _ => slice,
    };
    // An n-gram input reads its table's rows for each batch row's heads.
    let slice = match block.ngram {
        Some(ngram) => slice.scoped(
            "stored_rows",
            "M",
            [rows
                * ngram
                    .op
                    .hash
                    .heads()
                    .map_err(|error| error.to_string())?],
        ),
        None => slice,
    };
    block::feed_forward_class_slice(block, rows, grouped_rows, slice)
}

/// The blocks' output scales as their holder knows them. They are values of
/// a block's sealed graphs, so blocks share plans only at equal scales.
#[derive(Clone, Copy)]
pub(super) enum BlockOutputScales<'a> {
    /// As read from the artifact, per block.
    Read(&'a [OutputScales]),
    /// As the program plan fixes them: 1 for every sublayer that reads no
    /// layer scale (`reads_layer_scale`), and unknown for one that does,
    /// which the enumeration then takes as `UnreadScales` says.
    Planned(UnreadScales),
}

/// What an enumeration takes the layer scales it has not read to be. Blocks
/// of one shape share a plan exactly where their scales are equal, so the
/// two bound the plans a load seals from either side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UnreadScales {
    /// No two equal: every block that reads one seals its own plans. No
    /// load seals more.
    Distinct,
    /// All equal: blocks of one shape seal one block's plans. No load seals
    /// fewer.
    Equal,
}

/// The block graphs of a target: the (block, class) plans a load seals and
/// those a block holds of an earlier block's.
pub(super) struct BlockGraphEnumeration {
    /// The plans sealed, in sealing order: by block, then by staged class
    /// ascending (`StagedClass`'s order), as the sealer holds them.
    pub(super) sealing: Vec<(usize, StagedClass)>,
    /// The plans an earlier block of the same shape and output scales
    /// seals, with that block.
    pub(super) sharing: Vec<(usize, StagedClass, usize)>,
    /// Per block, its prefill rows' listing classes.
    pub(super) listings: Vec<BlockListings>,
    /// Where a launch's controls lie in its one upload of them.
    pub(super) controls: LaunchControlLayout,
    pub(super) launch_classes: std::collections::BTreeSet<BlockClass>,
    /// Whether every block's sharing was decided on known output scales:
    /// always under `BlockOutputScales::Read`, and under
    /// `BlockOutputScales::Planned` when no two blocks of one shape read a
    /// layer scale. Otherwise `sealing` bounds the plans a load seals, from
    /// the side its `UnreadScales` names.
    pub(super) exact: bool,
}

/// What a target's graph family follows from its plan and limits before any
/// graph is drafted: the one derivation of the planner and the sealer.
pub(super) struct FamilyOutline {
    pub(super) backend: BackendName,
    /// The launches' row classes, ascending, never none.
    pub(super) row_classes: Vec<u64>,
    pub(super) largest_rows: u64,
    /// The rows of each launch-row buffer (`chain_rows`).
    pub(super) chain_rows: u64,
    pub(super) max_slots: u64,
    /// The most history segments a launch has.
    pub(super) max_segments: u64,
    /// Per block, its mixer (`block_mixers`), its tap indices and its graph
    /// shape, with the width of the taps of a target that has them.
    pub(super) mixers: Vec<MixerKind>,
    pub(super) tapped: Vec<BlockTaps>,
    pub(super) shapes: Vec<BlockGraphShape>,
    pub(super) tap_width: Option<u64>,
}

impl FamilyOutline {
    pub(super) fn of(
        backend: BackendName,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
    ) -> Result<Self, String> {
        if plan.blocks().len() != geometry.blocks.len() {
            return Err("planned target block count disagrees with geometry".into());
        }
        let row_classes = magnitude_batching::row_classes(limits.max_launch_rows)
            .into_iter()
            .map(|rows| rows as u64)
            .collect::<Vec<_>>();
        let largest_rows = *row_classes.last().ok_or_else(|| {
            format!("target batch row bound {} has no row class", limits.max_launch_rows)
        })?;
        let (tap_width, tapped) = block_taps(plan, geometry)?;
        let shapes = block_graph_shapes(load, geometry, state, &tapped)?;
        Ok(Self {
            backend,
            row_classes,
            largest_rows,
            chain_rows: chain_rows(largest_rows, limits)?,
            max_slots: u64::try_from(limits.max_launch_slots)
                .map_err(|_| "target request slot bound exceeds u64")?,
            max_segments: u64::try_from(
                state
                    .target_state()
                    .span_limit()
                    .checked_next_power_of_two()
                    .ok_or("target segment class overflows")?,
            )
            .map_err(|_| "target segment class exceeds u64")?,
            mixers: block_mixers(geometry)?,
            tapped,
            shapes,
            tap_width,
        })
    }

    /// The first block of `index`'s shape, whose layouts it holds.
    pub(super) fn first_of_shape(&self, index: usize) -> usize {
        (0..index)
            .find(|&first| self.shapes[first] == self.shapes[index])
            .unwrap_or(index)
    }
}

/// Enumerate the block graphs of `plan` over `outline`'s row classes: the one
/// enumeration of the sealer (`PreparedTargetGraphs::prepare`, with the
/// scales it read) and of the planner (`certify_target_family`, with the
/// planned ones), so the planned sealed graphs are the sealed ones. The
/// blocks' groups run `passes`.
pub(super) fn enumerate_block_graphs(
    outline: &FamilyOutline,
    served: &[BlockServed],
    passes: Passes,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    scales: BlockOutputScales<'_>,
) -> Result<BlockGraphEnumeration, String> {
    let FamilyOutline { backend, ref row_classes, max_slots, max_segments, ref mixers, ref shapes, .. } = *outline;
    let blocks = plan.blocks();
    let unit = |index: usize| {
        !crate::native::output_tails(blocks[index])
            .into_iter()
            .any(crate::native::reads_layer_scale)
    };
    let equal_scales = |first: usize, index: usize| match scales {
        BlockOutputScales::Read(scales) => scales[first] == scales[index],
        BlockOutputScales::Planned(UnreadScales::Distinct) => unit(first) && unit(index),
        // Blocks of one shape read the same scales or none.
        BlockOutputScales::Planned(UnreadScales::Equal) => true,
    };
    let shared = (0..blocks.len())
        .map(|index| {
            (0..index).find(|&first| shapes[first] == shapes[index] && equal_scales(first, index))
        })
        .collect::<Vec<_>>();
    let exact = match scales {
        BlockOutputScales::Read(_) => true,
        BlockOutputScales::Planned(_) => (0..blocks.len()).all(|index| {
            (0..index).all(|first| shapes[first] != shapes[index] || (unit(first) && unit(index)))
        }),
    };
    // Per row class, each block's listing classes; per block, those of the
    // largest row class that has any.
    let listings_at = row_classes
        .iter()
        .map(|&rows| {
            blocks
                .iter()
                .enumerate()
                .map(|(index, &slot)| block_listings(state, backend, slot, index, rows))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut block_listed = vec![BlockListings::default(); blocks.len()];
    for listings in &listings_at {
        for (index, listings) in listings.iter().enumerate() {
            if !listings.tiles.is_empty() {
                block_listed[index] = listings.clone();
            }
        }
    }
    let mut enumeration = BlockGraphEnumeration {
        sealing: Vec::new(),
        sharing: Vec::new(),
        controls: LaunchControlLayout::of(
            &state.target_state().history_reads()?,
            &block_listed,
            outline.largest_rows,
            max_segments,
            max_slots,
        )?,
        listings: block_listed,
        launch_classes: std::collections::BTreeSet::new(),
        exact,
    };
    for (&rows, listings) in row_classes.iter().zip(&listings_at) {
        for (index, &slot) in blocks.iter().enumerate() {
            for class in block_classes(
                mixers[index],
                rows,
                max_slots,
                max_segments,
                &listings[index],
                slot_scores(slot),
            ) {
                enumeration.launch_classes.insert(class);
                for &stage in block_stages(served, index, rows) {
                    let staged = (class, Stage::Run(stage));
                    match shared[index] {
                        Some(first) => enumeration.sharing.push((index, staged, first)),
                        None => enumeration.sealing.push((index, staged)),
                    }
                }
            }
        }
    }
    // Each block's graphs sealed once; then every block's classes in
    // ascending order, as the sealer holds them.
    for (index, block) in served.iter().enumerate() {
        for (stage, rows) in once_graphs(block.streams, block.split.is_some(), passes) {
            match shared[index] {
                Some(first) => enumeration.sharing.push((index, once_class(rows, stage), first)),
                None => enumeration.sealing.push((index, once_class(rows, stage))),
            }
        }
    }
    enumeration.sealing.sort_unstable();
    enumeration.sharing.sort_unstable();
    Ok(enumeration)
}

/// The mixer kind of each decoder block, which decides its graph classes.
fn block_mixers(geometry: &Decoder) -> Result<Vec<MixerKind>, String> {
    geometry
        .blocks
        .iter()
        .map(|block| {
            paired_block(block)
                .map(|paired| paired.mixer.kind())
                .map_err(|error| error.to_string())
        })
        .collect()
}
