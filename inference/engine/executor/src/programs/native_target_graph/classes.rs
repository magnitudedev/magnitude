use super::prepared::{
    BlockGraphShape, BlockTaps, block_graph_shapes, block_scopes, block_taps, chain_rows,
};
use crate::{
    ModelLoadPlan, ResourceLimits, StateResourcePlan, TargetBlockProgramSlot, TargetProgramPlan,
    native::OutputScales,
    operators::{
        BlockKeeps, PairedBlock, block, paired_block,
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

/// Whether a launch of `class` may be a decode step's: one of no more than
/// its rows (`ResourceLimits::decode_rows`). Their
/// layouts differ (`RowForm`, and whether a class lists or scores), so what
/// a family keeps for its decode launches in flight is of every one of them
/// (`seismic::NativeGraphFamily::with_launches`).
pub(super) fn decode_launch(class: BlockClass, limits: ResourceLimits) -> bool {
    class.0 <= limits.decode_rows()
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

/// How a block is served, which decides its stages (`RunStage::served`,
/// `RunStage::sealed`): whether the load streams its routed layer's experts
/// and whether that layer has a resident form (`streamed::resident_form`),
/// and the values a split layer (`block_split`) crosses and merges over the
/// chain's rows.
#[derive(Clone, Debug)]
pub(super) struct BlockServed {
    pub(super) streams: bool,
    pub(super) resident: bool,
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
                resident: streamed::resident_form(load, block_scopes(index)?[1]),
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

/// The stages block `index`'s graphs of `rows` rows are sealed in
/// (`RunStage::sealed`): its runs' stages and its resident form.
pub(super) fn block_sealed_stages(served: &[BlockServed], index: usize, rows: u64) -> &'static [RunStage] {
    RunStage::sealed(served[index].split.is_some(), served[index].resident, rows)
}

/// Whether a launch of `rows` rows splits at block `index`
/// (`RunStage::Front`): only such a launch runs the block's graphs sealed
/// once (`once_graphs`), its fill and its passes.
fn block_splits(served: &[BlockServed], index: usize, rows: u64) -> bool {
    block_stages(served, index, rows).contains(&RunStage::Front)
}

/// The graphs each launch class runs, of a family's staged block graphs,
/// each held as a `T`: a run (`Stage::Run`) only where its class is the
/// launch's, and a graph the block runs once a group (`Stage::Once`, its
/// fill and its passes) only where the launch splits at the block
/// (`block_splits`): a class served whole, as decode is, runs none. A decode
/// launch of a bank planned whole runs either form of a streamed block,
/// as its bank is whole or not as it begins (`RunStage::Resident`): it counts
/// both, so what is held for its every class (`decode_launch`;
/// `LaunchGraphs::held`, and on CUDA the graphs
/// `seismic::NativeGraphFamily::record_decode` records ahead) holds either
/// form's, and a step that switches records none. The plan's census
/// (`certify_target_family`) and the sealed family (`PreparedTargetGraphs`)
/// both count a launch's graphs by it.
pub(super) struct LaunchRuns<'s, T> {
    served: &'s [BlockServed],
    /// The runs, by their class.
    runs: std::collections::BTreeMap<BlockClass, Vec<T>>,
    /// The graphs run once a group, by their block.
    once: std::collections::BTreeMap<usize, Vec<T>>,
}

impl<'s, T> LaunchRuns<'s, T> {
    /// The index of `graphs`, each its block, its staged class and what it
    /// holds, of blocks served as `served` (`block_served`).
    pub(super) fn of(served: &'s [BlockServed], graphs: impl IntoIterator<Item = (usize, StagedClass, T)>) -> Self {
        let mut index = Self { served, runs: Default::default(), once: Default::default() };
        for (block, (class, stage), graph) in graphs {
            match stage {
                Stage::Run(_) => index.runs.entry(class).or_default().push(graph),
                Stage::Once(_) => index.once.entry(block).or_default().push(graph),
            }
        }
        index
    }

    /// The graphs a launch of `class` runs.
    pub(super) fn of_class(&self, class: BlockClass) -> impl Iterator<Item = &T> {
        let once = self
            .once
            .iter()
            .filter(move |&(&block, _)| block_splits(self.served, block, class.0))
            .flat_map(|(_, graphs)| graphs);
        self.runs.get(&class).into_iter().flatten().chain(once)
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
    /// A picked pooled source's Block read: each row's complete groups
    /// (`[rows]`, the rows its visible spans list, `(t + 1) / rate`).
    Complete { read: usize },
    /// A picked pooled source's Block read: the launch's picks table
    /// (`[rows, top]`, uploaded -1), which its own layer's graph stores, a
    /// later Reindex layer's graph stores over, and the Reuse layers' graphs
    /// after each read (`operators::attention::pooled`).
    Picks { read: usize },
    /// A picked pooled source's Block read whose own indexer produces a
    /// candidate pool: the launch's pool table (`[rows, blocks + segments,
    /// 2]` spans, uploaded -1), which that layer's graph stores and the
    /// restricted indexers' graphs read.
    Pool { read: usize },
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
                | Self::Complete { .. }
                | Self::Picks { .. }
                | Self::Pool { .. }
        )
    }
}

/// The index lists over a picked pooled source's Block read: the picks
/// per row of every indexer over it (`top`), and the blocks of the
/// candidate pool its own indexer produces, if it does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PickedRead {
    pub top: u64,
    pub pool_blocks: Option<u64>,
}

/// The I32 controls of a target's launches, in the order a launch uploads
/// them (`seismic::NativeGraphFamilySlot::upload`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LaunchControlLayout {
    /// Per history read of a launch row, whether a block reading it lists
    /// its prefill rows' history tiles in classes sealed for the most
    /// segments alone (`BlockListings::segments`).
    pub(super) padded: Vec<bool>,
    /// Per history read, the index lists over it: a picked pooled source's
    /// Block read takes its rows' complete groups, the launch's picks table
    /// and, when its own indexer produces one, its candidate pool table.
    pub(super) picked: Vec<Option<PickedRead>>,
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
        picked: &[(usize, PickedRead)],
        rows: u64,
        max_segments: u64,
        slots: u64,
    ) -> Result<Self, String> {
        let mut picks = vec![None; reads.len()];
        for &(read, lists) in picked {
            *picks.get_mut(read).ok_or("a picked source's read is not the store's")? = Some(lists);
        }
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
            picked: picks,
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
        let per_read = self.padded.iter().zip(&self.picked).enumerate().flat_map(
            |(read, (&padded, &picks))| {
                [
                    Some((LaunchControl::Visible { read }, vec![rows, segments, 2])),
                    (padded && pads).then(|| {
                        (LaunchControl::Padded { read }, vec![rows, self.max_segments, 2])
                    }),
                    Some((LaunchControl::Fresh { read }, vec![rows, 2])),
                    Some((LaunchControl::Destinations { read }, vec![rows])),
                    picks.map(|_| (LaunchControl::Complete { read }, vec![rows])),
                    picks.map(|lists| (LaunchControl::Picks { read }, vec![rows, lists.top])),
                    picks.and_then(|lists| lists.pool_blocks).map(|blocks| {
                        (LaunchControl::Pool { read }, vec![rows, blocks + segments, 2])
                    }),
                ]
                .into_iter()
                .flatten()
            },
        );
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

    /// The passes of a merge set of `merged` runs: as many of
    /// [`Self::launches`] runs as it fills, then one of the rest, if any.
    fn merge_passes(self, merged: usize) -> (usize, usize) {
        (merged / self.launches, merged % self.launches)
    }

    /// The passes, as ranges of its runs, a group of `runs` runs makes at
    /// a block whose merge set is `merged` runs: a pass of each single run
    /// first, then the merge set's ([`Self::merge_passes`]). The one split
    /// of a group into passes: the engine runs it (`run_sets`) and the
    /// graph census counts it ([`chained_once_runs`]).
    pub(crate) fn sets(self, runs: usize, merged: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
        let merged = merged.clamp(1, runs.max(1)).min(runs);
        let singles = runs - merged;
        let (full, rest) = self.merge_passes(merged);
        let most = self.launches;
        (0..singles)
            .map(|start| start..start + 1)
            .chain((0..full).map(move |pass| singles + pass * most..singles + (pass + 1) * most))
            .chain((rest > 0).then(|| runs - rest..runs))
    }
}

/// What a chain of `launches` launches, in one submission, runs of a
/// split block's graphs run once a group beyond one run of each, which a
/// launch's own graphs hold (`launch_runs`): of each pass length the most
/// passes of it any merge set makes ([`Passes::sets`]), and a fill for
/// every launch; each `(stage, rows)` with its further runs, each a graph
/// of its own where one graph serves one run of its shape a submission
/// (`BackendName::graphs_update_in_flight`). The plan's census
/// (`certify_target_family`) and the sealed family (`PreparedTargetGraphs`)
/// both count a chain's graphs by it.
pub(super) fn chained_once_runs(passes: Passes, launches: usize) -> Vec<((OnceStage, u64), u64)> {
    // Passes by length (in runs), the most any merge set makes.
    let mut most = vec![0u64; passes.launches + 1];
    for merged in 1..=launches {
        // Its single runs' passes, then its merge set's (`Passes::sets`).
        let (singles, (full, rest)) = (launches - merged, passes.merge_passes(merged));
        let of_length = |length: usize| {
            usize::from(length == 1) * singles + usize::from(length == passes.launches) * full + usize::from(length == rest)
        };
        for length in [1, passes.launches, rest].into_iter().filter(|&length| length > 0) {
            most[length] = most[length].max(of_length(length) as u64);
        }
    }
    most.into_iter()
        .enumerate()
        .skip(1)
        .map(|(length, count)| ((OnceStage::Pass, passes.of_chain(length)), count.saturating_sub(1)))
        .chain([((OnceStage::Fill, passes.rows()), launches.saturating_sub(1) as u64)])
        .filter(|&(_, further)| further > 0)
        .collect()
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
            state.decoded_history_form(backend, binding.shape.widths.width)
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

/// The classes a block's sealed graph distinguishes for one row class, by
/// the state it `keeps`. A block reading a history is classed by its
/// history segments: each segment count has a class per `listed` tile count
/// and one that lists none, and, of a block that `scores` a selected read,
/// one that scores (and lists none). A block keeping a recurrent bank binds
/// exact per-slot bank tables, so it is classed by every slot count a launch
/// of `rows` rows can serve, up to the launch slot bound. A block keeping
/// both has a class of each slot count for each of those. Everything else
/// is per row.
pub(super) fn block_classes(
    keeps: BlockKeeps,
    rows: u64,
    max_slots: u64,
    max_segments: u64,
    listings: &BlockListings,
    scores: bool,
) -> Vec<BlockClass> {
    let slots = match keeps.bank() {
        true => 1..=rows.min(max_slots),
        false => 1..=1,
    };
    let topologies: Vec<(u64, u64, u64)> = match keeps.history() {
        true => segment_classes(max_segments)
            .flat_map(|segments| {
                let listed = listings
                    .tiles
                    .iter()
                    .copied()
                    .filter(move |_| listings.segments(segments, max_segments) == segments);
                std::iter::once(0)
                    .chain(listed)
                    .map(move |listed| (segments, listed, 0))
                    .chain(scores.then_some((segments, 0, 1)))
            })
            .collect(),
        false => vec![(1, 0, 0)],
    };
    topologies
        .into_iter()
        .flat_map(|(segments, listed, scored)| {
            slots.clone().map(move |slots| (rows, segments, slots, listed, scored))
        })
        .collect()
}

/// The class of a block's graph serving a launch of `rows` rows over
/// `segments` history segments and `slots` requests, listing `listed` of the
/// launch's history row tiles, scoring a selected read when `scored`: the
/// axes of the state the block `keeps` (`block_classes`).
pub(super) fn block_class(
    keeps: BlockKeeps,
    rows: u64,
    segments: u64,
    slots: u64,
    listed: u64,
    scored: bool,
) -> BlockClass {
    let (history, bank) = (keeps.history(), keeps.bank());
    (
        rows,
        if history { segments } else { 1 },
        if bank { slots } else { 1 },
        if history { listed } else { 0 },
        u64::from(history && scored),
    )
}

/// The class dimensions of a block's slices (`block_class_slice`) that are
/// numbers of its split, `listed` of a regime that lists history tiles: what
/// its traces vary per class (`seismic::ClassFact`).
pub(super) fn block_class_numbers(
    block: &PairedBlock,
    listed: bool,
) -> Vec<(Option<&'static str>, &'static str, super::block::ClassNumber)> {
    use super::block::ClassNumber;
    let keeps = block.keeps();
    [(None, "M", ClassNumber::Rows), (None, "O", ClassNumber::Rows)]
        .into_iter()
        .chain(keeps.history().then_some((None, "R", ClassNumber::Segments)))
        .chain((keeps.history() && listed).then_some((None, "HT", ClassNumber::Listed)))
        .chain(match keeps {
            BlockKeeps::History => None,
            BlockKeeps::Bank => Some((None, "B", ClassNumber::Slots)),
            BlockKeeps::HistoryAndBank => Some((Some(crate::operators::ngram::graph::NGRAM_SLOTS), "B", ClassNumber::Slots)),
        })
        .collect()
}

/// The admitted classes of one block for a `rows`-row launch, as one slice:
/// the row count, and every history segment count (attention) or request
/// slot count (recurrent) that launch can serve. A routed feed-forward's
/// grouped entries size their tile table from the rows they group,
/// `grouped_rows` (`block::feed_forward_class_slice`). A candidate pool's
/// lists over the `picked` source take its blocks beside each segment count
/// (`pooled::candidate_class_segments`).
pub(super) fn block_class_slice(
    block: &PairedBlock,
    rows: u64,
    grouped_rows: u64,
    max_slots: u64,
    max_segments: u64,
    listed: &[u64],
    scored: bool,
    picked: Option<crate::PickedBinding>,
) -> Result<NativeGraphClassSlice, String> {
    let keeps = block.keeps();
    // The segment and slot counts are those of every listing class.
    let classes = block_classes(
        keeps,
        rows,
        max_slots,
        max_segments,
        &BlockListings::default(),
        false,
    );
    let slice = NativeGraphClassSlice::new()
        .dimension("M", [rows])
        .dimension("O", [rows]);
    let slice = match (keeps.history(), listed.is_empty()) {
        (false, _) => slice,
        (true, true) => slice.dimension("R", classes.iter().map(|&(_, segments, ..)| segments)),
        (true, false) => slice
            .dimension("R", classes.iter().map(|&(_, segments, ..)| segments))
            .dimension("HT", listed.iter().copied()),
    };
    // A recurrent mixer's slots are its whole block's `B`; an attention
    // block's are its n-gram input's alone (`NGRAM_SLOTS`).
    let slots = classes.iter().map(|&(_, _, slots, ..)| slots);
    let slice = match keeps {
        BlockKeeps::History => slice,
        BlockKeeps::Bank => slice.dimension("B", slots),
        BlockKeeps::HistoryAndBank => slice.scoped(crate::operators::ngram::graph::NGRAM_SLOTS, "B", slots),
    };
    let slice = match block.mixer {
        crate::operators::Mixer::Attention(operator) if scored => {
            crate::operators::attention::graph::decode_class_segments(slice, operator)
        }
        _ => slice,
    };
    let slice = crate::operators::attention::pooled::candidate_class_segments(
        slice,
        picked,
        classes.iter().map(|&(_, segments, ..)| segments),
    );
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
    /// The untapped graphs of the tapped blocks (`FamilyOutline::untapped`)
    /// sealed, as `sealing` holds them.
    pub(super) untapped_sealing: Vec<(usize, StagedClass)>,
    /// The untapped graphs of the tapped blocks another graph of their
    /// shape and output scales serves, with that graph's holder.
    pub(super) untapped_sharing: Vec<(usize, StagedClass, Holder)>,
    /// Whether every block's sharing was decided on known output scales:
    /// always under `BlockOutputScales::Read`, and under
    /// `BlockOutputScales::Planned` when no two blocks of one shape read a
    /// layer scale. Otherwise `sealing` bounds the plans a load seals, from
    /// the side its `UnreadScales` names.
    pub(super) exact: bool,
}

/// Which of a target's block graphs holds another's plan: a block's own
/// graphs (`BlockGraphs::runs`), or a tapped block's untapped ones
/// (`BlockGraphs::untapped`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Holder {
    Block(usize),
    Untapped(usize),
}

/// What a target's graph family follows from its plan and limits before any
/// graph is drafted: the one derivation of the planner and the sealer.
#[derive(Clone)]
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
    /// Per block, the state it keeps (`block_keeps`), its tap indices and
    /// its graph shape, with the width of the taps of a target that has
    /// them.
    pub(super) keeps: Vec<BlockKeeps>,
    pub(super) tapped: Vec<BlockTaps>,
    pub(super) shapes: Vec<BlockGraphShape>,
    /// Per block, the first block of its shape ([`Self::first_of_shape`]).
    pub(super) firsts: Vec<usize>,
    pub(super) tap_width: Option<u64>,
    /// The block that exports its output residual, which a head
    /// conditioned on it reads: the exit block, of such a head's target
    /// ([`Self::exports_residual`]).
    pub(super) residual: Option<usize>,
    /// The most rows a launch none of whose rows demands features takes:
    /// one row of each request, a decode step's
    /// (`ResourceLimits::decode_rows`, as `decode_launch`). A tapped block
    /// holds its untapped graphs of the classes of at most these rows
    /// (`Self::untapped`), which a decode launch's pins hold beside its
    /// tapped ones.
    pub(super) untapped_rows: u64,
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
        let residual = load.residual_conditioning().map(|_| geometry.blocks.len() - 1);
        let shapes = block_graph_shapes(load, geometry, state, &tapped, residual)?;
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
            keeps: block_keeps(geometry)?,
            tapped,
            firsts: firsts(&shapes),
            shapes,
            tap_width,
            residual,
            untapped_rows: limits.decode_rows(),
        })
    }

    /// The outline of the same target without its drafter's writes: no
    /// draft taps nor exported residual, so each block of a shape that
    /// differed only in them is of one shape again. Every other input of
    /// the family is the target's own.
    pub(super) fn undrafted(&self) -> Self {
        let shapes = self.shapes.iter().map(BlockGraphShape::undrafted).collect::<Vec<_>>();
        Self {
            tapped: vec![BlockTaps::default(); self.tapped.len()],
            firsts: firsts(&shapes),
            shapes,
            tap_width: None,
            residual: None,
            ..self.clone()
        }
    }

    /// The first block of `index`'s shape, whose layouts it holds.
    pub(super) fn first_of_shape(&self, index: usize) -> usize {
        self.firsts[index]
    }

    /// The shape of tapped block `index`'s untapped graphs, which a launch
    /// none of whose rows demands features runs in its place: the block
    /// without its drafter's writes (`BlockGraphShape::undrafted`), as a
    /// load without the drafter seals it. None for a block that taps
    /// nothing.
    pub(super) fn untapped(&self, index: usize) -> Option<BlockGraphShape> {
        self.tapped[index].positions().any().then(|| self.shapes[index].undrafted())
    }

    /// Where tapped block `index`'s untapped graphs take their layouts
    /// from: the first block of their shape, else the first tapped block's
    /// untapped graphs of it.
    pub(super) fn untapped_layouts(&self, index: usize) -> Option<Holder> {
        let shape = self.untapped(index)?;
        Some(match self.shapes.iter().position(|other| *other == shape) {
            Some(block) => Holder::Block(block),
            None => Holder::Untapped(
                (0..=index)
                    .find(|&first| self.untapped(first).as_ref() == Some(&shape))
                    .expect("a tapped block's untapped shape is its own"),
            ),
        })
    }

    /// Whether block `index`'s graphs export its output residual.
    pub(super) fn exports_residual(&self, index: usize) -> bool {
        self.residual == Some(index)
    }
}

/// Per block, the first block of its shape.
fn firsts(shapes: &[BlockGraphShape]) -> Vec<usize> {
    (0..shapes.len())
        .map(|index| (0..index).find(|&first| shapes[first] == shapes[index]).unwrap_or(index))
        .collect()
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
    let FamilyOutline { backend, ref row_classes, max_slots, max_segments, ref keeps, ref shapes, .. } = *outline;
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
    let untapped = (0..blocks.len()).map(|index| outline.untapped(index)).collect::<Vec<_>>();
    // A tapped block's untapped graph shares the plan of a block's own
    // graph of its shape, else of an earlier tapped block's untapped one.
    let untapped_shared = (0..blocks.len())
        .map(|index| {
            let shape = untapped[index].as_ref()?;
            Some(
                (0..blocks.len())
                    .find(|&other| shapes[other] == *shape && equal_scales(other, index))
                    .map(Holder::Block)
                    .or_else(|| {
                        (0..index)
                            .find(|&first| untapped[first].as_ref() == Some(shape) && equal_scales(first, index))
                            .map(Holder::Untapped)
                    }),
            )
        })
        .collect::<Vec<_>>();
    let exact = match scales {
        BlockOutputScales::Read(_) => true,
        BlockOutputScales::Planned(_) => (0..blocks.len()).all(|index| {
            (0..index).all(|first| shapes[first] != shapes[index] || (unit(first) && unit(index)))
                && untapped[index].as_ref().is_none_or(|shape| {
                    (0..blocks.len())
                        .filter(|&other| shapes[other] == *shape || (other < index && untapped[other].as_ref() == Some(shape)))
                        .all(|other| unit(other) && unit(index))
                })
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
    // Each picked pooled source's Block read, the picks per row of every
    // indexer over it (each storing to one picks table, so all alike) and
    // the blocks of the candidate pool its own indexer produces.
    let mut picked: Vec<(usize, PickedRead)> = Vec::new();
    for (index, slot) in blocks.iter().enumerate() {
        let crate::MixerProgramSlot::Attention(crate::AttentionBinding { picked: Some(binding), .. }) =
            slot.mixer()
        else {
            continue;
        };
        let (block, part, indexer) = match binding {
            crate::PickedBinding::Own(pooled) => (
                u32::try_from(index).map_err(|_| "block index exceeds u32")?,
                pooled.part,
                pooled.indexer,
            ),
            crate::PickedBinding::Reindex { source, part, indexer } => (source, part, indexer),
            crate::PickedBinding::Reused => continue,
        };
        let read = state
            .target_state()
            .pooled_read(block, part)
            .ok_or("a picked pooled source has no Block domain")?;
        let pool_blocks = match indexer.candidates {
            crate::CandidatesBinding::Produce(pool) => Some(pool.blocks),
            crate::CandidatesBinding::All | crate::CandidatesBinding::Restrict(_) => None,
        };
        match picked.iter().find(|(other, _)| *other == read) {
            None => picked.push((read, PickedRead { top: indexer.top, pool_blocks })),
            Some((_, lists)) if lists.top != indexer.top => {
                return Err(format!(
                    "indexers over one pooled read picking {} and {} rows",
                    lists.top, indexer.top
                ));
            }
            // The rows' own indexer, the one that produces a pool, comes
            // first.
            Some(_) if pool_blocks.is_some() => {
                return Err("a candidate pool produced by an indexer after another over its rows".into());
            }
            Some(_) => {}
        }
    }
    let mut enumeration = BlockGraphEnumeration {
        sealing: Vec::new(),
        sharing: Vec::new(),
        controls: LaunchControlLayout::of(
            &state.target_state().history_reads()?,
            &block_listed,
            &picked,
            outline.largest_rows,
            max_segments,
            max_slots,
        )?,
        listings: block_listed,
        launch_classes: std::collections::BTreeSet::new(),
        untapped_sealing: Vec::new(),
        untapped_sharing: Vec::new(),
        exact,
    };
    for (&rows, listings) in row_classes.iter().zip(&listings_at) {
        for (index, &slot) in blocks.iter().enumerate() {
            for class in block_classes(
                keeps[index],
                rows,
                max_slots,
                max_segments,
                &listings[index],
                slot_scores(slot),
            ) {
                enumeration.launch_classes.insert(class);
                for &stage in block_sealed_stages(served, index, rows) {
                    let staged = (class, Stage::Run(stage));
                    match shared[index] {
                        Some(first) => enumeration.sharing.push((index, staged, first)),
                        None => enumeration.sealing.push((index, staged)),
                    }
                    match untapped_shared[index] {
                        _ if rows > outline.untapped_rows => {}
                        None => {}
                        Some(Some(holder)) => enumeration.untapped_sharing.push((index, staged, holder)),
                        Some(None) => enumeration.untapped_sealing.push((index, staged)),
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
    enumeration.untapped_sealing.sort_unstable();
    enumeration.untapped_sharing.sort_unstable();
    Ok(enumeration)
}

/// The state each decoder block keeps, which decides its graph classes.
fn block_keeps(geometry: &Decoder) -> Result<Vec<BlockKeeps>, String> {
    geometry
        .blocks
        .iter()
        .map(|block| {
            paired_block(block)
                .map(|paired| paired.keeps())
                .map_err(|error| error.to_string())
        })
        .collect()
}
