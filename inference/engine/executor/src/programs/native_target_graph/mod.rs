//! Seismic-owned decoder block graphs. The engine supplies semantic model
//! topology and external state; checked entry contracts derive every stage
//! edge and intermediate allocation.

mod block;
mod bound;
mod certify;
mod classes;
mod entry;
mod heap;
#[cfg(test)]
mod outline_parity;
mod prepared;
#[cfg(test)]
mod tests;
mod weights;

#[cfg(test)]
pub(crate) use block::checked_block_graph_resources;
pub(crate) use block::CarryPorts;
pub(crate) use bound::{BankRun, BoundTargetGraphs, GroupBuffers, TokenRow, placed_banks};
pub(crate) use certify::{CheckedTargetFamily, checked_target_and_undrafted, checked_target_family};
#[cfg(test)]
pub(crate) use certify::checked_target_family_storage;
#[cfg(test)]
pub(crate) use outline_parity::{recorded_traces, target_outline_differences};
pub(crate) use classes::{
    LaunchControl, LaunchControlLayout, Passes, complete, packed_selection_scores, selection_scores,
};
#[cfg(test)]
pub(crate) use entry::checked_entry_graph_storage;
pub(crate) use entry::{EntryTokens, TARGET_TABLE, embedding_transform};
pub use heap::{PlannedSealedGraphs, SealedGraphsHeap};
pub(crate) use prepared::block_scopes;
pub use prepared::{PreparedTargetGraphs, SealReport};
pub(crate) use weights::{
    BankPort, CheckedLandingEntries, GatheredTable, GraphPark, GraphTimeline, GraphWeights, LandedReadPorts, LandingEntries,
    PackedWeight, PackingBound, PlannedWeights, RowDirectory, ScaledWeight, StoredReadPorts, WeightOperands, WeightPart, WeightPort, WeightScale, WeightView,
    activation, bank_port, packed_weight, relative_scope_of,
    RegisteredPorts, scaled_weight, scaled_weight_ports, stored_table_rows, weight, weight_port,
    weight_scale_port,
};

use crate::{
    operators::routed::split::{Extents, SplitValues},
};
use classes::BlockServed;
use seismic::Element;

/// Assignment of each picked history read's VRAM picks and candidate pool
/// tables into the minimal set of non-overlapping shared buffer slots.
///
/// Within a single chain place, blocks execute in strict index order
/// `0..blocks.len()`. Two reads whose live block intervals do not overlap
/// share the same VRAM buffer slot (sized to the widest read assigned to that
/// slot), keeping VRAM usage O(concurrently live reads) — O(1) when each
/// sparse-attention layer owns and consumes its own picks/pool in one block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct PickedBuffers {
    /// Per shared picks slot, the maximum `top` across reads assigned to it.
    picks_slots: Vec<u64>,
    /// Per history read, `(slot, top)` into `picks_slots`.
    picks_by_read: Vec<Option<(usize, u64)>>,
    /// Per shared candidate pool slot, the maximum `pool_blocks` across reads
    /// assigned to it.
    pool_slots: Vec<u64>,
    /// Per history read, `(slot, pool_blocks)` into `pool_slots`.
    pool_by_read: Vec<Option<(usize, u64)>>,
}

impl PickedBuffers {
    /// Derive the minimal shared slot assignment from the blocks' live
    /// intervals over each picked history read.
    pub(super) fn of(
        geometry: &magnitude_family_contracts::Decoder,
        state: &crate::StateResourcePlan,
        plan: &crate::TargetProgramPlan,
        picked: &[Option<classes::PickedRead>],
    ) -> Result<Self, String> {
        let read_count = picked.len();
        if picked.iter().all(Option::is_none) {
            return Ok(Self::default());
        }
        let mut picks_span: Vec<Option<(usize, usize)>> = vec![None; read_count];
        let mut pool_span: Vec<Option<(usize, usize)>> = vec![None; read_count];
        let record = |spans: &mut [Option<(usize, usize)>], read: usize, block: usize| {
            match &mut spans[read] {
                Some((_, last)) => *last = (*last).max(block),
                slot @ None => *slot = Some((block, block)),
            }
        };
        for (index, slot) in plan.blocks().iter().enumerate() {
            let crate::MixerProgramSlot::Attention(crate::AttentionBinding {
                picked: Some(binding),
                ..
            }) = slot.mixer()
            else {
                continue;
            };
            let block = u32::try_from(index).map_err(|_| "block index exceeds u32")?;
            let (source_block, part, uses_pool) = match binding {
                crate::PickedBinding::Own(pooled) => (
                    block,
                    pooled.part,
                    matches!(pooled.indexer.candidates, crate::CandidatesBinding::Produce(_)),
                ),
                crate::PickedBinding::Reindex { source, part, indexer } => (
                    source,
                    part,
                    matches!(indexer.candidates, crate::CandidatesBinding::Restrict(_)),
                ),
                crate::PickedBinding::Reused => {
                    let attention = geometry
                        .attention(magnitude_family_contracts::SublayerIndex { block, sublayer: 0 })
                        .ok_or("a reused picked block has no attention sublayer")?;
                    let pooled = crate::operators::attention::picked_source(attention)
                        .map_err(|error| error.to_string())?
                        .ok_or("a reused picked block has no picked source")?
                        .pooled(block);
                    (pooled.block, pooled.part, false)
                }
            };
            let read = state
                .target_state()
                .pooled_read(source_block, part)
                .ok_or("a picked pooled source has no Block domain")?;
            if read >= read_count {
                return Err(format!("picked history read {read} out of bounds ({read_count})"));
            }
            record(&mut picks_span, read, index);
            if uses_pool {
                record(&mut pool_span, read, index);
            }
        }
        let mut picks_intervals = Vec::new();
        let mut pool_intervals = Vec::new();
        for (read, entry) in picked.iter().enumerate() {
            let Some(lists) = entry else {
                continue;
            };
            let (first, last) = picks_span[read]
                .ok_or_else(|| format!("picked history read {read} has no using block"))?;
            picks_intervals.push((read, lists.top, first, last));
            if let Some(pool_blocks) = lists.pool_blocks {
                let (first, last) = pool_span[read]
                    .ok_or_else(|| format!("picked history read {read} pool has no using block"))?;
                pool_intervals.push((read, pool_blocks, first, last));
            }
        }
        Ok(Self::from_intervals(read_count, picks_intervals, pool_intervals))
    }

    /// Optimal greedy interval-graph coloring for picks and candidate pool
    /// tables. Each entry is `(read, width, first_block, last_block)`.
    fn from_intervals(
        read_count: usize,
        picks_intervals: Vec<(usize, u64, usize, usize)>,
        pool_intervals: Vec<(usize, u64, usize, usize)>,
    ) -> Self {
        let (picks_slots, picks_by_read) = color_intervals(read_count, picks_intervals);
        let (pool_slots, pool_by_read) = color_intervals(read_count, pool_intervals);
        Self {
            picks_slots,
            picks_by_read,
            pool_slots,
            pool_by_read,
        }
    }
}

fn color_intervals(
    read_count: usize,
    mut intervals: Vec<(usize, u64, usize, usize)>,
) -> (Vec<u64>, Vec<Option<(usize, u64)>>) {
    intervals.sort_unstable_by_key(|&(read, _, first, last)| (first, last, read));
    let mut slot_end: Vec<usize> = Vec::new();
    let mut slot_width: Vec<u64> = Vec::new();
    let mut by_read: Vec<Option<(usize, u64)>> = vec![None; read_count];
    for (read, width, first, last) in intervals {
        let slot = slot_end
            .iter()
            .enumerate()
            .filter(|&(_, &end)| end < first)
            .min_by_key(|&(s, _)| (slot_width[s] < width, slot_width[s].abs_diff(width), s))
            .map(|(s, _)| s);
        let slot = match slot {
            Some(s) => {
                slot_end[s] = last;
                slot_width[s] = slot_width[s].max(width);
                s
            }
            None => {
                let s = slot_end.len();
                slot_end.push(last);
                slot_width.push(width);
                s
            }
        };
        by_read[read] = Some((slot, width));
    }
    (slot_width, by_read)
}

/// The buffers a prompt group holds over its rows, a region per place
/// (`GroupBuffers`): the draft input rows of a tapped target, the per-layer
/// rows, the kept token rows and guesses, its split layers' crossing and merged
/// positions (`SplitValues::at_rows`) in the load's one buffer of each
/// (`SplitValues::widest`), and its per-place alternating operand carries
/// (`GroupBuffers::operands`), beside the merged experts and row table of the
/// group's pass, which every group shares (`pass_bytes`).
struct GroupExtents {
    taps: Option<Extents>,
    per_layer: Option<Extents>,
    tokens: Option<Extents>,
    guesses: Option<Extents>,
    split: Option<SplitValues<Extents>>,
    decode_crossing: Option<crate::operators::routed::split::DecodeCrossingBuffers<Extents>>,
    /// The `[rows, quantized]` extents of one operand carry and the group's
    /// places: each place holds two alternating carries.
    operand_carry: Option<([Extents; 2], usize)>,
    /// Shared VRAM picks buffers (`[rows, max_top]`, I32), one per
    /// non-overlapping live-interval slot, plus per-read `(slot, top)` mapping.
    picks: Vec<Extents>,
    picks_by_read: Vec<Option<(usize, u64)>>,
    /// Shared VRAM candidate pool buffers
    /// (`[rows, max_pool_blocks + max_segments, 2]`, I32), one per
    /// non-overlapping live-interval slot, plus per-read `(slot, pool_blocks)`
    /// mapping.
    pool: Vec<Extents>,
    pool_by_read: Vec<Option<(usize, u64)>>,
}

impl GroupExtents {
    /// A group of `rows` rows, of the launch-row buffers' extents over one
    /// region and the blocks' `served` values.
    fn of(
        taps: Option<(Element, [u64; 2])>,
        per_layer: Option<[u64; 2]>,
        tokens: Option<[u64; 2]>,
        guesses: Option<[u64; 2]>,
        served: &[BlockServed],
        rows: u64,
    ) -> Self {
        let decode_crossing = {
            let mut layers = served.iter().filter_map(|block| block.decode_crossing.clone());
            layers.next().map(|first| {
                let rest: Vec<_> = layers.collect();
                crate::operators::routed::split::DecodeCrossingBuffers::widest(&first, &rest)
            })
        };
        Self {
            taps: taps.map(|(element, [_, width])| (element, vec![rows, width])),
            per_layer: per_layer.map(|[_, width]| (Element::f32(), vec![rows, width])),
            tokens: tokens.map(|[_, width]| (Element::i32(), vec![rows, width])),
            guesses: guesses.map(|[_, width]| (Element::i32(), vec![rows, width])),
            split: SplitLayers::of(served).map(|layers| layers.at_rows(rows)),
            decode_crossing,
            operand_carry: None,
            picks: Vec::new(),
            picks_by_read: Vec::new(),
            pool: Vec::new(),
            pool_by_read: Vec::new(),
        }
    }

    /// Attach the per-carry `[rows, quantized]` extents for a group of `places`
    /// places, each holding two alternating carries (`GroupBuffers::operands`).
    pub(super) fn with_operand_carry(mut self, carry: Option<[(Element, [u64; 2]); 2]>, places: usize) -> Self {
        self.operand_carry = carry.map(|[(rows_el, rows_ext), (q_el, q_ext)]| {
            ([(rows_el, rows_ext.to_vec()), (q_el, q_ext.to_vec())], places)
        });
        self
    }

    /// Attach the shared VRAM picks and candidate pool extents over `rows`
    /// rows for the load's picked pooled sources.
    pub(super) fn with_picked(
        mut self,
        picked: &PickedBuffers,
        max_segments: u64,
        rows: u64,
    ) -> Self {
        self.picks = picked
            .picks_slots
            .iter()
            .map(|&max_top| (Element::i32(), vec![rows, max_top]))
            .collect();
        self.picks_by_read = picked.picks_by_read.clone();
        self.pool = picked
            .pool_slots
            .iter()
            .map(|&max_blocks| (Element::i32(), vec![rows, max_blocks + max_segments, 2]))
            .collect();
        self.pool_by_read = picked.pool_by_read.clone();
        self
    }

    /// The group's own buffers.
    fn owned(&self) -> impl Iterator<Item = &Extents> {
        let split = self.split.iter().flat_map(|split| split.crossing.iter().chain([&split.merged.inverse]));
        let decode_crossing = self.decode_crossing.iter().flat_map(|c| c.iter());
        let operand_carry = self
            .operand_carry
            .iter()
            .flat_map(|(pair, places)| (0..2 * *places).flat_map(move |_| pair.iter()));
        let picks = self.picks.iter();
        let pool = self.pool.iter();
        self.taps
            .iter()
            .chain(&self.per_layer)
            .chain(&self.tokens)
            .chain(&self.guesses)
            .chain(split)
            .chain(decode_crossing)
            .chain(operand_carry)
            .chain(picks)
            .chain(pool)
    }

    /// Bytes of the group's own buffers.
    fn bytes(&self) -> Result<u64, String> {
        extents_bytes(self.owned())
    }

    /// Bytes of the pass's merged experts and row table.
    fn pass_bytes(&self) -> Result<u64, String> {
        extents_bytes(
            self.split
                .iter()
                .flat_map(|split| [&split.merged.experts, &split.merged.table]),
        )
    }
}

/// The `[rows, quantized]` element and `[rows, cols]` extents of one operand
/// carry (`seismic::sizing::operand_carry_ports`), the shape every carry of a
/// load takes (`block::CARRIED_ROWS` by `hidden`).
fn operand_carry_extents(backend: seismic::BackendName, hidden: u64) -> [(Element, [u64; 2]); 2] {
    let [(rows_repr, rows_extents), (quantized_repr, quantized_extents)] =
        seismic::sizing::operand_carry_ports(backend, block::CARRIED_ROWS, hidden);
    [
        (Element::from_id(rows_repr), rows_extents),
        (Element::from_id(quantized_repr), quantized_extents),
    ]
}

/// What a load's prompt-group buffers span (`GroupExtents`): a region of
/// the largest row class per place, of the draft input rows of a tapped
/// target, the per-layer rows, the kept token rows and guesses, and the
/// per-place alternating operand carries. The one derivation the certificate
/// (what a plan charges), the prepared family (what a load holds), and
/// `GroupBuffers::new` (what binding allocates) read.
#[derive(Clone, Debug)]
struct GroupRows {
    /// The element and `[rows, taps · hidden]` extents of one region of the
    /// draft input rows, when a separate draft taps the target.
    taps: Option<(Element, [u64; 2])>,
    /// The `[rows, L·P]` F32 extents of one region of the per-layer rows
    /// every per-layer input sublayer reads.
    per_layer: Option<[u64; 2]>,
    /// The `[rows, 2]` I32 extents of one region of the token rows a model
    /// that reads them past its entry keeps
    /// (`ModelLoadPlan::keeps_token_rows`).
    tokens: Option<[u64; 2]>,
    /// The `[rows, 2]` I32 extents of one region of the host's guesses a
    /// load that fronts keeps (`PreparedTargetEntryGraph::kept_guesses`).
    guesses: Option<[u64; 2]>,
    /// The `[rows, quantized]` element and extents of one operand carry, when
    /// any block of the load carries (`FamilyOutline::carries`).
    operand_carry: Option<[(Element, [u64; 2]); 2]>,
    /// Shared slot assignment for VRAM picks and candidate pool tables
    /// (`GroupBuffers::picks_slots`, `GroupBuffers::pool_slots`).
    picked: PickedBuffers,
    /// The load's maximum segment class, bounding each candidate pool's span
    /// capacity (`blocks + max_segments`).
    max_segments: u64,
    /// The rows of one chain region of the launch-row buffers: the largest
    /// row class.
    region_rows: u64,
}

impl GroupRows {
    #[allow(clippy::too_many_arguments)]
    fn of(
        backend: seismic::BackendName,
        load: &crate::ModelLoadPlan,
        geometry: &magnitude_family_contracts::Decoder,
        state: &crate::StateResourcePlan,
        plan: &crate::TargetProgramPlan,
        tap_width: Option<u64>,
        largest_rows: u64,
        carries: bool,
        picked: &[Option<classes::PickedRead>],
        max_segments: u64,
    ) -> Result<Self, String> {
        Ok(Self {
            taps: tap_width.map(|width| (activation(geometry), [largest_rows, width])),
            per_layer: prepared::per_layer_rows(plan, largest_rows),
            tokens: entry::token_rows_extents(load, largest_rows),
            guesses: load.fronts().then_some([largest_rows, 2]),
            operand_carry: carries.then(|| operand_carry_extents(backend, geometry.hidden)),
            picked: PickedBuffers::of(geometry, state, plan, picked)?,
            max_segments,
            region_rows: largest_rows,
        })
    }

    /// The first row of chain region `region` of the launch-row buffers
    /// (`chain_rows`): the region the `region`th launch of a chain writes
    /// and reads.
    fn region_start(&self, region: usize) -> u64 {
        self.region_rows * region as u64
    }

    /// The buffers a prompt group of `places` places holds over the blocks
    /// `served`.
    fn extents(&self, served: &[BlockServed], places: usize) -> GroupExtents {
        let rows = self.region_start(places);
        GroupExtents::of(self.taps, self.per_layer, self.tokens, self.guesses, served, rows)
            .with_operand_carry(self.operand_carry, places)
            .with_picked(&self.picked, self.max_segments, rows)
    }

    /// The bytes a group of `places` places' own buffers hold beside the
    /// floor's (`BoundTargetGraphs::hold_group`): none of one place, which
    /// the floor serves.
    fn group_bytes(&self, served: &[BlockServed], places: usize) -> Result<u64, String> {
        match places {
            1 => Ok(0),
            _ => self.extents(served, places).bytes(),
        }
    }
}

/// The split values of a load's routed layers (`BlockServed::split`), of a
/// load whose blocks split: never empty.
#[derive(Clone)]
pub(super) struct SplitLayers {
    first: SplitValues<Extents>,
    rest: Vec<SplitValues<Extents>>,
}

impl SplitLayers {
    /// The split layers of the blocks `served`; none where no block splits.
    pub(super) fn of(served: &[BlockServed]) -> Option<Self> {
        let mut layers = served.iter().filter_map(|block| block.split.clone());
        let first = layers.next()?;
        Some(Self { first, rest: layers.collect() })
    }

    /// The load's buffers of their values over a group of `rows` rows
    /// (`SplitValues::widest`).
    pub(super) fn at_rows(&self, rows: u64) -> SplitValues<Extents> {
        let rest = self.rest.iter().map(|layer| layer.at_rows(rows)).collect::<Vec<_>>();
        SplitValues::widest(&self.first.at_rows(rows), &rest)
    }
}

fn extents_bytes<'a>(extents: impl IntoIterator<Item = &'a Extents>) -> Result<u64, String> {
    extents.into_iter().try_fold(0u64, |total, (element, extents)| {
        let bytes = element.canonical_byte_len(extents).map_err(|error| error.to_string())?;
        total.checked_add(bytes).ok_or_else(|| "group buffer bytes overflow".to_owned())
    })
}

#[cfg(test)]
mod group_extents_tests {
    use super::{GroupExtents, GroupRows, PickedBuffers, operand_carry_extents};
    use seismic::BackendName;

    #[test]
    fn group_extents_and_rows_account_for_operand_carry() {
        let carry = operand_carry_extents(BackendName::Cpu, 256);
        let pair_bytes = carry[0].0.canonical_byte_len(&carry[0].1).unwrap()
            + carry[1].0.canonical_byte_len(&carry[1].1).unwrap();
        assert!(pair_bytes > 0);

        let without = GroupExtents::of(None, None, None, None, &[], 64);
        assert_eq!(without.bytes().unwrap(), 0);
        assert_eq!(without.pass_bytes().unwrap(), 0);

        let one_place = GroupExtents::of(None, None, None, None, &[], 64)
            .with_operand_carry(Some(carry), 1);
        assert_eq!(one_place.bytes().unwrap(), 2 * pair_bytes);
        assert_eq!(one_place.pass_bytes().unwrap(), 0);

        let three_places = GroupExtents::of(None, None, None, None, &[], 192)
            .with_operand_carry(Some(carry), 3);
        assert_eq!(three_places.bytes().unwrap(), 6 * pair_bytes);

        let rows = GroupRows {
            taps: None,
            per_layer: None,
            tokens: None,
            guesses: None,
            operand_carry: Some(carry),
            picked: PickedBuffers::default(),
            max_segments: 0,
            region_rows: 64,
        };
        // Floor (1 place) is charged in binding_constant_bytes; group_bytes(1) is 0.
        assert_eq!(rows.extents(&[], 1).bytes().unwrap(), 2 * pair_bytes);
        assert_eq!(rows.group_bytes(&[], 1).unwrap(), 0);
        // Multi-place groups charge all places' alternating carries.
        assert_eq!(rows.group_bytes(&[], 3).unwrap(), 6 * pair_bytes);
    }

    #[test]
    fn shared_picks_and_pool_vram_is_o1_across_non_overlapping_layers() {
        // 1 sparse-attention layer vs 60 sparse-attention layers with identical
        // per-layer (top = 2048, pool_blocks = 128) and disjoint single-block
        // live intervals [b, b] must allocate the exact same O(1) VRAM bytes.
        let make_rows = |layers: usize| {
            let picks_intervals: Vec<_> = (0..layers).map(|b| (b, 2048, b, b)).collect();
            let pool_intervals: Vec<_> = (0..layers).map(|b| (b, 128, b, b)).collect();
            let picked = PickedBuffers::from_intervals(layers, picks_intervals, pool_intervals);
            GroupRows {
                taps: None,
                per_layer: None,
                tokens: None,
                guesses: None,
                operand_carry: None,
                picked,
                max_segments: 16,
                region_rows: 1024,
            }
        };

        let one = make_rows(1);
        let sixty = make_rows(60);

        // Single shared picks [1024, 2048] i32 + single shared pool [1024, 144, 2] i32.
        let expected_one_place = (1024 * 2048 + 1024 * (128 + 16) * 2) * 4;
        assert_eq!(one.picked.picks_slots, vec![2048]);
        assert_eq!(one.picked.pool_slots, vec![128]);
        assert_eq!(sixty.picked.picks_slots, vec![2048]);
        assert_eq!(sixty.picked.pool_slots, vec![128]);
        assert!(sixty.picked.picks_by_read.iter().all(|&entry| entry == Some((0, 2048))));
        assert!(sixty.picked.pool_by_read.iter().all(|&entry| entry == Some((0, 128))));

        assert_eq!(one.extents(&[], 1).bytes().unwrap(), expected_one_place);
        assert_eq!(sixty.extents(&[], 1).bytes().unwrap(), expected_one_place);
        assert_eq!(sixty.group_bytes(&[], 2).unwrap(), 2 * expected_one_place);
    }

    #[test]
    fn shared_picks_and_pool_separate_overlapping_intervals_and_preserve_widths() {
        // Read 0 is live across blocks [0, 2] (e.g. reused at block 2),
        // Read 1 is live at block [1, 1] (overlaps Read 0!),
        // Read 2 is live at block [3, 3] (reuses a finished slot, wider top=5).
        let picked = PickedBuffers::from_intervals(
            4,
            vec![(0, 2, 0, 2), (1, 3, 1, 1), (2, 5, 3, 3)],
            vec![(0, 8, 0, 2), (1, 4, 1, 1)],
        );
        // Reads 0 and 1 overlap -> 2 distinct picks slots and 2 distinct pool slots.
        assert_eq!(picked.picks_slots.len(), 2);
        assert_eq!(picked.pool_slots.len(), 2);
        let (slot0, top0) = picked.picks_by_read[0].unwrap();
        let (slot1, top1) = picked.picks_by_read[1].unwrap();
        let (slot2, top2) = picked.picks_by_read[2].unwrap();
        assert_ne!(slot0, slot1);
        assert_eq!((top0, top1, top2), (2, 3, 5));
        assert!(picked.picks_slots[slot0] >= top0);
        assert!(picked.picks_slots[slot1] >= top1);
        assert!(picked.picks_slots[slot2] >= top2);
        assert_eq!(picked.picks_by_read[3], None);

        let (pslot0, pb0) = picked.pool_by_read[0].unwrap();
        let (pslot1, pb1) = picked.pool_by_read[1].unwrap();
        assert_ne!(pslot0, pslot1);
        assert_eq!((pb0, pb1), (8, 4));
        assert_eq!(picked.pool_by_read[2], None);
        assert_eq!(picked.pool_by_read[3], None);

        let rows = GroupRows {
            taps: None,
            per_layer: None,
            tokens: None,
            guesses: None,
            operand_carry: None,
            picked,
            max_segments: 4,
            region_rows: 16,
        };
        let picks_total: u64 = rows.picked.picks_slots.iter().map(|&top| 16 * top * 4).sum();
        let pool_total: u64 = rows.picked.pool_slots.iter().map(|&b| 16 * (b + 4) * 2 * 4).sum();
        assert_eq!(rows.extents(&[], 1).bytes().unwrap(), picks_total + pool_total);
    }
}
