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

    /// The group's own buffers.
    fn owned(&self) -> impl Iterator<Item = &Extents> {
        let split = self.split.iter().flat_map(|split| split.crossing.iter().chain([&split.merged.inverse]));
        let decode_crossing = self.decode_crossing.iter().flat_map(|c| c.iter());
        let operand_carry = self
            .operand_carry
            .iter()
            .flat_map(|(pair, places)| (0..2 * *places).flat_map(move |_| pair.iter()));
        self.taps
            .iter()
            .chain(&self.per_layer)
            .chain(&self.tokens)
            .chain(&self.guesses)
            .chain(split)
            .chain(decode_crossing)
            .chain(operand_carry)
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
#[derive(Clone, Copy, Debug)]
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
    /// The rows of one chain region of the launch-row buffers: the largest
    /// row class.
    region_rows: u64,
}

impl GroupRows {
    fn of(
        backend: seismic::BackendName,
        load: &crate::ModelLoadPlan,
        geometry: &magnitude_family_contracts::Decoder,
        plan: &crate::TargetProgramPlan,
        tap_width: Option<u64>,
        largest_rows: u64,
        carries: bool,
    ) -> Self {
        Self {
            taps: tap_width.map(|width| (activation(geometry), [largest_rows, width])),
            per_layer: prepared::per_layer_rows(plan, largest_rows),
            tokens: entry::token_rows_extents(load, largest_rows),
            guesses: load.fronts().then_some([largest_rows, 2]),
            operand_carry: carries.then(|| operand_carry_extents(backend, geometry.hidden)),
            region_rows: largest_rows,
        }
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
        GroupExtents::of(self.taps, self.per_layer, self.tokens, self.guesses, served, self.region_start(places))
            .with_operand_carry(self.operand_carry, places)
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
    use super::{GroupExtents, GroupRows, operand_carry_extents};
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
            region_rows: 64,
        };
        // Floor (1 place) is charged in binding_constant_bytes; group_bytes(1) is 0.
        assert_eq!(rows.extents(&[], 1).bytes().unwrap(), 2 * pair_bytes);
        assert_eq!(rows.group_bytes(&[], 1).unwrap(), 0);
        // Multi-place groups charge all places' alternating carries.
        assert_eq!(rows.group_bytes(&[], 3).unwrap(), 6 * pair_bytes);
    }
}
