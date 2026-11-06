//! Seismic-owned decoder block graphs. The engine supplies semantic model
//! topology and external state; checked entry contracts derive every stage
//! edge and intermediate allocation.

mod block;
mod bound;
mod certify;
mod classes;
mod entry;
mod heap;
mod prepared;
#[cfg(test)]
mod tests;
mod weights;

#[cfg(test)]
pub(crate) use block::checked_block_graph_resources;
pub(crate) use bound::{BankRun, BoundTargetGraphs, GroupBuffers, TokenRow, placed_banks};
pub(crate) use certify::{CheckedTargetFamily, checked_target_and_undrafted, checked_target_family};
#[cfg(test)]
pub(crate) use certify::checked_target_family_storage;
pub(crate) use classes::{LaunchControl, LaunchControlLayout};
#[cfg(test)]
pub(crate) use entry::checked_entry_graph_storage;
pub(crate) use entry::{EntryTokens, TARGET_TABLE, embedding_transform};
pub use heap::{PlannedSealedGraphs, SealedGraphsHeap};
pub(crate) use prepared::block_scopes;
pub use prepared::{PreparedTargetGraphs, SealReport};
pub(crate) use weights::{
    BankPort, CheckedLandingEntries, GatheredTable, GraphPark, GraphTimeline, LandedReadPorts,
    LandingEntries, PackedWeight, PackingBound, RowDirectory, ScaledWeight, StoredReadPorts,
    WeightOperands, WeightPart, WeightPort, WeightScale, activation, bank_port, packed_weight,
    projection_ports, scaled_weight, scaled_weight_ports, stored_table_rows, weight, weight_port,
    weight_scale_port,
};

use crate::operators::routed::split::{Extents, SplitValues};
use classes::BlockServed;
use seismic::Element;

/// The buffers a prompt group holds over its rows, a region per place
/// (`GroupBuffers`): the draft input rows of a tapped target, the per-layer
/// rows, the kept token rows, and its split layers' crossing and merged
/// positions (`SplitValues::at_rows`) in the load's one buffer of each
/// (`SplitValues::widest`), beside the merged experts and row table of the
/// group's pass, which every group shares (`pass_bytes`).
struct GroupExtents {
    taps: Option<Extents>,
    per_layer: Option<Extents>,
    tokens: Option<Extents>,
    split: Option<SplitValues<Extents>>,
}

impl GroupExtents {
    /// A group of `rows` rows, of the launch-row buffers' extents over one
    /// region and the blocks' `served` values.
    fn of(
        taps: Option<(Element, [u64; 2])>,
        per_layer: Option<[u64; 2]>,
        tokens: Option<[u64; 2]>,
        served: &[BlockServed],
        rows: u64,
    ) -> Self {
        Self {
            taps: taps.map(|(element, [_, width])| (element, vec![rows, width])),
            per_layer: per_layer.map(|[_, width]| (Element::f32(), vec![rows, width])),
            tokens: tokens.map(|[_, width]| (Element::i32(), vec![rows, width])),
            split: SplitLayers::of(served).map(|layers| layers.at_rows(rows)),
        }
    }

    /// The group's own buffers.
    fn owned(&self) -> impl Iterator<Item = &Extents> {
        let split = self.split.iter().flat_map(|split| split.crossing.iter().chain([&split.merged.inverse]));
        self.taps.iter().chain(&self.per_layer).chain(&self.tokens).chain(split)
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
