//! Borrowed stage geometry: local ordering never renumbers persistent state.
use crate::LayerRef;
use magnitude_family_contracts::{Block, Decoder};
use std::ops::Range;

/// Semantic block identity within the original decoder (also its weight/KV key).
/// An index is not a cross-model identity; the borrowed view retains its decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalLayerId(u32);

impl GlobalLayerId {
    pub fn index(self) -> u32 {
        self.0
    }

    pub fn kv_layer(self) -> LayerRef {
        LayerRef::Target(self.0)
    }
}

/// Position in a stage's ordered graph/component lists, not a KV key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StageLayerOrdinal(usize);

impl StageLayerOrdinal {
    pub fn index(self) -> usize {
        self.0
    }
}

/// A nonempty contiguous projection of one immutable original decoder.
/// This borrows the original blocks, never a rebuilt or renumbered decoder.
#[derive(Clone, Debug)]
pub struct StageModelView<'a> {
    decoder: &'a Decoder,
    range: Range<usize>,
}

impl<'a> StageModelView<'a> {
    pub fn new(decoder: &'a Decoder, range: Range<usize>) -> Result<Self, String> {
        if range.start >= range.end || range.end > decoder.blocks.len() {
            return Err("stage must be a nonempty range of original decoder blocks".into());
        }
        u32::try_from(range.end - 1).map_err(|_| "global layer identity exceeds u32")?;
        Ok(Self { decoder, range })
    }

    pub fn decoder(&self) -> &'a Decoder {
        self.decoder
    }

    pub fn global_range(&self) -> Range<usize> {
        self.range.clone()
    }

    pub fn blocks(&self) -> &'a [Block] {
        &self.decoder.blocks[self.range.clone()]
    }

    pub fn layers(
        &self,
    ) -> impl Iterator<Item = (StageLayerOrdinal, GlobalLayerId, &'a Block)> + '_ {
        self.blocks().iter().enumerate().map(|(local, block)| {
            // Construction established the u32 bound for every selected layer.
            (
                StageLayerOrdinal(local),
                GlobalLayerId((self.range.start + local) as u32),
                block,
            )
        })
    }

    pub fn contains(&self, layer: u32) -> bool {
        self.range.contains(&(layer as usize))
    }
}
