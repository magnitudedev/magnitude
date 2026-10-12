//! Which of a fixed number of slots holds each row of a table.
//!
//! A slot holds one row tile: the rows a storage lays out together (one row
//! for a host cache and for most device storage). The directory gives every
//! row its slot, and frees slots by a clock: it takes the next slot whose
//! tile was not used since it last passed, and passes over a used one once.
//!
//! It serves the row cache and the value slots of a streamed host table
//! (`streamed_table`) and the slots of a device table (`device_table`).

pub(crate) struct SlotDirectory {
    /// Rows of one tile.
    tile_rows: u64,
    /// Per table row, its slot, or -1: a device table's directory as the
    /// device holds it.
    slot: Vec<i32>,
    /// Per tile of slots, the first row of the table tile it holds.
    tile: Vec<Option<u64>>,
    /// Per tile of slots, whether its tile was used since the clock last
    /// passed it.
    used: Vec<bool>,
    /// The tile of slots the clock looks at next.
    hand: usize,
}

impl SlotDirectory {
    /// A directory of `rows` table rows over `slots` slots, both whole tiles
    /// of `tile_rows` rows.
    pub fn new(rows: u64, slots: u64, tile_rows: u64) -> Self {
        let slot_tiles = (slots / tile_rows) as usize;
        Self {
            tile_rows,
            slot: vec![-1; rows as usize],
            tile: vec![None; slot_tiles],
            used: vec![false; slot_tiles],
            hand: 0,
        }
    }

    pub fn tile_rows(&self) -> u64 {
        self.tile_rows
    }

    /// Tiles of slots.
    pub fn slot_tiles(&self) -> usize {
        self.tile.len()
    }

    /// Whether `row` is a row of the table.
    pub fn has_row(&self, row: u64) -> bool {
        row < self.slot.len() as u64
    }

    /// The slot holding `row`; `None` of a row not held or outside the table.
    pub fn slot(&self, row: u64) -> Option<u64> {
        let slot = self.slot.get(usize::try_from(row).ok()?)?;
        u64::try_from(*slot).ok()
    }

    /// The slot holding `row`, whose tile is used from here on.
    pub fn use_slot(&mut self, row: u64) -> Option<u64> {
        let slot = self.slot(row)?;
        self.used[(slot / self.tile_rows) as usize] = true;
        Some(slot)
    }

    /// The first row of the table tile the tile of slots `slot_tile` holds.
    pub fn held(&self, slot_tile: usize) -> Option<u64> {
        self.tile[slot_tile]
    }

    /// The directory's entries of table rows `low..high`, little-endian I32.
    pub fn entry_bytes(&self, low: u64, high: u64) -> Vec<u8> {
        self.slot[low as usize..high as usize]
            .iter()
            .flat_map(|slot| slot.to_le_bytes())
            .collect()
    }

    /// Hold no tile in the tile of slots `slot_tile`: the first row of the
    /// one it held.
    pub fn release(&mut self, slot_tile: usize) -> Option<u64> {
        self.used[slot_tile] = false;
        let first = self.tile[slot_tile].take()?;
        self.slot[first as usize..(first + self.tile_rows) as usize].fill(-1);
        Some(first)
    }

    /// Hold the table tile of first row `first` in the tile of slots
    /// `slot_tile`, which holds none. `used`: whether the clock passes it
    /// once before it takes it.
    pub fn hold(&mut self, first: u64, slot_tile: usize, used: bool) {
        let first_slot = slot_tile as u64 * self.tile_rows;
        for row in 0..self.tile_rows {
            self.slot[(first + row) as usize] = (first_slot + row) as i32;
        }
        self.tile[slot_tile] = Some(first);
        self.used[slot_tile] = used;
    }

    /// A tile of slots to hold another tile in: the clock's next one that is
    /// neither used since it last passed nor `kept`, released, and the first
    /// row of the tile it held. The clock passes every tile of slots at most
    /// twice, the first time taking from each its use; `None` when it finds
    /// none, as of a directory without slots.
    pub fn free(&mut self, kept: impl Fn(usize) -> bool) -> Option<(usize, Option<u64>)> {
        for _ in 0..2 * self.tile.len() {
            let slot_tile = self.hand;
            self.hand = (self.hand + 1) % self.tile.len();
            if kept(slot_tile) || std::mem::take(&mut self.used[slot_tile]) {
                continue;
            }
            return Some((slot_tile, self.release(slot_tile)));
        }
        None
    }

    /// The first of `count` adjacent tiles of slots, each released: the
    /// clock's next ones. `None` of a directory with fewer.
    pub fn free_adjacent(&mut self, count: usize) -> Option<usize> {
        if count > self.tile.len() {
            return None;
        }
        if self.hand + count > self.tile.len() {
            self.hand = 0;
        }
        let first = self.hand;
        for slot_tile in first..first + count {
            self.release(slot_tile);
        }
        self.hand = (first + count) % self.tile.len();
        Some(first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_frees_a_slot_whose_row_was_not_used_since_it_last_passed() {
        let mut directory = SlotDirectory::new(8, 3, 1);
        assert_eq!(directory.slot(5), None);
        assert_eq!(directory.slot(8), None);
        for (row, slot) in [(5, 0), (6, 1), (7, 2)] {
            assert_eq!(directory.free(|_| false), Some((slot, None)));
            directory.hold(row, slot, true);
        }
        assert_eq!(directory.entry_bytes(5, 8), [0i32, 1, 2].map(i32::to_le_bytes).concat());
        // Every row was used: the clock passes them all once, then takes
        // the first. A row used again in between is passed again.
        assert_eq!(directory.free(|_| false), Some((0, Some(5))));
        assert_eq!(directory.slot(5), None);
        directory.hold(1, 0, true);
        assert_eq!(directory.use_slot(6), Some(1));
        assert_eq!(directory.free(|_| false), Some((2, Some(7))));
        directory.hold(2, 2, false);
        // A kept slot is never taken, and nothing is when all are kept.
        assert_eq!(directory.free(|slot| slot == 0), Some((1, Some(6))));
        assert_eq!(directory.free(|_| true), None);
        assert_eq!(SlotDirectory::new(8, 0, 1).free(|_| false), None);
    }

    #[test]
    fn tiles_of_several_rows_hold_and_release_together() {
        let mut directory = SlotDirectory::new(8, 4, 2);
        assert_eq!(directory.slot_tiles(), 2);
        directory.hold(6, 1, false);
        assert_eq!(directory.slot(6), Some(2));
        assert_eq!(directory.slot(7), Some(3));
        assert_eq!(directory.held(1), Some(6));
        assert_eq!(directory.release(1), Some(6));
        assert_eq!(directory.slot(7), None);
        assert_eq!(directory.release(1), None);
    }

    #[test]
    fn adjacent_slots_are_the_clocks_next_ones() {
        let mut directory = SlotDirectory::new(8, 4, 1);
        assert_eq!(directory.free_adjacent(5), None);
        assert_eq!(directory.free_adjacent(3), Some(0));
        directory.hold(4, 1, true);
        // Two more do not fit before the end: the clock starts over, and
        // the row there leaves its slot.
        assert_eq!(directory.free_adjacent(2), Some(0));
        assert_eq!(directory.free_adjacent(2), Some(2));
        assert_eq!(directory.slot(4), None);
    }
}
