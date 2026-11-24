//! Which of a fixed number of slots holds each row of a table.
//!
//! A slot holds one row tile: the rows a storage lays out together (one row
//! for a host cache and for most device storage). The directory gives every
//! held row its slot, and frees slots by a clock: it takes the next slot
//! whose tile was not used since it last passed, and passes over a used one
//! once.
//!
//! The directory is also the device's: a reader finds a row's slot in it
//! (`gathered_row`, target.seismic) in one of two forms (`DirectoryForm`),
//! and the host writes the entries that changed (`take_changes`).
//!
//! It serves the row cache and the value slots of a streamed host table
//! (`streamed_table`) and the slots of a device table (`device_table`).

use crate::planning::DirectoryForm;

/// A hashed directory's empty bucket, which ends a probe.
const EMPTY: i32 = -1;
/// A hashed directory's bucket whose row left, which a probe passes.
const TOMBSTONE: i32 = -2;
/// Sentinel in `SlotDirectory::tile` indicating that a slot tile holds no table tile.
/// Valid table row indices are strictly less than `u32::MAX` (in fact `<= i32::MAX`).
const NO_TILE: u32 = u32::MAX;

/// Compact fixed-length bitset backed by `Vec<u64>` (1 bit per slot instead of 1 byte).
pub(crate) struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            words: vec![0u64; len.div_ceil(64)],
            len,
        }
    }

    #[inline]
    pub(crate) fn get(&self, index: usize) -> bool {
        assert!(index < self.len, "bitset index {index} out of bounds for len {}", self.len);
        (self.words[index >> 6] & (1u64 << (index & 63))) != 0
    }

    #[inline]
    pub(crate) fn set(&mut self, index: usize) {
        assert!(index < self.len, "bitset index {index} out of bounds for len {}", self.len);
        self.words[index >> 6] |= 1u64 << (index & 63);
    }

    #[inline]
    pub(crate) fn clear(&mut self, index: usize) {
        assert!(index < self.len, "bitset index {index} out of bounds for len {}", self.len);
        self.words[index >> 6] &= !(1u64 << (index & 63));
    }

    #[inline]
    pub(crate) fn set_to(&mut self, index: usize, value: bool) {
        if value {
            self.set(index);
        } else {
            self.clear(index);
        }
    }

    #[inline]
    pub(crate) fn or_assign(&mut self, index: usize, value: bool) {
        if value {
            self.set(index);
        }
    }

    #[inline]
    pub(crate) fn take(&mut self, index: usize) -> bool {
        assert!(index < self.len, "bitset index {index} out of bounds for len {}", self.len);
        let word = &mut self.words[index >> 6];
        let mask = 1u64 << (index & 63);
        let was = (*word & mask) != 0;
        *word &= !mask;
        was
    }
}

/// The bucket a hashed directory of `buckets` (a power of two) probes first
/// for `row`: murmur3's 32-bit finalizer, as `gathered_bucket`
/// (target.seismic) computes it.
pub(crate) fn bucket(row: i32, buckets: usize) -> usize {
    hash(row) as usize & (buckets - 1)
}

/// Murmur3's 32-bit finalizer of `row`.
pub(crate) fn hash(row: i32) -> u32 {
    let mut h = row as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// Every held row's slot, as the device reads it.
enum Entries {
    /// Per table row, its slot or -1. `changed`: the rows from the first to
    /// the last whose entry changed since the changes were last taken.
    Dense {
        slot: Vec<i32>,
        changed: Option<(u64, u64)>,
    },
    /// Linearly probed (row, slot) buckets, a power of two at least twice
    /// the slots. `changed`: the buckets written since the changes were
    /// last taken.
    Hashed {
        buckets: Vec<[i32; 2]>,
        held: usize,
        tombstones: usize,
        changed: Vec<usize>,
    },
}

impl Entries {
    fn slot(&self, row: u64) -> Option<u64> {
        let slot = match self {
            Self::Dense { slot, .. } => *slot.get(usize::try_from(row).ok()?)?,
            Self::Hashed { buckets, .. } => buckets[Self::find(buckets, row)?][1],
        };
        u64::try_from(slot).ok()
    }

    /// The bucket holding `row`.
    fn find(buckets: &[[i32; 2]], row: u64) -> Option<usize> {
        let key = i32::try_from(row).ok()?;
        let mut probe = bucket(key, buckets.len());
        for _ in 0..buckets.len() {
            match buckets[probe][0] {
                found if found == key => return Some(probe),
                EMPTY => return None,
                _ => probe = (probe + 1) & (buckets.len() - 1),
            }
        }
        None
    }

    fn set(&mut self, row: u64, held: i32) {
        match self {
            Self::Dense { slot, changed } => {
                slot[row as usize] = held;
                *changed = Some(changed.map_or((row, row + 1), |(low, high)| {
                    (low.min(row), high.max(row + 1))
                }));
            }
            Self::Hashed {
                buckets,
                held: count,
                tombstones,
                changed,
            } => {
                let key = i32::try_from(row).expect("a hashed table's rows are I32");
                match (Self::find(buckets, row), held >= 0) {
                    (Some(at), true) => buckets[at][1] = held,
                    (Some(at), false) => {
                        // The row leaves; probes pass its bucket still.
                        buckets[at] = [TOMBSTONE, EMPTY];
                        *count -= 1;
                        *tombstones += 1;
                        changed.push(at);
                    }
                    (None, false) => {}
                    (None, true) => {
                        // The first bucket from the row's own that holds no
                        // row: a tombstone the probe passes, or the empty
                        // bucket that ends it.
                        let mut probe = bucket(key, buckets.len());
                        while buckets[probe][0] >= 0 {
                            probe = (probe + 1) & (buckets.len() - 1);
                        }
                        if buckets[probe][0] == TOMBSTONE {
                            *tombstones -= 1;
                        }
                        buckets[probe] = [key, held];
                        *count += 1;
                        changed.push(probe);
                    }
                }
            }
        }
    }
}

pub(crate) struct SlotDirectory {
    /// Rows of the table.
    rows: u64,
    /// Rows of one tile.
    tile_rows: u64,
    entries: Entries,
    /// Per tile of slots, the first row of the table tile it holds, or
    /// `NO_TILE` (`u32::MAX`) when empty.
    tile: Vec<u32>,
    /// Per tile of slots, whether its tile was used since the clock last
    /// passed it.
    used: BitSet,
    /// The tile of slots the clock looks at next. The clock turns from the
    /// last tile of slots in use toward the first, so that of slots filled in
    /// order of value, first to last, it reaches the least valuable first.
    hand: usize,
    /// The leading tiles of slots in use; the others hold nothing.
    active: usize,
}

impl SlotDirectory {
    /// A directory of `rows` table rows over `slots` slots, both whole tiles
    /// of `tile_rows` rows, in `form`. A hashed directory's rows are I32.
    pub fn new(rows: u64, slots: u64, tile_rows: u64, form: DirectoryForm) -> Self {
        assert!(
            rows < u32::MAX as u64,
            "SlotDirectory rows ({rows}) must fit in u32 excluding NO_TILE sentinel"
        );
        let slot_tiles = (slots / tile_rows) as usize;
        let entries = match form {
            DirectoryForm::Dense => Entries::Dense {
                slot: vec![EMPTY; rows as usize],
                changed: None,
            },
            DirectoryForm::Hashed => Entries::Hashed {
                buckets: vec![[EMPTY, EMPTY]; form.entries(rows, slots) as usize],
                held: 0,
                tombstones: 0,
                changed: Vec::new(),
            },
        };
        Self {
            rows,
            tile_rows,
            entries,
            tile: vec![NO_TILE; slot_tiles],
            used: BitSet::new(slot_tiles),
            hand: slot_tiles.saturating_sub(1),
            active: slot_tiles,
        }
    }

    /// Use the leading `tiles` tiles of slots, at most every one: a tile
    /// past them is released.
    pub fn limit(&mut self, tiles: usize) {
        let tiles = tiles.min(self.tile.len());
        for tile in tiles..self.active {
            self.release(tile);
        }
        self.active = tiles;
        self.hand = tiles.saturating_sub(1);
    }

    pub fn tile_rows(&self) -> u64 {
        self.tile_rows
    }

    /// Whether `row` is a row of the table.
    pub fn has_row(&self, row: u64) -> bool {
        row < self.rows
    }

    /// The slot holding `row`; `None` of a row not held or outside the table.
    pub fn slot(&self, row: u64) -> Option<u64> {
        self.entries.slot(row)
    }

    /// The slot holding `row`, whose tile is used from here on.
    pub fn use_slot(&mut self, row: u64) -> Option<u64> {
        let slot = self.slot(row)?;
        self.used.set((slot / self.tile_rows) as usize);
        Some(slot)
    }

    /// The first row of the table tile the tile of slots `slot_tile` holds.
    pub fn held(&self, slot_tile: usize) -> Option<u64> {
        let first = self.tile[slot_tile];
        (first != NO_TILE).then_some(u64::from(first))
    }

    /// Every entry of the directory as the device holds it, little-endian
    /// I32, and the entries that changed since are none.
    pub fn entry_bytes(&mut self) -> Vec<u8> {
        match &mut self.entries {
            Entries::Dense { slot, changed } => {
                *changed = None;
                slot.iter().flat_map(|slot| slot.to_le_bytes()).collect()
            }
            Entries::Hashed {
                buckets, changed, ..
            } => {
                changed.clear();
                buckets.iter().flatten().flat_map(|field| field.to_le_bytes()).collect()
            }
        }
    }

    /// The entries that changed since the changes were last taken, as runs
    /// of adjacent entries: the first entry of each run and its bytes,
    /// little-endian I32, ascending. A dense directory's changes are one run
    /// from the first changed row to the last.
    pub fn take_changes(&mut self) -> Vec<(u64, Vec<u8>)> {
        match &mut self.entries {
            Entries::Dense { slot, changed } => changed
                .take()
                .map(|(low, high)| {
                    let bytes = slot[low as usize..high as usize]
                        .iter()
                        .flat_map(|slot| slot.to_le_bytes())
                        .collect();
                    (low, bytes)
                })
                .into_iter()
                .collect(),
            Entries::Hashed {
                buckets, changed, ..
            } => {
                // Buckets a few apart are written together.
                const GAP: usize = 16;
                let mut changed = std::mem::take(changed);
                changed.sort_unstable();
                changed.dedup();
                let mut runs = Vec::new();
                let mut at = 0;
                while at < changed.len() {
                    let mut end = at + 1;
                    while end < changed.len() && changed[end] - changed[end - 1] <= GAP {
                        end += 1;
                    }
                    let (first, last) = (changed[at], changed[end - 1]);
                    let bytes = buckets[first..=last]
                        .iter()
                        .flatten()
                        .flat_map(|field| field.to_le_bytes())
                        .collect();
                    runs.push((first as u64, bytes));
                    at = end;
                }
                runs
            }
        }
    }

    /// Whether a hashed directory's tombstones lengthen its probes enough
    /// that it should be rebuilt (`rebuild`): fewer than a quarter of its
    /// buckets are empty.
    pub fn crowded(&self) -> bool {
        match &self.entries {
            Entries::Dense { .. } => false,
            Entries::Hashed {
                buckets,
                held,
                tombstones,
                ..
            } => 4 * (held + tombstones) > 3 * buckets.len(),
        }
    }

    /// Enter every held row again into empty buckets, dropping the
    /// tombstones. Every bucket changes: a reader may not probe the
    /// directory meanwhile.
    pub fn rebuild(&mut self) {
        let rows = match &mut self.entries {
            Entries::Dense { .. } => return,
            Entries::Hashed {
                buckets,
                held,
                tombstones,
                ..
            } => {
                let rows = buckets
                    .iter()
                    .filter(|[row, _]| *row >= 0)
                    .copied()
                    .collect::<Vec<_>>();
                buckets.fill([EMPTY, EMPTY]);
                (*held, *tombstones) = (0, 0);
                rows
            }
        };
        for [row, slot] in rows {
            self.entries.set(row as u64, slot);
        }
        if let Entries::Hashed {
            buckets, changed, ..
        } = &mut self.entries
        {
            *changed = (0..buckets.len()).collect();
        }
    }

    /// Hold no tile in the tile of slots `slot_tile`: the first row of the
    /// one it held.
    pub fn release(&mut self, slot_tile: usize) -> Option<u64> {
        self.used.clear(slot_tile);
        let first = std::mem::replace(&mut self.tile[slot_tile], NO_TILE);
        if first == NO_TILE {
            return None;
        }
        let first = u64::from(first);
        for row in first..first + self.tile_rows {
            self.entries.set(row, EMPTY);
        }
        Some(first)
    }

    /// Hold the table tile of first row `first` in the tile of slots
    /// `slot_tile`, which holds none. `used`: whether the clock passes it
    /// once before it takes it.
    pub fn hold(&mut self, first: u64, slot_tile: usize, used: bool) {
        let first_u32 = u32::try_from(first).expect("held row fits u32");
        assert_ne!(first_u32, NO_TILE, "held row collides with NO_TILE sentinel");
        let first_slot = slot_tile as u64 * self.tile_rows;
        for row in 0..self.tile_rows {
            self.entries.set(first + row, (first_slot + row) as i32);
        }
        self.tile[slot_tile] = first_u32;
        self.used.set_to(slot_tile, used);
    }

    /// A tile of slots to hold another tile in: the clock's next one that is
    /// neither used since it last passed nor `kept`, released, and the first
    /// row of the tile it held. The clock passes every tile of slots at most
    /// twice, the first time taking from each its use; `None` when it finds
    /// none, as of a directory without slots. Only the tiles in use are
    /// taken ([`Self::limit`]).
    pub fn free(&mut self, kept: impl Fn(usize) -> bool) -> Option<(usize, Option<u64>)> {
        match self.try_free(|slot_tile| Ok::<_, std::convert::Infallible>(kept(slot_tile))) {
            Ok(free) => free,
            Err(never) => match never {},
        }
    }

    /// [`Self::free`] by a predicate that may fail to say whether a tile is
    /// kept: its first error ends the search, nothing released.
    pub fn try_free<E>(&mut self, kept: impl Fn(usize) -> Result<bool, E>) -> Result<Option<(usize, Option<u64>)>, E> {
        let tiles = self.active;
        for _ in 0..2 * tiles {
            let slot_tile = self.hand;
            self.hand = (self.hand + tiles - 1) % tiles;
            if kept(slot_tile)? || self.used.take(slot_tile) {
                continue;
            }
            return Ok(Some((slot_tile, self.release(slot_tile))));
        }
        Ok(None)
    }

    /// The first of `count` adjacent tiles of slots, each released: the
    /// clock's next ones, the tiles that end at the hand. `None` of a
    /// directory with fewer.
    pub fn free_adjacent(&mut self, count: usize) -> Option<usize> {
        let tiles = self.active;
        if count > tiles {
            return None;
        }
        if self.hand + 1 < count {
            self.hand = tiles - 1;
        }
        let first = self.hand + 1 - count;
        for slot_tile in first..first + count {
            self.release(slot_tile);
        }
        self.hand = (first + tiles - 1) % tiles;
        Some(first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|entry| i32::from_le_bytes(entry.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn the_clock_frees_a_slot_whose_row_was_not_used_since_it_last_passed() {
        let mut directory = SlotDirectory::new(8, 3, 1, DirectoryForm::Dense);
        assert_eq!(directory.slot(5), None);
        assert_eq!(directory.slot(8), None);
        // The clock turns from the last slot toward the first.
        for (row, slot) in [(5, 2), (6, 1), (7, 0)] {
            assert_eq!(directory.free(|_| false), Some((slot, None)));
            directory.hold(row, slot, true);
        }
        assert_eq!(
            directory.take_changes(),
            [(5, [2i32, 1, 0].map(i32::to_le_bytes).concat())]
        );
        assert!(directory.take_changes().is_empty());
        // Every row was used: the clock passes them all once, then takes
        // the last. A row used again in between is passed again.
        assert_eq!(directory.free(|_| false), Some((2, Some(5))));
        assert_eq!(directory.slot(5), None);
        directory.hold(1, 2, true);
        assert_eq!(directory.use_slot(6), Some(1));
        assert_eq!(directory.free(|_| false), Some((0, Some(7))));
        directory.hold(2, 0, false);
        // A kept slot is never taken, and nothing is when all are kept.
        assert_eq!(directory.free(|slot| slot == 2), Some((1, Some(6))));
        assert_eq!(directory.free(|_| true), None);
        assert_eq!(
            SlotDirectory::new(8, 0, 1, DirectoryForm::Dense).free(|_| false),
            None
        );
    }

    #[test]
    fn tiles_of_several_rows_hold_and_release_together() {
        for form in [DirectoryForm::Dense, DirectoryForm::Hashed] {
            let mut directory = SlotDirectory::new(8, 4, 2, form);
            directory.hold(6, 1, false);
            assert_eq!(directory.slot(6), Some(2));
            assert_eq!(directory.slot(7), Some(3));
            assert_eq!(directory.held(1), Some(6));
            assert_eq!(directory.release(1), Some(6));
            assert_eq!(directory.slot(7), None);
            assert_eq!(directory.release(1), None);
        }
    }

    #[test]
    fn adjacent_slots_are_the_clocks_next_ones() {
        let mut directory = SlotDirectory::new(8, 4, 1, DirectoryForm::Dense);
        assert_eq!(directory.free_adjacent(5), None);
        assert_eq!(directory.free_adjacent(3), Some(1));
        directory.hold(4, 1, true);
        // Two more do not fit before the first slot: the clock starts over
        // at the last, and the rows there leave their slots.
        assert_eq!(directory.free_adjacent(2), Some(2));
        assert_eq!(directory.free_adjacent(2), Some(0));
        assert_eq!(directory.slot(4), None);
    }

    /// The slot a reader finds for `row` in the device's entries, probing as
    /// `gathered_row` does.
    fn probed(device: &[i32], row: i32) -> Option<i32> {
        let buckets = device.len() / 2;
        let mut probe = bucket(row, buckets);
        for _ in 0..buckets {
            match device[2 * probe] {
                key if key == row => return Some(device[2 * probe + 1]),
                EMPTY => return None,
                _ => probe = (probe + 1) % buckets,
            }
        }
        None
    }

    #[test]
    fn a_hashed_directory_finds_what_it_holds_through_its_changes() {
        // A table far larger than its four slots, whose rows are found in
        // eight buckets.
        let rows = 1 << 30;
        let mut directory = SlotDirectory::new(rows, 4, 1, DirectoryForm::Hashed);
        let mut device = entries(&directory.entry_bytes());
        assert_eq!(device, [EMPTY; 16]);
        let apply = |directory: &mut SlotDirectory, device: &mut Vec<i32>| {
            for (first, bytes) in directory.take_changes() {
                let first = 2 * first as usize;
                let run = entries(&bytes);
                device[first..first + run.len()].copy_from_slice(&run);
            }
        };
        let mut held = std::collections::BTreeMap::new();
        // Rows enter and leave many times over, so tombstones accumulate
        // and are reused; what the device reads is always what the host
        // holds.
        let mut state = 7u64;
        for step in 0..4000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let row = (state >> 33) % 64 * 7919 % rows;
            if directory.slot(row).is_none() {
                let (slot, left) = directory.free(|_| false).unwrap();
                if let Some(left) = left {
                    held.remove(&left);
                }
                directory.hold(row, slot, false);
                held.insert(row, slot as i32);
            }
            if directory.crowded() {
                directory.rebuild();
            }
            apply(&mut directory, &mut device);
            if step % 97 == 0 {
                for probe in 0..64u64 {
                    let row = probe * 7919 % rows;
                    assert_eq!(directory.slot(row).map(|slot| slot as i32), held.get(&row).copied());
                    assert_eq!(probed(&device, row as i32), held.get(&row).copied(), "row {row}");
                }
            }
        }
        // At least one bucket is empty, as every probe needs.
        assert!(device.chunks_exact(2).any(|bucket| bucket[0] == EMPTY));
        assert_eq!(directory.slot(rows), None);
    }

    #[test]
    fn bitset_round_trips_across_word_boundaries() {
        let mut bits = BitSet::new(130);
        for i in 0..130 {
            assert!(!bits.get(i));
        }
        for &idx in &[0usize, 1, 63, 64, 65, 127, 128, 129] {
            bits.set(idx);
            assert!(bits.get(idx));
        }
        assert!(!bits.get(2));
        assert!(!bits.get(62));
        assert!(!bits.get(126));

        bits.or_assign(62, false);
        assert!(!bits.get(62));
        bits.or_assign(62, true);
        assert!(bits.get(62));

        assert!(bits.take(64));
        assert!(!bits.get(64));
        assert!(!bits.take(64));

        bits.set_to(129, false);
        assert!(!bits.get(129));
        bits.set_to(129, true);
        assert!(bits.get(129));
    }

    #[test]
    fn packed_tile_and_used_round_trip_edge_rows_and_word_boundaries() {
        // 130 slots spanning 3 bitset words; large table with rows up to i32::MAX.
        let rows = i32::MAX as u64;
        let mut directory = SlotDirectory::new(rows, 130, 1, DirectoryForm::Hashed);
        for slot in 0..130 {
            assert_eq!(directory.held(slot), None);
        }
        let max_row = rows - 1;
        directory.hold(0, 0, true);
        directory.hold(63, 63, false);
        directory.hold(64, 64, true);
        directory.hold(max_row, 129, true);

        assert_eq!(directory.held(0), Some(0));
        assert_eq!(directory.held(63), Some(63));
        assert_eq!(directory.held(64), Some(64));
        assert_eq!(directory.held(129), Some(max_row));
        assert_eq!(directory.slot(0), Some(0));
        assert_eq!(directory.slot(max_row), Some(129));

        // Slot 129 was marked used, so the clock passes it once and takes 128 (empty).
        assert_eq!(directory.free(|_| false), Some((128, None)));
        assert_eq!(directory.held(129), Some(max_row));

        assert_eq!(directory.release(129), Some(max_row));
        assert_eq!(directory.held(129), None);
        assert_eq!(directory.release(129), None);
        assert_eq!(directory.slot(max_row), None);
    }
}
