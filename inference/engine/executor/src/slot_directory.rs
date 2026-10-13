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

/// The bucket a hashed directory of `buckets` (a power of two) probes first
/// for `row`: murmur3's 32-bit finalizer, as `gathered_bucket`
/// (target.seismic) computes it.
pub(crate) fn bucket(row: i32, buckets: usize) -> usize {
    let mut h = row as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h as usize & (buckets - 1)
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
    /// of `tile_rows` rows, in `form`. A hashed directory's rows are I32.
    pub fn new(rows: u64, slots: u64, tile_rows: u64, form: DirectoryForm) -> Self {
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
        row < self.rows
    }

    /// The slot holding `row`; `None` of a row not held or outside the table.
    pub fn slot(&self, row: u64) -> Option<u64> {
        self.entries.slot(row)
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
        self.used[slot_tile] = false;
        let first = self.tile[slot_tile].take()?;
        for row in first..first + self.tile_rows {
            self.entries.set(row, EMPTY);
        }
        Some(first)
    }

    /// Hold the table tile of first row `first` in the tile of slots
    /// `slot_tile`, which holds none. `used`: whether the clock passes it
    /// once before it takes it.
    pub fn hold(&mut self, first: u64, slot_tile: usize, used: bool) {
        let first_slot = slot_tile as u64 * self.tile_rows;
        for row in 0..self.tile_rows {
            self.entries.set(first + row, (first_slot + row) as i32);
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
        for (row, slot) in [(5, 0), (6, 1), (7, 2)] {
            assert_eq!(directory.free(|_| false), Some((slot, None)));
            directory.hold(row, slot, true);
        }
        assert_eq!(
            directory.take_changes(),
            [(5, [0i32, 1, 2].map(i32::to_le_bytes).concat())]
        );
        assert!(directory.take_changes().is_empty());
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
        assert_eq!(
            SlotDirectory::new(8, 0, 1, DirectoryForm::Dense).free(|_| false),
            None
        );
    }

    #[test]
    fn tiles_of_several_rows_hold_and_release_together() {
        for form in [DirectoryForm::Dense, DirectoryForm::Hashed] {
            let mut directory = SlotDirectory::new(8, 4, 2, form);
            assert_eq!(directory.slot_tiles(), 2);
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
        assert_eq!(directory.free_adjacent(3), Some(0));
        directory.hold(4, 1, true);
        // Two more do not fit before the end: the clock starts over, and
        // the row there leaves its slot.
        assert_eq!(directory.free_adjacent(2), Some(0));
        assert_eq!(directory.free_adjacent(2), Some(2));
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
}
