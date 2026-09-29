// Slot and version addressing of the recurrent mixer entries that bind the
// bank version contract (recurrent.seismic, short_conv.seismic,
// state_space.seismic): the slots `segments`, `stop`, `previous_bank`,
// `previous_tape`, `following_bank`, over bank arenas stored in slabs. The
// counterpart of `metal/lib/recurrent/versions.h`,
// `cuda/lib/recurrent/versions.cuh` and `vulkan/lib/recurrent/versions.glsl`.

use seismic::cpu::Scalars;

/// One slot: rows [lo, hi), its publication row count `stop`, the version it
/// reads (bank `source` advanced by its first `taped` tape rows) and its
/// successor bank `target`.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub lo: usize,
    pub hi: usize,
    pub stop: usize,
    pub source: usize,
    pub taped: usize,
    pub target: usize,
}

impl Slot {
    /// The row after which the successor's state is published.
    #[inline(always)]
    pub fn publish(&self) -> usize {
        self.lo + self.stop
    }

    /// Tape rows the slot records in its successor: those after its stop
    /// row, at most `tape_rows` (T).
    #[inline(always)]
    pub fn recorded(&self, tape_rows: usize) -> usize {
        tape_rows.min(self.hi - self.lo - self.stop)
    }
}

/// The slot arguments of an entry, in contract order.
#[derive(Clone, Copy)]
pub struct Slots<'a> {
    pub segments: Scalars<'a, i32, 2>,
    pub stop: Scalars<'a, i32, 1>,
    pub previous_bank: Scalars<'a, i32, 1>,
    pub previous_tape: Scalars<'a, i32, 1>,
    pub following_bank: Scalars<'a, i32, 1>,
    /// B.
    pub count: usize,
}

impl Slots<'_> {
    #[inline(always)]
    pub fn slot(&self, index: usize) -> Slot {
        Slot {
            lo: self.segments.get([index, 0]) as usize,
            hi: self.segments.get([index, 1]) as usize,
            stop: self.stop.get([index]) as usize,
            source: self.previous_bank.get([index]) as usize,
            taped: self.previous_tape.get([index]) as usize,
            target: self.following_bank.get([index]) as usize,
        }
    }

    /// The slot holding `row`, if any: the slots partition a prefix of the
    /// rows in ascending order.
    #[inline(always)]
    pub fn slot_of_row(&self, row: usize) -> Option<Slot> {
        let (mut low, mut high) = (0, self.count);
        while low < high {
            let middle = (low + high) / 2;
            if self.segments.get([middle, 1]) as usize <= row {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        (low < self.count).then(|| self.slot(low))
    }

    /// The end of the rows the slots cover.
    #[inline(always)]
    pub fn covered_end(&self) -> usize {
        if self.count == 0 {
            0
        } else {
            self.segments.get([self.count - 1, 1]) as usize
        }
    }
}
