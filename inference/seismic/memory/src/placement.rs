//! Whole-graph workspace buffer placement: R_b, the two-ended rule over
//! birth order (`specs/26-11-16/model-assessment.md` §4.2).
//!
//! Every non-scratch buffer is coloured onto one end of a two-ended stack
//! (bottom or top) in birth order so that no two buffers on one end cross,
//! or lifted onto the top side where it would close an odd cycle of
//! crossings ([`colour_ends`]). Scratch buffers are lifted onto the bottom
//! side. R_b evaluates both forward and backward birth order and keeps the
//! placement with the smaller aligned arena, ties to forward ([`place`]).

use crate::BUFFER_ALIGNMENT;
use std::ops::Range;

/// A lifetime in steps, both ends live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// Whether the two are live at a common step. Endpoints count: a node's
    /// inputs end and its results start at the same step, so they never
    /// alias.
    pub fn overlaps(self, other: Span) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Whether `other` lives only while this one does.
    pub fn contains(self, other: Span) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Whether the two overlap and neither contains the other: they cannot
    /// lie on one end of a two-ended stack.
    pub fn crosses(self, other: Span) -> bool {
        self.overlaps(other) && !self.contains(other) && !other.contains(self)
    }

    /// Its order of birth, as the buffer at `index`: first born first, and
    /// of one birth the longest lived first.
    pub fn birth(self, index: usize) -> (usize, std::cmp::Reverse<usize>, usize) {
        (self.start, std::cmp::Reverse(self.end), index)
    }
}

/// One workspace buffer: its lifetime, its byte length, and whether it is a
/// node's scratch (call-private, or a paired operand living from its
/// producer to its consumer), which is lifted onto the bottom side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub span: Span,
    pub bytes: u64,
    pub scratch: bool,
}

/// An end of the two-ended stack, and the side of the gap between them a
/// lifted buffer lies against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Bottom,
    Top,
}

impl Side {
    pub const BOTH: [Side; 2] = [Side::Bottom, Side::Top];

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Where a buffer lies: nested on an end, or lifted against a side, past
/// everything that side holds over its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Colour {
    End(Side),
    Lifted(Side),
}

/// Each buffer's aligned offset in a workspace of `bytes` aligned bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub offsets: Vec<u64>,
    pub bytes: u64,
}

/// `bytes` rounded up to a [`BUFFER_ALIGNMENT`] boundary.
pub fn aligned(bytes: u64) -> u64 {
    bytes
        .checked_next_multiple_of(BUFFER_ALIGNMENT)
        .expect("workspace bytes overflow")
}

/// The bytes a buffer of `bytes` takes in the workspace: at least one,
/// padded to a [`BUFFER_ALIGNMENT`] boundary.
pub fn padded(bytes: u64) -> u64 {
    aligned(bytes.max(1))
}

#[derive(Clone, Debug, Default)]
struct Ends {
    spans: Vec<Span>,
    parent: Vec<usize>,
    parity: Vec<u8>,
    /// Previously coloured buffer IDs whose `end` is still `>=` the current
    /// span's `start`. Because `colour` is called in birth order (`start`
    /// non-decreasing), any span with `end < span.start` can never overlap or
    /// cross `span` or any later span.
    active: Vec<usize>,
}

impl Ends {
    fn with_capacity(cap: usize) -> Self {
        Self {
            spans: Vec::with_capacity(cap),
            parent: Vec::with_capacity(cap),
            parity: Vec::with_capacity(cap),
            active: Vec::new(),
        }
    }

    fn seed(&mut self, span: Span) -> usize {
        let id = self.spans.len();
        self.spans.push(span);
        self.parent.push(id);
        self.parity.push(0);
        id
    }

    fn colour(&mut self, span: Span) -> Option<usize> {
        if let Some(prev) = self.spans.last() {
            debug_assert!(
                prev.start <= span.start,
                "Ends::colour requires non-decreasing span.start (birth order)"
            );
        }
        self.active.retain(|&id| self.spans[id].end >= span.start);
        let mut required: Vec<(usize, u8)> = Vec::new();
        for &other in &self.active {
            if !span.crosses(self.spans[other]) {
                continue;
            }
            let (root, relative) = self.find(other);
            let want = relative ^ 1;
            match required.iter().find(|(found, _)| *found == root) {
                Some(&(_, wanted)) if wanted != want => return None,
                Some(_) => {}
                None => required.push((root, want)),
            }
        }
        let id = self.seed(span);
        self.active.push(id);
        // The latest joiner is its block's root.
        for (root, want) in required {
            self.parent[root] = id;
            self.parity[root] = want;
        }
        Some(id)
    }

    fn find(&self, mut id: usize) -> (usize, u8) {
        let mut relative = 0u8;
        while self.parent[id] != id {
            relative ^= self.parity[id];
            id = self.parent[id];
        }
        (id, relative)
    }
}

/// Colour the buffers of `spans` in birth order: one that would close an odd
/// cycle of crossings is lifted onto the top side, and each other lies on
/// the end of its parity relative to its block's root, the root on the
/// bottom. `scratch` buffers are not coloured: they are lifted onto the
/// bottom side.
pub fn colour_ends(spans: &[Span], scratch: &[bool]) -> Vec<Colour> {
    assert_eq!(spans.len(), scratch.len(), "a scratch flag for each buffer");
    let mut order = (0..spans.len())
        .filter(|&index| !scratch[index])
        .collect::<Vec<_>>();
    order.sort_by_key(|&index| spans[index].birth(index));
    let mut ends = Ends::with_capacity(order.len());
    let mut colours = vec![Colour::Lifted(Side::Bottom); spans.len()];
    let mut coloured = Vec::with_capacity(order.len());
    for index in order {
        match ends.colour(spans[index]) {
            Some(id) => coloured.push((index, id)),
            None => colours[index] = Colour::Lifted(Side::Top),
        }
    }
    for (index, id) in coloured {
        colours[index] = Colour::End(Side::BOTH[ends.find(id).1 as usize]);
    }
    colours
}

/// Precompiled lifetime structure for one birth-order direction (forward or
/// backward): colours, distinct start steps, each buffer's live step range,
/// and lifted buffers in birth order.
#[derive(Clone, Debug)]
struct CompiledDirection {
    colours: Vec<Colour>,
    steps: usize,
    life: Vec<Range<usize>>,
    lifted: Vec<(usize, Side)>,
}

impl CompiledDirection {
    fn of(spans: &[Span], scratch: &[bool]) -> Self {
        let colours = colour_ends(spans, scratch);
        let mut step_values = spans.iter().map(|span| span.start).collect::<Vec<_>>();
        step_values.sort_unstable();
        step_values.dedup();
        let life = spans
            .iter()
            .map(|span| {
                step_values.partition_point(|&step| step < span.start)
                    ..step_values.partition_point(|&step| step <= span.end)
            })
            .collect::<Vec<_>>();
        let mut lifted_indices = (0..spans.len())
            .filter(|&index| matches!(colours[index], Colour::Lifted(_)))
            .collect::<Vec<_>>();
        lifted_indices.sort_by_key(|&index| spans[index].birth(index));
        let lifted = lifted_indices
            .into_iter()
            .map(|index| {
                let Colour::Lifted(side) = colours[index] else { unreachable!() };
                (index, side)
            })
            .collect();
        Self {
            colours,
            steps: step_values.len(),
            life,
            lifted,
        }
    }

    /// Evaluate the aligned arena size for `padded_bytes` in this direction
    /// using O(N + S) difference-array prefix sums for coloured buffers.
    /// When `pasts` is `Some`, records each lifted buffer's `past` value for
    /// offset placement.
    fn eval_arena(&self, padded_bytes: &[u64], mut pasts: Option<&mut [u64]>) -> u64 {
        let mut add = [vec![0u64; self.steps], vec![0u64; self.steps]];
        let mut sub = [vec![0u64; self.steps + 1], vec![0u64; self.steps + 1]];
        for (index, &colour) in self.colours.iter().enumerate() {
            if let Colour::End(side) = colour {
                let lives = &self.life[index];
                if lives.start < lives.end {
                    let s = side.index();
                    let bytes = padded_bytes[index];
                    add[s][lives.start] = add[s][lives.start]
                        .checked_add(bytes)
                        .expect("workspace bytes overflow");
                    sub[s][lives.end] = sub[s][lives.end]
                        .checked_add(bytes)
                        .expect("workspace bytes overflow");
                }
            }
        }
        let mut coloured = [vec![0u64; self.steps], vec![0u64; self.steps]];
        for side in Side::BOTH {
            let s = side.index();
            let mut running = 0u64;
            for at in 0..self.steps {
                running -= sub[s][at];
                running = running
                    .checked_add(add[s][at])
                    .expect("workspace bytes overflow");
                coloured[s][at] = running;
            }
        }

        let mut pasts_at = [vec![0u64; self.steps], vec![0u64; self.steps]];
        for &(index, side) in &self.lifted {
            let s = side.index();
            let lives = self.life[index].clone();
            let mut most = 0u64;
            for at in lives.clone() {
                most = most.max(coloured[s][at]).max(pasts_at[s][at]);
            }
            let past = most
                .checked_add(padded_bytes[index])
                .expect("workspace bytes overflow");
            for at in lives {
                pasts_at[s][at] = pasts_at[s][at].max(past);
            }
            if let Some(pasts) = pasts.as_deref_mut() {
                pasts[index] = past;
            }
        }

        let mut arena = 0u64;
        for at in 0..self.steps {
            let bottom =
                pasts_at[Side::Bottom.index()][at].max(coloured[Side::Bottom.index()][at]);
            let top = pasts_at[Side::Top.index()][at].max(coloured[Side::Top.index()][at]);
            arena = arena.max(bottom.checked_add(top).expect("workspace bytes overflow"));
        }
        arena
    }

    /// Peak simultaneously live bytes over all buffers in this direction,
    /// using O(N + S) difference-array prefix sums.
    fn liveness_floor(&self, raw_bytes: &[u64]) -> u64 {
        liveness_floor_with_life(self.steps, &self.life, raw_bytes)
    }

    /// Compute every buffer's aligned offset within an arena of `arena` bytes
    /// given the `pasts` recorded by [`Self::eval_arena`]. On each end its
    /// buffers nest, each past those of its end that enclose it, from the
    /// bottom up or from the top down; a lifted buffer lies past everything
    /// its side holds over its life.
    fn place_offsets(
        &self,
        spans: &[Span],
        padded_bytes: &[u64],
        arena: u64,
        pasts: &[u64],
    ) -> Placed {
        let mut offsets = vec![0u64; spans.len()];
        let mut order = (0..spans.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| spans[index].birth(index));
        // Each end's buffers enclosing the one placed, innermost last, with
        // what they and those enclosing them take: an equal lifetime encloses
        // only a later-born one.
        let mut enclosing: [Vec<(usize, u64)>; 2] = [Vec::new(), Vec::new()];
        for index in order {
            match self.colours[index] {
                Colour::End(end) => {
                    let nest = &mut enclosing[end.index()];
                    while nest
                        .last()
                        .is_some_and(|&(outer, _)| spans[outer].end < spans[index].start)
                    {
                        nest.pop();
                    }
                    if let Some(&(outer, _)) = nest.last() {
                        assert!(
                            spans[outer].contains(spans[index]),
                            "invariant violated: buffers {outer} and {index} cross on one end"
                        );
                    }
                    let under = nest.last().map_or(0, |&(_, under)| under);
                    offsets[index] = match end {
                        Side::Bottom => under,
                        Side::Top => arena
                            .checked_sub(padded_bytes[index])
                            .and_then(|top| top.checked_sub(under))
                            .expect("invariant violated: a buffer lies outside its arena"),
                    };
                    nest.push((
                        index,
                        under
                            .checked_add(padded_bytes[index])
                            .expect("workspace bytes overflow"),
                    ));
                }
                Colour::Lifted(side) => {
                    let past = pasts[index];
                    offsets[index] = match side {
                        Side::Bottom => past
                            .checked_sub(padded_bytes[index])
                            .expect("invariant violated: a buffer lies outside its arena"),
                        Side::Top => arena
                            .checked_sub(past)
                            .expect("invariant violated: a buffer lies outside its arena"),
                    };
                }
            }
        }
        Placed { offsets, bytes: arena }
    }
}

fn liveness_floor_with_life(steps: usize, life: &[Range<usize>], raw_bytes: &[u64]) -> u64 {
    let mut add = vec![0u64; steps];
    let mut sub = vec![0u64; steps + 1];
    for (index, lives) in life.iter().enumerate() {
        if lives.start < lives.end {
            let bytes = raw_bytes[index];
            add[lives.start] = add[lives.start]
                .checked_add(bytes)
                .expect("live workspace bytes fit u64");
            sub[lives.end] = sub[lives.end]
                .checked_add(bytes)
                .expect("live workspace bytes fit u64");
        }
    }
    let mut running = 0u64;
    let mut floor = 0u64;
    for at in 0..steps {
        running -= sub[at];
        running = running
            .checked_add(add[at])
            .expect("live workspace bytes fit u64");
        floor = floor.max(running);
    }
    floor
}

/// Peak simultaneously live bytes over `spans` with `raw_bytes`: the liveness
/// floor no placement can go below.
pub fn liveness_floor(spans: &[Span], raw_bytes: &[u64]) -> u64 {
    assert_eq!(spans.len(), raw_bytes.len(), "a byte count for each buffer");
    let mut step_values = spans.iter().map(|span| span.start).collect::<Vec<_>>();
    step_values.sort_unstable();
    step_values.dedup();
    let life = spans
        .iter()
        .map(|span| {
            step_values.partition_point(|&step| step < span.start)
                ..step_values.partition_point(|&step| step <= span.end)
        })
        .collect::<Vec<_>>();
    liveness_floor_with_life(step_values.len(), &life, raw_bytes)
}

fn backward_spans(spans: &[Span]) -> Vec<Span> {
    let last = spans.iter().map(|span| span.end).max().unwrap_or(0);
    spans
        .iter()
        .map(|span| Span {
            start: last - span.end,
            end: last - span.start,
        })
        .collect()
}

/// Precompiled R_b lifetime structure for repeated arena sizing across
/// varying buffer byte lengths.
#[derive(Clone, Debug)]
pub struct CompiledPlacement {
    forward: CompiledDirection,
    backward: CompiledDirection,
    len: usize,
}

impl CompiledPlacement {
    /// Precompile forward and backward R_b colouring and step ranges from
    /// buffer lifetimes and scratch flags.
    pub fn of(spans: &[Span], scratch: &[bool]) -> Self {
        assert_eq!(spans.len(), scratch.len(), "a scratch flag for each buffer");
        let backward = backward_spans(spans);
        Self {
            forward: CompiledDirection::of(spans, scratch),
            backward: CompiledDirection::of(&backward, scratch),
            len: spans.len(),
        }
    }

    /// The aligned arena byte count that [`place`] produces for buffers of
    /// these lifetimes and `padded_bytes`, without computing offsets.
    pub fn arena_bytes(&self, padded_bytes: &[u64]) -> u64 {
        assert_eq!(
            padded_bytes.len(),
            self.len,
            "a padded byte count for each buffer"
        );
        let fwd = self.forward.eval_arena(padded_bytes, None);
        let bwd = self.backward.eval_arena(padded_bytes, None);
        fwd.min(bwd)
    }

    /// The peak simultaneously live bytes for buffers of these lifetimes and
    /// `raw_bytes`: the liveness floor no placement can go below.
    pub fn liveness_floor(&self, raw_bytes: &[u64]) -> u64 {
        assert_eq!(raw_bytes.len(), self.len, "a byte count for each buffer");
        self.forward.liveness_floor(raw_bytes)
    }
}

/// Place `buffers` by R_b: runs forward and backward birth order and keeps
/// the candidate with the smaller aligned arena (ties to forward). Asserts
/// that every buffer fits within the arena and no two overlapping buffers
/// alias.
pub fn place(buffers: &[Buffer]) -> Placed {
    let padded_bytes = buffers.iter().map(|buffer| padded(buffer.bytes)).collect::<Vec<_>>();
    let scratch = buffers.iter().map(|buffer| buffer.scratch).collect::<Vec<_>>();
    let forward = buffers.iter().map(|buffer| buffer.span).collect::<Vec<_>>();
    let backward = backward_spans(&forward);
    let fwd_dir = CompiledDirection::of(&forward, &scratch);
    let bwd_dir = CompiledDirection::of(&backward, &scratch);
    let mut fwd_pasts = vec![0u64; buffers.len()];
    let fwd_arena = fwd_dir.eval_arena(&padded_bytes, Some(&mut fwd_pasts));
    let mut bwd_pasts = vec![0u64; buffers.len()];
    let bwd_arena = bwd_dir.eval_arena(&padded_bytes, Some(&mut bwd_pasts));
    let placed = if fwd_arena <= bwd_arena {
        fwd_dir.place_offsets(&forward, &padded_bytes, fwd_arena, &fwd_pasts)
    } else {
        bwd_dir.place_offsets(&backward, &padded_bytes, bwd_arena, &bwd_pasts)
    };
    #[cfg(debug_assertions)]
    for a in 0..buffers.len() {
        let a_start = placed.offsets[a];
        let a_end = a_start + padded_bytes[a];
        debug_assert!(
            a_end <= placed.bytes,
            "invariant violated: buffer {a} [{a_start}, {a_end}) exceeds arena {}",
            placed.bytes
        );
        for b in (a + 1)..buffers.len() {
            if buffers[a].span.overlaps(buffers[b].span) {
                let b_start = placed.offsets[b];
                let b_end = b_start + padded_bytes[b];
                debug_assert!(
                    a_end <= b_start || b_end <= a_start,
                    "invariant violated: overlapping buffers {a} [{a_start}, {a_end}) and {b} [{b_start}, {b_end}) alias"
                );
            }
        }
    }
    placed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_compiled_matches_place(spans: &[Span], scratch: &[bool], byte_sets: &[&[u64]]) {
        let compiled = CompiledPlacement::of(spans, scratch);
        for &sizes in byte_sets {
            assert_eq!(sizes.len(), spans.len());
            let buffers = spans
                .iter()
                .zip(scratch)
                .zip(sizes)
                .map(|((&span, &scratch), &bytes)| Buffer {
                    span,
                    bytes,
                    scratch,
                })
                .collect::<Vec<_>>();
            let padded_bytes = sizes.iter().map(|&bytes| padded(bytes)).collect::<Vec<_>>();
            let placed = place(&buffers);
            assert_eq!(compiled.arena_bytes(&padded_bytes), placed.bytes);
        }
    }

    #[test]
    fn compiled_placement_matches_place_on_synthetic_cases() {
        // Empty.
        assert_compiled_matches_place(&[], &[], &[&[]]);

        // Single buffer (including 0-byte -> padded to BUFFER_ALIGNMENT).
        assert_compiled_matches_place(
            &[Span { start: 0, end: 1 }],
            &[false],
            &[&[0], &[1], &[256], &[4096]],
        );

        // Crossing pair (2-colourable: bottom and top ends) + nested + disjoint.
        assert_compiled_matches_place(
            &[
                Span { start: 0, end: 3 },
                Span { start: 1, end: 5 },
                Span { start: 2, end: 2 },
                Span { start: 6, end: 8 },
            ],
            &[false, false, false, false],
            &[
                &[512, 1024, 256, 2048],
                &[4096, 256, 1024, 512],
                &[0, 0, 0, 0],
            ],
        );

        // Odd cycle of crossings (5-cycle: forces Colour::Lifted(Side::Top))
        // combined with scratch buffers (Colour::Lifted(Side::Bottom)).
        let spans = [
            Span { start: 0, end: 3 },
            Span { start: 1, end: 5 },
            Span { start: 2, end: 7 },
            Span { start: 4, end: 9 },
            Span { start: 6, end: 10 },
            // Bridges back to create odd crossing cycles with earlier spans.
            Span { start: 2, end: 8 },
            // Scratch buffers overlapping different phases.
            Span { start: 1, end: 4 },
            Span { start: 5, end: 8 },
        ];
        let scratch = [false, false, false, false, false, false, true, true];
        assert_compiled_matches_place(
            &spans,
            &scratch,
            &[
                &[256, 512, 1024, 2048, 4096, 768, 512, 1024],
                &[8192, 256, 256, 256, 8192, 4096, 2048, 256],
                &[1, 1, 1, 1, 1, 1, 1, 1],
            ],
        );

        // Asymmetric crossing where backward birth order differs from forward.
        let asym_spans = [
            Span { start: 0, end: 4 },
            Span { start: 1, end: 6 },
            Span { start: 2, end: 5 },
            Span { start: 3, end: 8 },
            Span { start: 5, end: 9 },
            Span { start: 2, end: 3 },
        ];
        let asym_scratch = [false, false, false, false, false, true];
        assert_compiled_matches_place(
            &asym_spans,
            &asym_scratch,
            &[
                &[1024, 256, 4096, 512, 2048, 768],
                &[256, 4096, 512, 2048, 256, 1024],
            ],
        );
    }
}
