//! The CPU's executable form of a conversion recipe
//! (`registry::PacketRepackRecipe`): its bit routes and its values' source
//! fields compiled into byte moves, its computed values' postfix programs
//! (`seismic_lang::repack::PacketValues`) grouped by shape, as plain integer
//! records (no pointers, lookup tables by registered id), and the executor
//! over that form.
//!
//! A program reads some consecutive source packets and writes one output:
//! their packet forms ([`PacketForms`]), or wherever else a `destination`
//! map of [`RepackProgram::compile`] puts the packet forms' bits. Its input
//! is the source packets, then a slot per computed value; a slot per field a
//! value reads lies apart. It runs in three steps:
//!
//! 1. Field moves copy every field's source bits into its slot. A field is
//!    any part of an expression that only places source bits (a source
//!    field, shifted, or an OR of such with disjoint bits); a value that is
//!    one is routed from the source like a bit route, and has no slot.
//! 2. The values of each program shape are evaluated together, op by op
//!    over the slots of their fields, into their slots.
//! 3. Output moves move every output bit from its input bit: a bit route's
//!    source bit, or a value's bit.
//!
//! Moves ([`ByteMoves`]) are vector steps of [`BLOCK`] bytes: a span of
//! whole bytes consecutive on both sides is copied, and every other output
//! block is the OR of terms (zero of none), each picking bytes of one input
//! window and shifting and masking each. Moves that recur at fixed input
//! and output strides (across packets and rows) are executed as one
//! [`Repeat`].
//!
//! Inputs and outputs are segments, each addressed from its own base when
//! the program runs: the input's are the source packets of each of its rows
//! and the value slots, the output's are the caller's (a destination's
//! planes). A program so reads its source and writes its destination in
//! place ([`RepackExecutor::run_into`]), or stages both in its own buffers
//! ([`RepackExecutor::run`]).

use seismic_lang::registry::RepresentationConversion;
use seismic_lang::repack::{
    negate_if, slot, PacketBit, PacketValues, RepackOp, RepackShape, STACK,
};

/// Bytes of a vector step: an output block, a term's input window, a
/// span's least length.
pub const BLOCK: usize = 16;

/// Source bytes one execution of a program reads at least: an execution has
/// fixed costs (dispatching its moves and values), which a few packets
/// would not amortize. Measured (`repack_throughput`, M4 Max): fewer bytes
/// slow every conversion; twice as many speed most a little but slow `q8_0`
/// into `rows16`, whose moves are nearly all copies, by up to a quarter.
pub const EXECUTION_BYTES: usize = 4096;

/// `bytes` (at least [`BLOCK`]) bytes of input segment `input_segment` from
/// byte `input` copied to output segment `output_segment` from byte
/// `output`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ByteSpan {
    input_segment: u32,
    input: u32,
    output_segment: u32,
    output: u32,
    bytes: u32,
}

/// One input window's part of an output block: lane `l` takes byte
/// `input + indices[l]` of input segment `segment` (nothing for an index of
/// `0x80`), shifted left by `shifts[l]` (right when negative), under
/// `masks[l]`. The window is the [`BLOCK`] bytes from `input`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ByteTerm {
    segment: u32,
    input: u32,
    indices: [u8; BLOCK],
    shifts: [i8; BLOCK],
    masks: [u8; BLOCK],
}

/// The [`BLOCK`] bytes of output segment `segment` from byte `output`: the
/// OR of the terms from `terms` on, zero where none has bits; the first
/// `whole` take every lane's byte whole (no shift, every mask bit), the next
/// `masked` take bits unshifted, and the next `shifted` take bits shifted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ByteBlock {
    segment: u32,
    output: u32,
    terms: u32,
    whole: u32,
    masked: u32,
    shifted: u32,
}

/// How a term takes its lanes' bytes (a [`ByteBlock`]'s kinds of term).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Taking {
    Whole,
    Masked,
    Shifted,
}

/// How far a repetition moves its moves: by `input_segments` input
/// segments and `input` bytes, and `output_segments` output segments and
/// `output` bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Step {
    input_segments: u32,
    input: u32,
    output_segments: u32,
    output: u32,
}

impl Step {
    fn times(self, times: u32) -> Self {
        Self {
            input_segments: self.input_segments * times,
            input: self.input * times,
            output_segments: self.output_segments * times,
            output: self.output * times,
        }
    }

    /// Whether the step moves segments.
    fn segmented(self) -> bool {
        (self.input_segments, self.output_segments) != (0, 0)
    }
}

/// The moves `first..first + length` executed `count` times, the `i`-th
/// time moved `i` steps `step`, and all of that `outer` times, the `o`-th
/// time moved `o` steps `outer_step` (the repetitions of rows, say, of the
/// repetitions within a row).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Repeat {
    first: u32,
    length: u32,
    count: u32,
    step: Step,
    outer: u32,
    outer_step: Step,
}

/// Where a repetition of moves starts: input and output segments and bytes
/// past the moves' own.
#[derive(Clone, Copy)]
struct Origin {
    input_segment: usize,
    input: usize,
    output_segment: usize,
    output: usize,
}

impl Origin {
    fn of(step: Step, times: usize) -> Self {
        Self {
            input_segment: step.input_segments as usize * times,
            input: step.input as usize * times,
            output_segment: step.output_segments as usize * times,
            output: step.output as usize * times,
        }
    }

    fn then(self, step: Step, times: usize) -> Self {
        let more = Self::of(step, times);
        Self {
            input_segment: self.input_segment + more.input_segment,
            input: self.input + more.input,
            output_segment: self.output_segment + more.output_segment,
            output: self.output + more.output,
        }
    }
}

/// Moves from segmented input into segmented output: every block, then
/// every span (whose bytes a block covering them leaves zero). A block lies
/// within its segment's whole blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ByteMoves {
    spans: Vec<ByteSpan>,
    span_repeats: Vec<Repeat>,
    blocks: Vec<ByteBlock>,
    terms: Vec<ByteTerm>,
    block_repeats: Vec<Repeat>,
}

/// The `count` values of one shape. Value `v` is the low `bytes` (1, 2 or
/// 4) bytes of value slot bytes `values + v * bytes`; its field of op
/// `Field { prefix, width }` is field slot bytes
/// `fields + prefix * count + v * slot(width)`, its offset of op
/// `Offset { offset }` the program's offset `offsets + offset * count + v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValueGroup {
    shape: u32,
    count: u32,
    fields: u32,
    offsets: u32,
    values: u32,
    bytes: u32,
}

/// Values evaluated together, a lane each.
const LANES: usize = 64;
/// The longest period of moves one [`Repeat`] recurs.
const PERIOD: usize = 256;

/// The executable form of one recipe over `packets` consecutive source
/// packets of each of `rows` rows into segments of output (module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepackProgram {
    packet_bytes: u32,
    /// Source rows, and packets of each.
    rows: u32,
    packets: u32,
    /// Where each output segment starts in the output of a staged execution
    /// (the segments one after another), and that output's bytes.
    output_starts: Vec<u32>,
    output_bytes: u32,
    shapes: Vec<RepackShape>,
    ops: Vec<RepackOp>,
    groups: Vec<ValueGroup>,
    offsets: Vec<i32>,
    /// Bytes of the field slots (whole blocks) and of the value slots.
    field_bytes: u32,
    value_bytes: u32,
    /// Source bits into the field slots.
    fields: ByteMoves,
    /// Input bits into the output.
    output: ByteMoves,
}

/// The repeats covering `items`, a move each, in any order (they write
/// disjoint bytes): the items in order of their `shape` (a key equal for
/// moves equal but for where they move), a period of items recurs while
/// every item equals its first occurrence's moved by a multiple of one
/// [`Step`] (`step(first, later)`, `None` when no step makes them equal),
/// the period recurring most often taken; then a repeat recurs, in turn, as
/// later ones equal it moved by multiples of one step. `items` keeps one
/// period of each, which the repeats address.
fn repeats<T: Copy, K: Ord>(
    items: &mut Vec<T>,
    shape: impl Fn(&T) -> K,
    step: impl Fn(&T, &T) -> Option<Step>,
) -> Vec<Repeat> {
    let mut shaped = items
        .iter()
        .enumerate()
        .map(|(index, item)| (shape(item), index))
        .collect::<Vec<_>>();
    shaped.sort();
    *items = shaped.into_iter().map(|(_, index)| items[index]).collect();
    let count = items.len();
    let mut repeats: Vec<Repeat> = Vec::new();
    let mut first = 0;
    while first < count {
        let mut best = (1, 1, Step::default());
        for length in 1..=PERIOD.min((count - first) / 2) {
            let Some(once) = step(&items[first], &items[first + length]) else {
                continue;
            };
            let mut times = 1;
            while first + length * (times + 1) <= count
                && (0..length).all(|item| {
                    step(&items[first + item], &items[first + length * times + item])
                        == Some(once.times(times as u32))
                })
            {
                times += 1;
            }
            // The most repetitions, then the most moves: a short period
            // repeated many times before a long one repeated few.
            if times > 1 && (times, length * times) > (best.1, best.0 * best.1) {
                best = (length, times, once);
            }
        }
        let (length, times, once) = best;
        let repeat = Repeat {
            first: first as u32,
            length: length as u32,
            count: times as u32,
            step: once,
            outer: 1,
            outer_step: Step::default(),
        };
        first += length * times;
        // The repeat recurs as an earlier one's next outer time.
        let shape = |repeat: &Repeat| (repeat.length, repeat.count, repeat.step);
        let earlier = repeats.iter_mut().rev().find_map(|earlier| {
            if shape(earlier) != shape(&repeat) {
                return None;
            }
            let outer = (0..repeat.length as usize)
                .map(|item| {
                    step(
                        &items[earlier.first as usize + item],
                        &items[repeat.first as usize + item],
                    )
                })
                .reduce(|a, b| if a == b { a } else { None })
                .flatten()?;
            (earlier.outer == 1 || outer == earlier.outer_step.times(earlier.outer))
                .then_some((earlier, outer))
        });
        if let Some((earlier, outer)) = earlier {
            if earlier.outer == 1 {
                earlier.outer_step = outer;
            }
            earlier.outer += 1;
            continue;
        }
        repeats.push(repeat);
    }
    // One period of each repeat.
    let mut kept = Vec::new();
    for repeat in &mut repeats {
        let first = kept.len() as u32;
        kept.extend_from_slice(&items[repeat.first as usize..][..repeat.length as usize]);
        repeat.first = first;
    }
    *items = kept;
    repeats
}

/// The segment holding byte `byte` of a space whose segments start at
/// `starts` (the first at 0), and the byte within it.
fn segment(starts: &[u32], byte: u32) -> (u32, u32) {
    let segment = starts.partition_point(|start| *start <= byte) - 1;
    (segment as u32, byte - starts[segment])
}

/// ORs into `bytes`, the blocks of `TIMES` consecutive repetitions from
/// `origin` moving `step` each, the bits `terms` take as `taking`;
/// `SEGMENTED` when the step moves segments.
///
/// # Safety
/// As [`ByteMoves::execute`]'s.
#[inline(always)]
unsafe fn take<const TIMES: usize, const SEGMENTED: bool>(
    bytes: &mut [vector::Bytes; TIMES],
    terms: &[ByteTerm],
    inputs: &[*const u8],
    origin: Origin,
    step: Step,
    taking: Taking,
) {
    use vector::*;
    for term in terms {
        // SAFETY: (all below) the term's window lies in its segment, and
        // every operand holds a block.
        let indices = unsafe { load(term.indices.as_ptr()) };
        let shifts = unsafe { load(term.shifts.as_ptr().cast()) };
        let masks = unsafe { load(term.masks.as_ptr()) };
        let window = |origin: Origin| unsafe {
            inputs[term.segment as usize + origin.input_segment]
                .add(term.input as usize + origin.input)
        };
        let mut next = window(origin);
        for (time, bytes) in bytes.iter_mut().enumerate() {
            let window = if SEGMENTED {
                window(origin.then(step, time))
            } else {
                let window = next;
                // The next repetition's, past the allocation after the
                // last: wrapping, and read only when in it.
                next = next.wrapping_add(step.input as usize);
                window
            };
            // SAFETY: as this function's (the host has the features of
            // `vector`).
            let picked = unsafe { pick(load(window), indices) };
            // A block's terms have disjoint bits.
            *bytes = match taking {
                Taking::Whole => or(*bytes, picked),
                Taking::Masked => select(masks, picked, *bytes),
                // SAFETY: as above.
                Taking::Shifted => select(masks, unsafe { shift(picked, shifts) }, *bytes),
            };
        }
    }
}

impl ByteMoves {
    /// The moves writing every output bit `output` whose input bit is
    /// `inputs[output]` and zero every other; `inputs` covers whole bytes.
    /// The input's and the output's segments start at the bytes
    /// `input_starts` and `output_starts`; every output segment is whole
    /// blocks (a program's outputs, which `RepackProgram::compile` asserts,
    /// and its field slots, which it ends at a block).
    fn of(inputs: &[Option<u32>], input_starts: &[u32], output_starts: &[u32]) -> Self {
        let bytes = inputs.len() / 8;
        let ends = output_starts[1..]
            .iter()
            .map(|end| *end as usize)
            .chain([bytes])
            .collect::<Vec<_>>();
        // The input byte an output byte copies whole.
        let whole = |byte: usize| {
            let first = inputs[byte * 8]?;
            (first % 8 == 0 && (1..8).all(|bit| inputs[byte * 8 + bit] == Some(first + bit as u32)))
                .then_some(first / 8)
        };
        // Maximal spans of whole bytes consecutive on both sides, each in one
        // segment of each; those of a block or more are copied.
        let mut spans = Vec::new();
        let mut copied = vec![false; bytes];
        let mut byte = 0;
        while byte < bytes {
            let Some(input) = whole(byte) else {
                byte += 1;
                continue;
            };
            let (input_segment, input_byte) = segment(input_starts, input);
            let (output_segment, output_byte) = segment(output_starts, byte as u32);
            let mut end = byte + 1;
            while end < bytes
                && whole(end) == Some(input + (end - byte) as u32)
                && segment(input_starts, input + (end - byte) as u32).0 == input_segment
                && segment(output_starts, end as u32).0 == output_segment
            {
                end += 1;
            }
            if end - byte >= BLOCK {
                spans.push(ByteSpan {
                    input_segment,
                    input: input_byte,
                    output_segment,
                    output: output_byte,
                    bytes: (end - byte) as u32,
                });
                copied[byte..end].fill(true);
            }
            byte = end;
        }
        // Every other output byte's bits by (input byte, lane, shift), as
        // terms: each lane of a term takes one input byte of its window.
        let mut blocks = Vec::new();
        let mut terms: Vec<ByteTerm> = Vec::new();
        for (output_segment, (start, end)) in output_starts.iter().zip(&ends).enumerate() {
            for block in (*start as usize..*end).step_by(BLOCK) {
                // Each kind's pieces into terms of that kind: a term taking
                // fewer kinds of piece costs fewer operations.
                let mut pieces = std::collections::BTreeMap::<_, u8>::new();
                for byte in block..(block + BLOCK).min(*end) {
                    if copied[byte] {
                        continue;
                    }
                    for bit in 0..8 {
                        if let Some(input) = inputs[byte * 8 + bit] {
                            let shift = bit as i8 - (input % 8) as i8;
                            let (input_segment, input) = segment(input_starts, input / 8);
                            *pieces
                                .entry((input_segment, input, byte - block, shift))
                                .or_default() |= 1 << bit;
                        }
                    }
                }
                // A block of no bits is written (zero) unless spans copy all
                // of it.
                if pieces.is_empty() && copied[block..(block + BLOCK).min(*end)].iter().all(|c| *c)
                {
                    continue;
                }
                let mut pieces = pieces
                    .into_iter()
                    .map(|((segment, input, lane, shift), mask)| {
                        let taking = match (shift, mask) {
                            (0, 0xff) => Taking::Whole,
                            (0, _) => Taking::Masked,
                            _ => Taking::Shifted,
                        };
                        (taking, segment, input, lane, shift, mask)
                    })
                    .collect::<Vec<_>>();
                pieces.sort();
                let first = terms.len();
                let mut kinds = Vec::new();
                for (taking, segment, input, lane, shift, mask) in pieces {
                    let index = (first..terms.len()).find(|index| {
                        let term = &terms[*index];
                        kinds[index - first] == taking
                            && term.segment == segment
                            && term.indices[lane] == 0x80
                            && input - term.input < BLOCK as u32
                    });
                    let index = index.unwrap_or_else(|| {
                        terms.push(ByteTerm {
                            segment,
                            input,
                            indices: [0x80; BLOCK],
                            shifts: [0; BLOCK],
                            masks: [0; BLOCK],
                        });
                        kinds.push(taking);
                        terms.len() - 1
                    });
                    let term = &mut terms[index];
                    term.indices[lane] = (input - term.input) as u8;
                    term.shifts[lane] = shift;
                    term.masks[lane] = mask;
                }
                let of = |taking| kinds.iter().filter(|kind| **kind == taking).count() as u32;
                blocks.push(ByteBlock {
                    segment: output_segment as u32,
                    output: block as u32 - start,
                    terms: first as u32,
                    whole: of(Taking::Whole),
                    masked: of(Taking::Masked),
                    shifted: of(Taking::Shifted),
                });
            }
        }
        // A later move is an earlier one moved by a step whose segment
        // steps are not negative.
        let after = |first: u32, later: u32| later.checked_sub(first);
        let span_repeats = repeats(
            &mut spans,
            |span| span.bytes,
            |first, later| {
                (first.bytes == later.bytes)
                    .then(|| {
                        Some(Step {
                            input_segments: after(first.input_segment, later.input_segment)?,
                            input: after(first.input, later.input)?,
                            output_segments: after(first.output_segment, later.output_segment)?,
                            output: after(first.output, later.output)?,
                        })
                    })
                    .flatten()
            },
        );
        fn terms_of<'t>(terms: &'t [ByteTerm], block: &ByteBlock) -> &'t [ByteTerm] {
            &terms[block.terms as usize..][..(block.whole + block.masked + block.shifted) as usize]
        }
        // A block's shape: its kinds of term and its terms but for where
        // they read.
        let shape = |block: &ByteBlock| {
            let terms = terms_of(&terms, block);
            (
                (block.whole, block.masked, block.shifted),
                terms
                    .iter()
                    .map(|term| {
                        (
                            term.segment.wrapping_sub(terms[0].segment),
                            term.input.wrapping_sub(terms[0].input),
                            term.indices,
                            term.shifts,
                            term.masks,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let block_repeats = repeats(&mut blocks, shape, |first, later| {
            let kinds = |block: &ByteBlock| (block.whole, block.masked, block.shifted);
            let output = (
                after(first.segment, later.segment)?,
                after(first.output, later.output)?,
            );
            let (first_kinds, later_kinds) = (kinds(first), kinds(later));
            let (first, later) = (terms_of(&terms, first), terms_of(&terms, later));
            let input = match (first.first(), later.first()) {
                (Some(a), Some(b)) => (after(a.segment, b.segment)?, after(a.input, b.input)?),
                // Blocks of no terms read nothing.
                _ => (0, 0),
            };
            (first_kinds == later_kinds
                && first.iter().zip(later).all(|(a, b)| {
                    (after(a.segment, b.segment), after(a.input, b.input))
                        == (Some(input.0), Some(input.1))
                        && (a.indices, a.shifts, a.masks) == (b.indices, b.shifts, b.masks)
                }))
            .then_some(Step {
                input_segments: input.0,
                input: input.1,
                output_segments: output.0,
                output: output.1,
            })
        });
        // The terms of the blocks kept.
        let mut kept = Vec::new();
        for block in &mut blocks {
            let first = kept.len() as u32;
            kept.extend_from_slice(terms_of(&terms, block));
            block.terms = first;
        }
        let terms = kept;
        Self {
            spans,
            span_repeats,
            blocks,
            terms,
            block_repeats,
        }
    }

    /// Writes `count` repetitions of `blocks` from `origin`, moving `step`
    /// each: as many at a time as the registers hold the blocks of, then by
    /// halves, each term's operands loaded once for all.
    ///
    /// # Safety
    /// As [`execute`](Self::execute)'s.
    #[inline(always)]
    unsafe fn write_repeat<const SEGMENTED: bool>(
        &self,
        blocks: &[ByteBlock],
        inputs: &[*const u8],
        outputs: &[*mut u8],
        origin: Origin,
        count: u32,
        step: Step,
    ) {
        use vector::TOGETHER;
        let count = count as usize;
        let mut time = 0;
        // SAFETY: (all below) as this function's.
        while count - time >= TOGETHER {
            let origin = origin.then(step, time);
            unsafe {
                self.write_blocks::<TOGETHER, SEGMENTED>(blocks, inputs, outputs, origin, step)
            };
            time += TOGETHER;
        }
        if TOGETHER > 8 && count - time >= 8 {
            let origin = origin.then(step, time);
            unsafe { self.write_blocks::<8, SEGMENTED>(blocks, inputs, outputs, origin, step) };
            time += 8;
        }
        if count - time >= 4 {
            let origin = origin.then(step, time);
            unsafe { self.write_blocks::<4, SEGMENTED>(blocks, inputs, outputs, origin, step) };
            time += 4;
        }
        if count - time >= 2 {
            let origin = origin.then(step, time);
            unsafe { self.write_blocks::<2, SEGMENTED>(blocks, inputs, outputs, origin, step) };
            time += 2;
        }
        if count > time {
            let origin = origin.then(step, time);
            unsafe { self.write_blocks::<1, SEGMENTED>(blocks, inputs, outputs, origin, step) };
        }
    }

    /// Writes `blocks` of `TIMES` consecutive repetitions from `origin`
    /// moving `step` each; `SEGMENTED` when the step moves segments.
    ///
    /// # Safety
    /// As [`execute`](Self::execute)'s.
    #[inline(always)]
    unsafe fn write_blocks<const TIMES: usize, const SEGMENTED: bool>(
        &self,
        blocks: &[ByteBlock],
        inputs: &[*const u8],
        outputs: &[*mut u8],
        origin: Origin,
        step: Step,
    ) {
        use vector::*;
        for block in blocks {
            let mut bytes = [zero(); TIMES];
            let terms = &self.terms[block.terms as usize..];
            let (whole, terms) = terms.split_at(block.whole as usize);
            let (masked, terms) = terms.split_at(block.masked as usize);
            let shifted = &terms[..block.shifted as usize];
            // SAFETY: as this function's.
            unsafe {
                let take = take::<TIMES, SEGMENTED>;
                take(&mut bytes, whole, inputs, origin, step, Taking::Whole);
                take(&mut bytes, masked, inputs, origin, step, Taking::Masked);
                take(&mut bytes, shifted, inputs, origin, step, Taking::Shifted);
            }
            // SAFETY: (all below) the block lies in its segment.
            let output = |origin: Origin| unsafe {
                outputs[block.segment as usize + origin.output_segment]
                    .add(block.output as usize + origin.output)
            };
            let mut next = output(origin);
            for (time, bytes) in bytes.into_iter().enumerate() {
                let output = if SEGMENTED {
                    output(origin.then(step, time))
                } else {
                    let output = next;
                    // As `take`'s next window.
                    next = next.wrapping_add(step.output as usize);
                    output
                };
                unsafe { store(output, bytes) };
            }
        }
    }

    /// Writes the output segments at `outputs` from the input segments at
    /// `inputs`.
    ///
    /// # Safety
    /// Each input segment's bytes the moves read, and [`BLOCK`] bytes past
    /// them, are readable at its base; each output segment's bytes the moves
    /// write are writable at its base; the host has the features of
    /// [`vector`].
    #[inline(always)]
    unsafe fn execute(&self, inputs: &[*const u8], outputs: &[*mut u8]) {
        for repeat in &self.block_repeats {
            let blocks = &self.blocks[repeat.first as usize..][..repeat.length as usize];
            for outer in 0..repeat.outer as usize {
                let origin = Origin::of(repeat.outer_step, outer);
                // SAFETY: as this function's.
                unsafe {
                    if repeat.step.segmented() {
                        self.write_repeat::<true>(
                            blocks,
                            inputs,
                            outputs,
                            origin,
                            repeat.count,
                            repeat.step,
                        )
                    } else {
                        self.write_repeat::<false>(
                            blocks,
                            inputs,
                            outputs,
                            origin,
                            repeat.count,
                            repeat.step,
                        )
                    }
                }
            }
        }
        for repeat in &self.span_repeats {
            let spans = &self.spans[repeat.first as usize..][..repeat.length as usize];
            for outer in 0..repeat.outer as usize {
                let origin = Origin::of(repeat.outer_step, outer);
                for span in spans {
                    let (count, step) = (repeat.count as usize, repeat.step);
                    // SAFETY: as this function's.
                    unsafe {
                        // A span of up to four blocks is copied by as many
                        // loads and stores, unrolled.
                        match span.bytes.div_ceil(BLOCK as u32) {
                            1 => copy_spans::<1>(span, inputs, outputs, origin, count, step),
                            2 => copy_spans::<2>(span, inputs, outputs, origin, count, step),
                            3 => copy_spans::<3>(span, inputs, outputs, origin, count, step),
                            4 => copy_spans::<4>(span, inputs, outputs, origin, count, step),
                            _ => copy_spans::<0>(span, inputs, outputs, origin, count, step),
                        }
                    }
                }
            }
        }
    }
}

/// Copies `count` repetitions of `span` from `origin`, moving `step` each,
/// as [`copy_span`] copies `BLOCKS` blocks.
///
/// # Safety
/// As [`ByteMoves::execute`]'s.
#[inline(always)]
unsafe fn copy_spans<const BLOCKS: usize>(
    span: &ByteSpan,
    inputs: &[*const u8],
    outputs: &[*mut u8],
    origin: Origin,
    count: usize,
    step: Step,
) {
    let bytes = span.bytes as usize;
    // SAFETY: (all below) every repetition's bytes lie in its segments.
    unsafe {
        if step.segmented() {
            for time in 0..count {
                let (input, output) = span_at(span, inputs, outputs, origin.then(step, time));
                copy_span::<BLOCKS>(input, output, bytes);
            }
        } else {
            let (mut input, mut output) = span_at(span, inputs, outputs, origin);
            for _ in 0..count {
                copy_span::<BLOCKS>(input, output, bytes);
                // As `take`'s next window.
                input = input.wrapping_add(step.input as usize);
                output = output.wrapping_add(step.output as usize);
            }
        }
    }
}

/// The first input and output byte of `span` moved to `origin`.
///
/// # Safety
/// The span's bytes lie in its segments.
#[inline(always)]
unsafe fn span_at(
    span: &ByteSpan,
    inputs: &[*const u8],
    outputs: &[*mut u8],
    origin: Origin,
) -> (*const u8, *mut u8) {
    unsafe {
        (
            inputs[span.input_segment as usize + origin.input_segment]
                .add(span.input as usize + origin.input),
            outputs[span.output_segment as usize + origin.output_segment]
                .add(span.output as usize + origin.output),
        )
    }
}

/// Copies the `bytes` (at least [`BLOCK`]) bytes at `input` to `output`:
/// as `BLOCKS` blocks, the last ending at the span's end, when `bytes` takes
/// that many; any length four blocks at a time when `BLOCKS` is 0.
///
/// # Safety
/// The bytes are readable at `input` and writable at `output`; the host has
/// the features of [`vector`].
#[inline(always)]
unsafe fn copy_span<const BLOCKS: usize>(input: *const u8, output: *mut u8, bytes: usize) {
    use vector::*;
    let last = bytes - BLOCK;
    // SAFETY: (all below) every block lies in the span.
    unsafe {
        if BLOCKS > 0 {
            let at = |block: usize| {
                if block + 1 == BLOCKS {
                    last
                } else {
                    block * BLOCK
                }
            };
            let blocks: [Bytes; BLOCKS] = std::array::from_fn(|block| load(input.add(at(block))));
            for (block, bytes) in blocks.into_iter().enumerate() {
                store(output.add(at(block)), bytes);
            }
            return;
        }
        let mut at = 0;
        while at + 4 * BLOCK <= last {
            let bytes = [0, 1, 2, 3].map(|block| load(input.add(at + block * BLOCK)));
            for (block, bytes) in bytes.into_iter().enumerate() {
                store(output.add(at + block * BLOCK), bytes);
            }
            at += 4 * BLOCK;
        }
        while at < last {
            store(output.add(at), load(input.add(at)));
            at += BLOCK;
        }
        store(output.add(last), load(input.add(last)));
    }
}

/// The [`BLOCK`]-byte vector operations of the executor, inlined into its
/// `#[target_feature]` boundary: NEON, or SSSE3 on x86-64.
mod vector {
    #[cfg(target_arch = "aarch64")]
    pub use neon::*;
    #[cfg(target_arch = "x86_64")]
    pub use ssse3::*;

    #[cfg(target_arch = "aarch64")]
    mod neon {
        use std::arch::aarch64::*;

        pub type Bytes = uint8x16_t;

        /// Blocks built at once: half the 32 registers.
        pub const TOGETHER: usize = 16;

        /// # Safety
        /// `at` holds 16 bytes.
        #[inline(always)]
        pub unsafe fn load(at: *const u8) -> Bytes {
            unsafe { vld1q_u8(at) }
        }

        /// # Safety
        /// `at` holds 16 bytes.
        #[inline(always)]
        pub unsafe fn store(at: *mut u8, bytes: Bytes) {
            unsafe { vst1q_u8(at, bytes) }
        }

        #[inline(always)]
        pub fn zero() -> Bytes {
            unsafe { vdupq_n_u8(0) }
        }

        #[inline(always)]
        pub fn or(a: Bytes, b: Bytes) -> Bytes {
            unsafe { vorrq_u8(a, b) }
        }

        /// The bits of `a` under `mask`, of `b` elsewhere.
        #[inline(always)]
        pub fn select(mask: Bytes, a: Bytes, b: Bytes) -> Bytes {
            unsafe { vbslq_u8(mask, a, b) }
        }

        /// Lane `l` is `table[indices[l]]`, zero for an index with its high
        /// bit set.
        ///
        /// # Safety
        /// None on aarch64, whose every target has NEON (the SSSE3 form's
        /// condition).
        #[inline(always)]
        pub unsafe fn pick(table: Bytes, indices: Bytes) -> Bytes {
            unsafe { vqtbl1q_u8(table, indices) }
        }

        /// Lane `l` shifted left by the signed byte `shifts[l]` (right when
        /// negative; below 8 either way).
        ///
        /// # Safety
        /// As [`pick`]'s.
        #[inline(always)]
        pub unsafe fn shift(bytes: Bytes, shifts: Bytes) -> Bytes {
            unsafe { vshlq_u8(bytes, vreinterpretq_s8_u8(shifts)) }
        }

        /// Each word's binary16 widened in place to binary32 bits, as
        /// `registry::f16_to_f32` widens it.
        #[inline(always)]
        pub fn widen_halves(halves: &mut [u32; 4]) {
            unsafe {
                let words = vld1q_u32(halves.as_ptr());
                let magnitude = vshlq_n_u32::<13>(vandq_u32(words, vdupq_n_u32(0x7fff)));
                let exponent = vandq_u32(magnitude, vdupq_n_u32(0x0f80_0000));
                let normal = vaddq_u32(magnitude, vdupq_n_u32(0x3800_0000));
                let special = vaddq_u32(normal, vdupq_n_u32(0x3800_0000));
                let subnormal = vreinterpretq_u32_f32(vsubq_f32(
                    vreinterpretq_f32_u32(vaddq_u32(normal, vdupq_n_u32(0x0080_0000))),
                    vreinterpretq_f32_u32(vdupq_n_u32(0x3880_0000)),
                ));
                let bits = vbslq_u32(
                    vceqq_u32(exponent, vdupq_n_u32(0x0f80_0000)),
                    special,
                    vbslq_u32(vceqzq_u32(exponent), subnormal, normal),
                );
                let sign = vshlq_n_u32::<16>(vandq_u32(words, vdupq_n_u32(0x8000)));
                vst1q_u32(halves.as_mut_ptr(), vorrq_u32(bits, sign));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod ssse3 {
        use std::arch::x86_64::*;

        pub type Bytes = __m128i;

        /// Blocks built at once: half the 16 registers.
        pub const TOGETHER: usize = 8;

        /// # Safety
        /// `at` holds 16 bytes.
        #[inline(always)]
        pub unsafe fn load(at: *const u8) -> Bytes {
            unsafe { _mm_loadu_si128(at.cast()) }
        }

        /// # Safety
        /// `at` holds 16 bytes.
        #[inline(always)]
        pub unsafe fn store(at: *mut u8, bytes: Bytes) {
            unsafe { _mm_storeu_si128(at.cast(), bytes) }
        }

        #[inline(always)]
        pub fn zero() -> Bytes {
            unsafe { _mm_setzero_si128() }
        }

        #[inline(always)]
        pub fn or(a: Bytes, b: Bytes) -> Bytes {
            unsafe { _mm_or_si128(a, b) }
        }

        /// The bits of `a` under `mask`, of `b` elsewhere.
        #[inline(always)]
        pub fn select(mask: Bytes, a: Bytes, b: Bytes) -> Bytes {
            unsafe { _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b)) }
        }

        /// Lane `l` is `table[indices[l]]`, zero for an index with its high
        /// bit set.
        ///
        /// # Safety
        /// The host has SSSE3.
        #[inline(always)]
        pub unsafe fn pick(table: Bytes, indices: Bytes) -> Bytes {
            unsafe { _mm_shuffle_epi8(table, indices) }
        }

        /// Lane `l` shifted left by the signed byte `shifts[l]` (right when
        /// negative; below 8 either way): bits 8 to 15 of the lane times
        /// `2^(8 + shift)`, the even and odd lanes of each 16-bit lane
        /// multiplied apart.
        ///
        /// # Safety
        /// The host has SSSE3.
        #[inline(always)]
        pub unsafe fn shift(bytes: Bytes, shifts: Bytes) -> Bytes {
            unsafe {
                // 2^t for t = shift + 8 in 1..16, as its low and high byte.
                let t = _mm_add_epi8(shifts, _mm_set1_epi8(8));
                let low = _mm_shuffle_epi8(
                    _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0),
                    t,
                );
                let high = _mm_shuffle_epi8(
                    _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, -128),
                    t,
                );
                let even_bytes = _mm_set1_epi16(0x00ff);
                let even_factor =
                    _mm_or_si128(_mm_and_si128(low, even_bytes), _mm_slli_epi16::<8>(high));
                let odd_factor =
                    _mm_or_si128(_mm_srli_epi16::<8>(low), _mm_andnot_si128(even_bytes, high));
                let even = _mm_srli_epi16::<8>(_mm_mullo_epi16(
                    _mm_and_si128(bytes, even_bytes),
                    even_factor,
                ));
                let odd = _mm_andnot_si128(
                    even_bytes,
                    _mm_mullo_epi16(_mm_srli_epi16::<8>(bytes), odd_factor),
                );
                _mm_or_si128(even, odd)
            }
        }

        /// Each word's binary16 widened in place to binary32 bits, as
        /// `registry::f16_to_f32` widens it.
        #[inline(always)]
        pub fn widen_halves(halves: &mut [u32; 4]) {
            unsafe {
                let word = |bits: u32| _mm_set1_epi32(bits as i32);
                let words = _mm_loadu_si128(halves.as_ptr().cast());
                let magnitude = _mm_slli_epi32::<13>(_mm_and_si128(words, word(0x7fff)));
                let exponent = _mm_and_si128(magnitude, word(0x0f80_0000));
                let normal = _mm_add_epi32(magnitude, word(0x3800_0000));
                let special = _mm_add_epi32(normal, word(0x3800_0000));
                let subnormal = _mm_castps_si128(_mm_sub_ps(
                    _mm_castsi128_ps(_mm_add_epi32(normal, word(0x0080_0000))),
                    _mm_castsi128_ps(word(0x3880_0000)),
                ));
                let bits = select(
                    _mm_cmpeq_epi32(exponent, word(0x0f80_0000)),
                    special,
                    select(
                        _mm_cmpeq_epi32(exponent, _mm_setzero_si128()),
                        subnormal,
                        normal,
                    ),
                );
                let sign = _mm_slli_epi32::<16>(_mm_and_si128(words, word(0x8000)));
                _mm_storeu_si128(halves.as_mut_ptr().cast(), _mm_or_si128(bits, sign));
            }
        }
    }
}

impl RepackProgram {
    /// The program of `conversion`'s recipe over `packets` consecutive
    /// source packets of each of `rows` rows (an input segment each),
    /// writing output segments of `outputs` bytes, each whole blocks: bit
    /// `bit` of plane `plane` of the packet form of packet `p` of row `r`
    /// goes to bit `b` of output segment `s` for
    /// `destination(r, p, plane, bit) = Some((s, b))`, or nowhere for
    /// `None`. Output bits nothing maps to are zero.
    pub fn compile(
        conversion: &RepresentationConversion,
        rows: u32,
        packets: u32,
        outputs: &[u32],
        destination: impl Fn(u32, u32, usize, u32) -> Option<(usize, u32)>,
    ) -> Self {
        assert!(
            outputs
                .iter()
                .all(|bytes| (*bytes as usize).is_multiple_of(BLOCK)),
            "a repack output's segments are whole blocks"
        );
        let output_starts = outputs
            .iter()
            .scan(0, |start, bytes| {
                *start += bytes;
                Some(*start - bytes)
            })
            .collect::<Vec<_>>();
        let (recipe, source) = (&conversion.recipe, conversion.source_layout);
        let packet = conversion.destination_layout.packet();
        let (row_packets, packets) = (packets, rows * packets);
        let packet_bits = source.packet_size * 8;
        let values = PacketValues::of(recipe, packet);
        let (shapes, computed) = (values.shapes(), values.computed());
        // One group per shape, over every packet, its values' slots as
        // wide as the widest needs; the input bit of computed value `v` of
        // packet `p` is `slots[p * computed values + v]`, past the source
        // packets.
        let mut groups: Vec<ValueGroup> = Vec::new();
        let mut offsets = Vec::new();
        let mut members = Vec::new();
        let (mut field_bytes, mut value_bytes) = (0u32, 0u32);
        for (index, shape) in shapes.iter().enumerate() {
            let shape_members = (0..packets)
                .flat_map(|p| (0..computed.len()).map(move |v| (p, v)))
                .filter(|(_, v)| computed[*v].shape == index)
                .collect::<Vec<_>>();
            let count = shape_members.len() as u32;
            let bytes = slot(
                shape_members
                    .iter()
                    .map(|(_, v)| computed[*v].width)
                    .max()
                    .expect("a shape has values"),
            );
            groups.push(ValueGroup {
                shape: index as u32,
                count,
                fields: field_bytes,
                offsets: offsets.len() as u32,
                values: value_bytes,
                bytes,
            });
            for offset in 0..shape.offsets as usize {
                offsets.extend(
                    shape_members
                        .iter()
                        .map(|(_, v)| computed[*v].offsets[offset]),
                );
            }
            field_bytes += count * shape.fields;
            value_bytes += count * bytes;
            members.push(shape_members);
        }
        // The field slots end at a block: their move writes whole blocks.
        let field_bytes = field_bytes.next_multiple_of(BLOCK as u32);
        let values_at = source.packet_size * packets * 8;
        let mut slots = vec![0u32; computed.len() * packets as usize];
        let mut field_inputs: Vec<Option<u32>> = vec![None; field_bytes as usize * 8];
        for (group, members) in groups.iter().zip(&members) {
            for (lane, (p, v)) in members.iter().enumerate() {
                let lane = lane as u32;
                slots[*p as usize * computed.len() + v] =
                    values_at + (group.values + lane * group.bytes) * 8;
                // Each field's slot column, as its shape's `Field` op places
                // it.
                let shape = shapes[computed[*v].shape];
                let columns = values.ops()[shape.ops as usize..][..shape.length as usize]
                    .iter()
                    .filter_map(|op| match op {
                        RepackOp::Field { prefix, width } => Some((*prefix, slot(*width))),
                        _ => None,
                    });
                for ((prefix, bytes), field) in columns.zip(&computed[*v].fields) {
                    let at = (group.fields + prefix * group.count + lane * bytes) * 8;
                    for (offset, bit) in field {
                        field_inputs[(at + offset) as usize] = Some(p * packet_bits + bit);
                    }
                }
            }
        }
        // The input bit of every output bit.
        let output_bits = outputs.iter().sum::<u32>() as usize * 8;
        let mut inputs: Vec<Option<u32>> = vec![None; output_bits];
        let mut claimed = vec![false; output_bits];
        for p in 0..packets {
            for (index, plane) in packet.planes.iter().enumerate() {
                for bit in 0..plane.bytes_per_group * 8 {
                    let input = values.bit(index, bit).map(|input| match input {
                        PacketBit::Source(bit) => p * packet_bits + bit,
                        PacketBit::Value { value, bit } => {
                            slots[p as usize * computed.len() + value] + bit
                        }
                    });
                    if let Some((segment, output)) =
                        destination(p / row_packets, p % row_packets, index, bit)
                    {
                        assert!(
                            output < outputs[segment] * 8,
                            "a packet bit maps past its output segment"
                        );
                        let output = (output_starts[segment] * 8 + output) as usize;
                        assert!(
                            !claimed[output],
                            "two packet bits map to output bit {output}"
                        );
                        claimed[output] = true;
                        inputs[output] = input;
                    }
                }
            }
        }
        // The input segments: each row's source packets, then the value
        // slots.
        let input_starts = (0..rows)
            .map(|row| row * row_packets * source.packet_size)
            .chain([packets * source.packet_size])
            .collect::<Vec<_>>();
        let output = ByteMoves::of(&inputs, &input_starts, &output_starts);
        Self {
            packet_bytes: source.packet_size,
            rows,
            packets: row_packets,
            output_starts,
            output_bytes: outputs.iter().sum(),
            shapes: values.shapes().to_vec(),
            ops: values.ops().to_vec(),
            groups,
            offsets,
            field_bytes,
            value_bytes,
            fields: ByteMoves::of(&field_inputs, &input_starts, &[0]),
            output,
        }
    }

    /// Bytes of one source packet.
    fn packet_bytes(&self) -> usize {
        self.packet_bytes as usize
    }

    /// Source rows one execution reads.
    fn rows(&self) -> usize {
        self.rows as usize
    }

    /// Source packets of each row one execution reads.
    fn packets(&self) -> usize {
        self.packets as usize
    }

    /// Where each output segment starts in the output of a staged
    /// execution, the segments one after another.
    pub fn output_starts(&self) -> &[u32] {
        &self.output_starts
    }

    fn output_bytes(&self) -> usize {
        self.output_bytes as usize
    }

    fn field_bytes(&self) -> usize {
        self.field_bytes as usize
    }

    fn value_bytes(&self) -> usize {
        self.value_bytes as usize
    }

    /// Bytes of one row's source packets.
    fn row_bytes(&self) -> usize {
        self.packet_bytes() * self.packets()
    }

    /// An executor of this program, holding its working memory.
    pub fn executor(&self) -> RepackExecutor<'_> {
        #[cfg(target_arch = "x86_64")]
        assert!(
            std::arch::is_x86_feature_detected!("ssse3"),
            "the repack executor needs SSSE3"
        );
        RepackExecutor {
            program: self,
            // A move reads a block from any source byte.
            staging: vec![0u8; self.rows() * self.row_bytes() + BLOCK],
            slots: vec![0u8; self.value_bytes() + PADDING],
            fields: vec![0; self.field_bytes() + PADDING],
            output: vec![0u8; self.output_bytes()],
            inputs: vec![std::ptr::null(); self.rows() + 1],
            staged_sources: Vec::with_capacity(self.rows()),
            staged_outputs: Vec::with_capacity(self.output_starts.len()),
            prepared: self
                .ops
                .iter()
                .map(|op| match op {
                    RepackOp::Lookup { table } => {
                        table.bytes().map_or(Prepared::Nothing, Prepared::Entries)
                    }
                    RepackOp::ScaleF32 { exponent } => Prepared::Factor(2f32.powi(*exponent)),
                    _ => Prepared::Nothing,
                })
                .collect(),
            bytewise: self
                .shapes
                .iter()
                .zip(&self.groups)
                .map(|(shape, group)| {
                    group.bytes == 1
                        && self.ops[shape.ops as usize..][..shape.length as usize]
                            .iter()
                            .all(|op| match op {
                                RepackOp::Field { width, .. } => slot(*width) == 1,
                                RepackOp::ShiftLeft { .. }
                                | RepackOp::Or
                                | RepackOp::Offset { .. }
                                | RepackOp::NegateIf => true,
                                RepackOp::Lookup { table } => table.values().len() <= 256,
                                RepackOp::F16ToF32
                                | RepackOp::I32ToF32
                                | RepackOp::MultiplyF32
                                | RepackOp::ScaleF32 { .. } => false,
                            })
                })
                .collect(),
            words: [[0; LANES]; STACK],
            bytes: [[0; LANES]; STACK],
        }
    }
}

/// A recipe's program writing the packet forms (alignment padding zero) of
/// consecutive source packets, one after another.
pub struct PacketForms {
    program: RepackProgram,
}

impl PacketForms {
    /// The program of `conversion`'s recipe over the packets of one execution:
    /// at least [`EXECUTION_BYTES`] source bytes, and whole blocks of packet
    /// forms.
    pub fn compile(conversion: &RepresentationConversion) -> Self {
        let packet = conversion.destination_layout.packet();
        let least = EXECUTION_BYTES.div_ceil(conversion.source_layout.packet_size as usize) as u32;
        let form = packet.packet_size;
        let packets = (1..)
            .map(|times| times * least)
            .find(|packets| ((packets * form) as usize).is_multiple_of(BLOCK))
            .expect("some multiple of an execution is whole blocks");
        Self {
            program: RepackProgram::compile(
                conversion,
                1,
                packets,
                &[packets * form],
                |_, p, plane, bit| Some((0, (p * form + packet.planes[plane].offset) * 8 + bit)),
            ),
        }
    }

    /// The packet forms of the source packets `source`, across threads when
    /// there are many.
    pub fn execute(&self, source: &[u8]) -> Vec<u8> {
        let program = &self.program;
        let (packet_bytes, execution) = (program.packet_bytes(), program.row_bytes());
        assert_eq!(
            source.len() % packet_bytes,
            0,
            "a repack source is whole packets"
        );
        let produced = program.output_bytes();
        let form = produced / program.packets();
        let mut output = vec![0u8; source.len() / packet_bytes * form];
        let executions = source.len().div_ceil(execution);
        /// Source bytes each thread converts at least. Measured (whole
        /// tensors of 64 KiB to 16 MiB of `q8_0`, `q4_k`, `iq4_xs` and
        /// `q3_k` packets into their packet forms, M4 Max): a quarter or
        /// half of it speeds tensors of 1 MiB (`q3_k` 730 to 270 µs at a
        /// quarter) but slows the fastest of 4 MiB (`q8_0` 215 to 300 µs at
        /// a quarter, 245 µs at half); four times it slows every tensor of
        /// 4 MiB or more (`q3_k` 1.0 to 3.0 ms at 4 MiB).
        const BYTES_PER_THREAD: usize = 1 << 20;
        let threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(source.len().div_ceil(BYTES_PER_THREAD))
            .max(1);
        let chunk = executions.div_ceil(threads).max(1);
        // Executions `first..` writing `output` from its first: in place when
        // the source holds every byte an execution may read, else staged.
        let run = |first: usize, output: &mut [u8]| {
            let mut executor = program.executor();
            for (index, out) in output.chunks_mut(produced).enumerate() {
                let start = (first + index) * execution;
                if start + execution + BLOCK <= source.len() {
                    // SAFETY: the execution's source and a block past it lie
                    // in `source`, and its output in `out`.
                    unsafe {
                        executor.run_into(&[source.as_ptr().add(start)], &[out.as_mut_ptr()])
                    };
                } else {
                    // Past the packets, the staging holds what an earlier
                    // execution left: packets convert apart, and the forms
                    // past them are not kept.
                    let packets = &source[start..(start + execution).min(source.len())];
                    executor.source()[..packets.len()].copy_from_slice(packets);
                    out.copy_from_slice(&executor.run()[..out.len()]);
                }
            }
        };
        std::thread::scope(|scope| {
            for (index, output) in output.chunks_mut(chunk * produced).enumerate() {
                scope.spawn(move || run(index * chunk, output));
            }
        });
        output
    }
}

/// Executes one [`RepackProgram`].
pub struct RepackExecutor<'a> {
    program: &'a RepackProgram,
    /// The source rows of a staged execution, one after another, then a
    /// block; the value slots, then [`PADDING`] bytes.
    staging: Vec<u8>,
    slots: Vec<u8>,
    /// The field slots, to whole blocks, then [`PADDING`] bytes; a staged
    /// execution's output.
    fields: Vec<u8>,
    output: Vec<u8>,
    /// The bases of the input segments of the execution running (its
    /// source rows', then the value slots'), and of a staged execution's
    /// source rows and output segments: each set afresh for its execution
    /// (the buffers are only their storage).
    inputs: Vec<*const u8>,
    staged_sources: Vec<*const u8>,
    staged_outputs: Vec<*mut u8>,
    /// Per op, what it takes that the executor derives once.
    prepared: Vec<Prepared>,
    /// Per shape, whether its values evaluate in byte lanes: their slots
    /// are bytes, and so is every field, and the low byte of every op's
    /// result depends only on its operands' (no `f32` op, and a lookup's
    /// index, within a table of at most 256 entries, is its low byte).
    bytewise: Vec<bool>,
    /// The postfix stacks of word and of byte lanes.
    words: [[u32; LANES]; STACK],
    bytes: [[u8; LANES]; STACK],
}

/// What an op takes that the executor derives once.
#[derive(Clone, Copy, Debug)]
enum Prepared {
    Nothing,
    /// A lookup's table, looked up by a vector table instruction.
    Entries([u8; BLOCK]),
    /// A scale's factor.
    Factor(f32),
}

impl RepackExecutor<'_> {
    /// The source rows the next [`run`](Self::run) reads, one after another,
    /// as the last one left them: the caller writes them.
    pub fn source(&mut self) -> &mut [u8] {
        let bytes = self.program.rows() * self.program.row_bytes();
        &mut self.staging[..bytes]
    }

    /// The program's output of the staged source rows, its segments one
    /// after another: every byte no move writes is zero.
    pub fn run(&mut self) -> &[u8] {
        let (mut sources, mut outputs) = (
            std::mem::take(&mut self.staged_sources),
            std::mem::take(&mut self.staged_outputs),
        );
        // The bases derive from the buffers' own pointers (`Vec::as_ptr`,
        // `Vec::as_mut_ptr`, which make no reference to their elements), for
        // this run alone: no reference to `staging` or `output` is made
        // while the run reads and writes through them.
        let (staging, output) = (self.staging.as_ptr(), self.output.as_mut_ptr());
        let row_bytes = self.program.row_bytes();
        sources.clear();
        sources.extend((0..self.program.rows()).map(|row| staging.wrapping_add(row * row_bytes)));
        outputs.clear();
        outputs.extend(
            self.program
                .output_starts
                .iter()
                .map(|start| output.wrapping_add(*start as usize)),
        );
        // SAFETY: `staging` holds the rows and a block past each, and
        // `output` the segments, each apart from the others; neither is
        // otherwise referenced during the run (above).
        unsafe { self.run_into(&sources, &outputs) };
        (self.staged_sources, self.staged_outputs) = (sources, outputs);
        &self.output
    }

    /// Runs the program over the source rows at `sources` into the output
    /// segments at `outputs`, writing every byte of each segment.
    ///
    /// # Safety
    /// Each source row's bytes and a [`BLOCK`] past them are readable, and
    /// each output segment's bytes writable and no other reference's.
    pub unsafe fn run_into(&mut self, sources: &[*const u8], outputs: &[*mut u8]) {
        assert_eq!(sources.len(), self.program.rows(), "a source base per row");
        assert_eq!(
            outputs.len(),
            self.program.output_starts.len(),
            "an output base per segment"
        );
        // The value slots' base is the buffer's own pointer (`Vec::as_ptr`):
        // the references through which the evaluation then writes the slots
        // derive from it, so reading through it afterwards stays valid.
        let (rows, slots) = self.inputs.split_at_mut(sources.len());
        rows.copy_from_slice(sources);
        slots[0] = self.slots.as_ptr();
        #[cfg(target_arch = "aarch64")]
        // SAFETY: every aarch64 target has NEON, and as this function's.
        unsafe {
            self.evaluate(outputs)
        };
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `executor` found SSSE3, and as this function's.
        unsafe {
            evaluate_ssse3(self, outputs)
        };
    }

    /// # Safety
    /// As [`run_into`](Self::run_into)'s, with `inputs` the bases of the
    /// input segments; the host has the features of [`vector`].
    #[inline(always)]
    unsafe fn evaluate(&mut self, outputs: &[*mut u8]) {
        let program = self.program;
        // SAFETY: the fields lie in `fields`; as this function's.
        unsafe {
            program
                .fields
                .execute(&self.inputs, &[self.fields.as_mut_ptr()])
        };
        for group in &program.groups {
            let count = group.count as usize;
            for start in (0..count).step_by(LANES) {
                let lanes = (count - start).min(LANES);
                if self.bytewise[group.shape as usize] {
                    // SAFETY: the host has the features of `vector` (this
                    // function's condition).
                    unsafe { self.evaluate_bytes(group, start, lanes) };
                } else {
                    self.evaluate_words(group, start, lanes);
                }
            }
        }
        // SAFETY: as this function's.
        unsafe { program.output.execute(&self.inputs, outputs) };
    }

    /// Evaluates values `start..start + lanes` of `group` in byte lanes
    /// into their slots.
    ///
    /// # Safety
    /// The host has the features of [`vector`].
    #[inline(always)]
    unsafe fn evaluate_bytes(&mut self, group: &ValueGroup, start: usize, lanes: usize) {
        let count = group.count as usize;
        let mut height = 0;
        for (op, prepared) in shape_ops(self.program, &self.prepared, group) {
            let stack = &mut self.bytes;
            match op {
                RepackOp::Field { prefix, .. } => {
                    let at = group.fields as usize + prefix as usize * count + start;
                    copy_lanes(&mut stack[height], &self.fields[at..], lanes);
                    height += 1;
                }
                RepackOp::ShiftLeft { bits } => {
                    for value in &mut stack[height - 1][..lanes] {
                        *value = (u32::from(*value) << bits) as u8;
                    }
                }
                RepackOp::Or => {
                    height -= 1;
                    let (low, high) = stack.split_at_mut(height);
                    for (value, other) in low[height - 1][..lanes].iter_mut().zip(&high[0]) {
                        *value |= other;
                    }
                }
                RepackOp::NegateIf => {
                    height -= 1;
                    let (low, high) = stack.split_at_mut(height);
                    for (value, sign) in low[height - 1][..lanes].iter_mut().zip(&high[0]) {
                        // Exact on bytes: xor and a wrapping add commute
                        // with taking the low byte.
                        *value = negate_if(u32::from(*value), u32::from(*sign)) as u8;
                    }
                }
                RepackOp::Offset { offset } => {
                    let at = group.offsets as usize + offset as usize * count + start;
                    let offsets = &self.program.offsets[at..at + lanes];
                    for (value, offset) in stack[height - 1][..lanes].iter_mut().zip(offsets) {
                        *value = value.wrapping_add(*offset as u8);
                    }
                }
                RepackOp::Lookup { table } => match prepared {
                    // A block of lanes at a time; a lane past `lanes`
                    // looks up whatever it holds.
                    Prepared::Entries(entries) => {
                        use vector::*;
                        // SAFETY: `entries` and each column hold a block;
                        // the host has the features of `vector` (this
                        // function's condition).
                        let entries = unsafe { load(entries.as_ptr()) };
                        for column in stack[height - 1]
                            .chunks_exact_mut(BLOCK)
                            .take(lanes.div_ceil(BLOCK))
                        {
                            unsafe {
                                store(column.as_mut_ptr(), pick(entries, load(column.as_ptr())))
                            };
                        }
                    }
                    _ => {
                        let values = table.values();
                        for value in &mut stack[height - 1][..lanes] {
                            *value = values[usize::from(*value)] as u8;
                        }
                    }
                },
                RepackOp::F16ToF32
                | RepackOp::I32ToF32
                | RepackOp::MultiplyF32
                | RepackOp::ScaleF32 { .. } => unreachable!("a bytewise shape has no f32 op"),
            }
        }
        let at = group.values as usize + start;
        copy_lanes(&mut self.slots[at..], &self.bytes[0], lanes);
    }

    /// Evaluates values `start..start + lanes` of `group` in word lanes
    /// into their slots.
    #[inline(always)]
    fn evaluate_words(&mut self, group: &ValueGroup, start: usize, lanes: usize) {
        let count = group.count as usize;
        // Arithmetic takes whole chunks: a lane past `lanes` computes
        // whatever it holds ([`CHUNK`]).
        let chunks = lanes.div_ceil(CHUNK);
        let mut height = 0;
        for (op, prepared) in shape_ops(self.program, &self.prepared, group) {
            // The columns on the stack, and the next one.
            let (low, high) = self.words.split_at_mut(height);
            let top = |low| top_chunks(low, chunks);
            match op {
                RepackOp::Field { prefix, width } => {
                    let bytes = slot(width) as usize;
                    let at = group.fields as usize + prefix as usize * count + bytes * start;
                    load_column(&mut high[0], &self.fields[at..], bytes, lanes);
                    height += 1;
                }
                RepackOp::ShiftLeft { bits } => {
                    for chunk in top(low) {
                        *chunk = chunk.map(|value| value << bits);
                    }
                }
                RepackOp::Or => {
                    let (low, operand) = low.split_at_mut(height - 1);
                    let operand = operand[0].as_chunks::<CHUNK>().0;
                    for (chunk, other) in top(low).zip(operand) {
                        *chunk = std::array::from_fn(|lane| chunk[lane] | other[lane]);
                    }
                    height -= 1;
                }
                RepackOp::NegateIf => {
                    let (low, operand) = low.split_at_mut(height - 1);
                    let operand = operand[0].as_chunks::<CHUNK>().0;
                    for (chunk, sign) in top(low).zip(operand) {
                        *chunk = std::array::from_fn(|lane| negate_if(chunk[lane], sign[lane]));
                    }
                    height -= 1;
                }
                RepackOp::Offset { offset } => {
                    let at = group.offsets as usize + offset as usize * count + start;
                    let offsets = &self.program.offsets[at..at + lanes];
                    let column = low.last_mut().expect("an offset has an operand");
                    for (value, offset) in column[..lanes].iter_mut().zip(offsets) {
                        *value = value.wrapping_add(*offset as u32);
                    }
                }
                RepackOp::F16ToF32 => {
                    for chunk in top(low) {
                        for words in chunk.as_chunks_mut::<4>().0 {
                            vector::widen_halves(words);
                        }
                    }
                }
                RepackOp::I32ToF32 => {
                    for chunk in top(low) {
                        *chunk = chunk.map(|value| (value as i32 as f32).to_bits());
                    }
                }
                RepackOp::MultiplyF32 => {
                    let (low, operand) = low.split_at_mut(height - 1);
                    let operand = operand[0].as_chunks::<CHUNK>().0;
                    for (chunk, other) in top(low).zip(operand) {
                        *chunk = std::array::from_fn(|lane| {
                            (f32::from_bits(chunk[lane]) * f32::from_bits(other[lane])).to_bits()
                        });
                    }
                    height -= 1;
                }
                RepackOp::ScaleF32 { .. } => {
                    let Prepared::Factor(factor) = prepared else {
                        unreachable!("a scale's factor is prepared")
                    };
                    for chunk in top(low) {
                        *chunk = chunk.map(|value| (f32::from_bits(value) * factor).to_bits());
                    }
                }
                RepackOp::Lookup { table } => {
                    // Only the lanes present: past them an index is arbitrary.
                    let values = table.values();
                    let column = low.last_mut().expect("a lookup has an index");
                    for value in &mut column[..lanes] {
                        *value = values[*value as usize] as u32;
                    }
                }
            }
        }
        let bytes = group.bytes as usize;
        let at = group.values as usize + bytes * start;
        store_column(&self.words[0], &mut self.slots[at..], bytes, lanes);
    }
}

/// Lanes a column moves at once: past a column's lanes, a chunk's are
/// whatever the buffers hold (each is padded by [`PADDING`] bytes, and a
/// group's slots are followed by those of the groups evaluated after it).
const CHUNK: usize = 16;
/// Bytes past the slots of the field and of the value buffers: a chunk of
/// lanes of 4-byte slots, which a column moves past its lanes, and at least
/// a block, which a move reads from any slot byte.
const PADDING: usize = CHUNK * 4;
const _: () = assert!(PADDING >= BLOCK, "a move reads a block past any slot byte");

/// The first `chunks` chunks of the top column of `stack`.
#[inline(always)]
fn top_chunks(
    stack: &mut [[u32; LANES]],
    chunks: usize,
) -> impl Iterator<Item = &mut [u32; CHUNK]> {
    stack
        .last_mut()
        .expect("an op has an operand")
        .as_chunks_mut::<CHUNK>()
        .0
        .iter_mut()
        .take(chunks)
}

/// Copies the first `lanes` bytes of `from` to `to`, a chunk at a time.
#[inline(always)]
fn copy_lanes(to: &mut [u8], from: &[u8], lanes: usize) {
    for (to, from) in to
        .chunks_exact_mut(CHUNK)
        .zip(from.chunks_exact(CHUNK))
        .take(lanes.div_ceil(CHUNK))
    {
        to.copy_from_slice(from);
    }
}

/// Widens the first `lanes` slots of `slots`, each the `bytes` (1, 2 or
/// 4) little-endian bytes of a value, into `column`, a chunk at a time.
#[inline(always)]
fn load_column(column: &mut [u32; LANES], slots: &[u8], bytes: usize, lanes: usize) {
    #[inline(always)]
    fn chunks<const BYTES: usize>(column: &mut [u32; LANES], slots: &[u8], lanes: usize) {
        let slots = slots.as_chunks::<BYTES>().0;
        for (values, slots) in column
            .as_chunks_mut::<CHUNK>()
            .0
            .iter_mut()
            .zip(slots.as_chunks::<CHUNK>().0)
            .take(lanes.div_ceil(CHUNK))
        {
            *values = std::array::from_fn(|lane| {
                let mut word = [0; 4];
                word[..BYTES].copy_from_slice(&slots[lane]);
                u32::from_le_bytes(word)
            });
        }
    }
    match bytes {
        1 => chunks::<1>(column, slots, lanes),
        2 => chunks::<2>(column, slots, lanes),
        _ => chunks::<4>(column, slots, lanes),
    }
}

/// Narrows the first `lanes` words of `column` into slots of `bytes` (1, 2
/// or 4) little-endian bytes of `slots`, a chunk at a time.
#[inline(always)]
fn store_column(column: &[u32; LANES], slots: &mut [u8], bytes: usize, lanes: usize) {
    #[inline(always)]
    fn chunks<const BYTES: usize>(column: &[u32; LANES], slots: &mut [u8], lanes: usize) {
        let slots = slots.as_chunks_mut::<BYTES>().0;
        for (values, slots) in column
            .as_chunks::<CHUNK>()
            .0
            .iter()
            .zip(slots.as_chunks_mut::<CHUNK>().0)
            .take(lanes.div_ceil(CHUNK))
        {
            *slots =
                std::array::from_fn(|lane| values[lane].to_le_bytes()[..BYTES].try_into().unwrap());
        }
    }
    match bytes {
        1 => chunks::<1>(column, slots, lanes),
        2 => chunks::<2>(column, slots, lanes),
        _ => chunks::<4>(column, slots, lanes),
    }
}

/// The ops of `group`'s shape, each with its prepared form.
fn shape_ops<'a>(
    program: &'a RepackProgram,
    prepared: &'a [Prepared],
    group: &ValueGroup,
) -> impl Iterator<Item = (RepackOp, Prepared)> + 'a {
    let shape = program.shapes[group.shape as usize];
    let ops = shape.ops as usize..(shape.ops + shape.length) as usize;
    program.ops[ops.clone()]
        .iter()
        .copied()
        .zip(prepared[ops].iter().copied())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn evaluate_ssse3(executor: &mut RepackExecutor<'_>, outputs: &[*mut u8]) {
    unsafe { executor.evaluate(outputs) }
}

#[cfg(test)]
mod tests {
    use magnitude_ggml::Encoding;
    use seismic_lang::registry::{self, RepresentationConversion};

    /// The executable form of `conversion` repacks `source` (a tensor of
    /// `extents`) exactly as the reference interpreter does.
    fn assert_matches_the_interpreter(
        conversion: &RepresentationConversion,
        extents: &[u64],
        source: &[u8],
        what: &str,
    ) {
        let shape = extents
            .iter()
            .map(|extent| *extent as usize)
            .collect::<Vec<_>>();
        let expected =
            seismic_lang::interp::repack(conversion.id, &shape, source).expect("reference repack");
        let actual = super::super::convert(conversion, extents, source).expect("executable repack");
        assert_eq!(actual.len(), expected.len());
        if let Some(at) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
            panic!(
                "{} -> {} ({what}, {extents:?}): byte {at} is {:#04x}, the interpreter's {:#04x}",
                registry::representation_info(conversion.source).name,
                registry::representation_info(conversion.destination).name,
                actual[at],
                expected[at],
            );
        }
    }

    /// Every registered conversion's program writes the interpreter's bytes
    /// for random source packets in the conversion's domain, in every
    /// layout, on rows whose last packet is partial and over more packets
    /// than an execution's.
    #[test]
    fn programs_repack_random_packets_as_the_interpreter() {
        for conversion in registry::representation_conversions() {
            let layout = conversion.source_layout;
            let group = u64::from(layout.logical_group);
            let execution =
                super::PacketForms::compile(conversion).program.packets() as u64 * group;
            for extents in [
                [3, group],
                [5, 2 * group + group.div_ceil(3)],
                [3, 2 * execution + group.div_ceil(3)],
            ] {
                let packets = extents[0] * extents[1].div_ceil(group);
                let mut source = registry::pseudo_random_bytes(
                    conversion.id.index() as u64 ^ extents[1],
                    (packets * u64::from(layout.packet_size)) as usize,
                );
                conversion.admit_source(&mut source);
                assert_matches_the_interpreter(conversion, &extents, &source, "random");
            }
        }
    }

    /// The vector widening of binary16 is `registry::f16_to_f32` for every
    /// binary16, NaN payloads included.
    #[test]
    fn vector_widening_is_the_registry_widening() {
        let halves = (0..=u16::MAX).map(u32::from).collect::<Vec<_>>();
        for words in halves.chunks_exact(4) {
            let mut widened = words.try_into().unwrap();
            super::vector::widen_halves(&mut widened);
            for (word, widened) in words.iter().zip(widened) {
                assert_eq!(
                    widened,
                    registry::f16_to_f32(*word as u16).to_bits(),
                    "{word:#06x}"
                );
            }
        }
    }

    /// Every registered conversion's program writes the interpreter's bytes
    /// for the reference blocks of its GGUF type
    /// (`engine/kernels/tests/support/gguf_vectors`).
    #[test]
    fn programs_repack_reference_blocks_as_the_interpreter() {
        let vectors = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../engine/kernels/tests/support/gguf_vectors");
        for conversion in registry::representation_conversions() {
            let layout = conversion.source_layout;
            let group = u64::from(layout.logical_group);
            let encoding = Encoding::ALL
                .into_iter()
                .find(|encoding| registry::gguf_source(*encoding, group) == Some(conversion.source))
                .expect("a conversion's source is a GGUF type's");
            let path = vectors.join(format!("{}.bin", encoding.ggml_name()));
            let file =
                std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let blocks = u64::from(u32::from_le_bytes(file[..4].try_into().unwrap()));
            let packets = blocks * encoding.block_elements() / group;
            let source = &file[4..][..(packets * u64::from(layout.packet_size)) as usize];
            assert_matches_the_interpreter(conversion, &[packets, group], source, "reference");
        }
    }
}
