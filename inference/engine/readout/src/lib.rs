//! Prompt-only readout: forward passes that end in values read at chosen
//! rows instead of generated tokens.
//!
//! A [`ReadoutPlan`] is one or more sequences that share their first
//! [`ReadoutPlan::shared`] rows. The engine computes the shared rows once and
//! continues every sequence from that state. Each sequence declares its
//! attention [`Block`]s and the [`Readout`]s it wants; nothing here knows what
//! the values mean.

use magnitude_artifacts::TokenId;
use magnitude_family_contracts::{InputPreparationError, ModelDefinition, PreparedModelInput};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

/// Rows of one sequence that attend to each other in both directions on the
/// model's global attention layers.
///
/// A row of a block attends, on global attention layers, to every row of the
/// blocks it reads ([`Block::reads`], which include the block itself). Layers
/// that are not global attention (recurrent, convolution, sliding-window
/// attention) stay causal over the whole sequence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    rows: Range<usize>,
    reads: Vec<usize>,
}

impl Block {
    /// `reads` are indices of blocks of the same sequence, ascending and
    /// including this block's own.
    pub fn new(rows: Range<usize>, reads: Vec<usize>) -> Self {
        Self { rows, reads }
    }

    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }

    pub fn reads(&self) -> &[usize] {
        &self.reads
    }
}

/// What a readout returns at its row.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadoutKind {
    /// The output projection's logits of these tokens, in this order, from
    /// the row's final normalized hidden state.
    Logits { tokens: Arc<[TokenId]> },
    /// The model's readout head applied to the row's final normalized hidden
    /// state: one value per head output.
    Head,
    /// The scaled dot products `h · h_j / √d` of the row's final normalized
    /// hidden state `h` with those of rows `rows` of the same sequence.
    Scores { rows: Range<usize> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Readout {
    pub row: usize,
    pub kind: ReadoutKind,
}

/// One sequence of a plan: its whole prepared input, its attention blocks
/// and its readouts.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "RawReadoutSequence")]
pub struct ReadoutSequence {
    input: PreparedModelInput,
    blocks: Vec<Block>,
    readouts: Vec<Readout>,
}

impl ReadoutSequence {
    /// `blocks` either is empty (the sequence is causal) or partitions the
    /// sequence in order. Every readout row lies inside the sequence.
    pub fn new(
        input: PreparedModelInput,
        blocks: Vec<Block>,
        readouts: Vec<Readout>,
    ) -> Result<Self, PlanError> {
        let rows = input.tokens().len();
        if rows == 0 {
            return Err(PlanError::EmptySequence);
        }
        if !blocks.is_empty() {
            let mut next = 0;
            for (index, block) in blocks.iter().enumerate() {
                let reads_known = block.reads.windows(2).all(|pair| pair[0] < pair[1])
                    && block.reads.iter().all(|read| *read < blocks.len())
                    && block.reads.contains(&index);
                if block.rows.start != next || block.rows.end <= block.rows.start || !reads_known {
                    return Err(PlanError::Blocks);
                }
                next = block.rows.end;
            }
            if next != rows {
                return Err(PlanError::Blocks);
            }
        }
        for readout in &readouts {
            let inside = match &readout.kind {
                ReadoutKind::Logits { tokens } => !tokens.is_empty(),
                ReadoutKind::Head => true,
                ReadoutKind::Scores { rows: scored } => {
                    scored.start < scored.end && scored.end <= rows
                }
            };
            if readout.row >= rows || !inside {
                return Err(PlanError::Readout { row: readout.row });
            }
        }
        Ok(Self {
            input,
            blocks,
            readouts,
        })
    }

    pub fn input(&self) -> &PreparedModelInput {
        &self.input
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn readouts(&self) -> &[Readout] {
        &self.readouts
    }

    /// What each row reads on global attention layers.
    pub fn visibility(&self) -> Visibility {
        Visibility::of(self.blocks.iter().cloned().collect())
    }
}

/// Which rows each row of a sequence reads on the model's global attention
/// layers: every row up to and including itself (causal), or every row of
/// the blocks its block reads ([`Block`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Visibility {
    /// Empty for a causal sequence; otherwise blocks validated by
    /// [`ReadoutSequence::new`].
    blocks: Arc<[Block]>,
    /// The blocked sequence's boundaries, ascending (`Self::boundary`).
    boundaries: Arc<[usize]>,
}

impl Visibility {
    fn of(blocks: Arc<[Block]>) -> Self {
        // A block's end is a boundary when no block up to it reads past it.
        let mut boundaries = vec![0];
        let mut reach = 0;
        for block in blocks.iter() {
            reach = block
                .reads
                .iter()
                .map(|read| blocks[*read].rows.end)
                .fold(reach.max(block.rows.end), usize::max);
            if reach == block.rows.end {
                boundaries.push(block.rows.end);
            }
        }
        if blocks.is_empty() {
            boundaries.clear();
        }
        Self {
            blocks,
            boundaries: boundaries.into(),
        }
    }

    pub fn causal() -> Self {
        Self::default()
    }

    pub fn is_causal(&self) -> bool {
        self.blocks.is_empty()
    }

    /// The index of the block holding `row`.
    fn block_index(&self, row: usize) -> Option<usize> {
        let index = self.blocks.partition_point(|block| block.rows.end <= row);
        (index < self.blocks.len()).then_some(index)
    }

    /// The rows `row` reads, ascending, disjoint and never adjacent; `None`
    /// for a row outside a blocked sequence.
    pub fn reads(&self, row: usize) -> Option<Vec<Range<usize>>> {
        if self.is_causal() {
            return Some(vec![0..row + 1]);
        }
        let block = &self.blocks[self.block_index(row)?];
        let mut reads: Vec<Range<usize>> = Vec::with_capacity(block.reads.len());
        for read in &block.reads {
            let rows = self.blocks[*read].rows();
            match reads.last_mut() {
                Some(last) if last.end == rows.start => last.end = rows.end,
                _ => reads.push(rows),
            }
        }
        Some(reads)
    }

    /// The rows of `row`'s block (itself alone for a causal sequence);
    /// `None` for a row outside a blocked sequence.
    pub fn block(&self, row: usize) -> Option<Range<usize>> {
        if self.is_causal() {
            return Some(row..row + 1);
        }
        Some(self.blocks[self.block_index(row)?].rows())
    }

    /// Whether no row before `position` reads a row at or after it: the
    /// points where a sequence's rows may be split between computations,
    /// every row before one having been computed with everything it reads.
    pub fn boundary(&self, position: usize) -> bool {
        self.is_causal() || self.boundaries.binary_search(&position).is_ok()
    }

    /// The first [`Self::boundary`] at or after `position`, or `None` past a
    /// blocked sequence's end.
    pub fn boundary_from(&self, position: usize) -> Option<usize> {
        if self.is_causal() {
            return Some(position);
        }
        let index = self.boundaries.partition_point(|&boundary| boundary < position);
        self.boundaries.get(index).copied()
    }

    /// The last [`Self::boundary`] at or before `position`.
    pub fn boundary_to(&self, position: usize) -> usize {
        if self.is_causal() {
            return position;
        }
        let index = self.boundaries.partition_point(|&boundary| boundary <= position);
        self.boundaries[index - 1]
    }

    /// Whether a row of `rows` reads rows of them in more than one range:
    /// rows of them on both sides of rows it does not read.
    pub fn reads_apart(&self, rows: Range<usize>) -> bool {
        !self.is_causal()
            && rows.clone().any(|row| {
                self.reads(row).is_some_and(|reads| {
                    reads
                        .iter()
                        .filter(|read| read.end > rows.start && read.start < rows.end)
                        .count()
                        > 1
                })
            })
    }

    /// Whether the rows before `position` read alike in `self` and `other`.
    pub fn alike_before(&self, other: &Visibility, position: usize) -> bool {
        self.before(position) == other.before(position)
    }

    /// The blocks wholly before `position`.
    fn before(&self, position: usize) -> &[Block] {
        let count = self
            .blocks
            .partition_point(|block| block.rows.end <= position);
        &self.blocks[..count]
    }

    /// The visibility of the rows before `position`, a boundary: equal for
    /// two sequences exactly when those rows read alike.
    pub fn prefix(&self, position: usize) -> Visibility {
        Self::of(self.before(position).into())
    }
}

/// Sequences evaluated together, sharing their first `shared` rows.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "RawReadoutPlan")]
pub struct ReadoutPlan {
    shared: usize,
    sequences: Vec<ReadoutSequence>,
    retain_shared: bool,
}

impl ReadoutPlan {
    /// Every sequence has the same first `shared` tokens and the same blocks
    /// over them, and no row before `shared` reads a block that reaches past
    /// it, so the shared rows' state is the same for every sequence. With
    /// `retain_shared`, that state is kept for later plans with the same
    /// shared rows.
    pub fn new(
        shared: usize,
        sequences: Vec<ReadoutSequence>,
        retain_shared: bool,
    ) -> Result<Self, PlanError> {
        let Some(first) = sequences.first() else {
            return Err(PlanError::NoSequences);
        };
        if sequences.iter().any(|sequence| sequence.input.tokens().len() < shared) {
            return Err(PlanError::Shared);
        }
        let shared_blocks = |sequence: &ReadoutSequence| {
            sequence
                .blocks
                .iter()
                .take_while(|block| block.rows.start < shared)
                .cloned()
                .collect::<Vec<_>>()
        };
        let blocks = shared_blocks(first);
        let self_contained = blocks.iter().all(|block| {
            block.rows.end <= shared
                && block
                    .reads
                    .iter()
                    .all(|read| first.blocks[*read].rows.end <= shared)
        });
        let agree = sequences.iter().all(|sequence| {
            sequence.input.tokens()[..shared] == first.input.tokens()[..shared]
                && shared_blocks(sequence) == blocks
        });
        if !self_contained || !agree {
            return Err(PlanError::Shared);
        }
        Ok(Self {
            shared,
            sequences,
            retain_shared,
        })
    }

    pub fn shared(&self) -> usize {
        self.shared
    }

    pub fn sequences(&self) -> &[ReadoutSequence] {
        &self.sequences
    }

    pub fn retain_shared(&self) -> bool {
        self.retain_shared
    }

    /// Re-establish every invariant against the model `definition`: each
    /// sequence's input is valid for it (`PreparedModelInput::validated`)
    /// and within its context, and every logits readout reads tokens of its
    /// vocabulary. A plan decoded from another process is accepted only
    /// through this. Validation keeps every input's tokens, so the plan's
    /// own invariants still hold.
    pub fn validated(self, definition: &ModelDefinition) -> Result<Self, PlanValidationError> {
        let decoder = &definition.decoder;
        let sequences = self
            .sequences
            .into_iter()
            .map(|sequence| {
                let required = sequence.input.tokens().len() as u64;
                if required > decoder.context_limit {
                    return Err(PlanValidationError::Context {
                        required,
                        limit: decoder.context_limit,
                    });
                }
                let outside = sequence.readouts.iter().any(|readout| match &readout.kind {
                    ReadoutKind::Logits { tokens } => {
                        tokens.iter().any(|token| u64::from(token.0) >= decoder.vocabulary)
                    }
                    ReadoutKind::Head | ReadoutKind::Scores { .. } => false,
                });
                if outside {
                    return Err(PlanValidationError::Vocabulary);
                }
                Ok(ReadoutSequence {
                    input: sequence
                        .input
                        .validated(definition)
                        .map_err(PlanValidationError::Input)?,
                    ..sequence
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { sequences, ..self })
    }
}

/// Why a plan is invalid for a model ([`ReadoutPlan::validated`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanValidationError {
    Input(InputPreparationError),
    Context { required: u64, limit: u64 },
    Vocabulary,
}

impl fmt::Display for PlanValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(error) => write!(formatter, "{error}"),
            Self::Context { required, limit } => write!(
                formatter,
                "a readout sequence of {required} tokens exceeds the context of {limit}"
            ),
            Self::Vocabulary => {
                formatter.write_str("a logits readout names a token outside the vocabulary")
            }
        }
    }
}

impl std::error::Error for PlanValidationError {}

/// Deserialization revalidates through [`ReadoutSequence::new`].
#[derive(Deserialize)]
struct RawReadoutSequence {
    input: PreparedModelInput,
    blocks: Vec<Block>,
    readouts: Vec<Readout>,
}

impl TryFrom<RawReadoutSequence> for ReadoutSequence {
    type Error = PlanError;
    fn try_from(raw: RawReadoutSequence) -> Result<Self, PlanError> {
        Self::new(raw.input, raw.blocks, raw.readouts)
    }
}

/// Deserialization revalidates through [`ReadoutPlan::new`].
#[derive(Deserialize)]
struct RawReadoutPlan {
    shared: usize,
    sequences: Vec<ReadoutSequence>,
    retain_shared: bool,
}

impl TryFrom<RawReadoutPlan> for ReadoutPlan {
    type Error = PlanError;
    fn try_from(raw: RawReadoutPlan) -> Result<Self, PlanError> {
        Self::new(raw.shared, raw.sequences, raw.retain_shared)
    }
}

/// A plan's values: per sequence, per readout, in plan order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReadoutValues {
    pub sequences: Vec<Vec<Vec<f32>>>,
    pub usage: ReadoutUsage,
}

/// Rows the plan's execution read: every row it computed, and the leading
/// shared rows restored from retained state rather than computed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadoutUsage {
    pub computed_tokens: u64,
    pub cached_tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanError {
    NoSequences,
    EmptySequence,
    Blocks,
    Readout { row: usize },
    Shared,
}

impl fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSequences => formatter.write_str("a readout plan needs at least one sequence"),
            Self::EmptySequence => formatter.write_str("a readout sequence needs at least one row"),
            Self::Blocks => formatter.write_str(
                "blocks must partition the sequence in order and read known blocks including their own",
            ),
            Self::Readout { row } => write!(
                formatter,
                "readout at row {row} lies outside its sequence or reads nothing"
            ),
            Self::Shared => formatter.write_str(
                "shared rows must be identical, identically blocked and self-contained in every sequence",
            ),
        }
    }
}

impl std::error::Error for PlanError {}
