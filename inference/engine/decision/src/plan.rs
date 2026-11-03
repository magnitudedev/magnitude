//! A prepared decision's evaluation in prompt token positions, before the
//! family's input adapter expands its images into rows.

use magnitude_artifacts::TokenId;
use magnitude_chat::request::ImageInput;
use magnitude_family_contracts::PreparedModelInput;
use magnitude_readout::{Block, PlanError, Readout, ReadoutKind, ReadoutPlan, ReadoutSequence};
use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

/// What a readout reads at its position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenReadout {
    /// These tokens' logits, in order.
    Logits(Arc<[TokenId]>),
    /// The model's readout head.
    Head,
    /// Scores against the positions `positions` of the same sequence.
    Scores(Range<usize>),
}

/// One sequence: its tokens (one placeholder token per image), the request
/// images its placeholders stand for in order, its attention blocks over
/// token positions (empty: causal) and its readouts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenSequence {
    pub tokens: Vec<TokenId>,
    pub images: Vec<usize>,
    pub blocks: Vec<(Range<usize>, Vec<usize>)>,
    pub readouts: Vec<(usize, TokenReadout)>,
}

/// Sequences sharing their first `shared` tokens; `retain` keeps the shared
/// state for later requests with the same shared tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenPlan {
    pub shared: usize,
    pub sequences: Vec<TokenSequence>,
    pub retain: bool,
}

/// Why a token plan has no engine plan.
#[derive(Debug)]
pub enum ReadoutPlanError<E> {
    /// The caller's input preparation refused a sequence.
    Input(E),
    /// The prepared inputs do not admit the plan's blocks or readouts.
    Plan(PlanError),
}

impl<E: fmt::Display> fmt::Display for ReadoutPlanError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
        }
    }
}

impl TokenPlan {
    /// The engine plan: each sequence's tokens and own images prepared by
    /// `prepare` (the family's input adapter, bounded by the served
    /// context), and every position mapped to its row.
    pub fn readout_plan<E>(
        &self,
        images: &[ImageInput],
        mut prepare: impl FnMut(Vec<TokenId>, &[ImageInput]) -> Result<PreparedModelInput, E>,
    ) -> Result<ReadoutPlan, ReadoutPlanError<E>> {
        let mut shared = None;
        let mut sequences = Vec::with_capacity(self.sequences.len());
        for sequence in &self.sequences {
            let own: Vec<ImageInput> = sequence
                .images
                .iter()
                .map(|image| images[*image].clone())
                .collect();
            let input = prepare(sequence.tokens.clone(), &own).map_err(ReadoutPlanError::Input)?;
            let row = |position: usize| input.prompt_position_row(position);
            let rows_of = |range: &Range<usize>| row(range.start)..row(range.end);
            shared.get_or_insert(row(self.shared));
            let blocks = sequence
                .blocks
                .iter()
                .map(|(range, reads)| Block::new(rows_of(range), reads.clone()))
                .collect();
            let readouts = sequence
                .readouts
                .iter()
                .map(|(position, readout)| Readout {
                    row: row(*position),
                    kind: match readout {
                        TokenReadout::Logits(tokens) => ReadoutKind::Logits {
                            tokens: tokens.clone(),
                        },
                        TokenReadout::Head => ReadoutKind::Head,
                        TokenReadout::Scores(positions) => ReadoutKind::Scores {
                            rows: rows_of(positions),
                        },
                    },
                })
                .collect();
            sequences.push(
                ReadoutSequence::new(input, blocks, readouts).map_err(ReadoutPlanError::Plan)?,
            );
        }
        ReadoutPlan::new(shared.unwrap_or(0), sequences, self.retain)
            .map_err(ReadoutPlanError::Plan)
    }
}

/// How a request's questions are laid out after its state: the three facts
/// of a description's layout (`set_size`, `state_attends`,
/// `question_attends`), from which every sequence's blocks follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Questions laid out after one state in one sequence; `None` for all.
    pub set_size: Option<NonZeroUsize>,
    pub state_attends: StateAttends,
    pub question_attends: QuestionAttends,
}

/// What a state's rows read, both ways, on global attention layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateAttends {
    /// Itself.
    State,
    /// Itself and every question of its set.
    Set,
}

/// What a question's rows read, both ways, on global attention layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuestionAttends {
    /// The state and itself.
    Question,
    /// The state and every question of its set.
    Set,
}

/// One sequence before its blocks: its tokens, images and readouts, and the
/// token counts of its state and of each of its questions, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaidSequence {
    pub tokens: Vec<TokenId>,
    pub images: Vec<usize>,
    pub readouts: Vec<(usize, TokenReadout)>,
    pub state: usize,
    pub questions: Vec<usize>,
}

impl Layout {
    /// The question indices of each set of `questions` questions, in order.
    pub fn sets(&self, questions: usize) -> Vec<Range<usize>> {
        let size = self.set_size.map_or(questions.max(1), NonZeroUsize::get);
        (0..questions)
            .step_by(size)
            .map(|start| start..(start + size).min(questions))
            .collect()
    }

    /// The blocks of a sequence of `state` rows, then each question's rows,
    /// in their canonical form ([`canonical`]): laid out as the state, then
    /// each question, each reading what its layout says.
    pub fn blocks(&self, state: usize, questions: &[usize]) -> Vec<(Range<usize>, Vec<usize>)> {
        let set: Vec<usize> = (0..=questions.len()).collect();
        let mut blocks = vec![(
            0..state,
            match self.state_attends {
                StateAttends::State => vec![0],
                StateAttends::Set => set.clone(),
            },
        )];
        let mut start = state;
        for (index, length) in questions.iter().enumerate() {
            let reads = match self.question_attends {
                QuestionAttends::Question => vec![0, index + 1],
                QuestionAttends::Set => set.clone(),
            };
            blocks.push((start..start + length, reads));
            start += length;
        }
        canonical(&blocks)
    }

    /// The token plan of `sequences`, one per set. A state that reads only
    /// itself is the same in every sequence: it is computed once, shared,
    /// and retained for later requests with the same state.
    pub fn plan(&self, sequences: Vec<LaidSequence>) -> Result<TokenPlan, String> {
        let shared = match self.state_attends {
            StateAttends::Set => 0,
            StateAttends::State => {
                let first = &sequences[0];
                if sequences.iter().any(|sequence| {
                    sequence.state != first.state
                        || sequence.tokens[..sequence.state] != first.tokens[..first.state]
                }) {
                    return Err(
                        "the state does not end on a token boundary shared by every set".into(),
                    );
                }
                first.state
            }
        };
        Ok(TokenPlan {
            shared,
            sequences: sequences
                .into_iter()
                .map(|sequence| TokenSequence {
                    blocks: self.blocks(sequence.state, &sequence.questions),
                    tokens: sequence.tokens,
                    images: sequence.images,
                    readouts: sequence.readouts,
                })
                .collect(),
            retain: self.state_attends == StateAttends::State,
        })
    }
}

/// The same attention in the fewest blocks: neighbouring blocks that read
/// the same blocks, each other included, and that every block reads both or
/// neither of, are one block, each block's reads renumbered.
pub fn canonical(blocks: &[(Range<usize>, Vec<usize>)]) -> Vec<(Range<usize>, Vec<usize>)> {
    let mut merged: Vec<(Range<usize>, usize)> = Vec::new();
    let mut into = Vec::with_capacity(blocks.len());
    for (index, (rows, reads)) in blocks.iter().enumerate() {
        let joins = index > 0 && {
            let before = index - 1;
            *reads == blocks[before].1
                && reads.contains(&before)
                && reads.contains(&index)
                && blocks
                    .iter()
                    .all(|(_, reads)| reads.contains(&before) == reads.contains(&index))
        };
        match merged.last_mut() {
            Some((last, _)) if joins => last.end = rows.end,
            _ => merged.push((rows.clone(), index)),
        }
        into.push(merged.len() - 1);
    }
    merged
        .into_iter()
        .map(|(rows, first)| {
            let mut reads: Vec<usize> = blocks[first].1.iter().map(|read| into[*read]).collect();
            reads.dedup();
            (rows, reads)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FACTS: [(StateAttends, QuestionAttends); 4] = [
        (StateAttends::State, QuestionAttends::Question),
        (StateAttends::State, QuestionAttends::Set),
        (StateAttends::Set, QuestionAttends::Question),
        (StateAttends::Set, QuestionAttends::Set),
    ];

    fn laid(state: &[u32], questions: &[usize]) -> LaidSequence {
        let mut tokens: Vec<TokenId> = state.iter().copied().map(TokenId).collect();
        tokens
            .extend((0..questions.iter().sum::<usize>()).map(|token| TokenId(100 + token as u32)));
        LaidSequence {
            tokens,
            images: Vec::new(),
            readouts: Vec::new(),
            state: state.len(),
            questions: questions.to_vec(),
        }
    }

    /// Every layout the keys express: its sets, each block's rows and
    /// reads, and whether the state is shared and retained.
    #[test]
    fn every_layout_plans_from_its_facts() {
        for size in [None, NonZeroUsize::new(1), NonZeroUsize::new(2)] {
            for (state_attends, question_attends) in FACTS {
                let layout = Layout {
                    set_size: size,
                    state_attends,
                    question_attends,
                };
                let sets = layout.sets(3);
                assert_eq!(
                    sets,
                    match size.map(NonZeroUsize::get) {
                        None => vec![0..3],
                        Some(1) => vec![0..1, 1..2, 2..3],
                        _ => vec![0..2, 2..3],
                    }
                );
                let sequences: Vec<LaidSequence> = sets
                    .iter()
                    .map(|set| laid(&[1, 2], &vec![2; set.len()]))
                    .collect();
                let plan = layout.plan(sequences).unwrap();
                let shares = state_attends == StateAttends::State;
                assert_eq!(
                    (plan.shared, plan.retain),
                    (if shares { 2 } else { 0 }, shares)
                );
                for (sequence, set) in plan.sequences.iter().zip(&sets) {
                    let all: Vec<usize> = (0..=set.len()).collect();
                    let end = 2 + 2 * set.len();
                    let expected = match (state_attends, question_attends) {
                        // Blocks that read each other and are read alike
                        // are one: the whole set, or its questions.
                        (StateAttends::Set, QuestionAttends::Set) => vec![(0..end, vec![0])],
                        (StateAttends::Set, QuestionAttends::Question) if set.len() == 1 => {
                            vec![(0..end, vec![0])]
                        }
                        (StateAttends::State, QuestionAttends::Set) => {
                            vec![(0..2, vec![0]), (2..end, vec![0, 1])]
                        }
                        _ => {
                            let mut expected =
                                vec![(0..2, if shares { vec![0] } else { all.clone() })];
                            for member in 0..set.len() {
                                let start = 2 + 2 * member;
                                expected.push((
                                    start..start + 2,
                                    match question_attends {
                                        QuestionAttends::Question => vec![0, member + 1],
                                        QuestionAttends::Set => all.clone(),
                                    },
                                ));
                            }
                            expected
                        }
                    };
                    assert_eq!(sequence.blocks, expected);
                }
            }
        }
        assert!(Layout {
            set_size: None,
            state_attends: StateAttends::Set,
            question_attends: QuestionAttends::Set,
        }
        .sets(0)
        .is_empty());
    }

    /// A shared state must be the same tokens in every sequence.
    #[test]
    fn a_shared_state_differing_between_sets_is_refused() {
        let layout = Layout {
            set_size: NonZeroUsize::new(1),
            state_attends: StateAttends::State,
            question_attends: QuestionAttends::Question,
        };
        assert!(layout
            .plan(vec![laid(&[1, 2], &[1]), laid(&[1, 3], &[1])])
            .is_err());
        assert!(layout
            .plan(vec![laid(&[1, 2], &[1]), laid(&[1], &[2])])
            .is_err());
    }
}
