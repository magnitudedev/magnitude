//! Prompt-only readout plans, as the runs the target computes for them.
//!
//! A plan's sequences share their first `shared` rows. When more than one
//! sequence shares them, or the plan retains them, the shared rows are one
//! run (the trunk), computed once; every sequence that reads past them
//! continues in a run of its own, opened from a fork of the trunk's state.
//! Otherwise each sequence is one run. A run computes rows only through
//! the last row it reads or retains, then to the first boundary of its
//! sequence's visibility at or after it: nothing after it is read, and
//! every row before it is computed with every row it reads. A run's chunks
//! end at such boundaries, so the rows between two of them (a unit) share
//! one advance: one launch, or for a unit wider than any launch a launch of
//! its own that the executor computes in chunks (layer-major). Each readout
//! is answered once, by the run that computes its answer row
//! (`answer_row`); logits readouts of one row are read together as the
//! logits of the union of their tokens, and head readouts of one row as one
//! read of the head.
//!
//! A scores readout reads final hidden states of rows its sequence computed
//! before its answer row, possibly in the trunk. Every run retains, in the
//! executor's store for it (`ExecutorDomain::retain_features`), the states
//! of the rows it computes that a scores readout reads; a continuation also
//! reads the trunk's store, which lives while the continuation does. So a
//! trunk's rows are retained once for every sequence, and only the scores
//! reach the host.

use magnitude_executor::{Demand, Operation, RequestId, TokenId, WorkKind, answer_row};
use magnitude_family_contracts::{BoundaryRule, InputLayout, PreparedModelInput};
use magnitude_readout::{
    Readout, ReadoutKind, ReadoutPlan, ReadoutUsage, ReadoutValues, Visibility,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

/// Which rows of its plan a run computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunRole {
    /// The shared rows, for the continuations to fork.
    Trunk,
    /// A sequence's rows after the shared rows, from a fork of the trunk.
    Continuation,
    /// A whole sequence, from fresh or cached state.
    Whole,
}

/// A readout a run row answers.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Answer {
    sequence: usize,
    readout: usize,
    kind: ReadoutKind,
    /// The readout's own row (a scores readout's query row).
    query: usize,
}

/// What one row of a run reads: the logits of the ascending union of the
/// tokens of the logits readouts it answers (none when the union is
/// empty), the head when a readout it answers reads it, then each distinct
/// scores readout it answers (its query row and key rows).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RowRead {
    tokens: Arc<[TokenId]>,
    head: bool,
    scores: Vec<(usize, Range<usize>)>,
    answers: Vec<Answer>,
}

impl RowRead {
    fn add(&mut self, sequence: usize, readout: &Readout, index: usize) {
        match &readout.kind {
            ReadoutKind::Logits { tokens } => {
                let mut union = self.tokens.to_vec();
                union.extend(tokens.iter());
                union.sort_unstable();
                union.dedup();
                self.tokens = union.into();
            }
            ReadoutKind::Head => self.head = true,
            ReadoutKind::Scores { rows } => {
                if !self.scores.contains(&(readout.row, rows.clone())) {
                    self.scores.push((readout.row, rows.clone()));
                }
            }
        }
        self.answers.push(Answer {
            sequence,
            readout: index,
            kind: readout.kind.clone(),
            query: readout.row,
        });
    }

    /// The row's reads as the forward computes them, in order.
    fn readouts(&self, row: usize) -> impl Iterator<Item = Readout> + '_ {
        let logits = (!self.tokens.is_empty()).then(|| ReadoutKind::Logits {
            tokens: self.tokens.clone(),
        });
        let head = self.head.then_some(ReadoutKind::Head);
        logits
            .into_iter()
            .chain(head)
            .map(move |kind| Readout { row, kind })
            .chain(self.scores.iter().map(|(query, rows)| Readout {
                row: *query,
                kind: ReadoutKind::Scores { rows: rows.clone() },
            }))
    }
}

/// One run of a plan: the rows `start..end` of sequence `sequence`'s input
/// it computes, what its rows read and retain, and what they attend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    role: RunRole,
    sequence: usize,
    start: usize,
    end: usize,
    /// What its rows read, by answer row.
    reads: BTreeMap<usize, RowRead>,
    visibility: Visibility,
    /// The rows it computes whose final hidden states a scores readout
    /// reads, ascending.
    retained: Vec<usize>,
}

impl Run {
    pub fn role(&self) -> RunRole {
        self.role
    }

    /// The sequence whose input the run computes rows of (the first one,
    /// for the trunk).
    pub fn sequence(&self) -> usize {
        self.sequence
    }

    /// The position after the run's last row.
    pub fn end(&self) -> usize {
        self.end
    }

    /// What the run's rows read on global attention layers.
    pub fn visibility(&self) -> &Visibility {
        &self.visibility
    }

    /// The run's units: the rows between consecutive boundaries of its
    /// visibility, each computed in one advance. A causal run's are single
    /// rows, which are not listed.
    pub fn units(&self) -> Vec<std::ops::Range<usize>> {
        if self.visibility.is_causal() {
            return Vec::new();
        }
        let mut units = Vec::new();
        let mut start = self.start;
        while start < self.end {
            let end = self
                .visibility
                .boundary_from(start + 1)
                .expect("a run ends at a boundary");
            units.push(start..end);
            start = end;
        }
        units
    }

    /// Its parts of `layout`'s indivisible spans (`layout` its sequence's
    /// input's): beside its units, the rows it computes in one advance.
    pub fn indivisible(&self, layout: &InputLayout) -> Vec<std::ops::Range<usize>> {
        layout
            .spans()
            .iter()
            .filter(|span| span.boundaries == BoundaryRule::Indivisible)
            .map(|span| span.start.max(self.start)..span.end.min(self.end))
            .filter(|rows| !rows.is_empty())
            .collect()
    }

    /// The rows whose final hidden states it retains, ascending.
    pub fn retained(&self) -> &[usize] {
        &self.retained
    }

    /// Positions below which a retained state may open the run: it must
    /// compute its first readout row and first retained row, and needs
    /// nothing past its end.
    pub fn resume_bound(&self) -> usize {
        self.reads
            .keys()
            .next()
            .into_iter()
            .chain(self.retained.first())
            .min()
            .map_or(self.end + 1, |&row| row + 1)
    }

    /// The run's next chunk from `position`, a boundary: about `allowance`
    /// rows legal for `input`'s layout, ending at the last boundary within
    /// them or after its first unit when none is; a committed prompt chunk
    /// reading the values of its readout rows. A unit wider than a launch of
    /// `launch` rows is a chunk alone, which the executor computes in
    /// launch-sized chunks of its own; it takes a whole launch. `None` when
    /// the chunk exceeds the `limit` rows its launch has left.
    pub fn lower(
        &self,
        request: RequestId,
        input: &PreparedModelInput,
        position: usize,
        allowance: usize,
        limit: usize,
        launch: usize,
    ) -> Result<Option<Operation>, String> {
        let unit = self
            .visibility
            .boundary_from(position + 1)
            .ok_or("a run ends at a boundary")?;
        // The layout keeps an indivisible span whole; a unit is whole too.
        let end = input.layout().chunk_end(position, self.end, allowance)?;
        let end = self.visibility.boundary_to(end).max(unit);
        let rows = end - position;
        if rows > limit && !(rows > launch && limit == launch) {
            return Ok(None);
        }
        let readouts = self
            .reads
            .range(position..end)
            .flat_map(|(&row, read)| read.readouts(row))
            .collect::<Vec<_>>();
        let tokens = input.tokens()[position..end].to_vec();
        Ok(Some(Operation::Forward {
            request,
            kind: WorkKind::Prefill,
            committed: tokens.len(),
            tokens,
            position,
            conditioning: None,
            demand: if readouts.is_empty() {
                Demand::NONE
            } else {
                Demand::READOUT
            },
            select: Vec::new(),
            prime: None,
            readouts,
        }))
    }

    /// Answer the readouts of the row at `row` from its `values`, one list
    /// per read of the row in order (`RowRead::readouts`): the logits of
    /// its union of tokens, the head's values, then each scores read's.
    pub fn answer(
        &self,
        row: usize,
        values: &[Vec<f32>],
        plan: &mut PlanValues,
    ) -> Result<(), String> {
        let read = self
            .reads
            .get(&row)
            .ok_or("a readout value answers a row the run does not read")?;
        let logits = !read.tokens.is_empty();
        let first_scores = usize::from(logits) + usize::from(read.head);
        if values.len() != first_scores + read.scores.len()
            || logits && values[0].len() != read.tokens.len()
            || read
                .scores
                .iter()
                .zip(&values[first_scores..])
                .any(|((_, rows), values)| values.len() != rows.len())
        {
            return Err("a readout row's values differ from its reads".into());
        }
        for answer in &read.answers {
            let answered = match &answer.kind {
                ReadoutKind::Logits { tokens } => tokens
                    .iter()
                    .map(|token| {
                        read.tokens
                            .binary_search(token)
                            .map(|column| values[0][column])
                            .map_err(|_| "a readout token is absent from its row's read".to_owned())
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                ReadoutKind::Head => values[usize::from(logits)].clone(),
                ReadoutKind::Scores { rows } => {
                    let read = read
                        .scores
                        .iter()
                        .position(|scores| *scores == (answer.query, rows.clone()))
                        .ok_or("a scores readout is absent from its row's reads")?;
                    values[first_scores + read].clone()
                }
            };
            plan.record(answer.sequence, answer.readout, answered)?;
        }
        Ok(())
    }
}

/// Whether `plan`'s shared rows are retained: it asks, and it has shared
/// rows. Their end is an explicit cache point, which a later plan resumes
/// from at any depth (`CachePoint::Trunk`).
pub fn retains(plan: &ReadoutPlan) -> bool {
    plan.retain_shared() && plan.shared() > 0
}

/// The runs of `plan`, the trunk first when there is one. A sequence with
/// nothing to read past the shared rows has no continuation, and a run with
/// nothing to read and no state to share is not made.
pub fn runs(plan: &ReadoutPlan) -> Vec<Run> {
    let shared = plan.shared();
    let sequences = plan.sequences();
    let trunk = shared > 0 && (sequences.len() > 1 || retains(plan));
    // Which run computes row `row` of `sequence`: the trunk's rows are the
    // shared rows when there is one.
    let in_trunk = |row: usize| trunk && row < shared;
    let mut trunk_reads = BTreeMap::<usize, RowRead>::new();
    let mut trunk_retained = BTreeSet::new();
    let mut own = vec![(BTreeMap::<usize, RowRead>::new(), BTreeSet::new()); sequences.len()];
    for (sequence, input) in sequences.iter().enumerate() {
        for (index, readout) in input.readouts().iter().enumerate() {
            let answer = answer_row(readout);
            let reads = if in_trunk(answer) {
                &mut trunk_reads
            } else {
                &mut own[sequence].0
            };
            reads.entry(answer).or_default().add(sequence, readout, index);
            if let ReadoutKind::Scores { rows } = &readout.kind {
                for row in rows.clone().chain([readout.row]) {
                    if in_trunk(row) {
                        trunk_retained.insert(row);
                    } else {
                        own[sequence].1.insert(row);
                    }
                }
            }
        }
    }
    let mut runs = Vec::new();
    if trunk {
        runs.push(Run {
            role: RunRole::Trunk,
            sequence: 0,
            start: 0,
            end: shared,
            reads: trunk_reads,
            visibility: sequences[0].visibility(),
            retained: trunk_retained.into_iter().collect(),
        });
    }
    for (sequence, (reads, retained)) in own.into_iter().enumerate() {
        let Some(&last) = reads.keys().next_back() else {
            continue;
        };
        // A run ends where a chunk may: past its last answer row (the last
        // row it reads or retains), past the indivisible media span holding
        // it, and at a boundary of its visibility (admission refuses one
        // inside a media span).
        let input = &sequences[sequence];
        let visibility = input.visibility();
        let end = visibility
            .boundary_from(input.input().layout().next_boundary(last + 1))
            .expect("a sequence ends at a boundary");
        runs.push(Run {
            role: if trunk {
                RunRole::Continuation
            } else {
                RunRole::Whole
            },
            sequence,
            start: if trunk { shared } else { 0 },
            end,
            reads,
            visibility,
            retained: retained.into_iter().collect(),
        });
    }
    runs
}

/// A plan's answers as they arrive, and the rows its runs read.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanValues {
    sequences: Vec<Vec<Option<Vec<f32>>>>,
    usage: ReadoutUsage,
}

impl PlanValues {
    pub fn new(plan: &ReadoutPlan) -> Self {
        Self {
            sequences: plan
                .sequences()
                .iter()
                .map(|sequence| vec![None; sequence.readouts().len()])
                .collect(),
            usage: ReadoutUsage::default(),
        }
    }

    fn record(&mut self, sequence: usize, readout: usize, values: Vec<f32>) -> Result<(), String> {
        let slot = self
            .sequences
            .get_mut(sequence)
            .and_then(|readouts| readouts.get_mut(readout))
            .ok_or("an answer names no readout of its plan")?;
        if slot.replace(values).is_some() {
            return Err("a readout was answered twice".into());
        }
        Ok(())
    }

    pub fn computed(&mut self, rows: usize) {
        self.usage.computed_tokens += rows as u64;
    }

    pub fn cached(&mut self, rows: usize) {
        self.usage.cached_tokens += rows as u64;
    }

    /// Every readout's values, once each has been answered.
    pub fn finished(&self) -> Option<ReadoutValues> {
        Some(ReadoutValues {
            sequences: self
                .sequences
                .iter()
                .map(|readouts| readouts.iter().cloned().collect::<Option<Vec<_>>>())
                .collect::<Option<Vec<_>>>()?,
            usage: self.usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_family_contracts::{InputLayout, TokenPlan};
    use magnitude_readout::{Block, ReadoutSequence};

    fn input(tokens: &[u32]) -> PreparedModelInput {
        let tokens = tokens.iter().copied().map(TokenId).collect::<Vec<_>>();
        let count = tokens.len();
        PreparedModelInput::from_text_coordinates(
            TokenPlan::new(tokens, InputLayout::new(count, vec![]).unwrap()).unwrap(),
            (0..count).map(|row| [row as i32; 3]).collect(),
        )
        .unwrap()
    }

    fn logits(row: usize, tokens: &[u32]) -> Readout {
        Readout {
            row,
            kind: ReadoutKind::Logits {
                tokens: tokens.iter().copied().map(TokenId).collect(),
            },
        }
    }

    fn sequence(tokens: &[u32], readouts: Vec<Readout>) -> ReadoutSequence {
        ReadoutSequence::new(input(tokens), Vec::new(), readouts).unwrap()
    }

    /// Two branches of a four-row trunk: the trunk is one run reading the
    /// union of both branches' trunk readouts at its row, each branch reads
    /// only past it, through its last readout row.
    #[test]
    fn branches_fork_a_trunk_that_reads_their_shared_rows_once() {
        let plan = ReadoutPlan::new(
            4,
            vec![
                sequence(
                    &[1, 2, 3, 4, 5, 6, 7],
                    vec![logits(3, &[9, 5]), logits(5, &[1])],
                ),
                sequence(&[1, 2, 3, 4, 8], vec![logits(3, &[5, 7]), logits(4, &[2])]),
            ],
            false,
        )
        .unwrap();
        let runs = runs(&plan);
        assert_eq!(
            runs.iter()
                .map(|run| (run.role(), run.sequence(), run.end()))
                .collect::<Vec<_>>(),
            vec![
                (RunRole::Trunk, 0, 4),
                (RunRole::Continuation, 0, 6),
                (RunRole::Continuation, 1, 5),
            ]
        );
        let trunk = &runs[0];
        let chunk = trunk
            .lower(RequestId(1), plan.sequences()[0].input(), 0, 3, 8, 8)
            .unwrap()
            .unwrap();
        let Operation::Forward {
            tokens,
            readouts,
            demand,
            ..
        } = &chunk
        else {
            unreachable!()
        };
        assert_eq!(tokens.len(), 3);
        assert!(readouts.is_empty() && *demand == Demand::NONE);
        let chunk = trunk
            .lower(RequestId(1), plan.sequences()[0].input(), 3, 8, 8, 8)
            .unwrap()
            .unwrap();
        chunk.validate().unwrap();
        let Operation::Forward { readouts, .. } = &chunk else {
            unreachable!()
        };
        assert_eq!(readouts, &[logits(3, &[5, 7, 9])]);
        let mut values = PlanValues::new(&plan);
        trunk.answer(3, &[vec![0.5, 0.7, 0.9]], &mut values).unwrap();
        runs[1].answer(5, &[vec![0.1]], &mut values).unwrap();
        assert!(values.finished().is_none());
        runs[2].answer(4, &[vec![0.2]], &mut values).unwrap();
        assert_eq!(
            values.finished().unwrap().sequences,
            vec![
                vec![vec![0.9, 0.5], vec![0.1]],
                vec![vec![0.5, 0.7], vec![0.2]]
            ]
        );
        assert_eq!(runs[0].resume_bound(), 4);
    }

    /// A row's logits and head readouts are read once each, the logits of
    /// their union first; each readout is answered from its own read.
    #[test]
    fn a_row_reads_logits_and_the_head_once_each() {
        let head = |row| Readout {
            row,
            kind: ReadoutKind::Head,
        };
        let plan = ReadoutPlan::new(
            0,
            vec![sequence(
                &[1, 2, 3],
                vec![head(2), logits(2, &[8, 4]), head(2), logits(1, &[4])],
            )],
            false,
        )
        .unwrap();
        let run = &runs(&plan)[0];
        let chunk = run
            .lower(RequestId(1), plan.sequences()[0].input(), 0, 8, 8, 8)
            .unwrap()
            .unwrap();
        chunk.validate().unwrap();
        let Operation::Forward { readouts, .. } = &chunk else {
            unreachable!()
        };
        assert_eq!(readouts, &[logits(1, &[4]), logits(2, &[4, 8]), head(2)]);
        let mut values = PlanValues::new(&plan);
        run.answer(1, &[vec![0.4]], &mut values).unwrap();
        assert!(run.answer(2, &[vec![0.4, 0.8]], &mut values).is_err());
        run.answer(2, &[vec![0.4, 0.8], vec![1.0, 2.0]], &mut values)
            .unwrap();
        assert_eq!(
            values.finished().unwrap().sequences,
            vec![vec![vec![1.0, 2.0], vec![0.8, 0.4], vec![1.0, 2.0], vec![0.4]]]
        );
    }

    /// A scores readout is answered by the run computing the last row it
    /// reads; every row it reads is retained by the run computing it, the
    /// trunk's once for every sequence, and runs end past what they retain.
    #[test]
    fn scores_are_answered_where_their_last_row_is_and_retain_what_they_read() {
        let scores = |row, rows| Readout {
            row,
            kind: ReadoutKind::Scores { rows },
        };
        let plan = ReadoutPlan::new(
            4,
            vec![
                sequence(&[1, 2, 3, 4, 5, 6, 7, 8], vec![scores(6, 1..5)]),
                sequence(
                    &[1, 2, 3, 4, 9, 9, 9],
                    vec![scores(2, 0..3), scores(1, 4..6), logits(5, &[3])],
                ),
            ],
            false,
        )
        .unwrap();
        let runs = runs(&plan);
        assert_eq!(
            runs.iter()
                .map(|run| (run.role(), run.sequence(), run.end(), run.retained().to_vec()))
                .collect::<Vec<_>>(),
            vec![
                (RunRole::Trunk, 0, 4, vec![0, 1, 2, 3]),
                (RunRole::Continuation, 0, 7, vec![4, 6]),
                (RunRole::Continuation, 1, 6, vec![4, 5]),
            ]
        );
        assert_eq!(runs[1].resume_bound(), 5);
        let lowered = |run: &Run, position| {
            let chunk = run
                .lower(RequestId(1), plan.sequences()[run.sequence()].input(), position, 8, 8, 8)
                .unwrap()
                .unwrap();
            chunk.validate().unwrap();
            let Operation::Forward { readouts, .. } = chunk else {
                unreachable!()
            };
            readouts
        };
        assert_eq!(lowered(&runs[0], 0), vec![scores(2, 0..3)]);
        assert_eq!(lowered(&runs[1], 4), vec![scores(6, 1..5)]);
        assert_eq!(lowered(&runs[2], 4), vec![logits(5, &[3]), scores(1, 4..6)]);
        let mut values = PlanValues::new(&plan);
        runs[0].answer(2, &[vec![0.0, 0.1, 0.2]], &mut values).unwrap();
        runs[1].answer(6, &[vec![1.0, 1.1, 1.2, 1.3]], &mut values).unwrap();
        assert!(runs[2].answer(5, &[vec![0.3], vec![2.0]], &mut values).is_err());
        runs[2].answer(5, &[vec![0.3], vec![2.0, 2.1]], &mut values).unwrap();
        assert_eq!(
            values.finished().unwrap().sequences,
            vec![
                vec![vec![1.0, 1.1, 1.2, 1.3]],
                vec![vec![0.0, 0.1, 0.2], vec![2.0, 2.1], vec![0.3]]
            ]
        );
    }

    /// A lone sequence is one run unless it retains its shared rows, at any
    /// depth; a trunk with nothing to read may open at its end.
    #[test]
    fn a_lone_sequence_splits_only_to_retain_its_shared_rows() {
        let tokens = [1, 2, 3, 4, 5, 6];
        let lone = |shared, retain| {
            ReadoutPlan::new(
                shared,
                vec![sequence(&tokens, vec![logits(tokens.len() - 1, &[1])])],
                retain,
            )
            .unwrap()
        };
        let end = tokens.len();
        for (shared, retain) in [(3, false), (0, true)] {
            assert!(!retains(&lone(shared, retain)));
            assert_eq!(
                runs(&lone(shared, retain))
                    .iter()
                    .map(|run| (run.role(), run.end()))
                    .collect::<Vec<_>>(),
                vec![(RunRole::Whole, end)]
            );
        }
        let plan = lone(3, true);
        assert!(retains(&plan));
        let retained = runs(&plan);
        assert_eq!(
            retained
                .iter()
                .map(|run| (run.role(), run.end()))
                .collect::<Vec<_>>(),
            vec![(RunRole::Trunk, 3), (RunRole::Continuation, end)]
        );
        assert_eq!(retained[0].resume_bound(), 4);
    }

    fn blocked(tokens: &[u32], blocks: Vec<Block>, readouts: Vec<Readout>) -> ReadoutSequence {
        ReadoutSequence::new(input(tokens), blocks, readouts).unwrap()
    }

    /// A bidirectional trunk's branches continue past it and compute their
    /// whole block, though their last readout row comes earlier; a chunk
    /// never splits a block, and a block wider than the launch's rows left
    /// waits for a launch it fits, or one of its own when wider than any.
    #[test]
    fn blocked_runs_compute_whole_blocks_in_one_chunk() {
        let branch = |tokens: &[u32], readout: usize| {
            blocked(
                tokens,
                vec![Block::new(0..4, vec![0]), Block::new(4..tokens.len(), vec![0, 1])],
                vec![logits(readout, &[1])],
            )
        };
        let plan = ReadoutPlan::new(
            4,
            vec![branch(&[1, 2, 3, 4, 5, 6, 7], 4), branch(&[1, 2, 3, 4, 8, 9], 5)],
            false,
        )
        .unwrap();
        let runs = runs(&plan);
        assert_eq!(
            runs.iter()
                .map(|run| (run.role(), run.end(), run.units()))
                .collect::<Vec<_>>(),
            vec![
                (RunRole::Trunk, 4, vec![0..4]),
                (RunRole::Continuation, 7, vec![4..7]),
                (RunRole::Continuation, 6, vec![4..6]),
            ]
        );
        let input = plan.sequences()[0].input();
        let rows = |run: &Run, position, allowance, limit, launch| {
            run.lower(RequestId(1), input, position, allowance, limit, launch)
                .unwrap()
                .map(|operation| operation.row_count())
        };
        assert_eq!(rows(&runs[0], 0, 2, 8, 8), Some(4), "the trunk block is one chunk");
        assert_eq!(rows(&runs[0], 0, 2, 3, 8), None, "it waits for a launch it fits");
        assert_eq!(rows(&runs[0], 0, 2, 2, 3), None, "a wider block waits for a launch");
        assert_eq!(rows(&runs[0], 0, 3, 3, 3), Some(4), "and takes it whole");
        assert_eq!(rows(&runs[1], 4, 1, 8, 8), Some(3));
        assert!(!runs[1].visibility().reads_apart(4..7));
    }

    /// Head and scores readouts in blocked rows: a branch's scores reading
    /// the trunk are answered in its own block, which retains its rows and
    /// is computed whole, while the trunk retains the rows it reads.
    #[test]
    fn blocked_rows_read_the_head_and_scores_in_whole_blocks() {
        let tokens = [1, 2, 3, 4, 5, 6, 7];
        let scores = Readout {
            row: 5,
            kind: ReadoutKind::Scores { rows: 1..5 },
        };
        let head = Readout {
            row: 4,
            kind: ReadoutKind::Head,
        };
        let branch = blocked(
            &tokens,
            vec![Block::new(0..4, vec![0]), Block::new(4..7, vec![0, 1])],
            vec![scores.clone(), head.clone()],
        );
        let plan = ReadoutPlan::new(4, vec![branch.clone(), branch], false).unwrap();
        let runs = runs(&plan);
        assert_eq!(
            runs.iter()
                .map(|run| (run.role(), run.end(), run.retained().to_vec(), run.units()))
                .collect::<Vec<_>>(),
            vec![
                (RunRole::Trunk, 4, vec![1, 2, 3], vec![0..4]),
                (RunRole::Continuation, 7, vec![4, 5], vec![4..7]),
                (RunRole::Continuation, 7, vec![4, 5], vec![4..7]),
            ]
        );
        let chunk = runs[1]
            .lower(RequestId(1), plan.sequences()[0].input(), 4, 1, 8, 8)
            .unwrap()
            .unwrap();
        chunk.validate().unwrap();
        let Operation::Forward { tokens, readouts, .. } = chunk else {
            unreachable!()
        };
        assert_eq!((tokens.len(), readouts), (3, vec![head, scores]));
    }

    /// A packed trunk reads every block packed after it, so the whole
    /// sequence is one unit, whose question rows read the trunk and
    /// themselves around rows they do not read.
    #[test]
    fn a_packed_set_is_one_unit_read_out_of_order() {
        let sequence = blocked(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            vec![
                Block::new(0..3, vec![0, 1, 2]),
                Block::new(3..5, vec![0, 1]),
                Block::new(5..8, vec![0, 2]),
            ],
            vec![logits(4, &[1, 2]), logits(7, &[2, 3])],
        );
        let plan = ReadoutPlan::new(0, vec![sequence], false).unwrap();
        let runs = runs(&plan);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].units(), vec![0..8]);
        assert!(runs[0].visibility().reads_apart(0..8));
        let operation = runs[0]
            .lower(RequestId(1), plan.sequences()[0].input(), 0, 2, 8, 8)
            .unwrap()
            .unwrap();
        assert_eq!(operation.row_count(), 8);
    }

    /// A run's indivisible spans are its parts of the layout's indivisible
    /// spans: one after the trunk is in each continuation; a causal one in
    /// none.
    #[test]
    fn a_run_holds_its_part_of_each_indivisible_span() {
        use magnitude_family_contracts::{BoundaryRule, InputLayout, InputSpan};
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let plan = ReadoutPlan::new(
            2,
            vec![
                sequence(&tokens, vec![logits(7, &[1])]),
                sequence(&tokens, vec![logits(7, &[2])]),
            ],
            false,
        )
        .unwrap();
        let span = |start, end, boundaries| InputSpan {
            start,
            end,
            identity: format!("{start}"),
            boundaries,
            language_history: false,
        };
        let layout = InputLayout::new(
            8,
            vec![
                span(2, 5, BoundaryRule::Indivisible),
                span(6, 7, BoundaryRule::Causal),
            ],
        )
        .unwrap();
        let runs = runs(&plan);
        assert_eq!(
            runs.iter().map(|run| run.indivisible(&layout)).collect::<Vec<_>>(),
            vec![vec![], vec![2..5], vec![2..5]]
        );
    }
}
