use std::collections::{BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkKind {
    Decode,
    Prefill { remaining: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkCandidate {
    pub(crate) sequence_id: i32,
    pub(crate) kind: WorkKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchWork {
    Decode { sequence_id: i32 },
    Prefill { sequence_id: i32, tokens: usize },
}

/// Magnitude-owned, policy-light batch assembly.
///
/// Every runnable decode token is placed before prompt work. Prompt work is then split into
/// rotating round-robin quanta until the logical batch is full. This keeps token generation
/// responsive without importing llama-server's slot or cache policies, while still filling large
/// batches when enough prompt work exists.
#[derive(Debug)]
pub(crate) struct BatchPlanner {
    cursor: usize,
    prefill_quantum: usize,
}

impl BatchPlanner {
    pub(crate) fn new(prefill_quantum: usize) -> Self {
        assert!(prefill_quantum > 0);
        Self {
            cursor: 0,
            prefill_quantum,
        }
    }

    pub(crate) fn plan(&mut self, candidates: &[WorkCandidate], capacity: usize) -> Vec<BatchWork> {
        if candidates.is_empty() || capacity == 0 {
            return Vec::new();
        }

        let start = self.cursor % candidates.len();
        let ordered = (0..candidates.len())
            .map(|offset| candidates[(start + offset) % candidates.len()])
            .collect::<Vec<_>>();
        self.cursor = (start + 1) % candidates.len();

        let mut available = capacity;
        let mut result = Vec::new();

        // Decode-first is explicit: at most one next-token decode per sequence per iteration.
        for candidate in &ordered {
            if available == 0 {
                break;
            }
            if candidate.kind == WorkKind::Decode {
                result.push(BatchWork::Decode {
                    sequence_id: candidate.sequence_id,
                });
                available -= 1;
            }
        }

        let mut prefill = ordered
            .iter()
            .filter_map(|candidate| match candidate.kind {
                WorkKind::Decode => None,
                WorkKind::Prefill { remaining } => Some((candidate.sequence_id, remaining)),
            })
            .collect::<Vec<_>>();

        while available > 0 {
            let mut progressed = false;
            for (sequence_id, remaining) in &mut prefill {
                if available == 0 {
                    break;
                }
                let tokens = (*remaining).min(self.prefill_quantum).min(available);
                if tokens == 0 {
                    continue;
                }
                result.push(BatchWork::Prefill {
                    sequence_id: *sequence_id,
                    tokens,
                });
                *remaining -= tokens;
                available -= tokens;
                progressed = true;
            }
            if !progressed {
                break;
            }
        }

        result
    }
}

#[derive(Debug)]
pub(crate) struct SequencePool {
    free: VecDeque<i32>,
    owned: BTreeSet<i32>,
}

impl SequencePool {
    pub(crate) fn new(count: u32) -> Self {
        Self {
            free: (0..count)
                .map(|value| i32::try_from(value).expect("validated sequence count fits i32"))
                .collect(),
            owned: BTreeSet::new(),
        }
    }

    pub(crate) fn acquire(&mut self) -> Option<i32> {
        let sequence_id = self.free.pop_front()?;
        assert!(self.owned.insert(sequence_id), "sequence was already owned");
        Some(sequence_id)
    }

    pub(crate) fn release(&mut self, sequence_id: i32) {
        assert!(
            self.owned.remove(&sequence_id),
            "attempted to release an unowned sequence"
        );
        self.free.push_back(sequence_id);
    }

    /// Remove a sequence from service after native cleanup failed. Reusing it could expose one
    /// request to another request's resident state, so capacity is deliberately reduced instead.
    pub(crate) fn quarantine(&mut self, sequence_id: i32) {
        assert!(
            self.owned.remove(&sequence_id),
            "attempted to quarantine an unowned sequence"
        );
    }

    #[cfg(test)]
    pub(crate) fn owned(&self) -> &BTreeSet<i32> {
        &self.owned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_work_always_precedes_prefill() {
        let mut planner = BatchPlanner::new(2);
        let plan = planner.plan(
            &[
                WorkCandidate {
                    sequence_id: 0,
                    kind: WorkKind::Prefill { remaining: 10 },
                },
                WorkCandidate {
                    sequence_id: 1,
                    kind: WorkKind::Decode,
                },
                WorkCandidate {
                    sequence_id: 2,
                    kind: WorkKind::Prefill { remaining: 10 },
                },
            ],
            6,
        );
        assert_eq!(plan[0], BatchWork::Decode { sequence_id: 1 });
        assert_eq!(plan.iter().map(batch_size).sum::<usize>(), 6);
    }

    #[test]
    fn prompt_quanta_are_fair_and_fill_the_batch() {
        let mut planner = BatchPlanner::new(2);
        let candidates = [
            WorkCandidate {
                sequence_id: 0,
                kind: WorkKind::Prefill { remaining: 100 },
            },
            WorkCandidate {
                sequence_id: 1,
                kind: WorkKind::Prefill { remaining: 100 },
            },
        ];
        let first = planner.plan(&candidates, 6);
        let second = planner.plan(&candidates, 6);

        assert_eq!(first.iter().map(batch_size).sum::<usize>(), 6);
        assert_eq!(second.iter().map(batch_size).sum::<usize>(), 6);
        assert_eq!(
            first[0],
            BatchWork::Prefill {
                sequence_id: 0,
                tokens: 2
            }
        );
        assert_eq!(
            second[0],
            BatchWork::Prefill {
                sequence_id: 1,
                tokens: 2
            }
        );

        let allocated = |plan: &[BatchWork], sequence_id| {
            plan.iter()
                .filter_map(|work| match work {
                    BatchWork::Prefill {
                        sequence_id: id,
                        tokens,
                    } if *id == sequence_id => Some(*tokens),
                    _ => None,
                })
                .sum::<usize>()
        };
        assert!(allocated(&first, 0).abs_diff(allocated(&first, 1)) <= 2);
        assert!(allocated(&second, 0).abs_diff(allocated(&second, 1)) <= 2);
    }

    #[test]
    fn sequence_ownership_is_isolated_and_reused_only_after_release() {
        let mut pool = SequencePool::new(2);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        assert_ne!(first, second);
        assert_eq!(pool.acquire(), None);
        assert_eq!(pool.owned(), &BTreeSet::from([first, second]));

        pool.release(first);
        assert_eq!(pool.acquire(), Some(first));
        assert_eq!(pool.owned(), &BTreeSet::from([first, second]));
    }

    #[test]
    fn failed_cleanup_quarantines_only_the_affected_sequence() {
        let mut pool = SequencePool::new(2);
        let cancelled = pool.acquire().unwrap();
        let survivor = pool.acquire().unwrap();
        pool.quarantine(cancelled);

        assert_eq!(pool.acquire(), None);
        assert_eq!(pool.owned(), &BTreeSet::from([survivor]));
        pool.release(survivor);
        assert_eq!(pool.acquire(), Some(survivor));
        assert_eq!(pool.acquire(), None);
    }

    fn batch_size(work: &BatchWork) -> usize {
        match work {
            BatchWork::Decode { .. } => 1,
            BatchWork::Prefill { tokens, .. } => *tokens,
        }
    }
}
