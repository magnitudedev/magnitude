use std::collections::VecDeque;

use llama_cpp_2::LlamaSequenceState;
use llama_cpp_2::speculative::{SpeculativePosition, SpeculativePromptState};
use llama_cpp_2::token::LlamaToken;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PromptBoundary {
    pub(crate) logical_tokens: usize,
    pub(crate) native_position: i32,
}

impl PromptBoundary {
    pub(crate) fn speculative_position(self) -> Option<SpeculativePosition> {
        Some(SpeculativePosition {
            target: self.native_position,
            draft: i32::try_from(self.logical_tokens).ok()?,
        })
    }

    pub(crate) fn advance(self, tokens: usize) -> Option<Self> {
        Some(Self {
            logical_tokens: self.logical_tokens.checked_add(tokens)?,
            native_position: self
                .native_position
                .checked_add(i32::try_from(tokens).ok()?)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PromptSegment {
    Text(Vec<LlamaToken>),
    Media {
        identity: String,
        logical_tokens: usize,
        native_positions: i32,
    },
}

impl PromptSegment {
    fn logical_tokens(&self) -> usize {
        match self {
            Self::Text(tokens) => tokens.len(),
            Self::Media { logical_tokens, .. } => *logical_tokens,
        }
    }

    fn native_positions(&self) -> i32 {
        match self {
            Self::Text(tokens) => i32::try_from(tokens.len()).unwrap_or(i32::MAX),
            Self::Media {
                native_positions, ..
            } => *native_positions,
        }
    }
}

/// Semantic prompt input used to correlate native KV with future requests.
///
/// Text may be matched token-by-token. Media is an indivisible span identified by content and
/// preprocessing semantics, with logical token and native position counts kept separate for
/// M-RoPE models.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PromptLayout {
    segments: Vec<PromptSegment>,
}

impl PromptLayout {
    pub(crate) fn text(tokens: Vec<LlamaToken>) -> Self {
        Self {
            segments: vec![PromptSegment::Text(tokens)],
        }
    }

    pub(crate) fn new(segments: Vec<PromptSegment>) -> Self {
        Self { segments }
    }

    pub(crate) fn segments(&self) -> &[PromptSegment] {
        &self.segments
    }

    pub(crate) fn logical_tokens(&self) -> usize {
        self.segments
            .iter()
            .map(PromptSegment::logical_tokens)
            .sum()
    }

    pub(crate) fn text_tokens(&self) -> Vec<LlamaToken> {
        self.segments
            .iter()
            .filter_map(|segment| match segment {
                PromptSegment::Text(tokens) => Some(tokens.as_slice()),
                PromptSegment::Media { .. } => None,
            })
            .flatten()
            .copied()
            .collect()
    }

    pub(crate) fn text_tokens_at(&self, logical_token: usize) -> Option<&[LlamaToken]> {
        let mut start = 0usize;
        for segment in &self.segments {
            let end = start.checked_add(segment.logical_tokens())?;
            if logical_token < end {
                return match segment {
                    PromptSegment::Text(tokens) => Some(&tokens[logical_token - start..]),
                    PromptSegment::Media { .. } => None,
                };
            }
            start = end;
        }
        None
    }

    pub(crate) fn media_at(&self, logical_token: usize) -> Option<(usize, i32)> {
        let mut start = 0usize;
        for segment in &self.segments {
            if start == logical_token {
                return match segment {
                    PromptSegment::Media {
                        logical_tokens,
                        native_positions,
                        ..
                    } => Some((*logical_tokens, *native_positions)),
                    PromptSegment::Text(_) => None,
                };
            }
            start = start.checked_add(segment.logical_tokens())?;
        }
        None
    }

    pub(crate) fn common_prefix(&self, incoming: &Self) -> PromptBoundary {
        let mut boundary = PromptBoundary::default();
        let mut left = self.segments.iter();
        let mut right = incoming.segments.iter();
        loop {
            match (left.next(), right.next()) {
                (Some(PromptSegment::Text(left)), Some(PromptSegment::Text(right))) => {
                    let matched = left
                        .iter()
                        .zip(right)
                        .take_while(|(left, right)| left == right)
                        .count();
                    boundary.logical_tokens += matched;
                    boundary.native_position = boundary
                        .native_position
                        .saturating_add(i32::try_from(matched).unwrap_or(i32::MAX));
                    if matched != left.len() || matched != right.len() {
                        return boundary;
                    }
                }
                (
                    Some(PromptSegment::Media {
                        identity: left_identity,
                        logical_tokens: left_tokens,
                        native_positions: left_positions,
                    }),
                    Some(PromptSegment::Media {
                        identity: right_identity,
                        logical_tokens: right_tokens,
                        native_positions: right_positions,
                    }),
                ) if left_identity == right_identity
                    && left_tokens == right_tokens
                    && left_positions == right_positions =>
                {
                    boundary.logical_tokens += left_tokens;
                    boundary.native_position =
                        boundary.native_position.saturating_add(*left_positions);
                }
                _ => return boundary,
            }
        }
    }

    pub(crate) fn boundary_before_final_text_token(&self) -> Option<PromptBoundary> {
        let logical_tokens = self.logical_tokens().checked_sub(1)?;
        self.boundary_at(logical_tokens)
    }

    pub(crate) fn boundary_at(&self, logical_tokens: usize) -> Option<PromptBoundary> {
        let mut boundary = PromptBoundary::default();
        for segment in &self.segments {
            let segment_tokens = segment.logical_tokens();
            if boundary.logical_tokens + segment_tokens <= logical_tokens {
                boundary.logical_tokens += segment_tokens;
                boundary.native_position = boundary
                    .native_position
                    .checked_add(segment.native_positions())?;
                continue;
            }
            let within = logical_tokens.checked_sub(boundary.logical_tokens)?;
            match segment {
                PromptSegment::Text(_) => {
                    boundary.logical_tokens += within;
                    boundary.native_position = boundary
                        .native_position
                        .checked_add(i32::try_from(within).ok()?)?;
                    return Some(boundary);
                }
                PromptSegment::Media { .. } => return None,
            }
        }
        (boundary.logical_tokens == logical_tokens).then_some(boundary)
    }

    /// Return the first legal semantic boundary at or after a requested logical position.
    ///
    /// Text can be split token-by-token. Media cannot, so a position inside a media span advances
    /// to the end of that span instead of inventing a partially reusable media boundary.
    pub(crate) fn boundary_at_or_after(&self, logical_tokens: usize) -> Option<PromptBoundary> {
        let mut boundary = PromptBoundary::default();
        for segment in &self.segments {
            let segment_tokens = segment.logical_tokens();
            let segment_end = boundary.logical_tokens.checked_add(segment_tokens)?;
            if segment_end < logical_tokens {
                boundary.logical_tokens = segment_end;
                boundary.native_position = boundary
                    .native_position
                    .checked_add(segment.native_positions())?;
                continue;
            }
            if logical_tokens <= boundary.logical_tokens {
                return Some(boundary);
            }
            match segment {
                PromptSegment::Text(_) => {
                    let within = logical_tokens.checked_sub(boundary.logical_tokens)?;
                    boundary.logical_tokens = boundary.logical_tokens.checked_add(within)?;
                    boundary.native_position = boundary
                        .native_position
                        .checked_add(i32::try_from(within).ok()?)?;
                }
                PromptSegment::Media { .. } => {
                    boundary.logical_tokens = segment_end;
                    boundary.native_position = boundary
                        .native_position
                        .checked_add(segment.native_positions())?;
                }
            }
            return Some(boundary);
        }
        (boundary.logical_tokens == logical_tokens).then_some(boundary)
    }

    pub(crate) fn prefix(&self, boundary: PromptBoundary) -> Option<Self> {
        if self.boundary_at(boundary.logical_tokens)? != boundary {
            return None;
        }
        let mut remaining = boundary.logical_tokens;
        let mut segments = Vec::new();
        for segment in &self.segments {
            if remaining == 0 {
                break;
            }
            let segment_tokens = segment.logical_tokens();
            if segment_tokens <= remaining {
                segments.push(segment.clone());
                remaining -= segment_tokens;
            } else if let PromptSegment::Text(tokens) = segment {
                segments.push(PromptSegment::Text(tokens[..remaining].to_vec()));
                remaining = 0;
            } else {
                return None;
            }
        }
        (remaining == 0).then_some(Self { segments })
    }
}

#[derive(Clone, Debug)]
pub(crate) enum PromptCheckpointState {
    Target(LlamaSequenceState),
    Speculative(SpeculativePromptState),
}

#[derive(Clone, Debug)]
pub(crate) struct PromptCheckpoint {
    pub(crate) state: PromptCheckpointState,
    pub(crate) boundary: PromptBoundary,
}

#[derive(Debug)]
pub(crate) struct ReusablePrefix {
    pub(crate) layout: PromptLayout,
    pub(crate) checkpoints: Vec<PromptCheckpoint>,
}

#[derive(Debug)]
pub(crate) struct AvailableSequence {
    id: i32,
    pub(crate) reusable_prefix: Option<ReusablePrefix>,
}

impl AvailableSequence {
    pub(crate) fn id(&self) -> i32 {
        self.id
    }

    pub(crate) fn activate(self) -> ActiveSequence {
        ActiveSequence { id: self.id }
    }
}

#[derive(Debug)]
pub(crate) struct ActiveSequence {
    id: i32,
}

impl ActiveSequence {
    pub(crate) fn id(&self) -> i32 {
        self.id
    }

    pub(crate) fn into_available(
        self,
        reusable_prefix: Option<ReusablePrefix>,
    ) -> AvailableSequence {
        AvailableSequence {
            id: self.id,
            reusable_prefix,
        }
    }

    /// Consume capacity whose native state could not be made safe for another request.
    pub(crate) fn quarantine(self) {}
}

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

        let mut available = capacity;
        let mut result = Vec::new();

        // Decode-first is explicit: at most one next-token decode per sequence per iteration.
        // Keep resident sequence order stable across iterations, as llama-server does. Besides
        // making completion order predictable, this lets backends reuse an identical generation
        // graph instead of cycling sequence permutations when every decode already fits.
        let mut decode_candidates = candidates
            .iter()
            .filter(|candidate| candidate.kind == WorkKind::Decode)
            .collect::<Vec<_>>();
        decode_candidates.sort_unstable_by_key(|candidate| candidate.sequence_id);
        for candidate in decode_candidates {
            if available == 0 {
                break;
            }
            result.push(BatchWork::Decode {
                sequence_id: candidate.sequence_id,
            });
            available -= 1;
        }

        let prefill_candidates = candidates
            .iter()
            .filter_map(|candidate| match candidate.kind {
                WorkKind::Decode => None,
                WorkKind::Prefill { remaining } => Some((candidate.sequence_id, remaining)),
            })
            .collect::<Vec<_>>();
        if prefill_candidates.is_empty() {
            return result;
        }

        let start = self.cursor % prefill_candidates.len();
        let mut prefill = (0..prefill_candidates.len())
            .map(|offset| prefill_candidates[(start + offset) % prefill_candidates.len()])
            .collect::<Vec<_>>();
        self.cursor = (start + 1) % prefill_candidates.len();

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
    available: VecDeque<AvailableSequence>,
}

const SLOT_PROMPT_SIMILARITY_THRESHOLD: f32 = 0.1;

impl SequencePool {
    pub(crate) fn new(count: u32) -> Self {
        Self {
            available: (0..count)
                .map(|value| AvailableSequence {
                    id: i32::try_from(value).expect("validated sequence count fits i32"),
                    reusable_prefix: None,
                })
                .collect(),
        }
    }

    pub(crate) fn acquire(&mut self) -> Option<AvailableSequence> {
        self.available.pop_front()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.available.is_empty()
    }

    pub(crate) fn acquire_matching(&mut self, prompt: &PromptLayout) -> Option<AvailableSequence> {
        let prompt_tokens = prompt.logical_tokens();
        if prompt_tokens == 0 {
            return self.available.pop_back();
        }
        let best = self
            .available
            .iter()
            .enumerate()
            .filter_map(|(index, sequence)| {
                let prefix = sequence.reusable_prefix.as_ref()?;
                let common_prefix = prefix.layout.common_prefix(prompt).logical_tokens;
                let similarity = common_prefix as f32 / prompt_tokens as f32;
                (similarity > SLOT_PROMPT_SIMILARITY_THRESHOLD).then_some((index, common_prefix))
            })
            .max_by_key(|(_, common_prefix)| *common_prefix)
            .map(|(index, _)| index);
        match best {
            Some(index) => self.available.remove(index),
            None => self.available.pop_back(),
        }
    }

    pub(crate) fn release(&mut self, sequence: AvailableSequence) {
        self.available.push_front(sequence);
    }

    /// Native context reset erases KV for available sequences as well as active ones.
    pub(crate) fn invalidate_reuse(&mut self) {
        for sequence in &mut self.available {
            sequence.reusable_prefix = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linked_position_preserves_target_and_draft_coordinate_systems() {
        let boundary = PromptBoundary {
            logical_tokens: 4_096,
            native_position: 137,
        };
        assert_eq!(
            boundary.speculative_position(),
            Some(SpeculativePosition {
                target: 137,
                draft: 4_096,
            })
        );
        assert_eq!(
            boundary.advance(3),
            Some(PromptBoundary {
                logical_tokens: 4_099,
                native_position: 140,
            })
        );
    }

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
    fn decode_order_is_stable_when_every_sequence_fits() {
        let mut planner = BatchPlanner::new(2);
        let candidates = [
            WorkCandidate {
                sequence_id: 2,
                kind: WorkKind::Decode,
            },
            WorkCandidate {
                sequence_id: 0,
                kind: WorkKind::Decode,
            },
            WorkCandidate {
                sequence_id: 1,
                kind: WorkKind::Decode,
            },
        ];
        let expected = vec![
            BatchWork::Decode { sequence_id: 0 },
            BatchWork::Decode { sequence_id: 1 },
            BatchWork::Decode { sequence_id: 2 },
        ];
        assert_eq!(planner.plan(&candidates, 3), expected);
        assert_eq!(planner.plan(&candidates, 3), expected);
    }

    #[test]
    fn sequence_ownership_is_isolated_and_reused_only_after_release() {
        let mut pool = SequencePool::new(2);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        assert_ne!(first.id(), second.id());
        assert!(pool.acquire().is_none());

        let first_id = first.id();
        pool.release(first);
        assert_eq!(pool.acquire().unwrap().id(), first_id);
        assert!(pool.acquire().is_none());
        drop(second);
    }

    #[test]
    fn failed_cleanup_quarantines_only_the_affected_sequence() {
        let mut pool = SequencePool::new(2);
        let quarantined = pool.acquire().unwrap().activate();
        let survivor = pool.acquire().unwrap();
        quarantined.quarantine();

        assert!(pool.acquire().is_none());
        let survivor_id = survivor.id();
        pool.release(survivor);
        assert_eq!(pool.acquire().unwrap().id(), survivor_id);
        assert!(pool.acquire().is_none());
    }

    #[test]
    fn returning_an_unmodified_available_sequence_preserves_its_reusable_prefix() {
        let mut pool = SequencePool::new(2);
        let available = pool.acquire().unwrap();
        let sequence_id = available.id();
        let active = available.activate();
        pool.release(active.into_available(Some(ReusablePrefix {
            layout: PromptLayout::text(vec![LlamaToken::new(7)]),
            checkpoints: Vec::new(),
        })));
        let available = pool.acquire().unwrap();
        assert_eq!(available.id(), sequence_id);
        pool.release(available);
        let returned = pool.acquire().unwrap();
        let prefix = returned.reusable_prefix.unwrap();
        assert_eq!(prefix.layout.text_tokens(), vec![LlamaToken::new(7)]);
        assert!(prefix.checkpoints.is_empty());
    }

    #[test]
    fn matching_cache_is_selected_independently_of_free_order() {
        let mut pool = SequencePool::new(2);
        let first = pool.acquire().unwrap();
        let first_id = first.id();
        let second = pool.acquire().unwrap();
        pool.release(first.activate().into_available(Some(ReusablePrefix {
            layout: PromptLayout::text(vec![LlamaToken::new(1), LlamaToken::new(2)]),
            checkpoints: Vec::new(),
        })));
        pool.release(second.activate().into_available(Some(ReusablePrefix {
            layout: PromptLayout::text(vec![LlamaToken::new(7), LlamaToken::new(8)]),
            checkpoints: Vec::new(),
        })));

        let acquired = pool
            .acquire_matching(&PromptLayout::text(vec![
                LlamaToken::new(1),
                LlamaToken::new(9),
            ]))
            .unwrap();
        assert_eq!(acquired.id(), first_id);
    }

    #[test]
    fn weak_cache_match_uses_an_empty_lru_sequence() {
        let mut pool = SequencePool::new(2);
        let cached = pool.acquire().unwrap();
        let cached_id = cached.id();
        pool.release(cached.activate().into_available(Some(ReusablePrefix {
            layout: PromptLayout::text(vec![LlamaToken::new(1)]),
            checkpoints: Vec::new(),
        })));

        let acquired = pool
            .acquire_matching(&PromptLayout::text((1..=11).map(LlamaToken::new).collect()))
            .unwrap();

        assert_ne!(acquired.id(), cached_id);
        assert!(acquired.reusable_prefix.is_none());
    }

    #[test]
    fn cache_match_must_strictly_exceed_similarity_threshold() {
        let mut pool = SequencePool::new(2);
        let cached = pool.acquire().unwrap();
        let cached_id = cached.id();
        pool.release(cached.activate().into_available(Some(ReusablePrefix {
            layout: PromptLayout::text(vec![LlamaToken::new(1)]),
            checkpoints: Vec::new(),
        })));

        let acquired = pool
            .acquire_matching(&PromptLayout::text((1..=10).map(LlamaToken::new).collect()))
            .unwrap();

        assert_ne!(acquired.id(), cached_id);
        assert!(acquired.reusable_prefix.is_none());
    }

    #[test]
    fn qualifying_cache_match_wins_over_an_empty_lru_sequence() {
        let mut pool = SequencePool::new(2);
        let cached = pool.acquire().unwrap();
        let cached_id = cached.id();
        pool.release(cached.activate().into_available(Some(ReusablePrefix {
            layout: PromptLayout::text(vec![LlamaToken::new(1)]),
            checkpoints: Vec::new(),
        })));

        let acquired = pool
            .acquire_matching(&PromptLayout::text((1..=9).map(LlamaToken::new).collect()))
            .unwrap();

        assert_eq!(acquired.id(), cached_id);
    }

    #[test]
    fn missing_qualifying_match_uses_least_recently_used_cached_sequence() {
        let mut pool = SequencePool::new(2);
        let oldest = pool.acquire().unwrap();
        let oldest_id = oldest.id();
        let newest = pool.acquire().unwrap();
        pool.release(oldest.activate().into_available(Some(ReusablePrefix {
            layout: PromptLayout::text(vec![LlamaToken::new(1)]),
            checkpoints: Vec::new(),
        })));
        pool.release(newest.activate().into_available(Some(ReusablePrefix {
            layout: PromptLayout::text(vec![LlamaToken::new(2)]),
            checkpoints: Vec::new(),
        })));

        let acquired = pool
            .acquire_matching(&PromptLayout::text(vec![LlamaToken::new(3); 11]))
            .unwrap();

        assert_eq!(acquired.id(), oldest_id);
    }

    #[test]
    fn context_reset_invalidates_available_reusable_prefixes() {
        let mut pool = SequencePool::new(1);
        let sequence = pool.acquire().unwrap().activate();
        pool.release(sequence.into_available(Some(ReusablePrefix {
            layout: PromptLayout::text(vec![LlamaToken::new(7)]),
            checkpoints: Vec::new(),
        })));

        pool.invalidate_reuse();

        assert!(pool.acquire().unwrap().reusable_prefix.is_none());
    }

    #[test]
    fn multimodal_prefix_matches_identical_media_and_tracks_mrope_positions() {
        let cached = multimodal_layout("image-a", &[1, 2], 576, 1, &[3, 4]);
        let incoming = multimodal_layout("image-a", &[1, 2], 576, 1, &[3, 9]);

        let boundary = cached.common_prefix(&incoming);
        assert_eq!(
            boundary,
            PromptBoundary {
                logical_tokens: 2 + 576 + 1,
                native_position: 2 + 1 + 1,
            }
        );
        assert_eq!(
            boundary.speculative_position(),
            Some(SpeculativePosition {
                target: 4,
                draft: 579,
            })
        );
    }

    #[test]
    fn multimodal_prefix_stops_before_changed_media() {
        let cached = multimodal_layout("image-a", &[1, 2], 64, 64, &[3]);
        let incoming = multimodal_layout("image-b", &[1, 2], 64, 64, &[3]);

        assert_eq!(
            cached.common_prefix(&incoming),
            PromptBoundary {
                logical_tokens: 2,
                native_position: 2,
            }
        );
    }

    #[test]
    fn media_cannot_be_split_by_a_cache_boundary() {
        let layout = multimodal_layout("image-a", &[1], 64, 1, &[2]);

        assert!(layout.boundary_at(32).is_none());
        assert!(
            layout
                .prefix(PromptBoundary {
                    logical_tokens: 32,
                    native_position: 32,
                })
                .is_none()
        );
    }

    #[test]
    fn checkpoint_inside_media_advances_to_the_next_semantic_boundary() {
        let layout = multimodal_layout("image-a", &[1, 2], 512, 32, &[3, 4]);

        assert_eq!(
            layout.boundary_at_or_after(128),
            Some(PromptBoundary {
                logical_tokens: 514,
                native_position: 34,
            })
        );
        assert_eq!(
            layout.boundary_at_or_after(515),
            Some(PromptBoundary {
                logical_tokens: 515,
                native_position: 35,
            })
        );
    }

    #[test]
    fn retained_partial_text_prefix_matches_longer_text_segment() {
        let complete = PromptLayout::text(vec![
            LlamaToken::new(1),
            LlamaToken::new(2),
            LlamaToken::new(3),
        ]);
        let boundary = complete.boundary_at(2).unwrap();
        let retained = complete.prefix(boundary).unwrap();

        assert_eq!(retained.common_prefix(&complete), boundary);
    }

    fn multimodal_layout(
        identity: &str,
        before: &[i32],
        media_tokens: usize,
        media_positions: i32,
        after: &[i32],
    ) -> PromptLayout {
        PromptLayout::new(vec![
            PromptSegment::Text(before.iter().copied().map(LlamaToken::new).collect()),
            PromptSegment::Media {
                identity: identity.to_owned(),
                logical_tokens: media_tokens,
                native_positions: media_positions,
            },
            PromptSegment::Text(after.iter().copied().map(LlamaToken::new).collect()),
        ])
    }

    fn batch_size(work: &BatchWork) -> usize {
        match work {
            BatchWork::Decode { .. } => 1,
            BatchWork::Prefill { tokens, .. } => *tokens,
        }
    }
}
