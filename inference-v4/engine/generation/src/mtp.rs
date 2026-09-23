use crate::{
    Method, MethodCheckpoint, MethodCheckpointError, MethodEffects, MethodRequirements,
    MethodState, MtpCheckpoint, Propose, Verification,
};
use magnitude_model_executor::{
    Demand, FeatureRef, FeatureRetainer, FeatureSpan, Operation, Outcome, RequestId,
    RetainedFeatureSpan, SelectSpec, TokenId,
};
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Mtp {
    identity: String,
    capacity: usize,
}

impl Mtp {
    pub fn new(artifact: impl AsRef<str>, capacity: usize) -> Result<Self, String> {
        let artifact = artifact.as_ref();
        if artifact.is_empty() || capacity == 0 || capacity > u8::MAX as usize {
            return Err("MTP identity and proposal capacity must be nonempty and bounded".into());
        }
        Ok(Self {
            identity: format!("mtp:{artifact}:{capacity}"),
            capacity,
        })
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Method for Mtp {
    fn identity(&self) -> &str {
        &self.identity
    }
    fn requires(&self) -> MethodRequirements {
        MethodRequirements {
            prefill_demand: Demand::FEATURES,
            verify_demand: Demand::FEATURES,
            head: true,
        }
    }
    fn create(&self, checkpoint: Option<&MethodCheckpoint>) -> Box<dyn MethodState> {
        let checkpoint = checkpoint.map(|checkpoint| match checkpoint {
            MethodCheckpoint::Mtp(checkpoint) => checkpoint.clone(),
            MethodCheckpoint::Plain => panic!("plain checkpoint supplied to MTP method"),
        });
        Box::new(MtpState::new(self.capacity, checkpoint))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeadRole {
    Causal,
    Anchor,
    Speculative,
}

#[derive(Clone, Debug)]
struct HeadPlan {
    tokens: Vec<TokenId>,
    conditioning: FeatureSpan,
    retained: Vec<RetainedFeatureSpan>,
    role: HeadRole,
}

#[derive(Clone, Debug)]
enum FeatureRow {
    Transient(FeatureSpan),
    Retained(RetainedFeatureSpan),
}

impl FeatureRow {
    fn span(&self) -> FeatureSpan {
        match self {
            Self::Transient(span) => span.clone(),
            Self::Retained(feature) => feature.span(),
        }
    }

    fn into_parts(self) -> (FeatureSpan, Vec<RetainedFeatureSpan>) {
        match self {
            Self::Transient(span) => (span, Vec::new()),
            Self::Retained(feature) => (feature.span(), vec![feature]),
        }
    }

    fn retained_bytes(&self) -> u64 {
        match self {
            Self::Transient(_) => 0,
            Self::Retained(feature) => feature.bytes(),
        }
    }
}

#[derive(Clone, Debug)]
enum Pending {
    Head {
        operation: Operation,
        role: HeadRole,
        retained: Vec<RetainedFeatureSpan>,
    },
    Project(Operation),
}

#[derive(Clone)]
struct MtpState {
    capacity: usize,
    position: usize,
    pending_feature: Option<FeatureRow>,
    buffer: Vec<(TokenId, FeatureRow)>,
    queue: VecDeque<HeadPlan>,
    pending: Option<Pending>,
    draft_feature: Option<FeatureRef>,
    appended: usize,
    proposal_limit: usize,
    proposed: Vec<TokenId>,
    proposal_ready: bool,
}

impl MtpState {
    fn new(capacity: usize, checkpoint: Option<MtpCheckpoint>) -> Self {
        let checkpoint = checkpoint.unwrap_or(MtpCheckpoint {
            position: 0,
            pending: None,
            buffer: Vec::new(),
        });
        Self {
            capacity,
            position: checkpoint.position,
            pending_feature: checkpoint.pending.map(FeatureRow::Retained),
            buffer: checkpoint
                .buffer
                .into_iter()
                .map(|(token, feature)| (token, FeatureRow::Retained(feature)))
                .collect(),
            queue: VecDeque::new(),
            pending: None,
            draft_feature: None,
            appended: 0,
            proposal_limit: 0,
            proposed: Vec::new(),
            proposal_ready: false,
        }
    }

    fn ensure_idle(&self) -> Result<(), String> {
        if self.pending.is_some() || !self.queue.is_empty() || self.proposal_limit != 0 {
            return Err("MTP method already has unresolved work".into());
        }
        Ok(())
    }

    fn enqueue_buffer(&mut self) {
        let mut buffered = std::mem::take(&mut self.buffer).into_iter().peekable();
        while let Some((token, first)) = buffered.next() {
            let first_span = first.span();
            let mut tokens = vec![token];
            let mut retained = Vec::new();
            if let FeatureRow::Retained(feature) = first {
                retained.push(feature);
            }
            let mut count = 1usize;
            while let Some((_, next)) = buffered.peek() {
                let next = next.span();
                if next.features != first_span.features || next.start != first_span.start + count {
                    break;
                }
                let (token, next) = buffered.next().unwrap();
                if let FeatureRow::Retained(feature) = next {
                    retained.push(feature);
                }
                tokens.push(token);
                count += 1;
            }
            self.queue.push_back(HeadPlan {
                tokens,
                conditioning: FeatureSpan {
                    features: first_span.features,
                    start: first_span.start,
                    count,
                },
                retained,
                role: HeadRole::Causal,
            });
        }
    }

    fn emit_next_head(&mut self, request: RequestId) -> Result<Option<Operation>, String> {
        let Some(plan) = self.queue.pop_front() else {
            return Ok(None);
        };
        let operation = Operation::Head {
            request,
            tokens: plan.tokens,
            conditioning: plan.conditioning,
            position: self
                .position
                .checked_add(self.appended)
                .ok_or("MTP head position exhausted")?,
            demand: match plan.role {
                HeadRole::Causal => Demand::NONE,
                HeadRole::Anchor | HeadRole::Speculative => Demand::FEATURES,
            },
        };
        operation.validate().map_err(|error| error.to_string())?;
        self.pending = Some(Pending::Head {
            operation: operation.clone(),
            role: plan.role,
            retained: plan.retained,
        });
        Ok(Some(operation))
    }

    fn project(
        &mut self,
        request: RequestId,
        features: FeatureRef,
        select: SelectSpec,
    ) -> Operation {
        let operation = Operation::Project {
            request,
            features,
            select,
        };
        self.pending = Some(Pending::Project(operation.clone()));
        operation
    }

    fn effects(operation: Option<Operation>, head_prefix: Option<usize>) -> MethodEffects {
        MethodEffects {
            operations: operation.into_iter().collect(),
            head_prefix,
        }
    }
}

impl MethodState for MtpState {
    fn fork_transition(&self) -> Result<Box<dyn MethodState>, String> {
        Ok(Box::new(self.clone()))
    }
    fn prime(
        &mut self,
        request: RequestId,
        tokens: &[TokenId],
        features: FeatureRef,
    ) -> Result<MethodEffects, String> {
        self.ensure_idle()?;
        if tokens.is_empty() {
            return Err("MTP cannot prime an empty target chunk".into());
        }
        self.enqueue_buffer();
        if let Some(pending) = self.pending_feature.take() {
            let (conditioning, retained) = pending.into_parts();
            self.queue.push_back(HeadPlan {
                tokens: vec![tokens[0]],
                conditioning,
                retained,
                role: HeadRole::Causal,
            });
        }
        if tokens.len() > 1 {
            self.queue.push_back(HeadPlan {
                tokens: tokens[1..].to_vec(),
                conditioning: FeatureSpan {
                    features: features.clone(),
                    start: 0,
                    count: tokens.len() - 1,
                },
                retained: Vec::new(),
                role: HeadRole::Causal,
            });
        }
        self.pending_feature = Some(FeatureRow::Transient(FeatureSpan {
            features,
            start: tokens.len() - 1,
            count: 1,
        }));
        Ok(Self::effects(self.emit_next_head(request)?, None))
    }

    fn propose(
        &mut self,
        request: RequestId,
        context: &[TokenId],
        limit: usize,
        _first_select: SelectSpec,
    ) -> Propose {
        if self.proposal_ready {
            return Propose::Tokens(self.proposed.clone());
        }
        if self.pending.is_some() {
            return Propose::Pending(Vec::new());
        }
        let limit = limit.min(self.capacity);
        let Some(conditioning) = self.pending_feature.take() else {
            return Propose::Tokens(Vec::new());
        };
        let Some(&anchor) = context.last() else {
            self.pending_feature = Some(conditioning);
            return Propose::Tokens(Vec::new());
        };
        if limit == 0 {
            self.pending_feature = Some(conditioning);
            return Propose::Tokens(Vec::new());
        }
        self.enqueue_buffer();
        let (conditioning, retained) = conditioning.into_parts();
        self.queue.push_back(HeadPlan {
            tokens: vec![anchor],
            conditioning,
            retained,
            role: HeadRole::Anchor,
        });
        self.proposal_limit = limit;
        self.proposed.clear();
        self.appended = 0;
        self.draft_feature = None;
        match self.emit_next_head(request) {
            Ok(Some(operation)) => Propose::Pending(vec![operation]),
            Ok(None) | Err(_) => Propose::Pending(Vec::new()),
        }
    }

    fn observe(&mut self, verification: Verification<'_>) -> Result<MethodEffects, String> {
        if self.pending.is_some() || !self.queue.is_empty() {
            return Err("MTP cannot observe while method work is unresolved".into());
        }
        let features = verification
            .features
            .ok_or("MTP verification requires target features")?;
        if verification.inputs.is_empty()
            || verification.accepted == 0
            || verification.accepted > verification.inputs.len()
        {
            return Err("MTP verification prefix is invalid".into());
        }
        let accepted_after_anchor = verification.accepted - 1;
        let keep = accepted_after_anchor.min(self.appended);
        let head_prefix = (self.appended > 0).then_some(keep);
        if let Some(pending) = self.pending_feature.take() {
            self.buffer.push((verification.inputs[0], pending));
        }
        for index in keep..accepted_after_anchor {
            self.buffer.push((
                verification.inputs[index + 1],
                FeatureRow::Transient(FeatureSpan {
                    features: features.clone(),
                    start: index,
                    count: 1,
                }),
            ));
        }
        self.pending_feature = Some(FeatureRow::Transient(FeatureSpan {
            features,
            start: accepted_after_anchor,
            count: 1,
        }));
        self.position = self
            .position
            .checked_add(keep)
            .ok_or("MTP head position exhausted")?;
        self.appended = 0;
        self.proposal_limit = 0;
        self.proposed.clear();
        self.proposal_ready = false;
        self.draft_feature = None;
        Ok(Self::effects(None, head_prefix))
    }

    fn reconcile(
        &mut self,
        operation: &Operation,
        outcome: Outcome,
        next_select: Option<SelectSpec>,
    ) -> Result<MethodEffects, String> {
        let pending = self
            .pending
            .take()
            .ok_or("MTP has no pending method operation")?;
        let request = operation.request();
        match (pending, operation, outcome) {
            (
                Pending::Head {
                    operation: expected,
                    role,
                    retained: _retained,
                },
                operation @ Operation::Head { tokens, .. },
                Outcome::Head { features },
            ) if expected == *operation => match role {
                HeadRole::Causal => {
                    self.position = self
                        .position
                        .checked_add(tokens.len())
                        .ok_or("MTP head position exhausted")?;
                    Ok(Self::effects(
                        self.emit_next_head(request)?,
                        Some(tokens.len()),
                    ))
                }
                HeadRole::Anchor => {
                    self.position = self
                        .position
                        .checked_add(tokens.len())
                        .ok_or("MTP head position exhausted")?;
                    self.draft_feature = Some(features.clone());
                    let select = next_select.ok_or("MTP anchor has no proposal selection")?;
                    let project = self.project(request, features, select);
                    Ok(Self::effects(Some(project), Some(tokens.len())))
                }
                HeadRole::Speculative => {
                    self.appended = self
                        .appended
                        .checked_add(tokens.len())
                        .ok_or("MTP appended width exhausted")?;
                    self.draft_feature = Some(features.clone());
                    let select = next_select.ok_or("MTP speculative row has no selection")?;
                    let project = self.project(request, features, select);
                    Ok(Self::effects(Some(project), None))
                }
            },
            (
                Pending::Project(expected),
                operation @ Operation::Project { .. },
                Outcome::Project { selected },
            ) if expected == *operation => {
                let [selected] = selected.as_slice() else {
                    return Err("MTP project returned the wrong selection count".into());
                };
                self.proposed.push(selected.token);
                if let Some(_select) = next_select {
                    if self.proposed.len() >= self.proposal_limit {
                        return Err("MTP received a selection beyond its proposal limit".into());
                    }
                    let conditioning = self
                        .draft_feature
                        .clone()
                        .ok_or("MTP lost the preceding draft feature")?;
                    self.queue.push_back(HeadPlan {
                        tokens: vec![selected.token],
                        conditioning: FeatureSpan {
                            features: conditioning,
                            start: 0,
                            count: 1,
                        },
                        retained: Vec::new(),
                        role: HeadRole::Speculative,
                    });
                    Ok(Self::effects(self.emit_next_head(request)?, None))
                } else {
                    self.proposal_ready = true;
                    Ok(MethodEffects::default())
                }
            }
            _ => Err("MTP method operation or outcome does not match pending work".into()),
        }
    }

    fn stabilize(&mut self, retainer: &mut dyn FeatureRetainer) -> Result<(), String> {
        fn stabilize_row(
            row: &mut FeatureRow,
            retainer: &mut dyn FeatureRetainer,
        ) -> Result<(), String> {
            if let FeatureRow::Transient(span) = row {
                *row = FeatureRow::Retained(retainer.retain(span.clone())?);
            }
            Ok(())
        }

        if let Some(pending) = &mut self.pending_feature {
            stabilize_row(pending, retainer)?;
        }
        for (_, feature) in &mut self.buffer {
            stabilize_row(feature, retainer)?;
        }
        Ok(())
    }

    fn checkpoint(
        &self,
        retainer: &mut dyn FeatureRetainer,
    ) -> Result<MethodCheckpoint, MethodCheckpointError> {
        if self.pending.is_some()
            || !self.queue.is_empty()
            || self.proposal_limit != 0
            || self.proposal_ready
            || self.appended != 0
        {
            return Err(MethodCheckpointError::Unresolved);
        }
        let copy = |feature: &FeatureRow,
                    retainer: &mut dyn FeatureRetainer|
         -> Result<RetainedFeatureSpan, MethodCheckpointError> {
            retainer
                .retain(feature.span())
                .map_err(MethodCheckpointError::Retention)
        };
        Ok(MethodCheckpoint::Mtp(MtpCheckpoint {
            position: self.position,
            pending: self
                .pending_feature
                .as_ref()
                .map(|feature| copy(feature, retainer))
                .transpose()?,
            buffer: self
                .buffer
                .iter()
                .map(|(token, feature)| copy(feature, retainer).map(|feature| (*token, feature)))
                .collect::<Result<_, _>>()?,
        }))
    }

    fn evict(&mut self) {
        self.position = 0;
        self.pending_feature = None;
        self.buffer.clear();
        self.queue.clear();
        self.pending = None;
        self.draft_feature = None;
        self.appended = 0;
        self.proposal_limit = 0;
        self.proposed.clear();
        self.proposal_ready = false;
    }
    fn restore(&mut self) {}
    fn reclaimable(&self) -> u64 {
        self.pending_feature
            .iter()
            .chain(self.buffer.iter().map(|(_, feature)| feature))
            .fold(0, |total, feature| {
                total.saturating_add(feature.retained_bytes())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_model_executor::{ResourceDomainId, Sampling, Selected, Shaping};

    struct TestRetainer {
        next: u64,
        bytes_per_row: u64,
    }

    impl TestRetainer {
        fn new(next: u64) -> Self {
            Self {
                next,
                bytes_per_row: 32,
            }
        }
    }

    impl FeatureRetainer for TestRetainer {
        fn retain(&mut self, span: FeatureSpan) -> Result<RetainedFeatureSpan, String> {
            let id = self.next;
            self.next += 1;
            let copied = FeatureSpan {
                features: feature(id),
                start: 0,
                count: span.count,
            };
            RetainedFeatureSpan::new(copied, self.bytes_per_row * span.count as u64)
                .map_err(|error| error.to_string())
        }
    }

    fn feature(id: u64) -> FeatureRef {
        FeatureRef::logical(ResourceDomainId::new(format!("test-{id}")).unwrap(), 8, 8).unwrap()
    }

    fn select(position: usize) -> SelectSpec {
        SelectSpec {
            sampling: Sampling::Greedy,
            seed: 1,
            position,
            domain: 1,
            mask: None,
            shaping: Shaping {
                temperature: 0.0,
                ..Default::default()
            },
            history: None,
        }
    }

    fn head(effects: &MethodEffects) -> &Operation {
        let [operation @ Operation::Head { .. }] = effects.operations.as_slice() else {
            panic!("expected one head operation")
        };
        operation
    }

    fn primed_state() -> MtpState {
        let request = RequestId(4);
        let mut state = MtpState::new(2, None);
        let first = state
            .prime(request, &[TokenId(1), TokenId(2)], feature(10))
            .unwrap();
        state.stabilize(&mut TestRetainer::new(1_000)).unwrap();
        let Operation::Head {
            tokens,
            conditioning,
            position,
            demand,
            ..
        } = head(&first)
        else {
            unreachable!()
        };
        assert_eq!(tokens, &[TokenId(2)]);
        assert_eq!(
            *conditioning,
            FeatureSpan {
                features: feature(10),
                start: 0,
                count: 1
            }
        );
        assert_eq!((*position, *demand), (0, Demand::NONE));
        let committed = state
            .reconcile(
                head(&first),
                Outcome::Head {
                    features: feature(90),
                },
                None,
            )
            .unwrap();
        assert_eq!(committed.head_prefix, Some(1));
        assert!(committed.operations.is_empty());
        state
    }

    #[test]
    fn prime_pairs_shifted_rows_and_sequences_disparate_leases() {
        let request = RequestId(4);
        let mut state = primed_state();
        let first = state
            .prime(request, &[TokenId(3), TokenId(4)], feature(11))
            .unwrap();
        let Operation::Head {
            tokens,
            conditioning,
            position,
            ..
        } = head(&first)
        else {
            unreachable!()
        };
        assert_eq!(tokens, &[TokenId(3)]);
        assert_eq!(
            *conditioning,
            FeatureSpan {
                features: feature(1_000),
                start: 0,
                count: 1
            }
        );
        assert_eq!(*position, 1);
        let second = state
            .reconcile(
                head(&first),
                Outcome::Head {
                    features: feature(91),
                },
                None,
            )
            .unwrap();
        assert_eq!(second.head_prefix, Some(1));
        let Operation::Head {
            tokens,
            conditioning,
            position,
            ..
        } = head(&second)
        else {
            unreachable!()
        };
        assert_eq!(tokens, &[TokenId(4)]);
        assert_eq!(
            *conditioning,
            FeatureSpan {
                features: feature(11),
                start: 0,
                count: 1
            }
        );
        assert_eq!(*position, 2);
    }

    fn proposed_state() -> MtpState {
        let request = RequestId(4);
        let mut state = primed_state();
        let Propose::Pending(anchor) =
            state.propose(request, &[TokenId(1), TokenId(2), TokenId(5)], 2, select(0))
        else {
            panic!("expected anchor head")
        };
        let anchor_effects = state
            .reconcile(
                &anchor[0],
                Outcome::Head {
                    features: feature(20),
                },
                Some(select(0)),
            )
            .unwrap();
        assert_eq!(anchor_effects.head_prefix, Some(1));
        let first_project = &anchor_effects.operations[0];
        let speculative = state
            .reconcile(
                first_project,
                Outcome::Project {
                    selected: vec![Selected {
                        token: TokenId(6),
                        status: 0,
                    }],
                },
                Some(select(1)),
            )
            .unwrap();
        assert_eq!(speculative.head_prefix, None);
        let speculative_head = &speculative.operations[0];
        let Operation::Head {
            tokens, position, ..
        } = speculative_head
        else {
            panic!("expected speculative head")
        };
        assert_eq!(tokens, &[TokenId(6)]);
        assert_eq!(*position, 2);
        let second_project = state
            .reconcile(
                speculative_head,
                Outcome::Head {
                    features: feature(21),
                },
                Some(select(1)),
            )
            .unwrap();
        assert_eq!(second_project.head_prefix, None);
        let done = state
            .reconcile(
                &second_project.operations[0],
                Outcome::Project {
                    selected: vec![Selected {
                        token: TokenId(7),
                        status: 0,
                    }],
                },
                None,
            )
            .unwrap();
        assert_eq!(done, MethodEffects::default());
        assert_eq!(
            state.propose(request, &[TokenId(5)], 2, select(0)),
            Propose::Tokens(vec![TokenId(6), TokenId(7)])
        );
        state
    }

    #[test]
    fn coupled_proposals_chain_and_observe_partial_prefix() {
        let mut state = proposed_state();
        assert!(state.checkpoint(&mut TestRetainer::new(1_500)).is_err());
        let effects = state
            .observe(Verification {
                inputs: &[TokenId(5), TokenId(6), TokenId(7)],
                accepted: 2,
                next: TokenId(8),
                features: Some(feature(30)),
            })
            .unwrap();
        assert_eq!(effects.head_prefix, Some(1));
        let mut retainer = TestRetainer::new(2_000);
        state.stabilize(&mut retainer).unwrap();
        let MethodCheckpoint::Mtp(checkpoint) = state
            .checkpoint(&mut retainer)
            .expect("checkpoint copied retained feature rows")
        else {
            panic!("expected MTP checkpoint")
        };
        assert_eq!(checkpoint.retained_bytes(), 32);
    }

    #[test]
    fn observe_can_drop_all_or_keep_all_speculative_rows() {
        let mut dropped = proposed_state();
        let effects = dropped
            .observe(Verification {
                inputs: &[TokenId(5), TokenId(6), TokenId(7)],
                accepted: 1,
                next: TokenId(8),
                features: Some(feature(31)),
            })
            .unwrap();
        assert_eq!(effects.head_prefix, Some(0));

        let mut kept = proposed_state();
        let effects = kept
            .observe(Verification {
                inputs: &[TokenId(5), TokenId(6), TokenId(7)],
                accepted: 3,
                next: TokenId(8),
                features: Some(feature(32)),
            })
            .unwrap();
        assert_eq!(effects.head_prefix, Some(1));
        assert_eq!(kept.buffer.len(), 1);
        assert_eq!(kept.buffer[0].0, TokenId(7));
        assert_eq!(kept.pending_feature.as_ref().unwrap().span().start, 2);
        let mut retainer = TestRetainer::new(3_000);
        kept.stabilize(&mut retainer).unwrap();
        assert_eq!(kept.reclaimable(), 64);
        let MethodCheckpoint::Mtp(checkpoint) = kept
            .checkpoint(&mut retainer)
            .expect("buffer and pending rows are copy-stable")
        else {
            panic!("expected MTP checkpoint")
        };
        assert_eq!(checkpoint.retained_bytes(), 64);

        kept.evict();
        assert_eq!(kept.reclaimable(), 0);
        kept.restore();
        let MethodCheckpoint::Mtp(checkpoint) =
            kept.checkpoint(&mut TestRetainer::new(4_000)).unwrap()
        else {
            panic!("expected MTP checkpoint")
        };
        assert_eq!(checkpoint.position(), 0);
    }

    #[test]
    fn no_drafting_and_forced_blocks_preserve_pairing_without_head_rewind() {
        let request = RequestId(4);

        let mut no_draft = primed_state();
        assert_eq!(
            no_draft.propose(request, &[TokenId(2)], 0, select(0)),
            Propose::Tokens(Vec::new())
        );
        let effects = no_draft
            .observe(Verification {
                inputs: &[TokenId(2)],
                accepted: 1,
                next: TokenId(5),
                features: Some(feature(40)),
            })
            .unwrap();
        assert_eq!(effects.head_prefix, None);
        assert_eq!(no_draft.buffer.len(), 1);
        assert_eq!(no_draft.buffer[0].0, TokenId(2));
        assert_eq!(no_draft.pending_feature.unwrap().span().start, 0);

        let mut forced = primed_state();
        let effects = forced
            .observe(Verification {
                inputs: &[TokenId(2), TokenId(5), TokenId(6)],
                accepted: 3,
                next: TokenId(7),
                features: Some(feature(41)),
            })
            .unwrap();
        assert_eq!(effects.head_prefix, None);
        assert_eq!(
            forced
                .buffer
                .iter()
                .map(|(token, _)| *token)
                .collect::<Vec<_>>(),
            [TokenId(2), TokenId(5), TokenId(6)]
        );
        assert_eq!(forced.pending_feature.unwrap().span().start, 2);
    }

    #[test]
    fn checkpoint_copies_rows_and_forks_diverge_without_borrowed_leases() {
        let mut source = primed_state();
        source
            .observe(Verification {
                inputs: &[TokenId(2)],
                accepted: 1,
                next: TokenId(5),
                features: Some(feature(40)),
            })
            .unwrap();
        let mut live_retainer = TestRetainer::new(1_000);
        source.stabilize(&mut live_retainer).unwrap();
        assert_eq!(source.reclaimable(), 64);

        let mut first_retainer = TestRetainer::new(10_000);
        let MethodCheckpoint::Mtp(first_checkpoint) = source
            .checkpoint(&mut first_retainer)
            .expect("first checkpoint")
        else {
            panic!("expected MTP checkpoint")
        };
        let mut checkpoint_state = MtpState::new(2, Some(first_checkpoint.clone()));
        let mut second_retainer = TestRetainer::new(20_000);
        let MethodCheckpoint::Mtp(second_checkpoint) = checkpoint_state
            .checkpoint(&mut second_retainer)
            .expect("fork checkpoint")
        else {
            panic!("expected MTP checkpoint")
        };

        assert_eq!(first_checkpoint.retained_bytes(), 64);
        assert_eq!(second_checkpoint.retained_bytes(), 64);
        assert_ne!(
            first_checkpoint.pending.as_ref().unwrap().span().features,
            second_checkpoint.pending.as_ref().unwrap().span().features
        );
        assert_ne!(
            first_checkpoint.buffer[0].1.span().features,
            second_checkpoint.buffer[0].1.span().features
        );

        let mut first = MtpState::new(2, Some(first_checkpoint));
        let mut second = MtpState::new(2, Some(second_checkpoint));
        first
            .observe(Verification {
                inputs: &[TokenId(5)],
                accepted: 1,
                next: TokenId(6),
                features: Some(feature(50)),
            })
            .unwrap();
        second
            .observe(Verification {
                inputs: &[TokenId(9)],
                accepted: 1,
                next: TokenId(10),
                features: Some(feature(60)),
            })
            .unwrap();
        assert_eq!(first.buffer.last().unwrap().0, TokenId(5));
        assert_eq!(second.buffer.last().unwrap().0, TokenId(9));
        checkpoint_state.evict();
        assert_eq!(checkpoint_state.reclaimable(), 0);
    }

    #[test]
    fn eviction_drops_owned_rows_and_restore_is_reprimed_by_replay() {
        let request = RequestId(4);
        let mut state = primed_state();
        assert_eq!(state.reclaimable(), 32);
        state.evict();
        assert_eq!(state.reclaimable(), 0);
        state.restore();
        assert_eq!(
            state.propose(request, &[TokenId(2)], 1, select(0)),
            Propose::Tokens(Vec::new())
        );

        let replay = state
            .prime(request, &[TokenId(1), TokenId(2)], feature(70))
            .unwrap();
        state.stabilize(&mut TestRetainer::new(30_000)).unwrap();
        assert_eq!(state.reclaimable(), 32);
        let Operation::Head {
            tokens,
            conditioning,
            position,
            ..
        } = head(&replay)
        else {
            unreachable!()
        };
        assert_eq!(tokens, &[TokenId(2)]);
        assert_eq!(*conditioning, FeatureSpan::new(feature(70), 0, 1).unwrap());
        assert_eq!(*position, 0);
    }
}
