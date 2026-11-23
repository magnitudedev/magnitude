//! The MTP head of its own history on a load that fronts
//! (`magnitude_executor::TwoRowDecode`): its generations never propose on
//! their own. Every decode round is a pass in the form its launch grants
//! (§7.1), on the chain of guesses the head made past the pending token:
//!
//! - A **front pass** (`FrontGrant::UpTo`, `FrontGrant::Exactly`) runs the
//!   chain as front rows and enters the head on the device right behind it
//!   (`ChainEntry`): the head commits one entry row, the pass's exact
//!   selection paired with the exact row's feature, and its selections make
//!   the next pass's chain. No feature crosses to the host.
//! - A **verify pass** (`FrontGrant::Verify`) runs the chain as acceptable
//!   rows and enters nothing behind it: its accepted rows' pairs, their
//!   features read to the host, wait for the next round, which first enters
//!   them in one host head transaction that drafts the next chain from the
//!   last of them (the drafting MTP's [`HeadTransaction`]).
//!
//! So the head's history either ends level with the target, past it by the
//! pending token ([`DraftTail::Entered`]), where a round runs either form;
//! or trails it ([`DraftTail::Trailing`]), where a round is that head
//! transaction, whatever its grant, and leaves the head level again. The
//! prompt's closing chunk enters the head on the device behind its drafter
//! entry, with its selection and boundary feature, as a front pass does;
//! history the device did not enter (a replay, or a chunk entered on the
//! host because the head trailed it) trails, as a verify pass's does.

use crate::{
    method::{DraftTail, OpenFeature, PendingRows},
    mtp::{DrafterState, HeadTransaction},
    DraftCheckpoint, FrontEntry, MethodCheckpoint, MethodCheckpointError, MethodEffects, MethodState, Propose,
    Verification,
};
use magnitude_executor::{
    ChainEntry, DraftForm, FeatureReader, FeatureRef, FrontGrant, HeadPhase, Operation, Outcome, RequestId,
    SelectSpec, TokenId,
};

/// Unentered verify pairs whose conditioning features remain on the device
/// (`FeatureRef`, zero D2H copy) until the immediately following `Operation::Head`
/// consumes them on-device. Never retained inside a checkpoint.
#[derive(Clone)]
struct DevicePending {
    tokens: Vec<TokenId>,
    features: FeatureRef,
}

/// A head decoding in front and verify passes, of its own history.
#[derive(Clone)]
pub(crate) struct FrontState {
    /// Head rows entered: the next entry row's position.
    position: usize,
    /// Where its history ends: level with the target's rows once it entered
    /// their successor, on the device behind a selecting step or on the
    /// host by a head transaction; trailing them by the pairs a verify
    /// pass accepted, or history the device did not enter.
    tail: DraftTail,
    /// On-device pending pairs from a verify pass (`PassForm::Verify`), kept
    /// off the host until consumed by the next round's `Operation::Head`.
    pending_device: Option<DevicePending>,
    /// The in-flight round work, if any: a target pass started from the
    /// level head, or a host head transaction entering pairs.
    round: FrontRound,
}

/// The form of a pass ([`FrontState::front`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PassForm {
    /// Front rows, the head entering its selection behind it on the device.
    Front,
    /// Acceptable rows, entering nothing behind it.
    Verify,
}

/// In-flight round state of a fronting head: a target pass and a host head
/// transaction are mutually exclusive.
#[derive(Clone)]
enum FrontRound {
    /// No pass or host head transaction is in flight.
    Idle,
    /// A target pass started from the level head, until it is observed
    /// (or superseded / checkpointed after abandonment).
    Pass(PassForm),
    /// A host head transaction entering pairs, until it reconciles.
    Head(HeadTransaction),
}

impl FrontState {
    pub(crate) fn restore(checkpoint: Option<&MethodCheckpoint>) -> Result<Self, String> {
        let state = match checkpoint {
            None => {
                return Ok(Self {
                    position: 0,
                    tail: DraftTail::EMPTY,
                    pending_device: None,
                    round: FrontRound::Idle,
                });
            }
            Some(MethodCheckpoint::Mtp(state)) => state,
            Some(_) => return Err("another method's checkpoint cannot restore MTP state".into()),
        };
        if state.skipped != 0 {
            return Err("a head decoding in front passes omits no target row".into());
        }
        Ok(Self {
            position: state.position,
            tail: state.tail.clone(),
            pending_device: None,
            round: FrontRound::Idle,
        })
    }

    /// Enter `pairs` as a head transaction at the head's position, drafting
    /// a proposal per `proposals`.
    fn enter(
        &mut self,
        request: RequestId,
        pairs: PendingRows,
        phase: HeadPhase,
        proposals: &[SelectSpec],
    ) -> Result<MethodEffects, String> {
        let transaction =
            HeadTransaction::new(request, phase, pairs, self.position, proposals.to_vec(), DraftForm::Chained)?;
        let operation = transaction.operation().clone();
        self.round = FrontRound::Head(transaction);
        Ok(MethodEffects {
            operations: vec![operation],
        })
    }

    /// Enter `tokens` conditioned by leading `0..tokens.len()` rows of on-device
    /// `features` at the head's position, drafting a proposal per `proposals`.
    fn enter_device(
        &mut self,
        request: RequestId,
        tokens: Vec<TokenId>,
        features: FeatureRef,
        phase: HeadPhase,
        proposals: &[SelectSpec],
    ) -> Result<MethodEffects, String> {
        let transaction = HeadTransaction::new_device(
            request,
            phase,
            tokens,
            features,
            self.position,
            proposals.to_vec(),
            DraftForm::Chained,
        )?;
        let operation = transaction.operation().clone();
        self.round = FrontRound::Head(transaction);
        Ok(MethodEffects {
            operations: vec![operation],
        })
    }
}

impl MethodState for FrontState {
    fn fork_transition(&self) -> Box<dyn MethodState> {
        Box::new(self.clone())
    }

    /// The head enters a chunk on the device where it holds every pair
    /// before the chunk's: trailing it by the last row's feature, if any,
    /// the device's, its entry pairs the chunk's first token with that
    /// feature; level with the target, having entered the chunk's first
    /// token (a prefix resumed past a closing chunk's or a pass's entry),
    /// its entry is the chunk's pairs past that token (`Priming`).
    fn priming_position(&self) -> Option<usize> {
        if self.pending_device.is_some() {
            return None;
        }
        match (&self.tail, &self.round) {
            (
                DraftTail::Trailing {
                    pending: None,
                    open: None | Some(OpenFeature::Device(_)),
                }
                | DraftTail::Entered(_),
                FrontRound::Idle,
            ) => Some(self.position),
            _ => None,
        }
    }

    fn prime(
        &mut self,
        request: RequestId,
        tokens: &[TokenId],
        next: Option<TokenId>,
        features: Option<FeatureRef>,
        draft_from: usize,
        primed: Option<usize>,
        reader: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String> {
        match self.round {
            FrontRound::Idle => {}
            FrontRound::Pass(_) => return Err("MTP cannot prime while a pass is in flight".into()),
            FrontRound::Head(_) => return Err("MTP already has an unresolved head transaction".into()),
        }
        let features = features.ok_or("MTP priming requires the chunk's target features")?;
        if let Some(primed) = primed {
            if self.priming_position() != Some(self.position) || primed > tokens.len() {
                return Err("MTP state cannot take a device-primed chunk".into());
            }
            if matches!(self.tail, DraftTail::Entered(token) if tokens.first() != Some(&token)) {
                return Err("the chunk's first token differs from the successor MTP entered".into());
            }
            self.position = self
                .position
                .checked_add(primed)
                .ok_or("MTP head position exhausted")?;
            self.tail = match next {
                // The chunk closing the prompt selected `next`, and its head
                // launch behind it entered it with the chunk's boundary
                // feature (`ChainEntry`).
                Some(next) => {
                    self.position = self
                        .position
                        .checked_add(1)
                        .ok_or("MTP head position exhausted")?;
                    DraftTail::Entered(next)
                }
                None => DraftTail::Trailing {
                    pending: None,
                    open: Some(OpenFeature::Device(
                        features
                            .boundary()
                            .ok_or("MTP priming requires an independently owned boundary feature")?
                            .downgrade(),
                    )),
                },
            };
            return Ok(MethodEffects::default());
        }
        let Some((&first, rest)) = tokens.split_first() else {
            return Err("MTP cannot prime an empty target chunk".into());
        };
        let rows = DrafterState::read(reader, features, 0, tokens.len())?;
        // Each chunk row's feature selected the token after it; the head
        // pairs the chunk's first token with the feature before the chunk,
        // unless it entered that pair already.
        let tail = std::mem::replace(&mut self.tail, DraftTail::EMPTY);
        let mut pending = match (self.pending_device.take(), tail) {
            (Some(device), _) => {
                let device_rows = DrafterState::read(reader, device.features, 0, device.tokens.len())?;
                Some(PendingRows::joined(None, device.tokens, device_rows)?)
            }
            (None, DraftTail::Entered(token)) if token == first => None,
            (None, DraftTail::Entered(_)) => {
                return Err("the chunk's first token differs from the successor MTP entered".into());
            }
            (None, DraftTail::Trailing { pending, open: Some(open) }) => Some(PendingRows::joined(
                pending,
                vec![first],
                DrafterState::open_rows(reader, open)?,
            )?),
            (None, DraftTail::Trailing { pending, open: None }) => pending,
        };
        if !rest.is_empty() {
            pending = Some(PendingRows::joined(
                pending,
                rest.to_vec(),
                rows.slice(0, rest.len()).map_err(|error| error.to_string())?,
            )?);
        }
        let last = rows
            .slice(tokens.len() - 1, 1)
            .map_err(|error| error.to_string())?;
        self.tail = match next {
            Some(next) => {
                pending = Some(PendingRows::joined(pending, vec![next], last)?);
                DraftTail::Entered(next)
            }
            None => DraftTail::Trailing {
                pending: None,
                open: Some(OpenFeature::Host(last)),
            },
        };
        match pending {
            Some(pairs) => self.enter(request, pairs, HeadPhase::Priming { draft_from }, &[]),
            None => Ok(MethodEffects::default()),
        }
    }

    fn propose(&mut self, _: RequestId, _: &[SelectSpec]) -> Propose {
        unreachable!("a head decoding in front passes proposes nothing")
    }

    /// Level with the target, having entered the pending token, the round
    /// is its grant's pass: a front pass entering the head behind it from
    /// its position, or a verify pass entering nothing. Trailing it, the
    /// round is the head transaction entering every pair the head lacks,
    /// the last the pending token's (with the open feature that selected
    /// it, or a verify pass's last accepted pair), and drafting the next
    /// chain from it, a proposal per `selects`; it leaves the head level.
    fn front(
        &mut self,
        request: RequestId,
        pending: TokenId,
        grant: FrontGrant,
        selects: &[SelectSpec],
    ) -> Result<FrontEntry, String> {
        match self.round {
            // An abandoned target pass leaves Pass(_) without mutating position
            // or tail; starting a new round supersedes it.
            FrontRound::Idle | FrontRound::Pass(_) => {}
            FrontRound::Head(_) => return Err("MTP already has an unresolved head transaction".into()),
        }
        if let Some(device) = self.pending_device.take() {
            if device.tokens.last() != Some(&pending) {
                return Err("MTP trails a decode with no pair of its pending token to enter".into());
            }
            self.tail = DraftTail::Entered(pending);
            return Ok(FrontEntry::Pending(
                self.enter_device(request, device.tokens, device.features, HeadPhase::Generation, selects)?
                    .operations,
            ));
        }
        match &self.tail {
            DraftTail::Entered(token) if *token == pending => {
                let (form, entry) = match grant {
                    FrontGrant::UpTo(_) | FrontGrant::Exactly(_) => (
                        PassForm::Front,
                        FrontEntry::Chain(ChainEntry {
                            position: self.position,
                        }),
                    ),
                    FrontGrant::Verify(_) => (PassForm::Verify, FrontEntry::Verify),
                };
                self.round = FrontRound::Pass(form);
                return Ok(entry);
            }
            DraftTail::Entered(_) => return Err("MTP entered another successor than the pending token".into()),
            DraftTail::Trailing { .. } => {}
        }
        let DraftTail::Trailing { pending: pairs, open } =
            std::mem::replace(&mut self.tail, DraftTail::Entered(pending))
        else {
            unreachable!("matched above");
        };
        let pairs = match (pairs, open) {
            (pairs, Some(OpenFeature::Host(rows))) => PendingRows::joined(pairs, vec![pending], rows)?,
            (Some(pairs), None) if pairs.tokens.last() == Some(&pending) => pairs,
            _ => return Err("MTP trails a decode with no pair of its pending token to enter".into()),
        };
        Ok(FrontEntry::Pending(
            self.enter(request, pairs, HeadPhase::Generation, selects)?
                .operations,
        ))
    }

    fn trails(&self) -> bool {
        self.pending_device.is_some() || matches!(self.tail, DraftTail::Trailing { .. })
    }

    /// A pass committed from the token the head entered. A front pass
    /// committed its exact row alone, and its head launch entered the
    /// pass's selection with the exact row's feature on the device: the
    /// head moves one row. A verify pass committed its exact row and the
    /// guesses that stood: each committed row's feature selected the token
    /// after it (the next accepted input, or the pass's successor after the
    /// last), and those pairs trail the head until the next round enters
    /// them.
    fn observe(
        &mut self,
        _: RequestId,
        verification: Verification<'_>,
        reader: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String> {
        let form = match self.round {
            FrontRound::Pass(form) => form,
            FrontRound::Head(_) => return Err("MTP already has an unresolved head transaction".into()),
            FrontRound::Idle => return Err("MTP observes no pass it started".into()),
        };
        let accepted = verification.accepted;
        match self.tail {
            DraftTail::Entered(token)
                if verification.inputs.first() == Some(&token)
                    && accepted >= 1
                    && accepted <= verification.inputs.len() => {}
            _ => return Err("a pass commits from the token MTP entered".into()),
        }
        match form {
            PassForm::Front => {
                if accepted != 1 || verification.inputs.len() != 1 {
                    return Err("a front pass commits its exact row alone".into());
                }
                self.position = self
                    .position
                    .checked_add(1)
                    .ok_or("MTP head position exhausted")?;
                self.tail = DraftTail::Entered(verification.next);
            }
            PassForm::Verify => {
                let features = verification
                    .features
                    .ok_or("a verify pass publishes its rows' features")?;
                let mut tokens = verification.inputs[1..accepted].to_vec();
                tokens.push(verification.next);
                if features.is_device_backed() {
                    // Keep the target's output features on the device (zero D2H
                    // copy) until the immediately following `Operation::Head`
                    // consumes them directly via `HeadConditioning::Features`.
                    self.pending_device = Some(DevicePending { tokens, features });
                    self.tail = DraftTail::EMPTY;
                } else {
                    let rows = DrafterState::read(reader, features, 0, accepted)?;
                    self.tail = DraftTail::Trailing {
                        pending: Some(PendingRows::joined(None, tokens, rows)?),
                        open: None,
                    };
                }
            }
        }
        self.round = FrontRound::Idle;
        Ok(MethodEffects::default())
    }

    fn reconcile(&mut self, operation: &Operation, outcome: Outcome) -> Result<(), String> {
        let transaction = match std::mem::replace(&mut self.round, FrontRound::Idle) {
            FrontRound::Head(transaction) => transaction,
            other => {
                self.round = other;
                return Err("MTP has no pending head transaction".into());
            }
        };
        let drafted = transaction.reconcile(operation, outcome)?;
        self.position = self
            .position
            .checked_add(drafted.entered)
            .ok_or("MTP head position exhausted")?;
        Ok(())
    }

    /// Its position and tail; a verify pass's last accepted pair holds the
    /// sampled successor, past the retained prefix, so its feature is
    /// exported open and each restored request enters its own successor.
    fn checkpoint(&self) -> Result<MethodCheckpoint, MethodCheckpointError> {
        match self.round {
            // An abandoned target pass leaves Pass(_) without mutating position
            // or tail, so the head state is still resolved and checkpointable.
            FrontRound::Idle | FrontRound::Pass(_) => {}
            FrontRound::Head(_) => return Err(MethodCheckpointError::Unresolved),
        }
        let tail = if let Some(device) = &self.pending_device {
            let rows = device
                .features
                .read_rows_to_host(device.tokens.len())
                .map_err(|_| MethodCheckpointError::Unresolved)?;
            let pending = PendingRows::joined(None, device.tokens.clone(), rows)
                .map_err(|_| MethodCheckpointError::Unresolved)?;
            let (pending, last) = pending.opened();
            DraftTail::Trailing {
                pending,
                open: Some(OpenFeature::Host(last)),
            }
        } else {
            match &self.tail {
                DraftTail::Trailing {
                    pending: Some(pending),
                    open: None,
                } => {
                    let (pending, last) = pending.clone().opened();
                    DraftTail::Trailing {
                        pending,
                        open: Some(OpenFeature::Host(last)),
                    }
                }
                tail => tail.clone(),
            }
        };
        Ok(MethodCheckpoint::Mtp(DraftCheckpoint {
            position: self.position,
            tail,
            skipped: 0,
        }))
    }

    fn evict(&mut self) {
        self.position = 0;
        self.tail = DraftTail::EMPTY;
        self.pending_device = None;
        self.round = FrontRound::Idle;
    }

    fn reclaimable(&self) -> u64 {
        self.tail.host_bytes() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::{
        FeatureRows, FeatureSpan, HeadFeatures, ResourceDomainId, Sampling, Selected, Shaping,
    };

    struct CountingReader {
        reads: usize,
        row_bytes: usize,
    }

    impl FeatureReader for CountingReader {
        fn read(&mut self, span: &FeatureSpan) -> Result<FeatureRows, String> {
            self.reads += 1;
            let mut bytes = Vec::with_capacity(span.count * self.row_bytes);
            for row in 0..span.count {
                bytes.extend_from_slice(&((span.start + row) as u32).to_le_bytes());
            }
            FeatureRows::new(bytes.into(), span.count).map_err(|e| e.to_string())
        }
    }

    fn select() -> SelectSpec {
        SelectSpec {
            sampling: Sampling::Greedy,
            shaping: Shaping::default(),
            seed: 0,
            position: 0,
            domain: 0,
            mask: None,
            history: None,
        }
    }

    #[test]
    fn front_verify_logical_fallback_and_device_pending_enter_exact_tokens() {
        let domain = ResourceDomainId::new("front-test").unwrap();
        let req = RequestId(1);
        let mut reader = CountingReader { reads: 0, row_bytes: 4 };

        // Start entered at token 10, position 5.
        let mut state = FrontState {
            position: 5,
            tail: DraftTail::Entered(TokenId(10)),
            pending_device: None,
            round: FrontRound::Idle,
        };

        // Start a Verify pass of 3 tokens [10, 11, 12], accepting all 3 and producing next = 13.
        let entry = state
            .front(req, TokenId(10), FrontGrant::Verify(3), &[select(), select()])
            .unwrap();
        assert!(matches!(entry, FrontEntry::Verify));
        let logical = FeatureRef::logical(domain.clone(), 3, 1).unwrap();
        assert!(!logical.is_device_backed());
        state
            .observe(
                req,
                Verification {
                    inputs: &[TokenId(10), TokenId(11), TokenId(12)],
                    accepted: 3,
                    next: TokenId(13),
                    features: Some(logical),
                },
                &mut reader,
            )
            .unwrap();
        assert_eq!(reader.reads, 1, "logical fallback reads via FeatureReader");
        assert!(state.trails());

        let FrontEntry::Pending(ops) = state
            .front(req, TokenId(13), FrontGrant::UpTo(2), &[select(), select()])
            .unwrap()
        else {
            panic!("expected pending head transaction");
        };
        let [op @ Operation::Head { tokens, conditioning, position: 5, proposals, .. }] = ops.as_slice() else {
            panic!("expected one Head op: {ops:?}");
        };
        assert_eq!(tokens, &[TokenId(11), TokenId(12), TokenId(13)]);
        assert_eq!(conditioning.rows(), 3);
        assert!(conditioning.host().is_some());
        assert_eq!(proposals.len(), 2);
        state
            .reconcile(
                op,
                Outcome::Head {
                    proposals: vec![Selected::Token(TokenId(14)), Selected::Token(TokenId(15))],
                    confidence: None,
                },
            )
            .unwrap();
        assert_eq!(state.position, 8);
        assert!(!state.trails());

        // Now verify DevicePending path emits HeadFeatures::Device with ZERO FeatureReader reads.
        let device_ref = FeatureRef::logical(domain, 3, 1).unwrap();
        state.pending_device = Some(DevicePending {
            tokens: vec![TokenId(14), TokenId(15), TokenId(16)],
            features: device_ref.clone(),
        });
        state.tail = DraftTail::EMPTY;
        assert!(state.trails());
        assert_eq!(state.priming_position(), None);
        let reads_before = reader.reads;
        let FrontEntry::Pending(ops) = state
            .front(req, TokenId(16), FrontGrant::UpTo(2), &[select(), select()])
            .unwrap()
        else {
            panic!("expected device-conditioned pending head transaction");
        };
        assert_eq!(reader.reads, reads_before, "device-pending path must never call FeatureReader");
        let [op @ Operation::Head { tokens, conditioning, position: 8, proposals, .. }] = ops.as_slice() else {
            panic!("expected one device Head op: {ops:?}");
        };
        assert_eq!(tokens, &[TokenId(14), TokenId(15), TokenId(16)]);
        assert_eq!(
            conditioning,
            &HeadFeatures::Device { features: device_ref, rows: 3 }
        );
        assert_eq!(proposals.len(), 2);
        state
            .reconcile(
                op,
                Outcome::Head {
                    proposals: vec![Selected::Token(TokenId(17)), Selected::Token(TokenId(18))],
                    confidence: None,
                },
            )
            .unwrap();
        assert_eq!(state.position, 11);
        assert!(!state.trails());
    }
}
