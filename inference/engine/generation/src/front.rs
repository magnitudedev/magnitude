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
    /// The form of the pass a round started from the level head, until it
    /// is observed: whether the head entered its selection behind it.
    pass: Option<PassForm>,
    /// The host head transaction entering pairs, until it reconciles.
    head: Option<HeadTransaction>,
}

/// The form of a pass ([`FrontState::front`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PassForm {
    /// Front rows, the head entering its selection behind it on the device.
    Front,
    /// Acceptable rows, entering nothing behind it.
    Verify,
}

impl FrontState {
    pub(crate) fn restore(checkpoint: Option<&MethodCheckpoint>) -> Result<Self, String> {
        let state = match checkpoint {
            None => {
                return Ok(Self {
                    position: 0,
                    tail: DraftTail::EMPTY,
                    pass: None,
                    head: None,
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
            pass: None,
            head: None,
        })
    }

    fn idle(&self) -> Result<(), String> {
        match self.head {
            Some(_) => Err("MTP already has an unresolved head transaction".into()),
            None => Ok(()),
        }
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
        self.head = Some(transaction);
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
        match (&self.tail, &self.head) {
            (
                DraftTail::Trailing {
                    pending: None,
                    open: None | Some(OpenFeature::Device(_)),
                }
                | DraftTail::Entered(_),
                None,
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
        self.idle()?;
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
        let mut pending = match tail {
            DraftTail::Entered(token) if token == first => None,
            DraftTail::Entered(_) => {
                return Err("the chunk's first token differs from the successor MTP entered".into());
            }
            DraftTail::Trailing { pending, open: Some(open) } => Some(PendingRows::joined(
                pending,
                vec![first],
                DrafterState::open_rows(reader, open)?,
            )?),
            DraftTail::Trailing { pending, open: None } => pending,
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
        self.idle()?;
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
                self.pass = Some(form);
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
        self.pass = None;
        Ok(FrontEntry::Pending(
            self.enter(request, pairs, HeadPhase::Generation, selects)?
                .operations,
        ))
    }

    fn trails(&self) -> bool {
        matches!(self.tail, DraftTail::Trailing { .. })
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
        self.idle()?;
        let accepted = verification.accepted;
        match self.tail {
            DraftTail::Entered(token)
                if verification.inputs.first() == Some(&token)
                    && accepted >= 1
                    && accepted <= verification.inputs.len() => {}
            _ => return Err("a pass commits from the token MTP entered".into()),
        }
        match self.pass.take() {
            Some(PassForm::Front) => {
                if accepted != 1 || verification.inputs.len() != 1 {
                    return Err("a front pass commits its exact row alone".into());
                }
                self.position = self
                    .position
                    .checked_add(1)
                    .ok_or("MTP head position exhausted")?;
                self.tail = DraftTail::Entered(verification.next);
            }
            Some(PassForm::Verify) => {
                let features = verification
                    .features
                    .ok_or("a verify pass publishes its rows' features")?;
                let rows = DrafterState::read(reader, features, 0, accepted)?;
                let mut tokens = verification.inputs[1..accepted].to_vec();
                tokens.push(verification.next);
                self.tail = DraftTail::Trailing {
                    pending: Some(PendingRows::joined(None, tokens, rows)?),
                    open: None,
                };
            }
            None => return Err("MTP observes no pass it started".into()),
        }
        Ok(MethodEffects::default())
    }

    fn reconcile(&mut self, operation: &Operation, outcome: Outcome) -> Result<(), String> {
        let drafted = self
            .head
            .take()
            .ok_or("MTP has no pending head transaction")?
            .reconcile(operation, outcome)?;
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
        if self.head.is_some() {
            return Err(MethodCheckpointError::Unresolved);
        }
        let tail = match &self.tail {
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
        self.pass = None;
        self.head = None;
    }

    fn reclaimable(&self) -> u64 {
        self.tail.host_bytes() as u64
    }
}
