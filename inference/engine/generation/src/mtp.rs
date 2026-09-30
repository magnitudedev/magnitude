//! Multi-token prediction with the model's draft head, and the
//! target-conditioned drafter state it shares with DFlash (`dflash.rs`).
//!
//! The drafter's committed rows pair each accepted token with the target's
//! feature of the row before it (the feature that selected that token).
//! Those rows are target-conditioned only: rows the drafter conditioned on
//! its own features while drafting are never kept. A round's draft is one
//! drafter transaction whose entry rows are every pair not yet entered,
//! ending with the anchor (the last accepted token); the executor drafts the
//! proposals on the device, in the drafter's form (MTP chains head rows, a
//! DFlash draft reads one block).

use crate::{
    DraftCheckpoint, Method, MethodCheckpoint, MethodCheckpointError, MethodEffects,
    MethodRequirements, MethodState, Propose, Verification, method::PendingRows,
};
use magnitude_executor::{
    Demand, DraftForm, FeatureReader, FeatureRef, FeatureRows, FeatureSpan, HeadPhase, Operation,
    Outcome, RequestId, SelectSpec, TokenId,
};

#[derive(Clone, Debug)]
pub struct Mtp {
    identity: String,
    proposals: usize,
}

impl Mtp {
    pub fn new(artifact: impl AsRef<str>, proposals: usize) -> Result<Self, String> {
        let artifact = artifact.as_ref();
        if artifact.is_empty() || proposals == 0 || proposals > u8::MAX as usize {
            return Err("MTP identity and proposal width must be nonempty and bounded".into());
        }
        Ok(Self {
            identity: format!("mtp:{artifact}:{proposals}"),
            proposals,
        })
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
    fn proposals(&self) -> usize {
        self.proposals
    }
    fn create(
        &self,
        checkpoint: Option<&MethodCheckpoint>,
    ) -> Result<Box<dyn MethodState>, String> {
        Ok(Box::new(DrafterState::restore(
            Drafter::Mtp,
            self.proposals,
            checkpoint,
        )?))
    }
}

/// Which target-conditioned drafter a state drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Drafter {
    Mtp,
    DFlash,
}

impl Drafter {
    fn form(self) -> DraftForm {
        match self {
            Self::Mtp => DraftForm::Chained,
            Self::DFlash => DraftForm::Block,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Mtp => "MTP",
            Self::DFlash => "DFlash",
        }
    }

    fn checkpoint(self, state: DraftCheckpoint) -> MethodCheckpoint {
        match self {
            Self::Mtp => MethodCheckpoint::Mtp(state),
            Self::DFlash => MethodCheckpoint::DFlash(state),
        }
    }

    /// This drafter's state in `checkpoint`; another method's checkpoint
    /// cannot restore it.
    fn restored(self, checkpoint: &MethodCheckpoint) -> Option<&DraftCheckpoint> {
        match (self, checkpoint) {
            (Self::Mtp, MethodCheckpoint::Mtp(state))
            | (Self::DFlash, MethodCheckpoint::DFlash(state)) => Some(state),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct DrafterState {
    drafter: Drafter,
    proposals: usize,
    /// Drafter rows entered so far; the next entry row's position.
    position: usize,
    /// Complete pairs not yet entered; the last is the anchor.
    pending: Option<PendingRows>,
    /// A committed feature whose successor token is not known yet.
    open: Option<FeatureRows>,
    /// The drafter transaction `propose` returned, until it reconciles.
    head: Option<Operation>,
    /// Reconciled proposals awaiting their verification.
    proposal: Option<Vec<TokenId>>,
}

impl DrafterState {
    pub(crate) fn restore(
        drafter: Drafter,
        proposals: usize,
        checkpoint: Option<&MethodCheckpoint>,
    ) -> Result<Self, String> {
        let checkpoint = match checkpoint {
            None => DraftCheckpoint {
                position: 0,
                pending: None,
                open: None,
            },
            Some(checkpoint) => drafter.restored(checkpoint).cloned().ok_or_else(|| {
                format!(
                    "another method's checkpoint cannot restore {} state",
                    drafter.name()
                )
            })?,
        };
        Ok(Self {
            drafter,
            proposals,
            position: checkpoint.position,
            pending: checkpoint.pending,
            open: checkpoint.open,
            head: None,
            proposal: None,
        })
    }

    /// Entry rows a drafting transaction admits per request: a
    /// verification's accepted prefix and its anchor.
    fn entry_bound(&self) -> usize {
        self.proposals + 1
    }

    fn ensure_idle(&self) -> Result<(), String> {
        if self.head.is_some() || self.proposal.is_some() {
            return Err(format!(
                "{} method already has unresolved work",
                self.drafter.name()
            ));
        }
        Ok(())
    }

    fn append(&mut self, tokens: Vec<TokenId>, features: FeatureRows) -> Result<(), String> {
        if tokens.len() != features.rows() {
            return Err(format!(
                "{} pairs differ from their feature rows",
                self.drafter.name()
            ));
        }
        self.pending = Some(match self.pending.take() {
            None => PendingRows { tokens, features },
            Some(mut pending) => {
                pending.features = pending
                    .features
                    .concat(&features)
                    .map_err(|error| error.to_string())?;
                pending.tokens.extend(tokens);
                pending
            }
        });
        Ok(())
    }

    /// Enter every pending pair but the last `keep` as a drafter transaction
    /// without proposals.
    fn flush(
        &mut self,
        request: RequestId,
        keep: usize,
        phase: HeadPhase,
    ) -> Result<MethodEffects, String> {
        let Some(pending) = self.pending.take() else {
            return Ok(MethodEffects::default());
        };
        let rows = pending.tokens.len();
        if rows <= keep {
            self.pending = Some(pending);
            return Ok(MethodEffects::default());
        }
        let entered = rows - keep;
        let conditioning = pending
            .features
            .slice(0, entered)
            .map_err(|error| error.to_string())?;
        if keep > 0 {
            self.pending = Some(PendingRows {
                tokens: pending.tokens[entered..].to_vec(),
                features: pending
                    .features
                    .slice(entered, keep)
                    .map_err(|error| error.to_string())?,
            });
        }
        let operation = Operation::Head {
            request,
            phase,
            tokens: pending.tokens[..entered].to_vec(),
            conditioning,
            position: self.position,
            proposals: Vec::new(),
            form: self.drafter.form(),
        };
        operation.validate().map_err(|error| error.to_string())?;
        self.head = Some(operation.clone());
        Ok(MethodEffects {
            operations: vec![operation],
        })
    }

    fn read(
        reader: &mut dyn FeatureReader,
        features: FeatureRef,
        start: usize,
        count: usize,
    ) -> Result<FeatureRows, String> {
        let span = FeatureSpan::new(features, start, count).map_err(|error| error.to_string())?;
        reader.read(&span)
    }
}

impl MethodState for DrafterState {
    fn fork_transition(&self) -> Box<dyn MethodState> {
        Box::new(self.clone())
    }

    fn priming_position(&self) -> Option<usize> {
        (self.drafter == Drafter::DFlash
            && self.pending.is_none()
            && self.open.is_none()
            && self.head.is_none()
            && self.proposal.is_none())
        .then_some(self.position)
    }

    fn prime(
        &mut self,
        request: RequestId,
        tokens: &[TokenId],
        next: Option<TokenId>,
        features: FeatureRef,
        draft_from: usize,
        primed: usize,
        reader: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String> {
        if primed > 0 {
            // The chunk's entry was drafted behind it on the device: each
            // prompt token with the feature of the row before it, through
            // the chunk's last row when the prompt continues. A finishing
            // chunk's last feature selected `next`: the first draft's anchor.
            if self.priming_position() != Some(self.position) || primed > tokens.len() {
                return Err(format!(
                    "{} state cannot take a device-primed chunk",
                    self.drafter.name()
                ));
            }
            self.position = self
                .position
                .checked_add(primed)
                .ok_or_else(|| format!("{} drafter position exhausted", self.drafter.name()))?;
            if let Some(next) = next {
                let last = Self::read(reader, features, tokens.len() - 1, 1)?;
                self.append(vec![next], last)?;
            }
            return Ok(MethodEffects::default());
        }
        self.ensure_idle()?;
        let Some((&first, rest)) = tokens.split_first() else {
            return Err(format!(
                "{} cannot prime an empty target chunk",
                self.drafter.name()
            ));
        };
        let rows = Self::read(reader, features, 0, tokens.len())?;
        // The open feature selected this chunk's first token; each chunk
        // row's feature selected the token after it.
        if let Some(open) = self.open.take() {
            self.append(vec![first], open)?;
        }
        if !rest.is_empty() {
            self.append(
                rest.to_vec(),
                rows.slice(0, rest.len())
                    .map_err(|error| error.to_string())?,
            )?;
        }
        let last = rows
            .slice(tokens.len() - 1, 1)
            .map_err(|error| error.to_string())?;
        match next {
            Some(next) => {
                self.append(vec![next], last)?;
                // Keep the anchor for the first draft.
                self.flush(request, 1, HeadPhase::Priming { draft_from })
            }
            None => {
                self.open = Some(last);
                self.flush(request, 0, HeadPhase::Priming { draft_from })
            }
        }
    }

    fn propose(&mut self, request: RequestId, selects: &[SelectSpec]) -> Propose {
        if let Some(proposal) = self.proposal.as_ref() {
            return Propose::Tokens(proposal[..proposal.len().min(selects.len())].to_vec());
        }
        let (Some(pending), None, false) = (&self.pending, &self.head, selects.is_empty()) else {
            return Propose::Tokens(Vec::new());
        };
        if pending.tokens.len() > self.entry_bound() {
            return Propose::Tokens(Vec::new());
        }
        let operation = Operation::Head {
            request,
            phase: HeadPhase::Generation,
            tokens: pending.tokens.clone(),
            conditioning: pending.features.clone(),
            position: self.position,
            proposals: selects[..selects.len().min(self.proposals)].to_vec(),
            form: self.drafter.form(),
        };
        if operation.validate().is_err() {
            return Propose::Tokens(Vec::new());
        }
        self.head = Some(operation.clone());
        Propose::Pending(vec![operation])
    }

    fn observe(
        &mut self,
        request: RequestId,
        verification: Verification<'_>,
        reader: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String> {
        let name = self.drafter.name();
        if self.head.is_some() {
            return Err(format!(
                "{name} cannot observe while a drafter transaction is unresolved"
            ));
        }
        self.proposal = None;
        let features = verification
            .features
            .ok_or_else(|| format!("{name} verification requires target features"))?;
        let accepted = verification.accepted;
        if verification.inputs.is_empty() || accepted == 0 || accepted > verification.inputs.len() {
            return Err(format!("{name} verification prefix is invalid"));
        }
        // After a replay the last replayed row's feature selected the anchor.
        if let Some(open) = self.open.take() {
            self.append(vec![verification.inputs[0]], open)?;
        }
        if self
            .pending
            .as_ref()
            .and_then(|pending| pending.tokens.last())
            .is_some_and(|anchor| *anchor != verification.inputs[0])
        {
            return Err(format!(
                "{name} pending anchor differs from the verified anchor"
            ));
        }
        // Committed row i's feature selected the token after it: the next
        // accepted input, or the round's successor after the last row.
        let rows = Self::read(reader, features, 0, accepted)?;
        let mut tokens = verification.inputs[1..accepted].to_vec();
        tokens.push(verification.next);
        self.append(tokens, rows)?;
        let bound = self.entry_bound();
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.tokens.len() > bound)
        {
            return self.flush(request, 1, HeadPhase::Generation);
        }
        Ok(MethodEffects::default())
    }

    fn reconcile(&mut self, operation: &Operation, outcome: Outcome) -> Result<(), String> {
        let name = self.drafter.name();
        let head = self
            .head
            .take()
            .ok_or_else(|| format!("{name} has no pending drafter transaction"))?;
        let (
            Operation::Head {
                tokens, proposals, ..
            },
            Outcome::Head {
                proposals: selected,
            },
        ) = (&head, outcome)
        else {
            return Err(format!("{name} drafter outcome has the wrong kind"));
        };
        if head != *operation || selected.len() != proposals.len() {
            return Err(format!(
                "{name} drafter outcome differs from its transaction"
            ));
        }
        self.position = self
            .position
            .checked_add(tokens.len())
            .ok_or_else(|| format!("{name} drafter position exhausted"))?;
        if !proposals.is_empty() {
            self.pending = None;
            self.proposal = Some(
                selected
                    .iter()
                    .take_while(|selected| selected.status == 0)
                    .map(|selected| selected.token)
                    .collect(),
            );
        }
        Ok(())
    }

    fn checkpoint(&self) -> Result<MethodCheckpoint, MethodCheckpointError> {
        if self.head.is_some() || self.proposal.is_some() {
            return Err(MethodCheckpointError::Unresolved);
        }
        Ok(self.drafter.checkpoint(DraftCheckpoint {
            position: self.position,
            pending: self.pending.clone(),
            open: self.open.clone(),
        }))
    }

    fn evict(&mut self) {
        self.position = 0;
        self.pending = None;
        self.open = None;
        self.head = None;
        self.proposal = None;
    }

    fn reclaimable(&self) -> u64 {
        let pending = self
            .pending
            .as_ref()
            .map_or(0, |rows| rows.features.bytes().len());
        let open = self.open.as_ref().map_or(0, |rows| rows.bytes().len());
        (pending + open) as u64
    }
}
