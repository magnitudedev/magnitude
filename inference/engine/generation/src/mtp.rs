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
//!
//! A prompt chunk's complete pairs enter on the device behind the target.
//! MTP keeps its final feature independently owned by executor state, so the
//! next chunk supplies the successor token even when it resumes a prefix.
//! Replayed history with host features or pending pairs enters through
//! drafter transactions after each chunk.
//!
//! An MTP head that reads the target's history (an assistant drafter) keeps
//! none: nothing is primed, so no prompt row (a media row included) ever
//! enters it as a token. Its position is the target's, which only committed
//! target rows move, and a draft is one transaction of the anchor alone with
//! the target feature that selected it: the prompt's closing chunk's final
//! feature, then each verification's last accepted row's. The head chains
//! its proposals on the device over the target's committed history and
//! writes nothing.

use crate::{
    method::{OpenFeature, PendingRows},
    DraftBudget, DraftCheckpoint, DraftController, DraftLoad, DraftOutlook, Method, MethodCheckpoint,
    MethodCheckpointError, MethodEffects, MethodRequirements, MethodState, Propose, TimedLaunch,
    Verification,
};
use crate::{DraftHistory, DraftRoundState};
use magnitude_executor::{
    Demand, DraftForm, FeatureReader, FeatureRef, FeatureRows, FeatureSpan, HeadPhase, Operation,
    Outcome, RequestId, SelectSpec, TokenId,
};
use std::{sync::Arc, time::Duration};

#[derive(Clone, Debug)]
pub struct Mtp {
    identity: String,
    proposals: usize,
    drafter: Drafter,
    budget: DraftBudget,
    adaptive: Option<Arc<DraftLoad>>,
}

impl Mtp {
    /// MTP with a head that keeps its own history: every committed row
    /// enters it.
    pub fn new(artifact: impl AsRef<str>, proposals: usize) -> Result<Self, String> {
        Self::with(artifact, proposals, Drafter::Mtp)
    }

    /// MTP with a head that reads the target's history and keeps none: a
    /// draft needs only the anchor and the target feature that selected it.
    pub fn reading_target(artifact: impl AsRef<str>, proposals: usize) -> Result<Self, String> {
        Self::with(artifact, proposals, Drafter::TargetReadingMtp)
    }

    fn with(artifact: impl AsRef<str>, proposals: usize, drafter: Drafter) -> Result<Self, String> {
        let artifact = artifact.as_ref();
        if artifact.is_empty() || proposals == 0 || proposals > u8::MAX as usize {
            return Err("MTP identity and proposal width must be nonempty and bounded".into());
        }
        Ok(Self {
            identity: format!("mtp:{artifact}:{proposals}"),
            proposals,
            drafter,
            budget: DraftBudget {
                generated: proposals,
                verified: proposals,
            },
            adaptive: None,
        })
    }
    /// Draft and verify `budget` each round; its generated width must fit
    /// this load.
    pub fn with_budget(mut self, budget: DraftBudget) -> Result<Self, String> {
        budget.validate(self.proposals)?;
        self.identity.push_str(&format!(":{budget:?}"));
        self.budget = budget;
        Ok(self)
    }
    /// Choose each round's generated and verified widths from what the
    /// load's rounds measure, up to the proposal width.
    pub fn adaptive(mut self) -> Self {
        self.identity.push_str(":adaptive");
        self.adaptive = Some(Arc::default());
        self
    }
}

impl Method for Mtp {
    fn priming_requires_successor(&self) -> bool {
        self.drafter.keeps_history()
    }
    fn embeds_prompt_tokens(&self) -> bool {
        self.drafter.keeps_history()
    }
    fn identity(&self) -> &str {
        &self.identity
    }
    fn requires(&self) -> MethodRequirements {
        // An embedded head enters every prompt row with its feature; a head
        // reading the target's history needs only the feature that selected
        // the first anchor.
        let (prefill_demand, closing_demand) = if self.drafter.keeps_history() {
            (Demand::FEATURES, Demand::NONE)
        } else {
            (Demand::NONE, Demand::FEATURES)
        };
        MethodRequirements {
            prefill_demand,
            closing_demand,
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
            self.drafter,
            self.proposals,
            self.budget,
            self.adaptive.clone(),
            checkpoint,
        )?))
    }
}

/// Which target-conditioned drafter a state drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Drafter {
    Mtp,
    /// An MTP head that reads the target's history and keeps none. Its
    /// position is the target's: the anchor's row.
    TargetReadingMtp,
    DFlash,
}

impl Drafter {
    fn form(self) -> DraftForm {
        match self {
            Self::Mtp | Self::TargetReadingMtp => DraftForm::Chained,
            Self::DFlash => DraftForm::Block,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Mtp | Self::TargetReadingMtp => "MTP",
            Self::DFlash => "DFlash",
        }
    }

    /// Whether the drafter enters every committed row into its own history.
    fn keeps_history(self) -> bool {
        match self {
            Self::Mtp | Self::DFlash => true,
            Self::TargetReadingMtp => false,
        }
    }

    fn checkpoint(self, state: DraftCheckpoint) -> MethodCheckpoint {
        match self {
            Self::Mtp | Self::TargetReadingMtp => MethodCheckpoint::Mtp(state),
            Self::DFlash => MethodCheckpoint::DFlash(state),
        }
    }

    /// This drafter's state in `checkpoint`; another method's checkpoint
    /// cannot restore it.
    fn restored(self, checkpoint: &MethodCheckpoint) -> Option<&DraftCheckpoint> {
        match (self, checkpoint) {
            (Self::Mtp | Self::TargetReadingMtp, MethodCheckpoint::Mtp(state))
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
    open: Option<OpenFeature>,
    /// The drafter transaction `propose` returned, until it reconciles.
    head: Option<Operation>,
    /// Reconciled proposals awaiting their verification.
    proposal: Option<Vec<TokenId>>,
    budget: DraftBudget,
    /// The controller choosing each round's widths in place of `budget`.
    control: Option<DraftController>,
    /// This round's generated and verified widths, once chosen: a round
    /// started again proposes what it proposed before.
    generated: Option<usize>,
    verified: Option<usize>,
}

impl DrafterState {
    pub(crate) fn restore(
        drafter: Drafter,
        proposals: usize,
        budget: DraftBudget,
        adaptive: Option<Arc<DraftLoad>>,
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
        // A head reading the target's history retains nothing at a
        // boundary: the next chunk's closing feature selects its anchor.
        if !drafter.keeps_history() && (checkpoint.pending.is_some() || checkpoint.open.is_some()) {
            return Err("a head reading the target's history restores no feature rows".into());
        }
        Ok(Self {
            drafter,
            proposals,
            position: checkpoint.position,
            pending: checkpoint.pending,
            open: checkpoint.open,
            head: None,
            proposal: None,
            budget,
            control: adaptive.map(|load| DraftController::new(proposals, load)),
            generated: None,
            verified: None,
        })
    }

    /// Entry rows a drafting transaction admits per request: a
    /// verification's accepted prefix and its anchor.
    fn entry_bound(&self) -> usize {
        self.proposals + 1
    }

    fn round_state(&self) -> DraftRoundState {
        DraftRoundState::new(
            if self.drafter.keeps_history() {
                DraftHistory::Own
            } else {
                DraftHistory::Target
            },
            self.entry_bound(),
            self.position,
            self.pending
                .as_ref()
                .map_or(0, |pending| pending.tokens.len()),
            self.open.is_some(),
        )
        .expect("drafter state preserves its method boundary")
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

    fn advance(&mut self, rows: usize) -> Result<(), String> {
        // Target-reading heads advance only at target commitment; their
        // physical head transactions enter no rows.
        self.position = self
            .position
            .checked_add(rows)
            .ok_or_else(|| format!("{} drafter position exhausted", self.drafter.name()))?;
        Ok(())
    }

    /// Enter a committed target chunk into a head reading the target's
    /// history: nothing enters it, the chunk moves its position, and the
    /// closing chunk's final feature selected `next`, the first anchor.
    fn prime_reading(
        &mut self,
        tokens: &[TokenId],
        next: Option<TokenId>,
        features: Option<FeatureRef>,
        primed: Option<usize>,
        reader: &mut dyn FeatureReader,
    ) -> Result<(), String> {
        if primed.is_some() || tokens.is_empty() {
            return Err("a head reading the target's history primes nothing".into());
        }
        self.ensure_idle()?;
        self.pending = None;
        self.open = None;
        self.advance(tokens.len())?;
        if let Some(next) = next {
            let features =
                features.ok_or("the chunk closing the prompt requires its final target feature")?;
            let last = Self::read(reader, features, tokens.len() - 1, 1)?;
            self.append(vec![next], last)?;
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
        let (call, _) = self.round_state().flushing(keep);
        let Some(call) = call else {
            return Ok(MethodEffects::default());
        };
        let Some(pending) = self.pending.take() else {
            unreachable!("a described head entry owns pending pairs");
        };
        let entered = call.entry_rows;
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
            position: call.position,
            proposals: Vec::new(),
            form: self.drafter.form(),
        };
        operation.validate().map_err(|error| error.to_string())?;
        self.head = Some(operation.clone());
        Ok(MethodEffects {
            operations: vec![operation],
        })
    }

    fn open_rows(reader: &mut dyn FeatureReader, open: OpenFeature) -> Result<FeatureRows, String> {
        match open {
            OpenFeature::Host(rows) => Ok(rows),
            OpenFeature::Device(feature) => Self::read(
                reader,
                feature
                    .upgrade()
                    .ok_or("executor boundary feature is no longer resident")?,
                0,
                1,
            ),
        }
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
        // A drafter without history has no entry to draft behind a chunk.
        (self.drafter.keeps_history()
            && self.pending.is_none()
            && !matches!(self.open, Some(OpenFeature::Host(_)))
            && self.head.is_none()
            && self.proposal.is_none())
        .then_some(self.position)
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
        if !self.drafter.keeps_history() {
            self.prime_reading(tokens, next, features, primed, reader)?;
            return Ok(MethodEffects::default());
        }
        let features = features.ok_or_else(|| {
            format!(
                "{} priming requires the chunk's target features",
                self.drafter.name()
            )
        })?;
        if let Some(primed) = primed {
            // The device entered complete causal pairs. MTP leaves the
            // final feature open; a feature-only drafter may enter it now.
            // A finishing chunk's last feature selects the draft anchor.
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
            self.open = None;
            if self.drafter == Drafter::Mtp {
                let boundary = features
                    .boundary()
                    .cloned()
                    .ok_or("MTP priming requires an independently owned boundary feature")?;
                if let Some(next) = next {
                    self.append(vec![next], Self::read(reader, boundary, 0, 1)?)?;
                } else {
                    self.open = Some(OpenFeature::Device(boundary.downgrade()));
                }
            } else if let Some(next) = next {
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
            self.append(vec![first], Self::open_rows(reader, open)?)?;
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
                self.open = Some(OpenFeature::Host(last));
                self.flush(request, 0, HeadPhase::Priming { draft_from })
            }
        }
    }

    fn propose(&mut self, request: RequestId, selects: &[SelectSpec]) -> Propose {
        if let Some(proposal) = self.proposal.as_ref() {
            // The verified prefix is all a round offers the target.
            let count = match (&mut self.control, self.verified) {
                (_, Some(verified)) => verified,
                (Some(control), None) => control.verify(proposal.len(), selects.len()),
                (None, None) => self.budget.verified,
            }
            .min(proposal.len())
            .min(selects.len());
            self.verified = Some(count);
            return Propose::Tokens(proposal[..count].to_vec());
        }
        let (Some(pending), None, None, false) =
            (&self.pending, &self.head, &self.open, selects.is_empty())
        else {
            return Propose::Tokens(Vec::new());
        };
        if pending.tokens.len() > self.entry_bound() {
            return Propose::Tokens(Vec::new());
        }
        // A round drafts its whole generated width or nothing. Launch costs
        // depend on the history the round reads, the target's position.
        let position = self
            .round_state()
            .target_position()
            .expect("drafter state preserves its method boundary");
        let generated = match (&mut self.control, self.generated) {
            (_, Some(generated)) => generated,
            (Some(control), None) => control.generate(selects.len(), position),
            (None, None) => self.budget.generated,
        };
        self.generated = Some(generated);
        if generated == 0 || generated > selects.len() {
            return Propose::Tokens(Vec::new());
        }
        let (Some(call), _) = self
            .round_state()
            .proposing(generated)
            .expect("budgeted proposal fits the method boundary")
        else {
            unreachable!("positive proposal describes a head transaction");
        };
        let operation = Operation::Head {
            request,
            phase: HeadPhase::Generation,
            tokens: pending.tokens.clone(),
            conditioning: pending.features.clone(),
            position: call.position,
            proposals: selects[..generated].to_vec(),
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
        let transition = self.round_state().observed(accepted)?;
        self.generated = None;
        self.verified = None;
        if let Some(control) = &mut self.control {
            let mut committed = verification.inputs[1..accepted].to_vec();
            committed.push(verification.next);
            control.committed(&committed);
            if std::env::var_os("MAGNITUDE_TRACE_DRAFT_WIDTH").is_some() {
                let choice = control.choice();
                eprintln!(
                    "draft width request={request:?} generated={} verified={} look={} committed={} right={:.2?} measured={:.4?}",
                    choice.generated,
                    choice.verified,
                    choice.look,
                    committed.len(),
                    control.right(),
                    control.measured(),
                );
            }
        }
        // After a replay the last replayed row's feature selected the anchor.
        if let Some(open) = self.open.take() {
            self.append(vec![verification.inputs[0]], Self::open_rows(reader, open)?)?;
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
        if !self.drafter.keeps_history() {
            // The accepted rows moved the target; the last one's feature
            // selected the next anchor.
            let last = Self::read(reader, features, accepted - 1, 1)?;
            self.pending = None;
            self.position = transition.appended.position();
            self.append(vec![verification.next], last)?;
            debug_assert_eq!(self.round_state(), transition.appended);
            return Ok(MethodEffects::default());
        }
        // Committed row i's feature selected the token after it: the next
        // accepted input, or the round's successor after the last row.
        let rows = Self::read(reader, features, 0, accepted)?;
        let mut tokens = verification.inputs[1..accepted].to_vec();
        tokens.push(verification.next);
        self.append(tokens, rows)?;
        debug_assert_eq!(self.round_state(), transition.appended);
        if transition.maintenance.is_some() {
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
        // A drafter without history entered nothing: its position is the
        // target's, which only committed target rows move.
        if self.drafter.keeps_history() {
            self.position = self.round_state().entered(tokens.len())?.position();
        }
        if !proposals.is_empty() {
            self.pending = None;
            let proposal = selected
                .iter()
                .take_while(|selected| selected.status == 0)
                .map(|selected| selected.token)
                .collect::<Vec<_>>();
            if let Some(control) = &mut self.control {
                control.drafted(proposals.len(), &proposal);
            }
            self.proposal = Some(proposal);
        }
        Ok(())
    }

    fn timed(&mut self, launch: TimedLaunch, duration: Duration, sequences: usize, rows: usize) {
        if let Some(control) = &mut self.control {
            control.timed(launch, duration, sequences, rows);
        }
    }

    fn outlook(&self) -> Option<DraftOutlook> {
        match (&self.control, &self.proposal, self.verified) {
            (Some(control), Some(_), None) => control.outlook(),
            _ => None,
        }
    }

    fn checkpoint(&self) -> Result<MethodCheckpoint, MethodCheckpointError> {
        if self.head.is_some() || self.proposal.is_some() {
            return Err(MethodCheckpointError::Unresolved);
        }
        if !self.drafter.keeps_history() {
            // The pending anchor is the sampled successor, outside the
            // retained prefix; its request's next chunk selects its own.
            return Ok(self.drafter.checkpoint(DraftCheckpoint {
                position: self.position,
                pending: None,
                open: None,
            }));
        }
        let mut pending = self.pending.clone();
        let mut open = self.open.clone();
        // The final pending token is the sampled successor, outside the
        // retained target prefix. Export its feature without that token;
        // each restored request supplies its own successor.
        if open.is_none() {
            if let Some(rows) = pending.take() {
                let last = rows.tokens.len() - 1;
                open = Some(OpenFeature::Host(
                    rows.features.slice(last, 1).expect("pending row shape"),
                ));
                if last > 0 {
                    pending = Some(PendingRows {
                        tokens: rows.tokens[..last].to_vec(),
                        features: rows.features.slice(0, last).expect("pending row shape"),
                    });
                }
            }
        }
        Ok(self.drafter.checkpoint(DraftCheckpoint {
            position: self.position,
            pending,
            open,
        }))
    }

    fn evict(&mut self) {
        self.position = 0;
        self.pending = None;
        self.open = None;
        self.head = None;
        self.proposal = None;
        self.generated = None;
        self.verified = None;
    }

    fn reclaimable(&self) -> u64 {
        let pending = self
            .pending
            .as_ref()
            .map_or(0, |rows| rows.features.bytes().len());
        let open = self.open.as_ref().map_or(0, OpenFeature::host_bytes);
        (pending + open) as u64
    }
}
