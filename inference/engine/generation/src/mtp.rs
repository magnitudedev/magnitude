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
//!
//! On a load that fronts, MTP with a head of its own history proposes
//! nothing on its own: its generations decode in front passes on the chain
//! its head makes, on the device behind a front pass, or in the host head
//! transaction ([`HeadTransaction`]) that enters a verify pass's accepted
//! pairs (`front.rs`).

use crate::{
    method::{DraftTail, OpenFeature, PendingRows},
    DraftBudget, DraftCheckpoint, DraftController, DraftLoad, DraftOutlook, Method, MethodCheckpoint,
    MethodCheckpointError, MethodChoice, MethodEffects, MethodRequirements, MethodState, Propose,
    TimedLaunch, Verification,
};
use crate::{front::FrontState, DraftHistory, DraftRoundState};
use magnitude_executor::{
    Demand, DraftForm, FeatureReader, FeatureRef, FeatureRows, FeatureSpan, HeadPhase, LaunchDuration,
    Guess, GuessKind, Operation, Outcome, RequestId, SelectSpec, Selected, StreamedVerifyPrior, TokenId,
};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Mtp {
    identity: String,
    proposals: usize,
    drafter: Drafter,
    budget: DraftBudget,
    adaptive: Option<Arc<DraftLoad>>,
    /// Whether its generations decode in front passes ([`Self::fronting`]).
    fronting: bool,
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
            fronting: false,
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

    /// MTP on a load that fronts (`magnitude_executor::TwoRowDecode`): its
    /// generations never propose on their own, and decode in passes of the
    /// form their launches grant on the chain the head makes
    /// (`FrontState`). Only a head of its own history fronts.
    pub fn fronting(mut self) -> Result<Self, String> {
        if self.drafter != Drafter::Mtp {
            return Err("only a head of its own history decodes in front passes".into());
        }
        self.identity.push_str(":front");
        self.fronting = true;
        Ok(self)
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
    fn capability(&self) -> MethodChoice {
        MethodChoice::Mtp {
            proposals: self.proposals as u8,
        }
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
        if self.fronting {
            return Ok(Box::new(FrontState::restore(checkpoint)?));
        }
        Ok(Box::new(DrafterState::restore(
            self.drafter,
            self.proposals,
            self.budget,
            self.adaptive.clone(),
            checkpoint,
        )?))
    }
    fn decodes_two_rows(&self) -> bool {
        self.fronting
    }
    fn observe_streamed_prior(&self, prior: Option<StreamedVerifyPrior>) {
        if let Some(adaptive) = &self.adaptive {
            adaptive.observe_streamed(prior);
        }
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

    /// Whether publishing a round's feature costs its target step: a
    /// separate draft's fused taps do (the fusion and the tapped blocks'
    /// writes); an MTP head's feature is the step's own output row.
    fn fuses_taps(self) -> bool {
        match self {
            Self::DFlash => true,
            Self::Mtp | Self::TargetReadingMtp => false,
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

/// A host head transaction, until its outcome reconciles: the drafter
/// enters complete pairs, each token with the target feature of the row
/// before it, from its position, then drafts one proposal per select from
/// the last pair (none of a transaction that only enters). Every head
/// transaction a target-conditioned drafter submits from the host is one,
/// a drafting state's and a head decoding in front passes' alike.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HeadTransaction {
    operation: Operation,
}

/// What a head transaction did, once reconciled: the pairs it entered,
/// the proposals it was asked for, and its proposal ([`proposal`]) with the
/// drafter's confidence in each where it gave one.
pub(crate) struct Drafted {
    pub(crate) entered: usize,
    pub(crate) requested: usize,
    pub(crate) proposal: Vec<TokenId>,
    pub(crate) confidence: Option<Vec<f32>>,
}

/// A head transaction's proposal: its leading selected tokens up to the
/// first empty or declined slot, since each drafted proposal conditions on
/// the one before it. Any nonfinite selection anywhere in the batch fails
/// explicitly rather than silently truncating speculation.
pub(crate) fn proposal(selected: &[Selected]) -> Result<Vec<TokenId>, String> {
    if selected.contains(&Selected::NonFinite) {
        return Err("draft sampling distribution is nonfinite".into());
    }
    Ok(selected
        .iter()
        .map_while(|selected| match *selected {
            Selected::Token(token) => Some(token),
            Selected::Empty | Selected::Declined => None,
            Selected::NonFinite => unreachable!("checked above"),
        })
        .collect())
}

/// The chain a head transaction drafted for a head decoding in front
/// passes, from its selections (`Outcome::Head`'s): its proposal, each the
/// head's guess. The one reading of a drafted chain, generation's and the
/// evaluation harness's alike.
pub fn drafted_chain(selected: &[Selected]) -> Result<Vec<Guess>, String> {
    Ok(proposal(selected)?
        .into_iter()
        .map(|token| Guess { token, kind: GuessKind::Head })
        .collect())
}

impl HeadTransaction {
    /// Enter `pairs` at `position`, drafting a proposal per `proposals`.
    pub(crate) fn new(
        request: RequestId,
        phase: HeadPhase,
        pairs: PendingRows,
        position: usize,
        proposals: Vec<SelectSpec>,
        form: DraftForm,
    ) -> Result<Self, String> {
        let operation = Operation::Head {
            request,
            phase,
            tokens: pairs.tokens,
            conditioning: pairs.features,
            position,
            proposals,
            form,
        };
        operation.validate().map_err(|error| error.to_string())?;
        Ok(Self { operation })
    }

    pub(crate) fn operation(&self) -> &Operation {
        &self.operation
    }

    /// Its outcome, of exactly this transaction.
    pub(crate) fn reconcile(self, operation: &Operation, outcome: Outcome) -> Result<Drafted, String> {
        let (
            Operation::Head {
                tokens, proposals, ..
            },
            Outcome::Head {
                proposals: selected,
                confidence,
            },
        ) = (&self.operation, outcome)
        else {
            return Err("head transaction outcome has the wrong kind".into());
        };
        if self.operation != *operation
            || selected.len() != proposals.len()
            || confidence
                .as_ref()
                .is_some_and(|confidence| confidence.len() != selected.len())
        {
            return Err("head transaction outcome differs from its transaction".into());
        }
        let proposal = proposal(&selected)?;
        Ok(Drafted {
            entered: tokens.len(),
            requested: proposals.len(),
            confidence: confidence.map(|mut confidence| {
                confidence.truncate(proposal.len());
                confidence
            }),
            proposal,
        })
    }
}

/// How each round's draft and verify widths are chosen.
#[derive(Clone)]
enum WidthPolicy {
    Fixed(DraftBudget),
    Adaptive(DraftController),
}

/// One anchor token paired with the single target feature row that selected it.
#[derive(Clone)]
struct AnchorRow {
    token: TokenId,
    feature: FeatureRows,
}

impl AnchorRow {
    fn new(token: TokenId, feature: FeatureRows) -> Result<Self, String> {
        if feature.rows() != 1 {
            return Err("drafter pairs differ from their feature rows".into());
        }
        Ok(Self { token, feature })
    }

    fn into_pending(self) -> PendingRows {
        PendingRows {
            tokens: vec![self.token],
            features: self.feature,
        }
    }
}

/// History state of a target-conditioned drafter.
#[derive(Clone)]
enum HistoryState {
    /// A head reading the target's history: keeps no history of its own, only
    /// the target's position and the single anchor pair awaiting a draft.
    TargetReading {
        /// The target's position; only committed target rows move it.
        position: usize,
        /// The anchor token and the target feature row that selected it.
        anchor: Option<AnchorRow>,
    },
    /// A drafter keeping its own history of committed target-conditioned pairs.
    OwnHistory {
        /// Drafter rows entered so far; the next entry row's position.
        position: usize,
        /// Complete pairs not yet entered; the last is the anchor when `anchored`.
        pending: Option<PendingRows>,
        /// A committed feature whose successor token is not known yet.
        open: Option<OpenFeature>,
        /// Whether the drafter holds its next anchor's pair: the round before
        /// published its feature (or a prompt chunk closed). Under the
        /// controller a round that drafts nothing publishes none, so the next
        /// one cannot draft.
        anchored: bool,
        /// Target rows whose pairs an own-history drafter never entered: rows of
        /// rounds that published no feature. Its history goes on from the next
        /// pair it is given, so its positions trail the target's by these.
        skipped: usize,
    },
}

impl HistoryState {
    fn round_state(&self, entry_bound: usize) -> DraftRoundState {
        let (history, position, pending, open) = match self {
            Self::TargetReading { position, anchor } => {
                (DraftHistory::Target, *position, usize::from(anchor.is_some()), false)
            }
            Self::OwnHistory {
                position,
                pending,
                open,
                ..
            } => (
                DraftHistory::Own,
                *position,
                pending.as_ref().map_or(0, |pending| pending.tokens.len()),
                open.is_some(),
            ),
        };
        round_state_of(history, entry_bound, position, pending, open)
    }

    fn has_open(&self) -> bool {
        matches!(self, Self::OwnHistory { open: Some(_), .. })
    }

    fn skipped(&self) -> usize {
        match self {
            Self::TargetReading { .. } => 0,
            Self::OwnHistory { skipped, .. } => *skipped,
        }
    }

    fn draft_pairs(&self) -> Option<PendingRows> {
        match self {
            Self::TargetReading { anchor, .. } => anchor.clone().map(AnchorRow::into_pending),
            Self::OwnHistory { pending, .. } => pending.clone(),
        }
    }

    fn clear_pending(&mut self) {
        match self {
            Self::TargetReading { anchor, .. } => *anchor = None,
            Self::OwnHistory { pending, .. } => *pending = None,
        }
    }

    fn reset(&mut self) {
        match self {
            Self::TargetReading { position, anchor } => {
                *position = 0;
                *anchor = None;
            }
            Self::OwnHistory {
                position,
                pending,
                open,
                anchored,
                skipped,
            } => {
                *position = 0;
                *pending = None;
                *open = None;
                *anchored = false;
                *skipped = 0;
            }
        }
    }
}

fn round_state_of(
    history: DraftHistory,
    entry_bound: usize,
    position: usize,
    pending: usize,
    open: bool,
) -> DraftRoundState {
    DraftRoundState::new(history, entry_bound, position, pending, open)
        .expect("drafter state preserves its method boundary")
}

/// Lifecycle stage of the current round's drafter work.
#[derive(Clone)]
enum RoundStage {
    /// No unresolved drafter work and no width chosen for this round yet.
    Idle,
    /// This round chose not to draft, so repeated `propose` calls in the same
    /// round do not re-query the width policy.
    Declined,
    /// An entry-only head transaction (priming, maintenance, or catch-up) is
    /// in flight. When triggered by catch-up in `propose`, `resume_draft`
    /// holds the chosen width to draft once catch-up reconciles.
    Entering {
        transaction: HeadTransaction,
        resume_draft: Option<usize>,
    },
    /// Catch-up finished; the next `propose` drafts this chosen width.
    ResumeDraft(usize),
    /// A drafting head transaction is in flight.
    Drafting(HeadTransaction),
    /// Reconciled proposals awaiting verification and observation.
    Proposed {
        proposal: Vec<TokenId>,
        verified: Option<usize>,
    },
}

impl RoundStage {
    fn is_idle(&self) -> bool {
        matches!(self, Self::Idle | Self::Declined | Self::ResumeDraft(_))
    }

    fn has_transaction(&self) -> bool {
        matches!(self, Self::Entering { .. } | Self::Drafting(_))
    }

    fn is_drafting(&self) -> bool {
        matches!(
            self,
            Self::Entering {
                resume_draft: Some(_),
                ..
            } | Self::ResumeDraft(_)
                | Self::Drafting(_)
                | Self::Proposed { .. }
        )
    }
}

#[derive(Clone)]
pub(crate) struct DrafterState {
    drafter: Drafter,
    proposals: usize,
    width: WidthPolicy,
    round: RoundStage,
    history: HistoryState,
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
                tail: DraftTail::EMPTY,
                skipped: 0,
            },
            Some(checkpoint) => drafter.restored(checkpoint).cloned().ok_or_else(|| {
                format!(
                    "another method's checkpoint cannot restore {} state",
                    drafter.name()
                )
            })?,
        };
        // Only a head decoding in front passes enters past the target.
        let DraftTail::Trailing { pending, open } = checkpoint.tail else {
            return Err(format!("a drafting {} state enters no successor", drafter.name()));
        };
        let history = match drafter {
            Drafter::TargetReadingMtp => {
                // A head reading the target's history retains nothing at a
                // boundary: the next chunk's closing feature selects its anchor.
                if pending.is_some() || open.is_some() {
                    return Err(
                        "a head reading the target's history restores no feature rows".into(),
                    );
                }
                HistoryState::TargetReading {
                    position: checkpoint.position,
                    anchor: None,
                }
            }
            Drafter::Mtp | Drafter::DFlash => HistoryState::OwnHistory {
                position: checkpoint.position,
                pending,
                open,
                // The next prompt chunk anchors it.
                anchored: false,
                skipped: checkpoint.skipped,
            },
        };
        let width = match adaptive {
            Some(load) => WidthPolicy::Adaptive(DraftController::new(proposals, load)),
            None => WidthPolicy::Fixed(budget),
        };
        Ok(Self {
            drafter,
            proposals,
            width,
            round: RoundStage::Idle,
            history,
        })
    }

    /// Entry rows a drafting transaction admits per request: a
    /// verification's accepted prefix and its anchor.
    fn entry_bound(&self) -> usize {
        self.proposals + 1
    }

    fn round_state(&self) -> DraftRoundState {
        self.history.round_state(self.entry_bound())
    }

    /// Enter every pending pair but the last `keep` as a drafter transaction
    /// without proposals.
    fn flush(
        &mut self,
        request: RequestId,
        keep: usize,
        phase: HeadPhase,
        resume_draft: Option<usize>,
    ) -> Result<MethodEffects, String> {
        let call = self.round_state().flushing(keep).0;
        let (Some(call), HistoryState::OwnHistory { pending, .. }) = (call, &mut self.history)
        else {
            return Ok(MethodEffects::default());
        };
        let rows = pending
            .take()
            .expect("a described head entry owns pending pairs");
        let entered = call.entry_rows;
        let conditioning = rows
            .features
            .slice(0, entered)
            .map_err(|error| error.to_string())?;
        if keep > 0 {
            *pending = Some(PendingRows {
                tokens: rows.tokens[entered..].to_vec(),
                features: rows
                    .features
                    .slice(entered, keep)
                    .map_err(|error| error.to_string())?,
            });
        }
        let transaction = HeadTransaction::new(
            request,
            phase,
            PendingRows {
                tokens: rows.tokens[..entered].to_vec(),
                features: conditioning,
            },
            call.position,
            Vec::new(),
            self.drafter.form(),
        )?;
        let operation = transaction.operation().clone();
        self.round = RoundStage::Entering {
            transaction,
            resume_draft,
        };
        Ok(MethodEffects {
            operations: vec![operation],
        })
    }

    pub(crate) fn open_rows(reader: &mut dyn FeatureReader, open: OpenFeature) -> Result<FeatureRows, String> {
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

    pub(crate) fn read(
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
        // Its history's positions trail the target's past a gap, and a
        // device entry anchors at the target's (`Priming::draft_from`).
        match &self.history {
            HistoryState::OwnHistory {
                position,
                pending: None,
                open,
                skipped: 0,
                ..
            } if !matches!(open, Some(OpenFeature::Host(_))) && self.round.is_idle() => {
                Some(*position)
            }
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
        let is_idle = self.round.is_idle();
        let priming_pos = self.priming_position();
        let name = self.drafter.name();
        let (keep, draft_from) = match &mut self.history {
            // Enter a committed target chunk into a head reading the target's
            // history: nothing enters it, the chunk moves its position, and the
            // closing chunk's final feature selected `next`, the first anchor.
            HistoryState::TargetReading { position, anchor } => {
                if primed.is_some() || tokens.is_empty() {
                    return Err("a head reading the target's history primes nothing".into());
                }
                if !is_idle {
                    return Err(format!("{name} method already has unresolved work"));
                }
                let next_position = position
                    .checked_add(tokens.len())
                    .ok_or_else(|| format!("{name} drafter position exhausted"))?;
                *anchor = match next {
                    Some(next) => {
                        let features = features.ok_or(
                            "the chunk closing the prompt requires its final target feature",
                        )?;
                        let last = Self::read(reader, features, tokens.len() - 1, 1)?;
                        Some(AnchorRow::new(next, last)?)
                    }
                    None => None,
                };
                *position = next_position;
                return Ok(MethodEffects::default());
            }
            HistoryState::OwnHistory {
                position,
                pending,
                open,
                anchored,
                skipped,
            } => {
                let features = features.ok_or_else(|| {
                    format!("{name} priming requires the chunk's target features")
                })?;
                *anchored = true;
                // The first draft's anchor in the drafter's positions.
                let draft_from = draft_from.saturating_sub(*skipped);
                if let Some(primed) = primed {
                    // The device entered complete causal pairs. MTP leaves the
                    // final feature open; a feature-only drafter may enter it now.
                    // A finishing chunk's last feature selects the draft anchor.
                    if priming_pos != Some(*position) || primed > tokens.len() {
                        return Err(format!("{name} state cannot take a device-primed chunk"));
                    }
                    *position = position
                        .checked_add(primed)
                        .ok_or_else(|| format!("{name} drafter position exhausted"))?;
                    *open = None;
                    if self.drafter == Drafter::Mtp {
                        let boundary = features
                            .boundary()
                            .cloned()
                            .ok_or("MTP priming requires an independently owned boundary feature")?;
                        if let Some(next) = next {
                            *pending = Some(PendingRows::joined(
                                pending.take(),
                                vec![next],
                                Self::read(reader, boundary, 0, 1)?,
                            )?);
                        } else {
                            *open = Some(OpenFeature::Device(boundary.downgrade()));
                        }
                    } else if let Some(next) = next {
                        let last = Self::read(reader, features, tokens.len() - 1, 1)?;
                        *pending = Some(PendingRows::joined(pending.take(), vec![next], last)?);
                    }
                    return Ok(MethodEffects::default());
                }
                if !is_idle {
                    return Err(format!("{name} method already has unresolved work"));
                }
                let Some((&first, rest)) = tokens.split_first() else {
                    return Err(format!("{name} cannot prime an empty target chunk"));
                };
                let rows = Self::read(reader, features, 0, tokens.len())?;
                // The open feature selected this chunk's first token; each chunk
                // row's feature selected the token after it.
                if let Some(open_feature) = open.take() {
                    *pending = Some(PendingRows::joined(
                        pending.take(),
                        vec![first],
                        Self::open_rows(reader, open_feature)?,
                    )?);
                }
                if !rest.is_empty() {
                    *pending = Some(PendingRows::joined(
                        pending.take(),
                        rest.to_vec(),
                        rows.slice(0, rest.len())
                            .map_err(|error| error.to_string())?,
                    )?);
                }
                let last = rows
                    .slice(tokens.len() - 1, 1)
                    .map_err(|error| error.to_string())?;
                match next {
                    Some(next) => {
                        *pending = Some(PendingRows::joined(pending.take(), vec![next], last)?);
                        // Keep the anchor for the first draft.
                        (1, draft_from)
                    }
                    None => {
                        *open = Some(OpenFeature::Host(last));
                        (0, draft_from)
                    }
                }
            }
        };
        self.flush(request, keep, HeadPhase::Priming { draft_from }, None)
    }

    fn propose(&mut self, request: RequestId, selects: &[SelectSpec]) -> Propose {
        let entry_bound = self.entry_bound();
        let generated = match &mut self.round {
            RoundStage::Proposed { proposal, verified } => {
                // The verified prefix is all a round offers the target.
                let count = match (&mut self.width, *verified) {
                    (_, Some(verified)) => verified,
                    (WidthPolicy::Adaptive(control), None) => control.verify(selects.len()),
                    (WidthPolicy::Fixed(budget), None) => budget.verified,
                }
                .min(proposal.len())
                .min(selects.len());
                *verified = Some(count);
                return Propose::Tokens(proposal[..count].to_vec());
            }
            RoundStage::Declined | RoundStage::Entering { .. } | RoundStage::Drafting(_) => {
                return Propose::Tokens(Vec::new());
            }
            RoundStage::ResumeDraft(generated)
                if *generated > 0 && *generated <= selects.len() =>
            {
                if self.history.has_open() {
                    return Propose::Tokens(Vec::new());
                }
                *generated
            }
            stage @ (RoundStage::Idle | RoundStage::ResumeDraft(_)) => {
                if self.history.has_open() || selects.is_empty() {
                    return Propose::Tokens(Vec::new());
                }
                // A round drafts a prepared width that fits what the request can
                // still produce, or nothing; only from its anchor's pair. A
                // fused-taps drafter under the controller that a round publishing no
                // feature left without it asks the controller still, which prepares
                // the next round's draft (`DraftController::generate`); any other
                // drafter without its anchor's pair drafts nothing. Launch costs
                // depend on the history the round reads, the target's position.
                let (anchored, unanchored_for_prep) = match &self.history {
                    HistoryState::TargetReading { anchor, .. } => {
                        (anchor.is_some(), anchor.is_none())
                    }
                    HistoryState::OwnHistory {
                        pending, anchored, ..
                    } => (*anchored && pending.is_some(), !*anchored),
                };
                let prepares = unanchored_for_prep
                    && self.drafter.fuses_taps()
                    && matches!(self.width, WidthPolicy::Adaptive(_));
                if !anchored && !prepares {
                    return Propose::Tokens(Vec::new());
                }
                let position = self
                    .history
                    .round_state(entry_bound)
                    .target_position()
                    .expect("drafter state preserves its method boundary")
                    + self.history.skipped();
                let generated = match &mut self.width {
                    WidthPolicy::Adaptive(control) => {
                        control.generate(selects.len(), position, anchored)
                    }
                    WidthPolicy::Fixed(budget) => budget.generated_within(selects.len()),
                };
                if generated == 0 || generated > selects.len() {
                    *stage = RoundStage::Declined;
                    return Propose::Tokens(Vec::new());
                }
                generated
            }
        };
        // Rounds that drafted nothing left their pairs pending: the drafter
        // enters them now, then this round drafts from its anchor.
        if self.round_state().catching_up().0.is_some() {
            let effects = self
                .flush(request, 1, HeadPhase::Generation, Some(generated))
                .expect("pending pairs slice by their own rows");
            return Propose::Pending(effects.operations);
        }
        let pairs = self
            .history
            .draft_pairs()
            .expect("a drafting round holds its anchor");
        let (Some(call), _) = self
            .round_state()
            .proposing(generated)
            .expect("budgeted proposal fits the method boundary")
        else {
            unreachable!("positive proposal describes a head transaction");
        };
        let Ok(transaction) = HeadTransaction::new(
            request,
            HeadPhase::Generation,
            pairs,
            call.position,
            selects[..generated].to_vec(),
            self.drafter.form(),
        ) else {
            self.round = RoundStage::Declined;
            return Propose::Tokens(Vec::new());
        };
        let operation = transaction.operation().clone();
        self.round = RoundStage::Drafting(transaction);
        Propose::Pending(vec![operation])
    }

    fn observe(
        &mut self,
        request: RequestId,
        verification: Verification<'_>,
        reader: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String> {
        let name = self.drafter.name();
        if self.round.has_transaction() {
            return Err(format!(
                "{name} cannot observe while a drafter transaction is unresolved"
            ));
        }
        let accepted = verification.accepted;
        if verification.inputs.is_empty() || accepted == 0 || accepted > verification.inputs.len() {
            return Err(format!("{name} verification prefix is invalid"));
        }
        if verification.features.is_none() && matches!(self.width, WidthPolicy::Fixed(_)) {
            return Err(format!("{name} verification requires target features"));
        }
        self.round = RoundStage::Idle;
        if let WidthPolicy::Adaptive(control) = &mut self.width {
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
        let entry_bound = self.entry_bound();
        let needs_maintenance = match &mut self.history {
            HistoryState::TargetReading { position, anchor } => {
                if anchor
                    .as_ref()
                    .is_some_and(|row| row.token != verification.inputs[0])
                {
                    return Err(format!(
                        "{name} pending anchor differs from the verified anchor"
                    ));
                }
                let Some(features) = verification.features else {
                    *anchor = None;
                    *position = position
                        .checked_add(accepted)
                        .ok_or_else(|| format!("{name} drafter position exhausted"))?;
                    return Ok(MethodEffects::default());
                };
                let transition = round_state_of(
                    DraftHistory::Target,
                    entry_bound,
                    *position,
                    usize::from(anchor.is_some()),
                    false,
                )
                .observed(accepted)?;
                // The accepted rows moved the target; the last one's feature
                // selected the next anchor.
                let last = Self::read(reader, features, accepted - 1, 1)?;
                *anchor = Some(AnchorRow::new(verification.next, last)?);
                *position = transition.appended.position();
                debug_assert_eq!(
                    round_state_of(DraftHistory::Target, entry_bound, *position, 1, false),
                    transition.appended
                );
                false
            }
            HistoryState::OwnHistory {
                position,
                pending,
                open,
                anchored,
                skipped,
            } => {
                // After a replay the last replayed row's feature selected the anchor.
                if let Some(open_feature) = open.take() {
                    *pending = Some(PendingRows::joined(
                        pending.take(),
                        vec![verification.inputs[0]],
                        Self::open_rows(reader, open_feature)?,
                    )?);
                    *anchored = true;
                }
                if *anchored
                    && pending
                        .as_ref()
                        .and_then(|pending| pending.tokens.last())
                        .is_some_and(|anchor| *anchor != verification.inputs[0])
                {
                    return Err(format!(
                        "{name} pending anchor differs from the verified anchor"
                    ));
                }
                // A round that published no feature (one drafting nothing under the
                // controller) leaves the drafter without the pairs of its rows'
                // successors: the next round cannot draft, and an own history goes
                // on past them from the next pair it is given.
                let Some(features) = verification.features else {
                    *anchored = false;
                    *skipped += accepted;
                    return Ok(MethodEffects::default());
                };
                *anchored = true;
                let transition = round_state_of(
                    DraftHistory::Own,
                    entry_bound,
                    *position,
                    pending.as_ref().map_or(0, |pending| pending.tokens.len()),
                    false,
                )
                .observed(accepted)?;
                // Committed row i's feature selected the token after it: the next
                // accepted input, or the round's successor after the last row.
                let rows = Self::read(reader, features, 0, accepted)?;
                let mut tokens = verification.inputs[1..accepted].to_vec();
                tokens.push(verification.next);
                *pending = Some(PendingRows::joined(pending.take(), tokens, rows)?);
                debug_assert_eq!(
                    round_state_of(
                        DraftHistory::Own,
                        entry_bound,
                        *position,
                        pending.as_ref().map_or(0, |pending| pending.tokens.len()),
                        false,
                    ),
                    transition.appended
                );
                transition.maintenance.is_some()
            }
        };
        if needs_maintenance {
            return self.flush(request, 1, HeadPhase::Generation, None);
        }
        Ok(MethodEffects::default())
    }

    /// A drafter whose feature costs the step nothing, or one without the
    /// controller, takes every round's. Under the controller a fused-taps
    /// drafter takes one only for the next round's draft: from a round that
    /// drafted (its rows' successors are the next draft's pairs) or one
    /// preparing to draft, so a round drafting nothing costs a plain step.
    fn taps(&self) -> bool {
        match &self.width {
            WidthPolicy::Adaptive(control) if self.drafter.fuses_taps() => {
                self.round.is_drafting() || control.preparing()
            }
            _ => true,
        }
    }

    fn publishes_behind(&self) -> bool {
        self.drafter.fuses_taps()
            && matches!(
                &self.width,
                WidthPolicy::Adaptive(control) if control.publishes_behind()
            )
    }

    fn reconcile(&mut self, operation: &Operation, outcome: Outcome) -> Result<(), String> {
        enum ReconcileKind {
            Enter(Option<usize>),
            Draft,
        }
        let name = self.drafter.name();
        let (transaction, kind) = match std::mem::replace(&mut self.round, RoundStage::Idle) {
            RoundStage::Entering {
                transaction,
                resume_draft,
            } => (transaction, ReconcileKind::Enter(resume_draft)),
            RoundStage::Drafting(transaction) => (transaction, ReconcileKind::Draft),
            other => {
                self.round = other;
                return Err(format!("{name} has no pending drafter transaction"));
            }
        };
        let drafted = transaction
            .reconcile(operation, outcome)
            .map_err(|error| format!("{name}: {error}"))?;
        // A drafter without history entered nothing: its position is the
        // target's, which only committed target rows move.
        if let HistoryState::OwnHistory { .. } = &self.history {
            let new_position = self.round_state().entered(drafted.entered)?.position();
            if let HistoryState::OwnHistory { position, .. } = &mut self.history {
                *position = new_position;
            }
        }
        match kind {
            ReconcileKind::Enter(Some(generated)) => {
                self.round = RoundStage::ResumeDraft(generated);
            }
            ReconcileKind::Enter(None) => {}
            ReconcileKind::Draft => {
                self.history.clear_pending();
                if let WidthPolicy::Adaptive(control) = &mut self.width {
                    control.drafted(
                        drafted.requested,
                        &drafted.proposal,
                        drafted.confidence.as_deref(),
                    );
                }
                self.round = RoundStage::Proposed {
                    proposal: drafted.proposal,
                    verified: None,
                };
            }
        }
        Ok(())
    }

    fn timed(
        &mut self,
        launch: TimedLaunch,
        duration: Option<LaunchDuration>,
        sequences: usize,
        rows: usize,
    ) {
        if let WidthPolicy::Adaptive(control) = &mut self.width {
            control.timed(launch, duration, sequences, rows);
        }
    }

    fn outlook(&self) -> Option<DraftOutlook> {
        match (&self.width, &self.round) {
            (WidthPolicy::Adaptive(control), RoundStage::Proposed { verified: None, .. }) => {
                control.outlook()
            }
            _ => None,
        }
    }

    fn checkpoint(&self) -> Result<MethodCheckpoint, MethodCheckpointError> {
        if !self.round.is_idle() {
            return Err(MethodCheckpointError::Unresolved);
        }
        let state = match &self.history {
            HistoryState::TargetReading { position, .. } => {
                // The pending anchor is the sampled successor, outside the
                // retained prefix; its request's next chunk selects its own.
                DraftCheckpoint {
                    position: *position,
                    tail: DraftTail::EMPTY,
                    skipped: 0,
                }
            }
            HistoryState::OwnHistory {
                position,
                pending,
                open,
                anchored,
                skipped,
            } => {
                let mut pending = pending.clone();
                let mut open = open.clone();
                // The final pending token is the sampled successor, outside the
                // retained target prefix. Export its feature without that token;
                // each restored request supplies its own successor. A drafter
                // without its anchor's pair holds no successor: its pairs are all
                // inside the prefix.
                if open.is_none() && *anchored {
                    if let Some(rows) = pending.take() {
                        let (rest, last) = rows.opened();
                        pending = rest;
                        open = Some(OpenFeature::Host(last));
                    }
                }
                DraftCheckpoint {
                    position: *position,
                    tail: DraftTail::Trailing { pending, open },
                    skipped: *skipped,
                }
            }
        };
        Ok(self.drafter.checkpoint(state))
    }

    fn evict(&mut self) {
        self.history.reset();
        self.round = RoundStage::Idle;
    }

    fn reclaimable(&self) -> u64 {
        match &self.history {
            HistoryState::TargetReading { anchor, .. } => {
                anchor.as_ref().map_or(0, |row| row.feature.bytes().len()) as u64
            }
            HistoryState::OwnHistory { pending, open, .. } => {
                let pending = pending
                    .as_ref()
                    .map_or(0, |rows| rows.features.bytes().len());
                let open = open.as_ref().map_or(0, OpenFeature::host_bytes);
                (pending + open) as u64
            }
        }
    }
}
