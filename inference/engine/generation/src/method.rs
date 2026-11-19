use crate::{DraftOutlook, TimedLaunch};
use magnitude_executor::{
    ChainEntry, Demand, FeatureReader, FeatureRef, FeatureRows, FrontGrant, LaunchDuration, Operation, Outcome,
    RequestId, SelectSpec, TokenId, WeakFeatureRef,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MethodChoice {
    Plain,
    Mtp {
        proposals: u8,
    },
    /// A separate DFlash or DSpark draft model.
    DFlash {
        proposals: u8,
    },
}

impl MethodChoice {
    /// The most tokens one round drafts.
    pub fn proposals(self) -> usize {
        match self {
            Self::Plain => 0,
            Self::Mtp { proposals } | Self::DFlash { proposals } => usize::from(proposals),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MethodRequirements {
    /// Every prompt or replay chunk's demand.
    pub prefill_demand: Demand,
    /// Added to the prompt chunk that closes the prompt: the one selecting
    /// (or forcing) the first generated token. A drafter reading the
    /// target's history needs only that chunk's final feature.
    pub closing_demand: Demand,
    pub verify_demand: Demand,
    pub head: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MethodCheckpoint {
    Plain,
    Mtp(DraftCheckpoint),
    DFlash(DraftCheckpoint),
}

impl MethodCheckpoint {
    /// Whether a generation under `choice` resumes from this checkpoint
    /// taken at `rows` target rows, where its path holds `successor` next
    /// (none past its path's end): it was taken under the same method, so
    /// the drafter history the retained state holds (or lacks) is the one
    /// the generation needs; a drafter's accounts for exactly those rows;
    /// and one that entered the successor of its rows
    /// (`DraftTail::Entered`) entered this path's.
    pub fn resumes(&self, choice: MethodChoice, rows: usize, successor: Option<TokenId>) -> bool {
        match (choice, self) {
            (MethodChoice::Plain, Self::Plain) => true,
            (MethodChoice::Mtp { .. }, Self::Mtp(state)) | (MethodChoice::DFlash { .. }, Self::DFlash(state)) => {
                state.target_rows() == rows
                    && match state.tail {
                        DraftTail::Trailing { .. } => true,
                        DraftTail::Entered(entered) => successor == Some(entered),
                    }
            }
            _ => false,
        }
    }

    /// Whether the checkpoint accounts for exactly `rows` target rows, so it
    /// resumes beside a target state of that position.
    pub fn accounts_for(&self, rows: usize) -> bool {
        match self {
            Self::Plain => true,
            Self::Mtp(checkpoint) | Self::DFlash(checkpoint) => checkpoint.target_rows() == rows,
        }
    }

    /// Host bytes the checkpoint keeps.
    pub fn retained_bytes(&self) -> u64 {
        match self {
            Self::Plain => 0,
            Self::Mtp(checkpoint) | Self::DFlash(checkpoint) => checkpoint.retained_bytes(),
        }
    }
}

/// A target-conditioned drafter's reconciled state at a numerical boundary:
/// its position, where its history ends against the target's rows, and the
/// target rows its history omits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftCheckpoint {
    pub(crate) position: usize,
    pub(crate) tail: DraftTail,
    /// Target rows the drafter's history omits (`DrafterState::skipped`).
    pub(crate) skipped: usize,
}

/// Where a drafter's history ends against the target's rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DraftTail {
    /// Short of the target's rows: `pending` pairs, read on the host, await
    /// entry, then `open`, the target's last row's feature, awaits its
    /// successor token. Both none only before the first chunk, or where
    /// every pair is entered and no feature is held.
    Trailing {
        pending: Option<PendingRows>,
        open: Option<OpenFeature>,
    },
    /// Level with them: the head entered the successor of the target's last
    /// row, this token, with that row's feature. A head decoding in front
    /// passes enters each selecting step's selection on the device behind
    /// it (`FrontState`), so its history holds a token of its path the
    /// target's rows do not.
    Entered(TokenId),
}

impl DraftTail {
    /// Nothing awaits entry.
    pub(crate) const EMPTY: Self = Self::Trailing { pending: None, open: None };

    /// Target rows past the head's entered ones it accounts for: each
    /// pending pair and the open feature.
    fn rows(&self) -> usize {
        match self {
            Self::Trailing { pending, open } => {
                pending.as_ref().map_or(0, |rows| rows.tokens.len()) + usize::from(open.is_some())
            }
            Self::Entered(_) => 0,
        }
    }

    /// Host bytes it holds.
    pub(crate) fn host_bytes(&self) -> usize {
        match self {
            Self::Trailing { pending, open } => {
                pending.as_ref().map_or(0, |rows| rows.features.bytes().len())
                    + open.as_ref().map_or(0, OpenFeature::host_bytes)
            }
            Self::Entered(_) => 0,
        }
    }
}

/// One target feature awaiting its successor token. Device rows are owned by
/// executor numerical state; checkpoints never retain a graph output arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum OpenFeature {
    Host(FeatureRows),
    Device(WeakFeatureRef),
}

impl OpenFeature {
    pub(crate) fn host_bytes(&self) -> usize {
        match self {
            Self::Host(rows) => rows.bytes().len(),
            Self::Device(_) => 0,
        }
    }
}

/// Complete head input pairs: each token with the target feature of the
/// row before it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PendingRows {
    pub(crate) tokens: Vec<TokenId>,
    pub(crate) features: FeatureRows,
}

impl PendingRows {
    /// `pending` with the pairs of `tokens` and their `features` after it.
    pub(crate) fn joined(pending: Option<Self>, tokens: Vec<TokenId>, features: FeatureRows) -> Result<Self, String> {
        if tokens.len() != features.rows() {
            return Err("drafter pairs differ from their feature rows".into());
        }
        Ok(match pending {
            None => Self { tokens, features },
            Some(mut pending) => {
                pending.features = pending.features.concat(&features).map_err(|error| error.to_string())?;
                pending.tokens.extend(tokens);
                pending
            }
        })
    }

    /// Its pairs but the last, and the last pair's feature: the pairs of a
    /// retained prefix whose last row's successor, the last pair's token,
    /// lies past it.
    pub(crate) fn opened(self) -> (Option<Self>, FeatureRows) {
        let last = self.tokens.len() - 1;
        let open = self.features.slice(last, 1).expect("pending row shape");
        let rest = (last > 0).then(|| Self {
            tokens: self.tokens[..last].to_vec(),
            features: self.features.slice(0, last).expect("pending row shape"),
        });
        (rest, open)
    }
}

impl DraftCheckpoint {
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Target rows this state accounts for: every entered head row (each
    /// pairs a target row's feature with its successor), every pending
    /// pair, the open feature, and the rows whose pairs it omits.
    pub fn target_rows(&self) -> usize {
        self.position + self.tail.rows() + self.skipped
    }

    pub fn retained_bytes(&self) -> u64 {
        self.tail.host_bytes() as u64
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MethodCheckpointError {
    Unresolved,
}

impl std::fmt::Display for MethodCheckpointError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unresolved => formatter.write_str("cannot checkpoint unresolved method work"),
        }
    }
}

impl std::error::Error for MethodCheckpointError {}

#[derive(Clone, Debug)]
pub struct Verification<'a> {
    pub inputs: &'a [TokenId],
    /// Committed input rows, the anchor included.
    pub accepted: usize,
    pub next: TokenId,
    pub features: Option<FeatureRef>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Propose {
    Tokens(Vec<TokenId>),
    Pending(Vec<Operation>),
}

/// Executor work a method's state transition requires.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MethodEffects {
    pub operations: Vec<Operation>,
}

/// The next round of a method decoding in front passes
/// ([`MethodState::front`]), by the form its grant gives and where the
/// head's history ends.
#[derive(Clone, Debug, PartialEq)]
pub enum FrontEntry {
    /// The head holds every pair up to the pass's pending token, and the
    /// grant is a front pass's: the pass enters its exact row's selection
    /// behind it on the device, from here.
    Chain(ChainEntry),
    /// The head holds every pair up to the pass's pending token, and the
    /// grant is a verify pass's: the pass enters nothing behind it, its
    /// accepted rows' pairs reaching the head through the host
    /// ([`MethodState::observe`]).
    Verify,
    /// The head trails the target: the head transaction that enters the
    /// pairs it lacks on the host and drafts the next pass's chain from the
    /// last; the pass follows it, by its own grant.
    Pending(Vec<Operation>),
}

pub trait Method: Send + Sync {
    /// Whether each entered head row consumes its target feature's successor token.
    fn priming_requires_successor(&self) -> bool { false }
    /// Whether the drafter's history embeds every prompt token. Such a
    /// drafter cannot enter prompt rows whose input is not a token (a media
    /// span's rows are the projector's, under a placeholder token).
    fn embeds_prompt_tokens(&self) -> bool;
    fn identity(&self) -> &str;
    fn requires(&self) -> MethodRequirements;
    /// The most proposals one round may draft.
    fn proposals(&self) -> usize;
    fn create(&self, checkpoint: Option<&MethodCheckpoint>)
    -> Result<Box<dyn MethodState>, String>;
    /// Whether its generations decode in front passes
    /// (`magnitude_executor::TwoRowDecode`) on the chain the MTP head makes,
    /// each in the form its launch grants, and never propose on their own:
    /// the MTP method of a head of its own history on a load that fronts
    /// ([`crate::Mtp::fronting`]). Any other drafts, or decodes plainly,
    /// one row a pass.
    fn decodes_two_rows(&self) -> bool {
        false
    }
}

/// Per-request method state remains confined to its owner thread.
pub trait MethodState {
    /// Clone request-local method state for a fallible transition.
    fn fork_transition(&self) -> Box<dyn MethodState>;
    /// The drafter position a prompt chunk's entry takes when the drafter
    /// enters prompt chunks on the device, behind the chunk (a
    /// target-conditioned drafter with no rows pending); `None` when chunks
    /// are entered by method work after they commit.
    fn priming_position(&self) -> Option<usize>;
    /// Enter a committed target chunk. `next` is the token the chunk's last
    /// row selected, when it selected one; `draft_from` is the position the
    /// first draft after this history anchors at (the prefilled length).
    /// `primed` rows of the chunk's entry were drafted on the device behind
    /// it (see [`MethodState::priming_position`]). `features` are present
    /// exactly when the chunk demanded them (`prefill_demand`, and
    /// `closing_demand` when the chunk closes the prompt).
    #[allow(clippy::too_many_arguments)]
    fn prime(
        &mut self,
        request: RequestId,
        tokens: &[TokenId],
        next: Option<TokenId>,
        features: Option<FeatureRef>,
        draft_from: usize,
        primed: Option<usize>,
        reader: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String>;
    /// Propose one token per selection (keyed by the target's position), or
    /// return the executor work that drafts them. A state decoding in front
    /// passes proposes nothing and is never asked ([`Self::front`]).
    fn propose(&mut self, request: RequestId, selects: &[SelectSpec]) -> Propose;
    /// The next round of a state decoding in front passes
    /// ([`Method::decodes_two_rows`]), of pending token `pending`, its pass
    /// in the form `grant` gives; where its head trails the target, the
    /// head transaction drafting the chain first, a proposal per `selects`.
    /// It is asked in place of [`Self::propose`]; any other state never is.
    fn front(
        &mut self,
        _request: RequestId,
        _pending: TokenId,
        _grant: FrontGrant,
        _selects: &[SelectSpec],
    ) -> Result<FrontEntry, String> {
        unreachable!("a drafting method's decode proposes, decoding in no front pass")
    }
    /// Whether its head trails the target, so that its next decode round
    /// is the head transaction drafting the chain ([`Self::front`]), of a
    /// state decoding in front passes. Any other state is never asked.
    fn trails(&self) -> bool {
        unreachable!("a drafting method's head drafts no chain for a front pass")
    }
    fn observe(
        &mut self,
        request: RequestId,
        verification: Verification<'_>,
        reader: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String>;
    /// Whether this round's target step, once it verifies no proposal,
    /// publishes its rows' features (`MethodRequirements::verify_demand`):
    /// a method reading them only to draft takes them only for a round it
    /// drafts from next. Asked after `propose`.
    fn taps(&self) -> bool {
        true
    }
    /// Whether the decode queued behind this round's target step, which
    /// publishes no feature, publishes its own: the next round prepares to
    /// draft as that decode. Asked after `propose`.
    fn publishes_behind(&self) -> bool {
        false
    }
    /// Consume the outcome of the exact operation `propose` returned.
    fn reconcile(&mut self, operation: &Operation, outcome: Outcome) -> Result<(), String>;
    /// A launch carrying this request's work finished: `rows` rows of
    /// `sequences` requests, in `duration` when it replayed its graph.
    fn timed(
        &mut self,
        _launch: TimedLaunch,
        _duration: Option<LaunchDuration>,
        _sequences: usize,
        _rows: usize,
    ) {
    }
    /// The drafted block this request's next target round verifies, when
    /// the method chooses how much of it to verify.
    fn outlook(&self) -> Option<DraftOutlook> {
        None
    }
    fn checkpoint(&self) -> Result<MethodCheckpoint, MethodCheckpointError>;
    fn evict(&mut self);
    fn reclaimable(&self) -> u64;
}

#[cfg(test)]
mod tests {
    use super::{DraftCheckpoint, DraftTail, MethodCheckpoint};

    /// A drafter's checkpoint resumes only beside a target state of exactly
    /// the rows it accounts for (its entered rows and the rows it omits): a
    /// stop whose checkpoint trails or leads the target's rows is not a
    /// resume point. A plain request's checkpoint accounts for any rows.
    #[test]
    fn a_checkpoint_accounts_only_for_its_own_rows() {
        let draft = DraftCheckpoint { position: 70, tail: DraftTail::EMPTY, skipped: 2 };
        for checkpoint in [MethodCheckpoint::Mtp(draft.clone()), MethodCheckpoint::DFlash(draft)] {
            assert!(checkpoint.accounts_for(72), "{checkpoint:?}");
            assert!(!checkpoint.accounts_for(70), "{checkpoint:?}: rows it omits are still its own");
            assert!(!checkpoint.accounts_for(73), "{checkpoint:?}: a row it has not seen");
        }
        assert!(MethodCheckpoint::Plain.accounts_for(5));
    }
}
