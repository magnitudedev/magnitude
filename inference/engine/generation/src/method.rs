use magnitude_executor::{
    Demand, FeatureReader, FeatureRef, FeatureRows, Operation, Outcome, RequestId, SelectSpec,
    TokenId, WeakFeatureRef,
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
    /// Host bytes the checkpoint keeps.
    pub fn retained_bytes(&self) -> u64 {
        match self {
            Self::Plain => 0,
            Self::Mtp(checkpoint) | Self::DFlash(checkpoint) => checkpoint.retained_bytes(),
        }
    }
}

/// A target-conditioned drafter's reconciled state at a numerical boundary:
/// its position and the target-conditioned rows it has not yet entered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftCheckpoint {
    pub(crate) position: usize,
    pub(crate) pending: Option<PendingRows>,
    pub(crate) open: Option<OpenFeature>,
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

impl DraftCheckpoint {
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Target rows this state accounts for: every entered head row, every
    /// pending pair, and the open feature.
    pub fn target_rows(&self) -> usize {
        self.position
            + self.pending.as_ref().map_or(0, |rows| rows.tokens.len())
            + usize::from(self.open.is_some())
    }

    pub fn retained_bytes(&self) -> u64 {
        let pending = self
            .pending
            .as_ref()
            .map_or(0, |rows| rows.features.bytes().len());
        let open = self.open.as_ref().map_or(0, OpenFeature::host_bytes);
        (pending + open) as u64
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
    /// return the executor work that drafts them.
    fn propose(&mut self, request: RequestId, selects: &[SelectSpec]) -> Propose;
    fn observe(
        &mut self,
        request: RequestId,
        verification: Verification<'_>,
        reader: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String>;
    /// Consume the outcome of the exact operation `propose` returned.
    fn reconcile(&mut self, operation: &Operation, outcome: Outcome) -> Result<(), String>;
    fn checkpoint(&self) -> Result<MethodCheckpoint, MethodCheckpointError>;
    fn evict(&mut self);
    fn reclaimable(&self) -> u64;
}
