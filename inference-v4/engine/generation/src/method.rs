use magnitude_model_executor::{
    Demand, FeatureReader, FeatureRef, FeatureRows, Operation, Outcome, RequestId, SelectSpec,
    TokenId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MethodChoice {
    Plain,
    Mtp { proposals: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MethodRequirements {
    pub prefill_demand: Demand,
    pub verify_demand: Demand,
    pub head: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MethodCheckpoint {
    Plain,
    Mtp(MtpCheckpoint),
}

impl MethodCheckpoint {
    /// Host bytes the checkpoint keeps.
    pub fn retained_bytes(&self) -> u64 {
        match self {
            Self::Plain => 0,
            Self::Mtp(checkpoint) => checkpoint.retained_bytes(),
        }
    }
}

/// The draft head's reconciled state at a numerical boundary: the head
/// position and the target-conditioned rows it has not yet entered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtpCheckpoint {
    pub(crate) position: usize,
    pub(crate) pending: Option<PendingRows>,
    pub(crate) open: Option<FeatureRows>,
}

/// Complete head input pairs: each token with the target feature of the
/// row before it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PendingRows {
    pub(crate) tokens: Vec<TokenId>,
    pub(crate) features: FeatureRows,
}

impl MtpCheckpoint {
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Target rows this state accounts for: every entered head row, every
    /// pending pair, and the open feature.
    pub fn target_rows(&self) -> usize {
        self.position
            + self.pending.as_ref().map_or(0, |rows| rows.tokens.len())
            + self.open.as_ref().map_or(0, FeatureRows::rows)
    }

    pub fn retained_bytes(&self) -> u64 {
        let pending = self
            .pending
            .as_ref()
            .map_or(0, |rows| rows.features.bytes().len());
        let open = self.open.as_ref().map_or(0, |rows| rows.bytes().len());
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
    /// Enter a committed target chunk. `next` is the token the chunk's last
    /// row selected, when it selected one.
    fn prime(
        &mut self,
        request: RequestId,
        tokens: &[TokenId],
        next: Option<TokenId>,
        features: FeatureRef,
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
