use magnitude_model_executor::{
    Demand, FeatureRef, FeatureRetainer, Operation, Outcome, RequestId, RetainedFeatureSpan,
    SelectSpec, TokenId,
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
    pub fn retained_bytes(&self) -> u64 {
        match self {
            Self::Plain => 0,
            Self::Mtp(checkpoint) => checkpoint.retained_bytes(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtpCheckpoint {
    pub(crate) position: usize,
    pub(crate) pending: Option<RetainedFeatureSpan>,
    pub(crate) buffer: Vec<(TokenId, RetainedFeatureSpan)>,
}

impl MtpCheckpoint {
    pub const fn position(&self) -> usize {
        self.position
    }

    pub fn retained_bytes(&self) -> u64 {
        self.pending
            .iter()
            .chain(self.buffer.iter().map(|(_, feature)| feature))
            .fold(0, |total, feature| total.saturating_add(feature.bytes()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MethodCheckpointError {
    Unresolved,
    Retention(String),
}

impl std::fmt::Display for MethodCheckpointError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unresolved => formatter.write_str("cannot checkpoint unresolved method work"),
            Self::Retention(message) => {
                write!(formatter, "cannot retain method features: {message}")
            }
        }
    }
}

impl std::error::Error for MethodCheckpointError {}

#[derive(Clone, Debug)]
pub struct Verification<'a> {
    pub inputs: &'a [TokenId],
    pub accepted: usize,
    pub next: TokenId,
    pub features: Option<FeatureRef>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Propose {
    Tokens(Vec<TokenId>),
    Pending(Vec<Operation>),
}

/// Executor work and numerical branch acceptance decided by a generation
/// method. `head_prefix` is a row count within the head executor's current
/// pending branch, never an absolute sequence position.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MethodEffects {
    pub operations: Vec<Operation>,
    pub head_prefix: Option<usize>,
}

pub trait Method: Send + Sync {
    fn identity(&self) -> &str;
    fn requires(&self) -> MethodRequirements;
    fn create(&self, checkpoint: Option<&MethodCheckpoint>) -> Box<dyn MethodState>;
}

/// Per-request method state remains confined to its owner thread. In
/// particular, retained feature leases are intentionally not transportable
/// across executor domains.
pub trait MethodState {
    /// Clone request-local method state for a fallible transition. The clone
    /// shares immutable feature leases but cannot mutate the live method.
    fn fork_transition(&self) -> Result<Box<dyn MethodState>, String> {
        Err("method does not support staged reconciliation".into())
    }
    fn prime(
        &mut self,
        request: RequestId,
        tokens: &[TokenId],
        features: FeatureRef,
    ) -> Result<MethodEffects, String>;
    fn propose(
        &mut self,
        request: RequestId,
        context: &[TokenId],
        limit: usize,
        first_select: SelectSpec,
    ) -> Propose;
    fn observe(&mut self, verification: Verification<'_>) -> Result<MethodEffects, String>;
    /// Consume the outcome of the exact method operation previously returned
    /// by `prime` or `propose`. Method state, not service, owns its meaning.
    fn reconcile(
        &mut self,
        operation: &Operation,
        outcome: Outcome,
        next_select: Option<SelectSpec>,
    ) -> Result<MethodEffects, String>;
    /// Replace every feature row retained across operation boundaries with an
    /// independently owned resource-domain copy.
    fn stabilize(&mut self, _retainer: &mut dyn FeatureRetainer) -> Result<(), String> {
        Ok(())
    }
    fn checkpoint(
        &self,
        retainer: &mut dyn FeatureRetainer,
    ) -> Result<MethodCheckpoint, MethodCheckpointError>;
    fn evict(&mut self);
    fn restore(&mut self);
    fn reclaimable(&self) -> u64;
}
