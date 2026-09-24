//! Multi-token prediction with the model's draft head.
//!
//! The head's committed rows pair each accepted token with the target's
//! normalized output feature of the row before it (the feature that
//! selected that token). Those rows are target-conditioned only: rows the
//! head conditioned on its own features while drafting are never kept.
//! A round's draft is one head transaction whose entry rows are every pair
//! not yet entered, ending with the anchor (the last accepted token); the
//! executor chains the proposals on the device.

use crate::{
    Method, MethodCheckpoint, MethodCheckpointError, MethodEffects, MethodRequirements,
    MethodState, MtpCheckpoint, Propose, Verification, method::PendingRows,
};
use magnitude_model_executor::{
    Demand, FeatureReader, FeatureRef, FeatureRows, FeatureSpan, Operation, Outcome, RequestId,
    SelectSpec, TokenId,
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
        let checkpoint = match checkpoint {
            None => None,
            Some(MethodCheckpoint::Mtp(checkpoint)) => Some(checkpoint.clone()),
            Some(MethodCheckpoint::Plain) => {
                return Err("a plain checkpoint cannot restore MTP state".into());
            }
        };
        Ok(Box::new(MtpState::new(self.proposals, checkpoint)))
    }
}

#[derive(Clone)]
struct MtpState {
    proposals: usize,
    /// Head rows entered so far; the next entry row's head position.
    position: usize,
    /// Complete pairs not yet entered; the last is the anchor.
    pending: Option<PendingRows>,
    /// A committed feature whose successor token is not known yet.
    open: Option<FeatureRows>,
    /// The head transaction `propose` returned, until it reconciles.
    head: Option<Operation>,
    /// Reconciled proposals awaiting their verification.
    proposal: Option<Vec<TokenId>>,
}

impl MtpState {
    fn new(proposals: usize, checkpoint: Option<MtpCheckpoint>) -> Self {
        let checkpoint = checkpoint.unwrap_or(MtpCheckpoint {
            position: 0,
            pending: None,
            open: None,
        });
        Self {
            proposals,
            position: checkpoint.position,
            pending: checkpoint.pending,
            open: checkpoint.open,
            head: None,
            proposal: None,
        }
    }

    /// Entry rows a drafting head admits per request: a verification's
    /// accepted prefix and its anchor.
    fn entry_bound(&self) -> usize {
        self.proposals + 1
    }

    fn ensure_idle(&self) -> Result<(), String> {
        if self.head.is_some() || self.proposal.is_some() {
            return Err("MTP method already has unresolved work".into());
        }
        Ok(())
    }

    fn append(&mut self, tokens: Vec<TokenId>, features: FeatureRows) -> Result<(), String> {
        if tokens.len() != features.rows() {
            return Err("MTP pairs differ from their feature rows".into());
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

    /// Enter every pending pair but the last `keep` as a causal head
    /// transaction.
    fn flush(&mut self, request: RequestId, keep: usize) -> Result<MethodEffects, String> {
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
            tokens: pending.tokens[..entered].to_vec(),
            conditioning,
            position: self.position,
            proposals: Vec::new(),
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
        count: usize,
    ) -> Result<FeatureRows, String> {
        let span = FeatureSpan::new(features, 0, count).map_err(|error| error.to_string())?;
        reader.read(&span)
    }
}

impl MethodState for MtpState {
    fn fork_transition(&self) -> Box<dyn MethodState> {
        Box::new(self.clone())
    }

    fn prime(
        &mut self,
        request: RequestId,
        tokens: &[TokenId],
        next: Option<TokenId>,
        features: FeatureRef,
        reader: &mut dyn FeatureReader,
    ) -> Result<MethodEffects, String> {
        self.ensure_idle()?;
        let Some((&first, rest)) = tokens.split_first() else {
            return Err("MTP cannot prime an empty target chunk".into());
        };
        let rows = Self::read(reader, features, tokens.len())?;
        // The open feature selected this chunk's first token; each chunk
        // row's feature selected the token after it.
        if let Some(open) = self.open.take() {
            self.append(vec![first], open)?;
        }
        if !rest.is_empty() {
            self.append(
                rest.to_vec(),
                rows.slice(0, rest.len()).map_err(|error| error.to_string())?,
            )?;
        }
        let last = rows
            .slice(tokens.len() - 1, 1)
            .map_err(|error| error.to_string())?;
        match next {
            Some(next) => {
                self.append(vec![next], last)?;
                // Keep the anchor for the first draft.
                self.flush(request, 1)
            }
            None => {
                self.open = Some(last);
                self.flush(request, 0)
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
            tokens: pending.tokens.clone(),
            conditioning: pending.features.clone(),
            position: self.position,
            proposals: selects[..selects.len().min(self.proposals)].to_vec(),
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
        if self.head.is_some() {
            return Err("MTP cannot observe while a head transaction is unresolved".into());
        }
        self.proposal = None;
        let features = verification
            .features
            .ok_or("MTP verification requires target features")?;
        let accepted = verification.accepted;
        if verification.inputs.is_empty() || accepted == 0 || accepted > verification.inputs.len()
        {
            return Err("MTP verification prefix is invalid".into());
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
            return Err("MTP pending anchor differs from the verified anchor".into());
        }
        // Committed row i's feature selected the token after it: the next
        // accepted input, or the round's successor after the last row.
        let rows = Self::read(reader, features, accepted)?;
        let mut tokens = verification.inputs[1..accepted].to_vec();
        tokens.push(verification.next);
        self.append(tokens, rows)?;
        let bound = self.entry_bound();
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.tokens.len() > bound)
        {
            return self.flush(request, 1);
        }
        Ok(MethodEffects::default())
    }

    fn reconcile(&mut self, operation: &Operation, outcome: Outcome) -> Result<(), String> {
        let head = self
            .head
            .take()
            .ok_or("MTP has no pending head transaction")?;
        let (
            Operation::Head {
                tokens, proposals, ..
            },
            Outcome::Head { proposals: selected },
        ) = (&head, outcome)
        else {
            return Err("MTP head outcome has the wrong kind".into());
        };
        if head != *operation || selected.len() != proposals.len() {
            return Err("MTP head outcome differs from its transaction".into());
        }
        self.position = self
            .position
            .checked_add(tokens.len())
            .ok_or("MTP head position exhausted")?;
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
        Ok(MethodCheckpoint::Mtp(MtpCheckpoint {
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
