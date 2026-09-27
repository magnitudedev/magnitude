//! Drafting with a separate DFlash or DSpark draft model.
//!
//! The draft conditions on the target exactly as MTP's head does: its
//! entry rows pair each accepted token with the target's feature of the row
//! before it (for a DFlash model that feature is the fused residual taps), and
//! it enters them into its own history at the target's positions. A round's
//! draft is one transaction ending with the anchor; the executor then reads
//! one non-causal block `[anchor, mask, …]` after that history and selects
//! every proposal from it in one pass (`DraftForm::Block`).

use crate::mtp::{Drafter, DrafterState};
use crate::{Method, MethodCheckpoint, MethodRequirements, MethodState};
use magnitude_executor::Demand;

#[derive(Clone, Debug)]
pub struct DFlash {
    identity: String,
    proposals: usize,
}

impl DFlash {
    pub fn new(artifact: impl AsRef<str>, proposals: usize) -> Result<Self, String> {
        let artifact = artifact.as_ref();
        if artifact.is_empty() || proposals == 0 || proposals > u8::MAX as usize {
            return Err("DFlash identity and proposal width must be nonempty and bounded".into());
        }
        Ok(Self {
            identity: format!("dflash:{artifact}:{proposals}"),
            proposals,
        })
    }
}

impl Method for DFlash {
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
            Drafter::DFlash,
            self.proposals,
            checkpoint,
        )?))
    }
}
