//! Drafting with a separate DFlash or DSpark draft model.
//!
//! The draft conditions on the target exactly as MTP's head does: its
//! entry rows pair each accepted token with the target's feature of the row
//! before it (for a DFlash model that feature is the fused residual taps), and
//! it enters them into its own history at the target's positions. A round's
//! draft is one transaction ending with the anchor; the executor then reads
//! one block after that history using the artifact's attention and anchor
//! layout, and selects its proposals in one pass (`DraftForm::Block`).

use crate::mtp::{Drafter, DrafterState};
use crate::{DraftBudget, DraftLoad, Method, MethodCheckpoint, MethodRequirements, MethodState};
use magnitude_executor::Demand;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct DFlash {
    identity: String,
    proposals: usize,
    budget: DraftBudget,
    adaptive: Option<Arc<DraftLoad>>,
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
            budget: DraftBudget {
                generated: proposals,
                verified: proposals,
            },
            adaptive: None,
        })
    }
    /// Choose each round's generated and verified widths from what the
    /// load's rounds measure, up to the proposal width.
    pub fn adaptive(mut self) -> Self {
        self.identity.push_str(":adaptive");
        self.adaptive = Some(Arc::default());
        self
    }
    /// Draft and verify `budget` each round; its generated width must fit
    /// this load.
    pub fn with_budget(mut self, budget: DraftBudget) -> Result<Self, String> {
        budget.validate(self.proposals)?;
        self.identity.push_str(&format!(":{budget:?}"));
        self.budget = budget;
        Ok(self)
    }
}

impl Method for DFlash {
    fn identity(&self) -> &str {
        &self.identity
    }
    fn requires(&self) -> MethodRequirements {
        MethodRequirements {
            prefill_demand: Demand::FEATURES,
            closing_demand: Demand::NONE,
            verify_demand: Demand::FEATURES,
            head: true,
        }
    }
    /// Its history is the fused target features; only the block's anchor is
    /// embedded, and anchors are generated tokens.
    fn embeds_prompt_tokens(&self) -> bool {
        false
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
            self.budget,
            self.adaptive.clone(),
            checkpoint,
        )?))
    }
}
