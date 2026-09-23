#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapabilityPlan {
    pub(super) max_draft_proposals: Option<u8>,
    pub(super) vision: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedMethod {
    Plain,
    Mtp {
        greedy_proposals: u8,
        sampled_proposals: u8,
    },
}

impl CapabilityPlan {
    pub fn max_draft_proposals(&self) -> Option<u8> {
        self.max_draft_proposals
    }

    pub fn supports_vision(&self) -> bool {
        self.vision
    }
}
