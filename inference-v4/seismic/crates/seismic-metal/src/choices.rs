//! Typed original implementation domains. Search and path traversal belong to
//! the shared solver; these values define realization and reconstruction only.
use crate::reduction::Algorithm;
use seismic_accounting::choices::Choices;
use seismic_lang::{ir::LoadMode, normalize::loads};
use seismic_realization::dispatch::TilePlacement;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Transfer(crate::terminal::transfer::Choice),
    Traversal(crate::terminal::traversal::Choice),
    Fold(crate::execution::FoldChoice),
    Load(loads::Choice),
    Storage(crate::storage::StorageDecision),
    Reduction(crate::reduction::Decision),
    Allocation(crate::memory::AllocationChoices),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Alternative {
    Transfer(u8),
    Traversal(usize),
    Fold(crate::execution::FoldOwnership),
    Load(LoadMode),
    Storage(TilePlacement),
    Reduction(Algorithm),
    Allocation(usize),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain {
    pub decision: Decision,
}
impl Choices for Domain {
    type Alternative = Alternative;
    fn len(&self) -> usize {
        match &self.decision {
            Decision::Transfer(choice) => choice.len(),
            Decision::Traversal(choice) => choice.len(),
            Decision::Fold(choice) => choice.len(),
            Decision::Load(choice) => choice.modes().len(),
            Decision::Storage(choice) => choice.alternatives.len(),
            Decision::Reduction(choice) => choice.domain.algorithms().len(),
            Decision::Allocation(choice) => choice.len(),
        }
    }
    fn get(&self, index: usize) -> Option<Alternative> {
        match &self.decision {
            Decision::Transfer(choice) => choice.get(index).map(Alternative::Transfer),
            Decision::Traversal(choice) => choice.get(index).map(Alternative::Traversal),
            Decision::Fold(choice) => choice.get(index).map(Alternative::Fold),
            Decision::Load(choice) => choice.modes().get(index).copied().map(Alternative::Load),
            Decision::Storage(choice) => choice
                .alternatives
                .get(index)
                .cloned()
                .map(Alternative::Storage),
            Decision::Reduction(choice) => choice
                .domain
                .algorithms()
                .get(index)
                .copied()
                .map(Alternative::Reduction),
            Decision::Allocation(choice) => choice.get(index).map(Alternative::Allocation),
        }
    }
}
impl Domain {
    pub fn diagnostic(&self, loads: seismic_realization::LoadStrategy) -> Alternative {
        match &self.decision {
            Decision::Transfer(_) => Alternative::Transfer(1),
            Decision::Traversal(_) => Alternative::Traversal(1),
            Decision::Fold(_) => Alternative::Fold(crate::execution::FoldOwnership::Serial),
            Decision::Load(_) => Alternative::Load(match loads {
                seismic_realization::LoadStrategy::Materialize => LoadMode::Materialize,
                seismic_realization::LoadStrategy::BorrowProvenReadOnly => LoadMode::Borrow,
            }),
            Decision::Storage(choice) => Alternative::Storage(choice.diagnostic()),
            Decision::Reduction(choice) => Alternative::Reduction(choice.diagnostic()),
            Decision::Allocation(choice) => Alternative::Allocation(choice.new_slot),
        }
    }
    pub fn index(&self, alternative: &Alternative) -> Option<usize> {
        if let (Decision::Traversal(choice), Alternative::Traversal(width)) =
            (&self.decision, alternative)
        {
            return width.checked_sub(1).filter(|&index| index < choice.len());
        }
        (0..self.len()).find(|&index| self.get(index).as_ref() == Some(alternative))
    }
}
