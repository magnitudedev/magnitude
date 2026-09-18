//! Dependent storage/reduction domains for a fixed execution decomposition.
//! Preparation suspends at the first unresolved decision. No default policy,
//! emitted source or native compilation participates in this traversal.
use crate::{
    execution::{Config, Execution},
    reduction::Algorithm,
};
use seismic_accounting::selection::Choices;
use seismic_lang::{ir::LoadMode, lowered_ir::LoweredIr, normalize::loads};
use seismic_realization::dispatch::TilePlacement;
use std::cell::RefCell;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Fold(crate::execution::FoldChoice),
    Load(loads::Choice),
    Storage(crate::storage::StorageDecision),
    Reduction(crate::reduction::Decision),
    Allocation(crate::memory::AllocationChoices),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Alternative {
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
            Decision::Fold(choice) => choice.len(),
            Decision::Load(choice) => choice.modes().len(),
            Decision::Storage(choice) => choice.alternatives.len(),
            Decision::Reduction(choice) => choice.domain.algorithms().len(),
            Decision::Allocation(choice) => choice.len(),
        }
    }
    fn get(&self, index: usize) -> Option<Alternative> {
        match &self.decision {
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
        (0..self.len()).find(|&index| self.get(index).as_ref() == Some(alternative))
    }
}
pub enum Expansion {
    Choice(Domain),
    Execution {
        execution: Execution,
        consumed: usize,
    },
    Infeasible {
        launch: usize,
        required: u64,
        available: u64,
    },
}

/// The prefix selects indices in freshly derived domains. Storage choices can
/// change later reduction domains; those domains are not precomputed separately.
pub fn expand(function: &LoweredIr, config: Config, prefix: &[usize]) -> Result<Expansion, String> {
    expand_with_mappings(function, config, None, prefix)
}
pub fn expand_with_mappings(
    function: &LoweredIr,
    config: Config,
    mappings: Option<&[seismic_realization::dispatch::WorkMapping]>,
    prefix: &[usize],
) -> Result<Expansion, String> {
    struct Replay<'a> {
        prefix: &'a [usize],
        consumed: usize,
        pending: Option<Domain>,
    }
    impl Replay<'_> {
        fn select(&mut self, domain: Domain) -> Result<Alternative, String> {
            if domain.len() == 0 {
                return Err(format!(
                    "empty legal execution domain at {:?}",
                    domain.decision
                ));
            }
            let Some(index) = self.prefix.get(self.consumed) else {
                self.pending = Some(domain);
                return Err("execution preparation suspended at an unresolved decision".into());
            };
            let selected = domain
                .get(*index)
                .ok_or_else(|| format!("choice index {index} is outside {:?}", domain.decision))?;
            self.consumed += 1;
            Ok(selected)
        }
    }
    let replay = RefCell::new(Replay {
        prefix,
        consumed: 0,
        pending: None,
    });
    let available = u64::try_from(config.max_threadgroup_bytes)
        .map_err(|_| "negative shared-memory capacity")?;
    let mut unbounded_storage = config.clone();
    // Derive actual arrays first. Capacity exclusion below is explicit evidence,
    // rather than turning a compiler error string into permission to prune.
    unbounded_storage.max_threadgroup_bytes = i64::MAX;
    let result = crate::execution::prepare_with_participants(
        function,
        unbounded_storage,
        mappings,
        &mut |decision| match replay.borrow_mut().select(Domain {
            decision: Decision::Fold(decision.clone()),
        })? {
            Alternative::Fold(ownership) => Ok(ownership),
            _ => unreachable!("fold domain contains only ownerships"),
        },
        &mut |site, load| {
            if !load.can_borrow {
                return Ok(LoadMode::Materialize);
            }
            match replay.borrow_mut().select(Domain {
                decision: Decision::Load(loads::Choice {
                    site,
                    variable: load.variable,
                }),
            })? {
                Alternative::Load(mode) => Ok(mode),
                _ => unreachable!("load domain contains only modes"),
            }
        },
        &mut |decision| match replay.borrow_mut().select(Domain {
            decision: Decision::Storage(decision.clone()),
        })? {
            Alternative::Storage(placement) => Ok(placement),
            _ => unreachable!("storage domain contains only placements"),
        },
        &mut |decision| match replay.borrow_mut().select(Domain {
            decision: Decision::Reduction(decision.clone()),
        })? {
            Alternative::Reduction(algorithm) => Ok(algorithm),
            _ => unreachable!("reduction domain contains only algorithms"),
        },
        &mut |decision| {
            if decision.len() == 1 {
                return Ok(decision.new_slot);
            }
            match replay.borrow_mut().select(Domain {
                decision: Decision::Allocation(decision.clone()),
            })? {
                Alternative::Allocation(slot) => Ok(slot),
                _ => unreachable!("allocation domain contains only backing slots"),
            }
        },
    );
    let replay = replay.into_inner();
    if let Some(domain) = replay.pending {
        return Ok(Expansion::Choice(domain));
    }
    let mut execution = result?;
    execution.config = config;
    for (launch, memory) in execution.memory().launches().iter().enumerate() {
        if memory.shared_bytes_per_group > available {
            return Ok(Expansion::Infeasible {
                launch,
                required: memory.shared_bytes_per_group,
                available,
            });
        }
    }
    Ok(Expansion::Execution {
        execution,
        consumed: replay.consumed,
    })
}
