//! Dependent storage/reduction domains for a fixed execution decomposition.
//! Preparation suspends at the first unresolved decision. No default performance policy,
//! target-text inspection or native compilation participates in this traversal.
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
        if let (Decision::Traversal(choice), Alternative::Traversal(width)) = (&self.decision, alternative) {
            return width.checked_sub(1).filter(|&index| index < choice.len());
        }
        (0..self.len()).find(|&index| self.get(index).as_ref() == Some(alternative))
    }
}
pub enum Expansion {
    Choice(ExecutionChoice),
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
    if config.max_threadgroup_bytes < 0 {
        return Err("negative shared-memory capacity".into());
    }
    let mut unbounded = config.clone();
    unbounded.max_threadgroup_bytes = i64::MAX;
    let stage = crate::execution::prepare_initial(function, unbounded, mappings)?;
    finish(std::sync::Arc::new(stage), config, 0, prefix)
}

/// The retained owning stage and only that stage's resolved decision prefix.
#[derive(Clone)]
pub struct ExecutionChoice {
    stage: std::sync::Arc<crate::execution::Stage>,
    config: Config,
    base: usize,
    prefix: Vec<usize>,
    domain: Domain,
}
impl PartialEq for ExecutionChoice {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.base == other.base
            && self.prefix == other.prefix
            && self.domain == other.domain
            && (std::sync::Arc::ptr_eq(&self.stage, &other.stage) || self.stage == other.stage)
    }
}
impl std::fmt::Debug for ExecutionChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionChoice")
            .field("decision", &self.domain.decision)
            .field("base", &self.base)
            .field("prefix", &self.prefix)
            .finish()
    }
}
impl Choices for ExecutionChoice {
    type Alternative = Alternative;
    fn len(&self) -> usize {
        self.domain.len()
    }
    fn get(&self, index: usize) -> Option<Alternative> {
        self.domain.get(index)
    }
}
impl ExecutionChoice {
    pub fn diagnostic(&self, loads: seismic_realization::LoadStrategy) -> Alternative {
        self.domain.diagnostic(loads)
    }
    pub fn index(&self, alternative: &Alternative) -> Option<usize> {
        self.domain.index(alternative)
    }
    pub fn decision(&self) -> &Decision {
        &self.domain.decision
    }
    pub fn prepared(&self) -> &LoweredIr {
        self.stage.function()
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub(crate) fn relax(
        &self,
        hardware: &crate::model::Hardware,
        workload: &seismic_accounting::workload::ScalarWorkload,
        limits: seismic_accounting::workload::DerivationLimits,
    ) -> Result<Option<seismic_accounting::schedule::Demand>, String> {
        let maximum = u64::try_from(self.config.max_threads_per_threadgroup)
            .map_err(|_| "negative Metal thread capacity")?;
        let launches = self
            .stage
            .phases()
            .iter()
            .flat_map(|phase| std::iter::once(&phase.dispatch).chain(phase.merge_dispatch.as_ref()))
            .map(|dispatch| (dispatch.work_items, maximum / dispatch.lanes_per_item));
        let dispatch = crate::model::dispatch_demand(hardware, launches)?;
        let Some(account) = self.stage.refinement_account(workload, limits)? else { return Ok(dispatch); };
        let mut demand = match dispatch {
            Some(demand) => demand,
            None => seismic_accounting::schedule::Demand::new(hardware.timebase.clone(), hardware.resources.clone())?,
        };
        crate::model::include_preserved_demand(&mut demand, &account, hardware, |primitive| self.stage.preserves(primitive))?;
        Ok(Some(demand))
    }
    pub fn refine(&self, index: usize) -> Result<Expansion, String> {
        if self.get(index).is_none() {
            return Err("Metal execution choice is outside its derived domain".into());
        }
        let mut prefix = self.prefix.clone();
        prefix.push(index);
        finish(self.stage.clone(), self.config.clone(), self.base, &prefix)
    }
}

fn finish(
    mut stage: std::sync::Arc<crate::execution::Stage>,
    config: Config,
    mut base: usize,
    mut prefix: &[usize],
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
            // A forced implementation is not a search branch. Its owning plan
            // still validates and records the only legal implementation.
            if domain.len() == 1 {
                return Ok(domain.get(0).unwrap());
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
    loop {
        let replay = RefCell::new(Replay {
            prefix,
            consumed: 0,
            pending: None,
        });
        let result = crate::execution::advance(
            &stage,
            &mut |decision| match replay.borrow_mut().select(Domain {
                decision: Decision::Fold(decision.clone()),
            })? {
                Alternative::Fold(value) => Ok(value),
                _ => unreachable!(),
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
                    Alternative::Load(value) => Ok(value),
                    _ => unreachable!(),
                }
            },
            &mut |decision| match replay.borrow_mut().select(Domain {
                decision: Decision::Storage(decision.clone()),
            })? {
                Alternative::Storage(value) => Ok(value),
                _ => unreachable!(),
            },
            &mut |decision| match replay.borrow_mut().select(Domain {
                decision: Decision::Reduction(decision.clone()),
            })? {
                Alternative::Reduction(value) => Ok(value),
                _ => unreachable!(),
            },
            &mut |decision| {
                if decision.len() == 1 {
                    return Ok(decision.new_slot);
                }
                match replay.borrow_mut().select(Domain {
                    decision: Decision::Allocation(decision.clone()),
                })? {
                    Alternative::Allocation(value) => Ok(value),
                    _ => unreachable!(),
                }
            },
            &mut |choice| match replay.borrow_mut().select(Domain { decision: Decision::Transfer(choice.clone()) })? {
                Alternative::Transfer(width) => Ok(width), _ => unreachable!(),
            },
            &mut |choice| match replay.borrow_mut().select(Domain {
                decision: Decision::Traversal(choice.clone()),
            })? {
                Alternative::Traversal(width) => Ok(width),
                _ => unreachable!(),
            },
        );
        let replay = replay.into_inner();
        if let Some(domain) = replay.pending {
            return Ok(Expansion::Choice(ExecutionChoice {
                stage,
                config,
                base,
                prefix: prefix[..replay.consumed].to_vec(),
                domain,
            }));
        }
        base += replay.consumed;
        prefix = &prefix[replay.consumed..];
        match result? {
            crate::execution::Advance::Stage(next) => stage = std::sync::Arc::new(next),
            crate::execution::Advance::Execution(mut execution) => {
                let available = u64::try_from(config.max_threadgroup_bytes)
                    .map_err(|_| "negative shared-memory capacity")?;
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
                return Ok(Expansion::Execution {
                    execution,
                    consumed: base,
                });
            }
        }
    }
}
