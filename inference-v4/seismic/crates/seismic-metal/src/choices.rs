//! Dependent storage/reduction domains for a fixed execution decomposition.
//! Preparation suspends at the first unresolved decision. No default policy,
//! emitted source or native compilation participates in this traversal.
use crate::{
    execution::{Config, Execution},
    reduction::{Algorithm, Site},
};
use seismic_lang::{ir::VarId, lowered_ir::LoweredIr};
use seismic_realization::dispatch::TilePlacement;
use std::cell::RefCell;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Storage(VarId),
    Reduction(Site),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Alternative {
    Storage(TilePlacement),
    Reduction(Algorithm),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain {
    pub decision: Decision,
    pub alternatives: Vec<Alternative>,
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
    struct Replay<'a> {
        prefix: &'a [usize],
        consumed: usize,
        pending: Option<Domain>,
    }
    impl Replay<'_> {
        fn select(&mut self, domain: Domain) -> Result<Alternative, String> {
            if domain.alternatives.is_empty() {
                return Err(format!(
                    "empty legal execution domain at {:?}",
                    domain.decision
                ));
            }
            let Some(index) = self.prefix.get(self.consumed) else {
                self.pending = Some(domain);
                return Err("execution preparation suspended at an unresolved decision".into());
            };
            let selected =
                domain.alternatives.get(*index).cloned().ok_or_else(|| {
                    format!("choice index {index} is outside {:?}", domain.decision)
                })?;
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
    let result = crate::execution::prepare_selected(
        function,
        unbounded_storage,
        &mut |decision| match replay.borrow_mut().select(Domain {
            decision: Decision::Storage(decision.variable),
            alternatives: decision
                .alternatives
                .iter()
                .cloned()
                .map(Alternative::Storage)
                .collect(),
        })? {
            Alternative::Storage(placement) => Ok(placement),
            Alternative::Reduction(_) => unreachable!("storage domain contains only placements"),
        },
        &mut |decision| match replay.borrow_mut().select(Domain {
            decision: Decision::Reduction(decision.site),
            alternatives: decision
                .domain
                .algorithms()
                .iter()
                .copied()
                .map(Alternative::Reduction)
                .collect(),
        })? {
            Alternative::Reduction(algorithm) => Ok(algorithm),
            Alternative::Storage(_) => unreachable!("reduction domain contains only algorithms"),
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
