//! Exact selection over a structurally defined execution space. The space owns
//! legality and derives each leaf's execution and objective together. Traversal
//! order has no performance meaning, and every legal leaf participates.
//!
//! This module proves selection under that contract. It does not prove that a
//! backend's execution space or hardware model faithfully describes its target.
use std::collections::BTreeSet;

/// All alternatives use the same workload, target, execution form and objective.
/// Identities must describe immutable inputs, rather than device display names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Context {
    pub program: String,
    pub workload: String,
    pub target: String,
    pub contracts: String,
    pub execution_form: String,
    pub objective: String,
    /// Exact common time unit for the latency objective.
    pub seconds_numerator: u64,
    pub seconds_denominator: u64,
}
impl Context {
    fn validate(&self) -> Result<(), String> {
        if [
            &self.program,
            &self.workload,
            &self.target,
            &self.contracts,
            &self.execution_form,
            &self.objective,
        ]
        .iter()
        .any(|s| s.is_empty())
            || self.seconds_numerator == 0
            || self.seconds_denominator == 0
        {
            return Err("selection requires identified inputs and a positive time unit".into());
        }
        Ok(())
    }
}

/// A model-derived interval in the context's exact integer time units. Bounds
/// describe the same execution/workload conditions across every alternative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    lower: u64,
    upper: u64,
}
impl Cost {
    pub fn exact(ticks: u64) -> Self {
        Self {
            lower: ticks,
            upper: ticks,
        }
    }
    pub fn bounded(lower: u64, upper: u64) -> Result<Self, String> {
        if lower > upper {
            return Err("reversed execution cost bounds".into());
        }
        Ok(Self { lower, upper })
    }
    pub fn lower(self) -> u64 {
        self.lower
    }
    pub fn upper(self) -> u64 {
        self.upper
    }
}

/// A leaf cannot contain an execution without its derived modeled cost.
pub struct Realization<E> {
    pub execution: E,
    pub cost: Cost,
}
/// A violated physical capacity in the declared execution form. Verification
/// rederives this constraint from the original program and hardware inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityViolation {
    pub resource: String,
    pub required: u64,
    pub available: u64,
}
impl CapacityViolation {
    fn validate(&self) -> Result<(), String> {
        if self.resource.is_empty() || self.required <= self.available {
            return Err("infeasible branch does not contain a violated capacity".into());
        }
        Ok(())
    }
}
pub enum Node<E> {
    /// Complete legal alternatives at this point, including dependent choices.
    Choice {
        name: String,
        alternatives: Vec<String>,
    },
    Realization(Realization<E>),
    Infeasible(CapacityViolation),
}
/// Expand a decision prefix using IR and hardware contracts only. Each child is
/// addressed by its index in the freshly derived parent domain. Compiler failures
/// are errors, never evidence that an otherwise legal alternative can be omitted.
pub trait Space {
    type Execution;
    fn context(&self) -> &Context;
    fn expand(&self, prefix: &[usize]) -> Result<Node<Self::Execution>, String>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Evidence {
    Choice {
        name: String,
        alternatives: Vec<String>,
    },
    Realization(Cost),
    Infeasible(CapacityViolation),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub path: Vec<usize>,
    pub evidence: Evidence,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Certificate {
    pub context: Context,
    /// Complete preorder tree. Verification regenerates each domain and cost.
    pub records: Vec<Record>,
    pub selected: Vec<usize>,
}
/// The selected execution itself is retained, not reconstructed from settings at
/// native compilation. Construction is private to successful complete selection.
pub struct Selected<E> {
    execution: E,
    certificate: Certificate,
    cost: Cost,
}
impl<E> Selected<E> {
    pub fn execution(&self) -> &E {
        &self.execution
    }
    pub fn certificate(&self) -> &Certificate {
        &self.certificate
    }
    pub fn cost(&self) -> Cost {
        self.cost
    }
    pub fn into_execution(self) -> E {
        self.execution
    }
}

fn validate_choice(name: &str, alternatives: &[String]) -> Result<(), String> {
    let distinct: BTreeSet<_> = alternatives.iter().collect();
    if name.is_empty()
        || alternatives.is_empty()
        || distinct.len() != alternatives.len()
        || alternatives.iter().any(String::is_empty)
    {
        return Err(
            "execution choice requires a name and distinct nonempty legal alternatives".into(),
        );
    }
    Ok(())
}

/// A traversal budget interrupts work; it never redefines the legal domain or
/// produces a partially covered result. No target code or native feedback enters.
pub fn select<S: Space>(space: &S, node_budget: usize) -> Result<Selected<S::Execution>, String> {
    space.context().validate()?;
    let mut pending = vec![Vec::new()];
    let mut records = Vec::new();
    let mut best: Option<(Vec<usize>, Realization<S::Execution>)> = None;
    while let Some(path) = pending.pop() {
        if records.len() == node_budget {
            return Err(
                "selection budget exhausted before complete execution-space coverage".into(),
            );
        }
        match space.expand(&path)? {
            Node::Choice { name, alternatives } => {
                validate_choice(&name, &alternatives)?;
                for i in (0..alternatives.len()).rev() {
                    let mut child = path.clone();
                    child.push(i);
                    pending.push(child);
                }
                records.push(Record {
                    path,
                    evidence: Evidence::Choice { name, alternatives },
                });
            }
            Node::Infeasible(violation) => {
                violation.validate()?;
                records.push(Record {
                    path,
                    evidence: Evidence::Infeasible(violation),
                });
            }
            Node::Realization(realization) => {
                records.push(Record {
                    path: path.clone(),
                    evidence: Evidence::Realization(realization.cost),
                });
                if best.as_ref().is_none_or(|(_, old)| {
                    (realization.cost.upper, realization.cost.lower)
                        < (old.cost.upper, old.cost.lower)
                }) {
                    best = Some((path, realization));
                }
            }
        }
    }
    let (path, realization) = best.ok_or("execution space has no realization")?;
    for record in &records {
        if let Evidence::Realization(cost) = record.evidence {
            if record.path != path && realization.cost.upper > cost.lower {
                return Err("modeled intervals do not establish an optimal realization".into());
            }
        }
    }
    Ok(Selected {
        execution: realization.execution,
        cost: realization.cost,
        certificate: Certificate {
            context: space.context().clone(),
            records,
            selected: path,
        },
    })
}

/// Independent certificate replay: a stack of expected children establishes
/// coverage; neither claimed leaf count nor the optimizer's preferred order is
/// accepted as evidence. Costs are rederived from the original space.
pub fn verify<S: Space>(
    space: &S,
    certificate: &Certificate,
    node_budget: usize,
) -> Result<Cost, String> {
    space.context().validate()?;
    if space.context() != &certificate.context {
        return Err("selection assumptions changed".into());
    }
    let mut expected = vec![Vec::new()];
    let mut selected = None;
    let mut competitor_floor = u64::MAX;
    for (visited, record) in certificate.records.iter().enumerate() {
        if visited == node_budget {
            return Err("verification budget exhausted".into());
        }
        let path = expected.pop().ok_or("certificate contains extra nodes")?;
        if path != record.path {
            return Err("certificate omitted or reordered a legal branch".into());
        }
        match (space.expand(&path)?, &record.evidence) {
            (
                Node::Choice { name, alternatives },
                Evidence::Choice {
                    name: saved_name,
                    alternatives: saved,
                },
            ) => {
                validate_choice(&name, &alternatives)?;
                if name != *saved_name || alternatives != *saved {
                    return Err("legal choice domain changed".into());
                }
                for i in (0..alternatives.len()).rev() {
                    let mut child = path.clone();
                    child.push(i);
                    expected.push(child);
                }
            }
            (Node::Infeasible(violation), Evidence::Infeasible(saved)) => {
                violation.validate()?;
                if violation != *saved {
                    return Err("derived capacity violation changed".into());
                }
            }
            (Node::Realization(realization), Evidence::Realization(saved)) => {
                if realization.cost != *saved {
                    return Err("derived execution cost changed".into());
                }
                if path == certificate.selected {
                    selected = Some(realization.cost);
                } else {
                    competitor_floor = competitor_floor.min(realization.cost.lower);
                }
            }
            _ => return Err("certificate node does not match the execution space".into()),
        }
    }
    if !expected.is_empty() {
        return Err("certificate has incomplete coverage".into());
    }
    let cost = selected.ok_or("selected path is not a realization")?;
    if cost.upper > competitor_floor {
        return Err("certificate does not exclude a better realization".into());
    }
    Ok(cost)
}
