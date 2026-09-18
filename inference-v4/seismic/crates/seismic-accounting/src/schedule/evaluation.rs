//! One selection boundary for flat and structured execution constraints.
//! A representation never changes the completion requirement or supplies an
//! executable fallback when its scheduling frontier remains unresolved.
use super::{structured, *};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Model {
    Flat(super::Model),
    Structured { model: structured::Structured, expansion_limit: u64 },
}
impl From<super::Model> for Model { fn from(model: super::Model) -> Self { Self::Flat(model) } }
impl Model {
    /// An index accelerator, never semantic identity. Hash collisions and
    /// omitted fields are resolved by full constraint equality before reuse.
    pub(crate) fn sharing_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        fn operation(operation: &Operation, hash: &mut impl Hasher) {
            (
                &operation.name,
                operation.latency,
                &operation.predecessors,
                &operation.start_predecessors,
            )
                .hash(hash);
            operation.reservations.len().hash(hash);
            for r in &operation.reservations {
                (r.resource, r.offset, r.duration, r.units).hash(hash);
            }
        }
        fn node(value: &structured::Node, hash: &mut impl Hasher) {
            match value {
                structured::Node::Operation(op) => {
                    0u8.hash(hash);
                    operation(op, hash);
                }
                structured::Node::Compose { order, children } => {
                    (1u8, *order == structured::Order::Serial, children.len()).hash(hash);
                    for child in children {
                        node(child, hash);
                    }
                }
                structured::Node::Repeat { order, count, body } => {
                    (2u8, *order == structured::Order::Serial, count).hash(hash);
                    node(body, hash);
                }
                structured::Node::Scope { reservations, body } => {
                    (3u8, reservations).hash(hash);
                    node(body, hash);
                }
            }
        }
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        let (identity, timebase, resources, unmapped) = match self {
            Self::Flat(model) => {
                0u8.hash(&mut hash);
                model.operations.len().hash(&mut hash);
                for op in &model.operations {
                    operation(op, &mut hash);
                }
                (
                    &model.identity,
                    &model.timebase,
                    &model.resources,
                    &model.unmapped,
                )
            }
            Self::Structured {
                model,
                expansion_limit,
            } => {
                (1u8, expansion_limit).hash(&mut hash);
                node(&model.root, &mut hash);
                (
                    &model.identity,
                    &model.timebase,
                    &model.resources,
                    &model.unmapped,
                )
            }
        };
        (
            identity,
            timebase.seconds_numerator,
            timebase.seconds_denominator,
            unmapped,
        )
            .hash(&mut hash);
        for resource in resources {
            (&resource.name, resource.capacity).hash(&mut hash);
        }
        hash.finish()
    }
    pub(crate) fn lower_bound(&self) -> Result<u64, String> {
        match self {
            Self::Flat(model) => model.lower_bound(),
            Self::Structured { model, .. } => model.lower_bound(),
        }
    }
    pub fn into_flat(self) -> Result<super::Model, String> {
        match self { Self::Flat(model) => Ok(model), Self::Structured { .. } => Err("execution model retains structured work".into()) }
    }
    pub fn timebase(&self) -> &Timebase { match self { Self::Flat(m) => &m.timebase, Self::Structured { model, .. } => &model.timebase } }
    pub fn relationship(&self) -> &crate::authority::ModelRelationship { match self { Self::Flat(m) => &m.relationship, Self::Structured { model, .. } => &model.relationship } }
    pub(crate) fn start_search(self) -> Result<Search, String> {
        match self {
            Self::Flat(model) => Ok(Search::Flat { search: model.start_search()?, minimum: 0 }),
            Self::Structured { model, expansion_limit } => Ok(Search::Structured(
                structured::Refinement::new(model, expansion_limit)?)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Solution {
    Flat { solution: super::Solution, minimum: u64 },
    Structured(structured::Witness),
}
impl Solution {
    pub fn completion(&self) -> u64 { match self { Self::Flat { solution, .. } => solution.schedule().completion, Self::Structured(w) => w.completion() } }
    pub fn lower_bound(&self) -> u64 { match self { Self::Flat { solution, minimum } => solution.lower_bound().max(*minimum), Self::Structured(w) => w.lower_bound() } }
    pub fn is_optimal(&self) -> bool { self.lower_bound() == self.completion() }
    pub fn flat(&self) -> Option<(&super::Model, &Schedule)> { match self { Self::Flat { solution, .. } => Some((solution.model(), solution.schedule())), _ => None } }
    pub fn structured(&self) -> Option<&structured::Witness> { match self { Self::Structured(w) => Some(w), _ => None } }
    pub fn check_execution_upper(&self) -> Result<(), String> {
        match self {
            Self::Flat { solution, minimum } => {
                solution.model().check_execution_upper(solution.schedule())?;
                if *minimum > solution.schedule().completion { return Err("structured lower bound exceeds feasible completion".into()); }
            }
            Self::Structured(witness) => {
                witness.check_execution_upper()?;
            }
        }
        Ok(())
    }
}
pub(crate) enum Outcome { Feasible(Solution), Incomplete { lower_bound: u64 }, Infeasible }
pub(crate) enum Search {
    Flat { search: super::Search, minimum: u64 },
    Structured(structured::Refinement),
}
impl Search {
    /// Equality of every retained scheduling constraint, including mapping
    /// gaps and the model relationship. Selection may share this search only
    /// after deriving the same model from each execution under one request.
    pub(crate) fn matches_model(&self, model: &Model) -> bool {
        match (self, model) {
            (Self::Flat { search, .. }, Model::Flat(model)) => search.model() == model,
            (Self::Structured(search), Model::Structured { model, .. }) => search.model() == model,
            _ => false,
        }
    }
    pub fn unmapped(&self) -> &[String] { match self { Self::Flat { search, .. } => &search.model().unmapped, Self::Structured(search) => &search.model().unmapped } }
    pub fn advance(&mut self, budget: u64) -> Result<Outcome, String> {
        match self {
            Self::Flat { search, minimum } => Ok(match search.advance(budget)? {
                super::SearchOutcome::Feasible(solution) => Outcome::Feasible(Solution::Flat { solution, minimum: *minimum }),
                super::SearchOutcome::Incomplete { lower_bound } => Outcome::Incomplete { lower_bound: lower_bound.max(*minimum) },
                super::SearchOutcome::Infeasible => Outcome::Infeasible,
            }),
            Self::Structured(search) => Ok(match search.advance(budget)? {
                structured::RefinementOutcome::Feasible(witness) => Outcome::Feasible(Solution::Structured(witness)),
                structured::RefinementOutcome::Incomplete { lower_bound } => Outcome::Incomplete { lower_bound },
                structured::RefinementOutcome::Infeasible => Outcome::Infeasible,
            }),
        }
    }
}
