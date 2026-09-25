//! Proposal operators over domain-owned finite choices. No legality or score model.
use crate::candidate_domain::{CandidateCoordinate, CandidateDomain};
use seismic_lang::expr::compiled::InvocationValues;
use seismic_lang::expr::{PartialAssignment, SymbolValue};
use seismic_native_target::TargetFamily;

#[derive(Clone, Debug)]
pub(super) struct Random(pub u64);
impl Random {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
    pub fn index(&mut self, n: usize) -> usize {
        assert!(n > 0);
        let bound = n as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Operator {
    Mutate,
    Compound,
    Crossover,
    Fresh,
}

pub(super) fn propose<T: TargetFamily>(
    domain: &CandidateDomain<'_, T>,
    parent: Option<&CandidateCoordinate>,
    donor: Option<&CandidateCoordinate>,
    operator: Operator,
    random: &mut Random,
) -> Option<CandidateCoordinate> {
    propose_with_family(domain, parent, donor, operator, random, None)
}

pub(super) fn fresh_in_family<T: TargetFamily>(
    domain: &CandidateDomain<'_, T>,
    ordinal: usize,
    random: &mut Random,
) -> Option<CandidateCoordinate> {
    propose_with_family(domain, None, None, Operator::Fresh, random, Some(ordinal))
}

fn propose_with_family<T: TargetFamily>(
    domain: &CandidateDomain<'_, T>,
    parent: Option<&CandidateCoordinate>,
    donor: Option<&CandidateCoordinate>,
    operator: Operator,
    random: &mut Random,
    family_ordinal: Option<usize>,
) -> Option<CandidateCoordinate> {
    let families = domain.constructed().collect::<Vec<_>>();
    let family = match (operator, parent) {
        (Operator::Fresh, _) | (_, None) => {
            &families[family_ordinal
                .map(|i| i % families.len())
                .unwrap_or_else(|| random.index(families.len()))]
        }
        (_, Some(parent)) => families
            .iter()
            .find(|family| family.identity() == parent.family())?,
    };
    let choices = family.choices();
    let mutable = choices
        .iter()
        .filter(|choice| {
            domain
                .arena()
                .decision_domain(choice.decision())
                .values()
                .len()
                > 1
                && parent.is_none_or(|parent| {
                    parent
                        .decision_choices(choices)
                        .expect("checked parent axes")
                        .iter()
                        .any(|(decision, _)| *decision == choice.decision())
                })
        })
        .map(|choice| choice.decision())
        .collect::<Vec<_>>();
    if mutable.is_empty() && matches!(operator, Operator::Mutate | Operator::Compound) {
        return None;
    }
    let mut edits = Vec::new();
    if !mutable.is_empty() {
        let first = random.index(mutable.len());
        edits.push(mutable[first]);
        if matches!(operator, Operator::Compound) && mutable.len() > 1 {
            let second = (first + 1 + random.index(mutable.len() - 1)) % mutable.len();
            edits.push(mutable[second]);
        }
    }
    let mut fixed = PartialAssignment::new();
    let mut values = Vec::new();
    for choice in choices {
        let active = domain
            .arena()
            .compile_bool_with(choice.active_when(), &fixed);
        if !active.reads().is_empty() {
            return None;
        }
        if !active.evaluate(&InvocationValues::new()).ok()? {
            continue;
        }
        let decision = choice.decision();
        let arena = domain.arena();
        let allowed = arena.decision_domain(decision).values();
        let parent_value = parent
            .filter(|parent| parent.family() == family.identity())
            .and_then(|parent| {
                parent
                    .decision_choices(choices)
                    .expect("checked parent axes")
                    .iter()
                    .find(|(id, _)| *id == decision)
                    .map(|(_, value)| *value)
            });
        let donor_value = donor
            .filter(|donor| donor.family() == family.identity())
            .and_then(|donor| {
                donor
                    .decision_choices(choices)
                    .expect("checked donor axes")
                    .iter()
                    .find(|(id, _)| *id == decision)
                    .map(|(_, value)| *value)
            });
        let value = match operator {
            Operator::Fresh => allowed[random.index(allowed.len())],
            Operator::Crossover if donor_value.is_some() && random.next() & 1 == 0 => {
                donor_value.unwrap()
            }
            Operator::Mutate | Operator::Compound if edits.contains(&decision) => {
                let alternatives = allowed
                    .iter()
                    .copied()
                    .filter(|value| Some(*value) != parent_value)
                    .collect::<Vec<_>>();
                if alternatives.is_empty() {
                    allowed[0]
                } else {
                    alternatives[random.index(alternatives.len())]
                }
            }
            _ => parent_value.unwrap_or_else(|| allowed[random.index(allowed.len())]),
        };
        fixed.bind(
            domain.arena().decision_symbol(decision),
            SymbolValue::Int((value).into()),
        );
        values.push((decision, value));
    }
    let coordinate = domain
        .canonicalize(domain.proposal_for_decisions(family.identity().clone(), values))
        .ok()?;
    domain.check(&coordinate).ok()?;
    Some(coordinate)
}

/// Enumerate a genuinely small sealed domain in declaration dependency order.
/// The resource ceiling bounds traversal as well as output; `None` means the
/// finite set was not established, never that omitted coordinates are illegal.
pub(super) fn finite_coordinates<T: TargetFamily>(
    domain: &CandidateDomain<'_, T>,
    limit: usize,
) -> Option<Vec<CandidateCoordinate>> {
    let mut result = Vec::new();
    let mut visits = 0usize;
    for family in domain.constructed() {
        let choices = family.choices();
        let mut stack = vec![(0usize, Vec::new())];
        while let Some((index, values)) = stack.pop() {
            visits += 1;
            if visits > limit.saturating_mul(16) {
                return None;
            }
            if index == choices.len() {
                if let Ok(coordinate) = domain
                    .canonicalize(domain.proposal_for_decisions(family.identity().clone(), values))
                {
                    if domain.check(&coordinate).is_ok() {
                        if result.len() == limit {
                            return None;
                        }
                        result.push(coordinate);
                    }
                }
                continue;
            }
            let mut fixed = PartialAssignment::new();
            for (decision, value) in &values {
                fixed.bind(
                    domain.arena().decision_symbol(*decision),
                    SymbolValue::Int((*value).into()),
                );
            }
            let choice = &choices[index];
            let active = domain
                .arena()
                .compile_bool_with(choice.active_when(), &fixed);
            if !active.reads().is_empty() {
                return None;
            }
            if !active.evaluate(&InvocationValues::new()).ok()? {
                stack.push((index + 1, values));
                continue;
            }
            for value in domain
                .arena()
                .decision_domain(choice.decision())
                .values()
                .iter()
                .rev()
            {
                let mut child = values.clone();
                child.push((choice.decision(), *value));
                stack.push((index + 1, child));
                if stack.len() > limit.saturating_mul(16) {
                    return None;
                }
            }
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation_session::boundary_tests::domain_with_optional;
    use crate::realization::demand_driven_tests::registry;
    use crate::realization::demand_driven_tests::device;

    #[test]
    fn bounded_enumeration_is_complete_or_explicitly_unknown() {
        let device = device();
        let registry = registry();
        let (domain, optional) = domain_with_optional(&device, &registry);
        assert!(finite_coordinates(&domain, 1).is_none());
        let coordinates = finite_coordinates(&domain, 128).unwrap();
        assert!(coordinates.contains(&optional));
        assert!(coordinates.contains(&domain.canonicalize(domain.universal_proposal()).unwrap()));
        let unique = coordinates.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), coordinates.len());
        let mut random = Random(21);
        for _ in 0..128 {
            assert!(unique
                .contains(&propose(&domain, None, None, Operator::Fresh, &mut random).unwrap()));
        }
    }

    #[test]
    fn choice_free_family_has_no_mutation_and_crossover_respects_family_identity() {
        let device = device();
        let registry = registry();
        let (domain, parent) = domain_with_optional(&device, &registry);
        let mut random = Random(19);
        assert!(parent.choices().is_empty());
        assert!(propose(&domain, Some(&parent), None, Operator::Mutate, &mut random).is_none());
        assert!(propose(&domain, Some(&parent), None, Operator::Compound, &mut random).is_none());
        let general = domain.canonicalize(domain.universal_proposal()).unwrap();
        for _ in 0..32 {
            let child = propose(
                &domain,
                Some(&parent),
                Some(&general),
                Operator::Crossover,
                &mut random,
            )
            .unwrap();
            assert_eq!(child, parent);
            let child = propose(
                &domain,
                Some(&parent),
                Some(&parent),
                Operator::Crossover,
                &mut random,
            )
            .unwrap();
            assert_eq!(child, parent);
            assert!(domain.check(&child).is_ok());
        }
    }
    #[test]
    fn fresh_proposals_cover_families_without_a_parent() {
        let device = device();
        let registry = registry();
        let (domain, _) = domain_with_optional(&device, &registry);
        let mut random = Random(20);
        let mut families = std::collections::HashSet::new();
        for _ in 0..128 {
            let proposal = propose(&domain, None, None, Operator::Fresh, &mut random).unwrap();
            families.insert(proposal.family().clone());
        }
        assert_eq!(families.len(), domain.constructed().count());
    }
}
