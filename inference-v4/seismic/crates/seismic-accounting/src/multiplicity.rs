//! Evaluate structural execution counts with representation-specific constant facts.
//! The same rules serve scalar SSA and structured selected IR.
use crate::quantity::Count;
use seismic_realization::execution::Multiplicity;
use std::{collections::HashMap, sync::Arc};

/// The caller keeps all nodes alive for the lifetime of this pointer-keyed cache.
pub fn evaluate<V>(
    node: &Arc<Multiplicity<V>>,
    cache: &mut HashMap<usize, Count>,
    constant: &mut impl FnMut(&V) -> Option<i64>,
) -> Count {
    let key = Arc::as_ptr(node) as usize;
    if let Some(count) = cache.get(&key) {
        return count.clone();
    }
    let count = match &**node {
        Multiplicity::Constant(n) => Count::Exact(*n),
        Multiplicity::Unknown { reason } => Count::unknown(reason.clone()),
        Multiplicity::Product(a, b) => {
            evaluate(a, cache, constant).multiply(&evaluate(b, cache, constant))
        }
        Multiplicity::PlusOne(a) => evaluate(a, cache, constant).add(&Count::Exact(1)),
        Multiplicity::Iterations { lower, upper } => match (constant(lower), constant(upper)) {
            (Some(lo), Some(hi)) => Count::iterations(lo, hi),
            _ => Count::unknown("runtime or dependent execution extent"),
        },
        Multiplicity::Predicate { value, expected } => match constant(value) {
            Some(value) => Count::Exact(u64::from((value != 0) == *expected)),
            None => Count::interval(0, 1).unwrap(),
        },
    };
    cache.insert(key, count.clone());
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranges_predicates_and_unknowns_compose_without_enumerating_iterations() {
        let range = Arc::new(Multiplicity::Iterations {
            lower: Some(-3),
            upper: Some(4),
        });
        let repeated =
            Multiplicity::product(range, Arc::new(Multiplicity::Constant(1_000_000_000)));
        let conditional = Multiplicity::product(
            repeated,
            Arc::new(Multiplicity::Predicate {
                value: None,
                expected: true,
            }),
        );
        assert_eq!(
            evaluate(&conditional, &mut HashMap::new(), &mut |v| *v),
            Count::interval(0, 7_000_000_000).unwrap()
        );
        let unresolved: Arc<Multiplicity<Option<i64>>> = Arc::new(Multiplicity::Unknown {
            reason: "unresolved ownership".into(),
        });
        let skipped = Multiplicity::product(unresolved, Arc::new(Multiplicity::Constant(0)));
        assert_eq!(
            evaluate(&skipped, &mut HashMap::new(), &mut |v| *v),
            Count::Exact(0)
        );
        let full = Arc::new(Multiplicity::Iterations {
            lower: Some(i64::MIN),
            upper: Some(i64::MAX),
        });
        assert_eq!(
            evaluate(&full, &mut HashMap::new(), &mut |v| *v),
            Count::Exact(u64::MAX)
        );
    }
}
