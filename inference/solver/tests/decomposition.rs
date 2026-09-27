use magnitude_solver::{
    model::{Constraint, Cost, ModelBuilder},
    Domain, Limits, Options, Outcome, Search,
};
use std::sync::Arc;
#[test]
fn independent_problems_do_not_enumerate_the_cartesian_product() {
    let mut b = ModelBuilder::new();
    for i in 0..100 {
        let x = b.variable(format!("x{i}"), Domain::interval(0, 7).unwrap());
        b.cost(Cost::Table {
            variables: vec![x],
            entries: (0..8).map(|v| (vec![v], (7 - v) as u64)).collect(),
        });
    }
    let mut s = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    assert!(
        matches!(s.advance(Limits{work:100_000,..Default::default()}).unwrap(),Outcome::Optimal(ref x) if x.cost()==0)
    );
    assert!(s.stats().nodes < 2000, "{:?}", s.stats());
    assert!(s.stats().decompositions > 0);
}
#[test]
fn separator_contexts_are_shared_without_prefix_costs() {
    let mut b = ModelBuilder::new();
    let n = 18;
    let v: Vec<_> = (0..n)
        .map(|i| b.variable(format!("v{i}"), Domain::boolean()))
        .collect();
    for (i, &x) in v.iter().enumerate() {
        b.cost(Cost::Table {
            variables: vec![x],
            entries: vec![(vec![0], (i % 3) as u64), (vec![1], ((i + 1) % 3) as u64)],
        });
    }
    for w in v.windows(2) {
        b.constraint(Constraint::NotEqual {
            left: w[0],
            right: w[1],
        });
    }
    let mut s = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    assert!(matches!(
        s.advance(Limits {
            work: 100_000,
            ..Default::default()
        })
        .unwrap(),
        Outcome::Optimal(_)
    ));
    assert!(s.stats().nodes < 1000, "{:?}", s.stats());
}

#[test]
fn pairwise_chain_matches_dynamic_program_and_reuses_contexts() {
    let (n, d) = (10, 3);
    let mut b = ModelBuilder::new();
    let vars: Vec<_> = (0..n)
        .map(|i| b.variable(format!("stage{i}"), Domain::interval(0, d - 1).unwrap()))
        .collect();
    let mut optimum = vec![0_u64; d as usize];
    for i in 0..n {
        let local: Vec<_> = (0..d)
            .map(|v| ((i as i64 * 13 + v * 5 + 3) % 17) as u64)
            .collect();
        b.cost(Cost::Table {
            variables: vec![vars[i]],
            entries: (0..d).map(|v| (vec![v], local[v as usize])).collect(),
        });
        if i == 0 {
            optimum = local;
            continue;
        }
        let transition = |a: i64, z: i64| ((a * 7 + z * 11 + i as i64 * 3) % 13) as u64;
        b.cost(Cost::Table {
            variables: vec![vars[i - 1], vars[i]],
            entries: (0..d)
                .flat_map(|a| (0..d).map(move |z| (vec![a, z], transition(a, z))))
                .collect(),
        });
        optimum = (0..d)
            .map(|z| {
                local[z as usize]
                    + (0..d)
                        .map(|a| optimum[a as usize] + transition(a, z))
                        .min()
                        .unwrap()
            })
            .collect();
    }
    let expected = *optimum.iter().min().unwrap();
    let model = Arc::new(b.build().unwrap());
    let mut work = Vec::new();
    for memoize in [true, false] {
        let mut search = Search::new(
            model.clone(),
            Options {
                memoize,
                ..Default::default()
            },
        )
        .unwrap();
        let Outcome::Optimal(solution) = search
            .advance(Limits {
                work: 1_000_000,
                ..Default::default()
            })
            .unwrap()
        else {
            panic!("chain should complete")
        };
        assert_eq!(solution.cost(), expected);
        if memoize {
            assert!(search.stats().memo_hits > 0, "{:?}", search.stats());
        }
        work.push(search.stats().work);
    }
    assert!(work[0] < work[1], "cached and uncached work: {work:?}");
}
