use magnitude_solver::{
    model::{Arithmetic, Constraint, Cost, LinearTerm, Literal},
    Algorithm, Domain, Limits, ModelBuilder, NeighborhoodOptions, Options, Outcome, Search,
};
use std::sync::Arc;

fn options(seed: u64) -> Options {
    Options {
        algorithm: Algorithm::Neighborhood(NeighborhoodOptions {
            seed,
            repair_work: 97,
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn tiny_generated_models_preserve_every_claim_against_direct_enumeration() {
    for seed in 0..20_u64 {
        let mut b = ModelBuilder::new();
        let v: Vec<_> = (0..4)
            .map(|i| b.variable(format!("v{i}"), Domain::interval(0, 2).unwrap()))
            .collect();
        for i in 0..3 {
            b.constraint(Constraint::NotEqual {
                left: v[i],
                right: v[i + 1],
            });
            b.cost(Cost::Table {
                variables: vec![v[i], v[i + 1]],
                entries: (0..3)
                    .flat_map(|x| {
                        (0..3).map(move |y| {
                            (
                                vec![x, y],
                                (seed + x as u64 * 17 + y as u64 * 11 + i as u64 * 13) % 29,
                            )
                        })
                    })
                    .collect(),
            });
        }
        let model = Arc::new(b.build().unwrap());
        let optimum = (0..81)
            .filter_map(|mut a| {
                let x: Vec<_> = (0..4)
                    .map(|_| {
                        let x = a % 3;
                        a /= 3;
                        x
                    })
                    .collect();
                if x.windows(2).any(|w| w[0] == w[1]) {
                    return None;
                }
                Some(
                    (0..3)
                        .map(|i| {
                            (seed + x[i] as u64 * 17 + x[i + 1] as u64 * 11 + i as u64 * 13) % 29
                        })
                        .sum::<u64>(),
                )
            })
            .min()
            .unwrap();
        let mut search = Search::new(model.clone(), options(seed)).unwrap();
        let mut upper = u64::MAX;
        let mut lower = 0;
        for _ in 0..100 {
            match search
                .advance(Limits {
                    work: 29,
                    ..Default::default()
                })
                .unwrap()
            {
                Outcome::Optimal(w) => {
                    assert_eq!(w.cost(), optimum);
                    break;
                }
                Outcome::Infeasible => panic!("generated problem is feasible"),
                Outcome::Incomplete(p) => {
                    assert!(lower <= p.lower_bound && p.lower_bound <= optimum);
                    lower = p.lower_bound;
                    if let Some(w) = p.incumbent {
                        assert!(optimum <= w.cost() && w.cost() <= upper);
                        upper = w.cost();
                        let x = w.values();
                        assert!(x.windows(2).all(|w| w[0] != w[1]));
                        let cost = (0..3)
                            .map(|i| {
                                (seed + x[i] as u64 * 17 + x[i + 1] as u64 * 11 + i as u64 * 13)
                                    % 29
                            })
                            .sum::<u64>();
                        assert_eq!(cost, w.cost());
                    }
                }
            }
        }
    }
}

#[test]
fn conditional_arithmetic_and_private_values_remain_consistent() {
    let mut b = ModelBuilder::new();
    let mode = b.variable("mode", Domain::boolean());
    let layout = b.variable("layout", Domain::boolean());
    let tile = b.variable("tile", Domain::interval(1, 4).unwrap());
    let factor = b.variable("factor", Domain::singleton(3));
    let bytes = b.variable("bytes", Domain::interval(3, 12).unwrap());
    b.constraint(Constraint::Arithmetic(Arithmetic::Product {
        left: tile,
        right: factor,
        product: bytes,
    }));
    b.constraint(Constraint::InactiveValue {
        active: Literal::new(mode, 1),
        variable: layout,
        inactive: 0,
    });
    b.guarded_constraint(
        vec![Literal::new(mode, 1)],
        Constraint::InDomain {
            variable: layout,
            domain: Domain::singleton(1),
        },
    );
    b.constraint(Constraint::LinearLe {
        terms: vec![LinearTerm::new(bytes, 1), LinearTerm::new(mode, 2)],
        rhs: 12,
    });
    b.guarded_cost(vec![Literal::new(mode, 0)], Cost::Constant(20));
    b.guarded_cost(
        vec![Literal::new(mode, 1)],
        Cost::Linear {
            constant: 15,
            terms: vec![LinearTerm::new(tile, -3)],
        },
    );
    let model = Arc::new(b.build().unwrap());
    for seed in 0..8 {
        let mut search = Search::new(model.clone(), options(seed)).unwrap();
        let result = search
            .advance(Limits {
                work: 5000,
                ..Default::default()
            })
            .unwrap();
        let w = match result {
            Outcome::Optimal(w) => w.feasible().clone(),
            Outcome::Incomplete(p) => p.incumbent.expect("small feasible instance"),
            _ => panic!(),
        };
        let v = w.values();
        assert_eq!(v[bytes.0], 3 * v[tile.0]);
        assert_eq!(v[layout.0], v[mode.0]);
        assert!(v[bytes.0] + 2 * v[mode.0] <= 12);
        assert!(w.cost() >= 6);
    }
}

#[test]
fn unknown_coverage_never_certifies_missing_realizations() {
    let mut b = ModelBuilder::new();
    let v = b.variable("mode", Domain::boolean());
    b.guarded_cost(vec![Literal::new(v, 0)], Cost::Constant(10));
    b.unresolved(vec![Literal::new(v, 1)], "unimplemented region");
    let mut s = Search::new(Arc::new(b.build().unwrap()), options(0)).unwrap();
    for _ in 0..5 {
        let Outcome::Incomplete(p) = s
            .advance(Limits {
                work: 500,
                ..Default::default()
            })
            .unwrap()
        else {
            panic!("missing coverage cannot be proved optimal");
        };
        if let Some(w) = p.incumbent {
            assert_eq!(w.values()[0], 0);
            assert_eq!(w.cost(), 10);
        }
        assert_eq!(p.lower_bound, 0);
    }
}

#[test]
fn initialization_can_prove_global_infeasibility_and_errors_stick() {
    let mut b = ModelBuilder::new();
    let v = b.variable("v", Domain::boolean());
    b.constraint(Constraint::NotEqual { left: v, right: v });
    let mut s = Search::new(Arc::new(b.build().unwrap()), options(0)).unwrap();
    assert!(matches!(
        s.advance(Limits::default()).unwrap(),
        Outcome::Infeasible
    ));
    let mut b = ModelBuilder::new();
    b.cost(Cost::Constant(u64::MAX));
    b.cost(Cost::Constant(1));
    let mut s = Search::new(Arc::new(b.build().unwrap()), options(0)).unwrap();
    let first = s.advance(Limits::default()).unwrap_err().to_string();
    assert!(first.contains("overflow"));
    assert_eq!(s.advance(Limits::default()).unwrap_err().to_string(), first);
}

#[test]
fn options_are_validated_and_exact_remains_default() {
    assert!(matches!(Options::default().algorithm, Algorithm::Exact));
    for which in 0..4 {
        let mut c = NeighborhoodOptions::default();
        match which {
            0 => c.max_neighborhood_variables = 0,
            1 => c.repair_work = 0,
            2 => c.population_size = 0,
            _ => c.restart_after = 0,
        }
        assert!(Search::new(
            Arc::new(ModelBuilder::new().build().unwrap()),
            Options {
                algorithm: Algorithm::Neighborhood(c),
                ..Default::default()
            }
        )
        .is_err());
    }
}

#[cfg(feature = "serde")]
#[test]
fn serialized_algorithm_options_round_trip() {
    let opts = options(998);
    let json = serde_json::to_string(&opts).unwrap();
    let decoded: Options = serde_json::from_str(&json).unwrap();
    assert_eq!(json, serde_json::to_string(&decoded).unwrap());
}
