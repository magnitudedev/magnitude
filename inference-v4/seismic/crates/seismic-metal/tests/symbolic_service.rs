use magnitude_solver::{
    Limits, Options, Outcome, Search,
    model::{Cost, Domain, ModelBuilder},
};
use seismic_accounting::algebra::{Concrete, Symbolic};
use seismic_metal::{
    model::{Service, Timing, Units},
    terminal::Primitive,
};
use std::sync::Arc;

#[test]
fn shared_msl_service_equations_preserve_lane_subgroup_and_transaction_units() {
    let timing = Timing {
        primitive: Primitive::Return,
        latency: 3,
        services: vec![
            Service {
                resource: 0,
                offset: 0,
                duration: 1,
                units: Units::PerLane(2),
            },
            Service {
                resource: 1,
                offset: 1,
                duration: 2,
                units: Units::PerSubgroup(3),
            },
            Service {
                resource: 2,
                offset: 0,
                duration: 3,
                units: Units::PerTransaction {
                    bytes: 32,
                    units: 4,
                },
            },
        ],
    };
    for lanes in 0..=8u64 {
        for transactions in 0..=3u64 {
            let (_, concrete) = timing
                .account(&mut Concrete::<String>::default(), lanes, |_, bytes| {
                    assert_eq!(bytes, 32);
                    Ok(transactions)
                })
                .unwrap();
            let mut builder = ModelBuilder::new();
            let (latency, symbolic) = {
                let mut algebra = Symbolic::new(&mut builder, "MSL service");
                let lanes = algebra
                    .variable("lanes", Domain::singleton(lanes as i64))
                    .unwrap();
                let transactions = algebra
                    .variable("transactions", Domain::singleton(transactions as i64))
                    .unwrap();
                timing
                    .account(&mut algebra, lanes, |_, _| Ok(transactions))
                    .unwrap()
            };
            builder.cost(Cost::Constant(0));
            let mut search =
                Search::new(Arc::new(builder.build().unwrap()), Options::default()).unwrap();
            let Outcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
                panic!("fixed Metal service equations must complete")
            };
            assert_eq!(solution.values()[latency.id().0], 3);
            for ((symbolic, concrete), expected) in
                symbolic
                    .iter()
                    .zip(concrete)
                    .zip([lanes * 2, 3, transactions * 4])
            {
                assert_eq!(concrete.units, expected);
                assert_eq!(solution.values()[symbolic.units.id().0] as u64, expected);
                assert_eq!(
                    solution.values()[symbolic.duration.id().0] as u64,
                    concrete.duration
                );
            }
        }
    }
}
