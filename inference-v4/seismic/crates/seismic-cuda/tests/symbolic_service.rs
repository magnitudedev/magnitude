use magnitude_solver::{
    Limits, Options, Outcome, Search,
    model::{Cost, Domain, ModelBuilder},
};
use seismic_accounting::algebra::{Concrete, Symbolic};
use seismic_cuda::{
    model::{Amount, PrimitiveTiming, Quantity, Reservation, Ticks},
    ptx::Primitive,
};
use std::sync::Arc;

fn timing() -> PrimitiveTiming {
    // Reuse a real primitive kind; geometry and service are an explicit
    // hypothetical hardware contract, independent of kernel scores.
    PrimitiveTiming {
        primitive: Primitive::Return,
        latency: Ticks::Service {
            demand: Amount {
                quantity: Quantity::ActiveLanes,
                scale: 3,
            },
            per_tick: 4,
            base: 2,
        },
        reservations: vec![Reservation {
            resource: 0,
            offset: 1,
            duration: Ticks::Service {
                demand: Amount {
                    quantity: Quantity::ActiveLanes,
                    scale: 2,
                },
                per_tick: 4,
                base: 1,
            },
            units: Amount {
                quantity: Quantity::ActiveLanes,
                scale: 2,
            },
        }],
    }
}

#[test]
fn symbolic_terminal_service_matches_concrete_for_empty_and_partial_cohorts() {
    for lanes in 0..=8u64 {
        let timing = timing();
        let (concrete_latency, concrete_uses) = timing
            .account(&mut Concrete::<String>::default(), |_, q| {
                assert_eq!(q, Quantity::ActiveLanes);
                Ok(lanes)
            })
            .unwrap();
        assert_eq!(concrete_latency, (lanes * 3).div_ceil(4) + 2);
        let mut builder = ModelBuilder::new();
        let (latency, uses) = {
            let mut algebra = Symbolic::new(&mut builder, "PTX service");
            let lanes = algebra
                .variable("lanes", Domain::singleton(lanes as i64))
                .unwrap();
            timing
                .account(&mut algebra, |_, q| {
                    assert_eq!(q, Quantity::ActiveLanes);
                    Ok(lanes)
                })
                .unwrap()
        };
        builder.cost(Cost::Constant(0));
        let mut search =
            Search::new(Arc::new(builder.build().unwrap()), Options::default()).unwrap();
        let Outcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
            panic!("fixed service equations must complete")
        };
        assert_eq!(solution.values()[latency.id().0] as u64, concrete_latency);
        assert_eq!(
            solution.values()[uses[0].duration.id().0] as u64,
            concrete_uses[0].duration
        );
        assert_eq!(solution.values()[uses[0].units.id().0] as u64, lanes * 2);
        assert_eq!(uses[0].offset, concrete_uses[0].offset);
    }
}

#[test]
fn unresolved_service_size_does_not_enumerate_geometry_members() {
    let sizes: Vec<_> = [8, 1_000_000]
        .into_iter()
        .map(|maximum| {
            let mut builder = ModelBuilder::new();
            let mut algebra = Symbolic::new(&mut builder, "PTX service");
            let lanes = algebra
                .variable("lanes", Domain::interval(0, maximum).unwrap())
                .unwrap();
            timing().account(&mut algebra, |_, _| Ok(lanes)).unwrap();
            let model = builder.build().unwrap();
            (model.variables().len(), model.factors().len())
        })
        .collect();
    assert_eq!(sizes[0], sizes[1]);
}
