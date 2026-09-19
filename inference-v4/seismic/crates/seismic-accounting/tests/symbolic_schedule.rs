use magnitude_solver::{
    Limits, Options, Outcome, Search,
    model::{Constraint, Cost, Domain, LinearTerm, ModelBuilder},
    scheduling::{Activity, Demand},
};
use seismic_accounting::schedule::{CapacityUnit, Resource, symbolic::Encoding};
use std::sync::Arc;

#[test]
fn conditional_children_keep_shared_capacity_and_global_ranking() {
    let mut b = ModelBuilder::new();
    let short = b.variable("short private alternative", Domain::boolean());
    let long = b.variable("long private alternative", Domain::boolean());
    b.constraint(Constraint::ExactlyOne {
        variables: vec![short, long],
    });
    let resource = Resource {
        name: "joint storage".into(),
        capacity: 2,
        unit: CapacityUnit::Bytes,
    };
    let mut schedule = Encoding::new(&mut b, "joint", &[resource], 5).unwrap();
    // A one-tick isolated winner consumes all storage. A two-tick alternative
    // leaves room for its two-tick neighbor, and is therefore the global winner.
    for (name, duration, units, presence) in [
        ("short", 1, 2, Some(short)),
        ("long", 2, 1, Some(long)),
        ("neighbor", 2, 1, None),
    ] {
        let activity = Activity {
            start: b.variable(format!("{name}.start"), Domain::interval(0, 5).unwrap()),
            duration: b.variable(format!("{name}.duration"), Domain::singleton(duration)),
            end: b.variable(format!("{name}.end"), Domain::interval(0, 5).unwrap()),
            presence,
        };
        schedule.activity(&mut b, activity.clone());
        schedule
            .whole_activity(0, activity, Demand::Constant(units))
            .unwrap();
    }
    let completion = schedule.finish(&mut b).unwrap();
    b.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    let mut search = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    let Outcome::Optimal(result) = search
        .advance(Limits {
            work: 100_000,
            ..Limits::default()
        })
        .unwrap()
    else {
        panic!("small conditional relation must complete")
    };
    assert_eq!(result.cost(), 2);
    assert_eq!(result.values()[long.0], 1);
    assert_eq!(result.values()[short.0], 0);
}

#[test]
fn absent_activity_does_not_extend_completion_or_consume_capacity() {
    let mut b = ModelBuilder::new();
    let mut schedule = Encoding::new(
        &mut b,
        "absent",
        &[Resource {
            name: "slots".into(),
            capacity: 1,
            unit: CapacityUnit::Slots,
        }],
        10,
    )
    .unwrap();
    let absent = b.variable("presence", Domain::singleton(0));
    let activity = Activity {
        start: b.variable("unused start", Domain::singleton(3)),
        duration: b.variable("unused duration", Domain::singleton(2)),
        end: b.variable("unused end", Domain::singleton(10)),
        presence: Some(absent),
    };
    schedule.activity(&mut b, activity.clone());
    schedule
        .whole_activity(0, activity, Demand::Constant(100))
        .unwrap();
    let completion = schedule.finish(&mut b).unwrap();
    b.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    let mut search = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    let Outcome::Optimal(result) = search.advance(Limits::default()).unwrap() else {
        panic!("empty active family must complete")
    };
    assert_eq!(result.cost(), 0);
}

#[test]
fn symbolic_service_keeps_offsets_demands_and_presence_coupled() {
    use seismic_accounting::algebra::{Algebra, ResourceUse, Symbolic};
    for present in [false, true] {
        let mut builder = ModelBuilder::new();
        let presence = builder.variable("present", Domain::singleton(i64::from(present)));
        let mut schedule = Encoding::new(
            &mut builder,
            "service",
            &[Resource {
                name: "issue".into(),
                capacity: 1,
                unit: CapacityUnit::Slots,
            }],
            8,
        )
        .unwrap();
        let (duration, units) = {
            let mut algebra = Symbolic::new(&mut builder, "counts");
            (algebra.constant(2).unwrap(), algebra.constant(1).unwrap())
        };
        for index in 0..2 {
            let operation = Activity {
                start: builder
                    .variable(format!("op{index}.start"), Domain::interval(0, 4).unwrap()),
                duration: builder.variable(format!("op{index}.duration"), Domain::singleton(4)),
                end: builder.variable(format!("op{index}.end"), Domain::interval(0, 8).unwrap()),
                presence: Some(presence),
            };
            schedule.activity(&mut builder, operation.clone());
            schedule
                .service(
                    &mut builder,
                    &operation,
                    &[ResourceUse {
                        resource: 0,
                        offset: 1,
                        duration,
                        units,
                    }],
                )
                .unwrap();
        }
        let completion = schedule.finish(&mut builder).unwrap();
        builder.cost(Cost::Linear {
            constant: 0,
            terms: vec![LinearTerm::new(completion, 1)],
        });
        let mut search =
            Search::new(Arc::new(builder.build().unwrap()), Options::default()).unwrap();
        let Outcome::Optimal(solution) = search
            .advance(Limits {
                work: 100_000,
                ..Limits::default()
            })
            .unwrap()
        else {
            panic!("coupled service must complete")
        };
        // Partial issue services overlap operation latency: [1,3) and [3,5)
        // give completion 6, rather than serial operation completion 8.
        assert_eq!(solution.cost(), if present { 6 } else { 0 });
    }
}
