use magnitude_solver::model::{Cost, Domain, LinearTerm, ModelBuilder};
use magnitude_solver::{Limits, Options, Outcome, Search};
use seismic_accounting::{
    authority::ModelRelationship,
    schedule::{self, evaluation, export, structured, symbolic::Encoding},
};
use std::sync::Arc;

fn resources() -> Vec<schedule::Resource> {
    vec![schedule::Resource {
        name: "shared issue".into(),
        capacity: 1,
        unit: schedule::CapacityUnit::Slots,
    }]
}
fn operation(name: &str) -> schedule::Operation {
    schedule::Operation {
        name: name.into(),
        latency: 2,
        predecessors: vec![],
        start_predecessors: vec![],
        reservations: vec![schedule::Reservation {
            resource: 0,
            offset: 1,
            duration: 1,
            units: 1,
        }],
    }
}
fn flat(name: &str) -> evaluation::Model {
    schedule::Model {
        relationship: ModelRelationship::hypothetical_execution(),
        identity: name.into(),
        timebase: schedule::Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: resources(),
        operations: vec![operation(name)],
        lifetimes: vec![],
        static_orders: vec![],
        unmapped: vec![],
    }
    .into()
}
fn repeated(count: u64, expansion_limit: u64) -> evaluation::Model {
    evaluation::Model::Structured {
        expansion_limit,
        model: structured::Structured {
            relationship: ModelRelationship::hypothetical_execution(),
            identity: "repeated original occurrences".into(),
            timebase: schedule::Timebase {
                seconds_numerator: 1,
                seconds_denominator: 1,
            },
            resources: resources(),
            unmapped: vec![],
            root: Arc::new(structured::Node::Repeat {
                order: structured::Order::Parallel,
                count,
                body: Arc::new(structured::Node::Operation(operation("repeated"))),
            }),
        },
    }
}

#[test]
fn flat_and_repeated_fragments_compete_in_one_model_and_keep_occurrences() {
    let mut builder = ModelBuilder::new();
    let mut encoding = Encoding::new(&mut builder, "joint", &resources(), 6).unwrap();
    let explicit = export::Binding::append(
        &mut builder,
        &mut encoding,
        Arc::new(flat("explicit")),
        None,
    )
    .unwrap();
    let repeats =
        export::Binding::append(&mut builder, &mut encoding, Arc::new(repeated(2, 2)), None)
            .unwrap();
    let completion = encoding.finish(&mut builder).unwrap();
    builder.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    let model = Arc::new(builder.build().unwrap());
    let mut search = Search::new(model.clone(), Options::default()).unwrap();
    let Outcome::Optimal(solution) = search
        .advance(Limits {
            work: 100_000,
            ..Limits::default()
        })
        .unwrap()
    else {
        panic!("tiny shared model should complete")
    };
    model.validate_assignment(solution.values()).unwrap();
    // Three offset issue reservations cannot all start at time one. No private
    // fragment optimum or identical-copy schedule can satisfy this joint model.
    assert_eq!(solution.cost(), 4);
    let explicit = explicit.reconstruct(solution.values(), 0).unwrap().unwrap();
    let repeated = repeats.reconstruct(solution.values(), 0).unwrap().unwrap();
    assert_eq!(
        explicit.cost().upper().max(repeated.cost().upper()),
        solution.cost()
    );
    assert!(!repeated.cost().is_exact());
    let (original, schedule) = repeated.structured().unwrap().expand(2).unwrap();
    original.check_execution_upper(&schedule).unwrap();
    assert_ne!(schedule.starts[0], schedule.starts[1]);
}

#[test]
fn missing_compact_relation_retains_structure_and_conditional_obligation() {
    let original = Arc::new(repeated(1_000_000, 2));
    assert_eq!(export::horizon(&original).unwrap(), 2_000_000);
    let mut builder = ModelBuilder::new();
    let presence = builder.variable("active", Domain::singleton(0));
    let mut encoding = Encoding::new(&mut builder, "guarded", &resources(), 2_000_000).unwrap();
    let binding = export::Binding::append(
        &mut builder,
        &mut encoding,
        original.clone(),
        Some(presence),
    )
    .unwrap();
    assert!(Arc::ptr_eq(binding.original(), &original));
    assert!(binding.fragment().is_none());
    assert_eq!(binding.obligations().len(), 1);
    let completion = encoding.finish(&mut builder).unwrap();
    builder.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    let mut search = Search::new(Arc::new(builder.build().unwrap()), Options::default()).unwrap();
    let Outcome::Optimal(solution) = search
        .advance(Limits {
            work: 1000,
            ..Limits::default()
        })
        .unwrap()
    else {
        panic!("inactive missing relation must not block other regions")
    };
    assert_eq!(solution.cost(), 0);
    assert!(binding.reconstruct(solution.values(), 0).unwrap().is_none());
}

#[test]
fn reconstructed_objective_rejects_inconsistent_completion_and_global_bound() {
    let evaluation::Model::Flat(model) = flat("checked original") else {
        unreachable!()
    };
    let model = Arc::new(model);
    let valid = schedule::Schedule {
        starts: vec![0],
        completion: 2,
    };
    assert!(
        seismic_accounting::objective::Objective::from_flat(model.clone(), valid.clone(), 3)
            .is_err()
    );
    assert!(
        seismic_accounting::objective::Objective::from_flat(
            model.clone(),
            schedule::Schedule {
                starts: vec![0],
                completion: 1
            },
            0
        )
        .is_err()
    );
    let objective = seismic_accounting::objective::Objective::from_flat(model, valid, 0).unwrap();
    assert_eq!((objective.cost().lower(), objective.cost().upper()), (0, 2));
}
