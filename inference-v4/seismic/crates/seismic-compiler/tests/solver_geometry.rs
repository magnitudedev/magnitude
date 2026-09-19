use magnitude_solver::{
    model::{Constraint, Domain, LinearTerm, ModelBuilder},
    Limits, Options, Outcome, Search,
};
use seismic_compiler::tuner::geometry::{Error, Geometry};
use seismic_lang::types::DType;
use seismic_realization::dispatch::{TileDeclaration, TilePlacement};
use std::sync::Arc;

fn tile(placement: TilePlacement, capacity: u64) -> TileDeclaration {
    TileDeclaration {
        symbol: "tile".into(),
        dtype: DType::F32,
        capacity,
        placement,
    }
}

#[test]
fn complete_small_family_corresponds_to_concrete_geometry() {
    // Enumerate the ORIGINAL choice domain, including empty work and a scalar
    // mapping. No solver-selected optimum is used to define expected coverage.
    for extents in [vec![3, 2], vec![0, 3], vec![]] {
        let step_domains: Vec<_> = extents
            .iter()
            .map(|e| Domain::interval(1, (*e).max(1) as i64).unwrap())
            .collect();
        let steps: Vec<Vec<i64>> = if extents.is_empty() {
            vec![vec![]]
        } else {
            step_domains[0]
                .values()
                .flat_map(|a| step_domains[1].values().map(move |b| vec![a, b]))
                .collect()
        };
        for selected_steps in steps {
            for lanes in 1..=2 {
                for items in 1..=3 {
                    let mut b = ModelBuilder::new();
                    let g = Geometry::append(
                        &mut b,
                        "launch",
                        &extents,
                        &step_domains,
                        Domain::interval(1, 2).unwrap(),
                        Domain::interval(1, 3).unwrap(),
                        &[
                            tile(TilePlacement::Replicated, 0),
                            tile(TilePlacement::Distributed, 5),
                            tile(TilePlacement::GroupShared, 3),
                        ],
                    )
                    .unwrap();
                    for (&v, &value) in g
                        .steps
                        .iter()
                        .zip(&selected_steps)
                        .chain([(&g.lanes_per_item, &lanes), (&g.items_per_group, &items)])
                    {
                        b.constraint(Constraint::InDomain {
                            variable: v.id(),
                            domain: Domain::singleton(value),
                        });
                    }
                    let model = Arc::new(b.build().unwrap());
                    let mut search = Search::new(model.clone(), Options::default()).unwrap();
                    let Outcome::Optimal(solution) = search
                        .advance(Limits {
                            work: 100_000,
                            time: None,
                            memory_bytes: None,
                        })
                        .unwrap()
                    else {
                        panic!("every concrete choice must have a symbolic witness")
                    };
                    let concrete = g.reconstruct(solution.values()).unwrap();
                    let counts: Vec<_> = extents
                        .iter()
                        .zip(&selected_steps)
                        .map(|(&n, &s)| n.div_ceil(s as u64))
                        .collect();
                    assert_eq!(concrete.mapping.work_items(), counts.iter().product());
                    for item in 0..concrete.mapping.work_items() {
                        let coords = concrete.mapping.coordinates(item).unwrap();
                        let tails = concrete.mapping.extents(item).unwrap();
                        for axis in 0..extents.len() {
                            assert_eq!(
                                tails[axis],
                                (selected_steps[axis] as u64).min(extents[axis] - coords[axis])
                            );
                        }
                    }
                    assert_eq!(
                        concrete.dispatch.groups,
                        concrete.mapping.work_items().div_ceil(items as u64)
                    );
                    assert_eq!(
                        concrete.storage[1].private_bytes_per_lane,
                        5_u64.div_ceil(lanes as u64) * 4
                    );
                    assert_eq!(
                        concrete.storage[2].shared_bytes_per_group,
                        12 * items as u64
                    );
                    // Reverse check: a wrong derived quantity must not acquire a
                    // solver witness or pass the original realization check.
                    let mut wrong = solution.values().to_vec();
                    wrong[g.groups.id().0] += 1;
                    assert!(model
                        .validate_assignment(&wrong)
                        .map_or(true, |a| a.infeasible));
                    assert!(g.reconstruct(&wrong).is_err());
                }
            }
        }
    }
}

#[test]
fn storage_limit_couples_unresolved_group_width() {
    let mut b = ModelBuilder::new();
    let g = Geometry::append(
        &mut b,
        "launch",
        &[7],
        &[Domain::interval(1, 7).unwrap()],
        Domain::singleton(2),
        Domain::interval(1, 4).unwrap(),
        &[tile(TilePlacement::GroupShared, 3)],
    )
    .unwrap();
    b.constraint(Constraint::LinearLe {
        terms: vec![LinearTerm::new(g.storage[0].shared_bytes_per_group.id(), 1)],
        rhs: 24,
    });
    b.constraint(Constraint::InDomain {
        variable: g.items_per_group.id(),
        domain: Domain::interval(3, 4).unwrap(),
    });
    let mut search = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    assert!(matches!(
        search
            .advance(Limits {
                work: 100_000,
                time: None,
                memory_bytes: None
            })
            .unwrap(),
        Outcome::Infeasible
    ));
}

#[test]
fn construction_size_tracks_axes_not_candidate_count() {
    let build = |extent| {
        let mut b = ModelBuilder::new();
        Geometry::append(
            &mut b,
            "launch",
            &[extent],
            &[Domain::interval(1, extent as i64).unwrap()],
            Domain::singleton(32),
            Domain::interval(1, 32).unwrap(),
            &[],
        )
        .unwrap();
        b.build().unwrap()
    };
    let small = build(3);
    let large = build(1_000_000);
    assert_eq!(small.variables().len(), large.variables().len());
    assert_eq!(small.factors().len(), large.factors().len());
}

#[test]
fn unrepresentable_family_is_not_reported_infeasible() {
    let mut b = ModelBuilder::new();
    assert!(matches!(
        Geometry::append(
            &mut b,
            "launch",
            &[u64::MAX],
            &[Domain::singleton(1)],
            Domain::singleton(1),
            Domain::singleton(1),
            &[]
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn concrete_geometry_keeps_checked_arithmetic_and_empty_domains() {
    use seismic_realization::dispatch::{GroupDispatch, WorkMapping};
    assert!(WorkMapping::new(&[1], &[0]).is_err());
    assert!(WorkMapping::new(&[u64::MAX, 2], &[1, 1]).is_err());
    let empty = WorkMapping::new(&[u64::MAX, 0, 2], &[1, 1, 1]).unwrap();
    assert_eq!(empty.work_items(), 0);
    assert!(empty.axes().iter().all(|axis| axis.stride == 0));
    assert!(GroupDispatch::new(1, u64::MAX, 2).is_err());
    assert!(GroupDispatch::new(u64::MAX, 1, 2).is_err());
    assert!(tile(TilePlacement::GroupShared, u64::MAX)
        .layout(&GroupDispatch::new(1, 1, 1).unwrap())
        .is_err());
}
