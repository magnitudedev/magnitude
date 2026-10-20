use super::cases::DenseOutputTuning;
use super::*;
use crate::native::import::ImportKernels;
use crate::planning::tests::{fixture_definition, fixture_manifest};
use crate::{ComponentSelection, ModelLoadPlan};
use magnitude_kernels::{dense_output, shape_rows};
use seismic::TuningMethod;
use seismic::{BackendName, ConfigurationRecord, Outcome};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const LIMITS: TuningLimits = TuningLimits {
    max_rows: 512,
    max_projected_rows: 8,
    context_tokens: 16384,
};

/// What a demand walk collects for `case` alone, decoding without drafting:
/// the target's launches as the case represents them, and the objective
/// with the case's estimate of each term's time.
fn fixture_invocations<T: EntryTuning>(
    case: &T,
    statics: &NativeSpecialization,
    limits: TuningLimits,
) -> super::super::specialization::InvocationDemand {
    let terms = workload::terms(limits, &[]);
    let mut demand = super::super::specialization::InvocationDemand {
        units: HashMap::new(),
        objective: Objective::new(terms),
    };
    fixture_add(&mut demand, case, statics, limits);
    demand
}

/// Add what a demand walk collects for `case` to `demand`.
fn fixture_add<T: EntryTuning>(
    demand: &mut super::super::specialization::InvocationDemand,
    case: &T,
    statics: &NativeSpecialization,
    limits: TuningLimits,
) {
    let graph = crate::programs::invocation_demand::GraphInvocationDemand::new(
        crate::PlannedMethod::Plain,
        demand.objective.terms.clone(),
        None,
        limits.max_rows,
    );
    let mut representatives = case
        .points(limits)
        .into_iter()
        .map(|point| point.rows)
        .collect::<Vec<_>>();
    representatives.sort_unstable();
    representatives.dedup();
    let calls = graph
        .target
        .represented(|launch| case.invocation_rows(launch, &representatives))
        .scaled(case.launches());
    let mut points = case.points(limits);
    apply_invocation_counts(&mut points, &calls, &demand.objective);
    // The fixtures weigh each term by its points' relative cost: a load
    // estimates them from its entries' declarations ([`point_bytes`]).
    for point in &points {
        let time = point.cost();
        for (term, launches) in &point.terms {
            demand.objective.measured[*term] += launches * time;
        }
    }
    demand
        .units
        .entry((
            <T::Entry as seismic::Entry>::NAME,
            case.bindings(),
            statics.statics().clone(),
        ))
        .or_default()
        .add(&calls);
}

/// A unit's share is the mean over the objective's terms of its fraction
/// of each term's time: a unit that is a small part of the longest term and
/// most of a short one is worth as much as that short term.
#[test]
fn a_units_share_is_its_mean_fraction_of_every_terms_time() {
    // Two equal terms: a long prefill and a short decode. The projection
    // takes 90 of the prefill's 100 and 2 of the decode's 4; attention the
    // rest.
    let weights = [0.5, 0.5];
    let measured = [vec![90., 2.], vec![10., 2.]];
    let shares = shares(&measured, &weights);
    assert!(
        (shares[0] - (0.5 * 0.9 + 0.5 * 0.5)).abs() < 1e-12,
        "{shares:?}"
    );
    assert!(
        (shares[1] - (0.5 * 0.1 + 0.5 * 0.5)).abs() < 1e-12,
        "{shares:?}"
    );
}

/// The load's launches reach a unit's points: by rows and the sequences
/// declared at or below theirs, a launch's history divided between the two
/// declared histories around it and held to the longest (a window), each
/// point weighing its terms' launches by the term's weight over its time.
#[test]
fn launches_reach_the_points_at_their_rows_sequences_and_histories() {
    use crate::programs::invocation_demand::{InvocationCounts, Launch};
    let mut objective = Objective::new(workload::terms(LIMITS, &[]));
    objective.measured = vec![1.; objective.terms.len()];
    let mut counts = InvocationCounts::default();
    let launch = |term, rows, sequences, history| Launch {
        term,
        rows,
        sequences,
        history,
    };
    counts.launch(launch(0, 8, 8, 1024), 2.);
    counts.launch(launch(1, 8, 1, 4096), 1.);
    counts.launch(launch(2, 1, 1, 100_000), 1.);
    let windowed = |points: Vec<PointShape>| with_contexts(LIMITS, Some(4096), points);
    let mut points = windowed(vec![
        row_point(1, 1.),
        row_point(8, 1.),
        across_sequences(&row_point(8, 1.), 8),
    ]);
    apply_invocation_counts(&mut points, &counts, &objective);
    let at = |label: &str| {
        points
            .iter()
            .find(|point| point.label == label)
            .unwrap_or_else(|| panic!("{label}"))
            .terms
            .clone()
    };
    // 1024 lies between the empty history and 4096: a quarter of the way.
    let quarter = 1024. / 4096.;
    assert!(
        at("m8s8-c256").is_empty(),
        "numerical-only history has no timing demand"
    );
    assert_eq!(at("m8s8-c0"), [(0, 2. * (1. - quarter))]);
    assert_eq!(at("m8s8-c4096"), [(0, 2. * quarter)]);
    assert_eq!(at("m8-c4096"), [(1, 1.)]);
    // A history past the window is read at the window.
    assert_eq!(at("m1-c4096"), [(2, 1.)]);
    let weight = objective.terms[1].weight;
    let point = points
        .iter()
        .find(|point| point.label == "m8-c4096")
        .unwrap();
    assert!((point.weight - weight).abs() < 1e-12);
    // The census discovers the weighed points and the empty history of
    // their rows.
    let census = points
        .iter()
        .zip(census_points(&points))
        .filter(|(_, census)| *census)
        .map(|(point, _)| point.label.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        census,
        [
            "m1-c0",
            "m1-c4096",
            "m8-c0",
            "m8-c4096",
            "m8s8-c0",
            "m8s8-c4096"
        ]
    );
}

/// The requirement of the test units: `alpha` rounding steps of `bf16`.
fn steps(alpha: f64) -> seismic::NumericalRequirement {
    seismic::NumericalRequirement::RoundingSteps {
        alpha,
        activation: Some(DType::BF16),
    }
}

/// A candidate's measurement at `point`, made under the key `variant`: its
/// one `bf16` result is `ratio` rounding steps from the default's.
fn observation(point: &str, variant: u64, ratio: f64) -> seismic::NumericalObservation {
    seismic::NumericalObservation {
        point: point.into(),
        key: seismic::LaunchKey {
            launches: vec![0],
            values: BTreeMap::from([("ROWS".into(), variant)]),
        },
        rotations: vec![vec![seismic::SubjectMeasurement {
            subject: "r0".into(),
            comparison: seismic::SubjectComparison::Differs(seismic::Difference {
                stored: Some(DType::BF16),
                elements: 64,
                differing: 64,
                exact_mismatches: 0,
                non_finite_mismatches: 0,
                error_power: ratio * ratio,
                reference_power: 64.,
                maximum_absolute_error: ratio,
                step_power: vec![seismic::StepPower {
                    dtype: DType::BF16,
                    power: 1.,
                }],
            }),
        }]],
    }
}

#[test]
fn demanded_cache_requires_selected_native_default_evidence_for_every_case() {
    let shapes = [row_point(1, 1.), row_point(4, 0.)];
    let default = Configuration {
        statics: BTreeMap::new(),
        params: BTreeMap::from([("ROWS".into(), 1)]),
        launches: Vec::new(),
    };
    let overall = Configuration {
        params: BTreeMap::from([("ROWS".into(), 4)]),
        ..default.clone()
    };
    let mut result = TuningResult {
        disposition: seismic::TuningDisposition::QualifiedCandidate,
        tuning_identity: "fixture-device".into(),
        entry: "fixture".into(),
        backend: "fixture".into(),
        points: Vec::new(),
        requirement: steps(8.),
        measurement_identity: "fixture-measurements".into(),
        observations: Vec::new(),
        served: vec![default, overall.clone()],
        reselection: None,
        implementation_identity: "fixture-implementation".into(),
        parameters: Vec::new(),
        configurations: vec![ConfigurationRecord {
            formation: None,
            configuration: overall.clone(),
            artifact: Some("selected".into()),
            observations: Vec::new(),
            outcome: Outcome::Measured {
                points: Vec::new(),
                confirmed: Vec::new(),
            },
        }],
        overall,
        method: TuningMethod::Census,
        time: TuningTime::default(),
        coverage: None,
    };
    assert!(ChosenEvidence::of(&result).qualified_at(&shapes).is_err());
    result
        .observations
        .push(observation(&shapes[0].label, 4, 2.));
    result.configurations[0].observations.push(0);
    assert!(
        ChosenEvidence::of(&result).qualified_at(&shapes).is_err(),
        "untimed cases remain required"
    );
    // Another candidate's measurement at the untimed case is not the
    // selected one's.
    result
        .observations
        .push(observation(&shapes[1].label, 8, 2.));
    assert!(ChosenEvidence::of(&result).qualified_at(&shapes).is_err());
    result.configurations[0].observations.push(1);
    assert!(ChosenEvidence::of(&result).qualified_at(&shapes).is_ok());
    // Evidence at every case is not admission: one measurement beyond α
    // disqualifies the choice.
    result.observations[1] = observation(&shapes[1].label, 4, 8.5);
    assert!(
        ChosenEvidence::of(&result)
            .qualified_at(&shapes)
            .unwrap_err()
            .contains("error ratio 8.5 exceeds alpha 8")
    );
    result.observations[1] = observation(&shapes[1].label, 4, 2.);
    // A choice its search never promoted has no confirmation or coverage.
    let promoted = result.served.pop().unwrap();
    assert!(ChosenEvidence::of(&result).qualified_at(&shapes).is_err());
    result.served.push(promoted);
    assert!(ChosenEvidence::of(&result).qualified_at(&shapes).is_ok());

    // A recorded canonical fallback is usable without that evidence, unless
    // the canonical artifact itself was rejected.
    result.disposition = seismic::TuningDisposition::CanonicalFallback {
        artifact: "canonical".into(),
    };
    assert!(ChosenEvidence::of(&result).usable_fallback().is_ok());
    result.configurations[0].outcome =
        Outcome::Excluded(seismic::Exclusion::Execution("known defect".into()));
    assert!(ChosenEvidence::of(&result).usable_fallback().is_err());
}

#[test]
fn draft_readout_tuning_preserves_the_complete_artifact_vocabulary() {
    let mut definition = fixture_definition();
    let vocabulary = 248_320;
    definition.decoder.vocabulary = vocabulary;
    definition.decoder.entry.embedding.shape[0] = vocabulary;
    definition.decoder.exit.output.shape[0] = vocabulary;
    let manifest = fixture_manifest(&definition);
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        crate::resident_layout(crate::ExecutionPath::Native, BackendName::Metal),
    )
    .unwrap();
    let inputs = ModelInputs {
        definition: &definition,
        limits: LIMITS,
        load: &load,
    };
    let mtp = readout::PreparedProjectionTuning {
        scope: magnitude_family_contracts::WeightScope::Target,
        kind: magnitude_family_contracts::WeightKind::Output,
        columns: None,
        indexed: false,
        softcap: 0.0,
        weight: Element::f16(),
        activation: Element::bf16(),
    };
    let separate = readout::HeadRowsTuning {
        projection_scope: magnitude_family_contracts::WeightScope::Target,
        norm_scope: magnitude_family_contracts::WeightScope::Target,
        norm: Element::f16(),
        weight: Element::f16(),
        activation: Element::bf16(),
        epsilon: 1e-6,
        rows: None,
    };
    for statics in [
        mtp.statics(&inputs).unwrap(),
        separate.statics(&inputs).unwrap(),
    ] {
        assert_eq!(
            statics.iter().find(|(name, _)| *name == "V").unwrap().1,
            vocabulary
        );
    }
}

#[test]
fn persisted_point_weights_preserve_cache_identity() {
    // Real prefill weights: parsing these one ULP away caused every warm
    // model load to retune otherwise identical, fully qualified entries.
    for weight in [0.017857142857142853_f64, 0.026785714285714274] {
        let point = seismic::PointRecord {
            label: "prefill".into(),
            weight,
            class: Some("m16".into()),
        };
        let encoded = serde_json::to_vec(&point).unwrap();
        let restored: seismic::PointRecord = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            restored.weight.to_bits(),
            weight.to_bits(),
            "{}",
            String::from_utf8(encoded).unwrap()
        );
    }
}

#[test]
fn stored_cpu_choice_accepts_device_owned_parameters() {
    let device = seismic::Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let implementation = seismic::generated::native_implementation::<shape_rows::Entry>(&device)
        .unwrap()
        .unwrap();
    let choice = implementation
        .default_specialization(&NativeSpecialization::new())
        .unwrap()
        .with_param("cpu.tier", 0)
        .with_param("cpu.workers.0", 0)
        .with_param("cpu.workers.1", 0);
    assert!(implementation.validate(&choice).is_err());
    assert!(
        seismic::generated::native_specialization_valid::<shape_rows::Entry>(&device, &choice)
            .unwrap()
    );
}

/// A search's count collects each entry without tuning parameters once, and
/// its mandatory phase forms them beside the units' canonical programs and
/// reports them, so the load after the search forms none of them itself.
#[test]
fn the_mandatory_phase_forms_the_untuned_entries_the_count_met() {
    let device = seismic::Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let mut spec = crate::native::specialization::Specializer::count_untuned(&device);
    for (element, bindings) in [
        (seismic::Element::bf16(), "A=bf16"),
        (seismic::Element::u32(), "A=u32"),
        (seismic::Element::bf16(), "A=bf16"),
    ] {
        let entry = magnitude_kernels::copy_rows::native_entry_with(
            magnitude_kernels::copy_rows::Elements { A: element },
        );
        assert!(spec.fixed(bindings, &[], entry).unwrap().is_none());
    }
    let untuned = spec.into_untuned();
    assert_eq!(untuned.len(), 2);
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let recorder = Recorder::default();
    let tuner = Tuner::count(
        &device,
        TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: None,
            numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
            settings: TuningSettings::STANDARD,
        },
        LIMITS,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        Default::default(),
    );
    let tuner = tuner.census(untuned).unwrap();
    assert!(matches!(
        recorder.0.borrow().as_slice(),
        [TuningEvent::Untuned { programs: 2, .. }]
    ));
    assert!(tuner.search().unwrap().conclude().is_empty());
}

#[test]
fn the_tuning_cache_key_is_what_a_result_is_valid_for() {
    let device = seismic::Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let measured = |shapes: &[PointShape], activation, budget: Duration| {
        tuning_key_material(
            &device,
            "dense_output",
            "A=bf16",
            &NativeSpecialization::new(),
            "same-implementation",
            activation,
            shapes,
            budget,
        )
    };
    let budgeted =
        |shapes: &[PointShape], budget: Duration| measured(shapes, Some(DType::BF16), budget);
    let key = |shapes: &[PointShape]| budgeted(shapes, STANDARD_TUNING_BUDGET);
    let shapes = row_points(LIMITS);
    // Weights and order steer how a search spends its time, not what its
    // result is valid for.
    let mut reweighted = shapes
        .iter()
        .rev()
        .map(|shape| PointShape {
            weight: 1.0,
            ..shape.clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(key(&shapes), key(&reweighted));
    reweighted.pop();
    assert_ne!(key(&shapes), key(&reweighted));
    assert!(key(&shapes).contains(&format!("tuning {SEARCH_VERSION}\n")));
    // What is measured is result identity: the comparison's version and
    // the activation type subjects are measured against. The caller's α is
    // not: no part of the key is derived from a profile.
    assert!(key(&shapes).contains(&format!(
        "comparison {} steps Some(BF16)\n",
        seismic::COMPARISON_VERSION
    )));
    assert_ne!(
        key(&shapes),
        measured(&shapes, Some(DType::F16), STANDARD_TUNING_BUDGET)
    );
    assert!(!key(&shapes).contains("profile") && !key(&shapes).contains("alpha"));
    // A result searched within one budget is never read under another.
    assert_ne!(key(&shapes), budgeted(&shapes, STANDARD_TUNING_BUDGET * 2));
}

#[test]
fn a_units_window_is_the_longest_history_its_layers_keep() {
    use magnitude_family_contracts::{
        Attention, HistoryDomain, KeyValue, ModelDefinition, Operator, SublayerIndex, WeightScope,
    };
    fn mixer(definition: &mut ModelDefinition, block: usize) -> &mut Attention {
        match &mut definition.decoder.blocks[block].sublayers[0].op {
            Operator::Attention(attention) => &mut **attention,
            other => panic!("the fixture's mixer is {}", other.name()),
        }
    }
    let layer = |block: u32| SublayerIndex { block, sublayer: 0 };
    // Four attention layers: the whole context, windows of 1024 and 2048
    // rows, and a layer reading the 1024-row window's history.
    let (mut definition, load) = fixture_load();
    let block = definition.decoder.blocks[0].clone();
    definition.decoder.blocks = vec![block; 4];
    for (block, tokens) in [(1, 1024), (2, 2048)] {
        let KeyValue::Owned { domain, .. } = &mut mixer(&mut definition, block).key_value else {
            panic!("the fixture's attention owns its history");
        };
        *domain = HistoryDomain::Window { tokens };
    }
    mixer(&mut definition, 3).key_value = KeyValue::Shared { source: layer(1) };
    let inputs = ModelInputs {
        definition: &definition,
        limits: LIMITS,
        load: &load,
    };
    let kept = |blocks: &[u32]| {
        attention::history_window(
            &inputs,
            &blocks
                .iter()
                .map(|block| WeightScope::TargetSublayer(layer(*block)))
                .collect::<Vec<_>>(),
        )
    };
    assert_eq!(kept(&[1]), Ok(Some(1024)));
    // The largest window among a unit's layers.
    assert_eq!(kept(&[1, 2]), Ok(Some(2048)));
    // A Shared layer keeps what its source does.
    assert_eq!(kept(&[3]), Ok(Some(1024)));
    // Any layer keeping the whole context leaves the unit unbounded.
    assert_eq!(kept(&[0]), Ok(None));
    assert_eq!(kept(&[1, 0, 2]), Ok(None));
    assert!(kept(&[]).is_err());
    // One request's listed history holds what the layers keep: the whole
    // context, or the window, each in whole pages.
    let codec = magnitude_state::KvCodec::AffineK8V4;
    for window in [None, Some(1024), Some(2048)] {
        let rows = request_history_rows(&inputs, codec, window).unwrap();
        assert!(
            rows >= window.unwrap_or(LIMITS.context_tokens),
            "{window:?}: {rows}"
        );
    }
}

#[test]
fn numerical_histories_include_witnesses_and_the_window_limit() {
    let decode = || served_row_points(LIMITS, |rows| rows == 1);
    let labels = |window: Option<u64>| {
        with_contexts(LIMITS, window, decode())
            .into_iter()
            .map(|point| point.label)
            .collect::<Vec<_>>()
    };
    // Numerical witnesses remain present even when they carry no timing demand.
    let whole = ["m1-c0", "m1-c256", "m1-c4096", "m1-c16384"];
    assert_eq!(labels(None), whole);
    // A window shorter than the context is the longest history its layers
    // see, and the history of every prompt longer than it.
    assert_eq!(labels(Some(1024)), ["m1-c0", "m1-c256", "m1-c1024"]);
    assert_eq!(labels(Some(4096)), ["m1-c0", "m1-c256", "m1-c4096"]);
    assert_eq!(labels(Some(128)), ["m1-c0", "m1-c128"]);
    // A window that holds the whole context bounds nothing.
    assert_eq!(labels(Some(16384)), whole);
    assert_eq!(labels(Some(65536)), whole);
    let windowed = with_contexts(LIMITS, Some(1024), row_points(LIMITS));
    assert_eq!(windowed.len(), TUNING_ROWS.len() * 3);
    assert!((windowed.iter().map(|point| point.weight).sum::<f64>() - 1.0).abs() < 1e-12);
    // The window's lengths of one row point are its class, each row class
    // keeping the share it has at any history.
    assert_eq!(windowed[1].class.as_deref(), Some("m1"));
    let share = |points: &[PointShape], rows: u64| {
        points
            .iter()
            .filter(|point| point.rows == rows)
            .map(|point| point.weight)
            .sum::<f64>()
    };
    let whole = attention_points(LIMITS);
    for rows in TUNING_ROWS {
        assert!((share(&windowed, rows) - share(&whole, rows)).abs() < 1e-12);
    }
}

#[test]
fn point_cost_grows_with_rows_and_history() {
    let point = |rows: u64, context: Option<u64>| PointShape {
        terms: Vec::new(),
        label: String::new(),
        weight: 1.0,
        rows,
        sequences: 1,
        context,
        class: None,
    };
    assert_eq!(point(1, None).cost(), 1.0);
    // Without history, a point's work grows with its rows alone.
    assert_eq!(point(8, None).cost(), 8.0);
    assert_eq!(point(1, Some(4096)).cost(), 4097.0);
    assert!(point(8, Some(0)).cost() < point(1, Some(256)).cost());
}

#[test]
fn attention_points_cross_rows_with_served_contexts() {
    let points = attention_points(LIMITS);
    assert_eq!(
        points.len(),
        TUNING_ROWS.len() * 4,
        "all numerical histories through the served context"
    );
    assert!(points.iter().all(|point| point.context.unwrap() <= 16384));
    assert_eq!(points[0].label, "m1-c0");
    // The history lengths of one row point form its class.
    assert_eq!(points[0].class.as_deref(), Some("m1"));
    assert_eq!(points[2].class.as_deref(), Some("m1"));
    assert_eq!(points[4].class.as_deref(), Some("m2"));
    assert!((points.iter().map(|point| point.weight).sum::<f64>() - 1.0).abs() < 1e-12);
    let short = attention_points(TuningLimits {
        max_rows: 1,
        max_projected_rows: 1,
        context_tokens: 128,
    });
    assert_eq!(
        short.iter().map(|p| p.context.unwrap()).collect::<Vec<_>>(),
        [0, 128]
    );
}

#[test]
fn served_points_keep_an_entry_whose_rows_exceed_the_bound() {
    let chunked = served_row_points(LIMITS, |rows| rows >= 16);
    assert_eq!(
        chunked.iter().map(|point| point.rows).collect::<Vec<_>>(),
        [16, 32, 64, 128, 256, 512]
    );
    let decode = served_row_points(LIMITS, |rows| rows <= 8);
    assert_eq!(
        decode.iter().map(|point| point.rows).collect::<Vec<_>>(),
        [1, 2, 4, 8]
    );
    // A bound admits only the served rows at or below it.
    let short = served_row_points(
        TuningLimits {
            max_rows: 16,
            ..LIMITS
        },
        |rows| rows >= 16,
    );
    assert_eq!(
        short.iter().map(|point| point.rows).collect::<Vec<_>>(),
        [16]
    );
    assert!((chunked.iter().map(|point| point.weight).sum::<f64>() - 1.0).abs() < 1e-12);
    let stand_in = served_row_points(
        TuningLimits {
            max_rows: 8,
            ..LIMITS
        },
        |rows| rows >= 16,
    );
    assert_eq!(stand_in.len(), 1);
    assert_eq!(stand_in[0].rows, 16);
    assert_eq!(stand_in[0].weight, 1.0);
}

#[test]
fn rotations_take_distinct_layers_spread_over_depth() {
    let block = |block| {
        WeightScope::TargetSublayer(magnitude_family_contracts::SublayerIndex {
            block,
            sublayer: 0,
        })
    };
    let scopes = (0..32).map(block).collect::<Vec<_>>();
    let rows = |rows| served_row_points(LIMITS, move |served| served == rows).remove(0);
    assert_eq!(
        TuningInputs::rotation_scopes(&scopes, &rows(4)),
        [0, 8, 16, 24].map(block)
    );
    assert_eq!(
        TuningInputs::rotation_scopes(&scopes, &rows(32)),
        [block(0)]
    );
    let few = [WeightScope::HeadBlock(0)];
    assert_eq!(TuningInputs::rotation_scopes(&few, &rows(1)), few);
}

fn metal() -> Option<Device> {
    seismic::Devices::discover(&[])
        .ok()?
        .open_backend(BackendName::Metal)
        .ok()
}

fn fixture_load() -> (magnitude_family_contracts::ModelDefinition, ModelLoadPlan) {
    let definition = fixture_definition();
    let manifest = fixture_manifest(&definition);
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        crate::resident_layout(crate::ExecutionPath::Native, seismic::BackendName::Metal),
    )
    .unwrap();
    (definition, load)
}

#[derive(Default)]
struct Recorder(RefCell<Vec<TuningEvent>>);

impl TuningObserver for Recorder {
    fn event(&self, event: &TuningEvent) {
        self.0.borrow_mut().push(event.clone());
    }
}

/// A fake entry: its rotation uses case-owned scratch tensors, and its
/// "search" discovers every point it is given, one a piece, measures one
/// challenger from a fixed table and promotes it: `chosen`.
#[derive(Clone)]
struct FakeCase {
    /// Canonical preparation is complete before any unit builds its inputs.
    formations: Rc<Cell<usize>>,
    /// Every point built: its label, weight and rotation length.
    seen: Rc<RefCell<Vec<(String, f64, usize)>>>,
    /// Every piece of optional work performed, in order.
    stages: Rc<RefCell<Vec<seismic::Stage>>>,
    /// Serving artifacts published: a final promoted choice's, once.
    artifacts: Rc<Cell<usize>>,
    /// The implementation digest the case reports.
    digest: String,
    bindings: String,
    launches: usize,
    /// What the census reports the defaults take at every point.
    defaults_seconds: f64,
    defaults: Configuration,
    chosen: Configuration,
    /// When set, no candidate qualifies: the unit concludes on its
    /// canonical default, unqualified.
    unqualified: bool,
}

impl FakeCase {
    /// A case choosing the defaults with `ROWS` 2 (admissible, not the
    /// default).
    fn new(
        implementation: &NativeImplementation,
        statics: &NativeSpecialization,
        digest: &str,
    ) -> Self {
        let defaults = implementation.default_specialization(statics).unwrap();
        let chosen = defaults.clone().with_launch_param(0, "ROWS", 2);
        let configuration = |spec: &NativeSpecialization| Configuration {
            statics: spec.statics().clone(),
            params: spec.params().clone(),
            launches: implementation
                .launches
                .iter()
                .enumerate()
                .map(|(ordinal, launch)| {
                    launch
                        .params
                        .iter()
                        .map(|parameter| {
                            (
                                parameter.name.clone(),
                                spec.launch_param(ordinal, &parameter.name).unwrap(),
                            )
                        })
                        .collect()
                })
                .collect(),
        };
        Self {
            formations: Rc::new(Cell::new(0)),
            seen: Rc::new(RefCell::new(Vec::new())),
            stages: Rc::new(RefCell::new(Vec::new())),
            artifacts: Rc::new(Cell::new(0)),
            digest: digest.into(),
            bindings: "fake".into(),
            launches: 1,
            defaults_seconds: 1e-3,
            defaults: configuration(&defaults),
            chosen: configuration(&chosen),
            unqualified: false,
        }
    }
}

struct FakeArgs {
    residual: Tensor,
    product: Tensor,
    down: Tensor,
    out_rows: Tensor,
    absent_scale: Tensor,
}

impl EntryTuning for FakeCase {
    type Entry = dense_output::Entry;
    type Case = FakeArgs;

    fn launches(&self) -> usize {
        self.launches
    }

    fn bindings(&self) -> String {
        self.bindings.clone()
    }
    fn statics(&self, _inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(vec![("H", 8), ("F", 16), ("DS", 0), ("DB", 0)])
    }
    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }
    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<FakeArgs>, BuildError> {
        (0..ROTATION_LAYERS)
            .map(|layer| {
                Ok(FakeArgs {
                    residual: inputs.activation(Element::f32(), &[point.rows, 8], layer as u64)?,
                    product: inputs.scratch(Element::bf16(), &[point.rows, 16])?,
                    down: inputs.scratch(Element::bf16(), &[8, 16])?,
                    out_rows: inputs.every_row(point.rows)?,
                    absent_scale: inputs.activation(Element::f32(), &[0], 0)?,
                })
            })
            .collect()
    }
    fn args<'a>(case: &'a mut FakeArgs) -> dense_output::Args<'a> {
        dense_output::Args {
            residual: &case.residual,
            product: &case.product,
            down_weight: &case.down,
            out_rows: &case.out_rows,
            down_scale: &case.absent_scale,
            down_bound: &case.absent_scale,
        }
    }

    #[cfg(feature = "tuning-survey")]
    fn tune(
        &self,
        _device: &Device,
        _statics: &NativeSpecialization,
        _points: &mut dyn seismic::PointSource<'_, Self::Entry>,
        _validation: seismic::NumericalRequirement,
        strategy: Strategy,
    ) -> Result<TuningResult, TuneError> {
        panic!("the fake case is searched, not tuned by {strategy:?}")
    }

    fn open(
        &self,
        _device: &Device,
        statics: &NativeSpecialization,
        validation: seismic::NumericalRequirement,
    ) -> Result<Box<dyn EntrySearch<Self::Entry>>, TuneError> {
        assert_eq!(&self.chosen.statics, statics.statics());
        // The unit's activation type is the entry's `A` binding.
        let seismic::NumericalRequirement::RoundingSteps { alpha, activation } = validation else {
            panic!("the engine holds candidates to rounding steps, not {validation:?}")
        };
        assert_eq!(activation, Some(DType::BF16));
        Ok(Box::new(FakeSearch {
            case: self.clone(),
            alpha,
            specs: Vec::new(),
            formed: false,
            discovered: 0,
            measured: false,
            promoted: false,
            resident: false,
        }))
    }
    fn digest(
        &self,
        _device: &Device,
        _statics: &NativeSpecialization,
    ) -> Result<String, TuneError> {
        Ok(self.digest.clone())
    }
    fn entry(&self) -> seismic::BoundEntry<Self::Entry> {
        dense_output::native_entry_with(dense_output::Elements {
            DW: Element::bf16(),
            A: Element::bf16(),
        })
    }
}

/// The tuning of a [`FakeCase`]: every point discovered one a piece, then
/// one challenger measured at half the defaults' time, then promoted
/// (unless the case is `unqualified`: then nothing leads). It serves a
/// legal choice with a complete record throughout, like a real search. Its
/// challenger measures a quarter of the α it was opened under from the
/// defaults at every point; a second candidate measures half as much again
/// as α at its first point and is rejected there.
struct FakeSearch {
    case: FakeCase,
    alpha: f64,
    specs: Vec<seismic::PointSpec>,
    formed: bool,
    discovered: usize,
    measured: bool,
    promoted: bool,
    /// Whether the inputs built for its points are held.
    resident: bool,
}

impl FakeSearch {
    fn measurements(&self, count: usize, seconds: f64) -> Vec<seismic::PointMeasurement> {
        self.specs
            .iter()
            .take(count)
            .map(|spec| seismic::PointMeasurement {
                point: spec.label.clone(),
                key: seismic::LaunchKey {
                    launches: vec![0],
                    values: Default::default(),
                },
                median_seconds: seconds,
                deviation_seconds: 0.0,
                samples: vec![seconds],
                repetitions: 1,
                rotation_bytes: 0,
            })
            .collect()
    }

    fn piece(&self) -> Duration {
        Duration::from_secs_f64(self.case.defaults_seconds)
    }
}

impl EntrySearch<dense_output::Entry> for FakeSearch {
    fn grounding(
        &mut self,
        points: &mut dyn seismic::PointSource<'static, dense_output::Entry>,
    ) -> Result<Option<seismic::Grounding>, TuneError> {
        self.specs = points.points();
        Ok(None)
    }

    fn grounded(&mut self, grounded: Option<seismic::Grounded>) -> Result<(), TuneError> {
        assert!(grounded.is_none(), "the fake compiles nothing on a thread");
        self.case.formations.set(self.case.formations.get() + 1);
        self.formed = true;
        Ok(())
    }

    fn forming(&self) -> Option<seismic::Forming> {
        None
    }

    fn formed(&mut self, _: seismic::Formed) -> Result<Duration, TuneError> {
        unreachable!("the fake compiles nothing ahead")
    }

    fn plan(&mut self, _: seismic::UnitPlan) {}

    fn terminal(&self) -> Result<TuningResult, TuneError> {
        if !self.formed {
            return Err(TuneError::Declaration(
                "the defaults were not formed".into(),
            ));
        }
        let mut configurations = Vec::new();
        // This deterministic fixture supplies explicit synthetic
        // measurements of its candidates; it executes no arithmetic.
        let mut observations = Vec::new();
        if self.discovered > 0 {
            configurations.push(ConfigurationRecord {
                formation: None,
                configuration: self.case.defaults.clone(),
                artifact: Some("a".into()),
                // The default is the reference: it carries no evidence
                // against itself.
                observations: Vec::new(),
                outcome: Outcome::Measured {
                    points: self.measurements(self.discovered, self.case.defaults_seconds),
                    confirmed: Vec::new(),
                },
            });
        }
        if self.measured {
            let mut rejected = self.case.chosen.clone();
            rejected.launches[0].insert("ROWS".into(), 4);
            observations.extend(
                self.specs
                    .iter()
                    .map(|spec| observation(&spec.label, 2, self.alpha / 4.)),
            );
            configurations.push(ConfigurationRecord {
                formation: None,
                configuration: self.case.chosen.clone(),
                artifact: Some("b".into()),
                observations: (0..self.specs.len()).collect(),
                outcome: Outcome::Measured {
                    points: self.measurements(self.specs.len(), self.case.defaults_seconds / 2.0),
                    confirmed: Vec::new(),
                },
            });
            let beyond = observation(&self.specs[0].label, 4, self.alpha * 1.5);
            configurations.push(ConfigurationRecord {
                formation: None,
                configuration: rejected,
                artifact: Some("c".into()),
                observations: vec![observations.len()],
                outcome: Outcome::Excluded(
                    seismic::admit(&beyond, &steps(self.alpha))
                        .unwrap_err()
                        .exclusion(),
                ),
            });
            observations.push(beyond);
        }
        let candidates = configurations.len().saturating_sub(1);
        Ok(TuningResult {
            disposition: if self.promoted {
                seismic::TuningDisposition::QualifiedCandidate
            } else {
                seismic::TuningDisposition::CanonicalFallback {
                    artifact: "a".into(),
                }
            },
            tuning_identity: "fake-device".into(),
            entry: "dense_output".into(),
            backend: "metal".into(),
            points: self
                .specs
                .iter()
                .map(|spec| seismic::PointRecord {
                    label: spec.label.clone(),
                    weight: spec.weight,
                    class: spec.class.clone(),
                })
                .collect(),
            requirement: steps(self.alpha),
            measurement_identity: "synthetic-allocator-fixture".into(),
            observations,
            served: std::iter::once(self.case.defaults.clone())
                .chain(self.promoted.then(|| self.case.chosen.clone()))
                .collect(),
            reselection: None,
            implementation_identity: String::new(),
            parameters: Vec::new(),
            configurations,
            overall: if self.promoted {
                self.case.chosen.clone()
            } else {
                self.case.defaults.clone()
            },
            method: if self.discovered < self.specs.len() {
                TuningMethod::Census
            } else {
                TuningMethod::Search {
                    groups: 1,
                    candidates,
                    complete: self.next().is_none(),
                }
            },
            time: TuningTime::default(),
            coverage: None,
        })
    }

    fn next(&self) -> Option<seismic::Work> {
        let priced = seismic::Quote {
            priced: self.piece(),
            unpriced: 0,
        };
        if self.discovered < self.specs.len() {
            // A point's classes are first observed by its own discovery.
            Some(seismic::Work {
                stage: seismic::Stage::Discovery,
                quote: seismic::Quote {
                    priced: Duration::ZERO,
                    unpriced: 1,
                },
                forming: seismic::Quote::default(),
                // As a real unit's discovery is worth: each point one
                // operation of what remains.
                opportunity: 0.,
                footprint: seismic::Footprint::default(),
            })
        } else if !self.measured {
            Some(seismic::Work {
                stage: seismic::Stage::Measure,
                quote: priced,
                forming: seismic::Quote::default(),
                opportunity: 0.5,
                footprint: seismic::Footprint::default(),
            })
        } else if !self.promoted && !self.case.unqualified {
            Some(seismic::Work {
                stage: seismic::Stage::Promote,
                quote: priced,
                forming: seismic::Quote::default(),
                opportunity: 0.5,
                footprint: seismic::Footprint::default(),
            })
        } else {
            None
        }
    }

    fn perform(
        &mut self,
        points: &mut dyn seismic::PointSource<'static, dense_output::Entry>,
        _before: Option<Instant>,
    ) -> Result<seismic::Performed, TuneError> {
        let work = self.next().expect("a unit performs the work it named");
        let began = Instant::now();
        match work.stage {
            seismic::Stage::Discovery => {
                let spec = &self.specs[self.discovered];
                let built = points
                    .build(self.discovered)
                    .unwrap_or_else(|unavailable| panic!("{unavailable:?}"));
                self.resident = true;
                self.case.seen.borrow_mut().push((
                    spec.label.clone(),
                    spec.weight,
                    built.rotation.len(),
                ));
                self.discovered += 1;
            }
            seismic::Stage::Form | seismic::Stage::Measure => self.measured = true,
            seismic::Stage::Promote => self.promoted = true,
            seismic::Stage::Confirm => unreachable!("the fake's lead is resolved"),
        }
        self.case.stages.borrow_mut().push(work.stage);
        Ok(seismic::Performed {
            stage: work.stage,
            quote: work.quote,
            actual: began.elapsed(),
            interrupted: false,
            spent: seismic::Spent::default(),
            raises: Vec::new(),
            promoted: work.stage == seismic::Stage::Promote,
            footprint: work.footprint,
            memory: seismic::PieceMemory::default(),
        })
    }

    fn release(&mut self, points: &mut dyn seismic::PointSource<'static, dense_output::Entry>) {
        if std::mem::take(&mut self.resident) {
            for point in 0..self.discovered {
                points.release(point);
            }
        }
    }

    fn narrow(&mut self) -> bool {
        false
    }

    fn holding(&self) -> u64 {
        u64::from(self.resident)
    }

    fn programs(&self) -> u64 {
        0
    }

    fn give_up_programs(&mut self) {}

    fn largest_footprint(&self) -> u64 {
        0
    }

    fn longest_operation(&self) -> Duration {
        self.piece()
    }

    fn artifact_owed(&self, _worked: bool) -> seismic::Quote {
        seismic::Quote::default()
    }

    fn artifact_publication(&self, _worked: bool) -> seismic::Publication {
        seismic::Publication::default()
    }

    fn inputs_built(&self) -> seismic::Quote {
        seismic::Quote::default()
    }

    fn publish_artifact(&mut self) -> Result<Duration, TuneError> {
        assert!(
            self.promoted,
            "only a promoted choice's artifact is published"
        );
        self.case.artifacts.set(self.case.artifacts.get() + 1);
        Ok(Duration::ZERO)
    }

    fn service_measurements(&self) -> Vec<seismic::PointMeasurement> {
        self.measurements(self.discovered, self.case.defaults_seconds)
    }

    fn reweigh(&mut self, _: &[f64]) -> bool {
        !self.measured
    }

    fn discovered(&self) -> bool {
        self.discovered == self.specs.len()
    }

    fn standing(&self) -> Standing {
        Standing {
            cost: if self.promoted { 0.5 } else { 1. },
            measured: usize::from(self.discovered > 0) + usize::from(self.measured),
            generated: 2,
            requests: 2,
            declared: Some(2),
            exhausted: self.measured,
            programs: usize::from(self.measured),
            forming_seconds: 0.,
        }
    }
}

/// The mandatory phase comes first and leaves every unit legal: abandoning
/// the tuner at any instant after it leaves a cache that a second load
/// consumes with zero searches. A deterministic search and real typed point
/// factories; no model inference or claimed numerical qualification.
#[test]
fn abandoning_the_tuner_after_the_mandatory_phase_leaves_a_cache_a_second_load_consumes() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let recorder = Recorder::default();
    let context = TuningContext {
        definition: &definition,
        weights: &ZeroTuningWeights,
        observer: &recorder,
        cache: None,
        numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
        settings: TuningSettings::STANDARD,
    };
    let limits = TuningLimits {
        max_rows: 64,
        max_projected_rows: 8,
        context_tokens: 256,
    };
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(
                name.clone(),
                if name == "DS" || name == "DB" {
                    0
                } else if name == "H" {
                    8
                } else {
                    16
                },
            )
        });
    // The first load is abandoned after `performed` pieces of its optional
    // work: none at all, some of its discovery, its measurement, its
    // promotion. Whenever that is, the cache holds the unit's legal record
    // as it then stood.
    struct Abandoning {
        after: usize,
        stages: Cell<usize>,
    }
    impl TuningObserver for Abandoning {
        fn event(&self, event: &TuningEvent) {
            if matches!(event, TuningEvent::Stage(_)) {
                self.stages.set(self.stages.get() + 1);
                if self.stages.get() == self.after {
                    panic!("the tuner is abandoned here");
                }
            }
        }
    }
    let points = row_points(limits).len();
    for performed in 0..=points + 2 {
        let root = std::env::temp_dir().join(format!(
            "magnitude-tuning-abandoned-{}-{performed}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let cache = KernelCache::open(root.clone(), crate::DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let case = FakeCase::new(&implementation, &statics, "abandoned");
        let abandoning = Abandoning {
            after: performed,
            stages: Cell::new(0),
        };
        let mut tuner = Tuner::count(
            &device,
            TuningContext {
                observer: &abandoning,
                cache: Some(&cache),
                ..context
            },
            limits,
            TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
            fixture_invocations(&case, &statics, limits),
        );
        tuner.tune(&case, &implementation, &statics).unwrap();
        // The mandatory phase forms the canonical artifact and publishes
        // the unit's record before any input is built.
        let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
        assert_eq!(case.formations.get(), 1);
        assert!(case.seen.borrow().is_empty());
        tuner.tune(&case, &implementation, &statics).unwrap();
        if performed == 0 {
            drop(tuner);
        } else {
            let abandoned =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tuner.search().is_ok()));
            assert!(abandoned.is_err(), "the load was abandoned mid-search");
        }
        assert_eq!(case.stages.borrow().len(), performed);
        // A second load consumes that record: nothing is formed, built,
        // measured or searched, and the choice served is the one that
        // stood: the challenger only once it was promoted.
        let again = FakeCase::new(&implementation, &statics, "abandoned");
        let mut tuner = Tuner::count(
            &device,
            TuningContext {
                cache: Some(&cache),
                ..context
            },
            limits,
            TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
            fixture_invocations(&again, &statics, limits),
        );
        tuner.tune(&again, &implementation, &statics).unwrap();
        let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
        tuner.tune(&again, &implementation, &statics).unwrap();
        let mut tuner = tuner.search().unwrap();
        let chosen = tuner.tune(&again, &implementation, &statics).unwrap();
        assert_eq!(again.formations.get(), 0);
        assert!(again.seen.borrow().is_empty() && again.stages.borrow().is_empty());
        let promoted = performed == points + 2;
        if promoted {
            assert_eq!(chosen.launch_param(0, "ROWS"), Some(2));
        } else {
            assert_eq!(
                chosen,
                implementation.default_specialization(&statics).unwrap(),
                "after {performed} pieces"
            );
        }
        let tuned = tuner.conclude();
        assert_eq!(tuned.len(), 1);
        assert_eq!(tuned[0].origin, TuningOrigin::Stored);
        assert_eq!(tuned[0].qualified, promoted);
        // The record is the one last published: the canonical default's
        // from the mandatory phase, which measured nothing, or the
        // promotion's, with everything measured until then. What a piece
        // measured without promoting was the abandoned load's alone.
        assert_eq!(tuned[0].measured, if promoted { 2 } else { 0 });
        std::fs::remove_dir_all(root).unwrap();
    }
}

/// MEASUREMENT: what compiling the engine's dense output projection costs
/// the process while it runs on worker threads, as a load's challenger
/// compiles do (`cargo test --release -p magnitude-executor --lib --
/// --ignored compiles_on_workers --nocapture`). Each distinct value of a
/// launch's code parameters is a compile the toolchain has not seen. The
/// process's physical footprint is sampled every millisecond: before,
/// its most while four compiles run at once, with every kernel held, and
/// with all of them released.
#[test]
#[ignore]
fn profile_what_compiles_on_workers_add_to_the_process_footprint() {
    let Some(device) = metal() else {
        return;
    };
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(
                name.clone(),
                if name == "DS" || name == "DB" {
                    0
                } else if name == "H" {
                    1024
                } else {
                    4096
                },
            )
        });
    let default = implementation.default_specialization(&statics).unwrap();
    // One specialization per other value of each launch's code parameters.
    let mut specializations = Vec::new();
    for (launch, declared) in implementation.launches.iter().enumerate() {
        for parameter in declared.params.iter().filter(|parameter| parameter.code) {
            for value in &parameter.values {
                let changed =
                    default
                        .clone()
                        .with_launch_param(launch, parameter.name.clone(), *value);
                if changed != default && implementation.validate(&changed).is_ok() {
                    specializations.push(changed);
                }
            }
        }
    }
    let elements = || dense_output::Elements {
        DW: Element::bf16(),
        A: Element::bf16(),
    };
    let footprint = || super::process_footprint_bytes().expect("macOS reports a footprint");
    let mib = |bytes: i64| bytes as f64 / (1 << 20) as f64;
    // What stays once a compile's kernel is released follows the compiles:
    // the first half of the specializations, compiled and released one at
    // a time, then all of them on four threads.
    let start = footprint();
    let half = specializations.len() / 2;
    for specialization in &specializations[..half] {
        drop(dense_output::native_for_device_with(
            &device,
            elements(),
            specialization,
        ));
    }
    eprintln!(
        "{half} compiled and released one at a time: +{:.1} MiB stays",
        mib(footprint() as i64 - start as i64)
    );
    let before = footprint();
    let done = std::sync::atomic::AtomicBool::new(false);
    let (most, kernels) = std::thread::scope(|scope| {
        let sampler = scope.spawn(|| {
            let mut most = 0;
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                most = most.max(footprint());
                std::thread::sleep(Duration::from_millis(1));
            }
            most
        });
        let began = Instant::now();
        let workers = specializations
            .chunks(specializations.len().div_ceil(4).max(1))
            .map(|chunk| {
                let device = &device;
                scope.spawn(move || {
                    chunk
                        .iter()
                        .filter_map(|specialization| {
                            dense_output::native_for_device_with(device, elements(), specialization)
                                .ok()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        let kernels = workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        eprintln!(
            "{} of {} specializations compiled on 4 threads in {:.2} s",
            kernels.len(),
            specializations.len(),
            began.elapsed().as_secs_f64()
        );
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        (sampler.join().unwrap(), kernels)
    });
    let held = footprint();
    let count = kernels.len().max(1) as i64;
    drop(kernels);
    let released = footprint();
    eprintln!(
        "footprint: before {:.1} MiB; most while compiling +{:.1} MiB; with {count} kernels held \
         +{:.1} MiB ({:.2} MiB a kernel); all released +{:.1} MiB",
        mib(before as i64),
        mib(most as i64 - before as i64),
        mib(held as i64 - before as i64),
        mib(held as i64 - before as i64) / count as f64,
        mib(released as i64 - before as i64),
    );
}

/// A unit's record is stored when its served choice changes, with that
/// choice's serving artifact formed, and once when the optional work stops
/// if a piece changed it since, not with every piece: the record a later
/// load reads is the mandatory phase's until a promotion, and a unit that
/// measured without promoting publishes its measurements once, at the stop.
#[test]
fn a_record_is_published_with_a_promotion_and_once_when_the_work_stops() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let limits = TuningLimits {
        max_rows: 64,
        max_projected_rows: 8,
        context_tokens: 256,
    };
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(
                name.clone(),
                if name == "DS" || name == "DB" {
                    0
                } else if name == "H" {
                    8
                } else {
                    16
                },
            )
        });
    /// After each piece and each closing publication: what happened, and
    /// the configurations the stored record holds as measured.
    struct Stored<'c> {
        cache: &'c KernelCache,
        seen: RefCell<Vec<(&'static str, usize)>>,
    }
    impl Stored<'_> {
        fn measured(&self) -> usize {
            let stored = std::fs::read_dir(self.cache.root().join("evidence"))
                .unwrap()
                .flatten()
                .map(|record| record.path())
                .collect::<Vec<_>>();
            let [record] = stored.as_slice() else {
                panic!("one unit stores one record: {stored:?}");
            };
            serde_json::from_slice::<TuningResult>(&std::fs::read(record).unwrap())
                .unwrap()
                .configurations
                .iter()
                .filter(|record| matches!(record.outcome, Outcome::Measured { .. }))
                .count()
        }
    }
    impl TuningObserver for Stored<'_> {
        fn event(&self, event: &TuningEvent) {
            let what = match event {
                TuningEvent::Stage(stage) if stage.promoted => "promoted",
                TuningEvent::Stage(_) => "piece",
                TuningEvent::Published { .. } => "published",
                _ => return,
            };
            self.seen.borrow_mut().push((what, self.measured()));
        }
    }
    let points = row_points(limits).len();
    for unqualified in [false, true] {
        let root = std::env::temp_dir().join(format!(
            "magnitude-tuning-publication-{}-{unqualified}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let cache = KernelCache::open(root.clone(), crate::DEFAULT_KERNEL_CACHE_BYTES).unwrap();
        let stored = Stored {
            cache: &cache,
            seen: RefCell::new(Vec::new()),
        };
        let mut case = FakeCase::new(&implementation, &statics, "published");
        case.unqualified = unqualified;
        let mut tuner = Tuner::count(
            &device,
            TuningContext {
                definition: &definition,
                weights: &ZeroTuningWeights,
                observer: &stored,
                cache: Some(&cache),
                numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
                settings: TuningSettings::STANDARD,
            },
            limits,
            TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
            fixture_invocations(&case, &statics, limits),
        );
        tuner.tune(&case, &implementation, &statics).unwrap();
        let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
        // The mandatory phase published the canonical default's record.
        assert_eq!(stored.measured(), 0);
        tuner.tune(&case, &implementation, &statics).unwrap();
        let tuner = tuner.search().unwrap();
        // Discovery and the challenger's measurement store nothing.
        let mut expected = vec![("piece", 0); points + 1];
        if unqualified {
            // Nothing was promoted: the measurements are published once,
            // when the work stops.
            expected.push(("published", 2));
        } else {
            // The promotion forms the promoted choice's serving artifact
            // and publishes the record with everything measured. Nothing
            // changed it after, so nothing is owed when the work stops.
            expected.push(("promoted", 2));
        }
        assert_eq!(*stored.seen.borrow(), expected);
        // Only a promoted choice has an artifact to publish: the canonical
        // default's is the mandatory phase's own.
        assert_eq!(case.artifacts.get(), usize::from(!unqualified));
        let tuned = tuner.conclude();
        assert_eq!(tuned[0].measured, 2);
        assert_eq!(stored.measured(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn no_demand_composed_choices_are_published_in_the_mandatory_phase_and_time_fails_nothing() {
    struct Recipe {
        published: Rc<RefCell<Vec<bool>>>,
        delay: Duration,
    }
    impl<'a> ComposedContender<'a> for Recipe {
        fn dependencies(&self) -> Vec<TuningKey> {
            Vec::new()
        }
        fn key(&self) -> String {
            "unused".into()
        }
        fn label(&self) -> &'static str {
            "test_recipe"
        }
        fn cached(&self) -> Option<bool> {
            None
        }
        fn set_invocations(
            &mut self,
            _: &crate::programs::invocation_demand::InvocationCounts,
            _: &Objective,
        ) {
        }
        fn census(
            &mut self,
            _: &mut TuningInputs<'_, 'a>,
        ) -> Result<Option<ComposedCensus>, CatalogFailure> {
            panic!("an unused recipe must not construct inputs");
        }
        fn qualify(
            &mut self,
            _: &mut TuningInputs<'_, 'a>,
            _before: Instant,
        ) -> Result<bool, CatalogFailure> {
            panic!("an unused recipe must not qualify");
        }
        fn conclude(&mut self, selected: bool) -> Result<(), CatalogFailure> {
            std::thread::sleep(self.delay);
            self.published.borrow_mut().push(selected);
            Ok(())
        }
    }
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let recorder = Recorder::default();
    for delayed in [false, true] {
        let published = Rc::new(RefCell::new(Vec::new()));
        let recipe = || {
            Box::new(Recipe {
                published: published.clone(),
                delay: if delayed {
                    Duration::from_millis(100)
                } else {
                    Duration::ZERO
                },
            })
        };
        let mut tuner = Tuner::count(
            &device,
            TuningContext {
                definition: &definition,
                weights: &ZeroTuningWeights,
                observer: &recorder,
                cache: None,
                numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
                settings: TuningSettings {
                    budget: Duration::from_millis(50),
                    fresh: true,
                },
            },
            LIMITS,
            TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
            super::super::specialization::InvocationDemand {
                units: HashMap::from([(
                    ("test_recipe", "unused".into(), BTreeMap::new()),
                    crate::programs::invocation_demand::InvocationCounts::default(),
                )]),
                objective: Objective::new(workload::terms(LIMITS, &[])),
            },
        );
        assert_eq!(tuner.register_composed(recipe()).unwrap(), None);
        assert!(
            published.borrow().is_empty(),
            "counting cannot publish a choice"
        );
        // The recipe's mandatory work publishes the canonical choice it
        // serves until it is qualified: nothing is owed at the end.
        let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
        assert_eq!(tuner.register_composed(recipe()).unwrap(), None);
        assert_eq!(*published.borrow(), [false]);
        // A publication that outlasts the whole budget fails nothing.
        let mut tuner = tuner.search().unwrap();
        assert_eq!(tuner.register_composed(recipe()).unwrap(), Some(false));
        assert_eq!(
            *published.borrow(),
            [false],
            "an unused recipe's choice is not republished"
        );
    }
}

/// A pinned unit's choice is final at the count, so a composed recipe that
/// depends on it resolves its prerequisite on an empty cache: a defaults
/// load censuses the recipe and qualifies it against the unit's default.
#[cfg(feature = "pinned-tuning")]
#[test]
fn a_composed_recipe_finds_its_pinned_prerequisite_on_an_empty_cache() {
    struct Recipe {
        dependency: TuningKey,
        expected: NativeSpecialization,
        qualified: Rc<Cell<bool>>,
    }
    impl Recipe {
        fn point() -> PointShape {
            PointShape {
                label: "m1".into(),
                weight: 1.,
                rows: 1,
                sequences: 1,
                context: None,
                class: None,
            }
        }
    }
    impl<'a> ComposedContender<'a> for Recipe {
        fn dependencies(&self) -> Vec<TuningKey> {
            vec![self.dependency.clone()]
        }
        fn key(&self) -> String {
            "pinned-prerequisite".into()
        }
        fn label(&self) -> &'static str {
            "test_recipe"
        }
        fn cached(&self) -> Option<bool> {
            None
        }
        fn set_invocations(
            &mut self,
            _: &crate::programs::invocation_demand::InvocationCounts,
            _: &Objective,
        ) {
        }
        fn census(
            &mut self,
            _: &mut TuningInputs<'_, 'a>,
        ) -> Result<Option<ComposedCensus>, CatalogFailure> {
            Ok(Some(ComposedCensus {
                served: vec![Self::point()],
                measured: vec![Self::point()],
                seconds: vec![1e-3],
                qualification: Duration::from_millis(1),
            }))
        }
        fn qualify(
            &mut self,
            inputs: &mut TuningInputs<'_, 'a>,
            _before: Instant,
        ) -> Result<bool, CatalogFailure> {
            assert_eq!(inputs.choice(&self.dependency), Some(&self.expected));
            self.qualified.set(true);
            Ok(false)
        }
        fn conclude(&mut self, _: bool) -> Result<(), CatalogFailure> {
            Ok(())
        }
    }
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let recorder = Recorder::default();
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(name.clone(), if name == "DS" || name == "DB" { 0 } else { 8 })
        });
    let defaults = implementation.default_specialization(&statics).unwrap();
    let case = FakeCase::new(&implementation, &statics, "pinned-prerequisite");
    let dependency: TuningKey = (
        <dense_output::Entry as seismic::Entry>::NAME,
        case.bindings(),
        statics.statics().clone(),
    );
    let mut invocations = fixture_invocations(&case, &statics, LIMITS);
    invocations.units.insert(
        ("test_recipe", "pinned-prerequisite".into(), BTreeMap::new()),
        {
            let mut counts = crate::programs::invocation_demand::InvocationCounts::default();
            counts.launch(
                crate::programs::invocation_demand::Launch {
                    term: 0,
                    rows: 1,
                    sequences: 1,
                    history: 0,
                },
                1.,
            );
            counts
        },
    );
    let qualified = Rc::new(Cell::new(false));
    let recipe = || {
        Box::new(Recipe {
            dependency: dependency.clone(),
            expected: defaults.clone(),
            qualified: qualified.clone(),
        })
    };
    let _pin = pinned::defaults_on_this_thread();
    let mut tuner = Tuner::count(
        &device,
        TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: None,
            numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
            settings: TuningSettings::STANDARD,
        },
        LIMITS,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        invocations,
    );
    tuner.tune(&case, &implementation, &statics).unwrap();
    assert_eq!(tuner.register_composed(recipe()).unwrap(), None);
    // The pinned unit serves its choice from the count on, and forms and
    // searches nothing.
    assert_eq!(tuner.choice(&dependency), Some(&defaults));
    let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
    assert_eq!(case.formations.get(), 0);
    tuner.tune(&case, &implementation, &statics).unwrap();
    assert_eq!(tuner.register_composed(recipe()).unwrap(), None);
    let mut tuner = tuner.search().unwrap();
    assert!(
        qualified.get(),
        "the recipe was qualified against its prerequisite's pinned choice"
    );
    assert_eq!(
        tuner.tune(&case, &implementation, &statics).unwrap(),
        defaults
    );
    assert_eq!(tuner.register_composed(recipe()).unwrap(), Some(false));
    assert!(case.seen.borrow().is_empty() && case.stages.borrow().is_empty());
}

/// Work that runs past the budget completes, and the load succeeds with
/// every unit's record: exceeding the budget is never a failure.
#[test]
fn ordinary_and_zero_demand_units_outlasting_the_budget_fail_nothing() {
    struct LateObserver(Cell<bool>);
    impl TuningObserver for LateObserver {
        fn event(&self, event: &TuningEvent) {
            if matches!(event, TuningEvent::Finished(_)) {
                self.0.set(true);
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(name.clone(), if name == "DS" || name == "DB" { 0 } else { 8 })
        });
    for launches in [0, 1] {
        let mut case = FakeCase::new(&implementation, &statics, "late-publication");
        case.launches = launches;
        let observer = LateObserver(Cell::new(false));
        let mut tuner = Tuner::count(
            &device,
            TuningContext {
                definition: &definition,
                weights: &ZeroTuningWeights,
                observer: &observer,
                cache: None,
                numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
                settings: TuningSettings {
                    budget: Duration::from_millis(50),
                    fresh: true,
                },
            },
            LIMITS,
            TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
            fixture_invocations(&case, &statics, LIMITS),
        );
        tuner.tune(&case, &implementation, &statics).unwrap();
        let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
        tuner.tune(&case, &implementation, &statics).unwrap();
        let tuner = tuner.search().unwrap();
        assert!(
            observer.0.get(),
            "the regression exercises completion, not just admission"
        );
        assert!(tuner.started.elapsed() > Duration::from_millis(50));
        assert_eq!(tuner.conclude().len(), 1);
    }
}

#[test]
fn the_tuner_drives_a_registered_case_and_reports_progress() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let recorder = Recorder::default();
    let context = TuningContext {
        definition: &definition,
        weights: &ZeroTuningWeights,
        observer: &recorder,
        cache: None,
        numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
        settings: TuningSettings::STANDARD,
    };
    let limits = TuningLimits {
        max_rows: 64,
        max_projected_rows: 8,
        context_tokens: 256,
    };
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(
                name.clone(),
                if name == "DS" || name == "DB" {
                    0
                } else if name == "H" {
                    8
                } else {
                    16
                },
            )
        });
    let case = FakeCase::new(&implementation, &statics, "fake");
    // The count records the unit without forming or measuring anything.
    let mut tuner = Tuner::count(
        &device,
        context,
        limits,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        fixture_invocations(&case, &statics, limits),
    );
    let defaults = implementation.default_specialization(&statics).unwrap();
    assert_eq!(
        tuner.tune(&case, &implementation, &statics).unwrap(),
        defaults
    );
    assert!(case.seen.borrow().is_empty() && recorder.0.borrow().is_empty());
    // The mandatory phase forms the canonical artifact and builds nothing;
    // the walk after it still returns the defaults, now the published
    // choice.
    let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
    assert_eq!(case.formations.get(), 1);
    assert_eq!(
        tuner.tune(&case, &implementation, &statics).unwrap(),
        defaults
    );
    assert!(case.seen.borrow().is_empty());
    // The optional phase builds each point once, cheapest first, as its
    // discovery reaches it; the walk after it reads the unit's choice.
    let labels = ["m1", "m2", "m4", "m8", "m16", "m32", "m64"];
    let mut tuner = tuner.search().unwrap();
    assert_eq!(
        case.seen
            .borrow()
            .iter()
            .map(|(label, _, _)| label.clone())
            .collect::<Vec<_>>(),
        labels
    );
    let chosen = tuner.tune(&case, &implementation, &statics).unwrap();
    assert_eq!(chosen.launch_param(0, "ROWS"), Some(2));
    assert_eq!(case.seen.borrow().len(), labels.len());
    assert!(
        case.seen
            .borrow()
            .iter()
            .all(|(_, _, rotation)| *rotation == ROTATION_LAYERS)
    );
    // Discovery first, then the challenger, then its conversion.
    let mut expected = vec![seismic::Stage::Discovery; labels.len()];
    expected.extend([seismic::Stage::Measure, seismic::Stage::Promote]);
    assert_eq!(*case.stages.borrow(), expected);
    let events = recorder.0.borrow();
    // Progress counts milliseconds of the tuning time from the count.
    let total = STANDARD_TUNING_BUDGET.as_millis() as usize;
    // The mandatory phase is attributed first: forming and publishing.
    assert!(matches!(
        &events[0],
        TuningEvent::Mandatory {
            entry: "dense_output",
            ..
        }
    ));
    assert!(matches!(events[1], TuningEvent::Progress { total: all, .. } if all == total));
    assert!(matches!(
        &events[2],
        TuningEvent::Started {
            entry: "dense_output",
            points: 7,
            ..
        }
    ));
    // Every piece of work is reported with its forecast and what it took.
    let stages = events
        .iter()
        .filter_map(|event| match event {
            TuningEvent::Stage(stage) => Some(stage),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        stages.iter().map(|stage| stage.stage).collect::<Vec<_>>(),
        expected
    );
    assert!(
        stages[..labels.len()]
            .iter()
            .all(|stage| stage.forecast_seconds.is_none() && stage.unpriced == 1),
        "discovery is unpriced"
    );
    for stage in &stages[labels.len()..] {
        // Its runtime price and the publication of the unit's record.
        assert!(stage.forecast_seconds.unwrap() >= case.defaults_seconds);
        assert_eq!(stage.share, 1.);
    }
    assert_eq!(
        stages
            .iter()
            .map(|stage| stage.promoted)
            .filter(|promoted| *promoted)
            .count(),
        1
    );
    // One complete record per unit, then what became of the budget.
    let finished = events
        .iter()
        .filter_map(|event| match event {
            TuningEvent::Finished(tuned) => Some(tuned),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [tuned] = finished.as_slice() else {
        panic!("tuning reports one record per unit");
    };
    let Some(TuningEvent::Budget(budget)) = events.last() else {
        panic!("tuning reports what became of its budget");
    };
    assert_eq!(budget.budget_seconds, STANDARD_TUNING_BUDGET.as_secs_f64());
    assert_eq!(budget.stages, expected.len());
    assert_eq!(budget.overrun_seconds, 0.);
    assert!(budget.unused_seconds > 0. && budget.refused.is_empty());
    assert!((budget.used_seconds + budget.unused_seconds - budget.budget_seconds).abs() < 1e-9);
    // A unit with no work left holds no inputs: the inputs its pieces
    // built were on the device while it worked, and none of them is when
    // the optional work ends, before the tuning does.
    assert!(budget.memory.peak_bytes > budget.memory.mandatory_bytes);
    assert_eq!(budget.memory.ended_bytes, budget.memory.mandatory_bytes);
    // What the search reached is reported with the unit, not stored.
    let progress = tuned
        .progress
        .expect("a searched unit reports its progress");
    assert_eq!(progress.share, 1.);
    assert_eq!(progress.stages, expected.len());
    assert_eq!(progress.standing.cost, 0.5);
    // The defaults and the promoted challenger are measured; one is
    // rejected.
    assert_eq!(
        (tuned.measured, tuned.excluded, tuned.rejections),
        (2, 1, 1)
    );
    assert!(tuned.qualified);
    assert_eq!(tuned.stop, Some(SearchStop::Exhausted));
    assert_eq!(tuner.conclude(), [(*tuned).clone()]);
}

/// Stored tuning results (tuning spec §C2): a miss tunes and stores the
/// result; a hit offers the stored choice to runtime validation; a change to any
/// of the key's material is a miss.
#[test]
fn stored_results_are_offered_to_runtime_validation_and_changed_keys_miss() {
    let Some(device) = metal() else {
        return;
    };
    eprintln!(
        "terminal-cache reload qualification backend: {:?}",
        device.backend()
    );
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let root = std::env::temp_dir().join(format!("magnitude-tuning-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let cache = KernelCache::open(root.clone(), crate::DEFAULT_KERNEL_CACHE_BYTES).unwrap();
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(
                name.clone(),
                if name == "DS" || name == "DB" {
                    0
                } else if name == "H" {
                    8
                } else {
                    16
                },
            )
        });
    let limits = TuningLimits {
        max_rows: 64,
        max_projected_rows: 8,
        context_tokens: 256,
    };
    // One load: a count, a census and a search. Returns the unit's outcome,
    // whether its search started, and whether tuning reported progress.
    let load_with = |case: &FakeCase,
                     limits: TuningLimits,
                     settings: TuningSettings|
     -> (TunedEntry, bool, bool) {
        let recorder = Recorder::default();
        let context = TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: Some(&cache),
            numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
            settings,
        };
        let weights = || TuningWeights::new(&device, &load, &ZeroTuningWeights, &import);
        let mut tuner = Tuner::count(
            &device,
            context,
            limits,
            weights(),
            fixture_invocations(case, &statics, limits),
        );
        let formed = case.formations.get();
        tuner.tune(case, &implementation, &statics).unwrap();
        assert_eq!(
            case.formations.get(),
            formed,
            "counting cached defaults cannot compile"
        );
        let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
        tuner.tune(case, &implementation, &statics).unwrap();
        let mut tuner = tuner.search().unwrap();
        let chosen = tuner.tune(case, &implementation, &statics).unwrap();
        let tuned = tuner.conclude().pop().unwrap();
        assert_eq!(chosen, tuned.overall.specialization());
        let events = recorder.0.borrow();
        let started = events
            .iter()
            .any(|event| matches!(event, TuningEvent::Started { .. }));
        let progressed = events
            .iter()
            .any(|event| matches!(event, TuningEvent::Progress { .. }));
        (tuned, started, progressed)
    };
    let load_once =
        |case: &FakeCase, limits: TuningLimits| load_with(case, limits, TuningSettings::STANDARD);
    let stored_results = || std::fs::read_dir(root.join("tuning")).unwrap().count();

    let first = FakeCase::new(&implementation, &statics, "implementation a");
    let (searched, started, progressed) = load_once(&first, limits);
    assert_eq!(searched.origin, TuningOrigin::Searched);
    assert!(started && !first.seen.borrow().is_empty());
    assert!(progressed, "a unit without a stored result is tuned");
    assert_eq!(stored_results(), 1);

    let again = FakeCase::new(&implementation, &statics, "implementation a");
    let (stored, started, progressed) = load_once(&again, limits);
    assert_eq!(stored.origin, TuningOrigin::Stored);
    assert!(
        !started && again.seen.borrow().is_empty(),
        "a stored result is used without tuning"
    );
    assert!(
        !progressed,
        "the count finds the stored result before tuning"
    );
    assert_eq!(stored.overall, searched.overall);
    assert_eq!(stored.overall.launches[0]["ROWS"], 2);
    assert_eq!(
        (stored.measured, stored.excluded),
        (searched.measured, searched.excluded)
    );
    assert_eq!(stored.time, TuningTime::default());
    assert_eq!(stored.stop, searched.stop);
    assert_eq!(stored_results(), 1);

    // A different graph call frequency does not grant a new first-load search.
    let mut more_calls = FakeCase::new(&implementation, &statics, "implementation a");
    more_calls.launches = 7;
    let (reused, started, progressed) = load_once(&more_calls, limits);
    assert_eq!(reused.origin, TuningOrigin::Stored);
    assert_eq!(reused.overall, searched.overall);
    assert!(!started && !progressed && more_calls.seen.borrow().is_empty());
    assert_eq!(stored_results(), 1);

    // A changed implementation digest, and changed tuning points, are new
    // keys: each tunes and stores its own result.
    let changed = FakeCase::new(&implementation, &statics, "implementation b");
    assert_eq!(load_once(&changed, limits).0.origin, TuningOrigin::Searched);
    let narrower = TuningLimits {
        max_rows: 8,
        ..limits
    };
    let rebounded = FakeCase::new(&implementation, &statics, "implementation a");
    assert_eq!(
        load_once(&rebounded, narrower).0.origin,
        TuningOrigin::Searched
    );
    assert_eq!(stored_results(), 3);

    // A unit whose longest completed operation already outlasts the budget
    // starts no unpriced discovery: it keeps the default its mandatory
    // phase published, and every later load consumes that choice.
    let mut slow = FakeCase::new(&implementation, &statics, "implementation c");
    slow.defaults_seconds = STANDARD_TUNING_BUDGET.as_secs_f64();
    let (kept, started, _) = load_once(&slow, limits);
    assert!(slow.seen.borrow().is_empty() && slow.stages.borrow().is_empty());
    assert_eq!(kept.origin, TuningOrigin::Searched);
    assert!(!started);
    assert_eq!(kept.overall, slow.defaults);
    let canonical = kept.overall.specialization();
    // A final fallback needs the real declaration's complete launch maps;
    // a fake singleton ROWS map is not the same native specialization.
    let truncated = Configuration {
        launches: vec![
            [(
                "ROWS".to_owned(),
                canonical.launch_param(0, "ROWS").unwrap(),
            )]
            .into_iter()
            .collect(),
        ],
        ..kept.overall.clone()
    }
    .specialization();
    assert!(!slow.stored_valid(&device, &truncated));
    assert!(slow.stored_valid(&device, &canonical));
    assert_eq!(stored_results(), 4);
    let quick = FakeCase::new(&implementation, &statics, "implementation c");
    let (settled, started, progressed) = load_once(&quick, limits);
    assert_eq!(settled.origin, TuningOrigin::Stored);
    assert!(!started && !progressed && quick.seen.borrow().is_empty());
    assert_eq!(settled.overall, kept.overall);
    assert_eq!(stored_results(), 4);
    let mut inactive = FakeCase::new(&implementation, &statics, "implementation inactive");
    inactive.launches = 0;
    let (canonical, started, progressed) = load_once(&inactive, limits);
    assert_eq!(canonical.overall, inactive.defaults);
    assert!(
        !started && progressed,
        "canonical formation shares the first-load clock"
    );
    assert_eq!(inactive.formations.get(), 1);
    assert!(inactive.stages.borrow().is_empty());
    let mut activated = FakeCase::new(&implementation, &statics, "implementation inactive");
    activated.launches = 7;
    let (reused, started, progressed) = load_once(&activated, limits);
    assert_eq!(reused.overall, canonical.overall);
    assert_eq!(reused.origin, TuningOrigin::Stored);
    assert!(!started && !progressed);
    assert!(activated.seen.borrow().is_empty() && activated.stages.borrow().is_empty());
    assert_eq!(
        activated.formations.get(),
        0,
        "a later load is metadata only"
    );

    assert_eq!(stored_results(), 5);
    // A fresh load searches a unit whose result is stored, and replaces it.
    let fresh = FakeCase::new(&implementation, &statics, "implementation a");
    let fresh_settings = TuningSettings {
        fresh: true,
        ..TuningSettings::STANDARD
    };
    let (researched, started, progressed) = load_with(&fresh, limits, fresh_settings);
    assert_eq!(researched.origin, TuningOrigin::Searched);
    assert!(started && progressed && !fresh.seen.borrow().is_empty());
    assert_eq!(stored_results(), 5);
    // A result is read only under the budget it was searched within.
    let rebudgeted = FakeCase::new(&implementation, &statics, "implementation a");
    let half = TuningSettings {
        budget: STANDARD_TUNING_BUDGET / 2,
        ..TuningSettings::STANDARD
    };
    assert_eq!(
        load_with(&rebudgeted, limits, half).0.origin,
        TuningOrigin::Searched
    );
    assert_eq!(stored_results(), 6);
    std::fs::remove_dir_all(root).unwrap();
}

/// A unit's stored record holds every candidate's measurements, which are
/// the same under every α: a later load under another α reads the same
/// record and chooses again among the choices its search served, forming
/// and measuring nothing. A stricter α that no longer admits the promoted
/// challenger serves the choice before it; a looser α serves no candidate
/// the search rejected, since none was timed, confirmed or covered.
#[test]
fn a_stored_record_is_served_under_another_alpha_without_forming_or_measuring() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let root = std::env::temp_dir().join(format!("magnitude-tuning-alpha-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let cache = KernelCache::open(root.clone(), crate::DEFAULT_KERNEL_CACHE_BYTES).unwrap();
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(
                name.clone(),
                if name == "DS" || name == "DB" {
                    0
                } else if name == "H" {
                    8
                } else {
                    16
                },
            )
        });
    let limits = TuningLimits {
        max_rows: 64,
        max_projected_rows: 8,
        context_tokens: 256,
    };
    // One load under `alpha`: the unit's outcome, and its case, which
    // counts what the load formed, built and measured.
    let load_under = |alpha: f64| -> (TunedEntry, FakeCase) {
        let case = FakeCase::new(&implementation, &statics, "implementation a");
        let profile = super::precision::NumericalProfile {
            alpha: seismic::Limit::new(alpha).unwrap(),
        }
        .resolve()
        .unwrap();
        let recorder = Recorder::default();
        let context = TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: Some(&cache),
            numerical_profile: &profile,
            settings: TuningSettings::STANDARD,
        };
        let mut tuner = Tuner::count(
            &device,
            context,
            limits,
            TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
            fixture_invocations(&case, &statics, limits),
        );
        tuner.tune(&case, &implementation, &statics).unwrap();
        let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
        tuner.tune(&case, &implementation, &statics).unwrap();
        let mut tuner = tuner.search().unwrap();
        let chosen = tuner.tune(&case, &implementation, &statics).unwrap();
        let tuned = tuner.conclude().pop().unwrap();
        assert_eq!(chosen, tuned.overall.specialization());
        (tuned, case)
    };
    let untouched = |case: &FakeCase| {
        case.formations.get() == 0
            && case.seen.borrow().is_empty()
            && case.stages.borrow().is_empty()
    };
    let stored_results = || std::fs::read_dir(root.join("tuning")).unwrap().count();

    // The search: its challenger, a quarter of α from the defaults, is
    // promoted; another candidate, half as far again as α, is rejected.
    let (searched, first) = load_under(8.);
    assert_eq!(searched.origin, TuningOrigin::Searched);
    assert!(searched.qualified && first.formations.get() == 1);
    assert_eq!(searched.overall, first.chosen);
    let evidence = |tuned: &TunedEntry| tuned.receipt.read().unwrap();
    assert_eq!(
        (searched.rejections, evidence(&searched).reselection.clone()),
        (1, None)
    );
    assert_eq!(stored_results(), 1);

    // A stricter α than the challenger's ratio of 2: the same record, the
    // choice before the challenger, no work.
    let (strict, case) = load_under(1.);
    assert_eq!(strict.origin, TuningOrigin::Stored);
    assert!(untouched(&case), "re-selection forms and measures nothing");
    assert_eq!(strict.overall, case.defaults);
    assert!(!strict.qualified);
    // Chosen again from the evidence, which this load therefore holds.
    assert!(matches!(strict.receipt, Receipt::Held(_)));
    let strict_evidence = evidence(&strict);
    assert_eq!(strict_evidence.requirement, steps(1.));
    let reselection = strict_evidence.reselection.as_ref().unwrap();
    assert_eq!(reselection.searched, steps(8.));
    assert!(reselection.unserved.is_empty());
    // Both candidates are beyond this α; their measurements are unchanged.
    assert_eq!(strict.rejections, 2);
    assert_eq!(
        strict_evidence.observations,
        evidence(&searched).observations
    );
    assert_eq!(stored_results(), 1);

    // A looser α than the rejected candidate's ratio of 12 admits its
    // measurements, but it has no timing, confirmation or coverage: the
    // promoted challenger is still served, and the record says what was
    // left unserved.
    let (loose, case) = load_under(16.);
    assert_eq!(loose.origin, TuningOrigin::Stored);
    assert!(untouched(&case));
    assert_eq!(loose.overall, searched.overall);
    assert!(loose.qualified && loose.rejections == 0);
    let loose_evidence = evidence(&loose);
    let reselection = loose_evidence.reselection.as_ref().unwrap();
    assert_eq!(reselection.unserved.len(), 1);
    assert_eq!(reselection.unserved[0].launches[0]["ROWS"], 4);
    assert_ne!(reselection.unserved[0], loose.overall);

    // The α it was searched under again: the stored choice as it was.
    let (again, case) = load_under(8.);
    assert!(untouched(&case));
    assert_eq!(again.overall, searched.overall);
    assert_eq!(
        (again.rejections, again.measured),
        (searched.rejections, searched.measured)
    );
    // Its evidence is named, not read: the load decoded the choice alone.
    // Whoever states it reads what the search stored.
    assert!(matches!(again.receipt, Receipt::Stored(_)));
    assert_eq!(
        evidence(&again).observations,
        evidence(&searched).observations
    );
    assert_eq!(evidence(&again).reselection, None);
    assert_eq!(stored_results(), 1, "no α has a record of its own");
    std::fs::remove_dir_all(root).unwrap();
}

/// A first load whose every unit ends unqualified (its reference was
/// unavailable) stores that terminal disposition; a later load with the same
/// cache reuses it and never searches or reports tuning time.
#[test]
fn unqualified_terminal_choices_are_reused_with_their_disposition_and_without_search() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let root = std::env::temp_dir().join(format!(
        "magnitude-tuning-unqualified-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let cache = KernelCache::open(root.clone(), crate::DEFAULT_KERNEL_CACHE_BYTES).unwrap();
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(
                name.clone(),
                match name.as_str() {
                    "DS" | "DB" => 0,
                    "H" => 8,
                    _ => 16,
                },
            )
        });
    let limits = TuningLimits {
        max_rows: 64,
        max_projected_rows: 8,
        context_tokens: 256,
    };
    let load_once = |case: &FakeCase| -> (TunedEntry, bool) {
        let recorder = Recorder::default();
        let context = TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: Some(&cache),
            numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
            settings: TuningSettings::STANDARD,
        };
        let weights = || TuningWeights::new(&device, &load, &ZeroTuningWeights, &import);
        let mut tuner = Tuner::count(
            &device,
            context,
            limits,
            weights(),
            fixture_invocations(case, &statics, limits),
        );
        tuner.tune(case, &implementation, &statics).unwrap();
        let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
        tuner.tune(case, &implementation, &statics).unwrap();
        let mut tuner = tuner.search().unwrap();
        tuner.tune(case, &implementation, &statics).unwrap();
        let tuned = tuner.conclude().pop().unwrap();
        let searched = recorder.0.borrow().iter().any(|event| {
            matches!(
                event,
                TuningEvent::Started { .. } | TuningEvent::Progress { .. }
            )
        });
        (tuned, searched)
    };
    let mut first = FakeCase::new(&implementation, &statics, "implementation a");
    first.unqualified = true;
    let (tuned, searched) = load_once(&first);
    assert!(searched, "the first load tunes");
    assert_eq!(tuned.origin, TuningOrigin::Searched);
    assert!(!tuned.qualified);

    let mut again = FakeCase::new(&implementation, &statics, "implementation a");
    again.unqualified = true;
    let (stored, searched) = load_once(&again);
    assert!(
        !searched,
        "a stored unqualified choice is reused without search"
    );
    assert_eq!(stored.origin, TuningOrigin::Stored);
    assert!(!stored.qualified);
    assert_eq!(stored.overall, tuned.overall);
    // A stored disposition reports exactly what its first load decided.
    assert_eq!(super::trace::qualification(&tuned), "unqualified_rejected");
    assert_eq!(again.formations.get(), 0, "a later load is metadata only");
    assert_eq!(
        super::trace::qualification(&stored),
        super::trace::qualification(&tuned)
    );
    assert!(again.seen.borrow().is_empty());
    let _ = std::fs::remove_dir_all(&root);
}

/// The mandatory phase comes first: every canonical artifact is formed and
/// published before any input is built, whatever the budget.
#[test]
fn canonical_prerequisites_precede_every_census_and_a_spent_budget_starts_nothing_optional() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let recorder = Recorder::default();
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(name.clone(), if name == "DS" || name == "DB" { 0 } else { 8 })
        });
    let mut first = FakeCase::new(&implementation, &statics, "prerequisite-first");
    first.bindings = "first".into();
    let mut last = FakeCase::new(&implementation, &statics, "prerequisite-last");
    last.bindings = "last".into();
    let mut invocations = fixture_invocations(&first, &statics, LIMITS);
    fixture_add(&mut invocations, &last, &statics, LIMITS);
    let context = |budget| TuningContext {
        definition: &definition,
        weights: &ZeroTuningWeights,
        observer: &recorder,
        cache: None,
        numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
        settings: TuningSettings {
            budget,
            fresh: true,
        },
    };
    let mut tuner = Tuner::count(
        &device,
        context(STANDARD_TUNING_BUDGET),
        LIMITS,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        invocations.clone(),
    );
    for case in [&first, &last] {
        tuner.tune(case, &implementation, &statics).unwrap();
        assert_eq!(case.formations.get(), 0);
    }
    let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
    for case in [&first, &last] {
        assert_eq!(case.formations.get(), 1);
        assert!(
            case.seen.borrow().is_empty(),
            "no input construction during prerequisites"
        );
    }
    // The walk after the mandatory phase collects the units; it builds and
    // measures nothing either.
    for case in [&first, &last] {
        tuner.tune(case, &implementation, &statics).unwrap();
        assert!(case.seen.borrow().is_empty());
    }
    // Every unit serves its published default before any optional work.
    assert_eq!(
        tuner.chosen.len(),
        2,
        "each unit holds a served choice from the mandatory phase"
    );

    // A budget already spent starts nothing optional and fails nothing:
    // the mandatory phase still forms and publishes the unit's default.
    let mut spent = Tuner::count(
        &device,
        context(Duration::ZERO),
        LIMITS,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        invocations,
    );
    spent.tune(&last, &implementation, &statics).unwrap();
    let mut spent = spent.census(UntunedEntries::default()).unwrap();
    assert_eq!(last.formations.get(), 2);
    spent.tune(&last, &implementation, &statics).unwrap();
    let mut spent = spent.search().unwrap();
    let chosen = spent.tune(&last, &implementation, &statics).unwrap();
    assert_eq!(
        chosen,
        implementation.default_specialization(&statics).unwrap()
    );
    assert!(last.seen.borrow().is_empty() && last.stages.borrow().is_empty());
    let tuned = spent.conclude();
    assert_eq!(tuned.len(), 1);
    assert!(!tuned[0].qualified);
    let events = recorder.0.borrow();
    let Some(TuningEvent::Budget(budget)) = events.last() else {
        panic!("tuning reports what became of its budget");
    };
    assert_eq!((budget.stages, budget.refused.len()), (0, 1));
    assert_eq!(budget.unused_seconds, 0.);
    assert!(
        budget.overrun_seconds > 0.,
        "the mandatory phase alone ran past it"
    );
}

#[test]
fn the_count_forms_nothing_and_a_weight_without_its_import_entry_is_a_typed_failure() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let recorder = Recorder::default();
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(name.clone(), if name == "DS" || name == "DB" { 0 } else { 8 })
        });
    let case = DenseOutputTuning {
        down: Element::bf16(),
        activation: Element::bf16(),
        down_kind: WeightKind::DenseDown,
        scopes: vec![WeightScope::TargetSublayer(
            magnitude_family_contracts::SublayerIndex {
                block: 0,
                sublayer: 1,
            },
        )],
    };
    // A real case opens under the engine's standard caller profile.
    let standard = super::precision::NumericalProfile::standard()
        .resolve()
        .unwrap();
    let mut tuner = Tuner::count(
        &device,
        TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: None,
            numerical_profile: &standard,
            settings: TuningSettings::STANDARD,
        },
        LIMITS,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        fixture_invocations(&case, &statics, LIMITS),
    );
    // The count walk forms nothing. Entering census first forms all canonical
    // artifacts inside the clock, without importing weights.
    let formed = device.programs_formed();
    tuner.tune(&case, &implementation, &statics).unwrap();
    assert_eq!(device.programs_formed(), formed);
    // The mandatory phase forms the canonical artifact without inputs; the
    // unit's discovery builds them, its weights among them. A weight that
    // cannot be imported is a defect, not a matter of time: it is a typed
    // failure of the load.
    let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
    assert!(device.programs_formed() > formed);
    tuner.tune(&case, &implementation, &statics).unwrap();
    assert!(matches!(
        tuner.search(),
        Err(CatalogFailure::Tuning {
            entry: "dense_output",
            ..
        })
    ));

    // A unit with no workload still needs the canonical kernel. Counting it
    // cannot form that kernel before the clock, and its census needs no
    // weight imports or empirical self-comparison.
    let mut invocations = fixture_invocations(&case, &statics, LIMITS);
    for demand in invocations.units.values_mut() {
        *demand = demand.scaled(0);
    }
    let mut tuner = Tuner::count(
        &device,
        TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: None,
            numerical_profile: &standard,
            settings: TuningSettings::STANDARD,
        },
        LIMITS,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        invocations,
    );
    let formed = device.programs_formed();
    tuner.tune(&case, &implementation, &statics).unwrap();
    assert_eq!(device.programs_formed(), formed);
    assert!(tuner.tuned.is_empty());
    let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
    let formed = device.programs_formed();
    tuner.tune(&case, &implementation, &statics).unwrap();
    tuner.tune(&case, &implementation, &statics).unwrap();
    assert_eq!(
        device.programs_formed(),
        formed,
        "terminal selection forms nothing"
    );
    assert_eq!(
        tuner.tuned.len(),
        1,
        "one terminal choice across repeated calls"
    );
    let tuned = &tuner.tuned[0];
    assert!(tuned.time.forming_seconds > 0.0);
}

#[test]
fn tuning_batches_are_packed_by_the_batch_builder() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let mut weights = TuningWeights::new(&device, &load, &ZeroTuningWeights, &import);
    let mut shared = HashMap::new();
    let noise = Noise::default();
    let choices = HashMap::new();
    let mut inputs = TuningInputs {
        model: ModelInputs {
            definition: &definition,
            limits: LIMITS,
            load: &load,
        },
        device: &device,
        weights: &mut weights,
        noise: &noise,
        shared: &mut shared,
        choices: &choices,
    };
    let batch = inputs
        .batch(6, 256, &[vec![[0, 256]], vec![[256, 512]]], 512)
        .unwrap();
    assert_eq!(batch.actual_rows, 6);
    assert_eq!(batch.actual_slots, 2);
    assert_eq!(batch.class.rows(), 8);
    let history = &batch.histories[0];
    assert_eq!(&history.destinations[..6], [512, 513, 514, 515, 516, 517]);
    assert_eq!(history.visible[0][0], [0, 256]);
    assert_eq!(history.visible[3][0], [256, 512]);
    assert_eq!(batch.coordinates[3][0], 256);
    assert_eq!(&batch.bank[..2], [1, 2]);
    assert_eq!(&batch.following_bank[..2], [3, 4]);
    assert!(
        inputs
            .batch(1, 256, &[vec![[0, 256]], vec![[256, 512]]], 512)
            .is_err()
    );
    // Two requests that arrived together lie where the store places them:
    // apart, each row seeing its own request's ranges, in the span class
    // of the most ranges a request has.
    let point = across_sequences(
        &PointShape {
            context: Some(600),
            ..row_point(2, 1.)
        },
        2,
    );
    let placement =
        history_placement(&inputs, magnitude_state::KvCodec::AffineK8V4, &point).unwrap();
    assert_eq!(placement.sequences.len(), 2);
    assert_eq!(placement.sequences[0], [[0, 600]]);
    let second = &placement.sequences[1];
    assert!(second[0][0] > 600 + 256, "{second:?}");
    assert_eq!(
        second.iter().map(|range| range[1] - range[0]).sum::<i32>(),
        600
    );
    let placed = inputs
        .batch(2, 600, &placement.sequences, placement.extent)
        .unwrap();
    assert_eq!(placed.class.segments(), second.len().next_power_of_two());
    let history = &placed.histories[0];
    assert_eq!(history.visible[0][0], [0, 600]);
    assert_eq!(history.visible[1][0], second[0]);
    assert_eq!(
        history.destinations[..2],
        [placement.extent as i32, placement.extent as i32 + 1]
    );
    // One request alone holds its history in one span.
    let alone = history_placement(
        &inputs,
        magnitude_state::KvCodec::AffineK8V4,
        &PointShape {
            context: Some(600),
            ..row_point(2, 1.)
        },
    )
    .unwrap();
    assert_eq!((alone.sequences, alone.extent), (vec![vec![[0, 600]]], 600));
    let scratch = inputs.scratch(Element::f32(), &[2, 3]).unwrap();
    assert_eq!(scratch.read_to_host().unwrap(), vec![0; 24]);
    let activation = inputs.activation(Element::bf16(), &[4, 4], 1).unwrap();
    assert_eq!(activation.extents(), [4, 4]);
    let first = inputs
        .shared("arena".into(), |inputs| {
            inputs.scratch(Element::f32(), &[4])
        })
        .unwrap();
    let again = inputs
        .shared("arena".into(), |_| Err("built twice".into()))
        .unwrap();
    assert_eq!(first.extents(), again.extents());
}

/// A case state restores exactly its written rows, through every handle to
/// the storage, and leaves the rest of the tensor alone.
#[test]
fn case_state_restores_its_written_rows() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let mut weights = TuningWeights::new(&device, &load, &ZeroTuningWeights, &import);
    let mut shared = HashMap::new();
    let noise = Noise::default();
    let choices = HashMap::new();
    let inputs = TuningInputs {
        model: ModelInputs {
            definition: &definition,
            limits: LIMITS,
            load: &load,
        },
        device: &device,
        weights: &mut weights,
        noise: &noise,
        shared: &mut shared,
        choices: &choices,
    };
    let values = (0..8).map(|value| value as f32).collect::<Vec<_>>();
    let tensor = inputs.f32s(&[4, 2], &values).unwrap();
    let mut state = inputs.state(tensor, 1..3).unwrap();
    let mut other = state.share();
    let mut written = initializer(vec![&state], false).unwrap().unwrap();
    // The guard point's restorer captures the complete storage before any
    // invocation.
    let mut complete = initializer(vec![&state], true).unwrap().unwrap();
    let mut restored = |restore: &mut TuningInitializer<'static>| {
        other.tensor_mut().write_from_host(&[0u8; 32]).unwrap();
        restore().unwrap();
        state
            .tensor_mut()
            .read_to_host()
            .unwrap()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        restored(&mut written),
        [0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0]
    );
    assert_eq!(
        restored(&mut complete),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );
    assert!(initializer(Vec::new(), false).unwrap().is_none());
}

/// Baselines precede candidate comparisons, without guessing a unit's cost;
/// every operation receives the same session deadline.
#[test]
fn units_measure_baselines_before_search() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let recorder = Recorder::default();
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(
                name.clone(),
                if name == "DS" || name == "DB" {
                    0
                } else if name == "H" {
                    8
                } else {
                    16
                },
            )
        });
    let mut small = FakeCase::new(&implementation, &statics, "fake");
    small.bindings = "small".into();
    let mut large = FakeCase::new(&implementation, &statics, "fake");
    large.bindings = "large".into();
    large.launches = 3;
    let limits = TuningLimits {
        max_rows: 64,
        max_projected_rows: 8,
        context_tokens: 256,
    };
    let mut tuner = Tuner::count(
        &device,
        TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: None,
            numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
            settings: TuningSettings::STANDARD,
        },
        limits,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        {
            let mut demand = fixture_invocations(&small, &statics, limits);
            fixture_add(&mut demand, &large, &statics, limits);
            demand
        },
    );
    tuner.tune(&small, &implementation, &statics).unwrap();
    tuner.tune(&large, &implementation, &statics).unwrap();
    let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
    tuner.tune(&small, &implementation, &statics).unwrap();
    tuner.tune(&large, &implementation, &statics).unwrap();
    let mut tuner = tuner.search().unwrap();
    tuner.tune(&small, &implementation, &statics).unwrap();
    tuner.tune(&large, &implementation, &statics).unwrap();
    let pieces = recorder
        .0
        .borrow()
        .iter()
        .filter_map(|event| match event {
            TuningEvent::Stage(stage) => Some((stage.bindings.clone(), stage.stage)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let first_candidate = pieces
        .iter()
        .position(|(_, stage)| *stage != seismic::Stage::Discovery)
        .expect("a candidate is measured");
    assert!(
        pieces[first_candidate..]
            .iter()
            .all(|(_, stage)| *stage != seismic::Stage::Discovery)
    );
    assert!(
        pieces[..first_candidate]
            .iter()
            .any(|(binding, _)| binding == "small")
    );
    assert!(
        pieces[..first_candidate]
            .iter()
            .any(|(binding, _)| binding == "large")
    );
    let shares = tuner
        .conclude()
        .iter()
        .map(|tuned| (tuned.bindings.clone(), tuned.progress.unwrap().share))
        .collect::<Vec<_>>();
    assert!(
        matches!(&shares[..], [(first, small), (second, large)]
            if first == "small" && second == "large" && (large - 0.75).abs() < 1e-9 && (small - 0.25).abs() < 1e-9),
        "{shares:?}"
    );
}

/// A cache can contain a choice that exceeds Metal's shared-memory limit at
/// a served span class, which the declaration no longer admits. Reading
/// that record must fail without reopening the search; a valid replacement
/// keeps the key.
#[test]
fn stored_attention_choice_must_admit_every_served_span_class() {
    use super::attention::{AttentionMix, AttentionPrefillK8V4Tuning, PrefillListing};
    let Some(device) = metal() else { return };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let case = AttentionPrefillK8V4Tuning(AttentionMix {
        codec: magnitude_state::KvCodec::AffineK8V4.spec(seismic::DType::BF16, 128, 128),
        listing: PrefillListing::Listed { capacity: 65536 },
        activation: Element::bf16(),
        shape: crate::AttentionShape {
            hidden: 2048,
            kv_heads: 2,
            group: 16,
            rotary_pairs: 32,
            width: 128,
            interleaved_gate: 0,
            separate_gate: 0,
            fresh: 1,
            head_norm: 1,
            value_norm: 0,
            projected_value: true,
        },
        scopes: Vec::new(),
        epsilon: 1.0e-5,
        decode_rows: None,
        window: None,
        selection: None,
    });
    let implementation = seismic::generated::native_implementation::<
        magnitude_kernels::attention_prefill_k8v4::Entry,
    >(&device)
    .unwrap()
    .unwrap();
    let dimensions: BTreeMap<_, _> = case
        .0
        .shape
        .mix_dimensions(64, 4352, 1)
        .into_iter()
        .chain([("L", 1), ("HT", 16), ("B", 0)])
        .chain(crate::operators::attention::graph::affine_dimensions(case.0.codec).unwrap())
        .map(|(name, value)| (name.to_owned(), value))
        .collect();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(name.clone(), dimensions[name])
        });
    let defaults = implementation.default_specialization(&statics).unwrap();
    let invalid = defaults
        .clone()
        .with_param("QT", 16)
        .with_param("HEADS", 2)
        .with_param("SPLIT_GROUPS", 256)
        .with_param("DIRECT", 1)
        .with_param("COISSUE", 1);
    // The declaration charges the largest segment class, so it refuses the
    // choice outright: four pairs need 33,328 threadgroup bytes there.
    assert!(!case.stored_valid(&device, &invalid));
    assert!(case.entry().prepare(&device, &invalid).is_err());

    let root = std::env::temp_dir().join(format!("magnitude-span-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let cache = KernelCache::open(root.clone(), crate::DEFAULT_KERNEL_CACHE_BYTES).unwrap();
    let recorder = Recorder::default();
    let limits = TuningLimits {
        max_rows: 64,
        max_projected_rows: 8,
        context_tokens: 4096,
    };
    let context = TuningContext {
        definition: &definition,
        weights: &ZeroTuningWeights,
        observer: &recorder,
        cache: Some(&cache),
        numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
        settings: TuningSettings::STANDARD,
    };
    let tuner = Tuner::count(
        &device,
        context,
        limits,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        fixture_invocations(&case, &statics, limits),
    );
    let points = case.points(limits);
    let (_, key, hit) = tuner.stored(&case, &statics, &points).unwrap().unwrap();
    assert!(hit.is_none());
    let configuration = |spec: &NativeSpecialization| Configuration {
        statics: spec.statics().clone(),
        params: spec.params().clone(),
        launches: Vec::new(),
    };
    let mut record = TuningResult {
        disposition: seismic::TuningDisposition::QualifiedCandidate,
        tuning_identity: device.tuning_identity(),
        entry: "attention_prefill_k8v4".into(),
        backend: "metal".into(),
        points: Vec::new(),
        requirement: context.numerical_profile.requirement(&case.entry()),
        measurement_identity: String::new(),
        observations: Vec::new(),
        served: vec![configuration(&defaults)],
        reselection: None,
        implementation_identity: String::new(),
        parameters: Vec::new(),
        configurations: Vec::new(),
        overall: configuration(&invalid),
        method: TuningMethod::Census,
        time: TuningTime::default(),
        coverage: None,
    };
    cache.store_tuning(&key, &StoredChoice::of(&record), &record);
    assert!(
        cache.tuning::<StoredChoice>(&key).is_some(),
        "the stored choice parses"
    );
    assert!(
        tuner
            .stored(&case, &statics, &points)
            .err()
            .expect("invalid terminal choice must fail")
            .contains("search cannot reopen"),
        "an invalid terminal choice must fail without another tuning allowance"
    );
    record.overall = configuration(&defaults);
    cache.store_tuning(&key, &StoredChoice::of(&record), &record);
    assert!(
        tuner
            .stored(&case, &statics, &points)
            .err()
            .expect("unqualified demanded choice fails")
            .contains("lacks matching numerical evidence")
    );
    // Geometry remains reusable for an inactive unit without observations.
    let inactive_points = points
        .iter()
        .cloned()
        .map(|mut point| {
            point.weight = 0.;
            point
        })
        .collect::<Vec<_>>();
    let hit = tuner
        .stored(&case, &statics, &inactive_points)
        .unwrap()
        .unwrap()
        .2
        .unwrap();
    assert_eq!(
        hit.0.overall, record.overall,
        "the valid replacement remains a cache hit"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn full_context_demand_is_nonnegative_and_covers_the_actual_longest_history() {
    let limits = TuningLimits {
        context_tokens: 262144,
        ..LIMITS
    };
    let points = with_contexts(limits, None, served_row_points(limits, |rows| rows == 1));
    assert!(
        points
            .iter()
            .all(|point| point.weight.is_finite() && point.weight >= 0.)
    );
    assert!(points.iter().any(|point| point.context == Some(262144)));
    assert!((points.iter().map(|point| point.weight).sum::<f64>() - 1.).abs() < 1e-12);
}
#[test]
fn sliding_demand_clamps_to_the_physical_history_window() {
    let limits = TuningLimits {
        context_tokens: 262144,
        ..LIMITS
    };
    let points = with_contexts(
        limits,
        Some(1024),
        served_row_points(limits, |rows| rows == 1),
    );
    assert!(
        points
            .iter()
            .all(|point| point.weight.is_finite() && point.weight >= 0.)
    );
    assert!(points.iter().all(|point| point.context.unwrap() <= 1024));
    assert!((points.iter().map(|point| point.weight).sum::<f64>() - 1.).abs() < 1e-12);
}

/// A load never searches. With every record stored it prepares from them,
/// in either mode. At a unit without a record it stops unprepared when it
/// reports, and serves the canonical default when it serves defaults,
/// forming, measuring and publishing nothing: the search that follows
/// still finds the unit. Fresh tuning makes the reporting load report every
/// unit; the load that serves defaults reads what is stored.
#[test]
fn a_load_reads_records_and_reports_or_serves_defaults_without_searching() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
        packing_bound: HashMap::new(),
    };
    let root = std::env::temp_dir().join(format!("magnitude-tuning-serve-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let cache = KernelCache::open(root.clone(), crate::DEFAULT_KERNEL_CACHE_BYTES).unwrap();
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(
                name.clone(),
                if name == "DS" || name == "DB" {
                    0
                } else if name == "H" {
                    8
                } else {
                    16
                },
            )
        });
    let limits = TuningLimits {
        max_rows: 64,
        max_projected_rows: 8,
        context_tokens: 256,
    };
    let defaults = implementation.default_specialization(&statics).unwrap();
    let invocations = |cases: &[&FakeCase]| {
        cases
            .iter()
            .fold(
                None,
                |demand: Option<super::super::specialization::InvocationDemand>, case| {
                    Some(match demand {
                        None => fixture_invocations(*case, &statics, limits),
                        Some(mut demand) => {
                            fixture_add(&mut demand, *case, &statics, limits);
                            demand
                        }
                    })
                },
            )
            .unwrap_or_default()
    };
    // One load of `cases`, in walk order: each unit's choice or the load's
    // stop, the units served unprepared, and the load's outcomes and events.
    let load_of = |cases: &[&FakeCase],
                   cache: Option<&KernelCache>,
                   unprepared: UnpreparedUnits,
                   fresh: bool| {
        let recorder = Recorder::default();
        let mut tuner = Tuner::serve(
            &device,
            TuningContext {
                definition: &definition,
                weights: &ZeroTuningWeights,
                observer: &recorder,
                cache,
                numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
                settings: TuningSettings {
                    fresh,
                    ..TuningSettings::STANDARD
                },
            },
            limits,
            TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
            invocations(cases),
            unprepared,
        );
        let chosen = cases
            .iter()
            .map(|case| tuner.tune(*case, &implementation, &statics))
            .collect::<Result<Vec<_>, _>>();
        let served_unprepared = tuner.unprepared();
        let tuned = tuner.conclude();
        let reported = recorder
            .0
            .borrow()
            .iter()
            .filter(|event| matches!(event, TuningEvent::Unprepared { .. }))
            .count();
        (chosen, served_unprepared, tuned, reported)
    };
    // The search of `case`: what a Prepare runs.
    let search = |case: &FakeCase| {
        let mut tuner = Tuner::count(
            &device,
            TuningContext {
                definition: &definition,
                weights: &ZeroTuningWeights,
                observer: &UnreportedTuning,
                cache: Some(&cache),
                numerical_profile: super::precision::ResolvedNumericalProfile::standard(),
                settings: TuningSettings::STANDARD,
            },
            limits,
            TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
            invocations(&[case]),
        );
        tuner.tune(case, &implementation, &statics).unwrap();
        let mut tuner = tuner.census(UntunedEntries::default()).unwrap();
        tuner.tune(case, &implementation, &statics).unwrap();
        let mut tuner = tuner.search().unwrap();
        // The search is over: the published choice is read, nothing more.
        let chosen = tuner.tune(case, &implementation, &statics).unwrap();
        let tuned = tuner.conclude().pop().unwrap();
        assert_eq!(tuned.origin, TuningOrigin::Searched);
        chosen
    };
    let stored_results =
        || std::fs::read_dir(root.join("tuning")).map_or(0, |entries| entries.count());
    let untouched = |case: &FakeCase| {
        case.formations.get() == 0
            && case.seen.borrow().is_empty()
            && case.stages.borrow().is_empty()
    };
    let unit = |digest: &str, bindings: &str| {
        let mut case = FakeCase::new(&implementation, &statics, digest);
        case.bindings = bindings.into();
        case
    };

    // No records: the reporting load stops at the unit, having formed,
    // measured and published nothing.
    let first = unit("served", "first");
    let (chosen, served_unprepared, tuned, reported) =
        load_of(&[&first], Some(&cache), UnpreparedUnits::Report, false);
    assert!(matches!(
        chosen,
        Err(CatalogFailure::Unprepared { entry, ref bindings })
            if entry == <dense_output::Entry as seismic::Entry>::NAME && bindings == "first"
    ));
    assert_eq!((served_unprepared, tuned.len(), reported), (0, 0, 0));
    assert!(untouched(&first));
    assert_eq!(stored_results(), 0);
    // The load that serves defaults is total, with or without a cache, and
    // still publishes nothing.
    for cache in [Some(&cache), None] {
        let (chosen, served_unprepared, tuned, reported) =
            load_of(&[&first], cache, UnpreparedUnits::ServeDefaults, false);
        assert_eq!(chosen.unwrap(), [defaults.clone()]);
        assert_eq!((served_unprepared, tuned.len(), reported), (1, 0, 1));
        assert!(untouched(&first));
        assert_eq!(stored_results(), 0);
    }

    // The search publishes the unit's record: complete records. Both modes
    // read it and are the same load.
    let searched = search(&first);
    assert_ne!(searched, defaults);
    assert_eq!(stored_results(), 1);
    for unprepared in [UnpreparedUnits::Report, UnpreparedUnits::ServeDefaults] {
        let again = unit("served", "first");
        let (chosen, served_unprepared, tuned, reported) =
            load_of(&[&again], Some(&cache), unprepared, false);
        assert_eq!(chosen.unwrap(), [searched.clone()]);
        assert_eq!((served_unprepared, tuned.len(), reported), (0, 1, 0));
        assert_eq!(tuned[0].origin, TuningOrigin::Stored);
        assert!(untouched(&again));
    }

    // Partial records: a second unit has none. The reporting load stops at
    // it; the load that serves defaults serves the first from its record
    // and the second at its default.
    let (again, second) = (unit("served", "first"), unit("served", "second"));
    let (chosen, ..) = load_of(
        &[&again, &second],
        Some(&cache),
        UnpreparedUnits::Report,
        false,
    );
    assert!(matches!(
        chosen,
        Err(CatalogFailure::Unprepared { ref bindings, .. }) if bindings == "second"
    ));
    let (chosen, served_unprepared, tuned, reported) = load_of(
        &[&again, &second],
        Some(&cache),
        UnpreparedUnits::ServeDefaults,
        false,
    );
    assert_eq!(chosen.unwrap(), [searched.clone(), defaults.clone()]);
    assert_eq!((served_unprepared, tuned.len(), reported), (1, 1, 1));
    assert!(untouched(&again) && untouched(&second));
    // The load published nothing for the second unit: a search still finds
    // it unsearched.
    assert_eq!(stored_results(), 1);
    search(&second);
    assert_eq!(stored_results(), 2);

    // Fresh tuning: the reporting load reads no record, so the host's
    // search runs again; the load that serves defaults reads the records.
    let (chosen, ..) = load_of(&[&again], Some(&cache), UnpreparedUnits::Report, true);
    assert!(matches!(chosen, Err(CatalogFailure::Unprepared { .. })));
    let (chosen, served_unprepared, ..) = load_of(
        &[&again],
        Some(&cache),
        UnpreparedUnits::ServeDefaults,
        true,
    );
    assert_eq!(chosen.unwrap(), [searched]);
    assert_eq!(served_unprepared, 0);
    std::fs::remove_dir_all(root).unwrap();
}

/// A canonical fallback stored on CPU carries the Seismic-owned worker and
/// tier parameters; the default it is checked against is the device's, not
/// the bare declaration's, or every later Prepare refuses to reopen it.
#[test]
fn stored_cpu_fallback_matches_the_device_default() {
    let device = seismic::Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let implementation = seismic::generated::native_implementation::<shape_rows::Entry>(&device)
        .unwrap()
        .unwrap();
    let declared = implementation
        .default_specialization(&NativeSpecialization::new())
        .unwrap();
    let stored = declared
        .clone()
        .with_param("cpu.tier", 0)
        .with_param("cpu.workers.0", 0)
        .with_param("cpu.workers.1", 0);
    assert_ne!(stored, declared);
    let case = super::readout::ShapeRowsTuning { vocabulary: 1024 };
    assert_eq!(
        case.device_default(&device, &NativeSpecialization::new()),
        Some(stored)
    );
}
