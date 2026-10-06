use super::cases::DenseOutputTuning;
use super::*;
use crate::native::import::ImportKernels;
use crate::planning::tests::{fixture_definition, fixture_manifest};
use crate::{ComponentSelection, ModelLoadPlan};
use magnitude_kernels::{dense_output, shape_rows};
use seismic::{BackendName, ConfigurationRecord, DeviceCatalog, Exclusion, Outcome};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const LIMITS: TuningLimits = TuningLimits {
    max_rows: 512,
    max_projected_rows: 8,
    context_tokens: 16384,
};

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
    let device = DeviceCatalog::discover()
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

#[test]
fn row_points_follow_the_shape_ladder_and_normalize_weights() {
    let points = row_points(LIMITS);
    assert_eq!(
        points.iter().map(|point| point.rows).collect::<Vec<_>>(),
        TUNING_ROWS
    );
    let total = points.iter().map(|point| point.weight).sum::<f64>();
    assert!((total - 1.0).abs() < 1e-12);
    // Each served row class retains its own share of step time.
    let weights = points.iter().map(|point| point.weight).collect::<Vec<_>>();
    for (weight, expected) in weights.iter().zip([
        0.40,
        0.20 / 3.0,
        0.20 / 3.0,
        0.20 / 3.0,
        0.05,
        0.05,
        0.075,
        0.075,
        0.075,
        0.075,
    ]) {
        assert!((weight - expected).abs() < 1e-12, "{weights:?}");
    }
    let bounded = row_points(TuningLimits {
        max_rows: 32,
        ..LIMITS
    });
    assert_eq!(
        bounded.iter().map(|point| point.rows).collect::<Vec<_>>(),
        [1, 2, 4, 8, 16, 32]
    );
    assert!((bounded.iter().map(|point| point.weight).sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn the_tuning_cache_key_is_what_a_result_is_valid_for() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let policy = seismic::PrecisionPolicy::bounded(seismic::precision::Tolerance {
        absolute: seismic::precision::Limit::new(0.01).unwrap(),
        relative: seismic::precision::Limit::new(0.01).unwrap(),
        relative_floor: seismic::precision::Limit::ZERO,
        ulps: None,
    });
    let key = |shapes: &[PointShape]| {
        tuning_key_material(
            &device,
            "dense_output",
            "A=bf16",
            &NativeSpecialization::new(),
            "same-implementation",
            &policy,
            &BTreeMap::new(),
            shapes,
        )
    };
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
    // An admitted error class the entry declares, and its envelope, are
    // what a choice was searched and validated under.
    let admitting = |limit: f64| {
        let limit = seismic::precision::Limit::new(limit).unwrap();
        tuning_key_material(
            &device,
            "dense_output",
            "A=bf16",
            &NativeSpecialization::new(),
            "same-implementation",
            &policy,
            &[(
                "int8_activations".to_owned(),
                seismic::ErrorEnvelope {
                    relative_rms: limit,
                    peak: limit,
                },
            )]
            .into_iter()
            .collect(),
            &shapes,
        )
    };
    assert_ne!(admitting(1e-2), key(&shapes));
    assert_ne!(admitting(1e-2), admitting(2e-2));
    assert!(admitting(1e-2).starts_with(&key(&shapes)));
}

#[test]
fn the_census_times_the_costliest_point_of_each_chunk_class() {
    let labels = |shapes: &[PointShape]| {
        shapes
            .iter()
            .zip(census_points(shapes))
            .filter(|(_, census)| *census)
            .map(|(shape, _)| shape.label.clone())
            .collect::<Vec<_>>()
    };
    // Every streaming row count, then the most rows of the short chunks and
    // of the prefill chunks.
    assert_eq!(
        labels(&row_points(LIMITS)),
        ["m1", "m2", "m4", "m8", "m32", "m512"]
    );
    // An entry bounded below the largest chunk is timed at its own largest.
    assert_eq!(
        labels(&served_row_points(256, |rows| rows >= 32)),
        ["m32", "m256"]
    );
    // Attention: the most rows at the longest history.
    let chunks = served_row_points(512, |rows| rows >= 16);
    assert_eq!(
        labels(&with_contexts(LIMITS, chunks)),
        ["m32-c16384", "m512-c16384"]
    );
}

#[test]
fn a_conclusion_is_estimated_from_what_the_start_leaves_to_confirm() {
    let settings = search_settings(BackendName::Cuda);
    let samples = (settings.confirmation_samples + 1) as u32;
    // One-row decode attention of one key/value head and sixteen query heads
    // of 512 columns at up to 65536 rows of history, on a GB10: the defaults'
    // sample over the census's points takes 1.21 s and their pass 5.81 s; the
    // other form's start measures in a fiftieth of that.
    let sample = Duration::from_millis(1210);
    let pass = Duration::from_millis(5810);
    // The defaults and the one form start.
    let conclusion = started_conclusion(sample, 1, false, &settings);
    assert_eq!(conclusion, sample * 2 * samples);
    // The start with its form start and that conclusion fit the tuning
    // time with room for the units after it.
    assert!(pass * 2 + conclusion < TUNING_TIME.mul_f64(0.6));
    // Without form starts the first refinement finds the first rival.
    assert_eq!(started_conclusion(sample, 0, false, &settings), conclusion);
    // No more rivals are confirmed than the settings' finalists.
    assert_eq!(
        started_conclusion(sample, 9, false, &settings),
        sample * (settings.confirmed as u32 + 1) * samples
    );
    // A launch-scoped search reserves them all from its start.
    assert_eq!(
        started_conclusion(sample, 1, true, &settings),
        sample * (settings.confirmed as u32 + 1) * samples
    );
}

#[test]
fn a_unit_serving_only_chunks_holds_its_part_of_the_prefill_classes() {
    // One unit serves every row count, one only chunks. The census timed
    // each at its census points: the first takes a second everywhere, the
    // second three.
    let every = row_points(LIMITS);
    let chunks = served_row_points(512, |rows| rows >= 16);
    let timed = |shapes: &[PointShape]| {
        shapes
            .iter()
            .zip(census_points(shapes))
            .filter(|(_, census)| *census)
            .map(|(shape, _)| shape.clone())
            .collect::<Vec<_>>()
    };
    let (every_timed, chunks_timed) = (timed(&every), timed(&chunks));
    let shares = step_shares(&[
        UnitTime {
            launches: 1,
            served: &every,
            shapes: &every_timed,
            seconds: &vec![1.; every_timed.len()],
        },
        UnitTime {
            launches: 1,
            served: &chunks,
            shapes: &chunks_timed,
            seconds: &vec![3.; chunks_timed.len()],
        },
    ]);
    // The streaming classes (0.6 of the step) are the first unit's alone;
    // the chunk classes (0.4) split one to three.
    assert!((shares[0] - 0.7).abs() < 1e-9, "{shares:?}");
    assert!((shares[1] - 0.3).abs() < 1e-9, "{shares:?}");
    // Timed at its cheapest point alone, as a census of required points
    // left it, the second unit held only its part of that one class.
    let cheapest = [chunks[0].clone()];
    let shares = step_shares(&[
        UnitTime {
            launches: 1,
            served: &every,
            shapes: &every_timed,
            seconds: &vec![1.; every_timed.len()],
        },
        UnitTime {
            launches: 1,
            served: &chunks,
            shapes: &cheapest,
            seconds: &[3.],
        },
    ]);
    assert!(shares[1] < 0.3, "{shares:?}");
}

#[test]
fn point_cost_grows_with_rows_and_history() {
    let point = |rows: u64, context: Option<u64>| PointShape {
        label: String::new(),
        weight: 1.0,
        rows,
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
        "64k exceeds the served context"
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
        [0]
    );
}

#[test]
fn served_points_keep_an_entry_whose_rows_exceed_the_bound() {
    let chunked = served_row_points(512, |rows| rows >= 16);
    assert_eq!(
        chunked.iter().map(|point| point.rows).collect::<Vec<_>>(),
        [16, 32, 64, 128, 256, 512]
    );
    let decode = served_row_points(512, |rows| rows <= 8);
    assert_eq!(
        decode.iter().map(|point| point.rows).collect::<Vec<_>>(),
        [1, 2, 4, 8]
    );
    // A bound admits only the served rows at or below it.
    let short = served_row_points(16, |rows| rows >= 16);
    assert_eq!(
        short.iter().map(|point| point.rows).collect::<Vec<_>>(),
        [16]
    );
    assert!((chunked.iter().map(|point| point.weight).sum::<f64>() - 1.0).abs() < 1e-12);
    let stand_in = served_row_points(8, |rows| rows >= 16);
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
    let rows = |rows| served_row_points(512, move |served| served == rows).remove(0);
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
    DeviceCatalog::discover()
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
/// "search" builds every point it is given at its census and at its start
/// and reports a fixed table choosing `chosen`.
#[derive(Clone)]
struct FakeCase {
    /// Every point built: its label, weight and rotation length.
    seen: Rc<RefCell<Vec<(String, f64, usize)>>>,
    /// The unit's share of the tuning time, as every search the case
    /// started was given it to admit points with.
    allowances: Rc<RefCell<Vec<Duration>>>,
    /// The implementation digest the case reports.
    digest: String,
    bindings: String,
    launches: usize,
    /// What the census reports the defaults take at every point.
    defaults_seconds: f64,
    chosen: Configuration,
}

impl FakeCase {
    /// A case choosing the defaults with `ROWS` 2 (admissible, not the
    /// default).
    fn new(
        implementation: &NativeImplementation,
        statics: &NativeSpecialization,
        digest: &str,
    ) -> Self {
        let chosen = implementation
            .default_specialization(statics)
            .unwrap()
            .with_launch_param(0, "ROWS", 2);
        Self {
            seen: Rc::new(RefCell::new(Vec::new())),
            allowances: Rc::new(RefCell::new(Vec::new())),
            digest: digest.into(),
            bindings: "fake".into(),
            launches: 1,
            defaults_seconds: 1e-3,
            chosen: Configuration {
                statics: chosen.statics().clone(),
                params: chosen.params().clone(),
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
                                    chosen.launch_param(ordinal, &parameter.name).unwrap(),
                                )
                            })
                            .collect()
                    })
                    .collect(),
            },
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
    fn precision(&self) -> Result<PrecisionPolicy, TuneError> {
        Ok(PrecisionPolicy::Exact)
    }
    type Entry = dense_output::Entry;
    type Case = FakeArgs;

    fn launches(&self) -> usize {
        self.launches
    }

    fn bindings(&self) -> String {
        self.bindings.clone()
    }
    fn statics(&self, _inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(vec![("H", 8), ("F", 16), ("DS", 0)])
    }
    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }
    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<FakeArgs>, String> {
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
        }
    }

    #[cfg(feature = "tuning-survey")]
    fn tune(
        &self,
        _device: &Device,
        _statics: &NativeSpecialization,
        _points: &mut dyn seismic::PointSource<'_, Self::Entry>,
        _validation: seismic::TuningPrecision,
        strategy: Strategy,
    ) -> Result<TuningResult, TuneError> {
        panic!("the fake case is searched, not tuned by {strategy:?}")
    }

    fn open(
        &self,
        _device: &Device,
        statics: &NativeSpecialization,
        validation: seismic::TuningPrecision,
    ) -> Result<Box<dyn EntrySearch<Self::Entry>>, TuneError> {
        assert_eq!(&self.chosen.statics, statics.statics());
        Ok(Box::new(FakeSearch {
            case: self.clone(),
            policy: validation.policy,
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

/// The search of a [`FakeCase`]: finished by its start.
struct FakeSearch {
    case: FakeCase,
    policy: PrecisionPolicy,
}

impl FakeSearch {
    /// Build every point and record it.
    fn build(
        &self,
        points: &mut dyn seismic::PointSource<'static, dense_output::Entry>,
    ) -> Vec<seismic::PointSpec> {
        let specs = points.points();
        for (index, spec) in specs.iter().enumerate() {
            let built = points
                .build(index, Duration::MAX)
                .unwrap_or_else(|unavailable| panic!("{unavailable:?}"));
            self.case
                .seen
                .borrow_mut()
                .push((spec.label.clone(), spec.weight, built.rotation.len()));
        }
        specs
    }

    fn result(
        &self,
        points: Vec<seismic::PointRecord>,
        configurations: Vec<ConfigurationRecord>,
        overall: Configuration,
        method: TuningMethod,
    ) -> TuningResult {
        TuningResult {
            tuning_identity: "fake-device".into(),
            entry: "dense_output".into(),
            backend: "metal".into(),
            points,
            validation: self.policy.clone(),
            numerical_evidence: Vec::new(),
            implementation_identity: String::new(),
            parameters: Vec::new(),
            configurations,
            overall,
            method,
            time: TuningTime::default(),
        }
    }
}

impl EntrySearch<dense_output::Entry> for FakeSearch {
    /// The defaults measured at every point, at the case's time each.
    fn census(
        &mut self,
        points: &mut dyn seismic::PointSource<'static, dense_output::Entry>,
        _plan: CensusPlan,
    ) -> Result<TuningResult, TuneError> {
        let specs = self.build(points);
        let defaults = Configuration {
            launches: vec![[("ROWS".to_owned(), 8)].into_iter().collect()],
            ..self.case.chosen.clone()
        };
        Ok(self.result(
            specs
                .iter()
                .map(|spec| seismic::PointRecord {
                    label: spec.label.clone(),
                    weight: 1.0,
                    class: None,
                })
                .collect(),
            vec![ConfigurationRecord {
                configuration: defaults.clone(),
                outcome: Outcome::Measured {
                    artifact: "a".into(),
                    points: specs
                        .iter()
                        .map(|spec| seismic::PointMeasurement {
                            point: spec.label.clone(),
                            key: seismic::PointKey {
                                launches: vec![0],
                                values: Default::default(),
                            },
                            median_seconds: self.case.defaults_seconds,
                            deviation_seconds: 0.0,
                            samples: vec![self.case.defaults_seconds],
                            repetitions: 1,
                            rotation_bytes: 0,
                        })
                        .collect(),
                    confirmed: Vec::new(),
                    validated: true,
                },
            }],
            defaults,
            TuningMethod::Census,
        ))
    }

    fn start(
        &mut self,
        points: &mut dyn seismic::PointSource<'static, dense_output::Entry>,
        plan: StartPlan,
        _until: Instant,
    ) -> Result<Standing, TuneError> {
        self.build(points);
        self.case
            .allowances
            .borrow_mut()
            .push(plan.admission * ADMISSION_SHARE);
        Ok(Standing {
            finished: true,
            cost: 1.,
            measured: 2,
            admissible: 2,
            programs: 0,
            declared_programs: 0,
            forming_seconds: 0.,
            cut: false,
        })
    }

    fn refine(&mut self, _slice: Instant, _until: Instant) -> Result<Standing, TuneError> {
        panic!("the fake search is finished by its start")
    }

    fn reserve(&self) -> Duration {
        Duration::ZERO
    }

    fn conclude(
        self: Box<Self>,
        points: &mut dyn seismic::PointSource<'static, dense_output::Entry>,
        _until: Instant,
        allowance: Duration,
    ) -> Result<TuningResult, TuneError> {
        let mut rejected = self.case.chosen.clone();
        rejected.launches[0].insert("ROWS".into(), 4);
        Ok(self.result(
            points
                .points()
                .iter()
                .map(|point| seismic::PointRecord {
                    label: point.label.clone(),
                    weight: point.weight,
                    class: point.class.clone(),
                })
                .collect(),
            vec![
                ConfigurationRecord {
                    configuration: self.case.chosen.clone(),
                    outcome: Outcome::Measured {
                        artifact: "a".into(),
                        points: Vec::new(),
                        confirmed: Vec::new(),
                        validated: true,
                    },
                },
                ConfigurationRecord {
                    configuration: rejected,
                    outcome: Outcome::Excluded(Exclusion::Validation {
                        point: "m1".into(),
                        detail: "differs".into(),
                    }),
                },
            ],
            self.case.chosen.clone(),
            TuningMethod::Search {
                allowance_seconds: allowance.as_secs_f64(),
                settings: SEARCH_SETTINGS,
                stop: SearchStop::Exhausted,
            },
        ))
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
    };
    let recorder = Recorder::default();
    let context = TuningContext {
        definition: &definition,
        weights: &ZeroTuningWeights,
        observer: &recorder,
        cache: None,
        error_classes: &NO_ERROR_CLASSES,
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
                if name == "DS" {
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
    );
    let defaults = implementation.default_specialization(&statics).unwrap();
    assert_eq!(
        tuner.tune(&case, &implementation, &statics).unwrap(),
        defaults
    );
    assert!(case.seen.borrow().is_empty() && recorder.0.borrow().is_empty());
    // The census builds the unit's points, cheapest first, and returns the
    // defaults.
    let mut tuner = tuner.census();
    assert_eq!(
        tuner.tune(&case, &implementation, &statics).unwrap(),
        defaults
    );
    let labels = ["m1", "m2", "m4", "m8", "m16", "m32", "m64"];
    assert_eq!(
        case.seen
            .borrow()
            .iter()
            .map(|(label, _, _)| label.clone())
            .collect::<Vec<_>>(),
        labels
    );
    // The search gets the points again, built once: from the census. The
    // walk after it reads the unit's choice.
    let mut tuner = tuner.search().unwrap();
    assert_eq!(case.seen.borrow().len(), 2 * labels.len());
    let chosen = tuner.tune(&case, &implementation, &statics).unwrap();
    assert_eq!(chosen.launch_param(0, "ROWS"), Some(2));
    assert_eq!(case.seen.borrow().len(), 2 * labels.len());
    assert!(case
        .seen
        .borrow()
        .iter()
        .all(|(_, _, rotation)| *rotation == ROTATION_LAYERS));
    // The only unit's share of step time is all of it.
    let allowance = case.allowances.borrow()[0];
    assert!(allowance <= TUNING_TIME && allowance > TUNING_TIME - Duration::from_millis(1));
    let events = recorder.0.borrow();
    // Progress counts milliseconds of the tuning time from the census.
    let total = TUNING_TIME.as_millis() as usize;
    assert_eq!(
        events[0],
        TuningEvent::Progress {
            completed: 0,
            total
        }
    );
    assert!(matches!(events[1], TuningEvent::Progress { .. }));
    assert!(matches!(
        &events[2],
        TuningEvent::Started {
            entry: "dense_output",
            points: 7,
            ..
        }
    ));
    assert!(matches!(events[3], TuningEvent::Progress { .. }));
    let TuningEvent::Finished(tuned) = &events[4] else {
        panic!("tuning reports completion");
    };
    assert!(matches!(events[5], TuningEvent::Progress { .. }));
    // What the search reached is reported with the unit, not stored.
    let progress = tuned.progress.expect("a searched unit reports its progress");
    assert_eq!(progress.share, 1.);
    assert!(progress.standing.is_some_and(|standing| standing.finished));
    assert_eq!(
        (tuned.measured, tuned.excluded, tuned.rejections),
        (1, 1, 1)
    );
    assert!(matches!(tuned.search, Some((_, SearchStop::Exhausted))));
    assert_eq!(tuner.tuned(), [tuned.clone()]);
}

/// Stored tuning results (tuning spec §C2): a miss tunes and stores the
/// result; a hit offers the stored choice to runtime validation; a change to any
/// of the key's material is a miss.
#[test]
fn stored_results_are_offered_to_runtime_validation_and_changed_keys_miss() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
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
                if name == "DS" {
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
    let load_once = |case: &FakeCase, limits: TuningLimits| -> (TunedEntry, bool, bool) {
        let recorder = Recorder::default();
        let context = TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: Some(&cache),
            error_classes: &NO_ERROR_CLASSES,
        };
        let weights = || TuningWeights::new(&device, &load, &ZeroTuningWeights, &import);
        let mut tuner = Tuner::count(&device, context, limits, weights());
        tuner.tune(case, &implementation, &statics).unwrap();
        let mut tuner = tuner.census();
        tuner.tune(case, &implementation, &statics).unwrap();
        let mut tuner = tuner.search().unwrap();
        let chosen = tuner.tune(case, &implementation, &statics).unwrap();
        let tuned = tuner.tuned().pop().unwrap();
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
    assert_eq!(
        (stored.search, stored.time.clone()),
        (None, TuningTime::default())
    );
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

    // A unit whose start never fits the tuning time (confirming defaults
    // this slow would not end in it) keeps its defaults unmeasured and
    // stores nothing, so the next load searches it instead of reading
    // those defaults as its result.
    let mut slow = FakeCase::new(&implementation, &statics, "implementation c");
    slow.defaults_seconds = TUNING_TIME.as_secs_f64();
    let (kept, started, _) = load_once(&slow, limits);
    assert_eq!(kept.origin, TuningOrigin::Searched);
    assert!(!started);
    assert_eq!(stored_results(), 3);
    let quick = FakeCase::new(&implementation, &statics, "implementation c");
    let (searched, started, _) = load_once(&quick, limits);
    assert_eq!(searched.origin, TuningOrigin::Searched);
    assert!(started && searched.search.is_some());
    assert_eq!(stored_results(), 4);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn tuning_a_weight_without_its_import_entry_is_a_typed_failure() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
    };
    let recorder = Recorder::default();
    let implementation = seismic::generated::native_implementation::<dense_output::Entry>(&device)
        .unwrap()
        .unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, name| {
            spec.with_static(name.clone(), if name == "DS" { 0 } else { 8 })
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
    let mut tuner = Tuner::count(
        &device,
        TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: None,
            error_classes: &NO_ERROR_CLASSES,
        },
        LIMITS,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
    );
    tuner.tune(&case, &implementation, &statics).unwrap();
    // The census builds the unit's inputs, its weights among them.
    let mut tuner = tuner.census();
    assert!(matches!(
        tuner.tune(&case, &implementation, &statics),
        Err(CatalogFailure::Tuning {
            entry: "dense_output",
            ..
        })
    ));
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
    };
    let mut weights = TuningWeights::new(&device, &load, &ZeroTuningWeights, &import);
    let mut shared = HashMap::new();
    let noise = Noise::default();
    let building = RefCell::new(BuildBudget::default());
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
        building: &building,
    };
    let batch = inputs.batch(6, 256, 2, 512).unwrap();
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
    assert!(inputs.batch(1, 256, 2, 512).is_err());
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
    };
    let mut weights = TuningWeights::new(&device, &load, &ZeroTuningWeights, &import);
    let mut shared = HashMap::new();
    let noise = Noise::default();
    let building = RefCell::new(BuildBudget::default());
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
        building: &building,
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

#[test]
fn building_predicted_not_to_fit_is_refused() {
    let mut budget = BuildBudget::default();
    // Nothing is measured yet: building runs.
    budget.limit(Duration::ZERO);
    assert!(budget.admit(Building::Import, 1 << 30).is_ok());
    budget.record(Building::Import, 1 << 20, 0.01);
    // A gigabyte at ten milliseconds per megabyte does not fit a second.
    budget.limit(Duration::from_secs(1));
    assert!(budget.admit(Building::Import, 1 << 30).is_err());
    assert!(budget.refused());
    // A megabyte does, and a new point starts unrefused.
    budget.limit(Duration::from_secs(1));
    assert!(!budget.refused());
    assert!(budget.admit(Building::Import, 1 << 20).is_ok());
    // Each kind of building has its own rate.
    assert!(budget.admit(Building::Generation, 1 << 30).is_ok());
}

/// Units are started largest share of step time first, whatever the order
/// they are prepared in, and each admits points within its share.
#[test]
fn units_start_by_their_share_of_step_time() {
    let Some(device) = metal() else {
        return;
    };
    let (definition, load) = fixture_load();
    let import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
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
                if name == "DS" {
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
            error_classes: &NO_ERROR_CLASSES,
        },
        limits,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
    );
    tuner.tune(&small, &implementation, &statics).unwrap();
    tuner.tune(&large, &implementation, &statics).unwrap();
    let mut tuner = tuner.census();
    tuner.tune(&small, &implementation, &statics).unwrap();
    tuner.tune(&large, &implementation, &statics).unwrap();
    let mut tuner = tuner.search().unwrap();
    tuner.tune(&small, &implementation, &statics).unwrap();
    tuner.tune(&large, &implementation, &statics).unwrap();
    // Equal times at every point: the shares of step time follow launches,
    // a quarter and three quarters.
    let near = |allowance: Duration, share: f64| {
        (allowance.as_secs_f64() - TUNING_TIME.as_secs_f64() * share).abs() < 1e-3
    };
    assert!(near(small.allowances.borrow()[0], 0.25));
    assert!(near(large.allowances.borrow()[0], 0.75));
    // The large unit is started first, though prepared second.
    let started = recorder
        .0
        .borrow()
        .iter()
        .filter_map(|event| match event {
            TuningEvent::Started { bindings, .. } => Some(bindings.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(started, ["large", "small"]);
    let shares = tuner
        .tuned()
        .iter()
        .map(|tuned| (tuned.bindings.clone(), tuned.progress.unwrap().share))
        .collect::<Vec<_>>();
    assert!(
        matches!(&shares[..], [(first, large), (second, small)]
            if first == "large" && second == "small" && (large - 0.75).abs() < 1e-9 && (small - 0.25).abs() < 1e-9),
        "{shares:?}"
    );
}
