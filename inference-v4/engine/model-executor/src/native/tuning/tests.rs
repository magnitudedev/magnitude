use super::cases::DenseOutputTuning;
use super::*;
use crate::native::import::ImportKernels;
use crate::planning::tests::{fixture_definition, fixture_manifest};
use crate::{ComponentSelection, ModelLoadPlan};
use magnitude_model_kernels::{dense_output, shape_rows};
use seismic::{BackendName, ConfigurationRecord, DeviceCatalog, Exclusion, LoadError, Outcome};
use std::cell::RefCell;
use std::collections::HashMap;

const LIMITS: TuningLimits = TuningLimits {
    max_rows: 512,
    max_projected_rows: 8,
    context_tokens: 16384,
};

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
fn cpu_projection_screening_folds_shares_but_preserves_served_points() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let points = row_points(LIMITS);
    let screening = cpu_projection_screening(&device, &points);
    assert_eq!(
        screening
            .iter()
            .map(|point| points[point.index].rows)
            .collect::<Vec<_>>(),
        [1, 8, 32, 128]
    );
    for (point, expected) in screening
        .iter()
        .zip([0.40 + 0.20 / 3.0, 0.40 / 3.0, 0.10, 0.30])
    {
        assert!((point.weight - expected).abs() < 1e-12, "{screening:?}");
    }
    assert!((screening.iter().map(|point| point.weight).sum::<f64>() - 1.0).abs() < 1e-12);
    assert_eq!(points.len(), 10);
    assert!((points.iter().map(|point| point.weight).sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn screening_policy_is_part_of_the_tuning_cache_identity() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let points = row_points(LIMITS);
    let full = tuning_key_material(
        &device,
        "dense_output",
        "A=bf16",
        &NativeSpecialization::new(),
        "same-implementation",
        7,
        &points,
        &[],
    );
    let screened = tuning_key_material(
        &device,
        "dense_output",
        "A=bf16",
        &NativeSpecialization::new(),
        "same-implementation",
        7,
        &points,
        &cpu_projection_screening(&device, &points),
    );
    assert_ne!(full, screened);
    assert!(!full.contains("\nscreening "));
    assert!(screened.contains("\nscreening cpu-projection-3 "));
    assert!(full.contains("samples: 3"));
    assert!(screened.contains("samples: 1"));
    assert!(full.contains("confirmation_samples: 7"));
    assert!(screened.contains("confirmation_samples: 5"));
}

#[cfg(target_os = "macos")]
#[test]
fn metal_sample_policy_is_part_of_the_tuning_cache_identity() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap();
    let key = tuning_key_material(
        &device,
        "dense_output",
        "A=bf16",
        &NativeSpecialization::new(),
        "same-implementation",
        7,
        &row_points(LIMITS),
        &[],
    );
    assert!(key.contains("search 9"));
    assert!(key.contains("samples: 1"));
    assert!(key.contains("confirmation_samples: 5"));
    assert!(key.contains("confirmed: 3"));
}

#[test]
fn attention_points_cross_rows_with_served_contexts() {
    let points = attention_points(LIMITS);
    assert_eq!(
        points.len(),
        TUNING_ROWS.len() * 3,
        "64k exceeds the served context"
    );
    assert!(points.iter().all(|point| point.context.unwrap() <= 16384));
    assert_eq!(points[0].label, "m1-c256");
    // The history lengths of one row point form its class.
    assert_eq!(points[0].class.as_deref(), Some("m1"));
    assert_eq!(points[2].class.as_deref(), Some("m1"));
    assert_eq!(points[3].class.as_deref(), Some("m2"));
    assert!((points.iter().map(|point| point.weight).sum::<f64>() - 1.0).abs() < 1e-12);
    let short = attention_points(TuningLimits {
        max_rows: 1,
        max_projected_rows: 1,
        context_tokens: 128,
    });
    assert_eq!(short.len(), 1);
    assert_eq!(short[0].context, Some(128));
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
    let scopes = (0..32).map(WeightScope::TargetBlock).collect::<Vec<_>>();
    let rows = |rows| served_row_points(512, move |served| served == rows).remove(0);
    assert_eq!(
        TuningInputs::rotation_scopes(&scopes, &rows(4)),
        [0, 8, 16, 24].map(WeightScope::TargetBlock)
    );
    assert_eq!(
        TuningInputs::rotation_scopes(&scopes, &rows(32)),
        [WeightScope::TargetBlock(0)]
    );
    let few = [WeightScope::HeadBlock(0)];
    assert_eq!(TuningInputs::rotation_scopes(&few, &rows(1)), few);
}

#[test]
fn the_model_budget_is_shared_equally_and_small_spaces_return_their_rest() {
    // 100 over four units: the 6-configuration unit needs only 6; the other
    // three share 94 (32, 31, 31).
    assert_eq!(allocate(100, &[1620, 6, 540, 108]), [32, 6, 31, 31]);
    // Every unit fits: each takes its size.
    assert_eq!(allocate(100, &[3, 4, 5]), [3, 4, 5]);
    // A budget below the unit count still evaluates every unit's defaults.
    assert_eq!(allocate(2, &[10, 10, 10]), [1, 1, 1]);
    // Redistribution cascades: after the 10 leaves, 45 each fits the 40.
    assert_eq!(allocate(100, &[10, 40, 500]), [10, 40, 50]);
}

#[test]
fn spaces_up_to_the_complete_size_are_searched_completely_first() {
    // The 24s are complete although an equal share (100 / 5 = 20) would
    // cut them; the two large units share the remaining 44.
    assert_eq!(allocate(100, &[24, 24, 8, 1620, 540]), [24, 24, 8, 22, 22]);
    // When the small spaces do not all fit, every unit shares equally.
    assert_eq!(allocate(40, &[24, 24, 8, 1620]), [11, 11, 8, 10]);
}

fn metal() -> Option<Device> {
    DeviceCatalog::discover()
        .ok()?
        .open_backend(BackendName::Metal)
        .ok()
}

fn fixture_load() -> (magnitude_model_contracts::ModelDefinition, ModelLoadPlan) {
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
/// "tuner" checks the points it receives and reports a fixed table choosing
/// `chosen`.
struct FakeCase {
    seen: RefCell<Vec<(String, f64, usize)>>,
    /// The implementation digest the case reports.
    digest: String,
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
            seen: RefCell::new(Vec::new()),
            digest: digest.into(),
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
}

impl EntryTuning for FakeCase {
    type Entry = dense_output::Entry;
    type Case = FakeArgs;

    fn bindings(&self) -> String {
        "fake".into()
    }
    fn statics(&self, _inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(vec![("H", 8), ("F", 16)])
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
        }
    }

    fn tune(
        &self,
        _device: &Device,
        statics: &NativeSpecialization,
        points: Vec<TuningPoint<'_, Self::Entry>>,
        validation: Validation,
        _strategy: Strategy,
    ) -> Result<TuningResult, TuneError> {
        self.seen.borrow_mut().extend(
            points
                .iter()
                .map(|point| (point.label.clone(), point.weight, point.rotation.len())),
        );
        let mut rejected = self.chosen.clone();
        rejected.launches[0].insert("ROWS".into(), 4);
        assert_eq!(&self.chosen.statics, statics.statics());
        Ok(TuningResult {
            tuning_identity: "fake-device".into(),
            entry: "dense_output".into(),
            backend: "metal".into(),
            points: points
                .iter()
                .map(|point| seismic::PointRecord {
                    label: point.label.clone(),
                    weight: point.weight,
                    class: point.class.clone(),
                })
                .collect(),
            validation,
            parameters: Vec::new(),
            configurations: vec![
                ConfigurationRecord {
                    configuration: self.chosen.clone(),
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
            overall: self.chosen.clone(),
            method: TuningMethod::Search {
                budget: 2,
                settings: SEARCH_SETTINGS,
                stop: SearchStop::Exhausted,
                screening: Vec::new(),
            },
            time: TuningTime::default(),
        })
    }
    fn digest(
        &self,
        _device: &Device,
        _statics: &NativeSpecialization,
    ) -> Result<String, TuneError> {
        Ok(self.digest.clone())
    }
    fn prepare(
        &self,
        device: &Device,
        specialization: &NativeSpecialization,
    ) -> Result<NativeKernel<Self::Entry>, LoadError> {
        dense_output::native_for_device_with(
            device,
            dense_output::Elements {
                DW: Element::bf16(),
                A: Element::bf16(),
            },
            specialization,
        )
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
            spec.with_static(name.clone(), if name == "H" { 8 } else { 16 })
        });
    let case = FakeCase::new(&implementation, &statics, "fake");
    let configurations = implementation.admissible(&statics).unwrap().len();
    // A census counts the unit without tuning it.
    let mut census = Tuner::census(
        &device,
        context,
        limits,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
    );
    let default = census.tune(&case, &implementation, &statics).unwrap();
    assert_eq!(
        default,
        implementation.default_specialization(&statics).unwrap()
    );
    assert!(case.seen.borrow().is_empty() && recorder.0.borrow().is_empty());
    let budgets = census.budgets();
    assert_eq!(
        budgets.0.values().copied().collect::<Vec<_>>(),
        [MODEL_BUDGET.min(configurations)]
    );
    let mut tuner = Tuner::new(
        &device,
        context,
        limits,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        budgets,
    );
    let chosen = tuner.tune(&case, &implementation, &statics).unwrap();
    assert_eq!(chosen.launch_param(0, "ROWS"), Some(2));
    let seen = case.seen.borrow();
    assert_eq!(
        seen.iter()
            .map(|(label, _, _)| label.as_str())
            .collect::<Vec<_>>(),
        ["m1", "m2", "m4", "m8", "m16", "m32", "m64"]
    );
    assert!(seen
        .iter()
        .all(|(_, _, rotation)| *rotation == ROTATION_LAYERS));
    let events = recorder.0.borrow();
    assert!(matches!(
        &events[0],
        TuningEvent::Started {
            entry: "dense_output",
            configurations: count,
            points: 7,
            ..
        } if *count == configurations
    ));
    let TuningEvent::Finished(tuned) = &events[1] else {
        panic!("tuning reports completion");
    };
    assert_eq!((tuned.measured, tuned.excluded, tuned.defects), (1, 1, 1));
    assert_eq!(tuner.tuned(), [tuned.clone()]);
}

/// Stored tuning results (tuning spec §C2): a miss tunes and stores the
/// result; a hit prepares the stored choice without tuning; a change to any
/// of the key's material is a miss.
#[test]
fn a_stored_tuning_result_is_used_without_tuning_and_a_changed_key_misses() {
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
            spec.with_static(name.clone(), if name == "H" { 8 } else { 16 })
        });
    let limits = TuningLimits {
        max_rows: 64,
        max_projected_rows: 8,
        context_tokens: 256,
    };
    // One load: a census, then tuning within its budgets. Returns the unit's
    // outcome and whether tuning started.
    let load_once = |case: &FakeCase, limits: TuningLimits| -> (TunedEntry, bool) {
        let recorder = Recorder::default();
        let context = TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: Some(&cache),
        };
        let weights = || TuningWeights::new(&device, &load, &ZeroTuningWeights, &import);
        let mut census = Tuner::census(&device, context, limits, weights());
        census.tune(case, &implementation, &statics).unwrap();
        let mut tuner = Tuner::new(&device, context, limits, weights(), census.budgets());
        let chosen = tuner.tune(case, &implementation, &statics).unwrap();
        let tuned = tuner.tuned().pop().unwrap();
        assert_eq!(chosen, tuned.overall.specialization());
        let started = recorder
            .0
            .borrow()
            .iter()
            .any(|event| matches!(event, TuningEvent::Started { .. }));
        (tuned, started)
    };
    let stored_results = || std::fs::read_dir(root.join("tuning")).unwrap().count();

    let first = FakeCase::new(&implementation, &statics, "implementation a");
    let (searched, started) = load_once(&first, limits);
    assert_eq!(searched.origin, TuningOrigin::Searched);
    assert!(started && !first.seen.borrow().is_empty());
    assert_eq!(stored_results(), 1);

    let again = FakeCase::new(&implementation, &statics, "implementation a");
    let (stored, started) = load_once(&again, limits);
    assert_eq!(stored.origin, TuningOrigin::Stored);
    assert!(
        !started && again.seen.borrow().is_empty(),
        "a stored result is not tuned"
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
            spec.with_static(name.clone(), 8)
        });
    let case = DenseOutputTuning {
        down: Element::bf16(),
        activation: Element::bf16(),
        scopes: vec![WeightScope::TargetBlock(0)],
    };
    let key = ("dense_output", case.bindings(), statics.statics().clone());
    let mut tuner = Tuner::new(
        &device,
        TuningContext {
            definition: &definition,
            weights: &ZeroTuningWeights,
            observer: &recorder,
            cache: None,
        },
        LIMITS,
        TuningWeights::new(&device, &load, &ZeroTuningWeights, &import),
        TuningBudgets([(key, 4)].into_iter().collect()),
    );
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
    let mut inputs = TuningInputs {
        device: &device,
        definition: &definition,
        limits: LIMITS,
        weights: &mut weights,
        shared: &mut shared,
    };
    let batch = inputs.batch(6, 256, 2, 512).unwrap();
    assert_eq!(batch.actual_rows, 6);
    assert_eq!(batch.actual_slots, 2);
    assert_eq!(batch.class.rows(), 8);
    assert_eq!(&batch.destinations[..6], [512, 513, 514, 515, 516, 517]);
    assert_eq!(batch.visible[0][0], [0, 256]);
    assert_eq!(batch.visible[3][0], [256, 512]);
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
    let inputs = TuningInputs {
        device: &device,
        definition: &definition,
        limits: LIMITS,
        weights: &mut weights,
        shared: &mut shared,
    };
    let values = (0..8).map(|value| value as f32).collect::<Vec<_>>();
    let tensor = inputs.f32s(&[4, 2], &values).unwrap();
    let mut state = inputs.state(tensor, 1..3).unwrap();
    let mut other = state.share();
    other.tensor_mut().write_from_host(&[0u8; 32]).unwrap();
    let mut restore = initializer(vec![&state]).unwrap();
    restore().unwrap();
    let bytes = state.tensor_mut().read_to_host().unwrap();
    let restored = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(restored, [0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0]);
    assert!(initializer(Vec::new()).is_none());
}
