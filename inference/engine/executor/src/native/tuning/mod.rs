//! Tuning at program preparation (route spec NR12, program spec E6, tuning
//! spec `specs/26-09-24/native-tuning-search-and-caching.md`).
//!
//! Every native entry whose implementation declares tuning parameters is
//! tuned on the opened device on the first load for each tuning key. Seismic
//! searches the declared domain within a budget (a share of
//! [`MODEL_BUDGET`] among the model's tuning units, counted by a census
//! before tuning), forming, measuring and validating what it reaches. With a
//! [`KernelCache`], each result is stored under a key over everything it
//! depends on (device and toolchain identity, unit, implementation digest,
//! search definition and model weight groups); a later load checks the stored
//! evidence against actual initialized inputs before reusing its choice. Nothing is shipped.
//! The engine supplies what only it knows, per entry, through an
//! [`EntryTuning`] case:
//!
//! - static values from model geometry;
//! - tuning points over the shape classes the entry serves (§5.3), with
//!   weights;
//! - rotations built from real resident weights of up to
//!   [`ROTATION_LAYERS`] distinct layers, so repeated calls stream from
//!   memory as a real step does;
//! - control tables built as the batch builder builds them
//!   ([`TuningInputs::batch`]);
//! - scratch state, KV history and routing tables owned by each case. An
//!   entry's `&mut` parameters bind [`CaseState`]s, whose complete contents are
//!   restored before each reference and validated invocation (Seismic's
//!   `TuningPoint::initialize`); real state is never bound.
//!
//! Each entry uses the compiler precision policy with explicit floating result/state limits.
//!
//! Adding an entry: implement [`EntryTuning`] for a case type in the module of
//! its block family, and prepare the entry through `Specializer::tuned` at
//! its preparation call site. An entry that declares parameters but is
//! prepared through `Specializer::fixed` fails preparation with a typed error
//! naming it.
//!
//! Entries sharing element bindings, static values and model weight groups can
//! reuse matching numerical evidence; equal geometry alone never authorizes reuse.

/// Implements [`EntryTuning::tune`] and [`EntryTuning::prepare`] through an
/// entry module's generated `native_tune[_with]` and
/// `native_for_device[_with]`. `$this => $elements` names the case and the
/// entry's element bindings built from it.
macro_rules! generated_entry {
    ($module:ident, $this:ident => $elements:expr) => {
        fn precision(&self) -> Result<seismic::PrecisionPolicy, seismic::TuneError> {
            let $this = self;
            super::precision::policy($module::native_numerical_subjects($elements)?)
        }

        fn tune(
            &self,
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
            points: Vec<seismic::TuningPoint<'_, Self::Entry>>,
            validation: seismic::PrecisionPolicy,
            strategy: seismic::Strategy,
            reuse: Option<seismic::TuningReuse<'_>>,
        ) -> Result<seismic::TuningResult, seismic::TuneError> {
            let $this = self;
            $module::native_tune_with(
                device,
                $elements,
                statics,
                points,
                validation,
                strategy,
                reuse,
                seismic::TuningReference::NativeDefault,
            )
        }

        fn digest(
            &self,
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
        ) -> Result<String, seismic::TuneError> {
            let $this = self;
            $module::native_digest_with(device, $elements, statics)
        }

        fn prepare(
            &self,
            device: &seismic::Device,
            specialization: &seismic::NativeSpecialization,
        ) -> Result<seismic::NativeKernel<Self::Entry>, seismic::LoadError> {
            let $this = self;
            $module::native_for_device_with(device, $elements, specialization)
        }
    };
    ($module:ident) => {
        fn precision(&self) -> Result<seismic::PrecisionPolicy, seismic::TuneError> {
            super::precision::policy($module::native_numerical_subjects()?)
        }

        fn tune(
            &self,
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
            points: Vec<seismic::TuningPoint<'_, Self::Entry>>,
            validation: seismic::PrecisionPolicy,
            strategy: seismic::Strategy,
            reuse: Option<seismic::TuningReuse<'_>>,
        ) -> Result<seismic::TuningResult, seismic::TuneError> {
            $module::native_tune(
                device,
                statics,
                points,
                validation,
                strategy,
                reuse,
                seismic::TuningReference::NativeDefault,
            )
        }

        fn digest(
            &self,
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
        ) -> Result<String, seismic::TuneError> {
            $module::native_digest(device, statics)
        }

        fn prepare(
            &self,
            device: &seismic::Device,
            specialization: &seismic::NativeSpecialization,
        ) -> Result<seismic::NativeKernel<Self::Entry>, seismic::LoadError> {
            $module::native_for_device(device, specialization)
        }
    };
}

pub(crate) mod attention;
pub(crate) mod cases;
pub(crate) mod general_routed;
pub(crate) mod per_layer;
#[cfg(feature = "pinned-tuning")]
pub mod pinned;
pub(crate) mod post_norm;
mod precision;
pub(crate) mod readout;
pub(crate) mod recurrent;
pub(crate) mod routed;
pub(crate) mod short_conv;
pub(crate) mod state_space;
#[cfg(feature = "tuning-survey")]
pub mod survey;
mod weights;

pub(crate) use weights::TuningWeights;
pub use weights::{TuningWeightSource, ZeroTuningWeights};

use super::CatalogFailure;
use crate::kernel_cache::{KernelCache, TuningCacheKey};
use magnitude_batching::{ClassLimits, Demand, PackedRowTables, Row, RowHistory, Slot};
use magnitude_family_contracts::{ModelDefinition, Operator, WeightKind, WeightScope};
use seismic::{
    Configuration, DType, Device, Element, NativeImplementation, NativeKernel,
    NativeSpecialization, ParameterValues, PrecisionPolicy, ScreeningPoint, SearchPlan,
    SearchSettings, SearchStop, Strategy, Tensor, TensorError, TuneError, TuningInitializer,
    TuningMethod, TuningPoint, TuningResult, TuningTime,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The row counts whose shares of step time weigh the objective (§5.3).
pub const TUNING_ROWS: [u64; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
/// History lengths of attention tuning points (§5.3).
// An empty history exercises the fresh-only path, where the Gemma G8
// defect reproduces. Longer histories plus their appended rows already
// cover partially occupied groups.
pub const TUNING_CONTEXTS: [u64; 5] = [0, 256, 4096, 16384, 65536];
/// Distinct layers a decode-row rotation cycles through where the model has
/// them. Decode rows stream every weight once per call, so repeated calls
/// must not find the weights cache resident. A prefill chunk reuses each
/// weight across its row tile and is compute bound, so one argument set
/// measures it.
pub const ROTATION_LAYERS: usize = 4;
/// The largest row count at which projections stream their weights (the K1
/// GEMV bound); larger counts run tiled GEMMs.
pub const STREAMING_ROWS: u64 = 8;

/// Version of the search procedure: part of every stored tuning result's
/// key. Bump it whenever the search could choose differently given the same
/// measurements.
/// 2: per-point keyed measurement and costs relative to the defaults.
/// 3: device warmed before each measured batch; a finalist whose confirmed
///    samples spread widely is excluded.
/// 4: points of one class (the same rows at different history lengths)
///    split their weight by real time.
/// 5: measure every served row class rather than folding shares into a few
///    representative rows.
/// 9: one device measurement serves duplicate point/active-set keys within a
///    factored sweep, removing noise differences between identical work.
/// 10: budgets follow the units' measured shares of step time; each form
///    parameter's values seed a start of their own.
/// 11: reserve enough of that budget to measure every admissible form start.
/// 12: workload hints may replace a form's nearest-default first measurement.
/// 13: bounded first-execution validation, complete state resets, short-history
///     coverage and 2 ms steady timing windows.
/// 14: budget and cache units distinguish model weight groups.
/// 15: batched timing without per-invocation resets, 200 us sample windows,
///     histories 0 and 256 upward, units by entry, bindings and statics,
///     validation of written state rows with one whole-state guard point.
pub const SEARCH_VERSION: u32 = 15;
/// Version of the CPU projection screening policy in keys that use it.
const CPU_PROJECTION_SCREENING_VERSION: u32 = 3;
/// Configurations one model's tuning may evaluate in all (`B_model`,
/// §D3), the census's defaults measurements included, allocated by
/// [`allocate`] in proportion to the units' shares of step time.
pub const MODEL_BUDGET: usize = 100;
/// Spaces of at most this many configurations are searched completely when
/// the model's budget holds all of them: at the budgets the searches get,
/// the replay found the search on 24-configuration spaces within 2% of the
/// best 0% of the time (tuning spec §E3).
pub const COMPLETE_SIZE: usize = 24;
/// The search's constants (§D2).
pub const SEARCH_SETTINGS: SearchSettings = SearchSettings {
    improvement: 0.01,
    restarts: 2,
    confirmed: 3,
    default_margin: 0.02,
    samples: 3,
    confirmation_samples: 7,
};

/// CPU projection screening uses one sample and five finalist samples: a
/// replay of 31 stored CPU searches preserved every seven-sample winner at
/// five, while three samples changed one winner. Metal uses the same one
/// screening sample and five finalist samples.
pub(crate) fn search_settings(backend: seismic::BackendName, screening: bool) -> SearchSettings {
    let mut settings = SEARCH_SETTINGS;
    if screening {
        settings.samples = 1;
        settings.confirmation_samples = 5;
    } else if backend == seismic::BackendName::Metal {
        // Metal candidates screen with one sample. The default and leading
        // candidates get five fresh confirmation samples; the recorded replay
        // matched every seven-sample winner while removing two measurements
        // from every finalist and validation point.
        settings.samples = 1;
        settings.confirmation_samples = 5;
    }
    settings
}
/// Minimum device time of one sample; device timestamps resolve
/// microseconds.
pub const MIN_SAMPLE_SECONDS: f64 = 0.0002;
/// The safety stop of one preparation's tuning: past it every search ends
/// with the best found so far, and its result is not stored. It exists for
/// pathological machines; budgets, not time, bound tuning otherwise.
pub const SAFETY_STOP: Duration = Duration::from_secs(600);

/// One workload an entry serves: its shape and its share of expected step
/// time.
#[derive(Clone, Debug, PartialEq)]
pub struct PointShape {
    pub label: String,
    pub weight: f64,
    pub rows: u64,
    /// Visible history rows for attention points.
    pub context: Option<u64>,
    /// The row point whose history lengths this point varies: points of one
    /// class split its weight by real time (`seismic::TuningPoint::class`).
    pub class: Option<String>,
}

/// The engine's shape bounds that decide which points exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TuningLimits {
    /// Rows of the largest target batch class.
    pub max_rows: u64,
    /// Rows of the largest launch that projects logits.
    pub max_projected_rows: u64,
    /// The served context: the longest history a row attends to.
    pub context_tokens: u64,
}

/// Provisional share of expected step time per row count. Decode (1 row)
/// dominates serving; verify and concurrency rows (2–8) come next; prefill
/// chunks share the rest. Weights of points beyond the engine's bounds are
/// dropped and the remainder renormalized.
fn row_share(rows: u64) -> f64 {
    match rows {
        1 => 0.40,
        2..=8 => 0.20 / 3.0,
        16 | 32 => 0.05,
        _ => 0.30 / 4.0,
    }
}

fn normalized(mut points: Vec<PointShape>) -> Vec<PointShape> {
    let total = points.iter().map(|point| point.weight).sum::<f64>();
    for point in &mut points {
        point.weight /= total;
    }
    points
}

fn row_point(rows: u64) -> PointShape {
    PointShape {
        label: format!("m{rows}"),
        weight: row_share(rows),
        rows,
        context: None,
        class: None,
    }
}

/// Row points up to the engine's row bound.
pub fn row_points(limits: TuningLimits) -> Vec<PointShape> {
    served_row_points(limits.max_rows, |_| true)
}

/// Every row point an entry serves up to `bound`. An entry whose served rows
/// all lie beyond the bound is still prepared (its graph classes do not
/// exist, but the kernel set is complete); its smallest served row count
/// stands in as the single point.
pub fn served_row_points(bound: u64, serves: impl Fn(u64) -> bool) -> Vec<PointShape> {
    let served = TUNING_ROWS
        .into_iter()
        .filter(|rows| serves(*rows))
        .collect::<Vec<_>>();
    let within = served
        .iter()
        .copied()
        .filter(|rows| *rows <= bound)
        .collect::<Vec<_>>();
    if within.is_empty() {
        return served
            .into_iter()
            .take(1)
            .map(|rows| PointShape {
                weight: 1.0,
                ..row_point(rows)
            })
            .collect();
    }
    normalized(within.into_iter().map(row_point).collect())
}

/// Screen the expensive CPU projection spaces on a small row sample. The
/// tuner confirms the default and finalists on every original row point.
pub(crate) fn cpu_projection_screening(
    device: &Device,
    points: &[PointShape],
) -> Vec<ScreeningPoint> {
    if device.backend() != seismic::BackendName::Cpu || points.len() <= 4 {
        return Vec::new();
    }
    const ROWS: [u64; 4] = [1, 8, 32, 128];
    let representatives = points
        .iter()
        .enumerate()
        .filter_map(|(index, point)| ROWS.contains(&point.rows).then_some(index))
        .collect::<Vec<_>>();
    if representatives.is_empty() {
        return Vec::new();
    }
    let mut screening = representatives
        .iter()
        .map(|index| ScreeningPoint {
            index: *index,
            weight: 0.0,
        })
        .collect::<Vec<_>>();
    for point in points {
        let nearest = screening
            .iter_mut()
            .min_by_key(|candidate| {
                let rows = points[candidate.index].rows;
                (
                    (point.rows.ilog2() as i32 - rows.ilog2() as i32).abs(),
                    u64::MAX - rows,
                )
            })
            .expect("the screening set is nonempty");
        nearest.weight += point.weight;
    }
    screening
}

/// `rows` crossed with the history lengths the engine serves.
pub fn with_contexts(limits: TuningLimits, rows: Vec<PointShape>) -> Vec<PointShape> {
    let contexts = TUNING_CONTEXTS
        .into_iter()
        .filter(|context| *context <= limits.context_tokens)
        .collect::<Vec<_>>();
    let contexts = if contexts.is_empty() {
        vec![limits.context_tokens]
    } else {
        contexts
    };
    let share = 1.0 / contexts.len() as f64;
    normalized(
        rows.into_iter()
            .flat_map(|point| {
                contexts.iter().map(move |&context| PointShape {
                    label: format!("{}-c{context}", point.label),
                    weight: point.weight * share,
                    rows: point.rows,
                    context: Some(context),
                    class: Some(point.label.clone()),
                })
            })
            .collect(),
    )
}

/// Row points crossed with the history lengths the engine serves.
pub fn attention_points(limits: TuningLimits) -> Vec<PointShape> {
    with_contexts(limits, row_points(limits))
}

/// In-place state a case lends to an entry's `&mut` parameter: the tensor,
/// the leading-axis rows the entry writes, and their initial contents. At
/// most points only those rows are restored before every reference and
/// validated invocation and observed by validation; the rest is input the
/// entry only reads. One point per unit ([`guard_point`]) restores and
/// observes the complete storage, so a write outside the declared rows is
/// rejected without paying for whole long histories everywhere.
pub(crate) struct CaseState {
    tensor: Tensor,
    written: Range<u64>,
    region: Tensor,
    initial: Arc<[u8]>,
    /// The complete physical storage behind `tensor`.
    backing: Tensor,
    _slab: Option<Arc<seismic::SlabTensor>>,
}

impl CaseState {
    pub fn tensor_mut(&mut self) -> &mut Tensor {
        &mut self.tensor
    }

    /// Another handle to the same storage, for a further argument set.
    pub fn share(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            written: self.written.clone(),
            region: self.region.clone(),
            initial: self.initial.clone(),
            backing: self.backing.clone(),
            _slab: self._slab.clone(),
        }
    }

    /// Restores the written rows.
    fn restorer(&self) -> Restorer {
        let mut region = self.region.clone();
        let initial = self.initial.clone();
        Box::new(move || region.write_from_host(&initial))
    }

    /// Restores the complete storage to its contents now, before any
    /// invocation of the unit.
    fn complete_restorer(&self) -> Result<Restorer, String> {
        let mut backing = self.backing.clone();
        let initial: Arc<[u8]> = backing
            .read_to_host()
            .map_err(|error| error.to_string())?
            .into();
        Ok(Box::new(move || backing.write_from_host(&initial)))
    }
}

type Restorer = Box<dyn FnMut() -> Result<(), TensorError>>;

/// Restores every writable state in every argument rotation of a point:
/// completely at the guard point, their written rows elsewhere.
fn initializer(
    states: Vec<&CaseState>,
    complete: bool,
) -> Result<Option<TuningInitializer<'static>>, String> {
    if states.is_empty() {
        return Ok(None);
    }
    let mut restorers = states
        .into_iter()
        .map(|state| {
            if complete {
                state.complete_restorer()
            } else {
                Ok(state.restorer())
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(Box::new(move || {
        restorers.iter_mut().try_for_each(|restore| restore())
    })))
}

/// The point whose states are observed whole: the one with the least state
/// storage, so the guard costs a short history, not a long one. `None` for a
/// unit without states.
fn guard_point<C>(rotations: &[Vec<C>], state: impl Fn(&C) -> u64) -> Option<usize> {
    rotations
        .iter()
        .enumerate()
        .filter_map(|(point, cases)| cases.first().map(|case| (point, state(case))))
        .filter(|(_, bytes)| *bytes > 0)
        .min_by_key(|(_, bytes)| *bytes)
        .map(|(point, _)| point)
}

/// Entry-specific tuning knowledge. One implementation exists per native
/// entry that declares tuning parameters.
pub(crate) trait EntryTuning {
    type Entry: seismic::Entry;
    fn precision(&self) -> Result<PrecisionPolicy, TuneError>;
    /// One argument set: every tensor its arguments borrow, owned, including
    /// the [`CaseState`]s its `&mut` parameters bind.
    type Case;

    /// The semantic bindings, for reports.
    fn bindings(&self) -> String;
    /// Launches of the entry in one step of a row class it serves: one per
    /// layer the case binds, one for a per-step entry.
    fn launches(&self) -> usize;
    /// The value of every dimension the model fixes.
    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String>;
    fn points(&self, limits: TuningLimits) -> Vec<PointShape>;
    /// Empty uses the complete served workload for every search measurement.
    fn screening(&self, _device: &Device, _points: &[PointShape]) -> Vec<ScreeningPoint> {
        Vec::new()
    }
    /// Workload-informed admissible configurations to measure at the start
    /// of a search. A hint in another structural form replaces that form's
    /// usual nearest-default start within the same budget slot.
    fn search_starts(
        &self,
        _device: &Device,
        _implementation: &NativeImplementation,
        _statics: &NativeSpecialization,
        _limits: TuningLimits,
    ) -> Vec<ParameterValues> {
        Vec::new()
    }
    /// The argument sets of one point, cycling distinct layers.
    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String>;
    fn args<'a>(case: &'a mut Self::Case) -> <Self::Entry as seismic::Entry>::Args<'a>;
    /// The states `case` lends through `&mut` parameters, by parameter name;
    /// none for an entry without them. Seismic rejects a point that binds a
    /// `&mut` parameter whose state is not listed here.
    fn state(_case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        Vec::new()
    }
    /// The entry's generated `native_tune[_with]`.
    fn tune(
        &self,
        device: &Device,
        statics: &NativeSpecialization,
        points: Vec<TuningPoint<'_, Self::Entry>>,
        validation: PrecisionPolicy,
        strategy: Strategy,
        reuse: Option<seismic::TuningReuse<'_>>,
    ) -> Result<TuningResult, TuneError>;
    /// The entry's generated `native_digest[_with]`.
    fn digest(&self, device: &Device, statics: &NativeSpecialization) -> Result<String, TuneError>;
    /// Check a stored choice against the same device-augmented declaration
    /// that native preparation uses.
    fn stored_valid(&self, device: &Device, specialization: &NativeSpecialization) -> bool {
        seismic::generated::native_specialization_valid::<Self::Entry>(device, specialization)
            .unwrap_or(false)
    }
    /// The entry's generated `native_for_device[_with]`.
    fn prepare(
        &self,
        device: &Device,
        specialization: &NativeSpecialization,
    ) -> Result<NativeKernel<Self::Entry>, seismic::LoadError>;
}

/// Progress of tuning at load, for readiness reporting.
#[derive(Clone, Debug, PartialEq)]
pub enum TuningEvent {
    /// The configuration budget of the units searched so far, of the units
    /// that search at this load. Reported when tuning begins (none searched)
    /// and after each searched unit; a total of zero means nothing searches.
    Progress {
        completed: usize,
        total: usize,
    },
    Started {
        entry: &'static str,
        bindings: String,
        configurations: usize,
        points: usize,
    },
    Finished(TunedEntry),
}

/// Where a tuning unit's choice came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuningOrigin {
    /// Searched on the device at this load.
    Searched,
    /// A stored result of the same key: nothing was formed or measured.
    Stored,
}

/// The outcome of tuning one entry.
#[derive(Clone, Debug, PartialEq)]
pub struct TunedEntry {
    pub entry: &'static str,
    pub bindings: String,
    pub overall: Configuration,
    pub origin: TuningOrigin,
    /// The search's budget and why it stopped (a searched unit).
    pub search: Option<(usize, SearchStop)>,
    /// Configurations the result records as measured (for a stored result,
    /// measured when it was searched).
    pub measured: usize,
    pub excluded: usize,
    /// Configurations rejected during formation or numerical qualification.
    /// A numerical rejection is a policy result, not necessarily a kernel defect.
    pub rejections: usize,
    /// The first qualification rejection's configuration and reason.
    pub first_rejection: Option<String>,
    /// Wall time of this unit at this load.
    pub seconds: f64,
    /// Where the search's time went; zero for a stored result.
    pub time: TuningTime,
}

pub trait TuningObserver {
    fn event(&self, event: &TuningEvent);
}

/// An observer for hosts that do not report tuning progress.
pub struct UnreportedTuning;

impl TuningObserver for UnreportedTuning {
    fn event(&self, _event: &TuningEvent) {}
}

/// What tuning needs from the host: the model, its weights, and where
/// progress goes.
#[derive(Clone, Copy)]
pub struct TuningContext<'a> {
    pub definition: &'a ModelDefinition,
    pub weights: &'a dyn TuningWeightSource,
    pub observer: &'a dyn TuningObserver,
    /// Where tuning results are stored between loads; `None` tunes every
    /// unit at every load.
    pub cache: Option<&'a KernelCache>,
}

/// Inputs a case builds its argument sets from. Every tensor it returns is
/// owned by the caller's case.
/// Pseudo-random activation values in [-1, 1), per element type, generated
/// once and shared by every tuning input: a tensor is a window of the pool
/// its seed places.
#[derive(Default)]
pub(crate) struct Noise(std::cell::RefCell<HashMap<DType, Vec<u8>>>);

impl Noise {
    /// `count` encoded values of `dtype` at the window `seed` places.
    fn window(&self, dtype: DType, count: usize, seed: u64) -> Result<Vec<u8>, String> {
        let (size, encode): (usize, fn(f32, &mut Vec<u8>)) = match dtype {
            DType::F32 => (4, |value, bytes| {
                bytes.extend_from_slice(&value.to_le_bytes())
            }),
            DType::BF16 => (2, |value, bytes| {
                bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes())
            }),
            DType::F16 => (2, |value, bytes| {
                bytes.extend_from_slice(&f16_bits(value).to_le_bytes())
            }),
            other => {
                return Err(format!(
                    "tuning activations support f32, bf16 and f16, not {other:?}"
                ))
            }
        };
        let mut pools = self.0.borrow_mut();
        let pool = pools.entry(dtype).or_default();
        // Twice the largest request, so windows at different seeds differ.
        let wanted = count
            .checked_mul(2 * size)
            .ok_or("tuning activation overflows")?;
        if pool.len() < wanted {
            let mut state = (pool.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            pool.reserve(wanted - pool.len());
            while pool.len() < wanted {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                encode(((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0, pool);
            }
        }
        let windows = (pool.len() / size - count + 1) as u64;
        let start = (seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 16) % windows;
        let start = start as usize * size;
        Ok(pool[start..start + count * size].to_vec())
    }
}

pub(crate) struct TuningInputs<'w, 'a> {
    pub device: &'a Device,
    pub definition: &'a ModelDefinition,
    pub limits: TuningLimits,
    pub weights: &'w mut TuningWeights<'a>,
    noise: &'w Noise,
    /// Tensors shared by every point of the entry being tuned.
    shared: &'w mut HashMap<String, Tensor>,
}

impl TuningInputs<'_, '_> {
    /// The operator of the sublayer whose weights `scope` names: the first
    /// scope of a case's layers (every layer of one binding shares its
    /// statics).
    pub fn operator(&self, scopes: &[WeightScope]) -> Result<&Operator, String> {
        let scope = *scopes
            .first()
            .ok_or("a tuning case needs at least one layer")?;
        // A branch's operator is its own within its sublayer's branches.
        if let WeightScope::TargetBranch { sublayer, branch } = scope {
            let Operator::Parallel(branches) =
                self.operator(&[WeightScope::TargetSublayer(sublayer)])?
            else {
                return Err(format!("{scope:?} names no parallel sublayer"));
            };
            return branches
                .get(branch as usize)
                .map(|value| &value.op)
                .ok_or_else(|| format!("{scope:?} names no branch of the model"));
        }
        let (blocks, index) = match scope {
            WeightScope::TargetSublayer(index) => (
                self.definition
                    .decoder
                    .blocks
                    .get(index.block as usize)
                    .map(|block| &block.sublayers),
                index,
            ),
            WeightScope::HeadSublayer(index) => (
                self.definition
                    .head
                    .as_ref()
                    .and_then(|head| head.blocks.get(index.block as usize))
                    .map(|block| &block.block.sublayers),
                index,
            ),
            WeightScope::DraftSublayer(index) => (
                self.definition
                    .draft
                    .as_ref()
                    .and_then(|draft| draft.blocks.get(index.block as usize))
                    .map(|block| &block.sublayers),
                index,
            ),
            other => return Err(format!("{other:?} names no sublayer")),
        };
        blocks
            .and_then(|sublayers| sublayers.get(index.sublayer as usize))
            .map(|sublayer| &sublayer.op)
            .ok_or_else(|| format!("{scope:?} names no sublayer of the model"))
    }

    /// The layers of `point`'s argument sets: up to [`ROTATION_LAYERS`] of
    /// `scopes`, spread over the model's depth, for decode rows (up to
    /// [`STREAMING_ROWS`]); the first layer alone for prefill rows.
    pub fn rotation_scopes(scopes: &[WeightScope], point: &PointShape) -> Vec<WeightScope> {
        let layers = if point.rows <= STREAMING_ROWS {
            ROTATION_LAYERS
        } else {
            1
        };
        if scopes.len() <= layers {
            return scopes.to_vec();
        }
        (0..layers)
            .map(|index| scopes[index * scopes.len() / layers])
            .collect()
    }

    /// The resident weight of one role, imported from the artifact into the
    /// resident representation and layout.
    pub fn weight(&mut self, scope: WeightScope, kind: WeightKind) -> Result<Tensor, String> {
        self.weights.weight(scope, kind)
    }

    /// The planned shape of one weight role.
    pub fn weight_shape(&self, scope: WeightScope, kind: WeightKind) -> Result<Vec<u64>, String> {
        self.weights.shape(scope, kind)
    }

    /// The planned extent of one weight role's accumulator-scale port (0
    /// without a second-level scale).
    pub fn scale_extent(&self, scope: WeightScope, kind: WeightKind) -> Result<u64, String> {
        self.weights.scale_extent(scope, kind)
    }

    /// A unit accumulator-scale port of `extent` (absent at 0): the scale's
    /// value does not bear on a configuration's timing or agreement.
    pub fn unit_scale(&self, extent: u64) -> Result<Tensor, String> {
        let count = usize::try_from(extent).map_err(|_| "scale extent exceeds usize")?;
        self.f32s(&[extent], &vec![1.0; count])
    }

    /// A deterministic pseudo-random activation in [-1, 1).
    pub fn activation(
        &self,
        element: Element,
        extents: &[u64],
        seed: u64,
    ) -> Result<Tensor, String> {
        let count = element_count(extents)?;
        let dtype = element
            .dtype()
            .ok_or_else(|| format!("tuning activations are not {}", element.name()))?;
        let bytes = self.noise.window(dtype, count, seed)?;
        Tensor::from_host(self.device, element, extents, &bytes).map_err(|error| error.to_string())
    }

    /// An `i32` tensor of `values`.
    pub fn i32s(&self, extents: &[u64], values: &[i32]) -> Result<Tensor, String> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Tensor::from_host(self.device, Element::i32(), extents, &bytes)
            .map_err(|error| error.to_string())
    }

    /// A `u32` tensor of `values`.
    pub fn u32s(&self, extents: &[u64], values: &[u32]) -> Result<Tensor, String> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Tensor::from_host(self.device, Element::u32(), extents, &bytes)
            .map_err(|error| error.to_string())
    }

    /// An `f32` tensor of `values`.
    pub fn f32s(&self, extents: &[u64], values: &[f32]) -> Result<Tensor, String> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Tensor::from_host(self.device, Element::f32(), extents, &bytes)
            .map_err(|error| error.to_string())
    }

    /// The `out_rows` gather selecting every one of `rows` rows in order.
    pub fn every_row(&self, rows: u64) -> Result<Tensor, String> {
        let count = i32::try_from(rows).map_err(|_| "tuning rows exceed i32")?;
        self.i32s(&[rows], &(0..count).collect::<Vec<_>>())
    }

    /// A zeroed tensor owned by the case: outputs and tables the entry
    /// overwrites.
    pub fn scratch(&self, element: Element, extents: &[u64]) -> Result<Tensor, String> {
        Tensor::zeros(self.device, element, extents).map_err(|error| error.to_string())
    }

    /// `tensor` lent to a `&mut` parameter that writes leading-axis rows
    /// `written`; their current contents are what every validated invocation
    /// starts from.
    pub fn state(&self, tensor: Tensor, written: Range<u64>) -> Result<CaseState, String> {
        let region = tensor
            .slice_leading(written.start, written.end)
            .map_err(|error| error.to_string())?;
        let initial = region.read_to_host().map_err(|error| error.to_string())?;
        Ok(CaseState {
            backing: tensor.clone(),
            tensor,
            written,
            region,
            initial: initial.into(),
            _slab: None,
        })
    }

    /// Back one tuning state plane with a single slab. The logical view is
    /// passed to the kernel; the physical span is retained for validation
    /// resets, which must write the slab rather than its address table.
    pub fn slab_state(
        &self,
        source: Tensor,
        view: u64,
        written: Range<u64>,
    ) -> Result<CaseState, String> {
        let extents = source.extents();
        let rows = *extents.first().ok_or("slab tuning state has no row axis")?;
        if view == 0 || view > rows || written.start >= written.end || written.end > view {
            return Err("slab tuning state has invalid row bounds".into());
        }
        let mut slab = seismic::SlabTensor::new(
            self.device,
            view,
            view,
            vec![seismic::SlabRegion {
                element: source.element(),
                row_shape: extents[1..].to_vec(),
            }],
        )
        .map_err(|error| error.to_string())?;
        slab.add_slab().map_err(|error| error.to_string())?;
        let bytes = source
            .slice_leading(0, view)
            .map_err(|error| error.to_string())?
            .read_to_host()
            .map_err(|error| error.to_string())?;
        slab.region_rows(0, 0, view)
            .map_err(|error| error.to_string())?
            .write_from_host(&bytes)
            .map_err(|error| error.to_string())?;
        let region = slab
            .region_rows(0, written.start, written.end - written.start)
            .map_err(|error| error.to_string())?;
        let tensor = slab.logical_region(0).map_err(|error| error.to_string())?;
        Ok(CaseState {
            tensor,
            written,
            initial: region
                .read_to_host()
                .map_err(|error| error.to_string())?
                .into(),
            region,
            backing: slab
                .region_rows(0, 0, view)
                .map_err(|error| error.to_string())?,
            _slab: Some(Arc::new(slab)),
        })
    }

    /// The tensor `name` shared by the points of the entry being tuned,
    /// built on first use.
    pub fn shared(
        &mut self,
        name: String,
        build: impl FnOnce(&Self) -> Result<Tensor, String>,
    ) -> Result<Tensor, String> {
        if let Some(tensor) = self.shared.get(&name) {
            return Ok(tensor.clone());
        }
        let tensor = build(self)?;
        self.shared.insert(name, tensor.clone());
        Ok(tensor)
    }

    /// Row tables for `rows` rows over `slots` requests, each with `context`
    /// accepted history rows, packed by the batch builder. Slot `s` sees
    /// history rows `[s·context, (s+1)·context)`; batch row `i` appends at
    /// history row `appends + i`, past every row any point reads, so no
    /// point's appends change another point's inputs.
    pub fn batch(
        &self,
        rows: u64,
        context: u64,
        slots: u64,
        appends: u64,
    ) -> Result<PackedRowTables, String> {
        let rows = usize::try_from(rows).map_err(|_| "tuning rows exceed usize")?;
        let slots = usize::try_from(slots.max(1)).map_err(|_| "tuning slots exceed usize")?;
        let context = i32::try_from(context).map_err(|_| "tuning context exceeds i32")?;
        let appends = i32::try_from(appends).map_err(|_| "tuning history exceeds i32")?;
        if slots > rows {
            return Err(format!("{slots} slots cannot share {rows} rows"));
        }
        let mut row_index = 0i32;
        let packed = (0..slots)
            .map(|slot| {
                let count = rows / slots + usize::from(slot < rows % slots);
                let base = slot as i32 * context;
                Slot {
                    rows: (0..count)
                        .map(|offset| {
                            let position = context + offset as i32;
                            let destination = appends + row_index;
                            row_index += 1;
                            Row {
                                token: (offset as i32 * 7919 + slot as i32) % 1024,
                                coordinates: [position, position, position, 0],
                                // One Token history domain.
                                histories: vec![RowHistory {
                                    visible: if context == 0 {
                                        Vec::new()
                                    } else {
                                        vec![[base, base + context]]
                                    },
                                    fresh_start: 0,
                                    bidirectional_end: None,
                                    destination,
                                }],
                                demand: Demand::NONE,
                                select: None,
                            }
                        })
                        .collect(),
                    bank: slot as i32 + 1,
                    previous_tape: 0,
                    following_bank: (slots + slot) as i32 + 1,
                    stop: count as i32,
                }
            })
            .collect::<Vec<_>>();
        let vocabulary = usize::try_from(self.definition.decoder.vocabulary)
            .map_err(|_| "vocabulary exceeds usize")?;
        // Each row sees at most one history span.
        let limits = ClassLimits { rows, segments: 1 };
        PackedRowTables::pack(&packed, vocabulary, limits).map_err(|error| error.to_string())
    }
}

fn element_count(extents: &[u64]) -> Result<usize, String> {
    extents
        .iter()
        .try_fold(1u64, |count, extent| count.checked_mul(*extent))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| "tuning tensor size overflows".to_owned())
}

fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        // |value| < 2^-14 in [-1, 1): flush to signed zero.
        return sign;
    }
    let rounded = (mantissa + 0x1000) >> 13;
    sign | (((exponent as u32) << 10) + rounded) as u16
}

/// An entry, its element bindings and its static values: what one tuning
/// result applies to (a tuning unit).
type TuningKey = (&'static str, String, BTreeMap<String, u64>);

/// Split `total` configurations among tuning units, each given as its
/// admissible configurations and its share of expected step time
/// ([`step_shares`], §D3). Units of at most [`COMPLETE_SIZE`] configurations
/// are searched completely when all of them fit in `total`; the rest is split
/// among the larger units in proportion to their shares ([`share`]). When
/// they do not all fit, every unit shares `total` so.
pub fn allocate(total: usize, units: &[(usize, f64)]) -> Vec<usize> {
    let small = units
        .iter()
        .map(|(size, _)| (*size).max(1))
        .filter(|size| *size <= COMPLETE_SIZE);
    let complete = small.clone().sum::<usize>();
    let large = units
        .iter()
        .filter(|(size, _)| *size > COMPLETE_SIZE)
        .count();
    if complete + large > total {
        return share(total, units);
    }
    let large_units = (0..units.len())
        .filter(|unit| units[*unit].0 > COMPLETE_SIZE)
        .collect::<Vec<_>>();
    let large_budgets = share(
        total - complete,
        &large_units
            .iter()
            .map(|unit| units[*unit])
            .collect::<Vec<_>>(),
    );
    let mut budgets = units
        .iter()
        .map(|(size, _)| (*size).max(1))
        .collect::<Vec<_>>();
    for (unit, budget) in large_units.into_iter().zip(large_budgets) {
        budgets[unit] = budget;
    }
    budgets
}

/// Move search slots from units with spare budget to units whose structural
/// forms would otherwise receive no first measurement. The initial allocation
/// still decides every unit's remaining share. When the model budget cannot
/// cover every floor, the defaults retain their one slot.
fn reserve_form_starts(budgets: &mut [usize], floors: &[usize]) {
    debug_assert_eq!(budgets.len(), floors.len());
    for unit in 0..budgets.len() {
        while budgets[unit] < floors[unit] {
            let donor = (0..budgets.len())
                .filter(|&candidate| candidate != unit && budgets[candidate] > floors[candidate])
                .max_by_key(|&candidate| {
                    (budgets[candidate] - floors[candidate], budgets[candidate])
                });
            let Some(donor) = donor else {
                break;
            };
            budgets[donor] -= 1;
            budgets[unit] += 1;
        }
    }
}

/// Split `total` configurations among tuning units of `(size, share)` in
/// proportion to their shares: a unit whose part covers its whole space
/// takes only its size, and what it leaves is split again among the others.
/// Every unit gets at least one configuration (its defaults). Parts are
/// rounded by largest remainder, ties to the first units, so the split is
/// deterministic given the units in order.
fn share(total: usize, units: &[(usize, f64)]) -> Vec<usize> {
    let mut budgets = vec![0; units.len()];
    let mut open = (0..units.len()).collect::<Vec<_>>();
    let mut remaining = total;
    loop {
        if open.is_empty() {
            return budgets;
        }
        let weight = open.iter().map(|unit| units[*unit].1).sum::<f64>();
        let part = |unit: usize| remaining as f64 * units[unit].1 / weight;
        let small = open
            .iter()
            .copied()
            .filter(|&unit| units[unit].0 as f64 <= part(unit))
            .collect::<Vec<_>>();
        if small.is_empty() {
            for &unit in &open {
                budgets[unit] = part(unit).floor() as usize;
            }
            let assigned = open.iter().map(|unit| budgets[*unit]).sum::<usize>();
            let mut order = open.clone();
            order.sort_by(|left, right| {
                let fraction = |unit: usize| part(unit) - part(unit).floor();
                fraction(*right).total_cmp(&fraction(*left))
            });
            for unit in order.into_iter().take(remaining.saturating_sub(assigned)) {
                budgets[unit] += 1;
            }
            for &unit in &open {
                budgets[unit] = budgets[unit].max(1);
            }
            return budgets;
        }
        for unit in small {
            budgets[unit] = units[unit].0.max(1);
            remaining -= units[unit].0;
            open.retain(|&candidate| candidate != unit);
        }
    }
}

/// One tuning unit's step time as a census knows it: its launches per step,
/// its points and its defaults' time at each.
#[derive(Clone, Copy)]
struct UnitTime<'u> {
    launches: usize,
    shapes: &'u [PointShape],
    seconds: &'u [f64],
}

/// Each unit's share of expected step time (§D3). Every row class holds its
/// share of step time ([`row_share`]), split among the units serving it by
/// their time there: launches per step times the defaults' mean time over
/// the class's points (its history lengths, equally likely). The defaults'
/// time is also what tuning can recover: a unit far from its best spends
/// more of the step and gets more of the budget.
fn step_shares(units: &[UnitTime<'_>]) -> Vec<f64> {
    let time = |unit: &UnitTime<'_>, rows: u64| {
        let class = unit
            .shapes
            .iter()
            .zip(unit.seconds)
            .filter(|(shape, _)| shape.rows == rows)
            .map(|(_, seconds)| *seconds)
            .collect::<Vec<_>>();
        if class.is_empty() {
            0.0
        } else {
            unit.launches as f64 * class.iter().sum::<f64>() / class.len() as f64
        }
    };
    let class_time = TUNING_ROWS
        .iter()
        .map(|rows| units.iter().map(|unit| time(unit, *rows)).sum::<f64>())
        .collect::<Vec<_>>();
    units
        .iter()
        .map(|unit| {
            TUNING_ROWS
                .iter()
                .zip(&class_time)
                .filter(|(_, total)| **total > 0.0)
                .map(|(rows, total)| row_share(*rows) * time(unit, *rows) / total)
                .sum()
        })
        .collect()
}

/// How the tuner treats each unit it is asked for.
enum Allocation {
    /// Counting the model's tuning units, their admissible sizes, whether
    /// each will search, and their defaults' step time: a unit that searches
    /// has its defaults measured, nothing else is formed or measured.
    Census(Vec<CensusUnit>),
    /// Tuning, each unit within its share of [`MODEL_BUDGET`].
    Budgets(HashMap<TuningKey, usize>),
}

/// One tuning unit a census counted.
struct CensusUnit {
    key: TuningKey,
    /// Admissible configurations.
    size: usize,
    /// Defaults plus one start per other admissible value of each form axis
    /// and each of the case's own search starts.
    form_starts: usize,
    /// Launches per step over every preparation of the unit.
    launches: usize,
    /// Whether it searches at this load: neither a stored result of its key
    /// nor a pinned choice exists.
    searches: bool,
    /// Its points and its defaults' time at each, when it spends the model
    /// budget: measured by the census when it searches, read from its stored
    /// result otherwise. A pinned unit has none, nor has a launch-scoped one
    /// (its factored search measures every candidate and spends no budget).
    defaults: Option<(Vec<PointShape>, Vec<f64>)>,
    /// The census's measurement of the defaults: wall seconds and where
    /// they went.
    measurement: Option<(f64, TuningResult)>,
}

/// The configurations each tuning unit of a model may evaluate, the units
/// that search at this load, and the census's measurement of each.
pub(crate) struct TuningBudgets {
    budgets: HashMap<TuningKey, usize>,
    searching: HashSet<TuningKey>,
    measurements: HashMap<TuningKey, (f64, TuningResult)>,
}

/// A unit's slot in the kernel cache: the cache, the unit's key, and the
/// valid result stored under it.
type StoredSlot<'c> = (&'c KernelCache, TuningCacheKey, Option<TuningResult>);

/// The defaults' time at each point of a stored result, when it measured
/// them.
fn defaults_seconds(result: &TuningResult, defaults: &ParameterValues) -> Option<Vec<f64>> {
    result
        .configurations
        .iter()
        .find(|record| record.configuration.params == *defaults)
        .and_then(|record| match &record.outcome {
            seismic::Outcome::Measured { points, .. } => {
                Some(points.iter().map(|point| point.median_seconds).collect())
            }
            seismic::Outcome::Excluded(_) => None,
        })
}

/// Resolves every native entry's specialization for the opened device:
/// static values, tuned parameters, and missing implementations.
pub(crate) struct Tuner<'a> {
    device: &'a Device,
    context: TuningContext<'a>,
    limits: TuningLimits,
    weights: TuningWeights<'a>,
    noise: Noise,
    allocation: Allocation,
    /// The safety stop of this preparation's tuning.
    deadline: Instant,
    tuned: Vec<TunedEntry>,
    chosen: HashMap<TuningKey, NativeSpecialization>,
    /// The latest choice for each parameter declaration of an entry, a start
    /// for the next unit with the same declaration.
    winners: HashMap<String, ParameterValues>,
    /// The units the census expected to search.
    searching: HashSet<TuningKey>,
    /// The census's measurement of each unit it measured, reported as part
    /// of that unit's tuning.
    measurements: HashMap<TuningKey, (f64, TuningResult)>,
    /// The budget of the units searched so far, of `searching_total`.
    searched: usize,
    searching_total: usize,
}

impl<'a> Tuner<'a> {
    /// A tuner that only counts tuning units, for [`Tuner::budgets`].
    pub fn census(
        device: &'a Device,
        context: TuningContext<'a>,
        limits: TuningLimits,
        weights: TuningWeights<'a>,
    ) -> Self {
        Self::with(
            device,
            context,
            limits,
            weights,
            Allocation::Census(Vec::new()),
            HashSet::new(),
            HashMap::new(),
        )
    }

    /// A tuner within `budgets`. It reports tuning progress from the start:
    /// the budget of the units that search, none searched yet.
    pub fn new(
        device: &'a Device,
        context: TuningContext<'a>,
        limits: TuningLimits,
        weights: TuningWeights<'a>,
        budgets: TuningBudgets,
    ) -> Self {
        let tuner = Self::with(
            device,
            context,
            limits,
            weights,
            Allocation::Budgets(budgets.budgets),
            budgets.searching,
            budgets.measurements,
        );
        tuner.report_progress();
        tuner
    }

    fn with(
        device: &'a Device,
        context: TuningContext<'a>,
        limits: TuningLimits,
        weights: TuningWeights<'a>,
        allocation: Allocation,
        searching: HashSet<TuningKey>,
        measurements: HashMap<TuningKey, (f64, TuningResult)>,
    ) -> Self {
        let searching_total = match &allocation {
            Allocation::Census(_) => 0,
            Allocation::Budgets(budgets) => searching.iter().map(|key| budgets[key]).sum(),
        };
        Self {
            device,
            context,
            limits,
            weights,
            noise: Noise::default(),
            allocation,
            deadline: Instant::now() + SAFETY_STOP,
            tuned: Vec::new(),
            chosen: HashMap::new(),
            winners: HashMap::new(),
            searching,
            measurements,
            searched: 0,
            searching_total,
        }
    }

    /// [`MODEL_BUDGET`] shared among the units a census counted by their
    /// shares of step time, and the units among them that will search: those
    /// without a stored result of their key or a pinned choice. Each
    /// defaults measurement the census made spends one configuration of the
    /// model budget. A unit that spends none of it (pinned, launch-scoped)
    /// takes no share and counts as one configuration in tuning progress.
    pub fn budgets(self) -> TuningBudgets {
        let Allocation::Census(units) = self.allocation else {
            unreachable!("budgets come from a census");
        };
        let budgeted = units
            .iter()
            .filter_map(|unit| {
                unit.defaults.as_ref().map(|(shapes, seconds)| {
                    (
                        unit,
                        UnitTime {
                            launches: unit.launches,
                            shapes,
                            seconds,
                        },
                    )
                })
            })
            .collect::<Vec<_>>();
        let shares = step_shares(&budgeted.iter().map(|(_, time)| *time).collect::<Vec<_>>());
        let measured: usize = units
            .iter()
            .filter_map(|unit| unit.measurement.as_ref())
            .map(|(_, result)| result.configurations.len())
            .sum();
        let mut allocated = allocate(
            MODEL_BUDGET.saturating_sub(measured),
            &budgeted
                .iter()
                .zip(&shares)
                .map(|((unit, _), share)| (unit.size, *share))
                .collect::<Vec<_>>(),
        );
        reserve_form_starts(
            &mut allocated,
            &budgeted
                .iter()
                .map(|(unit, _)| if unit.searches { unit.form_starts } else { 1 })
                .collect::<Vec<_>>(),
        );
        let mut budgets = budgeted
            .iter()
            .zip(allocated)
            .map(|((unit, _), budget)| {
                (
                    unit.key.clone(),
                    budget
                        + unit
                            .measurement
                            .as_ref()
                            .map_or(0, |(_, result)| result.configurations.len()),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut searching = HashSet::new();
        let mut measurements = HashMap::new();
        for unit in units {
            budgets.entry(unit.key.clone()).or_insert(1);
            if unit.searches {
                searching.insert(unit.key.clone());
            }
            if let Some(measurement) = unit.measurement {
                measurements.insert(unit.key, measurement);
            }
        }
        TuningBudgets {
            budgets,
            searching,
            measurements,
        }
    }

    /// Report the budget of the units searched so far, of those that search.
    fn report_progress(&self) {
        self.context.observer.event(&TuningEvent::Progress {
            completed: self.searched,
            total: self.searching_total,
        });
    }

    pub fn tuned(self) -> Vec<TunedEntry> {
        self.tuned
    }

    /// Tune `case` over its implementation's declared domain at `statics`
    /// and return the chosen configuration. An entry already tuned with the
    /// same bindings and static values reuses that result; a stored result
    /// of the same key is used without tuning. A census counts the unit,
    /// measures its defaults when it will search, and returns the defaults.
    pub fn tune<T: EntryTuning>(
        &mut self,
        case: &T,
        implementation: &NativeImplementation,
        statics: &NativeSpecialization,
    ) -> Result<NativeSpecialization, CatalogFailure> {
        let entry = <T::Entry as seismic::Entry>::NAME;
        let bindings = case.bindings();
        let key = (entry, bindings.clone(), statics.statics().clone());
        let failure = |outcome: String| CatalogFailure::Tuning {
            entry,
            bindings: bindings.clone(),
            outcome,
        };
        let configurations = || {
            implementation
                .admissible(statics)
                .map(|admissible| admissible.len())
                .map_err(|error| failure(error.to_string()))
        };
        let budget = match &mut self.allocation {
            Allocation::Census(units) => {
                match units.iter_mut().find(|unit| unit.key == key) {
                    Some(unit) => unit.launches += case.launches(),
                    None => {
                        let unit = self.census_unit(case, implementation, statics, key)?;
                        let Allocation::Census(units) = &mut self.allocation else {
                            unreachable!("a census stays a census");
                        };
                        units.push(unit);
                    }
                }
                return implementation
                    .default_specialization(statics)
                    .map_err(|error| failure(error.to_string()));
            }
            Allocation::Budgets(budgets) => *budgets
                .get(&key)
                .ok_or_else(|| failure("the tuning census did not count this unit".into()))?,
        };
        if let Some(chosen) = self.chosen.get(&key) {
            return Ok(chosen.clone());
        }
        #[cfg(feature = "pinned-tuning")]
        if let pinned::Pinned::Chosen(chosen) =
            pinned::lookup(&key, implementation).map_err(failure)?
        {
            self.chosen.insert(key, chosen.clone());
            return Ok(chosen);
        }
        let shapes = case.points(self.limits);
        if shapes.is_empty() {
            return Err(failure("the engine's bounds admit no tuning point".into()));
        }
        let screening = case.screening(self.device, &shapes);
        let declaration = format!("{entry}:{:?}", implementation.params);
        let began = Instant::now();
        let survey = survey_plan(entry);
        let stored = self
            .stored(case, statics, &shapes, &screening)
            .map_err(failure)?;
        if let Some((_, _, Some(result))) = &stored {
            let tuned = tuned_entry(entry, bindings, result, TuningOrigin::Stored, began);
            return Ok(self.finish(key, declaration, tuned));
        }
        self.context.observer.event(&TuningEvent::Started {
            entry,
            bindings: bindings.clone(),
            configurations: configurations()?,
            points: shapes.len(),
        });
        let surveyed = survey.is_some();
        let strategy = match survey {
            Some(plan) => Strategy::Survey(plan),
            None => Strategy::Search(SearchPlan {
                budget,
                settings: search_settings(self.device.backend(), !screening.is_empty()),
                min_sample_seconds: MIN_SAMPLE_SECONDS,
                start: case
                    .search_starts(self.device, implementation, statics, self.limits)
                    .into_iter()
                    .chain(self.winners.get(&declaration).cloned())
                    .collect(),
                deadline: Some(self.deadline),
                screening,
            }),
        };
        let census = self
            .measurements
            .get(&key)
            .map(|(_, result)| result.clone());
        let result = self
            .run(
                case,
                statics,
                &shapes,
                strategy,
                census.as_ref().map(seismic::TuningReuse::Seed),
            )
            .map_err(failure)?;
        #[cfg(feature = "tuning-survey")]
        if surveyed {
            survey::record(&key, budget, &result).map_err(failure)?;
        }
        let stopped = matches!(
            result.method,
            TuningMethod::Search {
                stop: SearchStop::Expired,
                ..
            } | TuningMethod::Factored {
                complete: false,
                ..
            }
        );
        // A search the safety stop ended is not the search its key names.
        if let Some((cache, key, None)) = &stored {
            if !stopped && !surveyed {
                cache.store_tuning(key, &result);
            }
        }
        let mut tuned = tuned_entry(entry, bindings, &result, TuningOrigin::Searched, began);
        // The census's measurement of the defaults is part of the unit's
        // tuning.
        if let Some((seconds, census)) = self.measurements.remove(&key) {
            let time = census.time;
            tuned.seconds += seconds;
            tuned.time.reference_seconds += time.reference_seconds;
            tuned.time.forming_seconds += time.forming_seconds;
            tuned.time.measuring_seconds += time.measuring_seconds;
            tuned.time.validating_seconds += time.validating_seconds;
        }
        // A unit the census expected to have a stored result (one that proved
        // invalid) adds its budget to the search when it searches.
        if !self.searching.contains(&key) {
            self.searching_total += budget;
        }
        self.searched += budget;
        let chosen = self.finish(key, declaration, tuned);
        self.report_progress();
        Ok(chosen)
    }

    /// Count the unit `key` names: whether it will search at this load, and
    /// its defaults' time at its points when it spends the model budget,
    /// measured when it searches. Mirrors [`Tuner::tune`].
    fn census_unit<T: EntryTuning>(
        &mut self,
        case: &T,
        implementation: &NativeImplementation,
        statics: &NativeSpecialization,
        key: TuningKey,
    ) -> Result<CensusUnit, CatalogFailure> {
        let failure = |outcome: String| CatalogFailure::Tuning {
            entry: key.0,
            bindings: key.1.clone(),
            outcome,
        };
        let admissible = implementation
            .admissible(statics)
            .map_err(|error| failure(error.to_string()))?;
        let defaults = implementation
            .default_specialization(statics)
            .map_err(|error| failure(error.to_string()))?;
        let form_starts = 1 + implementation
            .params
            .iter()
            .filter(|parameter| parameter.form)
            .map(|parameter| {
                admissible
                    .iter()
                    .filter_map(|choice| choice.param(&parameter.name))
                    .filter(|value| Some(*value) != defaults.param(&parameter.name))
                    .collect::<HashSet<_>>()
                    .len()
            })
            .sum::<usize>();
        // The case's own starts (measured good geometries) are first
        // measurements too: a unit left only its defaults would never try
        // them.
        let seeded = case
            .search_starts(self.device, implementation, statics, self.limits)
            .into_iter()
            .filter(|start| *start != *defaults.params())
            .collect::<HashSet<_>>()
            .len();
        let form_starts = (form_starts + seeded).min(admissible.len());
        let mut unit = CensusUnit {
            key: key.clone(),
            size: admissible.len(),
            form_starts,
            launches: case.launches(),
            searches: false,
            defaults: None,
            measurement: None,
        };
        #[cfg(feature = "pinned-tuning")]
        if let pinned::Pinned::Chosen(_) = pinned::lookup(&key, implementation).map_err(failure)? {
            return Ok(unit);
        }
        let shapes = case.points(self.limits);
        if shapes.is_empty() {
            return Err(failure("the engine's bounds admit no tuning point".into()));
        }
        let screening = case.screening(self.device, &shapes);
        match self
            .stored(case, statics, &shapes, &screening)
            .map_err(failure)?
        {
            Some((_, _, Some(result))) => {
                if !implementation.launch_scoped() {
                    unit.defaults = defaults_seconds(&result, &result.overall.params)
                        .map(|seconds| (shapes, seconds));
                }
            }
            Some((_, _, None)) | None => {
                unit.searches = true;
                if !implementation.launch_scoped() {
                    let began = Instant::now();
                    let result = self
                        .run(
                            case,
                            statics,
                            &shapes,
                            Strategy::Census {
                                min_sample_seconds: MIN_SAMPLE_SECONDS,
                                deadline: Some(self.deadline),
                            },
                            None,
                        )
                        .map_err(failure)?;
                    let seconds = defaults_seconds(&result, &result.overall.params)
                        .ok_or_else(|| failure("the defaults were not measured".into()))?;
                    unit.defaults = Some((shapes, seconds));
                    unit.measurement = Some((began.elapsed().as_secs_f64(), result));
                }
            }
        }
        Ok(unit)
    }

    /// The cache slot of the unit `case` is at `statics` and the valid result
    /// it holds; none without a cache or when a survey replaces the unit's
    /// search.
    fn stored<T: EntryTuning>(
        &self,
        case: &T,
        statics: &NativeSpecialization,
        shapes: &[PointShape],
        screening: &[ScreeningPoint],
    ) -> Result<Option<StoredSlot<'a>>, String> {
        let entry = <T::Entry as seismic::Entry>::NAME;
        let Some(cache) = self.context.cache.filter(|_| survey_plan(entry).is_none()) else {
            return Ok(None);
        };
        let digest = case
            .digest(self.device, statics)
            .map_err(|error| error.to_string())?;
        let material = tuning_key_material(
            self.device,
            entry,
            &case.bindings(),
            statics,
            &digest,
            shapes,
            screening,
        );
        let policy = case.precision().map_err(|error| error.to_string())?;
        let key = TuningCacheKey::of(&format!(
            "{material}\npolicy {:?}",
            seismic::precision::PolicyIdentity::of(&policy).0
        ));
        let hit = cache
            .tuning(&key)
            .filter(|result| case.stored_valid(self.device, &result.overall.specialization()));
        Ok(Some((cache, key, hit)))
    }

    /// Tune `case` at `statics` over argument sets built for its `shapes`,
    /// released afterwards.
    fn run<T: EntryTuning>(
        &mut self,
        case: &T,
        statics: &NativeSpecialization,
        shapes: &[PointShape],
        strategy: Strategy,
        reuse: Option<seismic::TuningReuse<'_>>,
    ) -> Result<TuningResult, String> {
        let mut shared = HashMap::new();
        let mut inputs = TuningInputs {
            device: self.device,
            definition: self.context.definition,
            limits: self.limits,
            weights: &mut self.weights,
            noise: &self.noise,
            shared: &mut shared,
        };
        let mut rotations = shapes
            .iter()
            .map(|point| case.rotation(&mut inputs, point))
            .collect::<Result<Vec<_>, _>>()?;
        let guard = guard_point(&rotations, |case| {
            T::state(case)
                .iter()
                .map(|(_, state)| state.backing.byte_len())
                .sum()
        });
        let initializers = rotations
            .iter()
            .enumerate()
            .map(|(point, cases)| {
                initializer(
                    cases
                        .iter()
                        .flat_map(T::state)
                        .map(|(_, state)| state)
                        .collect(),
                    Some(point) == guard,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Every argument set of a point writes the same rows; the guard
        // point declares none, so its states are observed whole.
        let written = rotations
            .iter()
            .enumerate()
            .map(|(point, cases)| {
                cases
                    .iter()
                    .take(1)
                    .filter(|_| Some(point) != guard)
                    .flat_map(T::state)
                    .map(|(name, state)| (name.to_owned(), state.written.clone()))
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        let points = shapes
            .iter()
            .zip(rotations.iter_mut())
            .zip(initializers)
            .zip(written)
            .map(|(((shape, cases), initialize), written)| TuningPoint {
                label: shape.label.clone(),
                weight: shape.weight,
                class: shape.class.clone(),
                rotation: cases.iter_mut().map(T::args).collect(),
                initialize,
                written,
            })
            .collect();
        let result = case
            .tune(
                self.device,
                statics,
                points,
                case.precision().map_err(|e| e.to_string())?,
                strategy,
                reuse,
            )
            .map_err(|error| error.to_string());
        drop(rotations);
        drop(shared);
        self.weights.release();
        result
    }

    /// Record a unit's outcome and report it.
    fn finish(
        &mut self,
        key: TuningKey,
        declaration: String,
        tuned: TunedEntry,
    ) -> NativeSpecialization {
        self.context
            .observer
            .event(&TuningEvent::Finished(tuned.clone()));
        let chosen = tuned.overall.specialization();
        #[cfg(feature = "pinned-tuning")]
        pinned::record(&key, &chosen);
        self.winners
            .insert(declaration, tuned.overall.params.clone());
        self.tuned.push(tuned);
        self.chosen.insert(key, chosen.clone());
        chosen
    }

    pub fn statics<T: EntryTuning>(
        &mut self,
        case: &T,
    ) -> Result<Vec<(&'static str, u64)>, CatalogFailure> {
        let mut shared = HashMap::new();
        let inputs = TuningInputs {
            device: self.device,
            definition: self.context.definition,
            limits: self.limits,
            weights: &mut self.weights,
            noise: &self.noise,
            shared: &mut shared,
        };
        case.statics(&inputs)
            .map_err(|outcome| CatalogFailure::Preparation {
                entry: <T::Entry as seismic::Entry>::NAME,
                bindings: case.bindings(),
                outcome,
            })
    }
}

/// The survey plan replacing `entry`'s search, when a development survey
/// names it (feature `tuning-survey`).
#[cfg(feature = "tuning-survey")]
fn survey_plan(entry: &str) -> Option<seismic::SurveyPlan> {
    survey::plan(entry)
}

#[cfg(not(feature = "tuning-survey"))]
fn survey_plan(_entry: &str) -> Option<seismic::SurveyPlan> {
    None
}

/// Everything a stored tuning result depends on (§C2), rendered
/// canonically: the device with its toolchain and driver, the unit, the
/// implementation's digest, and the search's definition. A unit's budget is
/// not: it follows the measured step-time shares of the load that searched
/// it, so the model budget stands for it.
fn tuning_key_material(
    device: &Device,
    entry: &str,
    bindings: &str,
    statics: &NativeSpecialization,
    digest: &str,
    shapes: &[PointShape],
    screening: &[ScreeningPoint],
) -> String {
    let points = shapes
        .iter()
        .map(|shape| match &shape.class {
            Some(class) => format!("{}={:?}@{class}", shape.label, shape.weight),
            None => format!("{}={:?}", shape.label, shape.weight),
        })
        .collect::<Vec<_>>()
        .join(",");
    let settings = search_settings(device.backend(), !screening.is_empty());
    let screening = if screening.is_empty() {
        String::new()
    } else {
        format!("\nscreening cpu-projection-{CPU_PROJECTION_SCREENING_VERSION} {screening:?}")
    };
    format!(
        "device {}\nentry {entry}\nbindings {bindings}\nstatics {:?}\nimplementation {digest}\n\
         search {SEARCH_VERSION}\nmodel budget {MODEL_BUDGET}\nsettings {settings:?}\n\
         points {points}{screening}\nvalidation bounded-per-subject-v1\nmin sample {MIN_SAMPLE_SECONDS:?}",
        device.tuning_identity(),
        statics.statics(),
    )
}

fn tuned_entry(
    entry: &'static str,
    bindings: String,
    result: &TuningResult,
    origin: TuningOrigin,
    began: Instant,
) -> TunedEntry {
    let measured = result
        .configurations
        .iter()
        .filter(|record| matches!(record.outcome, seismic::Outcome::Measured { .. }))
        .count();
    let (time, search) = match origin {
        TuningOrigin::Stored => (TuningTime::default(), None),
        TuningOrigin::Searched => (
            result.time.clone(),
            match &result.method {
                TuningMethod::Search { budget, stop, .. } => Some((*budget, *stop)),
                TuningMethod::Survey { .. } | TuningMethod::Factored { .. } => None,
            },
        ),
    };
    TunedEntry {
        entry,
        bindings,
        overall: result.overall.clone(),
        origin,
        search,
        measured,
        excluded: result.configurations.len() - measured,
        rejections: result.rejections().count(),
        first_rejection: result
            .rejections()
            .find_map(|record| match &record.outcome {
                seismic::Outcome::Excluded(exclusion) => {
                    Some(format!("{:?}: {exclusion:?}", record.configuration.params))
                }
                seismic::Outcome::Measured { .. } => None,
            }),
        seconds: began.elapsed().as_secs_f64(),
        time,
    }
}

#[cfg(test)]
mod tests;
