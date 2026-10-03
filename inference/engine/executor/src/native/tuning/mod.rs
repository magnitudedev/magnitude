//! Tuning at program preparation (route spec NR12, program spec E6, tuning
//! spec `specs/26-09-24/native-tuning-search-and-caching.md`).
//!
//! Every native entry whose implementation declares tuning parameters is
//! tuned on the opened device on the first load for each tuning key. A count
//! walk finds the units that will search; each then searches the declared
//! domain within its launches' share of [`TUNING_TIME`], building each
//! point's inputs only when Seismic admits the point, and forming,
//! measuring and validating what it reaches. With a [`KernelCache`], each
//! result is stored under a key over what it is valid for (tuning version,
//! device and toolchain identity, unit, implementation digest, precision
//! policy and served shapes). Nothing is shipped.
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

/// Implements [`EntryTuning::tune`] and [`EntryTuning::entry`] through an
/// entry module's generated `native_tune[_with]` and
/// `native_entry[_with]`. `$this => $elements` names the case and the
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
            points: &mut dyn seismic::PointSource<'_, Self::Entry>,
            validation: seismic::PrecisionPolicy,
            strategy: seismic::Strategy,
        ) -> Result<seismic::TuningResult, seismic::TuneError> {
            let $this = self;
            $module::native_tune_with(
                device,
                $elements,
                statics,
                points,
                validation,
                strategy,
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

        fn entry(&self) -> seismic::BoundEntry<Self::Entry> {
            let $this = self;
            $module::native_entry_with($elements)
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
            points: &mut dyn seismic::PointSource<'_, Self::Entry>,
            validation: seismic::PrecisionPolicy,
            strategy: seismic::Strategy,
        ) -> Result<seismic::TuningResult, seismic::TuneError> {
            $module::native_tune(
                device,
                statics,
                points,
                validation,
                strategy,
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

        fn entry(&self) -> seismic::BoundEntry<Self::Entry> {
            $module::native_entry()
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
use crate::ModelLoadPlan;
use magnitude_batching::{ClassLimits, Demand, PackedRowTables, Row, RowHistory, Slot};
use magnitude_family_contracts::{ModelDefinition, Operator, WeightKind, WeightScope};
use seismic::{
    CensusPlan, Configuration, DType, Device, Element, NativeImplementation, NativeSpecialization, ParameterValues, PrecisionPolicy, SearchPlan, SearchSettings, SearchStop,
    Strategy, Tensor, TensorError, TuneError, TuningInitializer, TuningMethod, TuningResult,
    TuningTime,
};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
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
/// GEMV bound); larger counts run tiled GEMMs. Every point of at most these
/// rows is required: they are the row counts at which implementations' code
/// paths differ (the CPU projections' 4-row tiles among them), so a
/// candidate is chosen only when validated at every path it runs in
/// serving.
pub const STREAMING_ROWS: u64 = 8;

/// Version of tuning: part of every stored tuning result's key, so bumping it
/// makes every user retune every model.
///
/// Do not bump it just because tuning changed. A stored result stays valid
/// across changes to the search, its settings, the tuning time or the
/// workload weights: it was validated on this device for this code and
/// precision policy at these shapes, which the rest of the key covers. Bump
/// it only with the maintainers' approval, when tuning has improved enough
/// that retuning every model for its better performance is worth the cost,
/// or when a stored result can no longer be read.
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
/// 16: tuning time instead of a configuration budget (results of 15 cannot be
///     read), and keys without the search's definition.
pub const SEARCH_VERSION: u32 = 16;
/// The time one preparation spends tuning, on any device.
pub const TUNING_TIME: Duration = Duration::from_secs(60);
/// A unit's search may take `1 / ADMISSION_SHARE` of its budget to admit
/// points beyond its census's, building them and measuring the defaults
/// there, leaving the rest to search: about as many evaluations of its
/// points as this share.
pub const ADMISSION_SHARE: u32 = 10;
/// The search's constants (§D2).
pub const SEARCH_SETTINGS: SearchSettings = SearchSettings {
    improvement: 0.01,
    restarts: 2,
    confirmed: 3,
    default_margin: 0.02,
    samples: 3,
    confirmation_samples: 7,
};

/// CPU and Metal candidates are measured with one sample and their finalists
/// confirmed with five: replays of 31 stored CPU searches and of Metal's
/// recorded searches preserved every seven-sample winner at five, while
/// three samples changed one CPU winner.
pub(crate) fn search_settings(backend: seismic::BackendName) -> SearchSettings {
    let mut settings = SEARCH_SETTINGS;
    if matches!(
        backend,
        seismic::BackendName::Cpu | seismic::BackendName::Metal
    ) {
        settings.samples = 1;
        settings.confirmation_samples = 5;
    }
    settings
}
/// Minimum device time of one sample; device timestamps resolve
/// microseconds.
pub const MIN_SAMPLE_SECONDS: f64 = 0.0002;

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

impl PointShape {
    /// The point's cost relative to its unit's other points: its rows, and
    /// for attention each row's visible history.
    pub fn cost(&self) -> f64 {
        match self.context {
            Some(context) => (self.rows * (self.rows + context)) as f64,
            None => self.rows as f64,
        }
    }
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

impl TuningLimits {
    /// The bounds a load under `limits` tunes `definition`'s entries within.
    pub(crate) fn of(limits: crate::ResourceLimits, definition: &ModelDefinition) -> Self {
        Self {
            max_rows: limits.max_launch_rows as u64,
            max_projected_rows: crate::programs::graph::readout::max_projected_rows(limits) as u64,
            context_tokens: definition.decoder.context_limit,
        }
    }
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

/// Entry-specific tuning knowledge. One implementation exists per native
/// entry that declares tuning parameters.
pub(crate) trait EntryTuning {
    type Entry: seismic::Entry;
    fn precision(&self) -> Result<PrecisionPolicy, TuneError>;
    /// One argument set: every tensor its arguments borrow, owned, including
    /// the [`CaseState`]s its `&mut` parameters bind. Kept from a unit's
    /// census for its search.
    type Case: 'static;

    /// The semantic bindings, for reports.
    fn bindings(&self) -> String;
    /// Launches of the entry in one step of a row class it serves: one per
    /// layer the case binds, one for a per-step entry.
    fn launches(&self) -> usize;
    /// The value of every dimension the model fixes.
    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String>;
    fn points(&self, limits: TuningLimits) -> Vec<PointShape>;
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
        points: &mut dyn seismic::PointSource<'_, Self::Entry>,
        validation: PrecisionPolicy,
        strategy: Strategy,
    ) -> Result<TuningResult, TuneError>;
    /// The entry's generated `native_digest[_with]`.
    fn digest(&self, device: &Device, statics: &NativeSpecialization) -> Result<String, TuneError>;
    /// Check a stored choice against the same device-augmented declaration
    /// that native preparation uses.
    fn stored_valid(&self, device: &Device, specialization: &NativeSpecialization) -> bool {
        seismic::generated::native_specialization_valid::<Self::Entry>(device, specialization)
            .unwrap_or(false)
    }
    /// The entry bound to the case's elements (its generated
    /// `native_entry[_with]`).
    fn entry(&self) -> seismic::BoundEntry<Self::Entry>;
}

/// Progress of tuning at load, for readiness reporting.
#[derive(Clone, Debug, PartialEq)]
pub enum TuningEvent {
    /// Milliseconds of tuning time spent so far, of [`TUNING_TIME`].
    /// Reported when the search walk begins and after each unit it
    /// searches; nothing is reported when every unit is stored.
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
    /// The search's allowance and why it stopped (a searched unit).
    pub search: Option<(Duration, SearchStop)>,
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

/// What a case's static values derive from: the model, its load plan and the
/// engine's bounds. No device.
#[derive(Clone, Copy)]
pub(crate) struct ModelInputs<'a> {
    pub definition: &'a ModelDefinition,
    pub limits: TuningLimits,
    pub load: &'a ModelLoadPlan,
}

impl ModelInputs<'_> {
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

    /// The planned shape of one weight role.
    pub fn weight_shape(&self, scope: WeightScope, kind: WeightKind) -> Result<Vec<u64>, String> {
        Ok(weights::weight_plan(self.load, scope, kind)?.shape.clone())
    }

    /// The planned extent of one weight role's accumulator-scale port (0
    /// without a second-level scale).
    pub fn scale_extent(&self, scope: WeightScope, kind: WeightKind) -> Result<u64, String> {
        Ok(weights::weight_plan(self.load, scope, kind)?.scale_extent())
    }
}

pub(crate) struct TuningInputs<'w, 'a> {
    model: ModelInputs<'a>,
    pub device: &'a Device,
    pub weights: &'w mut TuningWeights<'a>,
    noise: &'w Noise,
    /// Tensors shared by the units of one tuning, by a name that identifies
    /// their contents.
    shared: &'w mut HashMap<String, Tensor>,
    /// What the point being built may take; a weight import or activation
    /// predicted not to fit is refused.
    building: &'w RefCell<BuildBudget>,
}

impl<'a> std::ops::Deref for TuningInputs<'_, 'a> {
    type Target = ModelInputs<'a>;

    fn deref(&self) -> &ModelInputs<'a> {
        &self.model
    }
}

impl TuningInputs<'_, '_> {
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
        if self.weights.resident(scope, kind) {
            return self.weights.weight(scope, kind);
        }
        let bytes = self.weights.bytes(scope, kind)?;
        self.building.borrow_mut().admit(Building::Import, bytes)?;
        let started = Instant::now();
        let weight = self.weights.weight(scope, kind)?;
        self.building
            .borrow_mut()
            .record(Building::Import, bytes, started.elapsed().as_secs_f64());
        Ok(weight)
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
        let dtype = element
            .dtype()
            .ok_or_else(|| format!("tuning activations are not {}", element.name()))?;
        self.generated(element, extents, |count| {
            self.noise.window(dtype, count, seed)
        })
    }

    /// A tensor of `element` at `extents` whose bytes `generate` makes from
    /// its element count. Generating it is refused when predicted not to fit
    /// the point's time, and measured.
    pub fn generated(
        &self,
        element: Element,
        extents: &[u64],
        generate: impl FnOnce(usize) -> Result<Vec<u8>, String>,
    ) -> Result<Tensor, String> {
        let count = element_count(extents)?;
        let width = element
            .dtype()
            .ok_or_else(|| format!("generated tuning tensors are not {}", element.name()))?
            .bytes();
        let size = (count as u64).saturating_mul(u64::from(width));
        self.building
            .borrow_mut()
            .admit(Building::Generation, size)?;
        let started = Instant::now();
        let bytes = generate(count)?;
        let tensor = Tensor::from_host(self.device, element, extents, &bytes)
            .map_err(|error| error.to_string())?;
        self.building.borrow_mut().record(
            Building::Generation,
            size,
            started.elapsed().as_secs_f64(),
        );
        Ok(tensor)
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

/// Where a preparation's tuning is. One [`Tuner`] walks the program once per
/// phase.
enum Phase {
    /// Recording each unit, its launches per step and whether it will
    /// search; nothing is formed or measured.
    Count {
        launches: HashMap<TuningKey, usize>,
        searching: Vec<TuningKey>,
    },
    /// Measuring each searching unit's defaults at the points every
    /// candidate must pass within the tuning time left, keeping the inputs
    /// built for its search.
    Census {
        launches: HashMap<TuningKey, usize>,
        censused: HashMap<TuningKey, Censused>,
    },
    /// Searching each censused unit within its budget: its share of step
    /// time's part of the tuning time left, among the units still to search.
    /// Time a unit leaves goes to the units after it, and an overrun is
    /// taken from them, so tuning ends within the tuning time.
    Search {
        searching: HashMap<TuningKey, Searching>,
        /// The shares of the units not yet searched, summed.
        unsearched: f64,
    },
}

/// A unit as its census left it.
struct Censused {
    launches: usize,
    /// The unit's points, ascending in cost.
    shapes: Vec<PointShape>,
    /// The census's wall time.
    seconds: f64,
    result: TuningResult,
}

/// A unit at the search: its census and its share of step time.
struct Searching {
    census: TuningResult,
    /// The census's wall time.
    seconds: f64,
    share: f64,
}

/// One tuning unit's step time as its census measured it: its launches per
/// step, its timed points and its defaults' time at each.
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
/// more of the step and gets more of the time.
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

/// The defaults' time a census measured at each of its timed points, with
/// those points' shapes.
fn censused_times(shapes: &[PointShape], census: &TuningResult) -> (Vec<PointShape>, Vec<f64>) {
    let Some(measured) = census
        .configurations
        .iter()
        .find(|record| record.configuration == census.overall)
        .and_then(|record| match &record.outcome {
            seismic::Outcome::Measured { points, .. } => Some(points),
            seismic::Outcome::Excluded(_) => None,
        })
    else {
        return (Vec::new(), Vec::new());
    };
    shapes
        .iter()
        .filter_map(|shape| {
            measured
                .iter()
                .find(|point| point.point == shape.label)
                .map(|point| (shape.clone(), point.median_seconds))
        })
        .unzip()
}

/// One unit's built cases by point, kept from its census for its search,
/// and the point whose states are observed whole: the first built with
/// states, the cheapest, so the guard costs a short history.
struct UnitCases<C> {
    points: Vec<Option<Vec<C>>>,
    guard: Option<usize>,
}

/// A unit's slot in the kernel cache: the cache, the unit's key, and the
/// valid result stored under it.
type StoredSlot<'c> = (&'c KernelCache, TuningCacheKey, Option<TuningResult>);

/// Resolves every native entry's specialization for the opened device:
/// static values, tuned parameters, and missing implementations. Tuning
/// spends about [`TUNING_TIME`] from the start of the census. The census
/// measures each unit's defaults at the points every candidate must pass,
/// within the tuning time left (a unit whose required points do not fit
/// keeps its defaults); each unit then searches within its share of step
/// time's part of the time left among the units still to search.
pub(crate) struct Tuner<'a> {
    device: &'a Device,
    context: TuningContext<'a>,
    limits: TuningLimits,
    weights: TuningWeights<'a>,
    noise: Noise,
    /// Tensors shared by the units of this tuning (history planes).
    shared: HashMap<String, Tensor>,
    /// What building inputs costs on this machine, as measured so far.
    building: RefCell<BuildBudget>,
    /// Each censused unit's built cases ([`UnitCases`] of its case type),
    /// kept for its search.
    built: HashMap<TuningKey, Box<dyn std::any::Any>>,
    phase: Phase,
    /// When the census began: the start of the tuning time.
    started: Instant,
    tuned: Vec<TunedEntry>,
    chosen: HashMap<TuningKey, NativeSpecialization>,
    /// The latest choice for each parameter declaration of an entry, a start
    /// for the next unit with the same declaration.
    winners: HashMap<String, ParameterValues>,
}

impl<'a> Tuner<'a> {
    /// A tuner that counts the units of a walk: the first phase.
    pub fn count(
        device: &'a Device,
        context: TuningContext<'a>,
        limits: TuningLimits,
        weights: TuningWeights<'a>,
    ) -> Self {
        Self {
            device,
            context,
            limits,
            weights,
            noise: Noise::default(),
            shared: HashMap::new(),
            building: RefCell::new(BuildBudget::default()),
            built: HashMap::new(),
            phase: Phase::Count {
                launches: HashMap::new(),
                searching: Vec::new(),
            },
            started: Instant::now(),
            tuned: Vec::new(),
            chosen: HashMap::new(),
            winners: HashMap::new(),
        }
    }

    /// Begin the census of the counted units that will search: the tuning
    /// time starts.
    pub fn census(mut self) -> Self {
        let Phase::Count {
            launches,
            searching,
        } = self.phase
        else {
            unreachable!("a census follows the count");
        };
        let launches = searching
            .into_iter()
            .map(|key| {
                let count = launches[&key];
                (key, count)
            })
            .collect::<HashMap<_, _>>();
        let searches = !launches.is_empty();
        self.phase = Phase::Census {
            launches,
            censused: HashMap::new(),
        };
        self.started = Instant::now();
        if searches {
            self.report_progress();
        }
        self
    }

    /// Begin the search: each censused unit's share of step time decides its
    /// part of the tuning time left.
    pub fn search(mut self) -> Self {
        let Phase::Census { censused, .. } = self.phase else {
            unreachable!("a search follows the census");
        };
        let units = censused.into_iter().collect::<Vec<_>>();
        let times = units
            .iter()
            .map(|(_, unit)| censused_times(&unit.shapes, &unit.result))
            .collect::<Vec<_>>();
        let shares = step_shares(
            &units
                .iter()
                .zip(&times)
                .map(|((_, unit), (shapes, seconds))| UnitTime {
                    launches: unit.launches,
                    shapes,
                    seconds,
                })
                .collect::<Vec<_>>(),
        );
        let searching = units
            .into_iter()
            .zip(shares)
            .map(|((key, unit), share)| {
                (
                    key,
                    Searching {
                        census: unit.result,
                        seconds: unit.seconds,
                        share,
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let unsearched = searching.values().map(|unit| unit.share).sum();
        self.phase = Phase::Search {
            searching,
            unsearched,
        };
        self
    }

    /// Report the tuning time spent so far, of [`TUNING_TIME`].
    fn report_progress(&self) {
        let total = TUNING_TIME.as_millis() as usize;
        self.context.observer.event(&TuningEvent::Progress {
            completed: (self.started.elapsed().as_millis() as usize).min(total),
            total,
        });
    }

    pub fn tuned(self) -> Vec<TunedEntry> {
        self.tuned
    }

    /// The unit's tuning points, ascending in cost.
    fn shapes<T: EntryTuning>(&self, case: &T) -> Vec<PointShape> {
        let mut shapes = case.points(self.limits);
        shapes.sort_by(|left, right| left.cost().total_cmp(&right.cost()));
        shapes
    }

    /// Tune `case` over its implementation's declared domain at `statics`
    /// and return the chosen configuration. An entry already tuned with the
    /// same bindings and static values reuses that result; a stored result
    /// of the same key is used without tuning. The count returns the
    /// defaults.
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
        let defaults = implementation
            .default_specialization(statics)
            .map_err(|error| failure(error.to_string()))?;
        let shapes = self.shapes(case);
        if shapes.is_empty() {
            return Err(failure("the engine's bounds admit no tuning point".into()));
        }
        match self.phase {
            Phase::Count { .. } => self.count_unit(case, implementation, statics, key, &shapes),
            Phase::Census { .. } => self.census_unit(case, implementation, statics, key, shapes),
            Phase::Search { .. } => self
                .search_unit(case, implementation, statics, key, shapes)
                .map(Some),
        }
        .map(|chosen| chosen.unwrap_or(defaults))
    }

    /// Count one unit: its launches per step, and whether it will search.
    fn count_unit<T: EntryTuning>(
        &mut self,
        case: &T,
        implementation: &NativeImplementation,
        statics: &NativeSpecialization,
        key: TuningKey,
        shapes: &[PointShape],
    ) -> Result<Option<NativeSpecialization>, CatalogFailure> {
        let failure = |outcome: String| CatalogFailure::Tuning {
            entry: key.0,
            bindings: key.1.clone(),
            outcome,
        };
        let Phase::Count { launches, .. } = &mut self.phase else {
            unreachable!("the phase is the count");
        };
        if let Some(count) = launches.get_mut(&key) {
            *count += case.launches();
            return Ok(None);
        }
        launches.insert(key.clone(), case.launches());
        #[cfg(feature = "pinned-tuning")]
        if let pinned::Pinned::Chosen(_) = pinned::lookup(&key, implementation).map_err(failure)? {
            return Ok(None);
        }
        #[cfg(not(feature = "pinned-tuning"))]
        let _ = implementation;
        let stored = matches!(
            self.stored(case, statics, shapes).map_err(failure)?,
            Some((_, _, Some(_)))
        );
        if !stored {
            let Phase::Count { searching, .. } = &mut self.phase else {
                unreachable!("the phase is the count");
            };
            searching.push(key);
        }
        Ok(None)
    }

    /// Measure one searching unit's defaults at the points every candidate
    /// must pass, within the tuning time left, keeping the inputs built for
    /// its search.
    fn census_unit<T: EntryTuning>(
        &mut self,
        case: &T,
        implementation: &NativeImplementation,
        statics: &NativeSpecialization,
        key: TuningKey,
        shapes: Vec<PointShape>,
    ) -> Result<Option<NativeSpecialization>, CatalogFailure> {
        let failure = |outcome: String| CatalogFailure::Tuning {
            entry: key.0,
            bindings: key.1.clone(),
            outcome,
        };
        let Phase::Census { launches, censused } = &self.phase else {
            unreachable!("the phase is the census");
        };
        let Some(&count) = launches.get(&key) else {
            return Ok(None);
        };
        if censused.contains_key(&key) {
            return Ok(None);
        }
        let plan = CensusPlan {
            limit: TUNING_TIME.saturating_sub(self.started.elapsed()),
            min_sample_seconds: MIN_SAMPLE_SECONDS,
        };
        #[cfg(feature = "pinned-tuning")]
        if let pinned::Pinned::Chosen(_) = pinned::lookup(&key, implementation).map_err(failure)? {
            return Ok(None);
        }
        #[cfg(not(feature = "pinned-tuning"))]
        let _ = implementation;
        let began = Instant::now();
        let result = self
            .run(case, statics, &key, &shapes, Strategy::Census(plan))
            .map_err(failure)?;
        let Phase::Census { censused, .. } = &mut self.phase else {
            unreachable!("the phase is the census");
        };
        censused.insert(
            key,
            Censused {
                launches: count,
                shapes,
                seconds: began.elapsed().as_secs_f64(),
                result,
            },
        );
        self.report_progress();
        Ok(None)
    }

    /// Search one unit within its budget.
    fn search_unit<T: EntryTuning>(
        &mut self,
        case: &T,
        implementation: &NativeImplementation,
        statics: &NativeSpecialization,
        key: TuningKey,
        shapes: Vec<PointShape>,
    ) -> Result<NativeSpecialization, CatalogFailure> {
        let entry = key.0;
        let bindings = key.1.clone();
        let failure = |outcome: String| CatalogFailure::Tuning {
            entry,
            bindings: bindings.clone(),
            outcome,
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
        let declaration = format!("{entry}:{:?}", implementation.params);
        let began = Instant::now();
        let survey = survey_plan(entry);
        let stored = self.stored(case, statics, &shapes).map_err(failure)?;
        if let Some((_, _, Some(result))) = &stored {
            let tuned = tuned_entry(entry, bindings, result, TuningOrigin::Stored, began);
            return Ok(self.finish(key, declaration, tuned));
        }
        let Phase::Search {
            searching,
            unsearched,
        } = &mut self.phase
        else {
            unreachable!("the phase is the search");
        };
        let unit = searching
            .remove(&key)
            .ok_or_else(|| failure("the tuning census did not measure this unit".into()))?;
        let budget = if *unsearched > 0. {
            TUNING_TIME
                .saturating_sub(self.started.elapsed())
                .mul_f64((unit.share / *unsearched).min(1.))
        } else {
            Duration::ZERO
        };
        *unsearched -= unit.share;
        self.context.observer.event(&TuningEvent::Started {
            entry,
            bindings: bindings.clone(),
            configurations: implementation
                .admissible(statics)
                .map_err(|error| failure(error.to_string()))?
                .len(),
            points: shapes.len(),
        });
        let surveyed = survey.is_some();
        // A census that could not fit the unit's required points in the
        // tuning time kept its defaults: there is nothing to search.
        let affordable = !unit.census.configurations.is_empty();
        let result = match survey {
            Some(plan) => self
                .run(case, statics, &key, &shapes, Strategy::Survey(plan))
                .map_err(failure)?,
            None if !affordable => unit.census.clone(),
            None => {
                let plan = SearchPlan {
                    allowance: budget,
                    admission: budget / ADMISSION_SHARE,
                    // The census timed the required points.
                    required: budget,
                    settings: search_settings(self.device.backend()),
                    min_sample_seconds: MIN_SAMPLE_SECONDS,
                    start: case
                        .search_starts(self.device, implementation, statics, self.limits)
                        .into_iter()
                        .chain(self.winners.get(&declaration).cloned())
                        .collect(),
                };
                self.run(
                    case,
                    statics,
                    &key,
                    &shapes,
                    Strategy::Censused {
                        plan,
                        census: unit.census.clone(),
                    },
                )
                .map_err(failure)?
            }
        };
        #[cfg(feature = "tuning-survey")]
        if surveyed {
            survey::record(&key, budget, &result).map_err(failure)?;
        }
        if let Some((cache, key, None)) = &stored {
            if !surveyed {
                cache.store_tuning(key, &result);
            }
        }
        let mut tuned = tuned_entry(entry, bindings, &result, TuningOrigin::Searched, began);
        // The census's measurement of the defaults is part of the unit's
        // tuning.
        let census = &unit.census.time;
        tuned.seconds += unit.seconds;
        tuned.time.building_seconds += census.building_seconds;
        tuned.time.reference_seconds += census.reference_seconds;
        tuned.time.forming_seconds += census.forming_seconds;
        tuned.time.measuring_seconds += census.measuring_seconds;
        tuned.time.validating_seconds += census.validating_seconds;
        let chosen = self.finish(key, declaration, tuned);
        self.report_progress();
        Ok(chosen)
    }

    /// The cache slot of the unit `case` is at `statics` and the valid result
    /// it holds; none without a cache or when a survey replaces the unit's
    /// search.
    fn stored<T: EntryTuning>(
        &self,
        case: &T,
        statics: &NativeSpecialization,
        shapes: &[PointShape],
    ) -> Result<Option<StoredSlot<'a>>, String> {
        let entry = <T::Entry as seismic::Entry>::NAME;
        let Some(cache) = self.context.cache.filter(|_| survey_plan(entry).is_none()) else {
            return Ok(None);
        };
        let digest = case
            .digest(self.device, statics)
            .map_err(|error| error.to_string())?;
        let policy = case.precision().map_err(|error| error.to_string())?;
        let key = TuningCacheKey::of(&tuning_key_material(
            self.device,
            entry,
            &case.bindings(),
            statics,
            &digest,
            &policy,
            shapes,
        ));
        let hit = cache
            .tuning(&key)
            .filter(|result| case.stored_valid(self.device, &result.overall.specialization()));
        Ok(Some((cache, key, hit)))
    }

    /// Tune `case` at `statics` at `shapes`, building each point's inputs
    /// only when Seismic admits it. A census keeps the inputs it built for
    /// the unit's search, which releases them.
    fn run<T: EntryTuning>(
        &mut self,
        case: &T,
        statics: &NativeSpecialization,
        key: &TuningKey,
        shapes: &[PointShape],
        strategy: Strategy,
    ) -> Result<TuningResult, String> {
        let precision = case.precision().map_err(|error| error.to_string())?;
        let keep = matches!(strategy, Strategy::Census(_));
        let mut cases = self
            .built
            .remove(key)
            .map(|built| {
                *built
                    .downcast::<UnitCases<T::Case>>()
                    .expect("a unit's cases are its case type's")
            })
            .unwrap_or_else(|| UnitCases {
                points: shapes.iter().map(|_| None).collect(),
                guard: None,
            });
        let mut points = UnitPoints {
            case,
            shapes,
            inputs: TuningInputs {
                model: self.model(),
                device: self.device,
                weights: &mut self.weights,
                noise: &self.noise,
                shared: &mut self.shared,
                building: &self.building,
            },
            cases: &mut cases,
        };
        let result = case
            .tune(self.device, statics, &mut points, precision, strategy)
            .map_err(|error| error.to_string());
        drop(points);
        if keep {
            self.built.insert(key.clone(), Box::new(cases));
        }
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

    /// What the static values of this tuning's cases derive from.
    pub fn model(&self) -> ModelInputs<'a> {
        ModelInputs {
            definition: self.context.definition,
            limits: self.limits,
            load: self.weights.load(),
        }
    }
}

/// One unit's points, whose inputs are built when Seismic admits them, once
/// for the unit's census and search.
struct UnitPoints<'u, 'w, 'a, T: EntryTuning> {
    case: &'u T,
    shapes: &'u [PointShape],
    inputs: TuningInputs<'w, 'a>,
    cases: &'u mut UnitCases<T::Case>,
}

impl<T: EntryTuning> seismic::PointSource<'static, T::Entry> for UnitPoints<'_, '_, '_, T> {
    fn points(&self) -> Vec<seismic::PointSpec> {
        self.shapes
            .iter()
            .map(|shape| seismic::PointSpec {
                label: shape.label.clone(),
                weight: shape.weight,
                class: shape.class.clone(),
                cost: shape.cost(),
                required: shape.rows <= STREAMING_ROWS,
            })
            .collect()
    }

    fn build(
        &mut self,
        point: usize,
        limit: Duration,
    ) -> Result<seismic::PointInputs<'static>, seismic::PointUnavailable> {
        if self.cases.points[point].is_none() {
            self.inputs.building.borrow_mut().limit(limit);
            let built = self.case.rotation(&mut self.inputs, &self.shapes[point]);
            let refused = self.inputs.building.borrow().refused();
            let built = built.map_err(|error| {
                if refused {
                    seismic::PointUnavailable::Unaffordable
                } else {
                    seismic::PointUnavailable::Failed(error)
                }
            })?;
            if self.cases.guard.is_none()
                && built.first().is_some_and(|case| {
                    T::state(case)
                        .iter()
                        .any(|(_, state)| state.backing.byte_len() > 0)
                })
            {
                self.cases.guard = Some(point);
            }
            self.cases.points[point] = Some(built);
        }
        let guard = self.cases.guard == Some(point);
        let cases = self.cases.points[point]
            .as_mut()
            .expect("the point's cases were built above");
        let initialize = initializer(
            cases
                .iter()
                .flat_map(T::state)
                .map(|(_, state)| state)
                .collect(),
            guard,
        )
        .map_err(seismic::PointUnavailable::Failed)?;
        // Every argument set of a point writes the same rows; the guard
        // declares none, so its states are observed whole.
        let written = cases
            .iter()
            .take(1)
            .filter(|_| !guard)
            .flat_map(T::state)
            .map(|(name, state)| (name.to_owned(), state.written.clone()))
            .collect();
        Ok(seismic::point_inputs::<T::Entry>(
            cases.iter_mut().map(T::args).collect(),
            initialize,
            written,
        ))
    }
}

/// What building a point's inputs may take, and what building costs on
/// this machine: seconds per byte of importing weights and of generating
/// activations, measured as tuning goes.
#[derive(Default)]
pub(crate) struct BuildBudget {
    deadline: Option<Instant>,
    refused: bool,
    import: Rate,
    generation: Rate,
}

/// Seconds per byte of one kind of building, as measured so far.
#[derive(Clone, Copy, Default)]
struct Rate {
    seconds: f64,
    bytes: f64,
}

impl Rate {
    /// The time of `bytes`, once anything was measured.
    fn predict(self, bytes: u64) -> Option<f64> {
        (self.bytes > 0.).then(|| self.seconds * bytes as f64 / self.bytes)
    }
}

/// The kinds of building whose cost is predicted.
#[derive(Clone, Copy)]
enum Building {
    Import,
    Generation,
}

impl BuildBudget {
    /// Allow the next point's building `limit`.
    fn limit(&mut self, limit: Duration) {
        self.deadline = Instant::now().checked_add(limit);
        self.refused = false;
    }

    /// Whether the last building was refused for its predicted time.
    fn refused(&self) -> bool {
        self.refused
    }

    fn rate(&mut self, kind: Building) -> &mut Rate {
        match kind {
            Building::Import => &mut self.import,
            Building::Generation => &mut self.generation,
        }
    }

    /// Admit building `bytes` of `kind` when its predicted time fits.
    fn admit(&mut self, kind: Building, bytes: u64) -> Result<(), String> {
        let predicted = self.rate(kind).predict(bytes);
        let fits = match (self.deadline, predicted) {
            (Some(deadline), Some(seconds)) => {
                Instant::now() + Duration::from_secs_f64(seconds) <= deadline
            }
            _ => true,
        };
        if fits {
            Ok(())
        } else {
            self.refused = true;
            Err("building the point's inputs would not fit its time".into())
        }
    }

    /// Record that building `bytes` of `kind` took `seconds`.
    fn record(&mut self, kind: Building, bytes: u64, seconds: f64) {
        let rate = self.rate(kind);
        rate.seconds += seconds;
        rate.bytes += bytes as f64;
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

/// What a stored tuning result is valid for (§C2), rendered canonically: the
/// tuning version, the device with its toolchain and driver, the unit, the
/// implementation's digest, the precision policy its choice passed, and the
/// shapes it was validated at. How it was searched (settings, tuning time,
/// workload weights, the points timed) is not: a better search never
/// invalidates a result that is still correct; see [`SEARCH_VERSION`].
fn tuning_key_material(
    device: &Device,
    entry: &str,
    bindings: &str,
    statics: &NativeSpecialization,
    digest: &str,
    policy: &PrecisionPolicy,
    shapes: &[PointShape],
) -> String {
    let mut shapes = shapes
        .iter()
        .map(|shape| shape.label.as_str())
        .collect::<Vec<_>>();
    shapes.sort_unstable();
    format!(
        "tuning {SEARCH_VERSION}\ndevice {}\nentry {entry}\nbindings {bindings}\nstatics {:?}\n\
         implementation {digest}\npolicy {:?}\nshapes {}",
        device.tuning_identity(),
        statics.statics(),
        seismic::precision::PolicyIdentity::of(policy).0,
        shapes.join(","),
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
                TuningMethod::Search {
                    allowance_seconds,
                    stop,
                    ..
                } => Some((Duration::from_secs_f64(*allowance_seconds), *stop)),
                TuningMethod::Factored {
                    allowance_seconds,
                    complete,
                    ..
                } => Some((
                    Duration::from_secs_f64(*allowance_seconds),
                    if *complete {
                        SearchStop::Exhausted
                    } else {
                        SearchStop::Expired
                    },
                )),
                TuningMethod::Survey { .. } | TuningMethod::Census => None,
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
