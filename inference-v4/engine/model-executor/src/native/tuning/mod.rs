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
//! search definition); a later load with the same key prepares the stored
//! choice without forming or measuring anything to tune. Nothing is shipped.
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
//!   entry's `&mut` parameters bind [`CaseState`]s, whose written region is
//!   restored before each configuration's validation run (Seismic's
//!   `TuningPoint::initialize`); real state is never bound.
//!
//! Every entry shares one validation rule, [`ARITHMETIC_TOLERANCE`].
//!
//! Adding an entry: implement [`EntryTuning`] for a case type in the module of
//! its block family, and prepare the entry through `Specializer::tuned` at
//! its preparation call site. An entry that declares parameters but is
//! prepared through `Specializer::fixed` fails preparation with a typed error
//! naming it.
//!
//! Entries prepared more than once with identical element bindings and static
//! values (the MTP head's blocks share the target's shapes) are tuned once:
//! the tuner reuses the first result.

/// Implements [`EntryTuning::tune`] and [`EntryTuning::prepare`] through an
/// entry module's generated `native_tune[_with]` and
/// `native_for_device[_with]`. `$this => $elements` names the case and the
/// entry's element bindings built from it.
macro_rules! generated_entry {
    ($module:ident, $this:ident => $elements:expr) => {
        fn tune(
            &self,
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
            points: Vec<seismic::TuningPoint<'_, Self::Entry>>,
            validation: seismic::Validation,
            strategy: seismic::Strategy,
        ) -> Result<seismic::TuningResult, seismic::TuneError> {
            let $this = self;
            $module::native_tune_with(device, $elements, statics, points, validation, strategy)
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
        fn tune(
            &self,
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
            points: Vec<seismic::TuningPoint<'_, Self::Entry>>,
            validation: seismic::Validation,
            strategy: seismic::Strategy,
        ) -> Result<seismic::TuningResult, seismic::TuneError> {
            $module::native_tune(device, statics, points, validation, strategy)
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
#[cfg(feature = "pinned-tuning")]
pub mod pinned;
pub(crate) mod readout;
#[cfg(feature = "tuning-survey")]
pub mod survey;
pub(crate) mod routed;
pub(crate) mod recurrent;
mod weights;

pub(crate) use weights::TuningWeights;
pub use weights::{TuningWeightSource, ZeroTuningWeights};

use super::{CatalogFailure, MissingImplementation};
use crate::kernel_cache::{KernelCache, TuningCacheKey};
use magnitude_model_batching::{Demand, PackedRowTables, Row, Slot};
use magnitude_model_contracts::{ModelDefinition, WeightKind, WeightScope};
use seismic::{
    Configuration, DType, Device, Element, NativeImplementation, NativeKernel,
    NativeSpecialization, ParameterValues, SearchPlan, SearchSettings, SearchStop, Strategy,
    Tensor, TensorError, TuneError, TuningInitializer, TuningMethod, TuningPoint, TuningResult,
    TuningTime, Validation,
};
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The row counts whose shares of step time weigh the objective (§5.3).
pub const TUNING_ROWS: [u64; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
/// The row counts measured: one decode row, a verification/concurrency
/// batch, a small prefill chunk, and two large ones up to the largest row
/// class (`MAX_CLASS_ROWS`). Every row count of
/// [`TUNING_ROWS`] an entry serves lends its share to the nearest (in
/// log2, ties to the larger) representative the entry serves.
pub const REPRESENTATIVE_ROWS: [u64; 5] = [1, 4, 32, 256, 512];
/// History lengths of attention tuning points (§5.3).
pub const TUNING_CONTEXTS: [u64; 4] = [256, 4096, 16384, 65536];
/// Distinct layers a decode-row rotation cycles through where the model has
/// them. Decode rows stream every weight once per call, so repeated calls
/// must not find the weights cache resident. A prefill chunk reuses each
/// weight across its row tile and is compute bound, so one argument set
/// measures it.
pub const ROTATION_LAYERS: usize = 4;
/// The largest row count at which projections stream their weights (the K1
/// GEMV bound); larger counts run tiled GEMMs.
pub const STREAMING_ROWS: u64 = 8;

/// The agreement tuning requires of every entry's configurations whose
/// arithmetic parameters differ from the defaults': each dense result within
/// 5% of the reference's norm (Seismic `Validation::Relative`); integer
/// results exact. Configurations sharing the defaults' arithmetic parameters
/// (every configuration of an entry with only mapping parameters) must be
/// bit-exact.
///
/// This is a defect guard, not the precision gate. D4's end-to-end verdict
/// (`forward_bench qualify`) is the gate, and the guard must never reject a
/// configuration that would pass it. Error model, for a K-term projection
/// `y = W x` whose activations are quantized to q8_1 (the most lossy
/// arithmetic option any entry declares):
///
/// - q8_1 rounds each element of a 32-element block to a step of
///   `max|x| / 127`; the error is uniform within half a step, rms
///   `step / √12`. With `max/rms ≈ 2.4` for a Gaussian block (up to `√32 ≈
///   5.7` when one outlier holds the block), the relative activation error
///   is `2.4 / (127 · √12) ≈ 0.55%` (up to 1.3%).
/// - Independent errors through `W` keep that relative size in `‖y‖`
///   independently of K (the error norm and the output norm both grow as
///   `√K`); bf16 output rounding adds `2⁻⁹ / √3 ≈ 0.11%`, f16 operands
///   `≈ 0.03%`, reassociated f32 sums (split-K, chunked scans, partitioned
///   softmax) far less.
/// - llama.cpp CUDA runs every projection this way and passes D4 (mean KL
///   0.0043 against the 0.010 bound). KL grows with the square of the
///   per-layer error, so D4 admits per-layer errors up to about
///   `√(0.010 / 0.0043) ≈ 1.5×` q8_1's, i.e. below ~2%.
///
/// A 5% bound is over twice the largest per-layer error D4 can admit, so any
/// configuration it rejects would fail D4 too, while wrong results (a bad
/// index, a missing term) miss by O(1) and are caught.
pub const ARITHMETIC_TOLERANCE: Validation = Validation::Relative { error: 0.05 };

/// Version of the search procedure: part of every stored tuning result's
/// key. Bump it whenever the search could choose differently given the same
/// measurements.
pub const SEARCH_VERSION: u32 = 1;
/// Configurations one model's tuning may evaluate in all (`B_model`,
/// §D3), shared equally among its tuning units.
pub const MODEL_BUDGET: usize = 100;
/// The search's constants (§D2).
pub const SEARCH_SETTINGS: SearchSettings = SearchSettings {
    improvement: 0.01,
    restarts: 2,
    confirmed: 3,
    default_margin: 0.02,
    samples: 3,
    confirmation_samples: 7,
};
/// Minimum device time of one sample; device timestamps resolve
/// microseconds.
pub const MIN_SAMPLE_SECONDS: f64 = 0.0002;
/// The safety stop of one preparation's tuning: past it every search ends
/// with the best found so far, and its result is not stored. It exists for
/// pathological machines; budgets, not time, bound tuning otherwise.
pub const SAFETY_STOP: Duration = Duration::from_secs(120);

/// One workload an entry serves: its shape and its share of expected step
/// time.
#[derive(Clone, Debug, PartialEq)]
pub struct PointShape {
    pub label: String,
    pub weight: f64,
    pub rows: u64,
    /// Visible history rows for attention points.
    pub context: Option<u64>,
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
    }
}

/// Row points up to the engine's row bound.
pub fn row_points(limits: TuningLimits) -> Vec<PointShape> {
    served_row_points(limits.max_rows, |_| true)
}

/// Row points up to `bound` among the row counts an entry serves: the
/// representatives of those row counts, each weighted by the shares of the
/// row counts nearest to it. An entry whose served rows all lie beyond the
/// bound is still prepared (its graph classes do not exist, but the kernel
/// set is complete); its smallest served row count stands in as the single
/// point.
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
    let Some(&largest) = within.last() else {
        return served
            .into_iter()
            .take(1)
            .map(|rows| PointShape {
                weight: 1.0,
                ..row_point(rows)
            })
            .collect();
    };
    let representatives = REPRESENTATIVE_ROWS
        .into_iter()
        .filter(|rows| within.contains(rows))
        .collect::<Vec<_>>();
    let representatives = if representatives.is_empty() {
        vec![largest]
    } else {
        representatives
    };
    let distance = |rows: u64, to: u64| (rows.ilog2() as i64 - to.ilog2() as i64).abs();
    let mut points = representatives
        .iter()
        .map(|&rows| PointShape {
            weight: 0.0,
            ..row_point(rows)
        })
        .collect::<Vec<_>>();
    for rows in within {
        // Ties go to the larger representative: two rows are a batch.
        let nearest = points
            .iter_mut()
            .rev()
            .min_by_key(|point| distance(rows, point.rows))
            .expect("an entry has at least one representative");
        nearest.weight += row_share(rows);
    }
    normalized(points)
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
/// the leading-axis rows the entry writes, and their initial contents, which
/// are restored before each configuration's validation run.
pub(crate) struct CaseState {
    tensor: Tensor,
    written: Tensor,
    initial: Arc<[u8]>,
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
            initial: self.initial.clone(),
        }
    }

    fn restorer(&self) -> impl FnMut() -> Result<(), TensorError> + 'static {
        let mut written = self.written.clone();
        let initial = self.initial.clone();
        move || written.write_from_host(&initial)
    }
}

/// Restores every state of a point's validation argument set.
fn initializer(states: Vec<&CaseState>) -> Option<TuningInitializer<'static>> {
    if states.is_empty() {
        return None;
    }
    let mut restorers = states
        .into_iter()
        .map(CaseState::restorer)
        .collect::<Vec<_>>();
    Some(Box::new(move || {
        restorers.iter_mut().try_for_each(|restore| restore())
    }))
}

/// Entry-specific tuning knowledge. One implementation exists per native
/// entry that declares tuning parameters.
pub(crate) trait EntryTuning {
    type Entry: seismic::Entry;
    /// One argument set: every tensor its arguments borrow, owned, including
    /// the [`CaseState`]s its `&mut` parameters bind.
    type Case;

    /// The semantic bindings, for reports.
    fn bindings(&self) -> String;
    /// The value of every dimension the model fixes.
    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String>;
    fn points(&self, limits: TuningLimits) -> Vec<PointShape>;
    /// The argument sets of one point, cycling distinct layers.
    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String>;
    fn args<'a>(case: &'a mut Self::Case) -> <Self::Entry as seismic::Entry>::Args<'a>;
    /// The states `case` lends through `&mut` parameters; none for an entry
    /// without them. Seismic rejects a point that binds a `&mut` parameter
    /// whose state is not listed here.
    fn state(_case: &Self::Case) -> Vec<&CaseState> {
        Vec::new()
    }
    /// The entry's generated `native_tune[_with]`.
    fn tune(
        &self,
        device: &Device,
        statics: &NativeSpecialization,
        points: Vec<TuningPoint<'_, Self::Entry>>,
        validation: Validation,
        strategy: Strategy,
    ) -> Result<TuningResult, TuneError>;
    /// The entry's generated `native_digest[_with]`.
    fn digest(&self, device: &Device, statics: &NativeSpecialization) -> Result<String, TuneError>;
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
    /// Configurations excluded for authoring defects (formation failures,
    /// misclassified parameters, validation failures).
    pub defects: usize,
    /// The first defect's configuration and reason, for the kernel's author.
    pub first_defect: Option<String>,
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
pub(crate) struct TuningInputs<'w, 'a> {
    pub device: &'a Device,
    pub definition: &'a ModelDefinition,
    pub limits: TuningLimits,
    pub weights: &'w mut TuningWeights<'a>,
    /// Tensors shared by every point of the entry being tuned.
    shared: &'w mut HashMap<String, Tensor>,
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
        self.weights.weight(scope, kind)
    }

    /// The planned shape of one weight role.
    pub fn weight_shape(&self, scope: WeightScope, kind: WeightKind) -> Result<Vec<u64>, String> {
        self.weights.shape(scope, kind)
    }

    /// A deterministic pseudo-random activation in [-1, 1).
    pub fn activation(&self, element: Element, extents: &[u64], seed: u64) -> Result<Tensor, String> {
        let count = element_count(extents)?;
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let values = (0..count).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        });
        let bytes = match element.dtype() {
            Some(DType::F32) => values.flat_map(f32::to_le_bytes).collect::<Vec<_>>(),
            Some(DType::BF16) => values
                .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                .collect(),
            Some(DType::F16) => values.flat_map(|value| f16_bits(value).to_le_bytes()).collect(),
            _ => {
                return Err(format!(
                    "tuning activations support f32, bf16 and f16, not {}",
                    element.name()
                ))
            }
        };
        Tensor::from_host(self.device, element, extents, &bytes).map_err(|error| error.to_string())
    }

    /// An `i32` tensor of `values`.
    pub fn i32s(&self, extents: &[u64], values: &[i32]) -> Result<Tensor, String> {
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        Tensor::from_host(self.device, Element::i32(), extents, &bytes)
            .map_err(|error| error.to_string())
    }

    /// A `u32` tensor of `values`.
    pub fn u32s(&self, extents: &[u64], values: &[u32]) -> Result<Tensor, String> {
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        Tensor::from_host(self.device, Element::u32(), extents, &bytes)
            .map_err(|error| error.to_string())
    }

    /// An `f32` tensor of `values`.
    pub fn f32s(&self, extents: &[u64], values: &[f32]) -> Result<Tensor, String> {
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
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
    /// `written`; their current contents are what every configuration's
    /// validation run starts from.
    pub fn state(&self, tensor: Tensor, written: Range<u64>) -> Result<CaseState, String> {
        let region = tensor
            .slice_leading(written.start, written.end)
            .map_err(|error| error.to_string())?;
        let initial = region.read_to_host().map_err(|error| error.to_string())?;
        Ok(CaseState {
            tensor,
            written: region,
            initial: initial.into(),
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
                                visible: if context == 0 {
                                    Vec::new()
                                } else {
                                    vec![[base, base + context]]
                                },
                                destination,
                                demand: Demand::NONE,
                                select: None,
                            }
                        })
                        .collect(),
                    bank: slot as i32 + 1,
                    following_bank: (slots + slot) as i32 + 1,
                }
            })
            .collect::<Vec<_>>();
        let vocabulary = usize::try_from(self.definition.geometry.vocabulary)
            .map_err(|_| "vocabulary exceeds usize")?;
        PackedRowTables::pack(&packed, vocabulary, rows).map_err(|error| error.to_string())
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

/// Split `total` configurations equally among tuning units of `sizes`
/// admissible configurations each (§D3): a unit smaller than its share takes
/// only its size, and what it leaves is shared again among the others. Every
/// unit gets at least one configuration (its defaults). Deterministic given
/// the sizes in order; the remainder of an uneven split goes to the first
/// units.
pub fn allocate(total: usize, sizes: &[usize]) -> Vec<usize> {
    let mut budgets = vec![0; sizes.len()];
    let mut open = (0..sizes.len()).collect::<Vec<_>>();
    let mut remaining = total;
    loop {
        if open.is_empty() {
            return budgets;
        }
        let share = remaining / open.len();
        let small = open
            .iter()
            .copied()
            .filter(|&unit| sizes[unit] <= share)
            .collect::<Vec<_>>();
        if small.is_empty() {
            let extra = remaining % open.len();
            for (position, &unit) in open.iter().enumerate() {
                budgets[unit] = (share + usize::from(position < extra)).max(1);
            }
            return budgets;
        }
        for unit in small {
            budgets[unit] = sizes[unit].max(1);
            remaining -= sizes[unit];
            open.retain(|&candidate| candidate != unit);
        }
    }
}

/// How the tuner treats each unit it is asked for.
enum Allocation {
    /// Counting the model's tuning units and their admissible sizes: nothing
    /// is formed or measured.
    Census(Vec<(TuningKey, usize)>),
    /// Tuning, each unit within its share of [`MODEL_BUDGET`].
    Budgets(HashMap<TuningKey, usize>),
}

/// The configurations each tuning unit of a model may evaluate.
pub(crate) struct TuningBudgets(HashMap<TuningKey, usize>);

/// Resolves every native entry's specialization for the opened device:
/// static values, tuned parameters, and missing implementations.
pub(crate) struct Tuner<'a> {
    device: &'a Device,
    context: TuningContext<'a>,
    limits: TuningLimits,
    weights: TuningWeights<'a>,
    allocation: Allocation,
    /// The safety stop of this preparation's tuning.
    deadline: Instant,
    tuned: Vec<TunedEntry>,
    chosen: HashMap<TuningKey, NativeSpecialization>,
    /// The latest choice for each parameter declaration of an entry, a start
    /// for the next unit with the same declaration.
    winners: HashMap<String, ParameterValues>,
}

impl<'a> Tuner<'a> {
    /// A tuner that only counts tuning units, for [`Tuner::budgets`].
    pub fn census(
        device: &'a Device,
        context: TuningContext<'a>,
        limits: TuningLimits,
        weights: TuningWeights<'a>,
    ) -> Self {
        Self::with(device, context, limits, weights, Allocation::Census(Vec::new()))
    }

    pub fn new(
        device: &'a Device,
        context: TuningContext<'a>,
        limits: TuningLimits,
        weights: TuningWeights<'a>,
        budgets: TuningBudgets,
    ) -> Self {
        Self::with(device, context, limits, weights, Allocation::Budgets(budgets.0))
    }

    fn with(
        device: &'a Device,
        context: TuningContext<'a>,
        limits: TuningLimits,
        weights: TuningWeights<'a>,
        allocation: Allocation,
    ) -> Self {
        Self {
            device,
            context,
            limits,
            weights,
            allocation,
            deadline: Instant::now() + SAFETY_STOP,
            tuned: Vec::new(),
            chosen: HashMap::new(),
            winners: HashMap::new(),
        }
    }

    /// [`MODEL_BUDGET`] shared among the units a census counted.
    pub fn budgets(self) -> TuningBudgets {
        let Allocation::Census(units) = self.allocation else {
            unreachable!("budgets come from a census");
        };
        let sizes = units.iter().map(|(_, size)| *size).collect::<Vec<_>>();
        TuningBudgets(
            units
                .into_iter()
                .map(|(key, _)| key)
                .zip(allocate(MODEL_BUDGET, &sizes))
                .collect(),
        )
    }

    pub fn tuned(self) -> Vec<TunedEntry> {
        self.tuned
    }

    /// Tune `case` over its implementation's declared domain at `statics`
    /// and return the chosen configuration. An entry already tuned with the
    /// same bindings and static values reuses that result; a stored result
    /// of the same key is used without tuning. A census only counts the unit
    /// and returns the defaults.
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
        let configurations = implementation
            .admissible(statics)
            .map_err(|error| failure(error.to_string()))?
            .len();
        let budget = match &mut self.allocation {
            Allocation::Census(units) => {
                if !units.iter().any(|(counted, _)| *counted == key) {
                    units.push((key, configurations));
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
        let declaration = format!("{entry}:{:?}", implementation.params);
        let began = Instant::now();
        let survey = survey_plan(entry);
        let stored = match self.context.cache.filter(|_| survey.is_none()) {
            Some(cache) => {
                let digest = case
                    .digest(self.device, statics)
                    .map_err(|error| failure(error.to_string()))?;
                let key = TuningCacheKey::of(&tuning_key_material(
                    self.device,
                    entry,
                    &bindings,
                    statics,
                    &digest,
                    budget,
                    &shapes,
                ));
                let hit = cache
                    .tuning(&key)
                    .filter(|result| implementation.validate(&result.overall.specialization()).is_ok());
                Some((cache, key, hit))
            }
            None => None,
        };
        if let Some((_, _, Some(result))) = &stored {
            let tuned = tuned_entry(entry, bindings, result, TuningOrigin::Stored, began);
            return Ok(self.finish(key, declaration, tuned));
        }
        self.context.observer.event(&TuningEvent::Started {
            entry,
            bindings: bindings.clone(),
            configurations,
            points: shapes.len(),
        });
        let mut shared = HashMap::new();
        let mut inputs = TuningInputs {
            device: self.device,
            definition: self.context.definition,
            limits: self.limits,
            weights: &mut self.weights,
            shared: &mut shared,
        };
        let mut rotations = shapes
            .iter()
            .map(|point| case.rotation(&mut inputs, point))
            .collect::<Result<Vec<_>, _>>()
            .map_err(failure)?;
        let initializers = rotations
            .iter()
            .map(|cases| cases.first().and_then(|case| initializer(T::state(case))))
            .collect::<Vec<_>>();
        let points = shapes
            .iter()
            .zip(rotations.iter_mut())
            .zip(initializers)
            .map(|((shape, cases), initialize)| TuningPoint {
                label: shape.label.clone(),
                weight: shape.weight,
                rotation: cases.iter_mut().map(T::args).collect(),
                initialize,
            })
            .collect();
        let surveyed = survey.is_some();
        let strategy = match survey {
            Some(plan) => Strategy::Survey(plan),
            None => Strategy::Search(SearchPlan {
                budget,
                settings: SEARCH_SETTINGS,
                min_sample_seconds: MIN_SAMPLE_SECONDS,
                start: self.winners.get(&declaration).cloned().into_iter().collect(),
                deadline: Some(self.deadline),
            }),
        };
        #[cfg(feature = "tuning-survey")]
        let held = survey::hold(entry).map_err(failure)?;
        let result = case
            .tune(self.device, statics, points, ARITHMETIC_TOLERANCE, strategy)
            .map_err(|error| failure(error.to_string()))?;
        #[cfg(feature = "tuning-survey")]
        {
            drop(held);
            if surveyed {
                survey::record(&key, &result).map_err(failure)?;
            }
        }
        drop(rotations);
        drop(shared);
        self.weights.release();
        let stopped = matches!(
            result.method,
            TuningMethod::Search {
                stop: SearchStop::Expired,
                ..
            }
        );
        // A search the safety stop ended is not the search its key names.
        if let Some((cache, key, None)) = &stored {
            if !stopped && !surveyed {
                cache.store_tuning(key, &result);
            }
        }
        let tuned = tuned_entry(entry, bindings, &result, TuningOrigin::Searched, began);
        Ok(self.finish(key, declaration, tuned))
    }

    /// Record a unit's outcome and report it.
    fn finish(&mut self, key: TuningKey, declaration: String, tuned: TunedEntry) -> NativeSpecialization {
        self.context
            .observer
            .event(&TuningEvent::Finished(tuned.clone()));
        let chosen = tuned.overall.specialization();
        #[cfg(feature = "pinned-tuning")]
        pinned::record(&key, &chosen);
        self.winners.insert(declaration, tuned.overall.params.clone());
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
            shared: &mut shared,
        };
        case.statics(&inputs).map_err(|outcome| CatalogFailure::Preparation {
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
/// implementation's digest, and the search's definition.
fn tuning_key_material(
    device: &Device,
    entry: &str,
    bindings: &str,
    statics: &NativeSpecialization,
    digest: &str,
    budget: usize,
    shapes: &[PointShape],
) -> String {
    let points = shapes
        .iter()
        .map(|shape| format!("{}={:?}", shape.label, shape.weight))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "device {}\nentry {entry}\nbindings {bindings}\nstatics {:?}\nimplementation {digest}\n\
         search {SEARCH_VERSION}\nbudget {budget}\nsettings {SEARCH_SETTINGS:?}\n\
         points {points}\nvalidation {ARITHMETIC_TOLERANCE:?}\nmin sample {MIN_SAMPLE_SECONDS:?}",
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
                TuningMethod::Survey { .. } => None,
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
        defects: result.defects().count(),
        first_defect: result.defects().find_map(|record| match &record.outcome {
            seismic::Outcome::Excluded(exclusion) => {
                Some(format!("{:?}: {exclusion:?}", record.configuration.params))
            }
            seismic::Outcome::Measured { .. } => None,
        }),
        seconds: began.elapsed().as_secs_f64(),
        time,
    }
}

/// Collects every required entry that lacks an implementation for the
/// backend, so preparation reports all of them at once.
#[derive(Default)]
pub(crate) struct MissingImplementations(Vec<MissingImplementation>);

impl MissingImplementations {
    pub fn record(&mut self, entry: &'static str, bindings: &str) {
        if !self
            .0
            .iter()
            .any(|missing| missing.entry == entry && missing.bindings == bindings)
        {
            self.0.push(MissingImplementation {
                entry,
                bindings: bindings.to_owned(),
            });
        }
    }

    pub fn finish(self) -> Result<(), CatalogFailure> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(CatalogFailure::MissingImplementations(self.0))
        }
    }
}

#[cfg(test)]
mod tests;
