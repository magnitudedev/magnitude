//! Tuning at program preparation (route spec NR12, program spec E6, tuning
//! spec `specs/26-09-24/native-tuning-search-and-caching.md`).
//!
//! Every native entry whose implementation declares tuning parameters is
//! tuned on the opened device once for each tuning key, by a search
//! (`AttestedPrograms::search_draft`) that runs in a process of its own and
//! ends before the model is loaded. A load never searches: its one walk
//! ([`Tuner::serve`]) reads each unit's stored record and, at a unit without
//! one, stops and reports it or prepares the unit at its canonical default
//! ([`UnpreparedUnits`]). "First load" below names the search that precedes
//! it. A count walk finds the units without a stored choice. Their mandatory phase then
//! forms each one's canonical artifact, hands it to serving and publishes
//! the unit's record: from there every unit serves a legal published choice.
//! What is left of the first-load tuning budget ([`TuningSettings`]) is
//! spent on optional work by value ([`allocation`]): only work whose whole
//! price fits what remains is started, work begun runs to its end, and a
//! challenger that measured faster replaces a unit's choice only once it is
//! confirmed and covered. Time past the budget is reported, never a failure.
//! A unit builds each point's inputs only when its work reaches the point,
//! and forms only what it is about to measure. With a [`KernelCache`], every
//! unit's record is stored from its mandatory phase on, replaced when its
//! served choice changes and once more, with everything it measured, when
//! the optional work stops, including explicit unmeasured scope. A load
//! abandoned between two of those keeps the choice last stored.
//! Later loads only reuse that record, under a key over what its
//! measurements are valid for (tuning version, tuning budget,
//! device and toolchain identity, unit, implementation digest, the
//! comparison its measurements come from and served shapes), choosing from
//! it again when their α differs. A fresh preparation ([`TuningSettings::fresh`])
//! reads no stored result and replaces them. Nothing is shipped.
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
//! Every candidate is held to the caller's numerical profile against its
//! unit's default: each floating result and written state within α rounding
//! steps ([`NumericalProfile`]), everything else bit for bit. A unit's
//! measurements do not depend on α: a stored record is served under any
//! profile by choosing again among the choices its search served.
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
        #[cfg(feature = "tuning-survey")]
        fn tune(
            &self,
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
            points: &mut dyn seismic::PointSource<'_, Self::Entry>,
            validation: seismic::NumericalRequirement,
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
            )
        }

        fn open(
            &self,
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
            validation: seismic::NumericalRequirement,
        ) -> Result<Box<dyn super::EntrySearch<Self::Entry>>, seismic::TuneError> {
            let $this = self;
            Ok(Box::new($module::native_search_with(
                device,
                $elements,
                statics,
                validation,
            )?))
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
        #[cfg(feature = "tuning-survey")]
        fn tune(
            &self,
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
            points: &mut dyn seismic::PointSource<'_, Self::Entry>,
            validation: seismic::NumericalRequirement,
            strategy: seismic::Strategy,
        ) -> Result<seismic::TuningResult, seismic::TuneError> {
            $module::native_tune(
                device,
                statics,
                points,
                validation,
                strategy,
            )
        }

        fn open(
            &self,
            device: &seismic::Device,
            statics: &seismic::NativeSpecialization,
            validation: seismic::NumericalRequirement,
        ) -> Result<Box<dyn super::EntrySearch<Self::Entry>>, seismic::TuneError> {
            Ok(Box::new($module::native_search(
                device,
                statics,
                validation,
            )?))
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

mod allocation;
pub(crate) mod attention;
pub(crate) mod cases;
pub(crate) mod general_routed;
pub(crate) mod per_layer;
#[cfg(feature = "pinned-tuning")]
pub mod pinned;
pub(crate) mod post_norm;
mod precision;
pub(crate) mod readout;
pub(crate) mod readout_choice;
mod record;
pub(crate) mod recurrent;
pub(crate) mod routed;
pub(crate) mod short_conv;
pub(crate) mod state_space;
#[cfg(feature = "tuning-survey")]
pub mod survey;
mod trace;
mod weights;
mod workload;

pub use precision::{NumericalProfile, ResolvedNumericalProfile};
pub use record::Receipt;
use record::StoredChoice;
#[cfg(test)]
use record::ChosenEvidence;
pub use trace::{TRACE_SCHEMA, TracedWork, TuningTrace};
pub use weights::{TuningWeightSource, ZeroTuningWeights};
pub(crate) use weights::{TuningWeights, weight_plan};

use super::CatalogFailure;
use crate::ModelLoadPlan;
use crate::kernel_cache::{KernelCache, TuningCacheKey};
use magnitude_batching::{ClassLimits, Demand, PackedRowTables, Row, RowHistory, Slot};
use magnitude_family_contracts::{ModelDefinition, Operator, WeightKind, WeightScope};
#[cfg(feature = "tuning-survey")]
use seismic::Strategy;
use seismic::{
    Configuration, DType, Device, Element, NativeImplementation, NativeSpecialization,
    ParameterValues, SearchSettings, SearchStop, Standing, Tensor, TensorError,
    TuneError, TuningInitializer, TuningResult, TuningTime,
};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The row classes units are tuned at. The served workload's launches per
/// turn at each weigh the objective ([`invocation_base`]).
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
/// across changes to the search, its settings or the workload weights: it
/// was measured on this device for this code and comparison at these
/// shapes under this tuning budget, which the rest of the key covers. Bump
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
/// 28: production fidelity uses the exact associated portable computation;
/// canonical formation is pre-clock, observations share the original minute.
/// Earlier native self-comparison results are not valid under this identity.
/// 27: complete checked numerical computations and caller-owned exact-recipe approvals
/// replace error-class widening; reference identity binds the computation and actual elements.
/// 26: numerical witnesses and performance rotations are distinct; zero-demand histories
/// remain untimed witnesses, and expired validation is not a cached rejection.
/// 25: admitted optional packets survive forecast-slice handoff under hard deadlines;
/// affine served geometry includes codec planes and invalid terminal choices cannot reopen search.
/// 23: runtime-owned admissible start portfolios are priced atomic packets;
/// service-weighted coverage continues within the same first-load minute before descent.
/// 22: incomplete required admission uses the allocated first-load slot;
/// receipt-volume assembly/publication reserves bootstrap from actual census
/// work, with every attempted execution phase charged on all outcomes.
/// 21: launch boundaries use admissible local witnesses; partial census
/// measurements retain their cost and evidence within the first-load minute.
/// 31: candidates qualify on the device against the declaration's native
/// default under the caller's comparison; no portable or host reference.
/// 20: required canonical qualification precedes optimization;
/// deadline-only optional absence retains the safe declared default.
/// 19: one first-load deadline across admission and confirmation; corrected
/// confirmation reserve, retained evidenced incumbents and dependent choices.
/// 29: selected-artifact service receipts retain actual checked invocation
/// geometry for load forecasting. Pre-schema records have a different terminal
/// identity; they are neither reinterpreted nor reopened by this load.
/// 32: a stored result of 31 can no longer be read: a measured point states
/// its invocation geometry without repeating its entry and artifact, a
/// method records no per-unit allowance, and coverage outcomes are passed
/// or failed. A unit's record is published from its mandatory phase on and
/// replaced as its first load measures and promotes.
/// 33: a stored result of 32 can no longer be read: a searched unit's
/// method is one shape, `Search { groups, candidates, complete }`, whether
/// its launches are formed separately or its program whole (it was
/// `Factored` with those fields for the one and `Search { settings, stop }`
/// for the other), and recorded settings no longer carry `improvement`,
/// `restarts` or `confirmed`. A program formed whole is searched in the
/// order of the others: other forms first, a form's leader's neighbours
/// after.
/// 35: the objective's demand is the workload's launches at every row count:
/// a third of the decode steps run at 2, 4 and 8 rows, which a result of 34
/// was searched without (it weighed every decoded token at one row).
/// 36: a searched configuration's record states how it was formed (the
/// formation calls' seconds, the programs they formed, whether any compiler
/// ran), and a piece's price includes the planning that follows it; a
/// result of 35 records neither.
/// 37: every recorded time is read at the device's sustained clock, both
/// sides of a round with the same repetitions and each sample behind a lead
/// of its own work (a result of 36 mixed readings taken at different device
/// states, whose ratios do not compare), and a challenger that is timed
/// before it is witnessed and does not stay ahead records no numerical
/// observations.
/// 38: what a search admits changed with what its pieces are quoted: a
/// promoted choice owes its serving artifact only where the store lacks
/// it, a witness after rounds not yet observed is priced by the unit's
/// admissions witnessed first, and a unit's coverage checks not yet
/// observed are one unknown. A result of 37 stopped with seconds of its
/// budget unused and left measured leads unconverted.
/// 39: a record is stored as its choice and, apart from it, its evidence
/// ([`record`]); a result of 38 is one entry holding both, which a load
/// decoded whole. And a challenger leads the served choice when the sign
/// test of its matched rounds resolves that it is faster, by however
/// little: no margin is asked of a round, of a nominal lead or of a
/// promotion, and the search's settings hold the error rate alone (a
/// result of 38 served only a challenger faster by more than 2%, and
/// recorded that margin in its settings). And a search whose leading row
/// band boundary moves off the default goes on to its groups' sweeps, and
/// is recorded complete only when its domain is exhausted (in a result of
/// 38 such a unit stopped after the new leader's coverage, its launches'
/// mappings unrefined, and was recorded as exhausted). And a unit whose
/// remaining pieces did not fit the memory room when its load's optional
/// work stopped is stored with that stop ([`SearchStop::NoRoom`]; a
/// result of 38 recorded it as expired, as if the budget had ended it).
pub const SEARCH_VERSION: u32 = 39;
/// The tuning budget of a served load: the time one first-load preparation
/// spends tuning, on any device.
pub const STANDARD_TUNING_BUDGET: Duration = Duration::from_secs(60);

/// What a load does when it meets a tuning unit without a stored record
/// (and without a development pin deciding it). The two are one walk that
/// differs only at that moment; a load searches under neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UnpreparedUnits {
    /// Stop there, before anything of the model is placed, and report it
    /// ([`CatalogFailure::Unprepared`]): the host has the search run in a
    /// process of its own and loads again.
    Report,
    /// Prepare the unit at its canonical default and go on: the load is
    /// total.
    ServeDefaults,
}

/// How one preparation tunes, chosen by its host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TuningSettings {
    /// The time the first-load preparation spends tuning. It is part of every
    /// stored result's key: a result searched within one budget is never
    /// read under another.
    pub budget: Duration,
    /// Search every unit again: stored tuning results are not read, and the
    /// new results replace them. Formed programs are still reused. A load
    /// never searches: one that reports units without a record reports
    /// every unit under it, and one that serves defaults ignores it.
    pub fresh: bool,
}

impl TuningSettings {
    /// What a served load tunes with.
    pub const STANDARD: Self = Self {
        budget: STANDARD_TUNING_BUDGET,
        fresh: false,
    };
}
/// The search's constants (§D2), the same for every unit on every device.
///
/// `error_rate` is how often a resolved comparison of a challenger with the
/// served choice is wrong ([`SearchSettings::error_rate`]): one in twenty of
/// the comparisons of a configuration that is not faster than the served
/// choice promote it, and one in twenty of those of a configuration that
/// is not slower drop it. No margin is asked of a challenger beyond that:
/// one that is faster by however little leads once its rounds resolve it.
/// The rate is per comparison, and only a comparison given the rounds to
/// resolve can go wrong the first way: a challenger whose first round is
/// not ahead leaves after it, so those comparisons are a load's
/// conversions (48 promotions on Metal for Qwen3.5-4B Q4 in the load
/// recorded 2026-10-10, 42 with the 2% margin that was asked before). How many
/// samples a decision takes follows from this rate and from how the rounds
/// come out, nowhere from a count: a clear difference is settled in the
/// fewest rounds there are (seven for a lead at 0.05), a slower challenger
/// leaves after one, and neither depends on the device or on the unit's
/// share (what a decision is worth is weighed where its piece is admitted,
/// by the allocator).
pub const SEARCH_SETTINGS: SearchSettings = SearchSettings { error_rate: 0.05 };

/// Minimum device time of one sample: what makes the fixed device cost of
/// a submission, which both sides of a comparison carry, a small part of
/// what is timed (device timestamps resolve microseconds). No number of
/// rounds stands in for it: rounds average a sample's noise, and this is a
/// bias every round repeats.
pub const MIN_SAMPLE_SECONDS: f64 = 0.0002;

/// One workload an entry serves: its shape and its invocation demand.
#[derive(Clone, Debug, PartialEq)]
pub struct PointShape {
    pub label: String,
    /// Launches of this shape per turn of the served workload, once the
    /// load's graph has supplied the unit's calls
    /// ([`apply_invocation_counts`]); as a case declares its points, the
    /// workload's launches relative to the unit's other points. The
    /// objective is the sum over points of this times the time measured
    /// there, so a point of no demand votes for nothing.
    pub weight: f64,
    pub rows: u64,
    /// The sequences the point's rows belong to, each with a history of its
    /// own: 1 for rows of one sequence (a prefill chunk, a drafter's
    /// verification rows), the rows themselves for a decode step of that
    /// many requests. Only a unit whose time depends on it declares points
    /// of more than one ([`apply_invocation_counts`]).
    pub sequences: u64,
    /// Visible history rows for attention points.
    pub context: Option<u64>,
    /// The row point whose history lengths this point varies: the physical
    /// region the point belongs to, which normalizes nothing of its own
    /// (`seismic::TuningPoint::class`).
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

/// The served workload's launches per turn at every row class the engine
/// has ([`workload::class_load`]): prefill, single-row decode and the decode
/// steps of 2, 4 and 8 rows. Supplied to the load's graph invocation model,
/// which adds action multiplicities; the tuner does not infer them from
/// weights.
pub(crate) fn invocation_base(
    limits: TuningLimits,
) -> crate::programs::invocation_demand::InvocationCounts {
    // A prefill launch's rows are one sequence's. A decode step of several
    // rows is that many requests decoding together, one row each.
    crate::programs::invocation_demand::InvocationCounts::across(
        TUNING_ROWS
            .into_iter()
            .filter(|rows| *rows <= limits.max_rows)
            .flat_map(|rows| {
                let prefill = workload::prefill_load(rows, limits).launches;
                let decode = (workload::class_load(rows, limits).launches - prefill).max(0.);
                [(rows, 1, prefill), (rows, rows, decode)]
            }),
    )
}

/// The decode rounds of a turn that bound a speculative method's actions:
/// a round emits at least one of the turn's decoded tokens.
pub(crate) fn invocation_rounds() -> f64 {
    workload::decoded_tokens()
}

pub(crate) fn invocation_prefill_base(
    limits: TuningLimits,
) -> crate::programs::invocation_demand::InvocationCounts {
    crate::programs::invocation_demand::InvocationCounts::new(
        TUNING_ROWS
            .into_iter()
            .filter(|rows| *rows <= limits.max_rows)
            .map(|rows| (rows, workload::prefill_load(rows, limits).launches)),
    )
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

/// The row classes one census point stands for. Each streaming row count is
/// a class of its own. Numerical witnesses cover all histories; the census
/// times bootstrap and workload representatives. Beyond them the short chunks and the prefill chunks are one group
/// each: the census times a unit at the costliest point it serves in each
/// group, and that point's time, scaled by cost, stands for the unit at
/// every row count of the group it serves.
fn census_group(rows: u64) -> u64 {
    match rows {
        0..=STREAMING_ROWS => rows,
        16 | 32 => 32,
        _ => u64::MAX,
    }
}

/// Discover each served execution region at its widest rows, first without
/// history, then at the points representing its workload demand. Complete
/// numerical scope remains independent of these performance measurements.
fn census_points(shapes: &[PointShape]) -> Vec<bool> {
    shapes
        .iter()
        .map(|shape| {
            (shape.rows <= STREAMING_ROWS
                || shapes.iter().all(|other| {
                    census_group(other.rows) != census_group(shape.rows) || other.rows <= shape.rows
                }))
                && (shape.context.is_none_or(|history| history == 0) || shape.weight > 0.)
        })
        .collect()
}

fn normalized(mut points: Vec<PointShape>) -> Vec<PointShape> {
    let total = points.iter().map(|point| point.weight).sum::<f64>();
    for point in &mut points {
        point.weight /= total;
    }
    points
}

/// Retain every numerical point while replacing only performance demand.
/// History points preserve their conditional distribution within a row class.
fn apply_invocation_counts(
    points: &mut [PointShape],
    counts: &crate::programs::invocation_demand::InvocationCounts,
) {
    // Launches of a row count go to the points of that row count whose
    // sequences are the most declared that the launches have: a unit that
    // declares only points of one sequence takes them all there.
    let declared = points
        .iter()
        .fold(BTreeMap::<u64, Vec<u64>>::new(), |mut declared, point| {
            declared.entry(point.rows).or_default().push(point.sequences);
            declared
        });
    let mut supplied = BTreeMap::<(u64, u64), f64>::new();
    for (rows, declared) in &declared {
        for (sequences, count) in counts.sequences(*rows) {
            let at = declared
                .iter()
                .copied()
                .filter(|declared| *declared <= sequences)
                .max()
                .unwrap_or_else(|| *declared.iter().min().expect("a row count has a point"));
            *supplied.entry((*rows, at)).or_default() += count;
        }
    }
    let totals = points
        .iter()
        .fold(BTreeMap::<(u64, u64), f64>::new(), |mut sum, point| {
            *sum.entry((point.rows, point.sequences)).or_default() += point.weight;
            sum
        });
    for point in points {
        let shape = (point.rows, point.sequences);
        let total = totals[&shape];
        point.weight = if total > 0. {
            point.weight / total * supplied.get(&shape).copied().unwrap_or(0.)
        } else {
            0.
        };
    }
}

fn row_point(rows: u64, weight: f64) -> PointShape {
    PointShape {
        label: format!("m{rows}"),
        weight,
        rows,
        sequences: 1,
        context: None,
        class: None,
    }
}

/// Where a point's histories lie ([`TuningInputs::batch`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HistoryPlacement {
    /// Per sequence, its history's `[start, end)` row ranges in logical
    /// order.
    pub sequences: Vec<Vec<[i32; 2]>>,
    /// One past the last row any history holds.
    pub extent: u64,
}

/// Where `point`'s histories lie. One sequence's history is one span, as a
/// store keeps a request that grows alone. Several sequences' lie where the
/// target's Token history store, of `codec`, places requests that arrive
/// together ([`magnitude_state::fresh_history_ranges`]): apart from one
/// another, so a kernel form that reads its rows' histories as one range
/// is timed over what lies between them, as it is in service.
pub(crate) fn history_placement(
    model: &ModelInputs<'_>,
    codec: magnitude_state::KvCodec,
    point: &PointShape,
) -> Result<HistoryPlacement, String> {
    let context = point.context.unwrap_or(0);
    let bound = |row: u64| i32::try_from(row).map_err(|_| "tuning history exceeds i32".to_owned());
    if point.sequences == 1 {
        // A history of no rows has no span.
        let history = match context {
            0 => Vec::new(),
            context => vec![[0, bound(context)?]],
        };
        return Ok(HistoryPlacement {
            sequences: vec![history],
            extent: context,
        });
    }
    let layout = magnitude_state::ModelStateLayout::derive(&model.definition.decoder, &[], codec, 0)?;
    let geometry = layout
        .target_history
        .iter()
        .find(|domain| matches!(domain, magnitude_state::HistoryDomainLayout::Token { .. }))
        .ok_or("a point of several sequences needs a Token history domain")?
        .geometry(
            &layout.target_history,
            usize::try_from(model.limits.context_tokens).map_err(|_| "context exceeds usize")?,
            usize::try_from(model.limits.max_rows).map_err(|_| "rows exceed usize")?,
        )
        .map_err(|error| error.to_string())?;
    let sequences = magnitude_state::fresh_history_ranges(
        geometry,
        usize::try_from(point.sequences).map_err(|_| "sequences exceed usize")?,
        usize::try_from(context).map_err(|_| "context exceeds usize")?,
    )
    .into_iter()
    .map(|history| {
        history
            .into_iter()
            .map(|(start, count)| Ok([bound(start as u64)?, bound((start + count) as u64)?]))
            .collect::<Result<Vec<_>, String>>()
    })
    .collect::<Result<Vec<_>, String>>()?;
    let extent = sequences
        .iter()
        .flatten()
        .map(|range| range[1] as u64)
        .max()
        .unwrap_or(0);
    Ok(HistoryPlacement { sequences, extent })
}

/// `point`, a row point of one sequence, as `sequences` sequences' rows.
pub(crate) fn across_sequences(point: &PointShape, sequences: u64) -> PointShape {
    PointShape {
        label: format!("{}s{sequences}", point.label),
        sequences,
        ..point.clone()
    }
}

/// Row points up to the engine's row bound.
pub fn row_points(limits: TuningLimits) -> Vec<PointShape> {
    served_row_points(limits, |_| true)
}

/// Every row point an entry serves up to the row bound of `limits`. An entry
/// whose served rows all lie beyond the bound is still prepared (its graph
/// classes do not exist, but the kernel set is complete); its smallest served
/// row count stands in as the single point.
pub fn served_row_points(limits: TuningLimits, serves: impl Fn(u64) -> bool) -> Vec<PointShape> {
    let served = TUNING_ROWS
        .into_iter()
        .filter(|rows| serves(*rows))
        .collect::<Vec<_>>();
    let within = served
        .iter()
        .copied()
        .filter(|rows| *rows <= limits.max_rows)
        .collect::<Vec<_>>();
    if within.is_empty() {
        return served
            .into_iter()
            .take(1)
            .map(|rows| row_point(rows, 1.0))
            .collect();
    }
    normalized(
        within
            .into_iter()
            .map(|rows| row_point(rows, workload::class_weight(rows, limits)))
            .collect(),
    )
}

/// The longest history the engine serves to layers keeping at most `window`
/// history rows: the window when it is shorter than the context, else the
/// context.
pub fn longest_history(limits: TuningLimits, window: Option<u64>) -> u64 {
    window.map_or(limits.context_tokens, |window| {
        window.min(limits.context_tokens)
    })
}

/// `rows` crossed with the history lengths the engine serves to layers
/// keeping at most `window` history rows (the whole context without one).
///
/// A layer whose window is shorter than the context never sees a longer
/// history and sees exactly its window in every prompt longer than it, so
/// its lengths are the ladder's below the window and the window itself. A
/// window that holds the whole context bounds nothing.
pub fn with_contexts(
    limits: TuningLimits,
    window: Option<u64>,
    rows: Vec<PointShape>,
) -> Vec<PointShape> {
    let longest = longest_history(limits, window);
    let mut contexts = TUNING_CONTEXTS
        .into_iter()
        .filter(|context| *context <= longest)
        .collect::<Vec<_>>();
    contexts.push(longest);
    contexts.sort_unstable();
    contexts.dedup();
    normalized(
        rows.into_iter()
            .flat_map(|point| {
                // Linear history interpolation represents the declared mean
                // service history. Endpoints still belong to qualification.
                let history = (workload::class_load(point.rows, limits).history
                    * limits.context_tokens as f64)
                    .min(longest as f64);
                let upper = contexts
                    .iter()
                    .position(|&context| context as f64 >= history)
                    .unwrap_or(contexts.len() - 1);
                let lower = upper.saturating_sub(1);
                let low = contexts[lower] as f64;
                let high = contexts[upper] as f64;
                let high_share = if high > low {
                    ((history - low) / (high - low)).clamp(0., 1.)
                } else {
                    1.
                };
                contexts
                    .iter()
                    .enumerate()
                    .map(move |(index, &context)| PointShape {
                        label: format!("{}-c{context}", point.label),
                        weight: point.weight
                            * if index == upper {
                                high_share
                            } else if index == lower {
                                1. - high_share
                            } else {
                                0.
                            },
                        rows: point.rows,
                        sequences: point.sequences,
                        context: Some(context),
                        class: Some(point.label.clone()),
                    })
            })
            .collect(),
    )
}

/// Row points crossed with the history lengths the engine serves to layers
/// keeping the whole context.
pub fn attention_points(limits: TuningLimits) -> Vec<PointShape> {
    with_contexts(limits, None, row_points(limits))
}

/// In-place state a case lends to an entry's `&mut` parameter: the tensor,
/// the leading-axis rows the entry writes, and their initial contents. At
/// most points only those rows are restored before every reference and
/// validated invocation and observed by validation; the rest is input the
/// entry only reads. One point per unit ([`guard_point`]) restores and
/// observes the complete storage: there the rows outside the declared ones
/// must stay the default's bit for bit, so a write outside them is rejected
/// without paying for whole long histories everywhere.
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

/// One unit's tuning on the device, advanced one piece of work at a time
/// ([`seismic::NativeSearch`]). From its mandatory work on, the unit holds
/// a legal served choice and a complete record of it; every later piece is
/// optional and can only replace that choice with a better one.
pub(crate) trait EntrySearch<E: seismic::Entry> {
    /// The compile of the mandatory work, the unit's canonical program,
    /// to run on any thread; none when the unit has none to run there.
    fn grounding(
        &mut self,
        points: &mut dyn seismic::PointSource<'static, E>,
    ) -> Result<Option<seismic::Grounding>, TuneError>;
    /// The end of the mandatory work: take what the compile formed and
    /// hand the canonical program to serving.
    fn grounded(&mut self, grounded: Option<seismic::Grounded>) -> Result<(), TuneError>;
    fn plan(&mut self, plan: seismic::UnitPlan);
    /// The unit's complete record as it stands.
    fn terminal(&self) -> Result<TuningResult, TuneError>;
    /// The next piece of optional work, priced in full; none when the unit
    /// has none left.
    fn next(&self) -> Option<seismic::Work>;
    /// The compile of the next piece, to run on any thread ahead of it;
    /// none when the piece compiles nothing ahead.
    fn forming(&self) -> Option<seismic::Forming>;
    /// Hold what the next piece's compile formed. Returns what the compile
    /// took.
    fn formed(&mut self, formed: seismic::Formed) -> Result<Duration, TuneError>;
    /// Run the next piece of work to its end.
    fn perform(
        &mut self,
        points: &mut dyn seismic::PointSource<'static, E>,
    ) -> Result<seismic::Performed, TuneError>;
    /// Give up what only running work needs: the points' built inputs,
    /// and with no work left the held programs.
    fn release(&mut self, points: &mut dyn seismic::PointSource<'static, E>);
    /// Offer the next piece with one challenger fewer; false when it has
    /// no fewer to offer.
    fn narrow(&mut self) -> bool;
    /// The bytes of the inputs the unit holds; zero when it holds none.
    fn holding(&self) -> u64;
    /// The bytes of the formed programs the unit's search holds.
    fn programs(&self) -> u64;
    /// Give up the formed programs the unit's search holds.
    fn give_up_programs(&mut self);
    /// The most a single piece of the unit's work has added to the bytes
    /// held.
    fn largest_footprint(&self) -> u64;
    /// The longest single operation the unit has completed.
    fn longest_operation(&self) -> Duration;
    /// The price of publishing the serving artifact of the unit's served
    /// choice as it stands, or, `worked`, as it stands once its next piece
    /// has run: nothing where the store keeps it as a load requests it.
    fn artifact_owed(&self, worked: bool) -> seismic::Quote;
    /// What that publication does, known before it runs.
    fn artifact_publication(&self, worked: bool) -> seismic::Publication;
    /// What building the inputs the unit holds took.
    fn inputs_built(&self) -> seismic::Quote;
    /// Publish the serving artifact of the unit's served choice: form its
    /// programs exactly as a load requests them and release them.
    fn publish_artifact(&mut self) -> Result<Duration, TuneError>;
    fn service_measurements(&self) -> Vec<seismic::PointMeasurement>;
    fn standing(&self) -> Standing;
}

impl<E: seismic::Entry> EntrySearch<E> for seismic::NativeSearch<'static, E> {
    fn grounding(
        &mut self,
        points: &mut dyn seismic::PointSource<'static, E>,
    ) -> Result<Option<seismic::Grounding>, TuneError> {
        seismic::NativeSearch::grounding(self, points)
    }
    fn grounded(&mut self, grounded: Option<seismic::Grounded>) -> Result<(), TuneError> {
        match grounded {
            Some(grounded) => seismic::NativeSearch::grounded(self, grounded),
            None => Ok(()),
        }
    }
    fn plan(&mut self, plan: seismic::UnitPlan) {
        seismic::NativeSearch::plan(self, plan)
    }
    fn terminal(&self) -> Result<TuningResult, TuneError> {
        seismic::NativeSearch::terminal(self)
    }
    fn next(&self) -> Option<seismic::Work> {
        seismic::NativeSearch::next(self)
    }
    fn forming(&self) -> Option<seismic::Forming> {
        seismic::NativeSearch::forming(self)
    }
    fn formed(&mut self, formed: seismic::Formed) -> Result<Duration, TuneError> {
        seismic::NativeSearch::formed(self, formed)
    }
    fn perform(
        &mut self,
        points: &mut dyn seismic::PointSource<'static, E>,
    ) -> Result<seismic::Performed, TuneError> {
        seismic::NativeSearch::perform(self, points)
    }
    fn release(&mut self, points: &mut dyn seismic::PointSource<'static, E>) {
        seismic::NativeSearch::release(self, points)
    }
    fn narrow(&mut self) -> bool {
        seismic::NativeSearch::narrow(self)
    }
    fn holding(&self) -> u64 {
        // Inputs whose building added nothing counted are still held.
        if seismic::NativeSearch::holds_inputs(self) {
            seismic::NativeSearch::held_inputs(self).max(1)
        } else {
            0
        }
    }
    fn programs(&self) -> u64 {
        seismic::NativeSearch::held_programs(self)
    }
    fn give_up_programs(&mut self) {
        seismic::NativeSearch::give_up_programs(self)
    }
    fn largest_footprint(&self) -> u64 {
        seismic::NativeSearch::largest_footprint(self)
    }
    fn longest_operation(&self) -> Duration {
        seismic::NativeSearch::longest_operation(self)
    }
    fn artifact_owed(&self, worked: bool) -> seismic::Quote {
        seismic::NativeSearch::artifact_owed(self, worked)
    }
    fn artifact_publication(&self, worked: bool) -> seismic::Publication {
        seismic::NativeSearch::artifact_publication(self, worked)
    }
    fn inputs_built(&self) -> seismic::Quote {
        seismic::NativeSearch::inputs_built(self)
    }
    fn publish_artifact(&mut self) -> Result<Duration, TuneError> {
        seismic::NativeSearch::publish_artifact(self)
    }
    fn service_measurements(&self) -> Vec<seismic::PointMeasurement> {
        seismic::NativeSearch::service_measurements(self)
    }
    fn standing(&self) -> Standing {
        seismic::NativeSearch::standing(self)
    }
}

/// Entry-specific tuning knowledge. One implementation exists per native
/// entry that declares tuning parameters. A unit's search keeps its case
/// from its census to its conclusion.
pub(crate) trait EntryTuning: Clone + 'static {
    type Entry: seismic::Entry;
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

    /// Project a logical graph invocation onto this entry's performance
    /// representative. Numerical support remains `points`/served dimensions.
    fn invocation_rows(&self, rows: u64, representatives: &[u64]) -> Option<u64> {
        (rows > 0)
            .then(|| {
                representatives
                    .iter()
                    .copied()
                    .filter(|at| *at >= rows)
                    .min()
            })
            .flatten()
    }
    /// The dimensions of the calls the unit serves at shapes the inputs of
    /// its points `shapes` do not take (`seismic::PointSource::served`):
    /// none, unless a launch's requirements grow with a dimension every
    /// point holds at one value. A candidate that cannot launch at one of
    /// them is excluded as it is formed, and a stored choice that cannot is
    /// tuned again.
    fn served_dimensions(&self, _shapes: &[PointShape]) -> Vec<BTreeMap<String, u64>> {
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
    /// Numerical witnesses may include adversarial cases; only genuine served
    /// invocations belong to the performance workload. Real-weight rotations
    /// retain every layer by default.
    fn workload(&self, _cases: usize) -> seismic::WorkloadRotation {
        seismic::WorkloadRotation::All
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
    /// The entry's generated `native_tune[_with]`: a development survey of
    /// the unit.
    #[cfg(feature = "tuning-survey")]
    fn tune(
        &self,
        device: &Device,
        statics: &NativeSpecialization,
        points: &mut dyn seismic::PointSource<'_, Self::Entry>,
        validation: seismic::NumericalRequirement,
        strategy: Strategy,
    ) -> Result<TuningResult, TuneError>;
    /// Open the search of the unit at `statics`: the entry's generated
    /// `native_search[_with]`.
    fn open(
        &self,
        device: &Device,
        statics: &NativeSpecialization,
        validation: seismic::NumericalRequirement,
    ) -> Result<Box<dyn EntrySearch<Self::Entry>>, TuneError>;
    /// The entry's generated `native_digest[_with]`.
    fn digest(&self, device: &Device, statics: &NativeSpecialization) -> Result<String, TuneError>;
    /// Check a stored choice against the same device-augmented declaration
    /// that native preparation uses.
    fn stored_valid(&self, device: &Device, specialization: &NativeSpecialization) -> bool {
        self.entry()
            .specialization_valid(device, specialization)
            .unwrap_or(false)
    }
    /// The entry bound to the case's elements (its generated
    /// `native_entry[_with]`).
    fn entry(&self) -> seismic::BoundEntry<Self::Entry>;
}

/// Progress of tuning at load, for readiness reporting.
#[derive(Clone, Debug, PartialEq)]
pub enum TuningEvent {
    /// Milliseconds of tuning time spent so far, of the tuning budget.
    /// Reported when the search walk begins and after each unit it
    /// searches; nothing is reported when every unit is stored.
    Progress {
        completed: usize,
        total: usize,
    },
    /// A unit's optional work starts. How many configurations it can reach
    /// is reported with its result ([`SearchProgress`]): counting the
    /// declared domain here would enumerate it.
    Started {
        entry: &'static str,
        bindings: String,
        points: usize,
    },
    /// A unit's mandatory work is done: when the units' canonical compiles
    /// began on the tuning clock (what the count walk took), what forming
    /// its canonical artifact took (its own compile, which ran beside the
    /// other units', and handing it to serving), and what publishing its
    /// record took.
    Mandatory {
        entry: &'static str,
        bindings: String,
        began_seconds: f64,
        forming_seconds: f64,
        publishing_seconds: f64,
    },
    /// A piece of a unit's optional work ran to its end: its forecast and
    /// what it took.
    Stage(StageRecord),
    /// The optional work stopped and a unit whose record had changed since
    /// it was published published it: what the publication was priced at
    /// (none while its class was unobserved), when it began on the tuning
    /// clock and what it took.
    Published {
        entry: &'static str,
        bindings: String,
        forecast_seconds: Option<f64>,
        began_seconds: f64,
        actual_seconds: f64,
    },
    /// A unit's complete record: once per unit, when its work at this load
    /// is over.
    Finished(TunedEntry),
    /// The load's optional work is over: what became of its budget.
    Budget(BudgetReport),
    /// A load found no stored record for a unit and prepared it at its
    /// canonical default. Nothing is published for it.
    Unprepared {
        entry: &'static str,
        bindings: String,
        statics: BTreeMap<String, u64>,
    },
    /// A development tuning pin (feature `pinned-tuning`) fixed a unit's
    /// configuration: the unit was not tuned. `configuration` is what the
    /// unit runs.
    Pinned {
        entry: &'static str,
        bindings: String,
        statics: BTreeMap<String, u64>,
        choice: PinnedChoice,
        configuration: Configuration,
    },
}

/// Where a pinned unit's configuration came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinnedChoice {
    /// The configuration recorded for the unit.
    Replayed,
    /// Every unit runs its declared default (`TuningPin::Defaults`).
    Defaults,
    /// The recording has no configuration for the unit (a shape first
    /// prepared in this run): its declared default runs, untuned.
    NotReplayed,
}

/// One piece of optional work the allocator started, and what it took.
/// Times are seconds; `began_seconds` since the tuning time began.
#[derive(Clone, Debug, PartialEq)]
pub struct StageRecord {
    pub entry: &'static str,
    pub bindings: String,
    /// The unit's statics: with its entry and bindings, which unit.
    pub statics: BTreeMap<String, u64>,
    /// The unit's share of step time when the piece was started.
    pub share: f64,
    pub stage: seismic::Stage,
    /// The piece's whole price when it was started: the sum of its
    /// operations' class prices, and how many of its operations belonged
    /// to a class this load had not observed. A piece with an unpriced
    /// operation has no forecast.
    pub priced_seconds: f64,
    pub unpriced: usize,
    pub forecast_seconds: Option<f64>,
    pub began_seconds: f64,
    pub actual_seconds: f64,
    /// The piece's compile, when it ran ahead on a worker while the device
    /// did other work: its price when the piece was started, and what it
    /// took. The piece's price and time above are then of the rest of it,
    /// on the device.
    pub formed: Option<FormedAhead>,
    /// Where the piece's time went by operation class, and what publishing
    /// the unit's record after it took.
    pub spent: seismic::Spent,
    /// What each raise of the device's clock did within the piece, and
    /// each establishment of its gauge, in order.
    pub raises: Vec<seismic::Raise>,
    pub publishing_seconds: f64,
    /// The unit's served choice changed.
    pub promoted: bool,
}

/// What the platform lets the engine claim in each memory domain when the
/// search's device is opened: the ceiling of the search
/// ([`crate::platform::OpenedPlatform::tuning_room`]). It is the platform's
/// own bound on any engine claim, the availability it observed above its
/// planning reserve, and not the load's plan: a search holds its units'
/// weights and its working memory together, which on a small model is more
/// than the load will hold, and it has ended before the load begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TuningRoom {
    /// What may still be charged on the device's allocation domain.
    pub device_bytes: u64,
    /// What may still be claimed of host RAM, where the device's memory is
    /// apart from the host's. None where it is the host's: the two are one
    /// domain, which `device_bytes` bounds.
    pub host_bytes: Option<u64>,
}

/// What a search held, against the ceiling it is held to.
///
/// The search runs in a process of its own, before the served model's
/// weights are loaded: everything it holds is its own. It may hold what the
/// platform lets the engine claim ([`TuningRoom`]). A tuner given no room
/// has no ceiling. Work is admitted on what Seismic's ledgers count:
/// its charges on the device ([`seismic::MemoryUsage::charged`]) and the
/// host bytes of reference observations. Tuning's whole footprint, which
/// also holds host buffers, compiles and records, is read as each piece
/// ends and reported against its ceiling. The whole footprint is the process's and, where the
/// device's allocations lie outside it (`device_separate`), the device's
/// charges beside it, so that ceiling, peak and margin count the same
/// bytes on every backend.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TuningMemory {
    /// The most tuning's charges on the device may reach
    /// ([`TuningRoom::device_bytes`]); none when the tuner was given no
    /// room.
    pub ceiling_bytes: Option<u64>,
    /// The most tuning held as its ledgers count it: the device's
    /// high-water mark within any piece, with the host bytes of that
    /// piece's reference observations, since tuning began.
    pub held_peak_bytes: u64,
    /// The ceiling less that peak: negative when the invariant was broken.
    pub margin_bytes: Option<i64>,
    /// The process's footprint when the search's tuner was made, before it
    /// counted its units (what every growth of the process is counted
    /// from), the most it was when a piece ended, and what it was when the
    /// last one ended; none where the system reports none.
    pub process_began_bytes: Option<u64>,
    pub process_peak_footprint_bytes: Option<u64>,
    pub process_ended_bytes: Option<u64>,
    /// The device's allocations lie outside the process's footprint (a
    /// CUDA or Vulkan device, whose memory the driver holds): tuning's whole
    /// footprint is then the process's and the device's charges together.
    /// Elsewhere the process's footprint already counts them.
    pub device_separate: bool,
    /// What tuning's whole footprint may grow by, where the device's memory
    /// is the host's and the two are one domain: the device ceiling. None
    /// where the device's memory is apart.
    pub footprint_ceiling_bytes: Option<u64>,
    /// The most tuning's whole footprint grew since it began, as read when
    /// each piece ended: the growth of the process's footprint, with what
    /// tuning held on a separate device at most within that piece.
    pub footprint_peak_bytes: Option<u64>,
    /// That ceiling less that growth: negative when tuning's whole
    /// footprint passed it.
    pub footprint_margin_bytes: Option<i64>,
    /// Where the device's memory is apart from the host's
    /// (`device_separate`), the two do not substitute, and tuning is held
    /// in each: its charges on the device to `ceiling_bytes`, and what the
    /// process's footprint has grown by since tuning began to this, the
    /// host RAM the platform lets the engine claim
    /// ([`TuningRoom::host_bytes`]). None where the device's memory is the
    /// host's, which `footprint_ceiling_bytes` bounds as one.
    pub host_ceiling_bytes: Option<u64>,
    /// The most the process's footprint had grown since tuning began: the
    /// peak of the footprint ([`seismic::ProcessMemory::footprint_peak`])
    /// less the footprint tuning began at (`process_began_bytes`).
    pub host_peak_bytes: Option<u64>,
    /// That peak is the most of the readings tuning took of the footprint
    /// (around every piece, at the end of every worker thread's job, and
    /// where a compile holds its program and what it formed), not a mark
    /// the system keeps: what the footprint was between two readings is
    /// not in it ([`seismic::ProcessMemory::PEAK_IS_SAMPLED`]).
    pub host_peak_sampled: bool,
    /// What the process's footprint had grown by when the optional work
    /// began, the pages the mandatory phase freed returned: the canonical
    /// programs and what forming them left. The optional work has the
    /// host ceiling less this.
    pub host_mandatory_bytes: Option<u64>,
    /// Every piece and burst of compiles that added more to the host than
    /// it was priced at, in order: each raised its price to what it added.
    pub mispriced: Vec<Mispricing>,
    /// The optional work stopped because nothing offered fit the bytes
    /// free; none when it stopped for anything else.
    pub no_room: Option<NoRoom>,
    /// The host ceiling less that growth: negative when tuning's host
    /// working set passed the load's planned host peak.
    pub host_margin_bytes: Option<i64>,
    /// The most one piece of each unit's work added to the process's
    /// footprint: what the unit's next piece is taken to add to the host
    /// ([`allocation::Unit::host_footprint`]).
    pub host_added_by_unit: Vec<(String, u64)>,
    /// The host bytes one thread of a compile holds while it runs: the
    /// most the peak of the process's footprint rose over compiles that
    /// ran at once, for each thread that compiled. What a compile started
    /// ahead of its piece is admitted on
    /// ([`allocation::Unit::compile_host`]).
    pub compile_host_bytes: u64,
    /// Device charges when the mandatory phase began and when it ended.
    pub began_bytes: u64,
    pub mandatory_bytes: u64,
    /// The device's high-water mark over the optional work, within pieces
    /// included, and what was charged when the optional work ended.
    pub peak_bytes: u64,
    pub ended_bytes: u64,
    /// What the pieces of each class left held, in sum: discovery leaves a
    /// unit's built inputs and its defaults' observations, measurement its
    /// challengers' programs, kernels and observations.
    pub held_by_class: Vec<(seismic::Stage, i64)>,
    /// What each unit's pieces left held, in sum.
    pub held_by_unit: Vec<(String, i64)>,
    /// The process's peak resident set when the optional work ended, as
    /// the operating system counts it (`getrusage`): everything the
    /// process held, on the device or not, within pieces included.
    pub process_peak_bytes: Option<u64>,
    /// The process's memory by what holds it ([`seismic::ProcessMemory`]):
    /// when tuning began, when a piece ended at the most footprint, when
    /// the optional work ended, and once the pages tuning freed were
    /// returned to the system ([`seismic::return_freed_pages`]).
    pub breakdown_began: seismic::ProcessMemory,
    pub breakdown_peak: seismic::ProcessMemory,
    pub breakdown_ended: seismic::ProcessMemory,
    pub breakdown_returned: seismic::ProcessMemory,
    /// What returning freed pages took the footprint down by over the
    /// search, and what the returning took. The search's process exits
    /// when it ends and no load runs in it, so what the search leaves in
    /// its process is carried by nothing and is not reported.
    pub returned_bytes: u64,
    pub returning_seconds: f64,
}

/// Work that added more to the host than it was priced at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mispricing {
    /// A unit, for one of its pieces; `"compile"` for one thread of a
    /// burst of compiles.
    pub kind: String,
    /// What it was admitted on, and what it added.
    pub priced_bytes: u64,
    pub added_bytes: u64,
}

/// The stop of the optional work for bytes: with nothing in flight and the
/// pages tuning freed returned, the smallest piece still offered did not
/// fit what was free. It is neither the budget running out nor a unit's
/// domain exhausted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoRoom {
    /// The domains the smallest piece did not fit, each as it stood at the
    /// stop.
    pub domains: Vec<Room>,
    /// What a load holds on the host that the host ceiling has no term for
    /// ([`crate::planning::HostPeak::UNPLANNED`]): each would raise the
    /// ceiling by what a load really holds of it, were it planned.
    pub unplanned_holders: &'static [&'static str],
}

/// One memory domain at a stop for bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Room {
    /// `"device"`: the device's charges under what the platform lets the
    /// engine claim on it; `"footprint"`: where the device's memory is the
    /// host's, everything the process holds under that same ceiling;
    /// `"host"`: where it is apart, the process's footprint under what the
    /// platform lets the engine claim of host RAM.
    pub domain: &'static str,
    pub ceiling_bytes: u64,
    /// What tuning held in the domain, as measured, at the stop.
    pub held_bytes: u64,
    /// The least any refused piece was taken to add to the domain.
    pub smallest_footprint_bytes: u64,
    /// What the domain lacked for that piece.
    pub lacking_bytes: u64,
}

/// The process's peak resident set so far.
fn process_peak_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: `getrusage` writes a whole `rusage` to the pointer it is
        // given when it returns zero.
        let usage = unsafe {
            (libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0)
                .then(|| usage.assume_init())
        }?;
        let peak = u64::try_from(usage.ru_maxrss).ok()?;
        // Bytes on macOS, kibibytes elsewhere.
        Some(if cfg!(target_os = "macos") {
            peak
        } else {
            peak.saturating_mul(1024)
        })
    }
    #[cfg(not(unix))]
    None
}

/// Whether `device`'s allocations lie outside this process's footprint: the
/// driver of a CUDA or Vulkan device holds them, while a CPU device's are
/// the process's own memory and a Metal device's count in its footprint.
fn device_separate(device: &Device) -> bool {
    match device.backend() {
        seismic::BackendName::Cuda | seismic::BackendName::Vulkan => true,
        seismic::BackendName::Cpu | seismic::BackendName::Metal => false,
    }
}

/// The process's physical footprint now, where the system reports one: what
/// its memory pressure and its peak-footprint accounting count. On Linux
/// that is its resident anonymous and shared memory; resident pages of
/// mapped files, which the system reclaims without swapping, are not
/// footprint there or on macOS.
fn process_footprint_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v2>::uninit();
        // SAFETY: `proc_pid_rusage` writes a whole `rusage_info_v2` to the
        // buffer it is given for that flavor when it returns zero.
        unsafe {
            (libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V2,
                usage.as_mut_ptr().cast(),
            ) == 0)
                .then(|| usage.assume_init().ri_phys_footprint)
        }
    }
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let kibibytes = |field: &str| {
            status
                .lines()
                .find_map(|line| line.strip_prefix(field))?
                .trim()
                .strip_suffix("kB")?
                .trim()
                .parse::<u64>()
                .ok()
        };
        Some((kibibytes("RssAnon:")? + kibibytes("RssShmem:")?) * 1024)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    None
}

impl TuningMemory {
    /// What tuning holds now, as admission counts it: Seismic's charges on
    /// the device since tuning began.
    fn held(&self, device: &Device) -> u64 {
        device.memory_usage().charged.saturating_sub(self.began_bytes)
    }

    /// What tuning holds now in all: what the process's footprint has
    /// grown by since it began (heap, compiles and whatever a toolchain or
    /// framework keeps, which no ledger counts) and, where the device's
    /// allocations lie outside the process (`device_separate`), its charges
    /// on the device beside it. A CPU or Metal device's allocations are in
    /// the footprint. None where the system reports no footprint.
    fn held_whole(&self, device: &Device) -> Option<u64> {
        let grown = seismic::ProcessMemory::footprint_now()?
            .saturating_sub(self.breakdown_began.footprint_bytes?);
        Some(if self.device_separate {
            self.held(device) + grown
        } else {
            grown
        })
    }

    /// The bytes tuning may still take in the domain the device allocates
    /// in; none without a ceiling. Where that domain is the host's, two
    /// readings hold at once and what is free is the lesser room: its
    /// charges on the device, and everything it holds, the host's growth
    /// included, each under the device's ceiling. Where the device's memory is apart, this
    /// is the device's room alone and the host has its own
    /// ([`TuningMemory::host_free`]).
    fn free(&self, device: &Device) -> Option<u64> {
        let charged = self
            .ceiling_bytes
            .map(|ceiling| ceiling.saturating_sub(self.held(device)));
        if self.device_separate {
            return charged;
        }
        let whole = self
            .footprint_ceiling_bytes
            .zip(self.held_whole(device))
            .map(|(ceiling, held)| ceiling.saturating_sub(held));
        match (charged, whole) {
            (Some(charged), Some(whole)) => Some(charged.min(whole)),
            (charged, whole) => charged.or(whole),
        }
    }

    /// What the process's footprint has grown by since tuning began: its
    /// host working set, where the device's memory is apart. The footprint
    /// is the one quantity on every system (anonymous and shared resident
    /// memory on Linux, the physical footprint on macOS): resident pages
    /// of mapped files, a compiler's own among them, are in neither, here
    /// or in the load this is held to.
    fn host_grown(&self) -> Option<u64> {
        Some(seismic::ProcessMemory::footprint_now()?.saturating_sub(self.process_began_bytes?))
    }

    /// The host bytes tuning may still take where host RAM is a domain of
    /// its own: the load's planned host peak less what the process's
    /// footprint has grown by, which counts heap freed and not yet
    /// returned. Unbounded where the device's memory is the host's, and
    /// without a ceiling.
    fn host_free(&self) -> allocation::HostRoom {
        match self.host_ceiling_bytes.zip(self.host_grown()) {
            None => allocation::HostRoom::Unbounded,
            Some((ceiling, grown)) => ceiling
                .checked_sub(grown)
                .map_or(allocation::HostRoom::Exceeded, allocation::HostRoom::Free),
        }
    }

    /// The most a piece of `unit`'s work has added to the process's
    /// footprint; none before one has run.
    fn host_added(&self, unit: &str) -> Option<u64> {
        self.host_added_by_unit
            .iter()
            .find(|(name, _)| name == unit)
            .map(|(_, added)| *added)
    }

    /// Record what a piece of `unit`'s work added to the process's
    /// footprint: the unit's price is the most one has added, and a piece
    /// that added more than the price it was admitted on is kept as a
    /// mispricing.
    fn host_piece(&mut self, unit: &str, added: u64) {
        match self
            .host_added_by_unit
            .iter_mut()
            .find(|(name, _)| name == unit)
        {
            Some((_, most)) if added > *most => {
                self.mispriced.push(Mispricing {
                    kind: unit.to_owned(),
                    priced_bytes: *most,
                    added_bytes: added,
                });
                *most = added;
            }
            Some(_) => {}
            None => self.host_added_by_unit.push((unit.to_owned(), added)),
        }
    }

    /// The bytes free under the ceiling when the optional work holds
    /// nothing: the ceiling less what the mandatory phase left held, the
    /// canonical programs serving takes. None without a ceiling.
    fn capacity(&self) -> Option<u64> {
        self.ceiling_bytes.map(|ceiling| {
            ceiling.saturating_sub(self.mandatory_bytes.saturating_sub(self.began_bytes))
        })
    }

    /// Record what was held at most: `within`, a piece's own high-water
    /// mark, and what is held now that it ended.
    fn observe(&mut self, device: &Device, within: u64) {
        let footprint = process_footprint_bytes();
        if footprint > self.process_peak_footprint_bytes {
            self.breakdown_peak = seismic::ProcessMemory::read();
        }
        self.process_peak_footprint_bytes = self.process_peak_footprint_bytes.max(footprint);
        self.peak_bytes = self.peak_bytes.max(device.memory_usage().high_water);
        self.held_peak_bytes = self
            .held_peak_bytes
            .max(within.saturating_sub(self.began_bytes))
            .max(self.held(device));
        self.margin_bytes = self
            .ceiling_bytes
            .map(|ceiling| ceiling as i64 - self.held_peak_bytes as i64);
        self.process_ended_bytes = footprint;
        let held_within = within
            .saturating_sub(self.began_bytes)
            .max(self.held(device));
        let grown = footprint
            .zip(self.process_began_bytes)
            .map(|(now, began)| {
                now.saturating_sub(began) + if self.device_separate { held_within } else { 0 }
            });
        self.footprint_peak_bytes = self.footprint_peak_bytes.max(grown);
        self.footprint_margin_bytes = self
            .footprint_ceiling_bytes
            .zip(self.footprint_peak_bytes)
            .map(|(ceiling, peak)| ceiling as i64 - peak as i64);
        // The footprint's high-water mark: the system's own where it keeps
        // one, and what the process held between two readings is then in
        // it; elsewhere the most of the readings taken.
        self.host_peak_sampled = seismic::ProcessMemory::PEAK_IS_SAMPLED;
        self.host_peak_bytes = self.host_peak_bytes.max(
            seismic::ProcessMemory::footprint_peak()
                .zip(self.process_began_bytes)
                .map(|(peak, began)| peak.saturating_sub(began)),
        );
        self.host_margin_bytes = self
            .host_ceiling_bytes
            .zip(self.host_peak_bytes)
            .map(|(ceiling, peak)| ceiling as i64 - peak as i64);
    }

    /// The threads the host's room affords compiles that run at once, each
    /// at the price a thread's compile was observed to hold
    /// (`compile_host_bytes`); none where the host is not a domain apart,
    /// and nothing is then bounded. Before a compile has been observed no
    /// room can be divided by its price, and compiles run one at a time.
    /// Owed work is not refused, so it is afforded one thread at the
    /// least.
    fn compile_threads(&self) -> Option<usize> {
        let price = self.compile_host_bytes;
        match self.host_free() {
            allocation::HostRoom::Unbounded => None,
            _ if price == 0 => Some(1),
            allocation::HostRoom::Free(free) => Some((free / price) as usize),
            allocation::HostRoom::Exceeded => Some(0),
        }
    }

    /// Return the pages tuning freed to the system, and count what that
    /// took the footprint down by.
    fn return_freed_pages(&mut self) {
        let began = Instant::now();
        self.returned_bytes += seismic::return_freed_pages();
        self.returning_seconds += began.elapsed().as_secs_f64();
    }

    fn piece(&mut self, unit: String, stage: seismic::Stage, before: u64, after: u64) {
        let held = after as i64 - before as i64;
        match self.held_by_class.iter_mut().find(|(class, _)| *class == stage) {
            Some((_, sum)) => *sum += held,
            None => self.held_by_class.push((stage, held)),
        }
        match self.held_by_unit.iter_mut().find(|(name, _)| *name == unit) {
            Some((_, sum)) => *sum += held,
            None => self.held_by_unit.push((unit, held)),
        }
    }
}

/// A piece's compile that ran ahead of it: its price and what it took.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct FormedAhead {
    pub priced_seconds: f64,
    pub unpriced_operations: usize,
    pub actual_seconds: f64,
}

/// What became of one load's tuning budget. Times are seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct BudgetReport {
    pub budget_seconds: f64,
    /// Tuning time when the optional work ended, the records its units
    /// owed published.
    pub used_seconds: f64,
    /// What publishing those records took, part of `used_seconds`.
    pub settling_seconds: f64,
    /// Budget left then: no work remained whose price fit it.
    pub unused_seconds: f64,
    /// Time past the budget then: what the last piece started ran beyond
    /// its forecast. Recorded, never a failure.
    pub overrun_seconds: f64,
    /// Pieces of work started.
    pub stages: usize,
    /// The work still offered when the allocator stopped, none of which
    /// fit.
    pub refused: Vec<RefusedOffer>,
    /// The allocator's own work between the last piece and its stop.
    pub ending: EndingReport,
    pub memory: TuningMemory,
}

/// The allocator's own work from the end of the last piece performed to
/// its stop, beside the publications settled there
/// ([`allocation::Ending`]). Times are seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EndingReport {
    /// Tuning time when the last piece performed ended.
    pub began_seconds: f64,
    /// What the start of that piece held back for the allocator's own
    /// ending.
    pub held_seconds: f64,
    /// The allocator's passes since and what deciding in them took.
    pub passes: usize,
    pub deciding_seconds: f64,
    /// The pieces narrowed since and what narrowing them took.
    pub narrowed: usize,
    pub narrowing_seconds: f64,
    /// The units that gave their inputs up since and what that took.
    pub released: usize,
    pub releasing_seconds: f64,
}

/// A piece of work still offered when the allocator stopped, its price in
/// seconds and in bytes, and which of the two it did not fit.
#[derive(Clone, Debug, PartialEq)]
pub struct RefusedOffer {
    pub unit: String,
    pub stage: seismic::Stage,
    pub priced_seconds: f64,
    pub unpriced_operations: usize,
    pub footprint_bytes: u64,
    pub unpriced_footprints: usize,
    /// `"seconds"`: what remained of the budget; `"bytes"`: what was free
    /// under the ceiling with every other unit's inputs given up.
    pub reason: &'static str,
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
    /// The original receipt, shared with observers without copying candidate
    /// measurements. Cached loads retain the same evidence as the first load:
    /// a stored unit's is named where the store keeps it and read by the
    /// observer that states it, not by the load.
    pub receipt: Receipt,
    pub overall: Configuration,
    /// Exact static domain of this unit, fixed by the same cache key.
    pub statics: BTreeMap<String, u64>,
    pub origin: TuningOrigin,
    /// Whether the chosen configuration was numerically qualified against its
    /// reference. `false` is the explicit canonical fallback: the declaration
    /// default, kept because the first-load allowance could not qualify a
    /// candidate or every candidate was rejected.
    pub qualified: bool,
    /// Whether the result's search reached its own end, read from the
    /// result itself, so a stored result reports what its first load did.
    pub stop: Option<SearchStop>,
    /// The coverage checks of the unit's leaders, from the result itself.
    pub coverage: Option<seismic::CoverageCheck>,
    /// How far a searched unit's search got at this load. Reported, not
    /// stored: nothing reads it to tune a unit again.
    pub progress: Option<SearchProgress>,
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

/// How far one unit's search got within the tuning time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchProgress {
    /// The unit's share of expected step time at its defaults.
    pub share: f64,
    /// Pieces of optional work the unit was given.
    pub stages: usize,
    /// Where the unit stood when the load's optional work ended: its served
    /// cost, the configurations measured of those it can reach, and the
    /// programs formed.
    pub standing: Standing,
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
    /// Where final first-load choices are retained. `None` is an isolated
    /// preparation without durable reuse, not a continuation mechanism.
    pub cache: Option<&'a KernelCache>,
    /// The caller's numerical requirement: α.
    pub numerical_profile: &'a ResolvedNumericalProfile,
    pub settings: TuningSettings,
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
                ));
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
                    .and_then(|head| {
                        head.blocks
                            .iter()
                            .flat_map(|block| &block.blocks)
                            .nth(index.block as usize)
                    })
                    .map(|block| &block.sublayers),
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
    choices: &'w HashMap<TuningKey, NativeSpecialization>,
}

impl<'a> std::ops::Deref for TuningInputs<'_, 'a> {
    type Target = ModelInputs<'a>;

    fn deref(&self) -> &ModelInputs<'a> {
        &self.model
    }
}

impl TuningInputs<'_, '_> {
    pub(super) fn choice(&self, key: &TuningKey) -> Option<&NativeSpecialization> {
        self.choices.get(key)
    }

    /// Drop every weight and shared tensor no built input binds any more.
    fn release_unheld(&mut self) {
        self.weights.release_unheld();
        self.shared
            .retain(|_, tensor| weights::held_elsewhere(tensor));
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
        self.weights.weight(scope, kind)
    }

    /// Retain a requested shared readout operand by physical storage identity.
    pub fn shared_weight(
        &mut self,
        scope: WeightScope,
        kind: WeightKind,
    ) -> Result<Tensor, String> {
        let plan = weights::weight_plan(self.weights.load(), scope, kind)?;
        let key = format!(
            "shared_weight:{:?}:{:?}",
            plan.storage_identity(),
            plan.shape
        );
        if let Some(weight) = self.shared.get(&key) {
            return Ok(weight.clone());
        }
        let weight = self.weight(scope, kind)?;
        self.shared.insert(key, weight.clone());
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
    /// its element count.
    pub fn generated(
        &self,
        element: Element,
        extents: &[u64],
        generate: impl FnOnce(usize) -> Result<Vec<u8>, String>,
    ) -> Result<Tensor, String> {
        let count = element_count(extents)?;
        if element.dtype().is_none() {
            return Err(format!(
                "generated tuning tensors are not {}",
                element.name()
            ));
        }
        let bytes = generate(count)?;
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

    /// Row tables for `rows` rows over one request for each of `histories`,
    /// each with `context` accepted history rows in the ranges its entry
    /// names ([`history_placement`]), packed by the batch builder. Batch
    /// row `i` appends at history row `appends + i`, past every row any
    /// point reads, so no point's appends change another point's inputs.
    pub fn batch(
        &self,
        rows: u64,
        context: u64,
        histories: &[Vec<[i32; 2]>],
        appends: u64,
    ) -> Result<PackedRowTables, String> {
        let rows = usize::try_from(rows).map_err(|_| "tuning rows exceed usize")?;
        let slots = histories.len();
        let context = i32::try_from(context).map_err(|_| "tuning context exceeds i32")?;
        let appends = i32::try_from(appends).map_err(|_| "tuning history exceeds i32")?;
        if slots == 0 || slots > rows {
            return Err(format!("{slots} slots cannot share {rows} rows"));
        }
        let spans = histories.iter().map(Vec::len).max().unwrap_or(0);
        let mut row_index = 0i32;
        let packed = histories
            .iter()
            .enumerate()
            .map(|(slot, visible)| {
                let count = rows / slots + usize::from(slot < rows % slots);
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
                                    visible: visible.clone(),
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
        // The span class a store seals for the most spans a row sees.
        let limits = ClassLimits {
            rows,
            segments: spans.max(1).next_power_of_two(),
        };
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
pub(super) type TuningKey = (&'static str, String, BTreeMap<String, u64>);

/// Where a preparation's tuning is. One [`Tuner`] walks the program once per
/// phase.
enum Phase<'a> {
    /// Recording each unit, its launches per step and whether it will
    /// search; nothing is formed or measured.
    Count {
        launches: HashMap<TuningKey, usize>,
        searching: Vec<TuningKey>,
    },
    /// Every searching unit's mandatory work is done: it serves its
    /// published canonical default. The walk collects the units with
    /// optional work, in preparation order; nothing is formed or measured.
    Census {
        launches: HashMap<TuningKey, usize>,
        units: Vec<Box<dyn SearchedUnit<'a> + 'a>>,
    },
    /// Every censused unit was searched ([`allocation`]) and its record is
    /// published: the end of a search. Nothing walks after it; a unit's
    /// published choice can still be read.
    Searched,
    /// A load's only phase: one walk reads each unit's stored record and
    /// prepares its choice. At a unit without a record the walk stops and
    /// reports it, or prepares the unit at its canonical default, as its
    /// host asked. Nothing is searched, measured or published.
    Serve(UnpreparedUnits),
}

/// A composed execution recipe competes for the same first-load allowance
/// as entry tuning. The consumer owns its qualification evidence and cache.
pub(super) trait ComposedContender<'a> {
    fn dependencies(&self) -> Vec<TuningKey>;
    fn key(&self) -> String;
    fn label(&self) -> &'static str;
    fn cached(&self) -> Option<bool>;
    fn set_invocations(
        &mut self,
        invocations: &crate::programs::invocation_demand::InvocationCounts,
    );
    /// Form the contender and measure both recipes once: its discovery,
    /// run to its end. None when the contender cannot be formed or built.
    fn census(
        &mut self,
        inputs: &mut TuningInputs<'_, 'a>,
    ) -> Result<Option<ComposedCensus>, CatalogFailure>;
    /// Compare the contender with the canonical recipe, to the end: its
    /// one piece of work, admitted on the price its census gave it.
    fn qualify(&mut self, inputs: &mut TuningInputs<'_, 'a>) -> Result<bool, CatalogFailure>;
    /// Retain the choice as it stands: the canonical recipe from the
    /// mandatory phase on, the composed one once it is qualified.
    fn conclude(&mut self, selected: bool) -> Result<(), CatalogFailure>;
}

/// Actual census evidence prices a composed contender; no guessed grace period.
pub(super) struct ComposedCensus {
    pub served: Vec<PointShape>,
    pub measured: Vec<PointShape>,
    pub seconds: Vec<f64>,
    /// Remaining pair/confirmation work, priced from the actual census.
    pub qualification: Duration,
}

struct ComposedUnit<'a> {
    key: TuningKey,
    contender: Box<dyn ComposedContender<'a> + 'a>,
    census: Option<ComposedCensus>,
    share: f64,
    /// The recipe was qualified against its prerequisites' choices: its
    /// one piece of work is done.
    qualified: Option<bool>,
    prerequisites: Vec<usize>,
    demanded: bool,
}

impl<'a> SearchedUnit<'a> for ComposedUnit<'a> {
    fn census_optional(&mut self, inputs: &mut TuningInputs<'_, 'a>) -> Result<(), CatalogFailure> {
        self.census = if self.demanded {
            self.contender.census(inputs)?
        } else {
            None
        };
        if self.census.as_ref().is_some_and(|measured| {
            measured.measured.is_empty()
                || measured.measured.len() != measured.seconds.len()
                || measured
                    .seconds
                    .iter()
                    .any(|seconds| !seconds.is_finite() || *seconds <= 0.)
        }) {
            return Err(CatalogFailure::Tuning {
                entry: self.contender.label(),
                bindings: self.key.1.clone(),
                outcome: "composed census lacks finite measured service evidence".into(),
            });
        }
        Ok(())
    }
    fn dependency_keys(&self) -> Vec<TuningKey> {
        if self.census.is_some() {
            self.contender.dependencies()
        } else {
            Vec::new()
        }
    }
    fn set_prerequisites(&mut self, prerequisites: Vec<usize>) {
        self.prerequisites = prerequisites;
    }
    fn prerequisites(&self) -> Vec<usize> {
        self.prerequisites.clone()
    }
    fn composed(&self) -> bool {
        true
    }
    fn key(&self) -> &TuningKey {
        &self.key
    }
    fn time(&self) -> (usize, Vec<PointShape>, Vec<PointShape>, Vec<f64>) {
        self.census.as_ref().map_or_else(
            || (1, Vec::new(), Vec::new(), Vec::new()),
            |census| {
                (
                    1,
                    census.served.clone(),
                    census.measured.clone(),
                    census.seconds.clone(),
                )
            },
        )
    }
    fn share(&self) -> f64 {
        self.share
    }
    fn set_share(&mut self, share: f64) {
        self.share = share;
    }

    /// The recipe's one piece of work: its qualification against its
    /// prerequisites' choices as they stand, priced by its own census. It
    /// is one more contender beside the canonical recipe.
    fn next(&self) -> Option<seismic::Work> {
        let census = self.census.as_ref().filter(|_| self.qualified.is_none())?;
        Some(seismic::Work {
            stage: seismic::Stage::Measure,
            quote: seismic::Quote {
                priced: census.qualification,
                unpriced: 0,
            },
            forming: seismic::Quote::default(),
            gain: 1. / 2.,
            // What a recipe's qualification holds is its contender's own:
            // nothing of it was observed to price.
            footprint: seismic::Footprint::default(),
        })
    }

    fn forming(&self) -> Option<seismic::Forming> {
        None
    }

    fn formed(
        &mut self,
        _: seismic::Formed,
        _: Duration,
        _: &mut Tuner<'a>,
    ) -> Result<(), CatalogFailure> {
        unreachable!("a composed recipe compiles nothing ahead")
    }

    fn perform(&mut self, tuner: &mut Tuner<'a>) -> Result<Duration, CatalogFailure> {
        let work = self.next().expect("a unit performs the work it named");
        let began = Instant::now();
        let began_seconds = tuner.started.elapsed().as_secs_f64();
        let selected = self.contender.qualify(&mut tuner.inputs())?;
        let qualified = began.elapsed();
        self.qualified = Some(selected);
        let publishing = Instant::now();
        self.contender.conclude(selected)?;
        tuner.composed_choices.insert(self.key.1.clone(), selected);
        let published = publishing.elapsed();
        tuner.context.observer.event(&TuningEvent::Stage(StageRecord {
            entry: self.key.0,
            bindings: self.key.1.clone(),
            statics: self.key.2.clone(),
            share: self.share,
            stage: work.stage,
            priced_seconds: work.quote.priced.as_secs_f64(),
            unpriced: work.quote.unpriced,
            forecast_seconds: work.quote.forecast().map(|forecast| forecast.as_secs_f64()),
            began_seconds,
            actual_seconds: began.elapsed().as_secs_f64(),
            formed: None,
            spent: seismic::Spent::default(),
            raises: Vec::new(),
            publishing_seconds: published.as_secs_f64(),
            promoted: selected,
        }));
        Ok(qualified.max(published))
    }

    /// A recipe never qualified keeps the canonical choice it published;
    /// one that was censused releases what its census formed.
    fn finish(&mut self, tuner: &mut Tuner<'a>) -> Result<(), CatalogFailure> {
        if self.qualified.is_none() {
            if self.census.is_some() {
                self.contender.conclude(false)?;
            }
            tuner.composed_choices.insert(self.key.1.clone(), false);
        }
        Ok(())
    }
}

/// One unit with optional work, whatever its entry: its opened search, its
/// case and the inputs built for it. It serves a published legal choice
/// throughout.
trait SearchedUnit<'a> {
    /// Measure a composed recipe's service: its discovery.
    fn census_optional(&mut self, _: &mut TuningInputs<'_, 'a>) -> Result<(), CatalogFailure> {
        Ok(())
    }
    fn dependency_keys(&self) -> Vec<TuningKey> {
        Vec::new()
    }
    fn set_prerequisites(&mut self, _: Vec<usize>) {}
    fn prerequisites(&self) -> Vec<usize> {
        Vec::new()
    }
    fn composed(&self) -> bool {
        false
    }
    fn key(&self) -> &TuningKey;
    /// The unit's step time as its census measured it: its launches per
    /// step, the points it serves, its timed points and its defaults' time
    /// at each.
    fn time(&self) -> (usize, Vec<PointShape>, Vec<PointShape>, Vec<f64>);
    fn share(&self) -> f64;
    fn set_share(&mut self, share: f64);
    /// The unit's next piece of optional work, priced in full.
    fn next(&self) -> Option<seismic::Work>;
    /// The compile of that piece, to run on a worker ahead of it.
    fn forming(&self) -> Option<seismic::Forming>;
    /// Hold what that compile formed, which took `took`.
    fn formed(
        &mut self,
        formed: seismic::Formed,
        took: Duration,
        tuner: &mut Tuner<'a>,
    ) -> Result<(), CatalogFailure>;
    /// Run that piece to its end; when it changed the unit's served choice,
    /// publish the unit's record with it. Returns the longest single
    /// operation the unit has completed.
    fn perform(&mut self, tuner: &mut Tuner<'a>) -> Result<Duration, CatalogFailure>;
    /// What the unit owes when the optional work stops, priced
    /// ([`allocation::Unit::owed`]).
    fn owed(&self, _worked: bool) -> seismic::Quote {
        seismic::Quote::default()
    }
    /// Publish the unit's record if its work changed it since it was
    /// published: the optional work has stopped.
    fn settle(&mut self, _: &mut Tuner<'a>) -> Result<(), CatalogFailure> {
        Ok(())
    }
    /// The bytes of the inputs the unit holds; zero when it holds none.
    fn holding(&self) -> u64 {
        0
    }
    /// What building the inputs the unit holds took
    /// ([`allocation::Unit::built`]).
    fn built(&self) -> seismic::Quote {
        seismic::Quote::default()
    }
    /// The price of settling now, when that is the load's first
    /// observation of a class of owed work ([`allocation::Unit::sample`]).
    fn sample(&self) -> Option<seismic::Quote> {
        None
    }
    /// Give up the inputs the unit holds.
    fn release(&mut self, _: &mut Tuner<'a>) {}
    /// The bytes of the formed programs the unit holds; zero when it holds
    /// none.
    fn programs(&self) -> u64 {
        0
    }
    /// Give up the formed programs the unit holds.
    fn give_up(&mut self) {}
    /// Offer the next piece with one challenger fewer; false when it has
    /// no fewer to offer (a piece that is one whole thing never has).
    fn narrow(&mut self) -> bool {
        false
    }
    /// The optional work stopped with the unit's next piece refused for
    /// its bytes ([`allocation::Unit::refused_for_bytes`]).
    fn refused_for_bytes(&mut self, _: &mut Tuner<'a>) {}
    /// The most a single piece of the unit's work has added to the bytes
    /// held.
    fn largest_footprint(&self) -> u64 {
        0
    }
    /// The load's optional work is over: report the unit's record.
    fn finish(&mut self, tuner: &mut Tuner<'a>) -> Result<(), CatalogFailure>;
}

impl<'a> allocation::Unit for Box<dyn SearchedUnit<'a> + 'a> {
    fn refresh(units: &mut [Self], tuner: &mut Tuner<'a>) {
        let times = units.iter().map(|unit| unit.time()).collect::<Vec<_>>();
        let shares = step_shares(
            &times
                .iter()
                .map(|(_, served, shapes, seconds)| UnitTime {
                    invocations: crate::programs::invocation_demand::InvocationCounts::new(
                        served.iter().map(|point| (point.rows, point.weight)),
                    ),
                    served,
                    shapes,
                    seconds,
                })
                .collect::<Vec<_>>(),
            tuner.limits,
        );
        for (unit, share) in units.iter_mut().zip(shares) {
            unit.set_share(share);
        }
    }
    fn prerequisites(&self) -> Vec<usize> {
        (**self).prerequisites()
    }
    type Context = Tuner<'a>;
    type Failure = CatalogFailure;
    type Forming = seismic::Forming;
    type Formed = seismic::Formed;

    fn share(&self) -> f64 {
        (**self).share()
    }

    fn next(&self) -> Option<seismic::Work> {
        (**self).next()
    }

    fn forming(&self) -> Option<seismic::Forming> {
        (**self).forming()
    }

    fn formed(
        &mut self,
        formed: seismic::Formed,
        took: Duration,
        tuner: &mut Tuner<'a>,
    ) -> Result<(), CatalogFailure> {
        (**self).formed(formed, took, tuner)
    }

    fn perform(&mut self, tuner: &mut Tuner<'a>) -> Result<Duration, CatalogFailure> {
        let stage = (**self)
            .next()
            .expect("a unit performs the work it named")
            .stage;
        let before = tuner.device.memory_usage().charged;
        let host_before = seismic::ProcessMemory::footprint_now();
        let peak_before = seismic::ProcessMemory::footprint_peak();
        tuner.device.restart_memory_high_water();
        let longest = (**self).perform(tuner)?;
        let unit = unit_name((**self).key());
        // What the piece added at its most: where the footprint's peak
        // rose while it ran, the peak over where it began (with what the
        // compiles in flight took meanwhile, which only prices it higher);
        // otherwise what it left added when it ended.
        if let Some((before, after)) = host_before.zip(seismic::ProcessMemory::footprint_now()) {
            let most = peak_before
                .zip(seismic::ProcessMemory::footprint_peak())
                .filter(|(before, peak)| peak > before)
                .map_or(after, |(_, peak)| peak.max(after));
            tuner
                .memory
                .host_piece(&unit, most.saturating_sub(before));
        }
        tuner.memory.piece(
            unit,
            stage,
            before,
            tuner.device.memory_usage().charged,
        );
        let within = tuner.device.memory_usage().high_water;
        tuner.memory.observe(tuner.device, within);
        tuner.report_progress();
        Ok(longest)
    }

    fn holding(&self) -> u64 {
        (**self).holding()
    }

    fn built(&self) -> seismic::Quote {
        (**self).built()
    }

    fn sample(&self) -> Option<seismic::Quote> {
        (**self).sample()
    }

    fn release(&mut self, tuner: &mut Tuner<'a>) {
        // What the unit gives up is freed; its pages are returned when a
        // piece needs the room ([`allocation::Unit::reclaim`]) and when
        // tuning ends. Returning them at every release took 2.6 s of one
        // load's 47 s of tuning.
        (**self).release(tuner);
    }

    fn reclaim(tuner: &mut Tuner<'a>) {
        tuner.memory.return_freed_pages();
    }

    fn programs(&self) -> u64 {
        (**self).programs()
    }

    fn give_up(&mut self, _: &mut Tuner<'a>) {
        (**self).give_up();
    }

    fn narrow(&mut self, _: &mut Tuner<'a>) -> bool {
        (**self).narrow()
    }

    fn refused_for_bytes(&mut self, tuner: &mut Tuner<'a>) {
        (**self).refused_for_bytes(tuner);
    }

    fn largest_footprint(&self) -> u64 {
        (**self).largest_footprint()
    }

    fn owed(&self, worked: bool) -> seismic::Quote {
        (**self).owed(worked)
    }

    fn settle(&mut self, tuner: &mut Tuner<'a>) -> Result<(), CatalogFailure> {
        (**self).settle(tuner)
    }

    fn free(tuner: &Tuner<'a>) -> Option<u64> {
        tuner.memory.free(tuner.device)
    }

    fn host_free(tuner: &Tuner<'a>) -> allocation::HostRoom {
        tuner.memory.host_free()
    }

    fn host_footprint(&self, tuner: &Tuner<'a>) -> Option<u64> {
        tuner.memory.host_added(&unit_name((**self).key()))
    }

    fn compile_host(tuner: &Tuner<'a>) -> u64 {
        tuner.memory.compile_host_bytes
    }

    fn capacity(tuner: &Tuner<'a>) -> Option<u64> {
        tuner.memory.capacity()
    }
}

/// A unit as the memory report names it.
fn unit_name(key: &TuningKey) -> String {
    format!("{} [{}]", key.0, key.1)
}

/// What publishing a serving artifact has taken in this load: a class of
/// its own, observed against what the publication does
/// ([`seismic::Publication`]) and the threads its compiles had. A
/// publication that compiles nothing is one class, priced by its mean. One
/// that compiles is priced by its unit's canonical formation, in turn
/// ([`ArtifactPrice::work`]), as the load's compiled publications have
/// followed it ([`Followed`]). Before the load has run one of a class, the
/// unit's own stand-in prices it ([`seismic::NativeSearch::artifact_owed`]).
#[derive(Clone, Copy, Default)]
struct ArtifactPrice {
    kept: seismic::Observed,
    compiled: Followed,
    /// The threads a publication's compiles had when one last ran: what
    /// the host's room afforded of the preparation's
    /// ([`Tuner::narrow_publication`]).
    threads: Option<usize>,
}

/// What one kind of work has taken against a measure of each instance
/// known before it runs, both in seconds: how the work follows the
/// measure. One instance, or instances of one measure, give the ratio of
/// what they took to it. Instances of different measures give the line
/// through them where that has a cost besides the measure and a cost of
/// the measure, neither below nothing; otherwise the ratio that fits them
/// all best.
#[derive(Clone, Copy, Default)]
struct Followed {
    count: f64,
    measure: f64,
    seconds: f64,
    measure_squared: f64,
    measure_seconds: f64,
}

impl Followed {
    fn observe(&mut self, measure: f64, seconds: f64) {
        self.count += 1.;
        self.measure += measure;
        self.seconds += seconds;
        self.measure_squared += measure * measure;
        self.measure_seconds += measure * seconds;
    }

    /// The price of an instance of `measure`; none before an observation.
    fn price(&self, measure: f64) -> Option<f64> {
        if self.count == 0. {
            return None;
        }
        // Instances whose measure was nothing say only what they took.
        if self.measure_squared == 0. {
            return Some(self.seconds / self.count);
        }
        let ratio = self.measure_seconds / self.measure_squared;
        let spread = self.count * self.measure_squared - self.measure * self.measure;
        if spread <= self.measure_squared * f64::EPSILON * self.count {
            return Some(ratio * measure);
        }
        let each = (self.count * self.measure_seconds - self.measure * self.seconds) / spread;
        let fixed = (self.seconds - each * self.measure) / self.count;
        Some(if each < 0. || fixed < 0. {
            ratio * measure
        } else {
            fixed + each * measure
        })
    }
}

impl ArtifactPrice {
    /// The measure of a publication that compiles: its unit's canonical
    /// formation, the same source compiled once already in this process,
    /// for each turn its programs take on the threads they have (they
    /// compile at once, so one turn where the threads cover them).
    fn work(publication: seismic::Publication, threads: Option<usize>) -> f64 {
        let turns = threads.map_or(1, |threads| publication.sources.div_ceil(threads.max(1)));
        publication.canonical.as_secs_f64() * turns as f64
    }

    /// The class's price; none before the load has published one of it.
    fn price(&self, publication: seismic::Publication) -> Option<f64> {
        if publication.sources == 0 {
            self.kept.price()
        } else {
            self.compiled.price(Self::work(publication, self.threads))
        }
    }

    fn observe(&mut self, publication: seismic::Publication, seconds: f64) {
        if publication.sources == 0 {
            self.kept.observe(seconds);
        } else {
            self.compiled
                .observe(Self::work(publication, self.threads), seconds);
        }
    }
}

/// The tuning time since `0` began.
struct Elapsed(Instant);

impl allocation::Clock for Elapsed {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
}

/// A searching unit of case type `T`.
struct CaseUnit<'a, T: EntryTuning> {
    case: T,
    key: TuningKey,
    implementation: NativeImplementation,
    statics: NativeSpecialization,
    /// The unit's points, ascending in cost.
    shapes: Vec<PointShape>,
    launches: usize,
    /// The inputs built so far, kept for the unit's later work.
    cases: UnitCases<T::Case>,
    search: Box<dyn EntrySearch<T::Entry>>,
    share: f64,
    /// The wall time of the unit's work so far, its mandatory work
    /// included.
    spent: Duration,
    /// Pieces of optional work performed.
    stages: usize,
    /// The next piece's compile, which ran ahead of it.
    formed: Option<FormedAhead>,
    /// What publishing a unit's record has taken in this load: its class,
    /// the price of a conversion's publication and of the one the unit
    /// owes when the optional work stops.
    publishing: Rc<Cell<seismic::Observed>>,
    /// What publishing a serving artifact has taken in this load.
    artifacts: Rc<Cell<ArtifactPrice>>,
    /// The unit's record changed since it was published: a piece measured
    /// or judged something and promoted nothing.
    unpublished: bool,
    /// The unit's served choice was promoted since its serving artifact
    /// was published: the artifact a load requests of that choice is not
    /// yet formed. The canonical default's is, by the mandatory phase.
    unformed: bool,
    /// The record as it was published when the optional work stopped.
    settled: Option<TuningResult>,
    marker: std::marker::PhantomData<&'a ()>,
}

/// The failure of tuning the unit `key` with `error`.
fn unit_failure(key: &TuningKey, error: impl ToString) -> CatalogFailure {
    CatalogFailure::Tuning {
        entry: key.0,
        bindings: key.1.clone(),
        outcome: error.to_string(),
    }
}

impl<'a, T: EntryTuning> SearchedUnit<'a> for CaseUnit<'a, T> {
    fn key(&self) -> &TuningKey {
        &self.key
    }

    fn time(&self) -> (usize, Vec<PointShape>, Vec<PointShape>, Vec<f64>) {
        let measured = self.search.service_measurements();
        let (shapes, seconds) = self
            .shapes
            .iter()
            .filter_map(|shape| {
                measured
                    .iter()
                    .find(|point| point.point == shape.label)
                    .map(|point| (shape.clone(), point.median_seconds))
            })
            .unzip();
        (self.launches, self.shapes.clone(), shapes, seconds)
    }

    fn share(&self) -> f64 {
        self.share
    }

    fn set_share(&mut self, share: f64) {
        self.share = share;
    }

    /// The runtime prices the piece. A conversion that promotes publishes
    /// the unit's record with its new choice: the adapter's one more
    /// operation, of a class this load observed when it published the
    /// unit's canonical default.
    fn next(&self) -> Option<seismic::Work> {
        let mut work = self.search.next()?;
        if matches!(work.stage, seismic::Stage::Promote | seismic::Stage::Confirm) {
            work.quote.add(self.publishing.get().price(), 1);
        }
        Some(work)
    }

    /// A unit whose record changed since it was published owes its
    /// publication when the optional work stops, and one whose served
    /// choice changed owes that choice's serving artifact, priced by what
    /// its publication does: nothing where the store keeps the choice's
    /// programs as a load requests them. A piece changes the served choice
    /// only by promoting, so only a conversion adds an artifact to what
    /// the unit owes once it has worked.
    fn owed(&self, worked: bool) -> seismic::Quote {
        let mut owed = seismic::Quote::default();
        if self.unformed || worked {
            // The artifact's publication is priced by the publications of
            // its class this load has run, and by the unit's stand-in for
            // one until it has run any.
            let publication = self.search.artifact_publication(worked);
            match self.artifacts.get().price(publication) {
                Some(seconds) => owed.add(Some(seconds), 1),
                None => owed = self.search.artifact_owed(worked),
            }
        }
        if self.unpublished || self.unformed || worked {
            owed.add(self.publishing.get().price(), 1);
        }
        owed
    }

    /// The optional work has stopped, so the unit's served choice is
    /// final: its serving artifact is formed as a load will request it and
    /// released, once, and never for a choice that was superseded; then
    /// the unit's record is published as it stands.
    fn settle(&mut self, tuner: &mut Tuner<'a>) -> Result<(), CatalogFailure> {
        if !self.unpublished && !self.unformed {
            return Ok(());
        }
        let began = Instant::now();
        let began_seconds = tuner.started.elapsed().as_secs_f64();
        let owed = self.owed(false);
        if self.unformed {
            // The artifact's programs compile at once on the preparation's
            // threads, on as many as the host's room affords.
            let publication = self.search.artifact_publication(false);
            let mut artifacts = self.artifacts.get();
            let _narrowed = tuner.narrow_publication();
            // The threads the publication runs on: read after the
            // narrowing, which returns freed pages before it counts when
            // the room affords none, and held as it holds them, to the
            // caller at the least and the set with the caller at the most.
            let width = tuner.workers().width() + 1;
            artifacts.threads = Some(
                tuner
                    .memory
                    .compile_threads()
                    .map_or(width, |threads| threads.clamp(1, width)),
            );
            let took = self
                .search
                .publish_artifact()
                .map_err(|error| unit_failure(&self.key, error))?;
            artifacts.observe(publication, took.as_secs_f64());
            self.artifacts.set(artifacts);
            self.unformed = false;
        }
        let result = tuner
            .publish(&self.key, || self.search.terminal())
            .map_err(|error| unit_failure(&self.key, error))?;
        self.unpublished = false;
        self.settled = Some(result);
        self.spent += began.elapsed();
        tuner.context.observer.event(&TuningEvent::Published {
            entry: self.key.0,
            bindings: self.key.1.clone(),
            forecast_seconds: owed.forecast().map(|forecast| forecast.as_secs_f64()),
            began_seconds,
            actual_seconds: began.elapsed().as_secs_f64(),
        });
        Ok(())
    }

    fn forming(&self) -> Option<seismic::Forming> {
        self.search.forming()
    }

    fn formed(
        &mut self,
        formed: seismic::Formed,
        took: Duration,
        _: &mut Tuner<'a>,
    ) -> Result<(), CatalogFailure> {
        // The compile's price as the piece was started with it.
        let quote = self
            .search
            .next()
            .expect("a compile ran ahead of the unit's next piece")
            .forming;
        self.search
            .formed(formed)
            .map_err(|error| unit_failure(&self.key, error))?;
        self.spent += took;
        self.formed = Some(FormedAhead {
            priced_seconds: quote.priced.as_secs_f64(),
            unpriced_operations: quote.unpriced,
            actual_seconds: took.as_secs_f64(),
        });
        Ok(())
    }

    fn perform(&mut self, tuner: &mut Tuner<'a>) -> Result<Duration, CatalogFailure> {
        let work = self.next().expect("a unit performs the work it named");
        let began = Instant::now();
        let began_seconds = tuner.started.elapsed().as_secs_f64();
        if self.stages == 0 {
            tuner.context.observer.event(&TuningEvent::Started {
                entry: self.key.0,
                bindings: self.key.1.clone(),
                points: self.shapes.len(),
            });
        }
        self.search.plan(seismic::UnitPlan {
            settings: SEARCH_SETTINGS,
            min_sample_seconds: MIN_SAMPLE_SECONDS,
            start: self.case.search_starts(
                tuner.device,
                &self.implementation,
                &self.statics,
                tuner.limits,
            ),
        });
        let performed = self
            .search
            .perform(&mut UnitPoints {
                case: &self.case,
                shapes: &self.shapes,
                inputs: tuner.inputs(),
                cases: &mut self.cases,
            })
            .map_err(|error| unit_failure(&self.key, error))?;
        // A weight no built input binds leaves; one that a point of any
        // unit still binds stays shared.
        tuner.inputs().release_unheld();
        tuner
            .memory
            .observe(tuner.device, performed.memory.ledger_peak_bytes);
        // A promotion is published with the unit's record as it stands and
        // served from that instant. What a piece measures or judges besides
        // is the unit's in memory and published once, when the optional
        // work stops ([`SearchedUnit::settle`]): a load abandoned before
        // then keeps the choice last published.
        let publishing = Instant::now();
        if performed.promoted {
            tuner
                .publish(&self.key, || self.search.terminal())
                .map_err(|error| unit_failure(&self.key, error))?;
        }
        self.unpublished = !performed.promoted;
        // The promoted choice's serving artifact is formed when the choice
        // is final ([`SearchedUnit::settle`]): a later promotion supersedes
        // it, and a superseded choice's artifact is never formed.
        self.unformed |= performed.promoted;
        let published = publishing.elapsed();
        self.spent += began.elapsed();
        self.stages += 1;
        tuner.context.observer.event(&TuningEvent::Stage(StageRecord {
            entry: self.key.0,
            bindings: self.key.1.clone(),
            statics: self.key.2.clone(),
            share: self.share,
            stage: performed.stage,
            priced_seconds: work.quote.priced.as_secs_f64(),
            unpriced: work.quote.unpriced,
            forecast_seconds: work.quote.forecast().map(|forecast| forecast.as_secs_f64()),
            began_seconds,
            actual_seconds: began.elapsed().as_secs_f64(),
            formed: self.formed.take(),
            spent: performed.spent,
            raises: performed.raises,
            publishing_seconds: published.as_secs_f64(),
            promoted: performed.promoted,
        }));
        Ok(self.search.longest_operation().max(published))
    }

    fn holding(&self) -> u64 {
        self.search.holding()
    }

    fn built(&self) -> seismic::Quote {
        self.search.inputs_built()
    }

    /// A promoted unit whose artifact's publication is of a class this
    /// load has not run offers it: publications are priced by the load's
    /// own, and until one of a class has run by a stand-in. If a later
    /// promotion supersedes the choice, the artifact published here is not
    /// the unit's final one: it stays in the store as the formed program
    /// it is, keyed by its text and entries, until the store evicts it,
    /// and the unit owes its final choice's at the class's observed price.
    fn sample(&self) -> Option<seismic::Quote> {
        if !self.unformed {
            return None;
        }
        let publication = self.search.artifact_publication(false);
        self.artifacts
            .get()
            .price(publication)
            .is_none()
            .then(|| self.owed(false))
    }

    /// The unit's inputs leave: the search's points give theirs up, the
    /// cases built for them are dropped, and the weights and shared planes
    /// nothing else binds leave with them.
    fn release(&mut self, tuner: &mut Tuner<'a>) {
        self.search.release(&mut UnitPoints {
            case: &self.case,
            shapes: &self.shapes,
            inputs: tuner.inputs(),
            cases: &mut self.cases,
        });
    }

    fn programs(&self) -> u64 {
        self.search.programs()
    }

    fn give_up(&mut self) {
        self.search.give_up_programs();
    }

    fn narrow(&mut self) -> bool {
        self.search.narrow()
    }

    /// What stopped the unit's search is the memory room: the tuner is
    /// told, and the unit owes its record once more, so that the record
    /// published when the optional work stops says so.
    fn refused_for_bytes(&mut self, tuner: &mut Tuner<'a>) {
        tuner.without_room.insert(self.key.clone());
        self.unpublished = true;
    }

    fn largest_footprint(&self) -> u64 {
        self.search.largest_footprint()
    }

    fn finish(&mut self, tuner: &mut Tuner<'a>) -> Result<(), CatalogFailure> {
        // The record published when the optional work stopped; for a unit
        // that owed no publication then, its record as it stands.
        let result = match self.settled.take() {
            Some(result) => result,
            None => self
                .search
                .terminal()
                .map_err(|error| unit_failure(&self.key, error))?,
        };
        let mut tuned = searched_entry(self.key.0, self.key.1.clone(), result, Instant::now());
        if tuner.without_room.contains(&self.key) {
            tuned.stop = Some(SearchStop::NoRoom);
        }
        tuned.seconds = self.spent.as_secs_f64();
        tuned.progress = Some(SearchProgress {
            share: self.share,
            stages: self.stages,
            standing: self.search.standing(),
        });
        tuner.finish(self.key.clone(), tuned);
        Ok(())
    }
}

/// One tuning unit's step time as its census measured it: its launches per
/// step, the points it serves, its timed points and its defaults' time at
/// each.
#[derive(Clone)]
struct UnitTime<'u> {
    invocations: crate::programs::invocation_demand::InvocationCounts,
    served: &'u [PointShape],
    shapes: &'u [PointShape],
    seconds: &'u [f64],
}

/// Each unit's share of the served workload's time ([`workload`]): what the
/// workload's launches of every row class the unit serves take at the
/// defaults, at their mean history, against what every unit's take. A
/// unit's time at a class is the defaults' time the census measured there,
/// launches per step times it; for attention, at the class's mean history,
/// interpolated linearly in a point's cost (rows times rows plus history)
/// between the histories timed, or scaled from the one timed. A class the
/// census did not time takes the costliest timed point of its group
/// ([`census_group`]), scaled by the classes' cost. The defaults' time is
/// also what tuning can recover: a unit far from its best spends more of
/// the workload and gets more of the time.
fn step_shares(units: &[UnitTime<'_>], _limits: TuningLimits) -> Vec<f64> {
    let cost = |rows: u64| (rows as f64).max(workload::STREAMING_EQUIVALENT_ROWS);
    let time = |unit: &UnitTime<'_>, rows: u64| {
        if !unit.served.iter().any(|shape| shape.rows == rows) {
            return 0.0;
        }
        // Use the same conditional history demand as the search objective.
        // Zero-weight bootstrap witnesses establish numerical support only.
        let (history_sum, history_weight) = unit
            .served
            .iter()
            .filter(|shape| shape.rows == rows && shape.weight > 0.)
            .filter_map(|shape| {
                shape
                    .context
                    .map(|history| (history as f64 * shape.weight, shape.weight))
            })
            .fold((0., 0.), |(sum, weight), (next, demand)| {
                (sum + next, weight + demand)
            });
        let history = if history_weight > 0. {
            history_sum / history_weight
        } else {
            0.
        };
        let timed = |at: u64| {
            unit.shapes
                .iter()
                .zip(unit.seconds)
                .filter(|(shape, _)| shape.rows == at)
                .map(|(shape, seconds)| (shape.clone(), *seconds))
                .collect::<Vec<_>>()
        };
        let (at, class) = match timed(rows) {
            class if !class.is_empty() => (rows, class),
            _ => match unit
                .shapes
                .iter()
                .filter(|shape| census_group(shape.rows) == census_group(rows))
                .map(|shape| shape.rows)
                .max()
            {
                Some(stands) => (stands, timed(stands)),
                None => return 0.0,
            },
        };
        let seconds = at_history(&class, history) * cost(rows) / cost(at);
        unit.invocations.count(rows) * seconds
    };
    let totals = units
        .iter()
        .map(|unit| {
            TUNING_ROWS
                .iter()
                .map(|rows| time(unit, *rows))
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let total = totals.iter().sum::<f64>();
    totals
        .into_iter()
        .map(|time| if total > 0.0 { time / total } else { 0.0 })
        .collect()
}

/// The defaults' time at `history` from the points of one row class timed:
/// their mean for a class without history; for attention, linear in a
/// point's cost between the two timed histories around it (or the two
/// nearest), or scaled from the one timed.
fn at_history(class: &[(PointShape, f64)], history: f64) -> f64 {
    let costed = |shape: &PointShape, history: f64| {
        let rows = shape.rows as f64;
        rows * (rows + history)
    };
    let mut timed = class
        .iter()
        .filter_map(|(shape, seconds)| {
            shape
                .context
                .map(|context| (context as f64, shape, *seconds))
        })
        .collect::<Vec<_>>();
    if timed.is_empty() {
        return class.iter().map(|(_, seconds)| seconds).sum::<f64>() / class.len() as f64;
    }
    timed.sort_by(|left, right| left.0.total_cmp(&right.0));
    if timed.len() == 1 {
        let (context, shape, seconds) = timed[0];
        return seconds * costed(shape, history) / costed(shape, context);
    }
    let upper = timed
        .iter()
        .position(|(context, ..)| *context >= history)
        .unwrap_or(timed.len() - 1)
        .max(1);
    let (low, shape, low_seconds) = timed[upper - 1];
    let (high, _, high_seconds) = timed[upper];
    let (from, to) = (costed(shape, low), costed(shape, high));
    if to <= from {
        return low_seconds;
    }
    let at = costed(shape, history);
    (low_seconds + (high_seconds - low_seconds) * (at - from) / (to - from)).max(0.0)
}

/// One unit's built cases by point, kept from its census for its search,
/// and the point whose states are observed whole: the first built with
/// states, the cheapest, so the guard costs a short history.
struct UnitCases<C> {
    points: Vec<Option<Vec<C>>>,
    guard: Option<usize>,
}

/// A counted unit can form its canonical artifact without constructing inputs.
/// Type erasure keeps the prerequisite phase independent of another model walk;
/// the census recovers the original entry-specific search and cases afterward.
trait CanonicalUnit: std::any::Any {
    /// The compile of the unit's mandatory work, its canonical artifact,
    /// to run on any thread: units' canonical artifacts form concurrently.
    fn grounding(&mut self) -> Result<Option<seismic::Grounding>, TuneError>;
    /// The end of the unit's mandatory work: take what the compile formed
    /// and hand the canonical artifact to serving. The unit's record is
    /// legal from here on. Returns the longest operation it took.
    fn grounded(&mut self, grounded: Option<seismic::Grounded>) -> Result<Duration, TuneError>;
    /// The unit's record as it stands.
    fn record(&self) -> Result<TuningResult, TuneError>;
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any>;
}

struct PreparedUnit<T: EntryTuning> {
    case: T,
    shapes: Vec<PointShape>,
    search: Box<dyn EntrySearch<T::Entry>>,
    cases: UnitCases<T::Case>,
    /// What the unit's mandatory work took.
    mandatory: Duration,
}

impl<T: EntryTuning> CanonicalUnit for PreparedUnit<T> {
    fn grounding(&mut self) -> Result<Option<seismic::Grounding>, TuneError> {
        self.search.grounding(&mut CanonicalPoints {
            case: &self.case,
            shapes: &self.shapes,
        })
    }

    fn grounded(&mut self, grounded: Option<seismic::Grounded>) -> Result<Duration, TuneError> {
        let began = Instant::now();
        let formed = grounded.as_ref().map_or(Duration::ZERO, seismic::Grounded::elapsed);
        self.search.grounded(grounded)?;
        self.mandatory = formed + began.elapsed();
        Ok(self.search.longest_operation())
    }

    fn record(&self) -> Result<TuningResult, TuneError> {
        self.search.terminal()
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

/// What one native implementation on a device is a function of: its entry,
/// the entry's element bindings and its static values.
type ImplementationKey = (&'static str, Vec<(&'static str, Element)>, NativeSpecialization);

/// A unit's slot in the kernel cache: the cache, the unit's key, and the
/// valid choice stored under it with its evidence.
type StoredSlot<'c> = (&'c KernelCache, TuningCacheKey, Option<(StoredChoice, Receipt)>);

/// Resolves every native entry's specialization for the opened device:
/// static values, tuned parameters, and missing implementations. A first
/// load's tuning has a mandatory phase and an optional one. The mandatory
/// phase forms every searching unit's canonical artifact, hands it to
/// serving and publishes the unit's record with it: from then on every unit
/// serves a legal published choice, and a failure there is the only
/// preparation failure of tuning. The optional phase spends what is left of
/// the tuning budget by value ([`allocation`]): it starts only work whose
/// whole price fits what remains, runs it to its end, publishes a unit's
/// record when its served choice changes, and stops when nothing that fits
/// is left, publishing once more each record that changed since.
pub(crate) struct Tuner<'a> {
    device: &'a Device,
    context: TuningContext<'a>,
    limits: TuningLimits,
    weights: TuningWeights<'a>,
    invocations: super::specialization::InvocationDemand,
    noise: Noise,
    /// Tensors shared by the units of this tuning (history planes).
    shared: HashMap<String, Tensor>,
    phase: Phase<'a>,
    /// When counting began: the tuning clock. Metadata, canonical
    /// formation and all optional work are measured on it; the budget is
    /// counted from it. Counting forms nothing.
    started: Instant,
    /// This preparation has uncached units: it reports what became of its
    /// budget.
    first_load_search: bool,
    /// Units a load found no stored record for and prepared at their
    /// canonical defaults.
    unprepared: usize,
    /// Units a development pin decided: what depends on one (a composed
    /// recipe) is decided with it, canonically.
    pinned: std::collections::HashSet<TuningKey>,
    tuned: Vec<TunedEntry>,
    /// Every unit's served choice as it stands: its canonical default from
    /// the mandatory phase, replaced when a challenger is promoted.
    chosen: HashMap<TuningKey, NativeSpecialization>,
    /// The digest of each native implementation a unit's key was formed
    /// with, by its entry, element bindings and static values
    /// ([`Self::implementation_digest`]).
    digests: std::cell::RefCell<HashMap<ImplementationKey, String>>,
    composed_counted: std::collections::HashSet<String>,
    composed_censused: std::collections::HashSet<String>,
    composed_choices: HashMap<String, bool>,
    /// What publishing a unit's record (deriving it and storing it) has
    /// taken in this load: the class a promotion's publication and each
    /// publication owed when the optional work stops are priced by.
    publishing: Rc<Cell<seismic::Observed>>,
    /// What publishing a unit's serving artifact has taken in this load:
    /// the class each artifact owed is priced by once one was published.
    artifacts: Rc<Cell<ArtifactPrice>>,
    /// The longest single operation the load has completed: what an
    /// unpriced operation is held to ([`seismic::Quote::fits`]).
    longest: Duration,
    /// What the load's tuning has held on the device, for its report.
    memory: TuningMemory,
    /// The units whose next piece was refused for its bytes when the
    /// optional work stopped: their records state that stop
    /// ([`SearchStop::NoRoom`]).
    without_room: std::collections::HashSet<TuningKey>,
    /// What the platform lets the engine claim: the ceiling of what the
    /// search holds. None for a tuner given no room, which bounds nothing.
    room: Option<TuningRoom>,
    /// Where each uncached unit's record is published.
    slots: HashMap<TuningKey, TuningCacheKey>,
    prepared: HashMap<TuningKey, Box<dyn CanonicalUnit>>,
    /// The threads everything of this preparation runs on off the device's
    /// thread.
    /// None once the search has ended: the walk that follows it compiles
    /// nothing a stored record does not serve, and holds no thread.
    staff: Option<Staff<'a>>,
}

/// One preparation's worker threads, lent to its device for as long as the
/// preparation's tuner lives: the canonical compiles of the mandatory
/// phase, the optional work's compiles and every source the device forms
/// run on these threads and no others. The tuner owns the concurrency of
/// the load's formation; the device and its toolchain start no thread per
/// compile. The threads are joined when the tuner is done.
struct Staff<'a> {
    device: &'a Device,
    workers: Arc<seismic::Workers>,
}

impl<'a> Staff<'a> {
    fn of(device: &'a Device) -> Self {
        let workers = Arc::new(seismic::Workers::of_the_host());
        device.staff(Some(workers.clone()));
        Self { device, workers }
    }
}

impl Drop for Staff<'_> {
    fn drop(&mut self) {
        self.device.staff(None);
    }
}

impl<'a> Tuner<'a> {
    pub(super) fn choice(&self, key: &TuningKey) -> Option<&NativeSpecialization> {
        self.chosen.get(key)
    }
    pub(super) fn cache(&self) -> Option<&'a KernelCache> {
        self.context.cache
    }
    pub(super) fn device(&self) -> &'a Device {
        self.device
    }
    pub(super) fn settings(&self) -> TuningSettings {
        self.context.settings
    }
    fn budget(&self) -> Duration {
        self.context.settings.budget
    }
    /// Hold what the preparation compiles at once to the host's room, for
    /// as long as what is returned lives: a unit's serving artifact is
    /// owed work whose programs compile together, outside the pieces the
    /// allocator admits, and each compile holds host memory while it runs.
    /// When the room affords no thread, what is held for nothing is
    /// returned first; owed work then runs in turn if it must. Nothing
    /// where the host is not a domain apart.
    fn narrow_publication(&mut self) -> Option<seismic::Narrowed> {
        let mut threads = self.memory.compile_threads()?;
        if threads == 0 {
            self.memory.return_freed_pages();
            threads = self.memory.compile_threads()?;
        }
        Some(self.workers().narrowed(threads))
    }

    /// The stop of the optional work as a stop for bytes: none unless a
    /// piece still offered when the allocator stopped was refused for its
    /// bytes. The allocator refuses for bytes only with nothing in flight,
    /// the pages tuning freed returned, and nothing left that another unit
    /// could give up for the piece; each bounded domain is stated as it
    /// stands now that the allocator has stopped, with the least a refused
    /// piece was taken to add to it.
    fn no_room(
        &self,
        units: &[Box<dyn SearchedUnit<'a> + 'a>],
        report: &allocation::Report,
    ) -> Option<NoRoom> {
        use allocation::Unit as _;
        let refused = report
            .refused
            .iter()
            .filter(|refused| refused.refusal == allocation::Refusal::Bytes)
            .collect::<Vec<_>>();
        if refused.is_empty() {
            return None;
        }
        let room = |domain, ceiling: Option<u64>, held: Option<u64>, smallest: Option<u64>| {
            let ((ceiling, held), smallest) = ceiling.zip(held).zip(smallest)?;
            Some(Room {
                domain,
                ceiling_bytes: ceiling,
                held_bytes: held,
                smallest_footprint_bytes: smallest,
                lacking_bytes: held.saturating_add(smallest).saturating_sub(ceiling),
            })
        };
        // What a refused piece is taken to add where the device allocates:
        // its footprint, an unpriced part at the most a piece of its unit
        // has added.
        let charged = refused
            .iter()
            .map(|refused| {
                refused
                    .footprint
                    .forecast(units[refused.unit].largest_footprint())
            })
            .min();
        // And on the host: the most a piece of its unit has added, with
        // what its compile holds on each of its threads.
        let host = refused
            .iter()
            .map(|refused| {
                let unit = &units[refused.unit];
                let compile = unit.forming().map_or(0, |forming| {
                    self.memory
                        .compile_host_bytes
                        .saturating_mul(seismic::Forming::threads(&forming) as u64)
                });
                unit.host_footprint(self).unwrap_or(0).saturating_add(compile)
            })
            .min();
        let domains = [
            room(
                "device",
                self.memory.ceiling_bytes,
                Some(self.memory.held(self.device)),
                charged,
            ),
            room(
                "footprint",
                self.memory
                    .footprint_ceiling_bytes
                    .filter(|_| !self.memory.device_separate),
                self.memory.held_whole(self.device),
                charged,
            ),
            room(
                "host",
                self.memory.host_ceiling_bytes,
                self.memory.host_grown(),
                host,
            ),
        ];
        Some(NoRoom {
            domains: domains.into_iter().flatten().collect(),
            unplanned_holders: crate::planning::HostPeak::UNPLANNED,
        })
    }

    /// The preparation's worker threads, which it has until its search
    /// ends.
    fn workers(&self) -> Arc<seismic::Workers> {
        self.staff
            .as_ref()
            .expect("the mandatory phase and the search run before the tuner's threads are joined")
            .workers
            .clone()
    }

    /// A tuner that counts the units of a walk: the first phase of a
    /// search.
    pub fn count(
        device: &'a Device,
        context: TuningContext<'a>,
        limits: TuningLimits,
        weights: TuningWeights<'a>,
        invocations: super::specialization::InvocationDemand,
    ) -> Self {
        Self::in_phase(
            device,
            context,
            limits,
            weights,
            invocations,
            Phase::Count {
                launches: HashMap::new(),
                searching: Vec::new(),
            },
        )
    }

    /// A load's tuner: its one walk reads the stored records and prepares
    /// their choices. `unprepared` is what it does at a unit without a
    /// record. Under fresh tuning ([`TuningSettings::fresh`]) a load that
    /// reports reads no stored record, so every unit is one without: the
    /// search its host then runs replaces them all. The load that serves
    /// defaults reads whatever is stored: it comes after that search.
    pub fn serve(
        device: &'a Device,
        context: TuningContext<'a>,
        limits: TuningLimits,
        weights: TuningWeights<'a>,
        invocations: super::specialization::InvocationDemand,
        unprepared: UnpreparedUnits,
    ) -> Self {
        Self::in_phase(
            device,
            TuningContext {
                settings: TuningSettings {
                    fresh: context.settings.fresh && unprepared == UnpreparedUnits::Report,
                    ..context.settings
                },
                ..context
            },
            limits,
            weights,
            invocations,
            Phase::Serve(unprepared),
        )
    }

    fn in_phase(
        device: &'a Device,
        context: TuningContext<'a>,
        limits: TuningLimits,
        weights: TuningWeights<'a>,
        invocations: super::specialization::InvocationDemand,
        phase: Phase<'a>,
    ) -> Self {
        Self {
            device,
            context,
            limits,
            weights,
            invocations,
            noise: Noise::default(),
            shared: HashMap::new(),
            unprepared: 0,
            pinned: std::collections::HashSet::new(),
            started: Instant::now(),
            first_load_search: false,
            tuned: Vec::new(),
            chosen: HashMap::new(),
            digests: std::cell::RefCell::new(HashMap::new()),
            composed_counted: std::collections::HashSet::new(),
            composed_censused: std::collections::HashSet::new(),
            composed_choices: HashMap::new(),
            publishing: Rc::new(Cell::new(seismic::Observed::default())),
            artifacts: Rc::new(Cell::new(ArtifactPrice::default())),
            longest: Duration::ZERO,
            // What a search's process holds is counted from here, where
            // its tuner is made, not from its mandatory phase: counting
            // the units and opening their searches grows the process too
            // (116 MB before the mandatory phase of one Gemma 4 E2B search
            // on CUDA), and the room it is held to was observed at the
            // same place, its device just opened.
            memory: match &phase {
                Phase::Serve(_) => TuningMemory::default(),
                Phase::Count { .. } | Phase::Census { .. } | Phase::Searched => {
                    let began = seismic::ProcessMemory::read();
                    TuningMemory {
                        process_began_bytes: began.footprint_bytes,
                        breakdown_began: began,
                        breakdown_peak: began,
                        ..TuningMemory::default()
                    }
                }
            },
            without_room: std::collections::HashSet::new(),
            room: None,
            slots: HashMap::new(),
            prepared: HashMap::new(),
            // A search's compiles run on its own threads. A load's walk
            // forms its chosen kernels as it always has, after a search's
            // threads were joined: on none.
            staff: match phase {
                Phase::Serve(_) => None,
                Phase::Count { .. } | Phase::Census { .. } | Phase::Searched => {
                    Some(Staff::of(device))
                }
            },
            phase,
        }
    }

    /// Units this load found no stored record for and prepared at their
    /// canonical defaults.
    pub fn unprepared(&self) -> usize {
        self.unprepared
    }

    /// Hold the search to `room`.
    pub fn held_to(mut self, room: Option<TuningRoom>) -> Self {
        self.room = room;
        self
    }

    /// Publish a unit's record as it stands: derive it (`record`), store it
    /// where a later load reads it, and serve its choice. The whole is one
    /// observation of the publication class. Returns the record.
    fn publish(
        &mut self,
        key: &TuningKey,
        record: impl FnOnce() -> Result<TuningResult, TuneError>,
    ) -> Result<TuningResult, TuneError> {
        let began = Instant::now();
        let result = record()?;
        if let (Some(cache), Some(slot)) = (self.context.cache, self.slots.get(key)) {
            let choice = StoredChoice::of(&result);
            let choice = if self.without_room.contains(key) {
                choice.without_room()
            } else {
                choice
            };
            cache.store_tuning(slot, &choice, &result);
        }
        self.chosen
            .insert(key.clone(), result.overall.specialization());
        let elapsed = began.elapsed();
        let mut publishing = self.publishing.get();
        publishing.observe(elapsed.as_secs_f64());
        self.publishing.set(publishing);
        self.longest = self.longest.max(elapsed);
        Ok(result)
    }

    /// The mandatory phase: form every searching unit's canonical
    /// artifact, hand it to serving and publish the unit's record with it,
    /// before any input is built or anything optional is started. From
    /// here every unit serves a legal published choice; a failure here is
    /// the only preparation failure of tuning.
    pub fn census(mut self) -> Result<Self, CatalogFailure> {
        let Phase::Count {
            launches,
            searching,
        } = std::mem::replace(&mut self.phase, Phase::Searched)
        else {
            unreachable!("a census follows the count");
        };
        self.memory.began_bytes = self.device.memory_usage().charged;
        self.memory.device_separate = device_separate(self.device);
        // The search holds no more than the platform lets the engine claim
        // ([`TuningRoom`]). Where the device's memory is the host's, its
        // charges and the process's footprint are one domain under one
        // ceiling; where it is apart, host RAM is a second domain with the
        // platform's own room.
        if let Some(room) = self.room {
            self.memory.ceiling_bytes = Some(room.device_bytes);
            if self.memory.device_separate {
                self.memory.host_ceiling_bytes = room.host_bytes;
            } else {
                self.memory.footprint_ceiling_bytes = Some(room.device_bytes);
            }
        }
        // The units' canonical programs compile together on the load's
        // worker threads: a compile needs nothing of another unit or of
        // the device's queue.
        let began = self.started.elapsed();
        let groundings = searching
            .iter()
            .map(|key| {
                self.prepared
                    .get_mut(key)
                    .expect("counted unit has its opened search")
                    .grounding()
                    .map(std::sync::Mutex::new)
                    .map_err(|error| unit_failure(key, error))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let workers = self.workers();
        let ground = |grounding: &std::sync::Mutex<Option<seismic::Grounding>>| {
            grounding
                .lock()
                .expect("a grounding is taken once")
                .take()
                .map(seismic::Grounding::run)
        };
        // What the peak of the process's footprint rose by over compiles
        // that ran on `threads` threads with nothing else running, for
        // each thread: what one thread's compile holds. A span that stayed
        // under the peak already kept says nothing.
        let rose = |before: Option<u64>, peak: Option<u64>, threads: usize| {
            before
                .zip(peak)
                .zip(seismic::ProcessMemory::footprint_peak())
                .filter(|((_, peak), now)| now > peak)
                .map(|((before, _), now)| now.saturating_sub(before) / threads as u64)
        };
        // Without a host ceiling the canonical compiles all run at once.
        // Where the host has a ceiling of its own they run in waves. A
        // compile's memory is not known before one has run, and one unit's
        // compile says little of another's (the first of one Qwen3.5-4B
        // search held 10 MB, and the twenty-two threads then run at once
        // on that price held 25 MB each and took the process 104 MB past
        // its ceiling): so a wave runs at once as many as the host's room
        // affords at the most a thread's compile has been seen to hold,
        // and never more than have been observed so far. One runs alone,
        // then as many as have run, and the price each wave is admitted on
        // is the most seen over at least as many compiles as it starts.
        // What a wave freed is returned before the next reads its room, so
        // that room is what its compiles left held, as measured.
        let mut grounded = Vec::with_capacity(groundings.len());
        if self.memory.host_ceiling_bytes.is_none() {
            grounded.extend(workers.each(&groundings, ground));
        }
        while grounded.len() < groundings.len() {
            let observed = grounded.len();
            let threads = self
                .memory
                .compile_threads()
                .map_or(1, |threads| threads.max(1))
                .min(observed.max(1))
                .min(groundings.len() - observed);
            let narrowed = workers.narrowed(threads);
            let (before, peak) = (
                seismic::ProcessMemory::footprint_now(),
                seismic::ProcessMemory::footprint_peak(),
            );
            grounded.extend(workers.each(&groundings[observed..observed + threads], ground));
            drop(narrowed);
            if let Some(held) = rose(before, peak, threads) {
                let priced = self.memory.compile_host_bytes;
                if held > priced {
                    if priced > 0 {
                        self.memory.mispriced.push(Mispricing {
                            kind: "compile".into(),
                            priced_bytes: priced,
                            added_bytes: held,
                        });
                    }
                    self.memory.compile_host_bytes = held;
                }
            }
            self.memory.return_freed_pages();
        }
        // What the optional work is admitted against is what is held, as
        // measured, and what the mandatory phase left is stated.
        if self.memory.host_ceiling_bytes.is_some() {
            self.memory.host_mandatory_bytes = self.memory.host_grown();
        }
        for (key, grounded) in searching.iter().zip(grounded) {
            let forming = Instant::now();
            let formed = grounded
                .as_ref()
                .map_or(Duration::ZERO, seismic::Grounded::elapsed);
            let mut unit = self
                .prepared
                .remove(key)
                .expect("counted unit has its opened search");
            let longest = unit
                .grounded(grounded)
                .map_err(|error| unit_failure(key, error))?;
            let forming = formed + forming.elapsed();
            self.longest = self.longest.max(longest);
            let publishing = Instant::now();
            self.publish(key, || unit.record())
                .map_err(|error| unit_failure(key, error))?;
            let publishing = publishing.elapsed();
            self.prepared.insert(key.clone(), unit);
            self.context.observer.event(&TuningEvent::Mandatory {
                entry: key.0,
                bindings: key.1.clone(),
                began_seconds: began.as_secs_f64(),
                forming_seconds: forming.as_secs_f64(),
                publishing_seconds: publishing.as_secs_f64(),
            });
        }
        let launches = searching
            .into_iter()
            .map(|key| {
                let count = launches[&key];
                (key, count)
            })
            .collect::<HashMap<_, _>>();
        self.memory.mandatory_bytes = self.device.memory_usage().charged;
        self.memory.peak_bytes = self.memory.mandatory_bytes;
        let searches = !launches.is_empty() || !self.composed_counted.is_empty();
        self.first_load_search = searches;
        self.phase = Phase::Census {
            launches,
            units: Vec::new(),
        };
        if searches {
            self.report_progress();
        }
        Ok(self)
    }

    /// The optional phase: spend what is left of the tuning budget on the
    /// units' work by value ([`allocation`]), then report every unit's
    /// record and what became of the budget. Nothing walks after it: the
    /// records are published, and a load reads them.
    pub fn search(mut self) -> Result<Self, CatalogFailure> {
        let Phase::Census { mut units, .. } = std::mem::replace(&mut self.phase, Phase::Searched)
        else {
            unreachable!("a search follows the census");
        };
        let clock = Elapsed(self.started);
        let budget = self.budget();
        // A composed recipe's census is its discovery: work of a class
        // this load has not observed, started while it is not late.
        let discovery = seismic::Quote {
            priced: Duration::ZERO,
            unpriced: 1,
        };
        for index in 0..units.len() {
            if !units[index].composed()
                || !discovery.fits(budget.saturating_sub(self.started.elapsed()), self.longest)
            {
                continue;
            }
            let began = Instant::now();
            units[index].census_optional(&mut self.inputs())?;
            self.inputs().release_unheld();
            self.longest = self.longest.max(began.elapsed());
        }
        let indices = units
            .iter()
            .enumerate()
            .map(|(index, unit)| (unit.key().clone(), index))
            .collect::<HashMap<_, _>>();
        for unit in &mut units {
            let mut prerequisites = Vec::new();
            for key in unit.dependency_keys() {
                match indices.get(&key) {
                    Some(index) => prerequisites.push(*index),
                    // A prerequisite without optional work: its choice is
                    // final already.
                    None if self.chosen.contains_key(&key) => {}
                    None => {
                        return Err(unit_failure(
                            unit.key(),
                            format!("missing canonical prerequisite {} [{}]", key.0, key.1),
                        ));
                    }
                }
            }
            unit.set_prerequisites(prerequisites);
        }
        // A finite prerequisite graph must be acyclic before any unit seals.
        let mut pending = (0..units.len()).collect::<std::collections::HashSet<_>>();
        loop {
            let ready = pending
                .iter()
                .copied()
                .filter(|index| {
                    units[*index]
                        .prerequisites()
                        .iter()
                        .all(|dependency| !pending.contains(dependency))
                })
                .collect::<Vec<_>>();
            if ready.is_empty() {
                break;
            }
            for index in ready {
                pending.remove(&index);
            }
        }
        if let Some(&index) = pending.iter().min() {
            return Err(unit_failure(
                units[index].key(),
                "cyclic canonical prerequisites",
            ));
        }
        let longest = self.longest;
        // The pieces' compiles run on the load's worker threads while the
        // device measures; each started runs to its end.
        let mut threads = allocation::Threads::new(
            self.workers(),
            seismic::Forming::run,
            seismic::Forming::threads,
            self.memory.compile_host_bytes,
        );
        let (report, outcome) =
            allocation::allocate(&mut units, &mut self, &clock, &mut threads, budget, longest);
        self.memory.compile_host_bytes = self
            .memory
            .compile_host_bytes
            .max(allocation::Workers::compile_host(&threads));
        self.memory
            .mispriced
            .extend(threads.mispriced().iter().map(|(priced, added)| Mispricing {
                kind: "compile".into(),
                priced_bytes: *priced,
                added_bytes: *added,
            }));
        drop(threads);
        self.memory.no_room = self.no_room(&units, &report);
        self.memory.ended_bytes = self.device.memory_usage().charged;
        let ended = self.memory.ended_bytes;
        self.memory.observe(self.device, ended);
        self.memory.process_peak_bytes = process_peak_bytes();
        // Every unit reports its record as the allocator left it published,
        // whatever became of the work: each has served a published legal
        // choice throughout.
        let reported = units
            .iter_mut()
            .try_for_each(|unit| unit.finish(&mut self));
        let refused = report
            .refused
            .iter()
            .map(|refused| {
                let key = units[refused.unit].key();
                RefusedOffer {
                    unit: format!("{} [{}]", key.0, key.1),
                    stage: refused.stage,
                    priced_seconds: refused.quote.priced.as_secs_f64(),
                    unpriced_operations: refused.quote.unpriced,
                    footprint_bytes: refused.footprint.priced,
                    unpriced_footprints: refused.footprint.unpriced,
                    reason: match refused.refusal {
                        allocation::Refusal::Seconds => "seconds",
                        allocation::Refusal::Bytes => "bytes",
                    },
                }
            })
            .collect();
        // Tuning is over: what its units still hold is released and the
        // pages it freed are returned, and the report that follows states
        // the process's memory before and after.
        self.memory.breakdown_ended = seismic::ProcessMemory::read();
        drop(units);
        // What the tuner itself held for its units' work goes with them:
        // the opened search of a unit no walk reached, the weights and
        // shared planes no input binds any more, and the activation pools
        // the inputs were windows of.
        self.prepared.clear();
        self.inputs().release_unheld();
        self.noise = Noise::default();
        // Its threads go too, joined: what a thread freed is the
        // allocator's to keep for that thread while it lives.
        self.staff = None;
        self.memory.return_freed_pages();
        self.memory.breakdown_returned = seismic::ProcessMemory::read();
        if self.first_load_search {
            self.context
                .observer
                .event(&TuningEvent::Budget(BudgetReport {
                    budget_seconds: budget.as_secs_f64(),
                    used_seconds: report.ended.as_secs_f64(),
                    settling_seconds: report.settling.as_secs_f64(),
                    unused_seconds: report.unused().as_secs_f64(),
                    overrun_seconds: report.overrun().as_secs_f64(),
                    stages: report.admitted.len(),
                    refused,
                    ending: EndingReport {
                        began_seconds: report.ending.began.as_secs_f64(),
                        held_seconds: report.ending.held.as_secs_f64(),
                        passes: report.ending.passes,
                        deciding_seconds: report.ending.deciding.as_secs_f64(),
                        narrowed: report.ending.narrowed,
                        narrowing_seconds: report.ending.narrowing.as_secs_f64(),
                        released: report.ending.released,
                        releasing_seconds: report.ending.releasing.as_secs_f64(),
                    },
                    memory: self.memory.clone(),
                }));
        }
        outcome.and(reported)?;
        Ok(self)
    }

    /// Register the same contender in each walk. Cache hits never census or
    /// qualify; misses join ordinary allocation and conclude once this load.
    pub(super) fn register_composed(
        &mut self,
        mut contender: Box<dyn ComposedContender<'a> + 'a>,
    ) -> Result<Option<bool>, CatalogFailure> {
        let key = contender.key();
        let demand_key = (contender.label(), key.clone(), BTreeMap::new());
        let demand = self
            .invocations
            .get(&demand_key)
            .ok_or_else(|| unit_failure(&demand_key, "graph invocation demand is missing"))?;
        contender.set_invocations(demand);
        if let Some(selected) = self
            .composed_choices
            .get(&key)
            .copied()
            .or_else(|| contender.cached())
        {
            self.composed_choices.insert(key, selected);
            return Ok(Some(selected));
        }
        let demanded = demand.iter().any(|(_, count)| count > 0.);
        match &self.phase {
            Phase::Count { .. } => {
                self.composed_counted.insert(key);
                Ok(None)
            }
            Phase::Census { .. } => {
                if !self.composed_censused.insert(key.clone()) {
                    return Ok(None);
                }
                let label = contender.label();
                // The recipe's mandatory work: publish the canonical choice
                // it serves until it is qualified.
                contender.conclude(false)?;
                let Phase::Census { units, .. } = &mut self.phase else {
                    unreachable!()
                };
                units.push(Box::new(ComposedUnit {
                    key: (label, key, BTreeMap::new()),
                    contender,
                    census: None,
                    share: 0.,
                    qualified: None,
                    prerequisites: Vec::new(),
                    demanded,
                }));
                Ok(None)
            }
            Phase::Searched => Err(unit_failure(
                &demand_key,
                "composed choice was not finalized inside the tuning budget",
            )),
            // A load qualifies nothing: a recipe without a stored choice
            // serves its canonical one. A recipe over pinned units is
            // decided by the pin; any other is a unit without a record.
            Phase::Serve(_) => {
                let pinned = contender
                    .dependencies()
                    .iter()
                    .all(|dependency| self.pinned.contains(dependency));
                if !pinned {
                    self.unprepared_unit(contender.label(), key.clone(), BTreeMap::new())?;
                }
                self.composed_choices.insert(key, false);
                Ok(Some(false))
            }
        }
    }

    /// What a case builds its points' inputs from.
    fn inputs(&mut self) -> TuningInputs<'_, 'a> {
        TuningInputs {
            model: self.model(),
            device: self.device,
            weights: &mut self.weights,
            noise: &self.noise,
            shared: &mut self.shared,
            choices: &self.chosen,
        }
    }

    /// Report the tuning time spent so far, of the tuning budget.
    fn report_progress(&self) {
        let total = self.budget().as_millis() as usize;
        self.context.observer.event(&TuningEvent::Progress {
            completed: (self.started.elapsed().as_millis() as usize).min(total),
            total,
        });
    }

    /// The walk or the search is over: its units' outcomes, with nothing
    /// else of the tuner left in the process. A search ends its process, so
    /// what it freed needs no returning here; a load's walk searched
    /// nothing.
    pub fn conclude(mut self) -> Vec<TunedEntry> {
        std::mem::take(&mut self.tuned)
    }

    /// The unit's tuning points, ascending in cost.
    fn shapes<T: EntryTuning>(
        &self,
        case: &T,
        key: &TuningKey,
    ) -> Result<Vec<PointShape>, CatalogFailure> {
        let mut shapes = case.points(self.limits);
        let supplied = self
            .invocations
            .get(key)
            .ok_or_else(|| unit_failure(key, "graph invocation demand is missing"))?;
        apply_invocation_counts(&mut shapes, supplied);
        shapes.sort_by(|left, right| left.cost().total_cmp(&right.cost()));
        Ok(shapes)
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
        let shapes = self.shapes(case, &key)?;
        if shapes.is_empty() {
            return Err(failure("the engine's bounds admit no tuning point".into()));
        }
        match self.phase {
            Phase::Count { .. } => self.count_unit(case, implementation, statics, key, &shapes),
            Phase::Census { .. } => self.census_unit(case, implementation, statics, key, shapes),
            // The search is over: the unit's published choice is read, and
            // nothing is prepared.
            Phase::Searched => self.searched_unit(case, statics, key, shapes).map(Some),
            Phase::Serve(_) => self
                .served_unit(case, implementation, statics, key, shapes, &defaults)
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
        // A pinned unit's choice is final here, as a stored one is: it is
        // the unit's choice from the count on, so whatever depends on the
        // unit (a composed recipe's qualification) finds it in every phase.
        #[cfg(feature = "pinned-tuning")]
        if let pinned::Pinned::Chosen(chosen, choice) =
            pinned::lookup(&key, implementation).map_err(failure)?
        {
            self.context.observer.event(&TuningEvent::Pinned {
                entry: key.0,
                bindings: key.1.clone(),
                statics: key.2.clone(),
                choice,
                configuration: Configuration::of(&chosen),
            });
            self.chosen.insert(key, chosen);
            return Ok(None);
        }
        #[cfg(not(feature = "pinned-tuning"))]
        let _ = implementation;
        // A valid cached final choice is never searched again. The first
        // preparation alone owns the fixed tuning allowance.
        let began = Instant::now();
        let cached = self.stored(case, statics, shapes).map_err(failure)?;
        let slot = cached.as_ref().map(|(_, slot, _)| slot.clone());
        let settled = if let Some((_, _, Some((choice, receipt)))) = cached {
            // A later load is metadata only: the walk that prepares the
            // choice forms what it serves, once. A stored fallback is the
            // declared default or a violated contract.
            if matches!(
                choice.disposition,
                seismic::TuningDisposition::CanonicalFallback { .. }
            ) && implementation
                .default_specialization(statics)
                .map_or(true, |defaults| choice.overall.specialization() != defaults)
            {
                return Err(failure(
                    "terminal fallback: choice is not the declared canonical default; search cannot reopen"
                        .into(),
                ));
            }
            let tuned = tuned_entry(
                key.0,
                key.1.clone(),
                choice,
                receipt,
                TuningOrigin::Stored,
                TuningTime::default(),
                began,
            );
            self.finish(key.clone(), tuned);
            true
        } else {
            // Counting opens searches without formation. The mandatory
            // phase forms every canonical artifact together.
            let requirement = self.context.numerical_profile.requirement(&case.entry());
            let search = case
                .open(self.device, statics, requirement)
                .map_err(|error| failure(error.to_string()))?;
            let cases: UnitCases<T::Case> = UnitCases {
                points: shapes.iter().map(|_| None).collect(),
                guard: None,
            };
            self.prepared.insert(
                key.clone(),
                Box::new(PreparedUnit {
                    case: case.clone(),
                    shapes: shapes.to_vec(),
                    search,
                    cases,
                    mandatory: Duration::ZERO,
                }),
            );
            if let Some(slot) = slot {
                self.slots.insert(key.clone(), slot);
            }
            false
        };
        // A surveyed unit is surveyed by the walk after the search.
        if !settled && survey_plan(key.0).is_none() {
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
        let Phase::Census { launches, units } = &self.phase else {
            unreachable!("the phase is the census");
        };
        let Some(&count) = launches.get(&key) else {
            return Ok(None);
        };
        if units.iter().any(|unit| *unit.key() == key) {
            return Ok(None);
        }
        let prepared = self
            .prepared
            .remove(&key)
            .ok_or_else(|| failure("canonical preparation missing".into()))?;
        let PreparedUnit {
            search,
            cases,
            mandatory,
            ..
        } = *prepared
            .into_any()
            .downcast::<PreparedUnit<T>>()
            .map_err(|_| failure("canonical preparation case mismatch".into()))?;
        let mut unit = CaseUnit {
            case: case.clone(),
            key: key.clone(),
            implementation: implementation.clone(),
            statics: statics.clone(),
            shapes,
            launches: count,
            cases,
            search,
            share: 0.,
            spent: mandatory,
            stages: 0,
            formed: None,
            publishing: self.publishing.clone(),
            artifacts: self.artifacts.clone(),
            unpublished: false,
            unformed: false,
            settled: None,
            marker: std::marker::PhantomData,
        };
        if unit.shapes.iter().all(|shape| shape.weight == 0.) {
            // A unit without serving demand has no optional work: nothing
            // could be ranked. Its published canonical default is its
            // record.
            unit.finish(self)?;
            let Phase::Census { launches, .. } = &mut self.phase else {
                unreachable!("the phase is the census");
            };
            launches.remove(&key);
            return Ok(None);
        }
        let Phase::Census { units, .. } = &mut self.phase else {
            unreachable!("the phase is the census");
        };
        units.push(Box::new(unit));
        Ok(None)
    }

    /// A load met a unit without a stored record: stop and report it, or
    /// count it as served at its canonical default, as the load's host
    /// asked. Nothing is published for it: only a search publishes, so a
    /// later search still finds the unit unsearched.
    fn unprepared_unit(
        &mut self,
        entry: &'static str,
        bindings: String,
        statics: BTreeMap<String, u64>,
    ) -> Result<(), CatalogFailure> {
        match self.phase {
            Phase::Serve(UnpreparedUnits::Report) => {
                Err(CatalogFailure::Unprepared { entry, bindings })
            }
            Phase::Serve(UnpreparedUnits::ServeDefaults) => {
                self.unprepared += 1;
                self.context.observer.event(&TuningEvent::Unprepared {
                    entry,
                    bindings,
                    statics,
                });
                Ok(())
            }
            Phase::Count { .. } | Phase::Census { .. } | Phase::Searched => {
                unreachable!("only a load meets a unit without a record")
            }
        }
    }

    /// A unit's choice once the search is over: the one its record was
    /// published with, read without preparing anything. A unit a
    /// development survey owns (feature `tuning-survey`) took no part in
    /// the search and is surveyed here, in the search's process.
    fn searched_unit<T: EntryTuning>(
        &mut self,
        case: &T,
        statics: &NativeSpecialization,
        key: TuningKey,
        shapes: Vec<PointShape>,
    ) -> Result<NativeSpecialization, CatalogFailure> {
        if let Some(chosen) = self.chosen.get(&key) {
            return Ok(chosen.clone());
        }
        #[cfg(feature = "tuning-survey")]
        if let Some(plan) = survey::plan(key.0) {
            let (entry, bindings) = (key.0, key.1.clone());
            let failure = |outcome: String| CatalogFailure::Tuning {
                entry,
                bindings: bindings.clone(),
                outcome,
            };
            let began = Instant::now();
            self.context.observer.event(&TuningEvent::Started {
                entry,
                bindings: bindings.clone(),
                points: shapes.len(),
            });
            let result = self.survey(case, statics, &shapes, plan).map_err(failure)?;
            // A surveyed unit takes no part in the division of the tuning
            // time: the most a search of it could be given is all of it.
            survey::record(&key, self.budget(), &result).map_err(failure)?;
            let tuned = searched_entry(entry, bindings, result, began);
            return Ok(self.finish(key, tuned));
        }
        #[cfg(not(feature = "tuning-survey"))]
        let _ = (case, statics, shapes);
        Err(unit_failure(
            &key,
            "the search published no choice for this unit",
        ))
    }

    /// The choice a load prepares one unit with: a pinned choice, its
    /// stored record's, or, for a unit without a record, its canonical
    /// default (`defaults`) when the load serves defaults
    /// ([`Self::unprepared_unit`]).
    fn served_unit<T: EntryTuning>(
        &mut self,
        case: &T,
        implementation: &NativeImplementation,
        statics: &NativeSpecialization,
        key: TuningKey,
        shapes: Vec<PointShape>,
        defaults: &NativeSpecialization,
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
        if let pinned::Pinned::Chosen(chosen, choice) =
            pinned::lookup(&key, implementation).map_err(failure)?
        {
            self.context.observer.event(&TuningEvent::Pinned {
                entry,
                bindings: bindings.clone(),
                statics: key.2.clone(),
                choice,
                configuration: Configuration::of(&chosen),
            });
            self.pinned.insert(key.clone());
            self.chosen.insert(key, chosen.clone());
            return Ok(chosen);
        }
        #[cfg(not(feature = "pinned-tuning"))]
        let _ = implementation;
        let began = Instant::now();
        let stored = self.stored(case, statics, &shapes).map_err(failure)?;
        let Some((_, _, Some((choice, receipt)))) = stored else {
            self.unprepared_unit(entry, bindings, key.2.clone())?;
            self.chosen.insert(key, defaults.clone());
            return Ok(defaults.clone());
        };
        let tuned = tuned_entry(
            entry,
            bindings,
            choice,
            receipt,
            TuningOrigin::Stored,
            TuningTime::default(),
            began,
        );
        Ok(self.finish(key, tuned))
    }

    /// The digest of the native implementation `case` is tuned on at
    /// `statics`, computed once for each implementation. Computing it
    /// renders the implementation's source, and units that differ only in
    /// what they are measured at share one implementation: an attention
    /// decode unit per span class rendered the same source once each, 5 ms
    /// a time (GB10 and M4 Pro, 2026-10-10).
    fn implementation_digest<T: EntryTuning>(
        &self,
        case: &T,
        statics: &NativeSpecialization,
    ) -> Result<String, String> {
        let key = (
            <T::Entry as seismic::Entry>::NAME,
            case.entry().elements().to_vec(),
            statics.clone(),
        );
        if let Some(digest) = self.digests.borrow().get(&key) {
            return Ok(digest.clone());
        }
        let digest = case
            .digest(self.device, statics)
            .map_err(|error| error.to_string())?;
        self.digests.borrow_mut().insert(key, digest.clone());
        Ok(digest)
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
        let digest = self.implementation_digest(case, statics)?;
        let requirement = self.context.numerical_profile.requirement(&case.entry());
        let key = TuningCacheKey::of(&tuning_key_material(
            self.device,
            entry,
            &case.bindings(),
            statics,
            &digest,
            requirement.activation(),
            shapes,
            self.budget(),
        ));
        // A fresh preparation searches again and replaces the stored record.
        // Any other serves the stored record under this load's requirement.
        // Under the requirement it was searched under, that is its stored
        // choice, and its evidence stays unread in the store. Its
        // measurements are the same facts under every α, so under another
        // the evidence is read and chosen from again, with nothing formed
        // or measured. The store keeps a choice and its evidence together,
        // so evidence that cannot be read is a broken store, not a miss.
        let stored = match self.context.settings.fresh {
            true => None,
            false => cache.tuning::<StoredChoice>(&key),
        };
        let hit = match stored {
            None => None,
            Some((choice, evidence)) if choice.requirement == requirement => {
                Some((choice, Receipt::Stored(evidence)))
            }
            Some((stored, evidence)) => {
                let mut result = evidence.read()?;
                seismic::reselect(&mut result, &requirement);
                // What stopped the record's search is the stored choice's
                // to say: the evidence knows only whether it was complete.
                let choice = StoredChoice {
                    stop: stored.stop,
                    ..StoredChoice::of(&result)
                };
                Some((choice, Receipt::Held(Arc::new(result))))
            }
        };
        if let Some((choice, _)) = &hit {
            // Reading a stored choice is metadata only. Its key binds the
            // implementation, the device and the served shapes it was
            // formed and checked for; the walk that prepares it forms what
            // it serves, once.
            let chosen = choice.overall.specialization();
            if !case.stored_valid(self.device, &chosen) {
                return Err(format!(
                    "terminal tuning choice for {entry} fails its served contract; search cannot reopen"
                ));
            }
            match &choice.disposition {
                seismic::TuningDisposition::CanonicalFallback { .. } => {
                    choice.chosen.usable_fallback()?;
                }
                seismic::TuningDisposition::QualifiedCandidate => {
                    if shapes.iter().any(|shape| shape.weight > 0.) {
                        choice.chosen.qualified_at(shapes).map_err(|detail| {
                            format!("terminal tuning choice for {entry} lacks matching numerical evidence: {detail}; search cannot reopen")
                        })?;
                    }
                }
            }
        }
        Ok(Some((cache, key, hit)))
    }

    /// Survey `case` at `statics` at `shapes`: every admissible
    /// configuration measured at every point.
    #[cfg(feature = "tuning-survey")]
    fn survey<T: EntryTuning>(
        &mut self,
        case: &T,
        statics: &NativeSpecialization,
        shapes: &[PointShape],
        plan: seismic::SurveyPlan,
    ) -> Result<TuningResult, String> {
        let requirement = self.context.numerical_profile.requirement(&case.entry());
        let mut cases = UnitCases {
            points: shapes.iter().map(|_| None).collect(),
            guard: None,
        };
        let device = self.device;
        let result = case
            .tune(
                device,
                statics,
                &mut UnitPoints {
                    case,
                    shapes,
                    inputs: self.inputs(),
                    cases: &mut cases,
                },
                requirement,
                Strategy::Survey(plan),
            )
            .map_err(|error| error.to_string());
        // The survey's cases are gone: nothing binds what it imported.
        drop(cases);
        self.inputs().release_unheld();
        result
    }

    /// Record a unit's outcome and report it.
    fn finish(&mut self, key: TuningKey, mut tuned: TunedEntry) -> NativeSpecialization {
        tuned.statics = key.2.clone();
        self.context
            .observer
            .event(&TuningEvent::Finished(tuned.clone()));
        let chosen = tuned.overall.specialization();
        #[cfg(feature = "pinned-tuning")]
        pinned::record(&key, &chosen);
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

/// Canonical formation uses only served geometry. Input construction and
/// observations are unavailable during prerequisites and cache verification.
struct CanonicalPoints<'a, T> {
    case: &'a T,
    shapes: &'a [PointShape],
}
impl<T: EntryTuning> seismic::PointSource<'static, T::Entry> for CanonicalPoints<'_, T> {
    fn points(&self) -> Vec<seismic::PointSpec> {
        point_specs(self.shapes)
    }
    fn served(&self) -> Vec<BTreeMap<String, u64>> {
        self.case.served_dimensions(self.shapes)
    }
    fn build(&mut self, _: usize) -> Result<seismic::PointInputs<'static>, String> {
        Err("canonical formation cannot construct numerical observations".into())
    }
}
fn point_specs(shapes: &[PointShape]) -> Vec<seismic::PointSpec> {
    shapes
        .iter()
        .zip(census_points(shapes))
        .map(|(shape, census)| seismic::PointSpec {
            label: shape.label.clone(),
            weight: shape.weight,
            class: shape.class.clone(),
            cost: shape.cost(),
            census,
        })
        .collect()
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
    fn release(&mut self, point: usize) {
        self.cases.points[point] = None;
        self.inputs.release_unheld();
    }

    fn points(&self) -> Vec<seismic::PointSpec> {
        point_specs(self.shapes)
    }

    fn served(&self) -> Vec<BTreeMap<String, u64>> {
        self.case.served_dimensions(self.shapes)
    }

    fn build(&mut self, point: usize) -> Result<seismic::PointInputs<'static>, String> {
        if self.cases.points[point].is_none() {
            let built = self.case.rotation(&mut self.inputs, &self.shapes[point])?;
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
        )?;
        // Every argument set of a point writes the same rows. The guard
        // observes its states' other rows too: a candidate may leave no
        // mark there, whatever error its written rows are allowed.
        let written = cases
            .iter()
            .take(1)
            .flat_map(T::state)
            .map(|(name, state)| {
                (
                    name.to_owned(),
                    seismic::WrittenRows {
                        rows: state.written.clone(),
                        observe_rest: guard,
                    },
                )
            })
            .collect();
        let workload = self.case.workload(cases.len());
        Ok(seismic::point_inputs::<T::Entry>(
            cases.iter_mut().map(T::args).collect(),
            workload,
            initialize,
            written,
        ))
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
/// tuning version and budget, the device with its toolchain and driver, the
/// unit, the implementation's digest, the definition of its measurements
/// against the default (the comparison's version and the activation type
/// subjects are measured against, `activation`), and the shapes it was
/// measured at. The budget bounds how far the search could reach, so a
/// result is read only under the budget it was searched within. How it was
/// otherwise searched (settings, workload weights, the points timed) is not:
/// a better search never invalidates a result that is still correct; see
/// [`SEARCH_VERSION`]. Neither is the caller's α: it judges the stored
/// measurements and changes none, so a load under another α reads the same
/// record and chooses from it again ([`seismic::reselect`]).
fn tuning_key_material(
    device: &Device,
    entry: &str,
    bindings: &str,
    statics: &NativeSpecialization,
    digest: &str,
    activation: Option<DType>,
    shapes: &[PointShape],
    budget: Duration,
) -> String {
    let mut shapes = shapes
        .iter()
        .map(|shape| shape.label.as_str())
        .collect::<Vec<_>>();
    shapes.sort_unstable();
    format!(
        "tuning {SEARCH_VERSION}\nbudget {} ms\ndevice {}\nentry {entry}\nbindings {bindings}\n\
         statics {:?}\nimplementation {digest}\ncomparison {} steps {activation:?}\nshapes {}",
        budget.as_millis(),
        device.tuning_identity(),
        statics.statics(),
        seismic::COMPARISON_VERSION,
        shapes.join(","),
    )
}

/// The outcome of a unit whose search just produced `result`.
fn searched_entry(
    entry: &'static str,
    bindings: String,
    result: TuningResult,
    began: Instant,
) -> TunedEntry {
    let choice = StoredChoice::of(&result);
    let time = result.time.clone();
    tuned_entry(
        entry,
        bindings,
        choice,
        Receipt::Held(Arc::new(result)),
        TuningOrigin::Searched,
        time,
        began,
    )
}

/// The outcome of a unit that serves `choice`, whose evidence is `receipt`.
fn tuned_entry(
    entry: &'static str,
    bindings: String,
    choice: StoredChoice,
    receipt: Receipt,
    origin: TuningOrigin,
    time: TuningTime,
    began: Instant,
) -> TunedEntry {
    TunedEntry {
        entry,
        bindings,
        overall: choice.overall,
        statics: BTreeMap::new(),
        origin,
        qualified: matches!(
            choice.disposition,
            seismic::TuningDisposition::QualifiedCandidate
        ),
        stop: choice.stop,
        coverage: choice.coverage,
        progress: None,
        measured: choice.measured,
        excluded: choice.excluded,
        rejections: choice.rejections,
        first_rejection: choice.first_rejection,
        seconds: began.elapsed().as_secs_f64(),
        time,
        receipt,
    }
}

#[cfg(test)]
mod tests;
