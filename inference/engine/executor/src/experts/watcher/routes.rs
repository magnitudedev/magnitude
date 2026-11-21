//! The key records' one reader: of every decode-form step, each streamed
//! layer's rows' selection key of every expert, as its route wrote its key
//! records (streaming_words.seismic's keys), which placement ranks every
//! expert by (`reuse::StepKeys`); and, where a load records its routes
//! (`ExpertBankPlan::recording`), a trace of them with each layer's choices
//! and every front row's verdict.

use super::core::{Layer, RoutingRecord};
use super::steps::{DecodeLaunch, Step, Steps};
use crate::experts::Words;
use crate::experts::interest::Judged;
use crate::experts::reuse::StepKeys;
use crate::planning::RegionWords;
use magnitude_kernels::constants::VERIFY_ROWS;
use std::sync::Mutex;

/// One decode-form step's routes (a decode step's, or a prompt chunk's of
/// at most `VERIFY_ROWS` rows).
#[derive(Clone, Debug, PartialEq)]
pub struct RoutedStep {
    /// The step's sequence, which a front row's verdict names
    /// ([`RecordedRoutes::stood`]).
    pub sequence: u64,
    /// Each row's request and position, rows in order.
    pub rows: Vec<(u64, u64)>,
    /// Its rows' front bits (STEP_FRONT's: bit `r` where row `r` only runs
    /// ahead).
    pub front: u32,
    /// Per streamed layer, each row's `K` choices, rows in order.
    pub choices: Vec<Vec<u32>>,
    /// Per streamed layer, each row's selection key of every expert, `[rows,
    /// E]` row-major: the value its route ranked (a fused route's
    /// probability, a general route's score plus its bias), so a row's top
    /// `K` is its choices, ties to the higher expert.
    pub keys: Vec<Vec<f32>>,
}

/// What a recording load recorded since it was last taken
/// (`ExecutorDomain::take_routes`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecordedRoutes {
    /// `K`: a row's choices.
    pub selected: usize,
    /// Each decode-form step's routes, in completion order.
    pub steps: Vec<RoutedStep>,
    /// Each judged front row's verdict: the step that ran it, its batch row,
    /// and whether its guess stood.
    pub stood: Vec<(u64, usize, bool)>,
}

/// Every streamed layer's key records: the one reader of them.
pub(super) struct KeyRecords {
    /// Each streamed layer's key records and its experts.
    layers: Vec<(Words, usize)>,
    words: RegionWords,
    selected: usize,
}

impl KeyRecords {
    /// The key records of `layers`, of records shaped by `words`, `selected`
    /// choices a row.
    pub(super) fn of(layers: &[Layer], words: RegionWords, selected: usize) -> Self {
        Self { layers: layers.iter().map(|layer| (layer.keys.clone(), layer.experts)).collect(), words, selected }
    }

    /// Decode-form step `sequence`'s keys, of `launch`, which completes and
    /// so holds its records until it ends: each row's depth, the tokens it
    /// routes past its request's exact row (an exact row's 0, a front or
    /// verify row's its position past it; a row the step reads for of a
    /// prompt chunk, 0), whether its guessed rows are verify rows, each
    /// request's first exact row's position, and each layer's rows' keys,
    /// copied while it holds them.
    pub(super) fn read(&self, steps: &Steps, sequence: Step, launch: DecodeLaunch) -> StepKeys {
        let rows = launch.rows;
        let slot = (sequence % self.words.flight) as usize * VERIFY_ROWS as usize;
        let layers = steps
            .while_owned(sequence, |_| {
                self.layers
                    .iter()
                    .map(|(keys, experts)| {
                        let base = slot * experts;
                        (0..rows * experts).map(|word| f32::from_bits(keys.load(base + word) as u32)).collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .expect("a completing step owns its key records");
        let placed = launch.placed();
        let guessed = launch.guessed.mask();
        let exact = |request| {
            placed
                .iter()
                .enumerate()
                .filter(move |&(row, &(of, _))| of == request && guessed >> row & 1 == 0)
                .map(|(_, &(_, position))| position)
        };
        let depths = placed
            .iter()
            .enumerate()
            .map(|(row, &(request, position))| match guessed >> row & 1 {
                0 => 0,
                _ => {
                    let exact = exact(request).max().unwrap_or_else(|| panic!("step {sequence}'s guessed row {row} runs beside no exact row of its request"));
                    position
                        .checked_sub(exact)
                        .and_then(|depth| u8::try_from(depth).ok())
                        .filter(|&depth| depth > 0)
                        .unwrap_or_else(|| panic!("step {sequence}'s guessed row {row} at {position} is not past its exact row at {exact}"))
                }
            })
            .collect();
        let mut positions = Vec::<(u64, u64)>::new();
        for &(request, _) in placed {
            if !positions.iter().any(|&(of, _)| of == request.0) {
                positions.push((request.0, exact(request).min().unwrap_or_else(|| panic!("step {sequence}'s request {request:?} runs no exact row"))));
            }
        }
        StepKeys { choices: self.selected, depths, verifies: launch.verifies(), positions: positions.into(), layers }
    }
}

/// The trace of a load recording its routes.
pub(super) struct RouteRecorder {
    /// Each streamed layer's routing records.
    routing: Vec<Words>,
    words: RegionWords,
    recorded: Mutex<RecordedRoutes>,
}

impl RouteRecorder {
    /// The recorder of `layers`, of records shaped by `words` and `selected`
    /// choices a row.
    pub(super) fn of(layers: &[Layer], words: RegionWords, selected: usize) -> Self {
        Self {
            routing: layers.iter().map(|layer| layer.routing.clone()).collect(),
            words,
            recorded: Mutex::new(RecordedRoutes { selected, ..RecordedRoutes::default() }),
        }
    }

    /// Record decode-form step `sequence` of `launch`, its keys `keys`
    /// ([`KeyRecords::read`]), which completes and so holds its records
    /// until it ends: each layer's routing record, copied while it holds
    /// them, every layer's (a whole layer's route writes its routing record
    /// too, as every route writes its keys).
    pub(super) fn record(&self, steps: &Steps, sequence: Step, launch: DecodeLaunch, keys: &StepKeys) {
        let rows = launch.rows;
        let copied = steps
            .while_owned(sequence, |owned| self.routing.iter().map(|routing| RoutingRecord::copy(owned, routing, self.words)).collect::<Vec<_>>())
            .expect("a completing step owns its routing records");
        let mut recorded = self.recorded.lock().expect("the recorded routes");
        let selected = recorded.selected;
        let choices = copied
            .into_iter()
            .enumerate()
            .map(|(layer, record)| {
                let record = RoutingRecord::read(record, self.words, layer);
                assert!(record.count() >= rows * selected, "step {sequence}'s routing record at layer {layer} holds {} of its {rows} rows' choices", record.count());
                record
                    .choices()
                    .take(rows * selected)
                    .map(|(expert, _)| u32::try_from(expert).unwrap_or_else(|_| panic!("step {sequence} chose expert {expert} at layer {layer}")))
                    .collect()
            })
            .collect();
        let rows = launch.placed().iter().map(|&(request, position)| (request.0, position)).collect();
        recorded.steps.push(RoutedStep { sequence, rows, front: launch.front(), choices, keys: keys.layers.clone() });
    }

    /// The verdicts of judged front rows.
    pub(super) fn judged(&self, judged: &[Judged]) {
        let mut recorded = self.recorded.lock().expect("the recorded routes");
        recorded.stood.extend(judged.iter().filter_map(|judged| judged.stood.map(|stood| (judged.step, judged.row, stood))));
    }

    /// What was recorded since it was last taken.
    pub(super) fn take(&self) -> RecordedRoutes {
        let mut recorded = self.recorded.lock().expect("the recorded routes");
        let selected = recorded.selected;
        std::mem::replace(&mut *recorded, RecordedRoutes { selected, ..RecordedRoutes::default() })
    }
}
