//! The exchange (`streaming_words.seismic`): the one protocol by which the
//! device asks the host for anything and waits for its answer, its host
//! side. Every request is a lane of a slot of a table of its kind
//! ([`ExchangeTable`]): the file kind's (the decode route's experts neither the
//! bank nor the tier holds), the site kind's (the CPU's misses), the
//! scratch kind's (a scratch-served step's request and its rounds). A
//! lane's request and answer are each one u64 step; the answer is the step
//! or the step with [`FAILED`], and only moves forward. The wait rule is
//! the kernels' (`slot_answered` in routed.seismic), computed here alone
//! ([`ExchangeTable::answered`]), which a test runs against the reference body.
//! Its liveness is its watchdog's ([`Watchdog`]): a lane the device waits
//! on left unanswered past the load's bound ends the process with why. Its
//! host's own is its pulse's ([`Pulse`]): every table beats while it lives,
//! so a spinning wait whose host is gone ends.

mod pulse;
mod watchdog;

pub(crate) use pulse::Pulse;
pub(in crate::experts) use watchdog::{Bound, Bounds, Kind, Parked, Trip, Watchdog, Watched, Witness, abort};

use super::Words;
use magnitude_kernels::{SlotTable, constants};

/// The failure flag of an answer: bit 63, SLOT_FAILED of its high word.
pub(crate) const FAILED: u64 = (constants::SLOT_FAILED as u64) << 32;

/// What a lane's answer holds for a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Answer {
    /// The step's answer: its payload is whole.
    Answered,
    /// The step's failure.
    Failed,
}

/// One table of the exchange, host-shared, laid out as its kind's
/// [`SlotTable`].
#[derive(Clone)]
pub(crate) struct ExchangeTable {
    words: Words,
    table: SlotTable,
    /// Its place in the host's pulse, which beats it while it lives.
    _pulsed: std::sync::Arc<pulse::Pulsed>,
}

impl ExchangeTable {
    /// The table `table` over `tensor`'s words, every one zero (no lane is
    /// requested for any step: steps begin at one), joined to its load's
    /// `pulse`: no table a device can wait on goes unpulsed.
    pub(crate) fn new(tensor: seismic::Tensor, table: SlotTable, pulse: &Pulse) -> Result<Self, String> {
        let words = Words::new(tensor, table.words(), 0)?;
        Ok(Self { _pulsed: std::sync::Arc::new(pulse.join(words.clone(), table)?), words, table })
    }

    pub(crate) fn table(&self) -> SlotTable {
        self.table
    }

    /// The device whose steps it serves.
    pub(crate) fn device(&self) -> seismic::Device {
        self.words.tensor.device()
    }

    fn at(&self, step: u64, lane: u64, index: u64) -> usize {
        assert!(index < self.table.width(), "word {index} of a lane of {}", self.table.width());
        (self.table.lane(step, lane) + index) as usize
    }

    /// The step lane `lane` of `step`'s slot is requested for.
    pub(crate) fn request_of(&self, step: u64, lane: u64) -> u64 {
        self.words.load_u64(self.at(step, lane, constants::SLOT_REQUEST))
    }

    /// Whether lane `lane` of `step`'s slot is requested for `step`.
    pub(crate) fn requested(&self, step: u64, lane: u64) -> bool {
        self.request_of(step, lane) == step
    }

    /// The earliest of `steps` lane `lane` is requested for that `live`
    /// admits. It reads only those steps' slots, so what a look costs follows
    /// the steps it can find, not the slots of the flight.
    pub(crate) fn earliest_requested(&self, steps: std::ops::Range<u64>, lane: u64, live: impl Fn(u64) -> bool) -> Option<u64> {
        steps.into_iter().find(|&step| self.requested(step, lane) && live(step))
    }

    /// Request lane `lane` for `step`, after its payload: as the device does,
    /// for a request the host makes (a measurement's, a qualification's, a
    /// test's), or for a wait whose request is its arrival where the device
    /// cannot store it before its park (`watcher::parks`).
    pub(crate) fn request(&self, step: u64, lane: u64) {
        self.words.store_u64(self.at(step, lane, constants::SLOT_REQUEST), step);
    }

    /// What lane `lane`'s answer holds for `step`: none while it holds an
    /// earlier step's.
    pub(crate) fn answer_of(&self, step: u64, lane: u64) -> Option<Answer> {
        match self.words.load_u64(self.at(step, lane, constants::SLOT_ANSWER)) {
            held if held == step => Some(Answer::Answered),
            held if held == step | FAILED => Some(Answer::Failed),
            _ => None,
        }
    }

    /// The exchange's one wait rule (`slot_answered`): `step`'s wait on lane
    /// `lane` is over once the lane is not requested for it (nothing is
    /// awaited) or its answer holds the step or its failure.
    pub(crate) fn answered(&self, step: u64, lane: u64) -> bool {
        !self.requested(step, lane) || self.answer_of(step, lane).is_some()
    }

    /// Answer lane `lane` for `step` (`failed`: with the failure flag),
    /// after its payload: the answer only moves forward, so an answer for a
    /// step no later than the one it holds, its failure included, leaves
    /// it. Whether it published.
    pub(crate) fn answer(&self, step: u64, lane: u64, failed: bool) -> bool {
        let value = if failed { step | FAILED } else { step };
        self.words.update_u64(self.at(step, lane, constants::SLOT_ANSWER), |held| (held & !FAILED < step).then_some(value))
    }

    /// Slot `slot`'s lane `lane` as it holds them, its request's step and
    /// its answer: what a reader with no step at hand reads (a pool watching
    /// its slots, a watchdog).
    pub(crate) fn held(&self, slot: u64, lane: u64) -> (u64, u64) {
        assert!(slot < self.table.slots(), "slot {slot} of {}", self.table.slots());
        (
            self.words.load_u64(self.at(slot, lane, constants::SLOT_REQUEST)),
            self.words.load_u64(self.at(slot, lane, constants::SLOT_ANSWER)),
        )
    }

    /// Payload word `index` of lane `lane` of `step`'s slot.
    pub(crate) fn load(&self, step: u64, lane: u64, index: u64) -> i32 {
        self.words.load(self.at(step, lane, index))
    }

    /// Store payload word `index` of lane `lane` of `step`'s slot.
    pub(crate) fn store(&self, step: u64, lane: u64, index: u64, value: i32) {
        self.words.store(self.at(step, lane, index), value)
    }
}

#[cfg(test)]
mod tests {
    //! The host's rule is the kernels': `slot_answered`, routed.seismic's
    //! reference body, which every native's wait is admitted against, run
    //! in the interpreter over every state a lane can hold.
    use super::*;
    use seismic::{BackendName, Devices, Element, Tensor};
    use seismic_lang::checked::{CheckedModule, SourceFile, check_source};
    use seismic_lang::entry::ElementBindings;
    use seismic_lang::interp::{Arg, Interpreter, OutcomeValue, TensorData};
    use seismic_lang::reference_math::ReferenceScalar;
    use seismic_lang::types::DType;

    fn module() -> CheckedModule {
        let mut sources = seismic_std::sources();
        for (path, text) in [
            ("routed.seismic", include_str!("../../../kernels/kernels/routed.seismic")),
            ("streaming_words.seismic", include_str!("../../../kernels/kernels/streaming_words.seismic")),
            ("functions.seismic", include_str!("../../../kernels/kernels/functions.seismic")),
        ] {
            sources.push(SourceFile { path: path.into(), text: text.into() });
        }
        check_source(sources).unwrap()
    }

    /// `slot_answered` of slot 0 of a one-slot table holding `request` and
    /// `answer`, for step `q`.
    fn reference(module: &CheckedModule, request: u64, answer: u64, q: u64) -> bool {
        let entry = module.entry(module.entry_named("slot_answered").unwrap(), &ElementBindings::new()).unwrap();
        let mut interpreter = Interpreter::new(&entry);
        let mut words = vec![0f64; constants::SLOT_PAYLOAD as usize];
        for (at, value) in [(constants::SLOT_REQUEST, request), (constants::SLOT_ANSWER, answer)] {
            words[at as usize] = f64::from(value as u32 as i32);
            words[at as usize + 1] = f64::from((value >> 32) as u32 as i32);
        }
        let slots = Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::I32, vec![1, constants::SLOT_PAYLOAD as usize], words)));
        let outcome = interpreter
            .run(&[slots, Arg::Scalar(ReferenceScalar::I32(0)), Arg::Scalar(ReferenceScalar::U64(q))])
            .unwrap();
        let answered = match outcome.results().next().expect("slot_answered returns its rule").value() {
            OutcomeValue::Scalar(ReferenceScalar::Bool(answered)) => answered,
            _ => panic!("slot_answered returns a bool"),
        };
        answered
    }

    #[test]
    fn the_hosts_wait_rule_is_the_kernels() {
        let module = module();
        let device = Devices::discover(&[]).unwrap().open_backend(BackendName::Cpu).unwrap();
        let table = SlotTable::file(1, 1);
        let slots = ExchangeTable::new(Tensor::shared(&device, Element::i32(), &[table.words()]).unwrap(), table, &Pulse::new()).unwrap();
        let q = (7u64 << 32) | 5;
        let states = [0, q - 1, q, q + 1, q | FAILED, (q - 1) | FAILED, (q + 1) | FAILED];
        for &request in &states[..4] {
            for &answer in &states {
                slots.words.store_u64(constants::SLOT_REQUEST as usize, request);
                slots.words.store_u64(constants::SLOT_ANSWER as usize, answer);
                assert_eq!(slots.answered(q, 0), reference(&module, request, answer, q), "request {request:#x} answer {answer:#x}");
            }
        }
    }

    /// An answer only moves forward: a later step's leaves it, the step's
    /// failure after its answer does not overwrite it, and a failure is
    /// final for its step.
    #[test]
    fn an_answer_only_moves_forward() {
        let device = Devices::discover(&[]).unwrap().open_backend(BackendName::Cpu).unwrap();
        let table = SlotTable::scratch(2, 2);
        let slots = ExchangeTable::new(Tensor::shared(&device, Element::i32(), &[table.words()]).unwrap(), table, &Pulse::new()).unwrap();
        let lane = constants::SCRATCH_ROUND;
        slots.request(5, lane);
        assert!(!slots.answered(5, lane));
        assert!(slots.answered(5, 0), "the request's lane is not requested");
        assert!(slots.answer(5, lane, false));
        assert_eq!(slots.answer_of(5, lane), Some(Answer::Answered));
        assert!(!slots.answer(5, lane, true), "answered: a failure leaves it");
        assert!(!slots.answer(3, lane, false), "an earlier step's");
        assert!(slots.answer(7, lane, true));
        assert_eq!(slots.answer_of(7, lane), Some(Answer::Failed));
        assert!(!slots.answer(7, lane, false), "failed: final");
        assert_eq!(slots.answer_of(6, lane), None);
        // Every lane's words begin at a whole u64.
        assert_eq!(table.width() % 2, 0);
        assert_eq!(table.lane(7, lane) % 2, 0);
    }

    /// A table pulses by construction, no watchdog or load around it: every
    /// lane of every slot holds a beat from the moment it is made, and a new
    /// one within a few PULSE_MS, for as long as it lives.
    #[test]
    fn a_table_pulses_from_its_making() {
        let device = Devices::discover(&[]).unwrap().open_backend(BackendName::Cpu).unwrap();
        let table = SlotTable::scratch(3, 2);
        let slots = ExchangeTable::new(Tensor::shared(&device, Element::i32(), &[table.words()]).unwrap(), table, &Pulse::new()).unwrap();
        let beats = || {
            (0..table.slots())
                .flat_map(|slot| (0..table.lanes()).map(move |lane| (slot, lane)))
                .map(|(slot, lane)| slots.load(slot, lane, constants::SLOT_PULSE))
                .collect::<Vec<_>>()
        };
        let first = beats();
        assert!(first.iter().all(|&beat| beat != 0), "every lane beats at once: {first:?}");
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(50 * constants::PULSE_MS);
        loop {
            let now = beats();
            if now.iter().zip(&first).all(|(now, first)| now != first) {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "every lane beats again: {first:?} then {now:?}");
            std::thread::sleep(std::time::Duration::from_millis(constants::PULSE_MS));
        }
    }
}
