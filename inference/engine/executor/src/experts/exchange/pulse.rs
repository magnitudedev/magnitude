//! The host's pulse (streaming_words.seismic's SLOT_PULSE): the proof a
//! spinning wait reads that its host lives. Metal never ends a dead
//! process's running kernel, so a wait whose pulse stands still for
//! PULSE_SPINS spins ends as on its failure.
//!
//! Each load has one ([`Pulse`], its steps': `Steps::pulse`), and every
//! table of the load's exchange ([`super::ExchangeTable`]) joins it as it is made:
//! no table a device can wait on goes unpulsed. Its thread advances every
//! live table's pulse every PULSE_MS, from the first table's making until
//! the last drops. A wait ends only once its pulse stood still for its
//! least patience (PULSE_SPINS spins at the fastest spin measured, 105 ms),
//! so a beat that comes later than that less one period ([`late`]: the
//! host starved, a loaded machine) is the only way a live host's wait can
//! end before its answer: it is counted ([`Pulse::late_beats`]), and every
//! step of the load that spans one is interrupted, its seeds never mapped
//! (`ExpertStep`, `Core::advance`), and re-run from its accepted state
//! (the scheduler's round driver). A stall shorter than that touches
//! nothing, and so does any while no table lives: no wait reads the pulse.
//! A test that plays the device itself drives its pulse
//! ([`Pulse::driven`]), so a late beat is its input, never its host's load.
//! Where parks are submission boundaries (Vulkan,
//! `seismic::BackendName::parks_in_kernels` false) a wait reads its slot
//! once, after the host opened its park, and never its pulse: the load's
//! pulse is unread ([`Pulse::of`]), beats nothing and runs no thread: no
//! wait of it ends on its pulse, so none stores an expiry and no step of
//! it is interrupted.
//!
//! The live tables own the thread. The pulse keeps each one's words while
//! it lives and beats them under its lock, so the thread never holds a
//! table (nor the device its words are of), and the last table to drop
//! ends the thread and joins it: while no table lives, no pulse thread
//! runs, so none outlives its load.

use super::Words;
use magnitude_kernels::{SlotTable, constants};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The shortest spin of a device's wait measured (an idle M4 Pro's, 70 ns,
/// 2026-11-02), at which PULSE_SPINS spins are the device's least patience.
const FASTEST_SPIN: Duration = Duration::from_nanos(70);

/// How late a beat may come before a wait may have ended: the device's
/// least patience, less one period for the beat's own way to the device.
fn late() -> Duration {
    FASTEST_SPIN * constants::PULSE_SPINS as u32 - Duration::from_millis(constants::PULSE_MS)
}

/// A table's place in a pulse: its words are beaten for as long as it
/// lives, and the last place to drop joins the pulse's thread.
pub(super) struct Pulsed {
    inner: Arc<Inner>,
    id: u64,
}

/// A live table: its words, and every lane's SLOT_PULSE among them.
struct Beaten {
    id: u64,
    words: Words,
    pulses: Vec<usize>,
}

impl Beaten {
    /// Store `beat` as every lane's SLOT_PULSE: a wait reads only that it
    /// moves, so no beat orders anything.
    fn beat(&self, beat: i32) {
        let words = self.words.in_place();
        for &pulse in &self.pulses {
            words[pulse].store(beat as u32, Ordering::Relaxed);
        }
    }
}

impl Drop for Pulsed {
    /// Its words leave the pulse; the last table's drop ends the thread
    /// and joins it.
    fn drop(&mut self) {
        let thread = {
            let mut beating = self.inner.beating.lock().expect("a pulse never panics");
            if let Beater::Unread = beating.beater {
                return;
            }
            beating.tables.retain(|table| table.id != self.id);
            if !beating.tables.is_empty() {
                return;
            }
            if let Beater::Running = beating.beater {
                beating.beater = Beater::Stopped;
            }
            beating.generation += 1;
            self.inner.changed.notify_all();
            beating.thread.take()
        };
        if let Some(thread) = thread {
            thread.join().expect("the exchange's pulse never panics");
        }
    }
}

/// What beats a pulse.
enum Beater {
    /// Nothing: no table lives, so no wait reads the pulse and none can end
    /// before its answer, however long since the last beat.
    Stopped,
    /// Its thread, every PULSE_MS, while a table lives.
    Running,
    /// Nothing, ever: no wait of the load reads its pulse, so none can end
    /// on it, and none stores an expiry (SLOT_EXPIRED) for the host to
    /// interrupt its step by ([`Pulse::unread`]).
    Unread,
    /// The test, by [`Pulse::beat_after`]: when a beat comes is its input,
    /// never the host's load. For tables no device waits on (a test that
    /// plays the device).
    #[cfg(test)]
    Driven,
}

/// The tables a pulse beats, what beats them, its thread while a table
/// lives and the generation it serves (each last table's drop ends one, so
/// a thread it ends never beats a table joined after), the last beat, and
/// when it came.
struct Beating {
    tables: Vec<Beaten>,
    next_id: u64,
    beater: Beater,
    thread: Option<JoinHandle<()>>,
    generation: u64,
    beat: i32,
    beaten: Instant,
}

impl Beating {
    /// Beat every live table with the next beat, which came `gap` after
    /// the last: counted in `late` when later than [`late`].
    fn beat_tables(&mut self, gap: Duration, late_beats: &AtomicU64) {
        if gap > late() {
            late_beats.fetch_add(1, Ordering::AcqRel);
            tracing::warn!(
                "the exchange's pulse beat {gap:?} after its last beat, past {:?}, within a beat of the device's least patience; \
                 every step in flight is interrupted and re-run.",
                late()
            );
        }
        self.beat = self.beat.wrapping_add(1);
        for table in &self.tables {
            table.beat(self.beat);
        }
    }
}

struct Inner {
    beating: Mutex<Beating>,
    /// Woken as the last table drops, which ends the thread.
    changed: Condvar,
    /// The beats that came late.
    late: AtomicU64,
}

/// A load's pulse: every table of its exchange beats while it lives.
pub(crate) struct Pulse {
    inner: Arc<Inner>,
}

impl Pulse {
    /// A pulse its thread beats.
    pub(crate) fn new() -> Self {
        Self::beaten_by(Beater::Stopped)
    }

    /// The pulse of a load no wait of which reads it: its waits wait at
    /// parks the host opens, and read their slots once.
    fn unread() -> Self {
        Self::beaten_by(Beater::Unread)
    }

    /// The pulse of a load on `backend`: beaten where its waits spin in
    /// their kernels, unread where they park at submission boundaries.
    pub(crate) fn of(backend: seismic::BackendName) -> Self {
        if backend.parks_in_kernels() { Self::new() } else { Self::unread() }
    }

    /// A pulse the test beats ([`Self::beat_after`]).
    #[cfg(test)]
    pub(crate) fn driven() -> Self {
        Self::beaten_by(Beater::Driven)
    }

    fn beaten_by(beater: Beater) -> Self {
        let beating = Beating { tables: Vec::new(), next_id: 0, beater, thread: None, generation: 0, beat: 1, beaten: Instant::now() };
        Self { inner: Arc::new(Inner { beating: Mutex::new(beating), changed: Condvar::new(), late: AtomicU64::new(0) }) }
    }

    /// How many beats have come late ([`late`]), as a step begins.
    pub(crate) fn late_beats(&self) -> u64 {
        self.inner.late.load(Ordering::Acquire)
    }

    /// Whether a beat may have come late since `late_beats` (a step's
    /// begin's [`Self::late_beats`]): one came late since, or its thread
    /// has not beaten a live table for longer than [`late`] now (starved,
    /// its late beat not yet counted). No wait ends before its pulse stood
    /// still that long, and none reads a pulse no thread beats (no table
    /// lives), so a step for which this is false had no wait end before
    /// its answer.
    pub(crate) fn late_since(&self, late_beats: u64) -> bool {
        let beating = self.inner.beating.lock().expect("a pulse never panics");
        let starved = matches!(beating.beater, Beater::Running) && beating.beaten.elapsed() > late();
        self.late_beats() != late_beats || starved
    }

    /// `words` (laid out as `table`) joined to the pulse, beaten at once and
    /// then with every beat while the returned place lives.
    pub(super) fn join(&self, words: Words, table: SlotTable) -> Result<Pulsed, String> {
        let mut beating = self.inner.beating.lock().expect("a pulse never panics");
        let id = beating.next_id;
        beating.next_id += 1;
        if let Beater::Unread = beating.beater {
            return Ok(Pulsed { inner: self.inner.clone(), id });
        }
        let pulses = (0..table.slots())
            .flat_map(|slot| (0..table.lanes()).map(move |lane| (slot, lane)))
            .map(|(slot, lane)| (table.lane(slot, lane) + constants::SLOT_PULSE) as usize)
            .collect();
        let beaten = Beaten { id, words, pulses };
        beaten.beat(beating.beat);
        beating.tables.push(beaten);
        if let Beater::Stopped = beating.beater {
            beating.beaten = Instant::now();
            let (inner, generation) = (self.inner.clone(), beating.generation);
            // A thread its last table's drop ended is joined by that drop,
            // which no longer needs it here.
            let spawned = std::thread::Builder::new()
                .name("expert-exchange-pulse".into())
                .spawn(move || beat(&inner, generation));
            match spawned {
                Ok(thread) => beating.thread = Some(thread),
                Err(error) => {
                    beating.tables.pop();
                    return Err(format!("the exchange's pulse: {error}"));
                }
            }
            beating.beater = Beater::Running;
        }
        Ok(Pulsed { inner: self.inner.clone(), id })
    }

    /// Hold the pulse's thread while the returned guard lives (a starved
    /// host, in a test).
    #[cfg(test)]
    pub(crate) fn hold(&self) -> std::sync::MutexGuard<'_, impl Sized> {
        self.inner.beating.lock().expect("a pulse never panics")
    }

    /// Beat a driven pulse's live tables `gap` after its last beat, as its
    /// thread would have: counted when late ([`late`]).
    #[cfg(test)]
    pub(crate) fn beat_after(&self, gap: Duration) {
        let mut beating = self.inner.beating.lock().expect("a pulse never panics");
        assert!(matches!(beating.beater, Beater::Driven), "only a driven pulse is beaten by its test");
        beating.beat_tables(gap, &self.inner.late);
    }
}

/// A pulse's thread for `generation`: a beat of every live table every
/// PULSE_MS, under the pulse's lock, until the last table drops (which
/// ends the generation, wakes it and joins it); a beat later than [`late`]
/// is counted ([`Pulse::late_beats`]).
fn beat(inner: &Inner, generation: u64) {
    let period = Duration::from_millis(constants::PULSE_MS);
    let mut beating = inner.beating.lock().expect("a pulse never panics");
    loop {
        beating = inner
            .changed
            .wait_timeout_while(beating, period, |beating| beating.generation == generation)
            .expect("a pulse never panics")
            .0;
        if beating.generation != generation {
            return;
        }
        let now = Instant::now();
        let gap = now.duration_since(beating.beaten);
        beating.beaten = now;
        beating.beat_tables(gap, &inner.late);
    }
}
