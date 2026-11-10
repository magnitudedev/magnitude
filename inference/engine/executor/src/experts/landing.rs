//! A file miss's landing: the expert's stored record, read into a landing
//! record (a staging record, `source::Records`), becomes a record of its
//! group in the bank's resident form, then is mapped and each choice that
//! named it answered (`Answer`): its record word in the request's slot of
//! the layer's file table, and the slot itself once every absent expert is.
//! A read that failed lands nothing and answers -1. Who converts it is
//! fixed by who places the bank's records (`Placer`), never chosen per
//! request:
//!
//! - **Into a host tier** (`Placer::Tier`, a discrete device), and every
//!   read ahead's landing (a hint's, the tier fill's, which no launch waits
//!   on as it lands): the host converts it piece by piece into a ready tier
//!   record (`RowConversion::piece`, pieces of whole stored tiles of at most
//!   [`CONVERSION_PIECE`] source bytes, byte-equal to every device's
//!   repack), then maps it: the layer's tier map names it (`tier[e] = t`, a
//!   release store, only now that its bytes are whole), then answers. It
//!   runs on the host compute executor (`compute`), submitted where its
//!   read ends (the read engine's worker that ends it, `Ticket::on_end`) or
//!   by the watcher where the stored record is already whole in staging:
//!   its pieces convert on any of the executor's workers, then its end maps
//!   the record and answers ([`Converting`]), so no host round trip but the
//!   landing itself is on a miss's path.
//! - **Into the bank** (`Placer::Bank`, unified memory), every demand read
//!   a step's landing launch waits on (`routed_land`): the device converts
//!   it ([`DeviceLanding`]). Each expert's read ends on its own, and its end
//!   appends its entry (staging record, bank record) to the layer's landing
//!   list ([`LandingList`]) and answers its choices, the request's last
//!   absent expert the slot too; the launch's threadgroups, parked on the
//!   list's head, convert each entry as it lands, in landing order, while
//!   the request's later reads run, so past the last read byte only the
//!   last record's conversion remains on the path. The device stores each
//!   entry's DONE once its every item is placed, which frees its staging
//!   record; the device past the launch proves the record's bytes whole,
//!   which maps it and ends the landing.
//!
//! The watcher learns a landing once its end is recorded ([`Ended`],
//! `Directory::landed`).

use super::compute::Urgency;
use super::host_records::Grant;
use super::source::Records;
use super::watcher::Step;
use magnitude_kernels::{SlotTable, constants};
use seismic::cpu::repack::RowConversion;

/// What a choice's source word (SOURCE_RECORD) names, by who places the
/// bank's records (`Placer`): -1 no record (the expert absent); where the
/// device does, beside a host tier, tier record t of -2 - t, and a bank
/// record r >= 0 the device's own memory, which the host does not address;
/// where the host does (unified memory, one store), bank record r itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::experts) enum Source {
    /// No record: the expert is absent.
    Absent,
    /// A record of the device's own memory.
    Device,
    /// The host view's record.
    Host(usize),
}

impl Source {
    /// What `word` names where `placer` places. Where the host places, no
    /// word is below -1: the device writes none.
    pub(in crate::experts) fn of(placer: crate::planning::Placer, word: i32) -> Self {
        use crate::planning::Placer;
        match (placer, word) {
            (_, -1) => Self::Absent,
            (Placer::Tier, 0..) => Self::Device,
            (Placer::Tier, ..=-2) => Self::Host((-2 - word) as usize),
            (Placer::Bank, 0..) => Self::Host(word as usize),
            (Placer::Bank, ..=-2) => panic!("source {word} names no record where the bank places its own"),
        }
    }
}

/// The source word of host view record `record` where `placer` places
/// ([`Source::of`]'s inverse).
pub(in crate::experts) fn host_source(placer: crate::planning::Placer, record: usize) -> i32 {
    match placer {
        crate::planning::Placer::Tier => -2 - record as i32,
        crate::planning::Placer::Bank => record as i32,
    }
}

/// The most source bytes one piece of a landing's conversion converts: the
/// unit the host executor's workers share a record's conversion in, the
/// size its conversion was measured at (`seismic::cpu::repack`'s
/// `mma16_record_rate`), whatever size the record was read in.
const CONVERSION_PIECE: usize = 128 << 10;

/// A landing launch's landing list (streaming_words.seismic's: a decode
/// layer's, `routed_land`; a scratch round's of a layer, `routed_round`),
/// as its one host writer appends to it. Each record whose read landed is
/// an entry, in landing order: the record it converts from (LANDING_FROM:
/// a staging or conveyor record, or a scratch slot s as -1 - s), the record
/// it converts into (LANDING_TO), then the head (LANDING_HEAD: the step's
/// low word, and the count landed above it) in one 64-bit release store,
/// so the launch's threadgroups, each parked on the head for the entry it
/// claimed, convert every entry as it lands. A head whose low word is
/// another step's counts none: the first entry of a step is its first
/// head. The device stores an entry's DONE (LANDING_DONE, the step's low
/// word) once it has placed the entry's every item, so its record
/// converted from is read; the host never resets it, comparing it with the
/// step's low word.
pub(in crate::experts) struct LandingList {
    words: super::Words,
    /// What an entry may name: its from record below `from` (or one of
    /// `lent` scratch slots), its to record below `to`.
    bounds: super::ListBounds,
    entries: usize,
    /// The step the head lists and the entries appended for it, one
    /// appender at a time.
    head: std::sync::Mutex<(Step, usize)>,
}

impl LandingList {
    /// `words`, a landing list of every entry its words hold, naming
    /// records within `bounds`.
    pub fn new(words: super::Words, bounds: super::ListBounds) -> Self {
        let entries = (words.len() - constants::LANDING_ENTRIES as usize) / constants::LANDING_ENTRY_WORDS as usize;
        Self { words, bounds, entries, head: std::sync::Mutex::new((0, 0)) }
    }

    /// The word `word` of entry `entry`.
    fn word(entry: usize, word: u64) -> usize {
        (constants::LANDING_ENTRIES + constants::LANDING_ENTRY_WORDS * entry as u64 + word) as usize
    }

    /// Append `(from, to)` for step `sequence`: its entry, then the head
    /// counting it. None where the list lists a later step already: the
    /// device is past this step's launch, which converts nothing more. An
    /// entry past the list, or a pair outside its bounds, is an invariant
    /// violation.
    pub fn append(&self, sequence: Step, from: i32, to: i32) -> Option<usize> {
        let mut head = self.head.lock().expect("a landing list's head");
        if sequence < head.0 {
            return None;
        }
        if sequence > head.0 {
            *head = (sequence, 0);
        }
        let entry = head.1;
        assert!(entry < self.entries, "landing entry {entry} of a list of {}", self.entries);
        assert!(self.bounds.admits(from, to), "landing entry {entry}, ({from}, {to}), outside {:?}", self.bounds);
        self.words.store(Self::word(entry, constants::LANDING_FROM), from);
        self.words.store(Self::word(entry, constants::LANDING_TO), to);
        head.1 += 1;
        let low = super::watcher::low_word(sequence) as u32;
        self.words.store_u64(constants::LANDING_HEAD as usize, u64::from(low) | (head.1 as u64) << 32);
        Some(entry)
    }

    /// Whether the device stored entry `entry`'s DONE for step `sequence`:
    /// it placed the entry's every item, so its record converted from is
    /// read.
    pub fn done(&self, sequence: Step, entry: usize) -> bool {
        self.words.load(Self::word(entry, constants::LANDING_DONE)) == super::watcher::low_word(sequence)
    }

    /// The entries the head counts for step `sequence`, each (from, to), as
    /// the device reads them.
    #[cfg(test)]
    pub fn listed(&self, sequence: Step) -> Vec<(i32, i32)> {
        let head = self.words.load_u64(constants::LANDING_HEAD as usize);
        let count = match head as u32 == super::watcher::low_word(sequence) as u32 {
            true => (head >> 32) as usize,
            false => 0,
        };
        (0..count)
            .map(|entry| (self.words.load(Self::word(entry, constants::LANDING_FROM)), self.words.load(Self::word(entry, constants::LANDING_TO))))
            .collect()
    }

    /// Place entry `entry` as the device does once it converted it: its
    /// DONE, the step's low word.
    #[cfg(test)]
    pub fn place(&self, sequence: Step, entry: usize) {
        self.words.store(Self::word(entry, constants::LANDING_DONE), super::watcher::low_word(sequence));
    }
}

/// Where a [`DeviceLanding`] is: its read running; its entry appended to
/// its landing list, whose DONE frees its staging record, once seen;
/// ended.
enum Device {
    Reading,
    Appended { entry: usize, grant: Grant, done: bool },
    Ended,
}

/// A miss the device converts (`routed_land`, where the bank places): its
/// read's end appends its entry to its layer's landing list
/// ([`Self::append`]) and answers its choices; the launch converts it as
/// it lands and stores the entry's DONE, which frees its staging record
/// ([`Self::converted`]); the device past the launch proves its bytes
/// whole, which maps its record ([`Self::end`]). Until then its record is
/// a landing's, held by the directory as any landing's. One that drops
/// without its end never landed, as a [`Converting`].
pub(in crate::experts) struct DeviceLanding {
    list: std::sync::Arc<LandingList>,
    at: super::directory::Point,
    state: std::sync::Mutex<Device>,
    ended: Ended,
    account: std::sync::Arc<LandingAccount>,
}

impl DeviceLanding {
    /// The device converts a landing of the request at `at` listed in
    /// `list`, its layer's.
    pub fn new(list: std::sync::Arc<LandingList>, at: super::directory::Point, ended: Ended, account: std::sync::Arc<LandingAccount>) -> Self {
        Self { list, at, state: std::sync::Mutex::new(Device::Reading), ended, account }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, Device> {
        self.state.lock().expect("a device landing")
    }

    /// Its read landed whole in staging record `staged`, for the bank
    /// record `grant` takes: append its entry. Whether it did: not once the
    /// list lists a later step, the device past this one's launch, when it
    /// ends not landed.
    pub fn append(&self, grant: Grant, staged: u32) -> bool {
        let to = i32::try_from(grant.record()).expect("bank records fit i32");
        let from = i32::try_from(staged).expect("staging records fit i32");
        let mut state = self.state();
        assert!(matches!(*state, Device::Reading), "a device landing appends once");
        match self.list.append(self.at.sequence, from, to) {
            Some(entry) => {
                *state = Device::Appended { entry, grant, done: false };
                true
            }
            None => {
                *state = Device::Ended;
                drop(state);
                self.ended.set(false);
                false
            }
        }
    }

    /// Whether the device stored its entry's DONE: its staging record is
    /// read, and frees.
    pub fn converted(&self) -> bool {
        match &mut *self.state() {
            Device::Appended { entry, done, .. } => {
                *done = *done || self.list.done(self.at.sequence, *entry);
                *done
            }
            Device::Reading | Device::Ended => false,
        }
    }

    /// Whether the device is past its launch at `past`: then its record is
    /// mapped and its end set landed where the device stored its DONE, else
    /// (the launch stopped before it: its slot failed, or its wait expired)
    /// its end set not landed. One still reading ends once its read does.
    pub fn end(&self, past: super::directory::Point) {
        if self.at >= past {
            return;
        }
        let mut state = self.state();
        let Device::Appended { entry, done, .. } = &*state else {
            return;
        };
        let landed = *done || self.list.done(self.at.sequence, *entry);
        let Device::Appended { grant, .. } = std::mem::replace(&mut *state, Device::Ended) else {
            unreachable!("matched above");
        };
        drop(state);
        if landed {
            grant.map();
            self.account.device.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.ended.set(landed);
    }
}

impl Drop for DeviceLanding {
    fn drop(&mut self) {
        if self.ended.get().is_none() {
            self.ended.set(false);
        }
    }
}

/// How a stored record becomes its resident form: as stored, its bytes, or
/// by its row conversion over records of `rows` rows of `k` values.
#[derive(Clone, Copy, Debug)]
pub(in crate::experts) enum Conversion {
    Stored,
    Rows { conversion: &'static RowConversion, rows: usize, k: usize },
}

/// One routed tensor of a group's host tier: its records, and how a stored
/// record of it converts into one.
#[derive(Clone)]
pub(in crate::experts) struct TierTensor {
    pub records: Records,
    pub conversion: Conversion,
}

impl TierTensor {
    /// The records of `records` converting stored records of `source` into
    /// `resident`, `matrix` rows of values each, `stored` bytes: as stored,
    /// or by the CPU backend's registered row conversion, whose bytes equal
    /// every device repack's; why neither holds.
    pub fn new(records: Records, source: seismic::Element, resident: seismic::Element, matrix: [u64; 2], stored: u64) -> Result<Self, String> {
        let form = format!("{} -> {} [{}, {}]", source.name(), resident.name(), matrix[0], matrix[1]);
        if source == resident {
            return match stored <= records.bytes() {
                true => Ok(Self { records, conversion: Conversion::Stored }),
                false => Err(format!("{form}: a stored record exceeds its tier record")),
            };
        }
        let conversion = seismic::cpu::repack::resolve(source.name(), resident.name())
            .ok_or_else(|| format!("{form}: no host conversion into a row layout"))?;
        let [rows, k] = matrix.map(|extent| extent as usize);
        let converted = conversion
            .stored_row_bytes(k)
            .map(|row| rows.next_multiple_of(conversion.tile()) * row)
            .ok_or_else(|| format!("{form}: the target holds no such row"))?;
        if (rows * conversion.source_row_bytes(k)) as u64 != stored || converted as u64 > records.bytes() {
            return Err(format!("{form}: a stored record does not convert into one tier record"));
        }
        Ok(Self { records, conversion: Conversion::Rows { conversion, rows, k } })
    }

    /// Its record `record` alone, which a landing into it writes: held as
    /// records of their own (`Records::range`), so a landing in flight holds
    /// the tensor's storage only through the record it lands in, which a
    /// shrink keeps in place (`HostRecords::shrink`).
    pub fn landed(&self, record: u32) -> LandedRecord {
        let record = u64::from(record);
        let records = self.records.range(record, record + 1).expect("a committed tier record is records of its own");
        LandedRecord(Self { records, conversion: self.conversion })
    }
}

/// The one tier record of a tensor a landing writes ([`TierTensor::landed`]).
pub(in crate::experts) struct LandedRecord(TierTensor);

impl LandedRecord {
    /// The pieces converting a stored record into a tier record: (stored
    /// bytes, tier record bytes) of each, every one disjoint from the
    /// others, together covering the record. As stored, one piece; by a
    /// row conversion, pieces of whole stored tiles of at most
    /// [`CONVERSION_PIECE`] source bytes.
    fn pieces(&self, stored: u64) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
        match self.0.conversion {
            Conversion::Stored => {
                let bytes = usize::try_from(stored).expect("a stored record the host addresses");
                vec![(0..bytes, 0..bytes)]
            }
            Conversion::Rows { conversion, rows, k } => {
                let row = conversion.source_row_bytes(k) * conversion.tile();
                let tiles = std::num::NonZeroUsize::new((CONVERSION_PIECE / row).max(1))
                    .expect("at least one tile a piece");
                conversion.pieces(k, rows, tiles).expect("a tier tensor's rows convert").collect()
            }
        }
    }

    /// Convert piece `(from, to)` of stored record `stored` into the
    /// record.
    ///
    /// # Safety
    ///
    /// No read lands in `stored` and no device work reads or writes the
    /// record until its landing ends, and no other piece of it writes `to`.
    unsafe fn convert(&self, stored: &[u8], (from, to): (std::ops::Range<usize>, std::ops::Range<usize>)) {
        let LandedRecord(TierTensor { records, conversion }) = self;
        // SAFETY: the caller's; the pieces' ranges are disjoint, so this one
        // alone writes `to`.
        assert!(to.end as u64 <= records.bytes(), "a piece past its tier record");
        let into = unsafe { std::slice::from_raw_parts_mut(records.address(0).as_ptr().add(to.start), to.len()) };
        match *conversion {
            Conversion::Stored => into.copy_from_slice(&stored[from]),
            Conversion::Rows { conversion, k, .. } => {
                conversion.piece(k, &stored[from], into).expect("a tier tensor's pieces convert")
            }
        }
    }
}

/// One expert's landing, its claim's `grant` (its expert and tier record):
/// from each routed tensor's staging record (its records, the record and
/// its stored bytes) into the grant's tier record of the tensor (that
/// record alone, [`TierTensor::landed`]), then
/// mapped by the grant ([`Converting`], [`Grant::map`]), from the step its
/// claim names: where the landing is a parked step's, the first step that
/// has not resolved the layer, which the device cannot pass before the
/// landing answers; else the first step not yet begun.
pub(in crate::experts) struct Landing {
    pub tensors: Vec<(Records, u64, u64, LandedRecord)>,
    pub grant: Grant,
}

/// Whether a landing has ended and how: unset while it converts, then
/// whether it landed (its record mapped) or not (its read failed, or its
/// conversion never ended). Set once, before any answer it publishes, so a
/// device past its answer proves it set; setting it is one of its load's
/// events (`Events`).
#[derive(Clone)]
pub(in crate::experts) struct Ended {
    landed: std::sync::Arc<std::sync::OnceLock<bool>>,
    events: std::sync::Arc<super::events::Events>,
}

impl Ended {
    /// A landing's end, unset, which is an event of `events` once set.
    pub fn new(events: &std::sync::Arc<super::events::Events>) -> Self {
        Self { landed: Default::default(), events: events.clone() }
    }

    /// How it ended, none while it runs.
    pub fn get(&self) -> Option<bool> {
        self.landed.get().copied()
    }

    /// It ended, landed or not.
    pub fn set(&self, landed: bool) {
        self.landed.set(landed).expect("a landing ends once");
        self.events.notify();
    }
}

/// A landing as the executor runs it (`compute::Job`): every routed
/// tensor's pieces, then its end: the record mapped, its end set landed,
/// then `then` (its answers, as its submitter chose them). One that drops
/// without its end (a piece panicked, or the executor stopped first) ends
/// not landed: every landing ends.
pub(in crate::experts) struct Converting {
    landing: Landing,
    ended: Ended,
    /// Each piece: its tensor, and its (stored, tier) byte ranges.
    pieces: Vec<(usize, (std::ops::Range<usize>, std::ops::Range<usize>))>,
    then: std::sync::Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// How urgently its pieces run (`compute::Job::want`): a step's
    /// miss's, or a hint's until a request names its expert.
    urgency: std::sync::Arc<Urgency>,
    /// The load's landings' account; when it was submitted and when a
    /// worker began its first piece; its pieces' worker time so far, in
    /// nanoseconds.
    account: std::sync::Arc<LandingAccount>,
    submitted: std::time::Instant,
    began: std::sync::OnceLock<std::time::Instant>,
    worked: std::sync::atomic::AtomicU64,
}

/// One kind of landing's terms over a load, each summed in nanoseconds
/// over the landings it is measured on: its read (from its request's
/// notice to its read's end; only a landing a request named when it was
/// read has one), then on the executor its wait (from its submission to
/// its first piece begun), its conversion (from then to its end, the path
/// a waiting choice sees), and its pieces' worker time (summed over the
/// workers that ran them at once: what the conversion costs the host, not
/// how long it takes).
#[derive(Default)]
pub(in crate::experts) struct LandingTerms {
    landings: std::sync::atomic::AtomicU64,
    reads: std::sync::atomic::AtomicU64,
    read: std::sync::atomic::AtomicU64,
    queued: std::sync::atomic::AtomicU64,
    converting: std::sync::atomic::AtomicU64,
    worked: std::sync::atomic::AtomicU64,
}

/// The host's landings' account over a load, by their urgency as they
/// ended (waited: a step waited for it; else a hint's no request named),
/// the landings the device converted, and the landings that waited for a
/// record past grace and how long; logged as the watcher stops.
#[derive(Default)]
pub(in crate::experts) struct LandingAccount {
    demand: LandingTerms,
    hinted: LandingTerms,
    pub waits: std::sync::atomic::AtomicU64,
    pub waited: std::sync::atomic::AtomicU64,
    /// The landings the device converted (`routed_land`).
    pub device: std::sync::atomic::AtomicU64,
}

impl LandingAccount {
    pub fn add(counter: &std::sync::atomic::AtomicU64, elapsed: std::time::Duration) {
        counter.fetch_add(elapsed.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// A step's miss's read, from its request's notice to its end.
    pub fn read(&self, elapsed: std::time::Duration) {
        Self::add(&self.demand.read, elapsed);
        self.demand.reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// One line of the account: each kind's landings, and each term's mean
    /// over the landings it is measured on.
    pub fn line(&self) -> String {
        let load = |counter: &std::sync::atomic::AtomicU64| counter.load(std::sync::atomic::Ordering::Relaxed);
        let ms = |nanos: u64, over: u64| nanos as f64 / 1e6 / over.max(1) as f64;
        let terms = |name: &str, terms: &LandingTerms| {
            let landings = load(&terms.landings);
            format!(
                "{name} {landings}: read {:.3} ms (mean of {}), queued {:.3} ms, converting {:.3} ms, worker time {:.3} ms",
                ms(load(&terms.read), load(&terms.reads)),
                load(&terms.reads),
                ms(load(&terms.queued), landings),
                ms(load(&terms.converting), landings),
                ms(load(&terms.worked), landings),
            )
        };
        format!(
            "landings, each term a landing's mean: {}; {}; on the device {}; waits for a record {} ({:.1} ms in all)",
            terms("demand", &self.demand),
            terms("hinted", &self.hinted),
            load(&self.device),
            load(&self.waits),
            ms(load(&self.waited), 1),
        )
    }
}

impl Converting {
    /// `landing` converted at `urgency`, mapped, `ended` set landed, then
    /// `then`.
    ///
    /// # Safety
    ///
    /// The stored records are whole and no read lands in them, and no
    /// device work reads or writes the tier record (it is ready: unmapped
    /// past every step that could resolve it), until it ends.
    pub unsafe fn new(
        landing: Landing,
        urgency: std::sync::Arc<Urgency>,
        account: std::sync::Arc<LandingAccount>,
        ended: Ended,
        then: impl FnOnce() + Send + 'static,
    ) -> Self {
        let pieces = landing
            .tensors
            .iter()
            .enumerate()
            .flat_map(|(index, (_, _, bytes, tensor))| tensor.pieces(*bytes).into_iter().map(move |piece| (index, piece)))
            .collect();
        Self {
            landing,
            ended,
            pieces,
            then: std::sync::Mutex::new(Some(Box::new(then))),
            urgency,
            account,
            submitted: std::time::Instant::now(),
            began: std::sync::OnceLock::new(),
            worked: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl Drop for Converting {
    /// A landing dropped before its end never landed: its record mapped
    /// nothing, and its end says so.
    fn drop(&mut self) {
        if self.ended.get().is_none() {
            self.ended.set(false);
        }
    }
}

// SAFETY: the workers that run its pieces share it: each piece reads its
// stored record and writes its own range of the tier record, disjoint from
// every other piece's (`TierTensor::pieces`), and the end runs alone, after
// every piece.
unsafe impl Sync for Converting {}

impl super::compute::Job for Converting {
    fn pieces(&self) -> usize {
        self.pieces.len()
    }

    fn piece(&self, index: usize) {
        let (tensor, piece) = self.pieces[index].clone();
        let (staging, staged, bytes, tier) = &self.landing.tensors[tensor];
        // SAFETY: `new`'s: the stored record is whole and nothing lands in
        // it, nothing reads the tier record, and this piece alone writes
        // its range.
        let started = std::time::Instant::now();
        self.began.get_or_init(|| started);
        unsafe { tier.convert(staging.record(*staged, *bytes), piece) };
        LandingAccount::add(&self.worked, started.elapsed());
    }

    /// Every piece is converted, so the record's bytes are whole: map it
    /// ([`Grant::map`]), then answer.
    fn end(&self) {
        let terms = match self.urgency.is_waited() {
            true => &self.account.demand,
            false => &self.account.hinted,
        };
        let began = *self.began.get().expect("a landing's end follows its pieces");
        LandingAccount::add(&terms.queued, began - self.submitted);
        LandingAccount::add(&terms.converting, began.elapsed());
        terms.worked.fetch_add(self.worked.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
        terms.landings.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.landing.grant.map();
        self.ended.set(true);
        if let Some(then) = self.then.lock().expect("a landing's end").take() {
            then();
        }
    }

    fn want(&self) -> seismic_io::Want {
        self.urgency.get()
    }
}

/// The choices of a layer's file request one landing answers: their indices
/// in the request of step `sequence`, in its slot of the layer's file table
/// of the exchange.
#[derive(Clone)]
pub(in crate::experts) struct Answer {
    /// The layer whose request it answers.
    pub layer: usize,
    pub file: super::ExchangeTable,
    pub sequence: Step,
    pub choices: Vec<usize>,
    /// The request's absent experts not yet answered, shared by each of
    /// their answers: the last one answers the slot.
    pub pending: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Who placed what it answers (`Extents::placer`).
    pub placer: crate::planning::Placer,
    /// The watcher's steps, whose ownership of the step's slot an answer is
    /// written under (`Steps::while_owned`).
    pub steps: std::sync::Arc<super::watcher::Steps>,
    /// The load's parks, which an answer settles (a landing launch's park
    /// opens at the slot's answer where parks are submission boundaries).
    pub parks: std::sync::Arc<super::Parks>,
}

impl Answer {
    /// Answer each choice with the source its expert landed in (none: its
    /// read failed): its entry's FILE_RECORD (where the device
    /// places, tier record t as -2 - t; where the host does, bank record r
    /// itself; -1 for none); and, as the request's last absent expert
    /// answered, the slot, after every record word. It writes while the step
    /// owns its slot: a landing ending after its step did writes nothing, so
    /// no later step's answer has a second writer.
    pub fn publish(&self, record: Option<u32>) {
        let source = record.map_or(-1, |record| host_source(self.placer, record as usize));
        let sequence = self.sequence;
        self.steps.while_owned(sequence, |_| {
            for &choice in &self.choices {
                self.file.store(sequence, 0, SlotTable::file_entry(choice as u64) + constants::FILE_RECORD, source);
            }
            let slot = self.pending.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1;
            if slot {
                self.file.answer(sequence, 0, false);
            }
            seismic_io::timeline::note(seismic_io::timeline::Kind::Noted("answer"), sequence as u64, (self.layer as i64, record.is_some() as i64, slot as i64));
        });
        self.parks.settle();
    }
}
