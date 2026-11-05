use super::file::SourceFile;
use super::platform::{evict, uncached, volume_key};
use super::probe::{Urgent, VolumeLines, LEAST_READ, WIDEST_WINDOW};
use super::Priority;
use crate::Error;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// How a reader reads the volume its file lies on: as the volume's probe
/// chose for its urgent request ([`VolumeChoice`]), within what the volume's
/// readers share.
#[derive(Clone, Debug)]
pub(super) struct Volume {
    pub(super) key: VolumeKey,
    pub(super) choice: VolumeChoice,
    /// The volume's own bound on the bytes in flight a less urgent piece
    /// joins, whichever reader submits it: the narrowest window of its live
    /// urgent readers ([`Bound`]).
    pub(super) background: Arc<Bound>,
    /// What the volume reads and how long it is busy ([`Delivery`]), shared
    /// by every reader of it.
    pub(super) delivery: Arc<Delivery>,
}

/// A volume's identity within the process: its device (`st_dev`) on unix,
/// its serial number on Windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct VolumeKey(pub(super) u64);

/// How a volume is read for an urgent request, as its probe chose ("A
/// volume's window"): what the measurement job keeps ([`probe_volume`]),
/// which a load adopts ([`adopt_volume`]). The pool reads it in reads of at
/// most `read` bytes, a demand job's in reads of at most `demand_read`, the
/// size its lone record reads soonest at; a demand piece joins the bytes in
/// flight up to `demand`, the full rate's window, and any other piece only
/// up to `window`, the urgent request's, so a demand read waits behind no
/// more of the less urgent reads than its point of least `T` allows; its
/// reads at that rate occupy `cores` of the host's processors (the cores its
/// threads ran, in seconds a second).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeChoice {
    pub window: u64,
    pub demand: u64,
    pub read: u64,
    pub demand_read: u64,
    pub cores: f64,
}

impl VolumeChoice {
    /// The rule a choice is made by, a term of a stored choice's key, so a
    /// choice made by another is probed again: 3, a demand window the widest
    /// at the full rate and a lone request timed through the read engine
    /// (the first rule took the narrowest window and timed lone reads by
    /// threads).
    pub const SCHEMA: u64 = 3;

    /// The rule before [`SCHEMA`](Self::SCHEMA), whose stored choice a load
    /// reads while the job probes the current one: none, the first rule,
    /// whose records carry no schema term. Named beside the current rule so
    /// a new rule names its predecessor with it.
    pub const EARLIER_SCHEMA: Option<u64> = None;

    /// How a volume is read before any probe of it: at the least read size
    /// and the widest window, every piece at the most any device measured
    /// needs (the declared bounds of the search), which takes no measurement.
    const PENDING: Self = Self { window: WIDEST_WINDOW, demand: WIDEST_WINDOW, read: LEAST_READ, demand_read: LEAST_READ, cores: 0. };
}

/// Each volume's lines, by volume.
fn volume_lines(file: &File) -> Result<(VolumeKey, Arc<Mutex<VolumeLines>>), Error> {
    static VOLUMES: OnceLock<Mutex<HashMap<VolumeKey, Arc<Mutex<VolumeLines>>>>> = OnceLock::new();
    let key = volume_key(file)?;
    Ok((key, VOLUMES.get_or_init(Default::default).lock().expect("volumes").entry(key).or_default().clone()))
}

/// Read `source`'s volume for `urgent` as `choice` from now on.
pub fn adopt_volume(source: &crate::FileSource, urgent: Option<&Urgent>, choice: VolumeChoice) -> Result<(), Error> {
    let (key, lines) = volume_lines(source.file())?;
    let mut lines = lines.lock().expect("a volume's lines");
    let volume = lines.volume(key, choice);
    lines.chosen.insert(urgent.cloned(), volume);
    Ok(())
}

/// Probe `source`'s volume for `urgent`, reading it and evicting the file's
/// pages to time its storage: run only where nothing waits on it (the
/// measurement job, a tool), and read as it chose from now on.
pub fn probe_volume(source: &crate::FileSource, urgent: Option<&Urgent>) -> Result<VolumeChoice, Error> {
    let (key, lines) = volume_lines(source.file())?;
    // Held across the probe: no probe shares the storage with another of
    // the volume, and other volumes' readers wait on none of it.
    let mut lines = lines.lock().expect("a volume's lines");
    let probed = Arc::new(SourceFile { path: source.path().into(), size: source.size(), through: uncached(source.file())? });
    evict(source.file(), source.size())?;
    let choice = lines.choose(key, &probed, urgent)?;
    let volume = lines.volume(key, choice);
    lines.chosen.insert(urgent.cloned(), volume);
    Ok(choice)
}

impl Volume {
    /// How `file` is read for `urgent`: as its volume's choice for it, or
    /// [`VolumeChoice::PENDING`]. A reader measures nothing.
    pub(super) fn of(file: &File, urgent: Option<&Urgent>) -> Result<Self, Error> {
        let (key, lines) = volume_lines(file)?;
        let lines = lines.lock().expect("a volume's lines");
        Ok(lines.chosen.get(&urgent.cloned()).cloned().unwrap_or_else(|| lines.volume(key, VolumeChoice::PENDING)))
    }

    /// The workers that hold the volume's widest window in flight in pieces
    /// of the least size: as many as its pieces in flight can be (a range
    /// smaller than a read is a piece of its own).
    pub(super) fn workers(&self) -> usize {
        self.choice.demand.max(self.choice.window).div_ceil(LEAST_READ) as usize
    }

    /// The bytes in flight a piece of a job at `priority` may join: a
    /// demand piece its reader's full-rate window, any other its reader's
    /// window within the volume's own bound, so a load's import beside a
    /// serving reader puts no more ahead of a decode request than its `T`
    /// allows.
    pub(super) fn window_for(&self, priority: Priority) -> u64 {
        match priority {
            Priority::Demand => self.choice.demand,
            Priority::Prefetch | Priority::Load => self.choice.window.min(self.background.least()),
        }
    }
}

/// The cores the read engine's workers occupy reading the volumes `sources`
/// lie on for `urgent`, each at its full rate at once: each distinct
/// volume's, as its probe measured them ("A volume's window").
pub fn read_cores<'a>(sources: impl IntoIterator<Item = &'a crate::FileSource>, urgent: &Urgent) -> Result<f64, Error> {
    let mut volumes = Vec::<Volume>::new();
    for source in sources {
        let volume = Volume::of(source.file(), Some(urgent))?;
        if !volumes.iter().any(|known| known.key == volume.key) {
            volumes.push(volume);
        }
    }
    Ok(volumes.iter().map(|volume| volume.choice.cores).sum())
}

/// What a volume read: the bytes of every piece that ended, whatever its
/// job (demand, hint, fill), and the seconds it held a piece in flight (its
/// busy time). Its bytes over its busy time are the volume's own rate as it
/// serves now (a burst drained or not), and its busy time over a span the
/// share of the span it was busy: what the volume did, which no job's time
/// gives (a job's time from its submission holds the pieces queued and in
/// flight ahead of it, which a hint's own bytes can be).
#[derive(Debug, Default)]
pub struct Delivery(Mutex<Delivered>);

#[derive(Debug, Default)]
struct Delivered {
    bytes: u64,
    seconds: f64,
    /// Since when it holds a piece in flight, while it does: the busy time
    /// not yet counted.
    busy: Option<Instant>,
}

impl Delivery {
    /// The volume took a piece at `now` with none in flight: busy from now.
    pub(super) fn began(&self, now: Instant) {
        let mut delivered = self.0.lock().expect("a volume's delivery");
        assert!(delivered.busy.is_none(), "a volume begins busy once until idle");
        delivered.busy = Some(now);
    }

    /// A piece of `bytes` ended at `now`, the volume holding none in flight
    /// after it (`idle`) or some.
    pub(super) fn ended(&self, bytes: u64, idle: bool, now: Instant) {
        let mut delivered = self.0.lock().expect("a volume's delivery");
        delivered.bytes += bytes;
        if idle {
            let since = delivered.busy.take().expect("a volume with a piece in flight is busy");
            delivered.seconds += now.saturating_duration_since(since).as_secs_f64();
        }
    }

    /// The bytes read and the busy seconds since the last take, the busy
    /// time running now counted to now.
    pub fn take(&self) -> (u64, f64) {
        self.take_at(Instant::now())
    }

    pub(super) fn take_at(&self, now: Instant) -> (u64, f64) {
        let mut delivered = self.0.lock().expect("a volume's delivery");
        if let Some(since) = delivered.busy {
            delivered.seconds += now.saturating_duration_since(since).as_secs_f64();
            delivered.busy = Some(now);
        }
        (std::mem::take(&mut delivered.bytes), std::mem::take(&mut delivered.seconds))
    }
}

/// A volume's bound on its less urgent bytes in flight: the narrowest
/// window of its live urgent readers, none (`u64::MAX`) while it has none.
/// Each urgent reader holds its window in it from when it is opened until
/// its last clone drops ([`Registered`]), so a load unloaded throttles no
/// later one.
#[derive(Debug)]
pub(super) struct Bound {
    /// The live readers' windows, each with how many hold it.
    live: Mutex<BTreeMap<u64, usize>>,
    /// The least of them, read without the lock by every piece taken.
    least: std::sync::atomic::AtomicU64,
}

impl Default for Bound {
    fn default() -> Self {
        Self { live: Mutex::default(), least: std::sync::atomic::AtomicU64::new(u64::MAX) }
    }
}

impl Bound {
    fn least(&self) -> u64 {
        self.least.load(std::sync::atomic::Ordering::Acquire)
    }

    /// `change` the live windows, then their least.
    fn update(&self, change: impl FnOnce(&mut BTreeMap<u64, usize>)) {
        let mut live = self.live.lock().expect("a volume's bound");
        change(&mut live);
        let least = live.keys().next().copied().unwrap_or(u64::MAX);
        self.least.store(least, std::sync::atomic::Ordering::Release);
    }
}

/// An urgent reader's window, held in its volume's bound until it drops.
#[derive(Debug)]
pub(super) struct Registered {
    bound: Arc<Bound>,
    window: u64,
}

impl Registered {
    pub(super) fn new(bound: Arc<Bound>, window: u64) -> Self {
        bound.update(|live| *live.entry(window).or_default() += 1);
        Self { bound, window }
    }
}

impl Drop for Registered {
    fn drop(&mut self) {
        let window = self.window;
        self.bound.update(|live| {
            let count = live.get_mut(&window).expect("a registered window is live");
            *count -= 1;
            if *count == 0 {
                live.remove(&window);
            }
        });
    }
}
