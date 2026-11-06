use super::file::SourceFile;
use super::platform::{evict, uncached, volume_key};
use super::probe::{Urgent, VolumeLines, LEAST_READ, WIDEST_WINDOW};
use super::OpenFile;
use crate::{Volume, Error, Priority};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::sync::{Arc, Mutex, OnceLock};

/// How a reader reads the volume its file lies on: as the volume's probe
/// chose for its urgent request ([`VolumeChoice`]), within what the volume's
/// readers share.
#[derive(Clone, Debug)]
pub(crate) struct VolumeReads {
    pub(crate) id: Volume,
    pub(crate) choice: VolumeChoice,
    /// The volume's own bound on the bytes in flight a less urgent piece
    /// joins, whichever reader submits it: the narrowest window of its live
    /// urgent readers ([`Bound`]).
    pub(crate) background: Arc<Bound>,
}

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
fn volume_lines(file: &File) -> Result<(Volume, Arc<Mutex<VolumeLines>>), Error> {
    static VOLUMES: OnceLock<Mutex<HashMap<Volume, Arc<Mutex<VolumeLines>>>>> = OnceLock::new();
    let key = volume_key(file)?;
    Ok((key, VOLUMES.get_or_init(Default::default).lock().expect("volumes").entry(key).or_default().clone()))
}

/// Read `source`'s volume for `urgent` as `choice` from now on.
pub fn adopt_volume(source: &impl AsRef<OpenFile>, urgent: Option<&Urgent>, choice: VolumeChoice) -> Result<(), Error> {
    let source = source.as_ref();
    let (key, lines) = volume_lines(source.file())?;
    let mut lines = lines.lock().expect("a volume's lines");
    let volume = lines.volume(key, choice);
    lines.chosen.insert(urgent.cloned(), volume);
    Ok(())
}

/// Probe `source`'s volume for `urgent`, reading it and evicting the file's
/// pages to time its storage: run only where nothing waits on it (the
/// measurement job, a tool), and read as it chose from now on.
pub fn probe_volume(source: &impl AsRef<OpenFile>, urgent: Option<&Urgent>) -> Result<VolumeChoice, Error> {
    let source = source.as_ref();
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

impl VolumeReads {
    /// How `file` is read for `urgent`: as its volume's choice for it, or
    /// [`VolumeChoice::PENDING`]. A reader measures nothing.
    pub(crate) fn of(file: &File, urgent: Option<&Urgent>) -> Result<Self, Error> {
        let (key, lines) = volume_lines(file)?;
        let lines = lines.lock().expect("a volume's lines");
        Ok(lines.chosen.get(&urgent.cloned()).cloned().unwrap_or_else(|| lines.volume(key, VolumeChoice::PENDING)))
    }

    /// The workers that hold the volume's widest window in flight in pieces
    /// of the least size: as many as its pieces in flight can be (a range
    /// smaller than a read is a piece of its own).
    pub(crate) fn workers(&self) -> usize {
        self.choice.demand.max(self.choice.window).div_ceil(LEAST_READ) as usize
    }

    /// The bytes in flight a piece of a job at `priority` may join: a
    /// demand piece its reader's full-rate window, any other its reader's
    /// window within the volume's own bound, so a load's import beside a
    /// serving reader puts no more ahead of a decode request than its `T`
    /// allows.
    pub(crate) fn window_for(&self, priority: Priority) -> u64 {
        match priority {
            Priority::Demand => self.choice.demand,
            Priority::Prefetch | Priority::Load => self.choice.window.min(self.background.least()),
        }
    }
}

/// The cores the read engine's workers occupy reading the volumes `sources`
/// lie on for `urgent`, each at its full rate at once: each distinct
/// volume's, as its probe measured them ("A volume's window").
pub fn read_cores<'a, S: AsRef<OpenFile> + 'a>(sources: impl IntoIterator<Item = &'a S>, urgent: &Urgent) -> Result<f64, Error> {
    let mut volumes = Vec::<VolumeReads>::new();
    for source in sources {
        let volume = VolumeReads::of(source.as_ref().file(), Some(urgent))?;
        if !volumes.iter().any(|known| known.id == volume.id) {
            volumes.push(volume);
        }
    }
    Ok(volumes.iter().map(|volume| volume.choice.cores).sum())
}

/// A volume's bound on its less urgent bytes in flight: the narrowest
/// window of its live urgent readers, none (`u64::MAX`) while it has none.
/// Each urgent reader holds its window in it from when it is opened until
/// its last clone drops ([`Registered`]), so a load unloaded throttles no
/// later one.
#[derive(Debug)]
pub(crate) struct Bound {
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
pub(crate) struct Registered {
    bound: Arc<Bound>,
    window: u64,
}

impl Registered {
    pub(crate) fn new(bound: Arc<Bound>, window: u64) -> Self {
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
