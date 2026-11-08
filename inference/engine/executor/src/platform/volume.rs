//! A volume's reserve: what a writer to it (a model download) leaves free,
//! so that the OS and other programs keep room for their swap, caches and
//! logs.
//!
//! The reserve is the volume's motion's margin ([`StoreMotion::margin`]) over
//! the writer's reaction time, and nothing else: neither macOS nor Linux
//! reports a line of its own on a volume. Available space (`f_bavail`)
//! already leaves out what the filesystem reserves (ext4's reserved blocks),
//! and APFS's low-disk levels are notifications (`VQ_LOWDISK`,
//! `VQ_VERYLOWDISK`) that report no threshold.

use super::policy::{MotionMeter, READING_INTERVAL, StoreMotion, machine_rate};
use seismic::ObservationError;
use std::io::Write as _;
use seismic::VolumeId;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The bytes the write-rate probe writes and syncs.
const WRITE_PROBE_BYTES: usize = 64 << 20;

/// The reserve of the volume one writer writes, and the space it leaves
/// that writer.
///
/// The writer asks for its space before each write, giving every byte it
/// has written to the volume so far. The volume is read again once a reading
/// interval has passed since the last reading; between readings, the space
/// is the last reading's less what the writer wrote since. A writer refused
/// writes nothing more, so its reaction to a reading is the interval alone,
/// and the margin is the volume's motion over it: the fluctuation, plus
/// what other programs write at their fastest rate within one interval.
///
/// Other programs' writes are judged on available space plus the writer's
/// own written bytes, as [`MotionMeter`] judges a host's own holding: the
/// writer's writes lower the one and raise the other by the same bytes.
#[derive(Debug)]
pub struct VolumeReserve {
    meter: MotionMeter,
    last: Option<VolumeReading>,
}

#[derive(Clone, Copy, Debug)]
struct VolumeReading {
    at: Instant,
    space: u64,
    written: u64,
}

impl VolumeReserve {
    /// A reserve before any reading of the volume: `prior`, the volume's
    /// sequential write rate ([`StoreMotion::of_volume`]).
    pub fn new(prior: StoreMotion) -> Self {
        Self { meter: MotionMeter::new(prior), last: None }
    }

    /// The bytes the volume keeps free of its writer.
    pub fn margin(&self) -> u64 {
        self.meter.motion().margin(READING_INTERVAL, Duration::ZERO, None)
    }

    /// The bytes the writer may still write at `at`, having written
    /// `written` so far; `available` reads the volume's available space, and
    /// is called once a reading interval has passed since the last reading.
    pub fn space<E>(&mut self, at: Instant, written: u64, available: impl FnOnce() -> Result<u64, E>) -> Result<u64, E> {
        if let Some(last) = self.last.filter(|last| at.duration_since(last.at) < READING_INTERVAL) {
            return Ok(last.space.saturating_sub(written.saturating_sub(last.written)));
        }
        let available = available()?;
        self.meter.read(at, available, Some(written));
        let space = available.saturating_sub(self.margin());
        self.last = Some(VolumeReading { at, space, written });
        Ok(space)
    }
}

impl StoreMotion {
    /// The motion of the volume holding `path` before any reading of it: no
    /// measured fluctuation, and the volume's sequential write rate (bytes
    /// written to a new file there and synced, timed, then the file
    /// removed), the fastest one program writes it at.
    ///
    /// A generic figure of the volume, measured once: where `cache` names a
    /// kernel cache and the volume reports a UUID, the rate that cache's
    /// machine figures keep for it, measured and stored there when they
    /// keep none; elsewhere, measured.
    pub fn of_volume(path: &Path, volume: &VolumeId, cache: Option<&Path>) -> Result<Self, ObservationError> {
        let figure = cache.zip(volume.uuid.as_ref()).map(|(cache, uuid)| (cache, format!("volume-write:{uuid}")));
        Ok(Self::prior(machine_rate(figure, || write_rate(path).map_err(|error| failed(path, error)))?))
    }
}

fn failed(path: &Path, error: std::io::Error) -> ObservationError {
    ObservationError::Failed(format!("{}: {error}", path.display()))
}

/// The rate the volume holding the directory `path` takes a sequential
/// write at, synced, in bytes a second.
fn write_rate(path: &Path) -> std::io::Result<f64> {
    let probe = path.join(format!(".volume-write-probe-{}", std::process::id()));
    let block = vec![0xA5_u8; 8 << 20];
    let timed = (|| -> std::io::Result<Duration> {
        let mut file = std::fs::File::create(&probe)?;
        let started = Instant::now();
        for _ in 0..WRITE_PROBE_BYTES / block.len() {
            file.write_all(&block)?;
        }
        file.sync_all()?;
        Ok(started.elapsed())
    })();
    let removed = std::fs::remove_file(&probe);
    let seconds = timed?.as_secs_f64();
    removed?;
    Ok(WRITE_PROBE_BYTES as f64 / seconds.max(f64::MIN_POSITIVE))
}

/// The reserves of the volumes this process writes, one per volume: the one
/// figure its writers' space ([`Self::space`]) and its fit
/// ([`super::Fit::with_volumes`]) read of each.
#[derive(Debug)]
pub struct VolumeReserves {
    /// The kernel cache whose machine figures keep each volume's write rate.
    cache: Option<PathBuf>,
    volumes: Mutex<BTreeMap<VolumeId, VolumeReserve>>,
}

impl VolumeReserves {
    pub fn new(cache: Option<PathBuf>) -> Self {
        Self { cache, volumes: Mutex::new(BTreeMap::new()) }
    }

    /// The volume holding the directory `path`, written by this process:
    /// its reserve kept from now on, its prior the volume's write rate as
    /// the kernel cache keeps it ([`StoreMotion::of_volume`]).
    pub fn register(&self, path: &Path) -> Result<VolumeId, ObservationError> {
        let volume = seismic::volume_identity(path).map_err(|error| failed(path, error))?;
        let mut volumes = self.volumes();
        if !volumes.contains_key(&volume) {
            let prior = StoreMotion::of_volume(path, &volume, self.cache.as_deref())?;
            volumes.insert(volume.clone(), VolumeReserve::new(prior));
        }
        Ok(volume)
    }

    /// Volume `volume`'s margin; none where this process writes it not.
    pub fn margin(&self, volume: &VolumeId) -> Option<u64> {
        self.volumes().get(volume).map(VolumeReserve::margin)
    }

    /// The space a writer to the directory `path` may still write, given
    /// every byte it has written there so far: its volume's
    /// [`VolumeReserve`], registered at the first call.
    pub fn space(self: Arc<Self>, path: PathBuf) -> impl Fn(u64) -> Result<u64, ObservationError> + Send + Sync + 'static {
        let registered = OnceLock::<VolumeId>::new();
        move |written| {
            let volume = match registered.get() {
                Some(volume) => volume,
                None => {
                    let volume = self.register(&path)?;
                    registered.get_or_init(|| volume)
                }
            };
            let mut volumes = self.volumes();
            let reserve = volumes.get_mut(volume).expect("a registered volume keeps its reserve");
            reserve.space(Instant::now(), written, || seismic::volume_available(&path).map(|available| available.value))
        }
    }

    fn volumes(&self) -> std::sync::MutexGuard<'_, BTreeMap<VolumeId, VolumeReserve>> {
        self.volumes.lock().expect("a volume reserve's reader does not panic")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn unread() -> Result<u64, ObservationError> {
        panic!("the volume is read once a reading interval has passed")
    }

    /// Before any reading, the margin is the volume's write rate over one
    /// reading interval: no fixed figure, and none for a faster volume the
    /// same as a slower one's.
    #[test]
    fn a_volume_keeps_its_write_rate_over_a_reading_interval() {
        let start = Instant::now();
        for rate in [1e9, 3e9] {
            let mut reserve = VolumeReserve::new(StoreMotion::new(0, rate));
            let margin = (rate * READING_INTERVAL.as_secs_f64()) as u64;
            assert_eq!(reserve.margin(), margin);
            assert_eq!(reserve.space(start, 0, || Ok::<_, ObservationError>(100 * GB)), Ok(100 * GB - margin));
        }
    }

    /// The writer's own writes are no other program's: available space
    /// falling by exactly what it wrote measures no take, and between
    /// readings its space falls by what it writes.
    #[test]
    fn a_writers_own_writes_are_not_takes() {
        let start = Instant::now();
        let mut reserve = VolumeReserve::new(StoreMotion::new(0, 1e9));
        let margin = reserve.margin();
        for second in 0..5 {
            let at = start + Duration::from_secs(second);
            let written = second * 10 * GB;
            assert_eq!(reserve.space(at, written, || Ok::<_, ObservationError>(100 * GB - written)), Ok(100 * GB - written - margin));
        }
        assert_eq!(reserve.margin(), margin);
        let between = start + Duration::from_secs(4) + READING_INTERVAL / 2;
        assert_eq!(reserve.space(between, 41 * GB, unread), Ok(59 * GB - margin));
        // Another program's write is a take at its rate.
        let at = start + Duration::from_secs(5);
        assert_eq!(reserve.space(at, 50 * GB, || Ok::<_, ObservationError>(45 * GB)), Ok(45 * GB - 500_000_000));
        assert_eq!(reserve.margin(), 500_000_000);
    }

    /// The volume's write rate is a positive rate, measured once per volume
    /// with a cache: a later reader reads it rather than measuring again.
    #[test]
    fn a_volume_writes_at_a_measured_rate() {
        let directory = std::env::temp_dir().join(format!("volume-write-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let identity = seismic::volume_identity(&directory).unwrap();
        let measured = StoreMotion::of_volume(&directory, &identity, None).unwrap();
        assert!(measured.take_rate() > 0.0 && measured.take_rate().is_finite());
        assert_eq!(measured.fluctuation(), 0);
        if let Some(uuid) = &identity.uuid {
            let cache = directory.join("kernels");
            let stored = machine_rate(Some((cache.as_path(), format!("volume-write:{uuid}"))), || Ok::<_, std::convert::Infallible>(7e9));
            assert_eq!(stored, Ok(7e9));
            assert_eq!(StoreMotion::of_volume(&directory, &identity, Some(&cache)).unwrap().take_rate(), 7e9);
            // The fit and the writer read one reserve of the volume.
            let reserves = Arc::new(VolumeReserves::new(Some(cache)));
            assert_eq!(reserves.margin(&identity), None);
            let space = Arc::clone(&reserves).space(directory.clone());
            let available = seismic::volume_available(&directory).unwrap().value;
            let margin = (7e9 * READING_INTERVAL.as_secs_f64()) as u64;
            assert!(space(0).unwrap() <= available.saturating_sub(margin));
            assert_eq!(reserves.margin(&identity), Some(margin));
        }
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
