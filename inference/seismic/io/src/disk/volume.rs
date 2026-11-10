use super::file::SourceFile;
use super::platform::{evict, uncached, volume_key};
use super::probe::{Curve, Urgent, VolumeLines, LEAST_READ, WIDEST_WINDOW};
use super::OpenFile;
use crate::{Error, Volume};
use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex, OnceLock};

/// How a reader reads the volume its file lies on: its link's shape, as the
/// volume's curve gives it for the reader's request ([`VolumeLines::shape`]).
#[derive(Clone, Debug)]
pub(crate) struct VolumeReads {
    pub(crate) id: Volume,
    pub(crate) shape: Shape,
}

/// A link's depths, in bytes in flight, and its read sizes ("A volume's
/// curve", "Order and depth"): due pieces join the due bytes in flight up
/// to `due`; a piece read ahead joins everything in flight up to `knee`.
/// Due pieces are cut at `due_read`, the others at `ahead_read`.
/// Its reads at once occupy `cores` of the host's processors (the cores
/// its threads ran, in seconds a second).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub due: u64,
    pub knee: u64,
    pub ahead_read: u64,
    pub due_read: u64,
    pub cores: f64,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "due {} KiB in {} KiB reads, knee {} KiB in {} KiB reads, {:.2} cores",
            self.due >> 10,
            self.due_read >> 10,
            self.knee >> 10,
            self.ahead_read >> 10,
            self.cores,
        )
    }
}

impl Shape {
    /// How a volume is read before any probe of it: at the least read size
    /// and every depth the widest window, every piece at the most any device
    /// measured needs (the declared bounds of the search), which takes no
    /// measurement.
    pub(crate) const PENDING: Self = Self { due: WIDEST_WINDOW, knee: WIDEST_WINDOW, ahead_read: LEAST_READ, due_read: LEAST_READ, cores: 0. };

    /// The size a piece is cut at, due or not.
    pub(crate) fn read(&self, due: bool) -> u64 {
        if due { self.due_read } else { self.ahead_read }
    }

}

/// How the rule before the curve read a volume for a request: five figures
/// a load reads as they were while the job probes the curve ("A volume's
/// curve"). Pieces were cut at `read` bytes, a demand request's at
/// `demand_read`; a demand piece joined the bytes in flight up to `demand`,
/// any other up to `window`; its reads occupied `cores`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarlierChoice {
    pub window: u64,
    pub demand: u64,
    pub read: u64,
    pub demand_read: u64,
    pub cores: f64,
}

impl EarlierChoice {
    /// The rule's schema, a term of its stored choice's key.
    pub const SCHEMA: u64 = 3;

    /// Its figures as a link's shape: a demand piece is due, any other read
    /// ahead, which joined every byte in flight up to `window`.
    pub(crate) fn shape(&self) -> Shape {
        Shape { due: self.demand, knee: self.window, ahead_read: self.read, due_read: self.demand_read, cores: self.cores }
    }
}

impl Curve {
    /// The rule a curve is kept by, a term of a stored curve's key, so a
    /// curve probed by another is probed again: 6, every point kept,
    /// windows from one read, each lone time a step's whole request's, and
    /// the request's range line through the engine, its knee (5 had none
    /// and took the knee of the power-of-two read sizes; 4 timed one record
    /// of a request; [`EarlierChoice`] the rule before).
    pub const SCHEMA: u64 = 6;
}

/// Each volume's lines, by volume.
fn volume_lines(file: &File) -> Result<(Volume, Arc<Mutex<VolumeLines>>), Error> {
    static VOLUMES: OnceLock<Mutex<HashMap<Volume, Arc<Mutex<VolumeLines>>>>> = OnceLock::new();
    let key = volume_key(file)?;
    Ok((key, VOLUMES.get_or_init(Default::default).lock().expect("volumes").entry(key).or_default().clone()))
}

/// Read `source`'s volume by `curve`, probed for `urgent`, from now on:
/// for `urgent`, and for every request that adopted nothing.
pub fn adopt_volume(source: &impl AsRef<OpenFile>, urgent: Option<&Urgent>, curve: &Curve) -> Result<(), Error> {
    let source = source.as_ref();
    let (_, lines) = volume_lines(source.file())?;
    lines.lock().expect("a volume's lines").adopt(urgent, curve, source.size());
    Ok(())
}

/// Read `source`'s volume for `urgent` by the rule before the curve's
/// `choice` until a curve is adopted for it.
pub fn adopt_earlier(source: &impl AsRef<OpenFile>, urgent: Option<&Urgent>, choice: EarlierChoice) -> Result<(), Error> {
    let (_, lines) = volume_lines(source.as_ref().file())?;
    lines.lock().expect("a volume's lines").adopt_earlier(urgent, choice);
    Ok(())
}

/// Probe `source`'s volume for `urgent`, reading it and evicting the file's
/// pages to time its storage: run only where nothing waits on it (the
/// measurement job, a tool), and read by its curve from now on.
pub fn probe_volume(source: &impl AsRef<OpenFile>, urgent: Option<&Urgent>) -> Result<Curve, Error> {
    let source = source.as_ref();
    let (key, lines) = volume_lines(source.file())?;
    // Held across the probe: no probe shares the storage with another of
    // the volume, and other volumes' readers wait on none of it.
    let mut lines = lines.lock().expect("a volume's lines");
    let probed = Arc::new(SourceFile { path: source.path().into(), size: source.size(), through: uncached(source.file())? });
    evict(source.file(), source.size())?;
    lines.probe(key, &probed, urgent)
}

impl VolumeReads {
    /// How `file` is read for `urgent`: by its volume's shape for it. A
    /// reader measures nothing.
    pub(crate) fn of(file: &File, urgent: Option<&Urgent>) -> Result<Self, Error> {
        let (id, lines) = volume_lines(file)?;
        let shape = lines.lock().expect("a volume's lines").shape(urgent);
        Ok(Self { id, shape })
    }
}

/// How `source`'s volume is read for `urgent` now: its link's shape.
pub fn link_shape(source: &impl AsRef<OpenFile>, urgent: Option<&Urgent>) -> Result<Shape, Error> {
    Ok(VolumeReads::of(source.as_ref().file(), urgent)?.shape)
}

/// The cores the read engine's workers occupy reading the volumes `sources`
/// lie on for `urgent`, each at its deepest at once: each distinct
/// volume's, as its probe measured them ("A volume's curve").
pub fn read_cores<'a, S: AsRef<OpenFile> + 'a>(sources: impl IntoIterator<Item = &'a S>, urgent: &Urgent) -> Result<f64, Error> {
    let mut volumes = Vec::<VolumeReads>::new();
    for source in sources {
        let volume = VolumeReads::of(source.as_ref().file(), Some(urgent))?;
        if !volumes.iter().any(|known| known.id == volume.id) {
            volumes.push(volume);
        }
    }
    Ok(volumes.iter().map(|volume| volume.shape.cores).sum())
}
