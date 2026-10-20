//! Measuring Seismic work across devices: the sustained clocks of every
//! physical device a caller measures on ([`Measurer`]), and the keys its
//! records are kept under ([`PointKey`], [`KernelIdentity`]).

use crate::measure::{self, Attempt, HeldClocks, Measure, Outcome, RecordStore, Work};
use crate::{
    BackendName, BoundEntry, CallError, Entry, Identity, LoadError, NativeSpecialization,
    PhysicalIdentity,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// What a measured point is a function of, the key its record is kept
/// under: the executor it ran on, the kernels it runs, the configuration
/// they were formed under, and the point's own values (its shape, and how
/// it was timed). Its parts are named by the caller that measures.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PointKey {
    /// The executor the point ran on.
    pub executor: Identity,
    /// The kernels the point runs.
    pub kernels: Vec<KernelIdentity>,
    /// The configuration they were formed under, by name.
    pub configuration: BTreeMap<String, u64>,
    /// The point's own values (its shape, and how it was timed), by name.
    pub point: BTreeMap<String, u64>,
}

/// The identity of an entry's kernel on a backend, computed without a
/// device: its declaration and sources, element bindings, static values,
/// backend, the source this build renders of it, and for CUDA the loaded
/// NVRTC's release (`seismic_runtime::native::tune::kernel_digest`, which
/// says why each backend's toolchain is or is not in it). With the
/// executor's identity it names the program a device runs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KernelIdentity(String);

impl KernelIdentity {
    /// The identity of `entry`'s kernel on `backend` at `statics`.
    pub fn of<E: Entry>(
        backend: BackendName,
        entry: &BoundEntry<E>,
        statics: &NativeSpecialization,
    ) -> Result<Self, LoadError> {
        let module = E::module().map_err(LoadError::Bundle)?;
        let resolved = E::resolve(module).map_err(LoadError::Bundle)?;
        let bindings = entry.elements.iter().fold(
            seismic_lang::entry::ElementBindings::new(),
            |bindings, (name, element)| bindings.bind(name, element.id()),
        );
        seismic_runtime::native::tune::kernel_digest(
            backend,
            module.checked(),
            resolved.id(),
            &bindings,
            statics,
            entry.cpu,
        )
        .map(Self)
        .map_err(LoadError::from_prepare)
    }

    /// The digest, as hexadecimal text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The sustained clocks a caller holds across the devices it measures on,
/// one per physical device, each established by the first measurement there
/// and raised by every later one ([`HeldClocks`]).
pub struct Measurer {
    clocks: HeldClocks<PhysicalIdentity, CallError>,
}

impl Measurer {
    /// No clock held yet; clocks are established with readings of
    /// `reading_seconds` of device time and the evidence a lead takes at
    /// `error_rate`.
    pub fn new(error_rate: f64, reading_seconds: f64) -> Self {
        Self {
            clocks: HeldClocks::new(error_rate, reading_seconds),
        }
    }

    /// Measure the points `store` holds no record of, each formed only once
    /// `attempting` lets it proceed, and record each but one not formed now
    /// or whose timing failed ([`measure::measure_all`], the one writer of
    /// records).
    pub fn measure_all<K, W, G, F>(
        &mut self,
        points: Vec<(K, F)>,
        how: &Measure,
        store: &dyn RecordStore<K>,
        attempting: &mut dyn FnMut(&K) -> Attempt,
    ) -> Vec<(K, Outcome<CallError, G>)>
    where
        W: Work<Error = CallError, Clock = PhysicalIdentity>,
        G: std::fmt::Display,
        F: FnOnce() -> Result<W, measure::Unformed<G>>,
    {
        measure::measure_all(points, how, &mut self.clocks, store, attempting)
    }
}
