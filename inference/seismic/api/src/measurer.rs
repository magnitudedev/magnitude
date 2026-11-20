//! Measuring Seismic work across devices: the sustained clocks of every
//! physical device a caller measures on ([`Measurer`]), and the keys its
//! records are kept under ([`PointKey`], [`PointSubject`], [`KernelIdentity`]).

use crate::measure::{self, Attempt, HeldClocks, Measure, Outcome, RecordStore, Work};
use crate::{
    BackendName, BoundEntry, CallError, DeviceTopology, Entry, Identity, LoadError,
    NativeSpecialization, PhysicalIdentity, VolumeId,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// What a measured point measures: an executor (physical device, backend, and
/// normalized driver family), the host CPU/RAM itself, or a storage volume on
/// the host.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PointSubject {
    /// Work running on a specific executor (GPU or CPU backend + normalized driver).
    Executor(Identity),
    /// Host-only work (host memory sweeps, CPU site pool/admission) keyed by
    /// the host hardware identity (`PhysicalIdentity::Host`), independent of
    /// which GPU executor triggered the measurement.
    Host(PhysicalIdentity),
    /// Storage volume I/O keyed by the volume's identity and normalized OS,
    /// independent of any GPU executor.
    Volume { volume: VolumeId, os: String },
}

impl PointSubject {
    /// An executor subject with its driver normalized so OS/driver point
    /// patches do not invalidate stored measurements.
    pub fn executor(identity: &Identity) -> Self {
        Self::Executor(Identity {
            physical: identity.physical.clone(),
            backend: identity.backend,
            driver: normalize_driver(identity.backend, &identity.driver),
        })
    }

    /// A host-only subject keyed by `topology`'s host physical identity.
    pub fn host(topology: &DeviceTopology) -> Self {
        Self::Host(topology.host_device().identity().clone())
    }

    /// A volume I/O subject keyed by `volume` and `topology`'s normalized OS.
    pub fn volume(topology: &DeviceTopology, volume: VolumeId) -> Self {
        Self::Volume {
            volume,
            os: normalize_os(&topology.host().os_build),
        }
    }
}

/// Normalize an OS build string to the release level that affects performance
/// (dropping security point updates and build numbers):
/// - macOS (`"macOS 15.7.9 (24G830)"`, `"Version 15.7.9 (Build 24G830)"`): major version (`"macOS 15"`, `"Version 15"`).
/// - Linux (`"Linux 6.17.0-1026-nvidia #26-Ubuntu ..."`): major.minor kernel (`"Linux 6.17"`).
/// - Windows (`"Windows 10.0.26100"`, `"Windows 11"`): major.minor or major (`"Windows 10.0"`, `"Windows 11"`).
pub(crate) fn normalize_os(os_build: &str) -> String {
    let trimmed = os_build.trim();
    if let Some(rest) = trimmed.strip_prefix("macOS ") {
        return match major_component(rest) {
            Some(major) => format!("macOS {major}"),
            None => "macOS".to_owned(),
        };
    }
    if let Some(rest) = trimmed.strip_prefix("Version ") {
        return match major_component(rest) {
            Some(major) => format!("Version {major}"),
            None => "Version".to_owned(),
        };
    }
    if let Some(rest) = trimmed.strip_prefix("Linux ") {
        return match major_minor_components(rest) {
            Some((major, Some(minor))) => format!("Linux {major}.{minor}"),
            Some((major, None)) => format!("Linux {major}"),
            None => "Linux".to_owned(),
        };
    }
    if let Some(rest) = trimmed.strip_prefix("Windows ") {
        return match major_minor_components(rest) {
            Some((major, Some(minor))) => format!("Windows {major}.{minor}"),
            Some((major, None)) => format!("Windows {major}"),
            None => "Windows".to_owned(),
        };
    }
    trimmed.to_owned()
}

/// Normalize an executor's driver string for measurement keying:
/// - Metal and CPU use the host OS build -> normalized via [`normalize_os`].
/// - CUDA is already `major.minor` from `cuDriverGetVersion`.
/// - Vulkan (`"vulkan driver <id> version <ver>"`) keeps `driver_id` and
///   `major.minor`, dropping patch/build components.
pub(crate) fn normalize_driver(backend: BackendName, driver: &str) -> String {
    match backend {
        BackendName::Metal | BackendName::Cpu => normalize_os(driver),
        BackendName::Cuda => driver.trim().to_owned(),
        BackendName::Vulkan => normalize_vulkan_driver(driver),
    }
}

fn normalize_vulkan_driver(driver: &str) -> String {
    let trimmed = driver.trim();
    let Some(rest) = trimmed.strip_prefix("vulkan driver ") else {
        return trimmed.to_owned();
    };
    let Some((id_str, ver_str)) = rest.split_once(" version ") else {
        return trimmed.to_owned();
    };
    let id_str = id_str.trim();
    let ver_str = ver_str.trim();
    // Raw u32 VkPhysicalDeviceProperties::driverVersion.
    if let Ok(raw) = ver_str.parse::<u32>() {
        let (major, minor) = match id_str.parse::<i32>() {
            // VK_DRIVER_ID_NVIDIA_PROPRIETARY: 10-bit major, 8-bit minor, 8-bit secondary, 6-bit tertiary.
            Ok(4) => ((raw >> 22) & 0x3ff, (raw >> 14) & 0xff),
            // VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS: 18-bit major, 14-bit minor.
            Ok(5) => (raw >> 14, raw & 0x3fff),
            // Standard VK_MAKE_VERSION: major = bits 22..31, minor = bits 12..21, patch = bits 0..11.
            _ => (raw >> 22, (raw >> 12) & 0x3ff),
        };
        return format!("vulkan driver {id_str} version {major}.{minor}");
    }
    // Dotted version string (e.g. "615.71.9" -> "615.71").
    if let Some((major, Some(minor))) = major_minor_components(ver_str) {
        return format!("vulkan driver {id_str} version {major}.{minor}");
    }
    trimmed.to_owned()
}

fn major_component(text: &str) -> Option<&str> {
    let token = text.split_whitespace().next()?;
    let major = token.split('.').next()?;
    (!major.is_empty() && major.bytes().all(|b| b.is_ascii_digit())).then_some(major)
}

fn major_minor_components(text: &str) -> Option<(&str, Option<&str>)> {
    let token = text.split_whitespace().next()?;
    let mut parts = token.split('.');
    let major = parts.next()?;
    if major.is_empty() || !major.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let minor = parts.next().and_then(|part| {
        let digits_end = part.bytes().take_while(|b| b.is_ascii_digit()).count();
        (digits_end > 0).then_some(&part[..digits_end])
    });
    Some((major, minor))
}

/// What a measured point is a function of, the key its record is kept
/// under: the subject it measures (an executor, the host, or a storage
/// volume), the kernels it runs, the configuration they were formed under,
/// and the point's own values (its shape, and how it was timed). Its parts
/// are named by the caller that measures.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PointKey {
    /// What the point measures: an executor, the host, or a storage volume.
    pub subject: PointSubject,
    /// The kernels the point runs.
    pub kernels: Vec<KernelIdentity>,
    /// The configuration they were formed under, by name.
    pub configuration: BTreeMap<String, u64>,
    /// The point's own values (its shape, and how it was timed), by name.
    pub point: BTreeMap<String, u64>,
}

/// The identity of an entry's kernel on a backend, computed without opening a
/// device: its formed declaration (excluding `source_path` and tuning search
/// domains), element bindings, default specialization values, backend, the
/// source this build renders of it (with Metal `tensor_ops` when enabled),
/// and the backend toolchain (`seismic_runtime::native::tune::kernel_digest`).
/// With the executor's subject it names the program a device runs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KernelIdentity(String);

impl KernelIdentity {
    /// The identity of `entry`'s kernel on `backend` at `statics`.
    /// `tensor_ops` states whether a Metal executor forms Metal 4 tensor
    /// operations (`Executor::forms_tensor_operations`); ignored on other
    /// backends.
    pub fn of<E: Entry>(
        backend: BackendName,
        tensor_ops: bool,
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
            tensor_ops,
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
    /// `attempting` lets it proceed, none begun once `deadline` has passed
    /// and each begun timed to its end, and record each but one not formed
    /// now, whose timing failed or that was not begun
    /// ([`measure::measure_all`], the one writer of records).
    pub fn measure_all<K, W, G, F>(
        &mut self,
        points: Vec<(K, F)>,
        how: &Measure,
        deadline: measure::Deadline,
        store: &dyn RecordStore<K>,
        attempting: &mut dyn FnMut(&K) -> Attempt,
    ) -> Vec<(K, Outcome<CallError, G>)>
    where
        W: Work<Error = CallError, Clock = PhysicalIdentity>,
        G: std::fmt::Display,
        F: FnOnce() -> Result<W, measure::Unformed<G>>,
    {
        measure::measure_all(points, how, deadline, &mut self.clocks, store, attempting)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_os_strips_point_patches_and_build_numbers() {
        assert_eq!(normalize_os("macOS 15.7.9 (24G830)"), "macOS 15");
        assert_eq!(normalize_os("macOS 15.7.8 (24G700)"), "macOS 15");
        assert_eq!(normalize_os("macOS 16.0 (25A100)"), "macOS 16");
        assert_eq!(normalize_os("macOS (24G830)"), "macOS");
        assert_eq!(normalize_os("macOS"), "macOS");
        assert_eq!(normalize_os("Version 15.7.9 (Build 24G830)"), "Version 15");
        assert_eq!(
            normalize_os(
                "Linux 6.17.0-1026-nvidia #26-Ubuntu SMP PREEMPT_DYNAMIC Thu Jun 25 00:57:17 UTC 2026"
            ),
            "Linux 6.17"
        );
        assert_eq!(normalize_os("Linux"), "Linux");
        assert_eq!(normalize_os("Windows 10.0.26100"), "Windows 10.0");
        assert_eq!(normalize_os("Windows 11"), "Windows 11");
    }

    #[test]
    fn normalize_driver_ignores_os_and_vulkan_patch_updates() {
        assert_eq!(
            normalize_driver(BackendName::Metal, "macOS 15.7.9 (24G830)"),
            normalize_driver(BackendName::Metal, "macOS 15.7.8 (24G700)")
        );
        assert_ne!(
            normalize_driver(BackendName::Metal, "macOS 15.7.9 (24G830)"),
            normalize_driver(BackendName::Metal, "macOS 16.0 (25A100)")
        );
        assert_eq!(
            normalize_driver(BackendName::Cpu, "macOS 15.7.9 (24G830)"),
            "macOS 15"
        );
        assert_eq!(
            normalize_driver(BackendName::Cuda, "cuda driver 12.8"),
            "cuda driver 12.8"
        );
        // NVIDIA Vulkan raw u32: (580 << 22) | (159 << 14) | (patch << 6) | tertiary
        let nv_a = (580u32 << 22) | (159 << 14) | (3 << 6) | 0;
        let nv_b = (580u32 << 22) | (159 << 14) | (9 << 6) | 2;
        let nv_minor = (580u32 << 22) | (160 << 14);
        assert_eq!(
            normalize_driver(BackendName::Vulkan, &format!("vulkan driver 4 version {nv_a}")),
            "vulkan driver 4 version 580.159"
        );
        assert_eq!(
            normalize_driver(BackendName::Vulkan, &format!("vulkan driver 4 version {nv_a}")),
            normalize_driver(BackendName::Vulkan, &format!("vulkan driver 4 version {nv_b}"))
        );
        assert_ne!(
            normalize_driver(BackendName::Vulkan, &format!("vulkan driver 4 version {nv_a}")),
            normalize_driver(BackendName::Vulkan, &format!("vulkan driver 4 version {nv_minor}"))
        );
        // Standard VK_MAKE_VERSION (e.g. Mesa RADV driver_id = 3): major 24, minor 2, patch 3 vs 4
        let radv_a = (24u32 << 22) | (2 << 12) | 3;
        let radv_b = (24u32 << 22) | (2 << 12) | 4;
        let radv_minor = (24u32 << 22) | (3 << 12) | 0;
        assert_eq!(
            normalize_driver(BackendName::Vulkan, &format!("vulkan driver 3 version {radv_a}")),
            "vulkan driver 3 version 24.2"
        );
        assert_eq!(
            normalize_driver(BackendName::Vulkan, &format!("vulkan driver 3 version {radv_a}")),
            normalize_driver(BackendName::Vulkan, &format!("vulkan driver 3 version {radv_b}"))
        );
        assert_ne!(
            normalize_driver(BackendName::Vulkan, &format!("vulkan driver 3 version {radv_a}")),
            normalize_driver(BackendName::Vulkan, &format!("vulkan driver 3 version {radv_minor}"))
        );
        // Dotted Vulkan version strings (fixtures/tests)
        assert_eq!(
            normalize_driver(BackendName::Vulkan, "vulkan driver 4 version 615.71.9"),
            normalize_driver(BackendName::Vulkan, "vulkan driver 4 version 615.71.10")
        );
        assert_ne!(
            normalize_driver(BackendName::Vulkan, "vulkan driver 4 version 615.71.9"),
            normalize_driver(BackendName::Vulkan, "vulkan driver 4 version 616.92.0")
        );
    }

    #[test]
    fn point_key_subjects_store_and_load_in_record_store() {
        let store = crate::measure::HeldStore::default();
        let record = crate::measure::Record::Measured(crate::measure::Timing {
            median: 1e-3,
            interval: None,
            samples: 1,
        });
        let subjects = [
            PointSubject::executor(&Identity {
                physical: PhysicalIdentity::Registry(0x1234),
                backend: BackendName::Metal,
                driver: "macOS 15.7.9 (24G830)".into(),
            }),
            PointSubject::Host(PhysicalIdentity::Host {
                machine: "Apple M4 Pro;ram=51539607552".into(),
            }),
            PointSubject::Volume {
                volume: VolumeId {
                    device: 42,
                    uuid: Some("ABCD-1234".into()),
                },
                os: "macOS 15".into(),
            },
        ];
        for subject in subjects {
            let key = PointKey {
                subject,
                kernels: Vec::new(),
                configuration: BTreeMap::from([("storage".into(), 1)]),
                point: BTreeMap::from([("full".into(), 1)]),
            };
            store.store(&key, &record).unwrap();
            assert_eq!(store.load(&key).unwrap(), Some(record.clone()));
        }
    }
}
