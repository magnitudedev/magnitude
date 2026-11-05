//! What a native declaration reads of the device it runs on: `device.NAME`.
//!
//! Each backend declares its properties once ([`properties!`]): each
//! property's type and the sources that establish its value. The
//! declaration generates the backend's `Properties` struct (every field
//! required), its `describe`, which runs each property's sources and
//! nothing else, and the names its native declarations may read, so the
//! language and the runtime share one list. The declarations live here, in
//! the language, because the language checks a declaration's reads against
//! them and every backend crate sits above it; each backend implements its
//! declaration's `Queries`, one method per query, answering exactly what
//! the query names of the API, and hands `describe` what the measurement job
//! stored of its probes ([`ProbeResult`], the generated `Probed`).
//!
//! A property is a [`Support`] (it holds or not), a [`Count`] (a whole
//! number) or a [`Measured`] (a whole number only a probe establishes). None
//! has a constructor outside this module: the only way to make one is to run
//! its declared sources, so no device name, driver identity or constant can
//! set a property.
//!
//! Sources:
//! - `query(NAME)` asks the API one thing (a field, or whether its compiler
//!   forms a pipeline), running no kernel.
//! - `probe(NAME)` is the stored result of a probe kernel the backend owns,
//!   run by the measurement job once per device and driver version, in a
//!   process of its own, never by a load or by opening a device; it
//!   establishes what no query reports, or checks that a reported feature
//!   works where a driver has been seen to get it wrong.
//! - `a & b` holds only when both hold (`b` is not asked when `a` does not
//!   hold); `a | b` is the first of them the device answers.
//!
//! Until its probe has run for the device and driver, a probed property
//! takes its safe value: a first load claims nothing a probe has not shown,
//! so it may be slower, and a mode a probe decides (fp32 round-to-nearest-even
//! division) is the device's default until the probe holds:
//! - a `Support` probe does not hold: a capability a probe checks is never
//!   claimed before the probe shows it, and every reader of a `Support`
//!   already has its path for a device without it;
//! - a `Count` probe establishes a [`Measured`], which reads as not measured.
//!   No number is safe in its place (a guessed register file is a fact the
//!   device never reported), so a reader takes its own declared alternative
//!   through [`Measured::get`] or [`DeviceValues::measured`], never a value
//!   standing in for the device's.

use crate::registry::BackendName;

/// A capability of the device, established by its declared sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Support(bool);

/// A whole-number property of the device, established by its declared
/// sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Count(u64);

/// A whole-number property only a probe establishes: the device's value
/// once the measurement job has run the probe for its device and driver,
/// and not measured before. Its reader states what it takes in place of a
/// value not measured; no number stands in for the device's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Measured(Option<u64>);

impl Support {
    pub fn holds(self) -> bool {
        self.0
    }
}

impl Count {
    pub fn get(self) -> u64 {
        self.0
    }
}

impl Measured {
    /// The device's value; none until its probe has run for the device and
    /// driver version.
    pub fn get(self) -> Option<u64> {
        self.0
    }
}

/// The type of a property, as a declaration reads it: a [`Support`] reads
/// as 0 or 1 and can stand as a condition; a [`Count`] reads as its value;
/// a [`Measured`] reads only with the alternative its reader declares for a
/// device not measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PropertyKind {
    Support,
    Count,
    Measured,
}

/// A device's values of every property its backend provides, read by
/// name. Every [`Support`] and [`Count`] has a value; a [`Measured`] is read
/// apart, with its reader's alternative: there is no absent value at use.
pub trait DeviceValues {
    /// The value of `name`, a [`Support`] (0 or 1) or [`Count`] the
    /// device's backend provides ([`provided`]).
    fn value(&self, name: &str) -> u64;
    /// The value of `name`, a [`Measured`] the device's backend provides;
    /// none while not measured, where the reader takes its alternative.
    fn measured(&self, name: &str) -> Option<u64>;
}

/// What one run of a probe established, as the measurement job stores it
/// for a device and driver version.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProbeResult<T> {
    /// The probe ran and checked its result against what is expected of it.
    Answer(T),
    /// The probe could not establish its answer on this device and driver:
    /// its process failed (the driver's compiler crashed, for one), or the
    /// device lacks what it needs. The property takes its safe value, and
    /// the probe runs again only for another driver version.
    Unanswerable(String),
}

/// What the measurement job stored of one backend's probes for a device and
/// driver version (each backend's generated `Probed`).
pub trait Probes: Default + Clone + serde::Serialize + serde::de::DeserializeOwned {
    /// Every declared probe, with the type it answers.
    const PROBES: &'static [(&'static str, PropertyKind)];
    /// The declared probes with no stored result.
    fn unrun(&self) -> Vec<&'static str>;
    /// Keep probe `name`'s result, its answer as the probe reports it: a
    /// `Support` probe's 0 or 1, a `Count` probe's value.
    fn record(&mut self, name: &str, result: ProbeResult<u64>) -> Result<(), String>;
}

/// The line a probe's process writes to its standard output once it has
/// opened the device and begins the probe: a process that ends before it
/// says nothing of the device.
pub const PROBING: &str = "probing";

/// The line a probe's process writes to its standard output after
/// [`PROBING`]: `answer VALUE` or `unanswerable REASON`. Its process exits 0
/// after it; a process that fails otherwise says nothing of the device.
impl ProbeResult<u64> {
    pub fn line(&self) -> String {
        match self {
            Self::Answer(value) => format!("answer {value}"),
            Self::Unanswerable(reason) => format!("unanswerable {}", reason.replace('\n', " ")),
        }
    }

    /// The result `line` states; none for any other text.
    pub fn parse(line: &str) -> Option<Self> {
        let line = line.trim_end_matches('\n');
        if let Some(value) = line.strip_prefix("answer ") {
            return value.parse().ok().map(Self::Answer);
        }
        line.strip_prefix("unanswerable ").map(|reason| Self::Unanswerable(reason.to_owned()))
    }
}

/// How a probe's stored result answers its source: the safe value until it
/// is answered.
trait ProbeAnswer: Sized {
    type Answer;
    const KIND: PropertyKind;
    fn answer(result: Option<&ProbeResult<Self>>) -> Self::Answer;
    fn of(raw: u64) -> Option<Self>;
}

/// A `Support` probe: holds only once it answered that it holds.
impl ProbeAnswer for bool {
    type Answer = bool;
    const KIND: PropertyKind = PropertyKind::Support;
    fn answer(result: Option<&ProbeResult<bool>>) -> bool {
        matches!(result, Some(ProbeResult::Answer(true)))
    }
    fn of(raw: u64) -> Option<bool> {
        match raw {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

/// A `Count` probe: its value once answered, not measured before.
impl ProbeAnswer for u64 {
    type Answer = Option<u64>;
    const KIND: PropertyKind = PropertyKind::Count;
    fn answer(result: Option<&ProbeResult<u64>>) -> Option<u64> {
        match result {
            Some(ProbeResult::Answer(value)) => Some(*value),
            _ => None,
        }
    }
    fn of(raw: u64) -> Option<u64> {
        Some(raw)
    }
}

/// The type of property `name` when `backend` provides it.
pub fn provided(backend: BackendName, name: &str) -> Option<PropertyKind> {
    let names: &[(&str, PropertyKind)] = match backend {
        BackendName::Cuda => cuda::NAMES,
        BackendName::Vulkan => vulkan::NAMES,
        BackendName::Metal => metal::NAMES,
        BackendName::Cpu => cpu::NAMES,
    };
    names.iter().find(|(provided, _)| *provided == name).map(|(_, kind)| *kind)
}

/// One opened device's properties, of its backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum DeviceProperties {
    Cuda(cuda::Properties),
    Vulkan(vulkan::Properties),
    Metal(metal::Properties),
    Cpu(cpu::Properties),
}

impl DeviceProperties {
    /// The backend whose properties these are.
    pub fn backend(&self) -> BackendName {
        match self {
            Self::Cuda(_) => BackendName::Cuda,
            Self::Vulkan(_) => BackendName::Vulkan,
            Self::Metal(_) => BackendName::Metal,
            Self::Cpu(_) => BackendName::Cpu,
        }
    }
}

impl DeviceValues for DeviceProperties {
    fn value(&self, name: &str) -> u64 {
        match self {
            Self::Cuda(properties) => properties.value(name),
            Self::Vulkan(properties) => properties.value(name),
            Self::Metal(properties) => properties.value(name),
            Self::Cpu(properties) => properties.value(name),
        }
    }
    fn measured(&self, name: &str) -> Option<u64> {
        match self {
            Self::Cuda(properties) => properties.measured(name),
            Self::Vulkan(properties) => properties.measured(name),
            Self::Metal(properties) => properties.measured(name),
            Self::Cpu(properties) => properties.measured(name),
        }
    }
}

/// A device's sources did not establish property `.0`: no source of it
/// answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unanswered(pub &'static str);

impl std::fmt::Display for Unanswered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no source of device property `{}` answered", self.0)
    }
}

impl std::error::Error for Unanswered {}

/// How a property reads: a value, or a [`Measured`]'s value where measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reading {
    Value(u64),
    Measured(Option<u64>),
}

impl std::fmt::Display for Reading {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(value) | Self::Measured(Some(value)) => write!(f, "{value}"),
            Self::Measured(None) => f.write_str("unmeasured"),
        }
    }
}

/// How a property type is made from what its sources answer.
trait Established: Sized {
    type Answer;
    const KIND: PropertyKind;
    fn established(answer: Self::Answer) -> Self;
    fn reading(self) -> Reading;
}

impl Established for Support {
    type Answer = bool;
    const KIND: PropertyKind = PropertyKind::Support;
    fn established(answer: bool) -> Self {
        Self(answer)
    }
    fn reading(self) -> Reading {
        Reading::Value(u64::from(self.0))
    }
}

impl Established for Count {
    type Answer = u64;
    const KIND: PropertyKind = PropertyKind::Count;
    fn established(answer: u64) -> Self {
        Self(answer)
    }
    fn reading(self) -> Reading {
        Reading::Value(self.0)
    }
}

/// Established only by a `Count` probe, whose answer is the measured value
/// or none.
impl Established for Measured {
    type Answer = Option<u64>;
    const KIND: PropertyKind = PropertyKind::Measured;
    fn established(answer: Option<u64>) -> Self {
        Self(answer)
    }
    fn reading(self) -> Reading {
        Reading::Measured(self.0)
    }
}

/// A source as a property's declaration combines it: asked only when the
/// combination needs its answer.
struct Answer<'s, T>(Box<dyn FnOnce() -> Option<T> + 's>);

/// A declared query, as `query(NAME)` reads it.
struct Query<'s, T>(&'s dyn Fn() -> Option<T>);

/// A declared probe, as `probe(NAME)` reads it: its stored result, or its
/// safe value ([`ProbeAnswer`]).
struct Probe<'s, T>(&'s dyn Fn() -> Option<T>);

impl<T> Clone for Query<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Query<'_, T> {}
impl<T> Clone for Probe<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Probe<'_, T> {}

fn query<T>(source: Query<'_, T>) -> Answer<'_, T> {
    Answer(Box::new(move || (source.0)()))
}

fn probe<T>(source: Probe<'_, T>) -> Answer<'_, T> {
    Answer(Box::new(move || (source.0)()))
}

/// `a & b`: holds only when both hold; `b` is asked only when `a` holds.
impl<'s> std::ops::BitAnd for Answer<'s, bool> {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        Answer(Box::new(move || match (self.0)() {
            Some(true) => (other.0)(),
            not_held => not_held,
        }))
    }
}

/// `a | b`: the first of them the device answers.
impl<'s, T: 's> std::ops::BitOr for Answer<'s, T> {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Answer(Box::new(move || (self.0)().or_else(other.0)))
    }
}

/// One backend's property declaration. `queries` and `probes` name each
/// source once with the type it answers; `properties` gives each property
/// its type and its sources, combined with `&` and `|`. A `Count` probe
/// answers only a `Measured` property, which the types enforce: its source
/// answers a measured value or none, which no `Count` takes.
macro_rules! properties {
    (
        $backend:ident;
        queries { $($query:ident: $query_kind:ident),* $(,)? }
        probes { $($probe:ident: $probe_kind:ident),* $(,)? }
        properties { $($name:ident: $kind:ident = $sources:expr);* $(;)? }
    ) => {
        pub mod $backend {
            #[allow(unused_imports)]
            use super::{
                probe, query, Answer, Count, DeviceValues, Established, Measured, Probe, ProbeAnswer,
                ProbeResult, Probes, PropertyKind, Query, Reading, Support, Unanswered,
            };

            /// Every property the backend provides, each established by
            /// its declared sources.
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
            pub struct Properties {
                $(pub $name: $kind,)*
            }

            /// The backend's queries, one method per declared query: the
            /// one API field it names; none where the device does not
            /// answer it.
            pub trait Queries {
                $(fn $query(&self) -> Option<properties!(@answer $query_kind)>;)*
            }

            /// What the measurement job stored of each declared probe for
            /// one device and driver version; none for a probe not run.
            #[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
            pub struct Probed {
                $(pub $probe: Option<ProbeResult<properties!(@answer $probe_kind)>>,)*
            }

            impl Probes for Probed {
                const PROBES: &'static [(&'static str, PropertyKind)] =
                    &[$((stringify!($probe), <properties!(@answer $probe_kind) as ProbeAnswer>::KIND)),*];

                #[allow(unused_mut)]
                fn unrun(&self) -> Vec<&'static str> {
                    let mut unrun = Vec::new();
                    $(if self.$probe.is_none() {
                        unrun.push(stringify!($probe));
                    })*
                    unrun
                }

                #[allow(unused_variables)]
                fn record(&mut self, name: &str, result: ProbeResult<u64>) -> Result<(), String> {
                    match name {
                        $(stringify!($probe) => {
                            self.$probe = Some(match result {
                                ProbeResult::Answer(raw) => ProbeResult::Answer(
                                    <properties!(@answer $probe_kind) as ProbeAnswer>::of(raw).ok_or_else(|| {
                                        format!("probe `{name}` answered {raw}, not a {:?}", <properties!(@answer $probe_kind) as ProbeAnswer>::KIND)
                                    })?,
                                ),
                                ProbeResult::Unanswerable(reason) => ProbeResult::Unanswerable(reason),
                            });
                            Ok(())
                        })*
                        other => Err(format!(concat!("`{}` is not a ", stringify!($backend), " probe"), other)),
                    }
                }
            }

            /// The device's properties, each from its declared sources and
            /// nothing else: `queries` answers each query, and `probed` each
            /// probe, with its safe value for a probe not run.
            pub fn describe(queries: &impl Queries, probed: &Probed) -> Result<Properties, Unanswered> {
                let _ = probed;
                $(let $query = || queries.$query();
                let $query = Query::<properties!(@answer $query_kind)>(&$query);)*
                $(let $probe = || {
                    Some(<properties!(@answer $probe_kind) as ProbeAnswer>::answer(probed.$probe.as_ref()))
                };
                let $probe = Probe::<<properties!(@answer $probe_kind) as ProbeAnswer>::Answer>(&$probe);)*
                Ok(Properties {
                    $($name: {
                        let answer: Answer<'_, <$kind as Established>::Answer> = $sources;
                        <$kind as Established>::established(
                            (answer.0)().ok_or(Unanswered(stringify!($name)))?,
                        )
                    },)*
                })
            }

            /// The names the backend's native declarations may read.
            pub const NAMES: &[(&str, PropertyKind)] =
                &[$((stringify!($name), <$kind as Established>::KIND)),*];

            impl Properties {
                fn reading(&self, name: &str) -> Reading {
                    match name {
                        $(stringify!($name) => Established::reading(self.$name),)*
                        other => panic!(concat!("`{}` is not a ", stringify!($backend), " device property"), other),
                    }
                }

                /// Every property as `name=value`, in declaration order:
                /// the part of an identity the properties decide.
                pub fn identity(&self) -> String {
                    [$(format!("{}={}", stringify!($name), Established::reading(self.$name))),*].join(";")
                }
            }

            impl DeviceValues for Properties {
                fn value(&self, name: &str) -> u64 {
                    match self.reading(name) {
                        Reading::Value(value) => value,
                        Reading::Measured(_) => panic!("`{name}` is measured: it is read with its reader's alternative"),
                    }
                }
                fn measured(&self, name: &str) -> Option<u64> {
                    match self.reading(name) {
                        Reading::Measured(value) => value,
                        Reading::Value(_) => panic!("`{name}` is not a measured property"),
                    }
                }
            }
        }
    };
    (@answer Support) => { bool };
    (@answer Count) => { u64 };
}

properties! {
    cuda;
    queries {
        multiprocessor_count: Count,
        warp_size: Count,
        max_threads_per_block: Count,
        max_threads_per_multiprocessor: Count,
        max_shared_memory_per_block_optin: Count,
        l2_cache_size: Count,
        cooperative_launch: Support,
    }
    probes {}
    properties {
        units: Count = query(multiprocessor_count);
        subgroup_width: Count = query(warp_size);
        max_group_threads: Count = query(max_threads_per_block);
        threads_per_unit: Count = query(max_threads_per_multiprocessor);
        shared_bytes_per_group: Count = query(max_shared_memory_per_block_optin);
        l2_bytes: Count = query(l2_cache_size);
        cooperative_launch: Support = query(cooperative_launch);
    }
}

properties! {
    vulkan;
    queries {
        required_subgroup_32: Count,
        fixed_subgroup_size: Count,
        max_compute_work_group_invocations: Count,
        max_compute_shared_memory_size: Count,
        shader_float16: Support,
        shader_rounding_mode_rte_float16: Support,
        shader_signed_zero_inf_nan_preserve_float16: Support,
        cooperative_matrix: Support,
        cooperative_matrix_f16_16x16x16: Support,
        integer_dot_product_4x8_mixed_signedness_accelerated: Support,
        shader_buffer_float32_atomic_add: Support,
        shader_shared_int64_atomics: Support,
        shader_denorm_preserve_float32: Support,
        shader_fma_float32: Support,
        shader_rounding_mode_rte_float32: Support,
    }
    probes {
        // `seismic_vulkan::probe`: each probe's kernel and its expected result.
        rte_32_division: Support,
        wide_accumulators: Support,
        thread_registers: Count,
        register_file: Count,
    }
    properties {
        subgroup_width: Count = query(required_subgroup_32) | query(fixed_subgroup_size);
        max_group_threads: Count = query(max_compute_work_group_invocations);
        shared_bytes_per_group: Count = query(max_compute_shared_memory_size);
        float16: Support = query(shader_float16)
            & query(shader_rounding_mode_rte_float16)
            & query(shader_signed_zero_inf_nan_preserve_float16);
        matrix: Support = query(cooperative_matrix) & query(cooperative_matrix_f16_16x16x16);
        // A subgroup's cooperative-matrix accumulator arrays of 8 and 16
        // fragments in the flash-attention body compute correctly: the
        // NVIDIA 580 compiler was seen returning wrong outputs for wide
        // accumulator arrays in an earlier form of the body.
        wide_accumulators: Support = query(cooperative_matrix)
            & query(cooperative_matrix_f16_16x16x16)
            & probe(wide_accumulators);
        // `RoundingModeRTE 32` forms and rounds fp32 as declared: the NVIDIA
        // 580 driver's compiler traps on fp32 `OpFDiv` under it.
        rounding_rte_32: Support = query(shader_rounding_mode_rte_float32) & probe(rte_32_division);
        // Registers one thread may use, the most a register-heavy kernel of
        // 32 threads is given ("Register Count").
        registers_per_thread: Measured = probe(thread_registers);
        // Registers one workgroup may use: the same kernel's registers at
        // 1024 threads, times 1024.
        registers_per_group: Measured = probe(register_file);
        mixed_dot: Support = query(integer_dot_product_4x8_mixed_signedness_accelerated);
        f32_atomic_add: Support = query(shader_buffer_float32_atomic_add);
        shared_int64_atomics: Support = query(shader_shared_int64_atomics);
        denorm_preserve_32: Support = query(shader_denorm_preserve_float32);
        fma_32: Support = query(shader_fma_float32);
    }
}

properties! {
    metal;
    queries {
        max_threadgroup_memory_length: Count,
        apple10_family: Support,
        // The compiler forms a `matmul2d` pipeline: asked of the API at
        // open, running nothing.
        tensor_matmul_forms: Support,
    }
    probes {}
    properties {
        shared_bytes_per_group: Count = query(max_threadgroup_memory_length);
        tensor_ops: Support = query(apple10_family) & query(tensor_matmul_forms);
    }
}

properties! {
    cpu;
    queries {
        worker_cores: Count,
    }
    probes {}
    properties {
        units: Count = query(worker_cores);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Answers {
        matrix: Option<bool>,
        later_asked: std::cell::Cell<bool>,
    }

    impl vulkan::Queries for Answers {
        fn required_subgroup_32(&self) -> Option<u64> {
            None
        }
        fn fixed_subgroup_size(&self) -> Option<u64> {
            Some(64)
        }
        fn max_compute_work_group_invocations(&self) -> Option<u64> {
            Some(1024)
        }
        fn max_compute_shared_memory_size(&self) -> Option<u64> {
            Some(49152)
        }
        fn shader_float16(&self) -> Option<bool> {
            Some(true)
        }
        fn shader_rounding_mode_rte_float16(&self) -> Option<bool> {
            Some(true)
        }
        fn shader_signed_zero_inf_nan_preserve_float16(&self) -> Option<bool> {
            Some(false)
        }
        fn cooperative_matrix(&self) -> Option<bool> {
            self.matrix
        }
        fn cooperative_matrix_f16_16x16x16(&self) -> Option<bool> {
            self.later_asked.set(true);
            Some(true)
        }
        fn integer_dot_product_4x8_mixed_signedness_accelerated(&self) -> Option<bool> {
            Some(true)
        }
        fn shader_buffer_float32_atomic_add(&self) -> Option<bool> {
            Some(true)
        }
        fn shader_shared_int64_atomics(&self) -> Option<bool> {
            Some(true)
        }
        fn shader_denorm_preserve_float32(&self) -> Option<bool> {
            Some(true)
        }
        fn shader_fma_float32(&self) -> Option<bool> {
            Some(false)
        }
        fn shader_rounding_mode_rte_float32(&self) -> Option<bool> {
            Some(true)
        }
    }

    /// `|` takes the first source that answers, `&` needs every source to
    /// hold and asks no further once one does not, and a property no
    /// source answers is an error naming it.
    #[test]
    fn sources_combine_as_declared() {
        let answers = Answers { matrix: Some(false), later_asked: Default::default() };
        let properties = vulkan::describe(&answers, &vulkan::Probed::default()).unwrap();
        assert_eq!(properties.subgroup_width.get(), 64);
        assert!(!properties.float16.holds());
        assert!(!properties.matrix.holds());
        assert!(!answers.later_asked.get(), "no source is asked once one does not hold");
        assert_eq!(DeviceValues::value(&properties, "shared_bytes_per_group"), 49152);
        assert_eq!(provided(BackendName::Vulkan, "matrix"), Some(PropertyKind::Support));
        assert_eq!(provided(BackendName::Metal, "units"), None);
        let unanswered = Answers { matrix: None, later_asked: Default::default() };
        assert_eq!(vulkan::describe(&unanswered, &vulkan::Probed::default()).unwrap_err(), Unanswered("matrix"));
    }

    /// A probe not run takes its safe value: a `Support` probe does not
    /// hold, a `Count` probe's `Measured` reads as not measured; once
    /// stored, the probe's answer is the property's, and an unanswerable
    /// probe keeps the safe value without running again.
    #[test]
    fn probes_take_their_safe_value_until_stored() {
        let answers = Answers { matrix: Some(true), later_asked: Default::default() };
        let unmeasured = vulkan::describe(&answers, &vulkan::Probed::default()).unwrap();
        assert!(!unmeasured.rounding_rte_32.holds());
        assert!(!unmeasured.wide_accumulators.holds());
        assert_eq!(unmeasured.registers_per_group.get(), None);
        assert_eq!(DeviceValues::measured(&unmeasured, "registers_per_group"), None);
        assert_eq!(provided(BackendName::Vulkan, "registers_per_group"), Some(PropertyKind::Measured));

        let mut probed = vulkan::Probed::default();
        assert_eq!(probed.unrun().len(), <vulkan::Probed as Probes>::PROBES.len());
        probed.record("rte_32_division", ProbeResult::Answer(1)).unwrap();
        probed.record("wide_accumulators", ProbeResult::Unanswerable("the compiler crashed".into())).unwrap();
        probed.record("register_file", ProbeResult::Answer(65536)).unwrap();
        assert!(probed.record("rte_32_division", ProbeResult::Answer(2)).is_err(), "a Support probe answers 0 or 1");
        assert!(probed.record("matrix", ProbeResult::Answer(1)).is_err(), "only a declared probe is recorded");
        assert_eq!(probed.unrun(), ["thread_registers"]);
        let measured = vulkan::describe(&answers, &probed).unwrap();
        assert!(measured.rounding_rte_32.holds());
        assert!(!measured.wide_accumulators.holds());
        assert_eq!(measured.registers_per_group.get(), Some(65536));
        assert_eq!(measured.registers_per_thread.get(), None);
        assert!(measured.identity().contains("registers_per_thread=unmeasured;registers_per_group=65536"));
    }
}
