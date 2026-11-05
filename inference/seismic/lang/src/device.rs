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
//! declaration's `Sources`, one method per source, reading exactly the API
//! field or stored probe result the source names.
//!
//! A property is a [`Support`] (it holds or not) or a [`Count`] (a whole
//! number). Neither has a constructor outside this module: the only way to
//! make one is to run its declared sources, so no device name, driver
//! identity or constant can set a property.
//!
//! Sources:
//! - `query(NAME)` reads one API field.
//! - `probe(NAME)` is a check the backend owns, run when the device is
//!   described; it establishes what no query reports, or checks that a
//!   reported feature works where a driver has been seen to get it wrong.
//! - `a & b` holds only when both hold (`b` is not asked when `a` does not
//!   hold); `a | b` is the first of them the device answers.

use crate::registry::BackendName;

/// A capability of the device, established by its declared sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Support(bool);

/// A whole-number property of the device, established by its declared
/// sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Count(u64);

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

/// The type of a property, as a declaration reads it: a [`Support`] reads
/// as 0 or 1 and can stand as a condition; a [`Count`] reads as its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PropertyKind {
    Support,
    Count,
}

/// A device's values of every property its backend provides, read by
/// name. Every name the backend provides has a value: there is no absent
/// property at use.
pub trait DeviceValues {
    /// The value of `name`, which the device's backend provides
    /// ([`provided`]); a [`Support`] reads as 0 or 1.
    fn value(&self, name: &str) -> u64;
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

/// How a property type is made from what its sources answer.
trait Established: Sized {
    type Answer;
    const KIND: PropertyKind;
    fn established(answer: Self::Answer) -> Self;
    fn value(self) -> u64;
}

impl Established for Support {
    type Answer = bool;
    const KIND: PropertyKind = PropertyKind::Support;
    fn established(answer: bool) -> Self {
        Self(answer)
    }
    fn value(self) -> u64 {
        u64::from(self.0)
    }
}

impl Established for Count {
    type Answer = u64;
    const KIND: PropertyKind = PropertyKind::Count;
    fn established(answer: u64) -> Self {
        Self(answer)
    }
    fn value(self) -> u64 {
        self.0
    }
}

/// A source as a property's declaration combines it: asked only when the
/// combination needs its answer.
struct Answer<'s, T>(Box<dyn FnOnce() -> Option<T> + 's>);

/// A declared query, as `query(NAME)` reads it.
struct Query<'s, T>(&'s dyn Fn() -> Option<T>);

/// A declared probe, as `probe(NAME)` reads it.
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
/// its type and its sources, combined with `&` and `|`.
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
                probe, query, Answer, Count, DeviceValues, Established, Probe, PropertyKind, Query,
                Support, Unanswered,
            };

            /// Every property the backend provides, each established by
            /// its declared sources.
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
            pub struct Properties {
                $(pub $name: $kind,)*
            }

            /// The backend's sources, one method per declared query (the
            /// one API field it names) and probe (its result); none where
            /// the device does not answer it.
            pub trait Sources {
                $(fn $query(&self) -> Option<properties!(@answer $query_kind)>;)*
                $(fn $probe(&self) -> Option<properties!(@answer $probe_kind)>;)*
            }

            /// The device's properties, each from its declared sources and
            /// nothing else.
            pub fn describe(sources: &impl Sources) -> Result<Properties, Unanswered> {
                $(let $query = || sources.$query();
                let $query = Query::<properties!(@answer $query_kind)>(&$query);)*
                $(let $probe = || sources.$probe();
                let $probe = Probe::<properties!(@answer $probe_kind)>(&$probe);)*
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

            impl DeviceValues for Properties {
                fn value(&self, name: &str) -> u64 {
                    match name {
                        $(stringify!($name) => Established::value(self.$name),)*
                        other => panic!(concat!("`{}` is not a ", stringify!($backend), " device property"), other),
                    }
                }
            }

            impl Properties {
                /// Every property as `name=value`, in declaration order:
                /// the part of an identity the properties decide.
                pub fn identity(&self) -> String {
                    [$(format!("{}={}", stringify!($name), Established::value(self.$name))),*].join(";")
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
    }
    probes {}
    properties {
        subgroup_width: Count = query(required_subgroup_32) | query(fixed_subgroup_size);
        max_group_threads: Count = query(max_compute_work_group_invocations);
        shared_bytes_per_group: Count = query(max_compute_shared_memory_size);
        float16: Support = query(shader_float16)
            & query(shader_rounding_mode_rte_float16)
            & query(shader_signed_zero_inf_nan_preserve_float16);
        matrix: Support = query(cooperative_matrix) & query(cooperative_matrix_f16_16x16x16);
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
    }
    probes {
        tensor_matmul_forms: Support,
    }
    properties {
        shared_bytes_per_group: Count = query(max_threadgroup_memory_length);
        tensor_ops: Support = query(apple10_family) & probe(tensor_matmul_forms);
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

    impl vulkan::Sources for Answers {
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
    }

    /// `|` takes the first source that answers, `&` needs every source to
    /// hold and asks no further once one does not, and a property no
    /// source answers is an error naming it.
    #[test]
    fn sources_combine_as_declared() {
        let answers = Answers { matrix: Some(false), later_asked: Default::default() };
        let properties = vulkan::describe(&answers).unwrap();
        assert_eq!(properties.subgroup_width.get(), 64);
        assert!(!properties.float16.holds());
        assert!(!properties.matrix.holds());
        assert!(!answers.later_asked.get(), "no source is asked once one does not hold");
        assert_eq!(DeviceValues::value(&properties, "shared_bytes_per_group"), 49152);
        assert_eq!(provided(BackendName::Vulkan, "matrix"), Some(PropertyKind::Support));
        assert_eq!(provided(BackendName::Metal, "units"), None);
        let unanswered = Answers { matrix: None, later_asked: Default::default() };
        assert_eq!(vulkan::describe(&unanswered).unwrap_err(), Unanswered("matrix"));
    }
}
