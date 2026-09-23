//! Versioned CPU-native numerical helpers. Canonical exact transcendental
//! semantics expand to ordinary kernel IR in `seismic-compiler`; the host
//! transcendental calls here are used only by explicitly approximate math
//! operations. Native physical rounding, packed field extraction, atomics, and
//! team barriers remain backend-owned native services. Packed decode arithmetic
//! is instantiated from language recipes before kernel closure.

use seismic_lang::registry::{bf16_round, f16_bits, f16_round, f16_to_f32};
use seismic_lang::types::DType;
use std::sync::{Mutex, OnceLock};

/// Name and version of the host sequence set.
pub const MATH_IDENTITY: &str = "seismic_math";
pub const MATH_VERSION: u32 = 2;

/// Conversion used by selected native physical operations. Narrow conversion
/// follows this backend sequence; the language-owned exact recipe defines
/// source scalar rounding independently of native instruction choices.
pub fn round_to(dtype: DType, value: f64) -> f64 {
    match dtype {
        DType::F32 => value as f32 as f64,
        DType::BF16 => bf16_round(value as f32) as f64,
        DType::F16 => f16_round(value as f32) as f64,
        DType::I32 => value as i32 as f64,
        DType::U32 => value as u32 as f64,
        DType::Bool => {
            if value != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

extern "C" fn seismic_approx_exp_f32(x: f32) -> f32 {
    x.exp()
}
extern "C" fn seismic_approx_log_f32(x: f32) -> f32 {
    x.ln()
}
extern "C" fn seismic_approx_sin_f32(x: f32) -> f32 {
    x.sin()
}
extern "C" fn seismic_approx_cos_f32(x: f32) -> f32 {
    x.cos()
}
/// The registry `max`: exact, a NaN operand is ignored.
extern "C" fn seismic_fmax(a: f64, b: f64) -> f64 {
    a.max(b)
}
extern "C" fn seismic_fmin(a: f64, b: f64) -> f64 {
    a.min(b)
}
macro_rules! round_helper {
    ($name:ident, $dtype:expr) => {
        extern "C" fn $name(x: f64) -> f64 {
            round_to($dtype, x)
        }
    };
}
round_helper!(seismic_round_f32, DType::F32);
round_helper!(seismic_round_f16, DType::F16);
round_helper!(seismic_round_bf16, DType::BF16);
round_helper!(seismic_round_i32, DType::I32);
round_helper!(seismic_round_u32, DType::U32);
round_helper!(seismic_round_bool, DType::Bool);
extern "C" fn seismic_f16_load(bits: i32) -> f32 {
    f16_to_f32((bits & 0xffff) as u16)
}
extern "C" fn seismic_f16_store(x: f32) -> i32 {
    i32::from(f16_bits(x))
}
extern "C" fn seismic_packed_bits(base: *const u8, bit: u64, bits: u32) -> u32 {
    let byte = (bit / 8) as usize;
    let shift = (bit % 8) as u32;
    let mut raw = 0u64;
    // A packed plane is padded to whole bytes. Reading bytewise avoids any
    // alignment assumption and reads only bytes touched by the entry.
    let needed = ((shift + bits + 7) / 8) as usize;
    for ordinal in 0..needed {
        let value = unsafe { *base.add(byte + ordinal) };
        raw |= u64::from(value) << (ordinal * 8);
    }
    ((raw >> shift) & ((1u64 << bits) - 1)) as u32
}
#[derive(Clone, Copy)]
enum AtomicOperation {
    Add,
    Max,
    Min,
}

#[derive(Clone, Copy)]
enum AtomicDType {
    F32,
    F16,
    BF16,
    I32,
    U32,
}

unsafe fn atomic(pointer: *mut u8, dtype: AtomicDType, op: AtomicOperation, incoming: u64) {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        // This mutex carries no state; it only serializes the update. A
        // cancelled launch may unwind through the helper while holding it,
        // and that does not invalidate the next launch's serialization.
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    unsafe {
        match dtype {
            AtomicDType::F32 => {
                let current = f32::from_bits(std::ptr::read_unaligned(pointer.cast::<u32>()));
                let incoming = f32::from_bits(incoming as u32);
                let value = match op {
                    AtomicOperation::Add => current + incoming,
                    AtomicOperation::Max => current.max(incoming),
                    AtomicOperation::Min => current.min(incoming),
                };
                std::ptr::write_unaligned(pointer.cast::<u32>(), value.to_bits());
            }
            AtomicDType::F16 => {
                let current = f16_to_f32(std::ptr::read_unaligned(pointer.cast::<u16>()));
                let incoming = f16_to_f32(incoming as u16);
                let value = match op {
                    AtomicOperation::Add => current + incoming,
                    AtomicOperation::Max => current.max(incoming),
                    AtomicOperation::Min => current.min(incoming),
                };
                std::ptr::write_unaligned(pointer.cast::<u16>(), f16_bits(value));
            }
            AtomicDType::BF16 => {
                let raw = std::ptr::read_unaligned(pointer.cast::<u16>());
                let current = f32::from_bits(u32::from(raw) << 16);
                let incoming = f32::from_bits(u32::from(incoming as u16) << 16);
                let value = match op {
                    AtomicOperation::Add => current + incoming,
                    AtomicOperation::Max => current.max(incoming),
                    AtomicOperation::Min => current.min(incoming),
                };
                std::ptr::write_unaligned(
                    pointer.cast::<u16>(),
                    (bf16_round(value).to_bits() >> 16) as u16,
                );
            }
            AtomicDType::I32 => {
                let current = std::ptr::read_unaligned(pointer.cast::<i32>());
                let incoming = incoming as u32 as i32;
                let value = match op {
                    AtomicOperation::Add => current.wrapping_add(incoming),
                    AtomicOperation::Max => current.max(incoming),
                    AtomicOperation::Min => current.min(incoming),
                };
                std::ptr::write_unaligned(pointer.cast::<i32>(), value);
            }
            AtomicDType::U32 => {
                let current = std::ptr::read_unaligned(pointer.cast::<u32>());
                let incoming = incoming as u32;
                let value = match op {
                    AtomicOperation::Add => current.wrapping_add(incoming),
                    AtomicOperation::Max => current.max(incoming),
                    AtomicOperation::Min => current.min(incoming),
                };
                std::ptr::write_unaligned(pointer.cast::<u32>(), value);
            }
        }
    }
}

macro_rules! atomic_helper {
    ($name:ident, $dtype:expr, $operation:expr) => {
        extern "C" fn $name(pointer: *mut u8, incoming: u64) {
            unsafe { atomic(pointer, $dtype, $operation, incoming) }
        }
    };
}
atomic_helper!(
    seismic_atomic_add_f32,
    AtomicDType::F32,
    AtomicOperation::Add
);
atomic_helper!(
    seismic_atomic_max_f32,
    AtomicDType::F32,
    AtomicOperation::Max
);
atomic_helper!(
    seismic_atomic_min_f32,
    AtomicDType::F32,
    AtomicOperation::Min
);
atomic_helper!(
    seismic_atomic_add_f16,
    AtomicDType::F16,
    AtomicOperation::Add
);
atomic_helper!(
    seismic_atomic_max_f16,
    AtomicDType::F16,
    AtomicOperation::Max
);
atomic_helper!(
    seismic_atomic_min_f16,
    AtomicDType::F16,
    AtomicOperation::Min
);
atomic_helper!(
    seismic_atomic_add_bf16,
    AtomicDType::BF16,
    AtomicOperation::Add
);
atomic_helper!(
    seismic_atomic_max_bf16,
    AtomicDType::BF16,
    AtomicOperation::Max
);
atomic_helper!(
    seismic_atomic_min_bf16,
    AtomicDType::BF16,
    AtomicOperation::Min
);
atomic_helper!(
    seismic_atomic_add_i32,
    AtomicDType::I32,
    AtomicOperation::Add
);
atomic_helper!(
    seismic_atomic_max_i32,
    AtomicDType::I32,
    AtomicOperation::Max
);
atomic_helper!(
    seismic_atomic_min_i32,
    AtomicDType::I32,
    AtomicOperation::Min
);
atomic_helper!(
    seismic_atomic_add_u32,
    AtomicDType::U32,
    AtomicOperation::Add
);
atomic_helper!(
    seismic_atomic_max_u32,
    AtomicDType::U32,
    AtomicOperation::Max
);
atomic_helper!(
    seismic_atomic_min_u32,
    AtomicDType::U32,
    AtomicOperation::Min
);

/// The host symbols a JIT module resolves: the math sequences and the team
/// barrier entry points.
pub(crate) fn host_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        (
            "seismic_approx_exp_f32",
            seismic_approx_exp_f32 as *const u8,
        ),
        (
            "seismic_approx_log_f32",
            seismic_approx_log_f32 as *const u8,
        ),
        (
            "seismic_approx_sin_f32",
            seismic_approx_sin_f32 as *const u8,
        ),
        (
            "seismic_approx_cos_f32",
            seismic_approx_cos_f32 as *const u8,
        ),
        ("seismic_fmax", seismic_fmax as *const u8),
        ("seismic_fmin", seismic_fmin as *const u8),
        ("seismic_round_f32", seismic_round_f32 as *const u8),
        ("seismic_round_f16", seismic_round_f16 as *const u8),
        ("seismic_round_bf16", seismic_round_bf16 as *const u8),
        ("seismic_round_i32", seismic_round_i32 as *const u8),
        ("seismic_round_u32", seismic_round_u32 as *const u8),
        ("seismic_round_bool", seismic_round_bool as *const u8),
        ("seismic_f16_load", seismic_f16_load as *const u8),
        ("seismic_f16_store", seismic_f16_store as *const u8),
        ("seismic_packed_bits", seismic_packed_bits as *const u8),
        (
            "seismic_atomic_add_f32",
            seismic_atomic_add_f32 as *const u8,
        ),
        (
            "seismic_atomic_max_f32",
            seismic_atomic_max_f32 as *const u8,
        ),
        (
            "seismic_atomic_min_f32",
            seismic_atomic_min_f32 as *const u8,
        ),
        (
            "seismic_atomic_add_f16",
            seismic_atomic_add_f16 as *const u8,
        ),
        (
            "seismic_atomic_max_f16",
            seismic_atomic_max_f16 as *const u8,
        ),
        (
            "seismic_atomic_min_f16",
            seismic_atomic_min_f16 as *const u8,
        ),
        (
            "seismic_atomic_add_bf16",
            seismic_atomic_add_bf16 as *const u8,
        ),
        (
            "seismic_atomic_max_bf16",
            seismic_atomic_max_bf16 as *const u8,
        ),
        (
            "seismic_atomic_min_bf16",
            seismic_atomic_min_bf16 as *const u8,
        ),
        (
            "seismic_atomic_add_i32",
            seismic_atomic_add_i32 as *const u8,
        ),
        (
            "seismic_atomic_max_i32",
            seismic_atomic_max_i32 as *const u8,
        ),
        (
            "seismic_atomic_min_i32",
            seismic_atomic_min_i32 as *const u8,
        ),
        (
            "seismic_atomic_add_u32",
            seismic_atomic_add_u32 as *const u8,
        ),
        (
            "seismic_atomic_max_u32",
            seismic_atomic_max_u32 as *const u8,
        ),
        (
            "seismic_atomic_min_u32",
            seismic_atomic_min_u32 as *const u8,
        ),
        (
            "seismic_cpu_barrier",
            crate::workers::seismic_cpu_barrier as *const u8,
        ),
    ]
}
