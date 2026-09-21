//! The CPU backend: effective target profile, mapping catalog, vacuous
//! intrinsic catalog, exhaustive mechanical encoder over sealed launches,
//! Cranelift native assembler, and the executor over one prepared
//! invocation.
//!
//! There is no retry, candidate, or repair API: the encoder is total over
//! the sealed kernel algebra, assembly validates the reflected facts
//! against the selected resource contract and seals one native tree that
//! mirrors the physical schedule exactly once, and execution validates
//! nothing — the invocation was validated once by preparation; the executor
//! evaluates retained geometry, skips zero-work launches, distributes
//! participants across the worker pool in retained order, and reports the
//! first recorded safety error after synchronous completion.

mod buffer;
pub mod catalog;
pub mod codegen;
pub mod encode;
pub mod intrinsics;
pub mod mapping;
pub mod native;
pub mod runtime;
mod workers;

pub use buffer::Buffer;
pub use intrinsics::{CpuDialect, CpuIntrinsic};
pub use mapping::{Cpu, Limits, TARGET};
pub use native::NativeArtifact;
pub use runtime::{CpuExecutor, ExecutionInputs, InvocationOutputs, ScalarOutput};
pub use workers::Workers;

// ---------------------------------------------------------------------------
// The versioned host sequences (the software math reference)
// ---------------------------------------------------------------------------

/// The `seismic_math` sequences: binary64 host evaluation rounded once.
/// These are the same code paths the reference interpreter runs, so the
/// reference bits agree by construction.
extern "C" fn seismic_exp_f64(x: f64) -> f64 {
    x.exp()
}
extern "C" fn seismic_log_f64(x: f64) -> f64 {
    x.ln()
}
extern "C" fn seismic_sin_f64(x: f64) -> f64 {
    x.sin()
}
extern "C" fn seismic_cos_f64(x: f64) -> f64 {
    x.cos()
}
extern "C" fn seismic_fmax(a: f64, b: f64) -> f64 {
    a.max(b)
}
extern "C" fn seismic_fmin(a: f64, b: f64) -> f64 {
    a.min(b)
}
extern "C" fn seismic_fmod(a: f64, b: f64) -> f64 {
    a % b
}
extern "C" fn seismic_round_to(tag: i32, x: f64) -> f64 {
    let dtype = match tag {
        0 => seismic_lang::types::DType::F32,
        1 => seismic_lang::types::DType::BF16,
        2 => seismic_lang::types::DType::F16,
        3 => seismic_lang::types::DType::I32,
        4 => seismic_lang::types::DType::U32,
        _ => seismic_lang::types::DType::Bool,
    };
    seismic_lang::interp::round_to(dtype, x)
}
extern "C" fn seismic_f16_load(bits: i32) -> f32 {
    seismic_lang::numeric::f16_to_f32((bits & 0xffff) as u16)
}
extern "C" fn seismic_f16_store(x: f32) -> i32 {
    seismic_lang::numeric::f16_bits(x) as i32
}

/// The host symbols the JIT resolves (the versioned sequences). The
/// identity and version of this set are the CPU capability fingerprint's
/// software-math component (`mapping::SEISMIC_MATH_IDENTITY`).
pub(crate) fn host_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        ("seismic_exp_f64", seismic_exp_f64 as *const u8),
        ("seismic_log_f64", seismic_log_f64 as *const u8),
        ("seismic_sin_f64", seismic_sin_f64 as *const u8),
        ("seismic_cos_f64", seismic_cos_f64 as *const u8),
        ("seismic_fmax", seismic_fmax as *const u8),
        ("seismic_fmin", seismic_fmin as *const u8),
        ("seismic_fmod", seismic_fmod as *const u8),
        ("seismic_round_to", seismic_round_to as *const u8),
        ("seismic_f16_load", seismic_f16_load as *const u8),
        ("seismic_f16_store", seismic_f16_store as *const u8),
    ]
}
