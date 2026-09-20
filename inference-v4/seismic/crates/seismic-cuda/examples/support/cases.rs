//! Kernel cases of the CUDA sweeps: the canonical std sweep table from `seismic-std`
//! (`seismic_std::sweep`), shared with `seismic-runtime`'s gate sweep so kernel coverage
//! cannot drift between consumers. Device-facing helpers stay local to the CUDA examples.
use seismic_lang::family::Workload;
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::repr;
use seismic_lang::sir::{Definition, Program};
use seismic_lang::types::{DType, Elem};

pub use seismic_std::sweep::{cases, packed_cases, Case};

pub fn element(name: &str) -> Result<Elem, String> {
    match DType::from_name(name) {
        Some(dtype) => Ok(Elem::Dtype(dtype)),
        None if repr::lookup(name).is_some() => Ok(Elem::Repr(name.into())),
        None => Err(format!("unknown element {name}")),
    }
}

pub fn workload(case: &Case, precision: PrecisionPolicy) -> Result<Workload, String> {
    Ok(Workload {
        shapes: case
            .shapes
            .iter()
            .map(|(n, v)| (n.to_string(), *v))
            .collect(),
        elems: case
            .elems
            .iter()
            .map(|(n, e)| Ok((n.to_string(), element(e)?)))
            .collect::<Result<_, String>>()?,
        precision,
    })
}

/// A family definition whose parameters are the entry's invocation ABI.
#[allow(dead_code)]
pub fn abi<'a>(program: &'a Program, entry: &str) -> Result<&'a Definition, String> {
    let family = program.resolve_family(entry)?;
    family
        .bodies
        .iter()
        .chain(&family.lowerings)
        .map(|id| program.definition(*id))
        .next()
        .ok_or_else(|| format!("{entry} has no implementation"))
}

/// seismic-std checked for the CUDA target.
pub fn program() -> Result<Program, String> {
    seismic_lang::program::compile(&seismic_std::sources()).map_err(|errors| {
        errors
            .iter()
            .map(|e| e.render())
            .collect::<Vec<_>>()
            .join("\n")
    })
}

/// `PRECISION=unconstrained` enables exploration instead of the default exact policy.
pub fn precision() -> Result<PrecisionPolicy, String> {
    match std::env::var("PRECISION").as_deref() {
        Err(_) | Ok("exact") => Ok(PrecisionPolicy::Exact),
        Ok("unconstrained") => Ok(PrecisionPolicy::Unconstrained),
        Ok(other) => Err(format!(
            "PRECISION={other}: expected `exact` or `unconstrained`"
        )),
    }
}

/// The complete canonical table, kept by `KERNELS=<substring>` and not dropped by
/// `KERNELS_SKIP=<substring>`.
pub fn selected() -> Vec<Case> {
    let (filter, skip) = (
        std::env::var("KERNELS").ok(),
        std::env::var("KERNELS_SKIP").ok(),
    );
    cases()
        .into_iter()
        .chain(packed_cases())
        .filter(|c| {
            filter.as_ref().is_none_or(|f| c.label.contains(f.as_str()))
                && !skip.as_ref().is_some_and(|s| c.label.contains(s.as_str()))
        })
        .collect()
}

/// Outcome of one case with panics reported as errors (the pipeline is under test).
pub fn guarded<T>(run: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|panic| {
        let text = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()));
        Err(format!(
            "panic: {}",
            text.unwrap_or_else(|| "non-string payload".into())
        ))
    })
}
