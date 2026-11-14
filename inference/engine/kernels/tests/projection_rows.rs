//! The row thresholds `common::bound` models a backend's projections by are
//! the ones `dense.seismic` declares: the most rows its GEMV launches serve.

mod common;

use common::bound::gemv_rows;
use seismic::BackendName;

/// The declarations the model's thresholds are pinned to, embedded so the
/// test names no source tree at run time.
const DENSE: &str = include_str!("../kernels/dense.seismic");

/// `project_rows`'s native for `backend` in `dense.seismic`.
fn native(backend: &str) -> String {
    let header = format!("native project_rows for {backend} ");
    let start = DENSE.find(&header).unwrap_or_else(|| panic!("no `{header}`"));
    let rest = &DENSE[start + header.len()..];
    rest[..rest.find("\nnative ").unwrap_or(rest.len())].to_owned()
}

#[test]
fn the_models_gemv_row_thresholds_are_the_declared_launches() {
    // The first launch past the GEMV band: Metal's and CUDA's small GEMM
    // (`M > 16`), Vulkan's GEMM (`M > 8`).
    for (backend, name) in [
        (BackendName::Metal, "metal"),
        (BackendName::Vulkan, "vulkan"),
        (BackendName::Cuda, "cuda"),
    ] {
        let declared = native(name);
        let past = format!("when M > {} ", gemv_rows(backend));
        let past_end = format!("when M > {}:", gemv_rows(backend));
        assert!(
            declared.contains(&past) || declared.contains(&past_end),
            "{backend:?}: no launch starts past {} rows",
            gemv_rows(backend)
        );
    }
}
