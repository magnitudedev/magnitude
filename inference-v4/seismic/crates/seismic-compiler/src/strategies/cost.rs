//! Backend-calibrated cost identity plumbing.
//!
//! Every strategy exposes a `CostExpr` (a symbolic expression over the
//! family's plan parameters) together with a `CostModelId` naming the cost
//! model that produced it. Metal and CUDA register *measured* model
//! identities with provenance through their dialect surfaces;
//! an uncalibrated identity affects ranking only and never legality.
//! This module owns the identity type and the shared expression helpers — it
//! never decides a cost itself.

use seismic_lang::sym::Sym;

/// Identity and provenance of the cost model behind a strategy's `CostExpr`.
/// `measured: true` records are registered by the backend that ran the
/// measurement; the strategy library only carries the identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CostModelId {
    /// Backend the model was measured on (`"metal"`, `"cuda"`, …).
    pub backend: String,
    /// Model name/revision (e.g. `"metal-m4pro-timing-v3"`).
    pub model: String,
    /// Provenance of the measurement (device, date, harness).
    pub provenance: String,
    /// Whether the model is measured (`false` = analytic/uncalibrated).
    pub measured: bool,
}

impl CostModelId {
    /// An uncalibrated analytic identity: ranking input only, never legality.
    pub fn uncalibrated(backend: impl Into<String>) -> Self {
        CostModelId {
            backend: backend.into(),
            model: "analytic".into(),
            provenance: "seismic-compiler::strategies".into(),
            measured: false,
        }
    }

    /// A measured model identity. Registered by the owning backend stream
    /// (E/F/G); the strategy library accepts it as an input and attaches it
    /// to every receipt it produces.
    pub fn measured(
        backend: impl Into<String>,
        model: impl Into<String>,
        provenance: impl Into<String>,
    ) -> Self {
        CostModelId {
            backend: backend.into(),
            model: model.into(),
            provenance: provenance.into(),
            measured: true,
        }
    }
}

/// Cost of one launch dominated by a fixed per-launch constant (dispatch
/// overhead): `launches * constant`.
pub fn launch_overhead(launches: &Sym, per_launch: i64) -> Sym {
    launches.scale(per_launch)
}

/// Cost of a traversal over `total` elements at `per_element` units, split
/// across `participants`: `total * per_element` (work is conserved; the
/// participant count enters through the launch/occupancy terms, not the
/// elementwise work).
pub fn elementwise_work(total: &Sym, per_element: i64) -> Sym {
    total.scale(per_element)
}

/// Number of fixed-width windows covering `total` at width `width`:
/// `ceil(total / width)` as a symbolic expression (`total` is the checked
/// capacity bound, always ≥ 1; the retained runtime total is always ≤ it).
pub fn window_count(total: &Sym, width: &Sym) -> Sym {
    let one = Sym::constant(1);
    total.add(&width.sub(&one)).quot(width)
}
