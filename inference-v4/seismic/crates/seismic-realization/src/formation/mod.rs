//! Private construction state of the closed compiler.
//!
//! Every mutable builder, pending set, union-find, and partially built
//! artifact lives here and nowhere else. Nothing in this module is public;
//! the sealed artifacts in the crate root modules are the only outputs. Each
//! submodule is owned by exactly one package:
//!
//! - `occurrence` (O1): `OccurrenceForest::expand`.
//! - `strategy`   (S1): `StrategyFormer::form` and the universal rule catalog.
//! - `dataflow`   (D1): `DataflowFormer::form`.
//! - `kernel`     (K1): `KernelFormer::form`.
//! - `seal`       (P1): `form_plan_space`, `solver_model`, `resolve`.

pub(crate) mod dataflow;
pub(crate) mod kernel;
pub(crate) mod occurrence;
pub(crate) mod seal;
pub(crate) mod strategy;
