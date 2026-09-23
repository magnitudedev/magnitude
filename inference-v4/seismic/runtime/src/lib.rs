//! Generic device, tensor, and prepared-kernel machinery over the frozen
//! backend contracts (spec §2.3, §12, §24.1 R9). No planning lives here.
//!
//! The public surface is [`devices`] (the one device catalog, identities,
//! memory pools and scoped observations) and [`api`] (the opened device, the
//! tensor, and the call driver) which the public `seismic` crate composes.
//! Everything else is private:
//!
//! - `memory`: per-pool allocation ledgers and per-device limits.
//! - `driver`: the backend-generic opened device, preparation cache, authored
//!   native route, and the narrow primitive that issues one already-admitted
//!   node.
//! - `workflow`: pure descriptor binding followed by graph-wide atomic
//!   resource admission.
//! - `execution`: consuming submission and completion ownership for an
//!   already-admitted run, including closed-output decoding.
//! - `backends`: the closed sum over backends, the one place backend crates
//!   are named.
//! - `layout`: the canonical dense layout of a representation.
//! - `telemetry`: OpenTelemetry spans and metrics at the public boundaries.
//!
//! Runtime duties: bind dependency-closed workflows before allocation, retain
//! their admitted resources through completion, evaluate chosen
//! guards/durations/layout/geometry, submit owned
//! asynchronous work, report data checks and external failures, and emit
//! telemetry. It never infers placement, matches
//! value kinds, reconciles joins, clamps copies, validates geometry against
//! limits, chooses an alternative after a failure, compiles on call, or
//! interprets the portable body.

pub mod api;
pub mod devices;

mod backends;
mod driver;
mod execution;
mod layout;
#[cfg(target_os = "macos")]
pub mod native_graph;
mod memory;
mod resources;
mod telemetry;
