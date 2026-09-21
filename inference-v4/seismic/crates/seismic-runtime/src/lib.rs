//! Generic device, tensor, and prepared-kernel machinery over the frozen
//! backend contracts (spec §2.3, §12, §24.1 R9). No planning lives here.
//!
//! The public surface is [`api`]: the device catalog, the opened device,
//! the tensor, and the call driver the public `seismic` crate composes.
//! Everything else is private:
//!
//! - `driver`: the runtime generic over one backend `B: Backend` — the
//!   opened device with its service, device contract, execution profile,
//!   concurrency-safe submission factory, and preparation cache; workflow
//!   closure; joint invocation admission; execution and result decoding.
//! - `backends`: the closed sum over backends, the one place backend crates
//!   are named.
//! - `layout`: the canonical dense layout of a representation.
//! - `telemetry`: OpenTelemetry spans and metrics at the public boundaries.
//!
//! Runtime duties: bind dependency-closed workflows, atomically reserve their
//! resources, evaluate chosen guards/durations/layout/geometry, submit owned
//! asynchronous work, report data checks and external failures, and emit
//! telemetry. It never infers placement, matches
//! value kinds, reconciles joins, clamps copies, validates geometry against
//! limits, chooses an alternative after a failure, compiles on call, or
//! interprets the portable body.

pub mod api;

mod backends;
mod driver;
mod layout;
mod telemetry;
