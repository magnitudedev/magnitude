//! Timing work on a device (specs/26-10-18/device-topology.md §2.6),
//! generically: what is timed is a [`Work`], which this crate submits and
//! reads the seconds of; it knows nothing of devices, kernels or why a
//! caller asks.
//!
//! - **The session** ([`session`]): placed works sampled round by round,
//!   each sample one submission of its passes queued behind an untimed
//!   lead, its repetitions calibrated so a sample covers a minimum device
//!   time. Tuning's comparisons take their times through it.
//! - **The clock** ([`clock`]): the device's sustained clock as a holder's
//!   own work reads it, raised before a session ([`Clock`], [`warm`]), or
//!   by a caller that reads its work itself ([`SustainedClock`]).
//! - **The statistics** ([`statistics`]): the median, the paired sign test
//!   a comparison resolves by ([`Tally`]), and the order of a round's
//!   samples ([`Shuffle`]).
//! - **Deadlines** ([`Deadline`]): optional work stops at its cutoff.
//! - **A work's own time** ([`measure`]): its median over the samples its
//!   caller asks for.

pub mod clock;
mod deadline;
mod measure;
pub mod session;
pub mod statistics;
mod work;

pub use clock::{Clock, Raise, RaiseOutcome, SustainedClock, warm};
pub use deadline::{Deadline, WorkError};
pub use measure::{Measure, measure};
pub use session::{Counted, Measurement, Rounds, Slot, SlotFailure, Slotted, sample};
pub use statistics::{Shuffle, Tally, Verdict, median, median_of};
pub use work::{Gauge, Pending, Work};
