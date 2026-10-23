//! Measuring work on a device (specs/26-10-18/device-topology.md §2.6),
//! generically: what is timed is a [`Work`], which this crate submits and
//! reads the seconds of; it knows nothing of devices, kernels or why a
//! caller asks.
//!
//! - **The session** ([`session`]): a placed work sampled round by round,
//!   each sample one submission of its passes queued behind its lead (an
//!   untimed pass, or a work measured alone's sample before it), its
//!   repetitions calibrated so a sample covers a minimum device time.
//!   Tuning's comparisons and every point measured here take their times
//!   through it.
//! - **The clock** ([`clock`]): a session runs at the device's sustained
//!   clock, raised by a gauge of the holder's own work and held per clock
//!   key ([`HeldClocks`]); work without a clock (a CPU) is not raised.
//! - **The statistics** ([`statistics`]): the median and its anytime-valid
//!   interval, the paired sign test a comparison resolves by, and the one
//!   stopping rule ([`Timing::resolved`], [`MEDIAN_TOLERANCE`]).
//! - **Deadlines** ([`Deadline`]): optional work stops at its cutoff.
//! - **Records** ([`measure_all`], [`Record`], [`RecordStore`]): measured
//!   points kept under the caller's keys, through a store the host
//!   implements, read bounded and written atomically ([`JsonFileStore`]).

pub mod clock;
mod deadline;
mod measure;
pub mod session;
pub mod statistics;
mod store;
mod work;

pub use clock::{warm, Clock, HeldClocks, Raise, RaiseOutcome, SustainedClock};
pub use deadline::{Deadline, WorkError};
pub use measure::{
    measure, measure_all, Attempt, Measure, Outcome, Reason, Record, RecordStore, Stop,
    StoreError, Unformed,
};
pub use session::{sample, Counted, Measurement, Passes, Rounds, Slot, SlotFailure, Slotted};
pub use statistics::{
    fewest_bounding, median, median_interval, median_of, Shuffle, Tally, Timing, Verdict,
    MEDIAN_TOLERANCE,
};
pub use store::JsonFileStore;
pub use work::{Gauge, Pending, Work};
