//! What is timed: a [`Work`], which submits passes of itself to a device
//! and reports their seconds. The runtime implements it for its measurable
//! execution features (a formed kernel's rotation of calls, a sealed
//! graph's runs); this crate knows nothing of devices.

use std::hash::Hash;

/// Submitted passes of a [`Work`], queued and not waited on; for work timed
/// on the host, already complete ([`Work::submit`]).
pub trait Pending<E> {
    /// The seconds of the submission, once it completes (waiting for it):
    /// device time where the work is timed on the device.
    fn seconds(&self) -> Result<f64, E>;
    /// Wait for the submission and report its outcome.
    fn wait(&self) -> Result<(), E>;
}

/// Work a held clock gauges with: a fresh placement of the work it was made
/// from ([`Work::gauge`]), submitting passes the same way.
pub trait Gauge<E> {
    /// Submit `passes` passes as one unit of work, without waiting.
    fn submit(&self, passes: usize) -> Result<Box<dyn Pending<E>>, E>;
}

/// Work this crate times: passes of it are submitted as one unit and
/// timed, each sample queued behind a lead of the same work.
pub trait Work {
    type Error;
    type Pending: Pending<Self::Error>;
    /// What identifies the sustained clock the work runs under: the
    /// physical device's, shared by every executor of it.
    type Clock: Eq + Hash + Clone;
    /// Submit `passes` passes over the work as one unit, without waiting.
    /// Work timed on the host (from its submission to its completion) runs
    /// its passes before this returns: its pending passes are complete, and
    /// a session's lead before a sample (an untimed pass, or the sample
    /// before it) has completed when the sample is submitted.
    fn submit(&self, passes: usize) -> Result<Self::Pending, Self::Error>;
    /// The calls one pass makes: a sample's seconds per call divide by it.
    fn calls_per_pass(&self) -> usize;
    /// Submit `passes` cold passes as one unit, without waiting: each is
    /// the work's calls, then the calls of its copies (the same work over
    /// storage of its own), so that no call finds its bytes left in a cache
    /// by an earlier call of the submission, as served work does not (the
    /// session's module documentation, "What a pass reads"). Work without
    /// copies submits its passes.
    fn submit_cold(&self, passes: usize) -> Result<Self::Pending, Self::Error>;
    /// The calls one cold pass makes: the work's, then its copies'.
    fn cold_calls_per_pass(&self) -> usize;
    /// The clock the work runs under; none where no clock is raised (a CPU,
    /// whose raised and unraised times read within the tolerance).
    fn clock(&self) -> Option<Self::Clock>;
    /// A fresh placement of the same work, for a clock to gauge with: a
    /// gauge's work is its own, separate from what it raises the device
    /// for.
    fn gauge(&self) -> Result<Box<dyn Gauge<Self::Error>>, Self::Error>;
}

impl<W: Work> Work for &W {
    type Error = W::Error;
    type Pending = W::Pending;
    type Clock = W::Clock;
    fn submit(&self, passes: usize) -> Result<Self::Pending, Self::Error> {
        (**self).submit(passes)
    }
    fn calls_per_pass(&self) -> usize {
        (**self).calls_per_pass()
    }
    fn submit_cold(&self, passes: usize) -> Result<Self::Pending, Self::Error> {
        (**self).submit_cold(passes)
    }
    fn cold_calls_per_pass(&self) -> usize {
        (**self).cold_calls_per_pass()
    }
    fn clock(&self) -> Option<Self::Clock> {
        (**self).clock()
    }
    fn gauge(&self) -> Result<Box<dyn Gauge<Self::Error>>, Self::Error> {
        (**self).gauge()
    }
}
