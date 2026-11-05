//! Why the host raised a gate a device's work waited at, which a trace of
//! that work keeps: one vocabulary for the backends that record it and the
//! runtime that reports it.

/// Why the host answers a park.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParkAnswer {
    pub by: AnsweredBy,
    /// The park's kind as its owner names it (a trace label, e.g. the slot
    /// its call waits on).
    pub park: &'static str,
    /// The owner's index of where the park is (a streamed layer).
    pub layer: usize,
}

/// Who answered a park, and so what the device's wait at it was for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnsweredBy {
    /// An answerer, as the answer became ready: what the call after the
    /// park reads was put in place just then.
    Ready,
    /// The host on seeing the device arrive at the park: the answer was
    /// ready before (or the call waits for nothing).
    Arrival,
}

/// Why the host raised a gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateOpening {
    /// A park's answer.
    Answered(ParkAnswer),
    /// A plain host signal: a wait node's gate, or a release.
    Signalled,
}
