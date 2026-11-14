//! The one decision that a store's shortage is real and whether it persists
//! (design §3.3, §4.5). It reads only the ledger's estimate
//! ([`super::ledger::Estimate`]) and its boundary; the scheduler's owner
//! asks at every event ([`super::ledger::Ledger::shortage`]), which answers
//! each reading once, and carries out the response it returns.
//!
//! A shortage is real when, on the estimate:
//! - the budget cannot hold what is committed and the guarantee (the store
//!   cannot hold what is admitted plus one reading's real growth);
//! - others' harm rises with no growth of the process's within λ_h;
//! - a new pressure event came with the OS's pressure level set.
//!
//! The response, in order:
//! 1. cache yields in rank order, at once: what the budget lacks, or the
//!    harm's bytes; under pressure, which names no bytes, what the process
//!    holds above the boundary B (`memory::supply`), which each new
//!    pressure event lowers by one block;
//! 2. admission waits while the bank sits at its floor;
//! 3. named sequences are preempted, the lowest value first and just
//!    enough, never the earliest admitted;
//! 4. unload, only where the shortage holds on readings spanning more than
//!    the store's λ with no change of the process's pending and nothing
//!    left to yield or preempt, and only with a sequence admitted. That
//!    rules out any shortage the process's own action made: its own changes
//!    are shown by then, and the rise term of the guarantee falls to zero
//!    once others stop rising. An idle load never unloads: it holds nothing
//!    to give but its floor, so its first new sequence is refused instead
//!    (`memory::ledger`'s admission).

use crate::operation::RequestId;

use super::ledger::{Estimate, Sampled};

/// A real shortage and the response to it.
#[derive(Clone, Debug)]
pub struct Shortage {
    pub cause: Cause,
    pub response: Response,
}

/// Why a shortage is real.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// The budget is `over` bytes short of what is committed and the
    /// guarantee.
    Room { over: u64 },
    /// Others brought back `bytes` past their baseline with no growth of
    /// the process's within λ_h.
    Harm { bytes: u64 },
    /// The OS's pressure level, with a new pressure event at the reading.
    Pressure(seismic::PressureLevel),
}

/// What the owner does, in order.
#[derive(Clone, Debug)]
pub struct Response {
    /// Cache bytes to yield now, in rank order
    /// ([`super::ledger::Ledger::yield_cache`]).
    pub yields: u64,
    /// Admission waits: the cache yields whole, so the bank sits at its
    /// floor.
    pub admission_waits: bool,
    /// Sequences to preempt, the lowest value first.
    pub preempt: Vec<RequestId>,
    /// The shortage is real, persists and nothing is left to give: unload.
    pub unload: bool,
}

/// A store's shortage across readings: the instant of its first reading
/// short, none while it is not.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Persistence {
    since: Option<Sampled>,
}

impl Persistence {
    /// Fold in the reading at `at`, short by `cause` or not.
    pub(crate) fn observe(&mut self, at: Sampled, cause: Option<Cause>) {
        self.since = cause.map(|_| self.since.unwrap_or(at));
    }
}

/// The cause of a shortage on `estimate`, its boundary's harm and pressure
/// beside it: the room first, then harm, then pressure.
pub(crate) fn cause(estimate: &Estimate, harmed: Option<u64>, pressed: Option<seismic::PressureLevel>) -> Option<Cause> {
    let needed = estimate.committed().saturating_add(estimate.guarantee());
    match needed.checked_sub(estimate.budget()) {
        Some(over) if over > 0 => Some(Cause::Room { over }),
        _ => harmed.map(|bytes| Cause::Harm { bytes }).or(pressed.map(Cause::Pressure)),
    }
}

/// Cache bytes `cause` asks of `estimate` before admission waits or
/// sequences preempt (§4.5).
///
/// # Panics
/// Under pressure with no boundary standing: a pressure event sets B
/// (`memory::supply::Boundary`).
pub(crate) fn lacks(estimate: &Estimate, cause: Cause) -> u64 {
    match cause {
        Cause::Room { over } => estimate.cache().saturating_add(over),
        Cause::Harm { bytes } => bytes,
        // What the process holds above B, its yielded bytes still releasing
        // left out: sized from the boundary's state, so a later answer
        // yields only what B lowered since.
        Cause::Pressure(_) => {
            let cap = estimate.cap().expect("invariant violated: a pressure event set no boundary");
            estimate.allocated().saturating_sub(estimate.releasing()).saturating_sub(cap)
        }
    }
}

/// The response to `cause` on `estimate`, `preemptible` the sequences a
/// shortage may preempt with their state (lowest value first), the
/// shortage having stood since `persistence`'s first reading, `admitted`
/// whether any sequence is admitted in the store.
///
/// # Panics
/// Under pressure with no boundary standing: a pressure event sets B
/// (`memory::supply::Boundary`).
pub(crate) fn respond(estimate: &Estimate, cause: Cause, preemptible: &[(RequestId, u64)], admitted: bool, persistence: &Persistence) -> Shortage {
    let cache = estimate.cache();
    let lacks = lacks(estimate, cause);
    let yields = lacks.min(cache);
    let mut short = lacks - yields;
    let mut preempt = Vec::new();
    for (sequence, bytes) in preemptible {
        if short == 0 {
            break;
        }
        short = short.saturating_sub(*bytes);
        preempt.push(*sequence);
    }
    let at = estimate.at();
    let spans = persistence.since.is_some_and(|since| at.at().duration_since(since.at()) > estimate.lag());
    let exhausted = cache == 0 && estimate.releasing() == 0 && preemptible.is_empty();
    let unload = admitted && spans && exhausted && !estimate.pending();
    Shortage { cause, response: Response { yields, admission_waits: yields == cache, preempt, unload } }
}
