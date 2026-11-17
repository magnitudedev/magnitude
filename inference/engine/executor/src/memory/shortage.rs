//! The one decision that a store's shortage is real and whether it persists
//! (design §3.3, §4.5). It reads only the ledger's estimate
//! ([`super::ledger::Estimate`]) and its boundary; the scheduler's owner
//! asks at every event ([`super::ledger::Ledger::shortage`]), which answers
//! each reading once, and carries out the response it returns.
//!
//! A shortage is real when, on the estimate:
//! - the budget cannot hold what is committed (a floor shortfall), or what
//!   is committed and others' rise beside it (a rise shortfall);
//! - others' harm rises with no growth of the process's within λ_h;
//! - a new pressure event came with the OS's pressure level set.
//!
//! The response is one relief plan, carried out whole or not at all
//! (`memory::ledger::relief`): cache in rank order, then (but under
//! pressure) requests, the lowest value first and never the earliest
//! admitted, just enough. With no plan nothing yields and new work waits.
//! New work also waits while committed memory or others' rise is short.
//! The model unloads only on a floor shortfall that holds on readings
//! spanning more than the store's λ, with no change of the process's
//! pending and no plan: what is committed cannot be held. Under pressure
//! the boundary never falls below what is committed
//! (`memory::supply::Boundary`), so pressure sheds cache alone.

use super::ledger::{Estimate, Relief, ReliefPlan, Sampled};

/// A real shortage and the response to it.
#[derive(Clone, Debug)]
pub struct Shortage {
    pub cause: Cause,
    pub response: Response,
}

/// Why a shortage is real.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// The budget is `over` bytes short of what is committed: the only cause
    /// a model unloads for.
    FloorShort { over: u64 },
    /// The budget holds what is committed but is `over` bytes short of it
    /// and others' rise beside it: relieved, new work refused, never an
    /// unload.
    RiseShort { over: u64 },
    /// Others brought back `bytes` past their baseline with no growth of
    /// the process's within λ_h.
    Harm { bytes: u64 },
    /// The OS's pressure level, with a new pressure event at the reading:
    /// cache above the boundary B, which never falls below what is
    /// committed.
    Pressure(seismic::PressureLevel),
}

/// What the owner does, in order.
#[derive(Clone, Debug)]
pub struct Response {
    /// Relief known to cover the shortage, carried out whole
    /// ([`super::ledger::Ledger::execute`]); none where nothing covers it.
    pub relief: Option<ReliefPlan>,
    /// New work waits: committed memory or others' rise is short, or no
    /// relief covers the shortage.
    pub admission_waits: bool,
    /// A floor shortfall persists with nothing to give and nothing of ours
    /// pending: unload.
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
/// beside it: the floor first, then the rise, then harm, then pressure.
pub(crate) fn cause(estimate: &Estimate, harmed: Option<u64>, pressed: Option<seismic::PressureLevel>) -> Option<Cause> {
    let (budget, committed) = (estimate.budget(), estimate.committed());
    if let Some(over) = committed.checked_sub(budget).filter(|over| *over > 0) {
        return Some(Cause::FloorShort { over });
    }
    match committed.saturating_add(estimate.guarantee()).checked_sub(budget) {
        Some(over) if over > 0 => Some(Cause::RiseShort { over }),
        _ => harmed.map(|bytes| Cause::Harm { bytes }).or(pressed.map(Cause::Pressure)),
    }
}

/// The bytes `cause` asks relief for on `estimate` (§4.5).
///
/// # Panics
/// Under pressure with no boundary standing: a pressure event sets B
/// (`memory::supply::Boundary`).
pub(crate) fn lacks(estimate: &Estimate, cause: Cause) -> u64 {
    match cause {
        Cause::FloorShort { over } | Cause::RiseShort { over } => over,
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

/// What relief `cause` may take: pressure names no shortfall of committed
/// memory, so it takes cache alone; any other cause, cache and then the
/// `evictable` requests.
pub(crate) fn allows(cause: Cause, evictable: &std::collections::BTreeSet<crate::operation::RequestId>) -> Relief<'_> {
    match cause {
        Cause::Pressure(_) => Relief::CacheOnly,
        Cause::FloorShort { .. } | Cause::RiseShort { .. } | Cause::Harm { .. } => Relief::CacheAndRequests(evictable),
    }
}

/// The response to `cause` on `estimate`, `relief` the plan that covers it
/// (none where nothing does), the shortage having stood since
/// `persistence`'s first reading.
pub(crate) fn respond(estimate: &Estimate, cause: Cause, relief: Option<ReliefPlan>, persistence: &Persistence) -> Shortage {
    let at = estimate.at();
    let spans = persistence.since.is_some_and(|since| at.at().duration_since(since.at()) > estimate.lag());
    let committed_short = matches!(cause, Cause::FloorShort { .. } | Cause::RiseShort { .. });
    let unload = matches!(cause, Cause::FloorShort { .. }) && spans && relief.is_none() && estimate.releasing() == 0 && !estimate.pending();
    let admission_waits = committed_short || relief.is_none();
    Shortage { cause, response: Response { relief, admission_waits, unload } }
}
