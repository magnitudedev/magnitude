//! The ledger: the one record of what the process holds in each store.
//!
//! - **One ledger per store.** Unified (or device) memory, and host RAM
//!   where it is separate. Every byte the process holds there is one
//!   [`Entry`]: an owner kind, its bytes, its [`Holding`] (committed or
//!   cache), and how many of its bytes its holder has allocated.
//! - **Claimed, allocated, released.** Every allocation takes its entry
//!   first, admitted by comparing only its new bytes against free room;
//!   Seismic's allocation charging the entry is the one path that moves
//!   what is allocated ([`seismic::memory::Entry`]: take, took, freed), so
//!   one allocation counts once; recording a claim or a cache holding moves
//!   only what is held, and dropping the entry frees what it still holds.
//! - **One charge per pool block.** A store holds a granule pool of each
//!   class (`PoolClass`: the state's slabs, the bank's records), each at its
//!   own granule. The backend charges each block of a pool once, to that
//!   pool's entry ([`Ledger::pool_block`]), which carries what is allocated
//!   for every granule of it. Bank slots and KV slabs are holdings over
//!   their own class's granules that allocate nothing of their own: moving
//!   a granule between holdings of one class moves what is held and never
//!   what is allocated, and granules no holding of the class holds are
//!   spare. Memory moves between classes only through the OS: one pool
//!   returns blocks (`seismic::Pool::return_freed`), the other grows. Seismic never re-points a charge. An
//!   allocation outside the pool (the arena, weights, driver objects) is
//!   charged to its own claim's entry ([`Claim::entry`]). A reading
//!   attributes to others only what the process's allocated bytes do not
//!   account for, and free room excludes every entry, allocated or not: an
//!   entry taken but not yet allocated never reads as free, and memory
//!   allocated under an entry never reads as others'.
//! - **The only record.** An allocation is backed by an entry or it cannot
//!   be made: committed memory exists only as a [`Claim`], taken by
//!   [`Ledger::claim`] from a [`Need`], and cache only as a [`Cached`],
//!   taken by [`Ledger::cache`]. The allocation paths (the granule pool's
//!   blocks, the arena, weights) take one of them as their right to
//!   allocate, and dropping it is the free.
//! - **Every byte to its owner at the reading's instant.** A reading is
//!   one sample of the OS at one instant ([`Sample`]), taken off the
//!   owner's thread and paired with the ledger later. The ledger journals
//!   its own allocation changes (`memory::journal`), each with its instant
//!   and its place among the scope's changes, and pairs a reading with what
//!   the process had allocated as the reading shows it: its allocations
//!   now, less every change the sample does not show yet (made after it, or
//!   within the store's lag before it). An allocation change is its OS
//!   effect and its record run as one change the ledger owns ([`Changing`]):
//!   every record of one takes it, and no sample beside one stands. So
//!   `others = used − allocated_visible`, of which only the bytes in RAM
//!   count (those out of it are `Estimate::nonresident` and
//!   `Estimate::unobserved_resident`), and the process's own action never
//!   moves it.
//!   Bytes a GPU fence still reads stay held in their own entry until the
//!   fence completes and the free returns: the holder drops its handle
//!   then. So what the ledger holds covers them.
//! - **Allocated is what exists.** An entry's allocated bytes are memory
//!   that exists for it: a bank's growth is allocated as the pool backs its
//!   slots ([`Ledger::backing`]), and cache a claim outside the pool takes
//!   in process stays allocated, releasing, until Seismic frees it through
//!   the entry that allocated it: the claim allocates those bytes only then
//!   ([`Estimate::allocatable`]). Seismic allocates only through the
//!   entries' [`seismic::memory::Entry`] and the stores' [`seismic::memory::Store`]: every
//!   byte it takes, block slack and driver objects included, is an entry's.
//! - **One OS input.** The ledger reads the OS only through
//!   [`super::supply::Reading`]: a store's total, its supply, its wired
//!   ceiling and its paging signals at one sample. A failed reading is an
//!   invariant violation in the reader, naming the figure.
//! - **One estimate per decision.** Every figure a decision reads comes from
//!   one [`Estimate`]: the entries and the reading at one instant, so no
//!   decision mixes readings. The budget is `min(grant, in_use + supply)`,
//!   `in_use` the process's `allocated_visible` the store's use holds; the
//!   guarantee others' rise; the bank's target
//!   what the budget leaves beside them, within the boundary
//!   (`memory::supply::Boundary`).
//!
//! Sizes are never computed here. A [`Need`] names what is needed and with
//! what inputs; the page rule ([`magnitude_state::state_bytes`]) sizes state inside
//! [`Ledger::reserve`] through the one private `size`,
//! the same functions plan and assessment predict with.
//!
//! No OS is queried here: readings are handed in.

mod admission;
mod changes;
mod charges;
mod claims;
mod estimate;
mod grant;
mod relief;
mod reservation;
mod state;
mod view;
mod vocabulary;
mod pressure;
pub use pressure::PressureWatch;
mod store;
pub use store::*;

pub use admission::{Admit, Over, Short};
pub use changes::{Changes, Changing, OpenChange, Owner, Sample, Sampled};
pub use claims::{ArenaClaim, Cached, Claim};
pub use estimate::Estimate;
pub use grant::{Footprints, Grant, SystemFootprints};
pub use relief::{Owed, Relief, ReliefPlan};
pub use reservation::Reservation;
pub use view::{ChargeClasses, Ledger, Standing};
pub use vocabulary::{CacheKind, CommittedKind, Entry, GrantRole, Holding, LoadShape, Need, PoolClass, StoreShape, Value};

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::planning::BankTier;
#[cfg(test)] use crate::operation::RequestId;

use state::{State, Store};
use super::supply::Reading;
#[cfg(test)] use super::shortage;

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod dynamics_tests;
