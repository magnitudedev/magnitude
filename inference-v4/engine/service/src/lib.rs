//! Family-neutral request scheduling and execution ownership.
//!
//! The crate observes logical generation work, byte-denominated capacity, and
//! completion events. Model-family execution and protocol publication remain
//! in composition crates above this boundary.

pub mod domain;
pub mod owner;
pub mod policy;
pub mod protocol;
pub mod publication;
pub mod retention;
pub mod round_driver;
pub mod worker;

pub use policy::ServiceLimits;
