//! Authoritative executable IR and its coordinated construction owner.
//! No compiler, native backend, solver, or runtime dependency.

pub mod construction;
pub mod execution;
pub mod identity;
pub mod kernel;
pub mod repr;
pub mod schedule;
pub mod specialization;
pub mod storage;
pub mod target;

pub mod metal;
