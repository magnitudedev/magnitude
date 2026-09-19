//! The execution IR: the concrete, fully instantiated program a backend realizes.
//!
//! `instantiate` is its only producer. Every structural extent, region result and
//! selected implementation has already been resolved; nothing here is a choice.
//! `ir` holds the typed nodes, `lowered_ir` one instantiated entry, `types` their
//! types, `verify` the structural checks, `normalize` the canonical forms backends
//! share, and `effects`/`writes`/`demand` read-only analyses over the nodes.
//! `reduction` is the numerical contract of `reduce`.
pub mod demand;
pub mod effects;
pub mod ir;
pub mod lowered_ir;
pub mod normalize;
pub mod reduction;
pub mod types;
pub mod verify;
pub mod writes;
