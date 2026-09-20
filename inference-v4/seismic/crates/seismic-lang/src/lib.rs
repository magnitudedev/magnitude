//! The Seismic language: source syntax checked into the semantic program,
//! typed through the intrinsic registry with canonical types and traversal.
//!
//! `syntax` -> `check` -> `sir` (checked) -> reference interpretation (`interp`).
//! Logical construction and specialization consume the checked program next.

pub mod abi;
pub mod check;
pub mod family;
pub mod intrinsics;
pub mod layout;
pub mod logical;
pub mod numeric;
pub mod precision;
pub mod program;
pub mod repr;
pub mod sir;
pub mod span;
pub mod sym;
pub mod syntax;
pub mod types;

pub mod interp;

pub use span::{Diagnostic, Span};
