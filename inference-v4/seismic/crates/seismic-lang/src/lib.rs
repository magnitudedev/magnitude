//! The Seismic language: authored execution structure, checked into a structured
//! IR, selected jointly, then instantiated into the execution IR.
//!
//! `syntax` -> `check` -> `sir` -> (`interp` | `family` -> selection -> `instantiate` -> `exec`).

pub mod abi;
pub mod check;
pub mod exec;
pub mod family;
pub mod instantiate;
pub mod interp;
pub mod intrinsics;
pub mod layout;
pub mod numeric;
pub mod precision;
pub mod program;
pub mod repr;
pub mod sir;
pub mod span;
pub mod sym;
pub mod syntax;
pub mod types;

pub use span::{Diagnostic, Span};
