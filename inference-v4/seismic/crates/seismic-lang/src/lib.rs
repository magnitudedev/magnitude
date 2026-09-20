//! The Seismic language: authored execution structure checked into semantic and
//! logical programs for constructive physical compilation.
//!
//! `syntax` -> `check` -> `sir` -> logical specialization -> physical planning.

pub mod abi;
pub mod check;
pub mod family;
pub mod interp;
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

pub use span::{Diagnostic, Span};
