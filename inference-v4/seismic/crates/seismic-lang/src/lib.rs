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
pub mod program;
pub mod repr;
pub mod sir;
pub mod span;
pub mod sym;
pub mod syntax;
pub mod types;

pub use span::{Diagnostic, Span};

/// The scope a file's extension selects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    Portable,
    Backend(String),
}

/// Parse a file name of the form `<name>.seismic.<scope>`.
pub fn scope_of_path(path: &str) -> Option<Scope> {
    let file = path.rsplit('/').next()?;
    let (_, rest) = file.split_once(".seismic.")?;
    if rest.is_empty() || rest.contains('.') {
        return None;
    }
    Some(if rest == "portable" { Scope::Portable } else { Scope::Backend(rest.to_string()) })
}
