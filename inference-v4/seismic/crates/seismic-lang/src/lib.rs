//! The Seismic language: parser, type/effect checker, the opaque
//! `CheckedModule`, the checked-bundle boundary, `LogicalEntry`/`CallSchema`,
//! typed registry identities, and the one symbolic expression arena.
//!
//! Public checked semantics are the opaque `checked`/`bundle` boundary and
//! the read-only `entry` graph. Primitive enums remain in `intrinsics`; the
//! string-keyed intrinsic and representation tables are private registry
//! construction details. The semantic oracle is a validation consumer of the
//! same entry graph, never a production execution path.

pub mod bundle;
pub mod checked;
pub mod entry;
pub mod expr;
pub mod ids;
pub mod registry;

mod check;
pub mod intrinsics;
pub mod precision;
pub mod reference_math;
pub(crate) mod repr;
pub mod span;
pub mod syntax;
pub mod types;

pub mod interp;

pub use span::{Diagnostic, Span};
