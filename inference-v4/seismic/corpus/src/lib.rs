//! The Seismic end-to-end corpus (design A10 §2.2): scenario files, their
//! header grammar, deterministic argument generation, the construct x context
//! matrix and the library invocation files.
//!
//! This library depends only on `seismic-lang`, so runtime-internal tests
//! (the member sweep) can read the corpus without a dependency cycle. The
//! public-API runner lives in `tests/common`.
pub mod inputs;
pub mod library;
pub mod matrix;
pub mod scenario;
