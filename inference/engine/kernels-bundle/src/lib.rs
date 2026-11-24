//! Embedded checked Seismic bundle and bundle identity for the engine-owned
//! numerical kernel catalog.
//!
//! Split from `magnitude-kernels` so GPU shader edits recompile only this
//! ~30-line crate and leave downstream `.rmeta` files byte-identical.

include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
