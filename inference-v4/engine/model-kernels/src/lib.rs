//! Checked, generated bindings for the engine-owned numerical kernel catalog.
//!
//! The build script links `kernels` with `seismic-std`, checks the whole
//! source module once, and emits this typed surface.  Engine code uses these
//! entry modules exclusively; it never assembles compiler or runtime bindings.

#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
