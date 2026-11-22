//! Checked, generated bindings for the development-only machine line probe
//! kernels (`line/`), separated from `magnitude-kernels` so served engines and
//! `magnitude-harness` share one `magnitude-kernels` build.

#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/line.rs"));
