//! Compiled CPU native SIMD kernel launchers for `magnitude-kernels`.
//!
//! Only this kernel crate is compiled at `opt-level = 3` in `dev`; its build
//! script watches only `.seismic` declarations and `kernels/cpu/**`.

#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
