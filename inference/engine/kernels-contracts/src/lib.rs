//! Compiled per-backend contract helper functions (`admit`, `default_params`,
//! `sizing`, `scratch_configs`, `active_launches`, etc.) for `magnitude-kernels`.
//!
//! Compiled at `opt-level = 0` in `dev` (`opt-level = 3` in `release`),
//! backend-gated per `target_os`, and invalidated only by `.seismic` edits.

#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
