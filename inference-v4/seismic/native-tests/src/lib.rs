//! Fixtures of the direct-native route: one checked module whose entries
//! carry Metal, CUDA and CPU implementations with static dimensions, tuning
//! parameters, scratch, several launches and shared memory. The tests run
//! on every backend the host can open.

#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/fixtures.rs"));
