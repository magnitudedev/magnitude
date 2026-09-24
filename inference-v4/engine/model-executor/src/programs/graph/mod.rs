//! Per-family decoder block graph construction. `native_target_graph` seals
//! each block by calling its mixer and feed-forward family here.

pub(crate) mod attention;
pub(crate) mod dense;
pub(crate) mod readout;
pub(crate) mod recurrent;
pub(crate) mod routed;
