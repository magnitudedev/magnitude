//! Per-family decoder block graph construction. `native_target_graph` seals
//! each block by calling its mixer and feed-forward family here.

pub(crate) mod attention;
pub(crate) mod dense;
pub(crate) mod draft;
pub(crate) mod readout;
pub(crate) mod recurrent;
pub(crate) mod routed;

/// The structural form a row class selects in every row-dependent branch of
/// graph construction. The topology code branches on these same predicates,
/// so classes of equal form share node order, edges and exports, and one
/// certified layout can serve all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RowForm {
    attention_decode: bool,
    recurrent_chunked: bool,
    routed_decode: bool,
}

impl RowForm {
    pub(crate) fn of(rows: u64) -> Self {
        Self {
            attention_decode: attention::decodes(rows),
            recurrent_chunked: recurrent::chunked(rows),
            routed_decode: routed::decodes(rows),
        }
    }
}
