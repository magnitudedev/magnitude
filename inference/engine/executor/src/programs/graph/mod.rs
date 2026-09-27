//! Graph construction shared by every operator: the draft abstraction over
//! prepared and checked entries, the readout, and the draft taps. Each
//! operator's own block fragment lives with the operator
//! (`operators::<op>::graph`).

pub(crate) mod draft;
pub(crate) mod readout;
pub(crate) mod tap;

use crate::operators::{attention, gated_delta, routed};

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
            attention_decode: attention::graph::decodes(rows),
            recurrent_chunked: gated_delta::graph::chunked(rows),
            routed_decode: routed::fused_graph::decodes(rows),
        }
    }
}
