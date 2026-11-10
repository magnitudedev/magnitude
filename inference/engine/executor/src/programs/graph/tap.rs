//! Target taps of a separate draft (DFlash, DSpark). A tapped block's graph
//! rounds a residual into its tap's column block of the draft input rows (a
//! program-owned `[rows, taps · hidden]` buffer bound per run) at up to three
//! positions: the residual entering the block, the residual entering its
//! feed-forward sublayer, and the block's output (the last block's output is
//! the decoder's final pre-norm residual). The readout then fuses the taps
//! into the draft's conditioning features. Tap indices are run inputs, so
//! blocks tapped at the same positions share a plan.

use super::{GraphError, draft::GraphDraft};
use magnitude_kernels::tap_rows;
use seismic::{Dim, Element, NativePort, WorkflowTensor};

/// Where a block's graph taps its residual stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct TapPositions {
    /// The residual entering the block.
    pub entry: bool,
    /// The residual entering the block's feed-forward sublayer.
    pub middle: bool,
    /// The block's output residual.
    pub output: bool,
}

impl TapPositions {
    pub fn any(self) -> bool {
        self.entry || self.middle || self.output
    }
}

/// A tapped graph's entry and the positions the block taps.
pub(crate) struct TapEntry<'a, G: GraphDraft + 'a> {
    pub entry: G::Binding<'a, tap_rows::Entry>,
    pub positions: TapPositions,
}

/// The per-run ports of a block's taps: the bound draft input rows and each
/// tapped position's column block index.
#[derive(Clone)]
pub(crate) struct TapPorts {
    pub taps: NativePort,
    pub entry: Option<NativePort>,
    pub middle: Option<NativePort>,
    pub output: Option<NativePort>,
}

/// The draft input rows of a tapped graph, before any position is tapped.
pub(crate) struct Taps<'a, G: GraphDraft + 'a> {
    entry: G::Binding<'a, tap_rows::Entry>,
    /// `T`: the taps the draft input rows hold.
    taps: Dim,
    hidden_width: Dim,
    rows: Dim,
    pub positions: TapPositions,
    pub ports: TapPorts,
}

impl<'a, G: GraphDraft + 'a> Taps<'a, G> {
    /// The taps of `tap` into draft input rows `width` wide.
    pub fn new(
        graph: &mut G,
        tap: TapEntry<'a, G>,
        width: &Dim,
        rows: &Dim,
        hidden_width: &Dim,
        activation: Element,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            entry: tap.entry,
            taps: width.checked_div(hidden_width).ok_or("draft input rows of no hidden width")?,
            hidden_width: hidden_width.clone(),
            rows: rows.clone(),
            positions: tap.positions,
            ports: TapPorts {
                taps: graph.port_with_class_extent(activation, &[rows, width], 0, "M")?,
                entry: None,
                middle: None,
                output: None,
            },
        })
    }

    /// Round `hidden`'s rows into the column block the returned index port
    /// names.
    pub fn tap(
        &mut self,
        graph: &mut G,
        hidden: &WorkflowTensor,
    ) -> Result<NativePort, GraphError> {
        let dimensions = [
            ("M", &self.rows),
            ("D", &self.hidden_width),
            ("T", &self.taps),
        ];
        let index = graph.input_for(self.entry, "index", &dimensions)?;
        graph.enqueue(
            self.entry,
            &dimensions,
            tap_rows::WorkflowArgs {
                residual: hidden.into(),
                index: index.tensor().into(),
                taps: self.ports.taps.tensor_mut().into(),
            },
        )?;
        Ok(index)
    }
}
