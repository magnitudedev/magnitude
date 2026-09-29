//! The gated short convolution mixer (`Operator::ShortConv`, LFM2,
//! model-family plan §4.2).
//!
//! One layer runs three entries over the block's residual rows:
//!
//! 1. the input RMS and the `B`, `C` and `X` projections in one segmented
//!    launch, `short_conv_project`, publishing F32 rows `u = B⊙X | C`;
//! 2. the depthwise causal convolution over `u`, the `C` gate and the window
//!    publication, `short_conv_rows` (one computation for every row class:
//!    each row is independent);
//! 3. the output projection plus the residual, `attention_output` over one
//!    head of `channels` channels.
//!
//! The bank holds one component, the F32 window of `u` rows, which is also
//! the layer's tape; slots and versions follow the recurrent contract, so
//! the block binds the recurrent control ports.

pub(crate) mod graph;

use crate::error::PlanError;
use magnitude_family_contracts::ShortConv;
use seismic::Element;

/// Every static axis of a short convolution over a `hidden`-wide residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShortConvShape {
    /// `H`: the hidden width.
    pub hidden: u64,
    /// `CH`: convolved channels.
    pub channels: u64,
    /// `C`: convolution taps, the current row included.
    pub convolution_width: u64,
}

impl ShortConvShape {
    /// The shape of an admitted operator.
    pub(crate) fn of(hidden: u64, conv: &ShortConv) -> Result<Self, PlanError> {
        admit(conv)?;
        Ok(Self {
            hidden,
            channels: conv.channels,
            convolution_width: conv.width,
        })
    }

    /// `short_conv_project`'s dimensions.
    pub fn project_dimensions(&self, rows: u64) -> [(&'static str, u64); 3] {
        [("M", rows), ("H", self.hidden), ("CH", self.channels)]
    }

    /// `short_conv_rows`' dimensions for `slots` slots over `banks` banks
    /// with `tape_rows` tape rows each.
    pub fn rows_dimensions(
        &self,
        rows: u64,
        slots: u64,
        banks: u64,
        tape_rows: u64,
    ) -> [(&'static str, u64); 6] {
        [
            ("M", rows),
            ("B", slots),
            ("S", banks),
            ("CH", self.channels),
            ("C", self.convolution_width),
            ("T", tape_rows),
        ]
    }

    /// `attention_output`'s dimensions: one head of `channels`.
    pub fn output_dimensions(&self, rows: u64) -> [(&'static str, u64); 4] {
        [
            ("M", rows),
            ("D", self.hidden),
            ("Q", 1),
            ("W", self.channels),
        ]
    }

    /// The tape rows of a window of `window_rows` rows.
    pub fn tape_rows(&self, window_rows: u64) -> Option<u64> {
        window_rows.checked_sub(self.convolution_width - 1)
    }
}

/// The resident elements of a short convolution's weights that are not
/// fixed F32 (the taps are).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShortConvBinding {
    pub shape: ShortConvShape,
    /// The input RMS weight.
    pub norm: Element,
    /// `B`.
    pub input_gate: Element,
    /// `C`.
    pub output_gate: Element,
    /// `X`.
    pub value: Element,
    pub output: Element,
    pub activation: Element,
}

/// The short convolutions the entries implement: at least one tap before
/// the current row (the window keeps `C - 1` rows).
pub(crate) fn admit(conv: &ShortConv) -> Result<(), PlanError> {
    if conv.width < 2 || conv.channels == 0 {
        return Err(PlanError::Unsupported("short convolution geometry"));
    }
    Ok(())
}

/// The weights the operator binds, in the order its entries consume them.
pub(super) fn weights<'a>(conv: &'a ShortConv, push: &mut super::WeightPush<'_, 'a>) {
    use magnitude_family_contracts::WeightKind;
    push(WeightKind::ShortConvInputGate, &conv.input_gate);
    push(WeightKind::ShortConvValue, &conv.value);
    push(WeightKind::ShortConvOutputGate, &conv.output_gate);
    push(WeightKind::RecurrentConvolution, &conv.convolution);
    push(WeightKind::RecurrentOutput, &conv.output);
}

/// Every numerical parameter of the operator a sealed graph holds apart from
/// its weights.
pub(super) fn shape_key(conv: &ShortConv) -> String {
    format!("short conv {:?}", ShortConvShape::of(0, conv))
}

/// The program binding of a short-convolution sublayer; `lookup` resolves
/// the planned element of a role in its scope.
pub(super) fn binding(
    conv: &ShortConv,
    hidden: u64,
    lookup: impl Fn(magnitude_family_contracts::WeightKind) -> Result<Element, PlanError>,
    activation: Element,
) -> Result<ShortConvBinding, PlanError> {
    use magnitude_family_contracts::WeightKind;
    Ok(ShortConvBinding {
        shape: ShortConvShape::of(hidden, conv)?,
        norm: lookup(WeightKind::InputNorm)?,
        input_gate: lookup(WeightKind::ShortConvInputGate)?,
        output_gate: lookup(WeightKind::ShortConvOutputGate)?,
        value: lookup(WeightKind::ShortConvValue)?,
        output: lookup(WeightKind::RecurrentOutput)?,
        activation,
    })
}
