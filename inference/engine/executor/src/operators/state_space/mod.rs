//! The Mamba-2 selective state-space mixer (`Operator::StateSpace`,
//! model-family plan §4.6).
//!
//! One layer runs four entries over the block's residual rows:
//!
//! 1. the input RMS and the whole `z | x | B | C | dt` projection row,
//!    `attention_project` with only its query segment (the K1 projection with
//!    an RMS prologue; the other segments are empty);
//! 2. the state advance over the layer's bank, `state_space_step` for row
//!    classes below [`CHUNKED_ROWS`](crate::operators::gated_delta::graph::CHUNKED_ROWS),
//!    `state_space_chunk` from there;
//! 3. the SiLU(z) gate and the grouped RMS, `state_space_gate`;
//! 4. the output projection plus the residual, `attention_output` over
//!    `heads` heads of `head_width` channels.
//!
//! The bank holds three components in the gated delta's order (window,
//! state, tape) and follows the same slot and version contract, so the
//! block's state and control ports are the recurrent ones.

pub(crate) mod graph;

use crate::error::PlanError;
use magnitude_family_contracts::StateSpace;
use seismic::Element;

/// Every static axis of a state-space mixer over a `hidden`-wide residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StateSpaceShape {
    /// `D`: the hidden width.
    pub hidden: u64,
    /// `NH`.
    pub heads: u64,
    /// `P`: channels per head.
    pub head_width: u64,
    /// `N`: state columns per head channel.
    pub state: u64,
    /// `G`: groups of heads sharing `B` and `C`.
    pub groups: u64,
    /// `C`: convolution taps, the current row included.
    pub convolution_width: u64,
    /// `U`: heads per output norm group.
    pub norm_heads: u64,
}

impl StateSpaceShape {
    /// The shape of an admitted operator.
    pub(crate) fn of(hidden: u64, space: &StateSpace) -> Result<Self, PlanError> {
        admit(space)?;
        Ok(Self {
            hidden,
            heads: space.heads,
            head_width: space.head_width,
            state: space.state,
            groups: space.groups,
            convolution_width: space.convolution_width,
            norm_heads: space.norm_group / space.head_width,
        })
    }

    pub fn inner(&self) -> u64 {
        self.heads * self.head_width
    }

    /// Convolved channels: `x | B | C`.
    pub fn channels(&self) -> u64 {
        self.inner() + 2 * self.groups * self.state
    }

    /// `J`: the projection row, `z | x | B | C | dt`.
    pub fn projection_width(&self) -> u64 {
        self.inner() + self.channels() + self.heads
    }

    /// Output norm groups.
    pub fn norm_groups(&self) -> u64 {
        self.heads / self.norm_heads
    }

    /// `attention_project`'s dimensions: the query segment only.
    pub fn project_dimensions(&self, rows: u64) -> [(&'static str, u64); 6] {
        [
            ("M", rows),
            ("D", self.hidden),
            ("Q", self.projection_width()),
            ("GR", 0),
            ("K", 0),
            ("V", 0),
        ]
    }

    /// The statics the state entries fix.
    pub fn state_statics(&self) -> [(&'static str, u64); 5] {
        [
            ("NH", self.heads),
            ("P", self.head_width),
            ("G", self.groups),
            ("N", self.state),
            ("C", self.convolution_width),
        ]
    }

    /// The state entries' dimensions for `slots` slots over `banks` banks
    /// with `tape_rows` tape rows each.
    pub fn state_dimensions(
        &self,
        rows: u64,
        slots: u64,
        banks: u64,
        tape_rows: u64,
    ) -> [(&'static str, u64); 9] {
        let [heads, width, groups, state, taps] = self.state_statics();
        [
            ("M", rows),
            ("B", slots),
            ("S", banks),
            heads,
            width,
            groups,
            state,
            taps,
            ("T", tape_rows),
        ]
    }

    /// `state_space_gate`'s dimensions.
    pub fn gate_dimensions(&self, rows: u64) -> [(&'static str, u64); 5] {
        [
            ("M", rows),
            ("G", self.norm_groups()),
            ("U", self.norm_heads),
            ("P", self.head_width),
            ("J", self.projection_width()),
        ]
    }

    /// `attention_output`'s dimensions: `heads` heads of `head_width`.
    pub fn output_dimensions(&self, rows: u64) -> [(&'static str, u64); 4] {
        [
            ("M", rows),
            ("D", self.hidden),
            ("Q", self.heads),
            ("W", self.head_width),
        ]
    }

    /// Row width of the bank's tape: `u = δ·x | B | decay`.
    pub fn tape_width(&self) -> u64 {
        self.inner() + self.groups * self.state + self.heads
    }
}

/// The resident elements of a state-space layer's weights that are not
/// fixed F32 (the convolution, its bias, decay, time bias, skip and output
/// norm are).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StateSpaceBinding {
    pub shape: StateSpaceShape,
    /// The input RMS weight.
    pub norm: Element,
    pub projection: Element,
    pub output: Element,
    pub activation: Element,
}

/// The state-space forms the entries implement: an output norm group of
/// whole heads.
pub(crate) fn admit(space: &StateSpace) -> Result<(), PlanError> {
    if space.head_width == 0
        || !space.norm_group.is_multiple_of(space.head_width)
        || !space.heads.is_multiple_of(space.norm_group / space.head_width)
    {
        return Err(PlanError::Unsupported(
            "state-space output norm group of partial heads",
        ));
    }
    Ok(())
}

/// The weights the operator binds, in the order its entries consume them.
pub(super) fn weights<'a>(space: &'a StateSpace, push: &mut super::WeightPush<'_, 'a>) {
    use magnitude_family_contracts::WeightKind;
    push(WeightKind::StateSpaceProjection, &space.projection);
    push(WeightKind::RecurrentConvolution, &space.convolution);
    push(WeightKind::RecurrentConvolutionBias, &space.convolution_bias);
    push(WeightKind::RecurrentTimeBias, &space.time_bias);
    push(WeightKind::RecurrentDecay, &space.decay);
    push(WeightKind::StateSpaceSkip, &space.skip);
    push(WeightKind::StateSpaceNorm, &space.norm.weight);
    push(WeightKind::RecurrentOutput, &space.output);
}

/// Every numerical parameter of the operator a sealed graph holds apart from
/// its weights.
pub(super) fn shape_key(space: &StateSpace) -> String {
    format!(
        "state space {:?} {}",
        StateSpaceShape::of(0, space),
        space.norm.epsilon
    )
}

/// The program binding of a state-space sublayer; `lookup` resolves the
/// planned element of a role in its scope.
pub(super) fn binding(
    space: &StateSpace,
    hidden: u64,
    lookup: impl Fn(magnitude_family_contracts::WeightKind) -> Result<Element, PlanError>,
    activation: Element,
) -> Result<StateSpaceBinding, PlanError> {
    use magnitude_family_contracts::WeightKind;
    Ok(StateSpaceBinding {
        shape: StateSpaceShape::of(hidden, space)?,
        norm: lookup(WeightKind::InputNorm)?,
        projection: lookup(WeightKind::StateSpaceProjection)?,
        output: lookup(WeightKind::RecurrentOutput)?,
        activation,
    })
}
