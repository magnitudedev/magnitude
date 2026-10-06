//! Tuning cases of the state-space block (`operators::state_space`): the
//! normed projection row (`attention_project`, query segment only), the
//! in-place state advance (`state_space_step` below [`CHUNKED_ROWS`] rows,
//! `state_space_chunk` from there) and the output projection
//! (`attention_output` over `heads` heads). `state_space_gate` declares no
//! parameters.
//!
//! The state entries write the layer's window, state and tape in place.
//! Each argument set owns small arenas of three banks (the zero seed, the
//! bank the slot reads, the bank it publishes to), filled with pseudo-random
//! state; the published bank is the case state restored before each
//! validation run.

use super::cases::projection_shape;
use super::{
    row_points, served_row_points, CaseState, EntryTuning, ModelInputs, PointShape, TuningInputs,
    TuningLimits,
};
use crate::operators::gated_delta::graph::CHUNKED_ROWS;
use crate::{StateSpaceBinding, StateSpaceShape};
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::{attention_output, attention_project, state_space_chunk, state_space_step};
use seismic::{Element, Tensor};

/// Banks of a tuning arena: the zero seed, the bank the slot reads and the
/// bank it publishes to.
const TUNING_BANKS: u64 = 3;
/// The bank each tuning slot publishes to; the only bank a state entry writes.
const PUBLISHED_BANK: u64 = 2;
/// Tape rows of a tuning case's banks: one, the least a store holds. Cases
/// publish after every row (stop = rows), so the tape is never written and
/// its size does not change the cost.
const TUNING_TAPE_ROWS: u64 = 1;

/// `attention_project` as the state-space input projection.
#[derive(Clone)]
pub(crate) struct StateSpaceProjectTuning {
    pub binding: StateSpaceBinding,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct StateSpaceProjectCase {
    hidden: Tensor,
    input_norm: Tensor,
    projection: Tensor,
    empty: Tensor,
    epsilon: f32,
}

impl StateSpaceProjectTuning {
    fn elements(&self) -> attention_project::Elements {
        let b = self.binding;
        attention_project::Elements {
            NW: b.norm,
            QW: b.projection,
            GW: b.projection,
            KW: b.projection,
            VW: b.projection,
            A: b.activation,
        }
    }
}

impl EntryTuning for StateSpaceProjectTuning {
    type Entry = attention_project::Entry;
    type Case = StateSpaceProjectCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        let b = self.binding;
        format!(
            "NW={},QW={},GW={},KW={},VW={},A={}",
            b.norm.name(),
            b.projection.name(),
            b.projection.name(),
            b.projection.name(),
            b.projection.name(),
            b.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        if projection_shape(inputs, &self.scopes, WeightKind::StateSpaceProjection)?
            != (shape.projection_width(), shape.hidden)
        {
            return Err("the state-space projection disagrees with the binding".into());
        }
        let [_, statics @ ..] = shape.project_dimensions(0);
        Ok(statics.to_vec())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let hidden = self.binding.shape.hidden;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let projection = inputs.weight(scope, WeightKind::StateSpaceProjection)?;
                Ok(StateSpaceProjectCase {
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, hidden],
                        index as u64 + 1,
                    )?,
                    input_norm: inputs.weight(scope, WeightKind::InputNorm)?,
                    empty: projection
                        .slice_leading(0, 0)
                        .map_err(|error| error.to_string())?,
                    projection,
                    epsilon: self.epsilon,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> attention_project::Args<'a> {
        attention_project::Args {
            hidden: &case.hidden,
            input_norm: &case.input_norm,
            query_weight: &case.projection,
            gate_weight: &case.empty,
            key_weight: &case.empty,
            value_weight: &case.empty,
            epsilon: case.epsilon,
            project_mode: 0,
        }
    }

    generated_entry!(attention_project, this => this.elements());
}

/// `attention_output` as the state-space output projection plus residual.
#[derive(Clone)]
pub(crate) struct StateSpaceOutputTuning {
    pub binding: StateSpaceBinding,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct StateSpaceOutputCase {
    hidden: Tensor,
    gated: Tensor,
    output: Tensor,
}

impl EntryTuning for StateSpaceOutputTuning {
    type Entry = attention_output::Entry;
    type Case = StateSpaceOutputCase;

    fn launches(&self) -> usize {
        self.scopes.len()
    }

    fn bindings(&self) -> String {
        format!(
            "OW={},A={}",
            self.binding.output.name(),
            self.binding.activation.name()
        )
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.binding.shape;
        if projection_shape(inputs, &self.scopes, WeightKind::RecurrentOutput)?
            != (shape.hidden, shape.inner())
        {
            return Err("the state-space output projection disagrees with the binding".into());
        }
        let [_, statics @ ..] = shape.output_dimensions(0);
        Ok(statics.to_vec())
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let shape = self.binding.shape;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 2 * index as u64;
                Ok(StateSpaceOutputCase {
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, shape.hidden],
                        seed + 1,
                    )?,
                    gated: inputs.activation(
                        self.binding.activation,
                        &[point.rows, shape.heads, shape.head_width],
                        seed + 2,
                    )?,
                    output: inputs.weight(scope, WeightKind::RecurrentOutput)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> attention_output::Args<'a> {
        attention_output::Args {
            hidden: &case.hidden,
            gated: &case.gated,
            output_weight: &case.output,
        }
    }

    generated_entry!(attention_output, this => attention_output::Elements {
        OW: this.binding.output,
        A: this.binding.activation,
    }, rounded to this.binding.activation);
}

/// What both state entries tune over: the advance of one request's rows from
/// the bank it reads to the bank it publishes.
#[derive(Clone)]
pub(crate) struct StateSpaceState {
    pub activation: Element,
    pub shape: StateSpaceShape,
    pub scopes: Vec<WeightScope>,
}

/// `state_space_step`, for row classes below [`CHUNKED_ROWS`]. Its
/// parameters are mappings: every configuration is bit-exact.
#[derive(Clone)]
pub(crate) struct StateSpaceStepTuning(pub StateSpaceState);

/// `state_space_chunk`, for row classes of [`CHUNKED_ROWS`] and more.
#[derive(Clone)]
pub(crate) struct StateSpaceChunkTuning(pub StateSpaceState);

/// One argument set of either state entry; they share one contract.
pub(crate) struct StateSpaceStateCase {
    projection: Tensor,
    convolution: Tensor,
    convolution_bias: Tensor,
    rate: Tensor,
    time_bias: Tensor,
    skip: Tensor,
    segments: Tensor,
    stop: Tensor,
    previous_bank: Tensor,
    previous_tape: Tensor,
    following_bank: Tensor,
    window: CaseState,
    state: CaseState,
    tape: CaseState,
    slab_banks: u32,
}

impl StateSpaceState {
    fn case(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        scope: WeightScope,
        rows: u64,
        seed: u64,
    ) -> Result<StateSpaceStateCase, String> {
        let shape = &self.shape;
        let tables = inputs.batch(rows, 0, 1, 0)?;
        let slots = tables.actual_slots;
        let segments = tables.segments[..=slots]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let stop = tables.segments[..slots]
            .iter()
            .map(|[start, end]| end - start)
            .collect::<Vec<_>>();
        let banks = slots as u64;
        let window = inputs.activation(
            self.activation,
            &[
                TUNING_BANKS,
                shape.convolution_width - 1 + TUNING_TAPE_ROWS,
                shape.channels(),
            ],
            seed + 2,
        )?;
        let state = inputs.activation(
            Element::f32(),
            &[TUNING_BANKS, shape.heads, shape.head_width, shape.state],
            seed + 3,
        )?;
        let tape = inputs.activation(
            Element::f32(),
            &[TUNING_BANKS, TUNING_TAPE_ROWS, shape.tape_width()],
            seed + 4,
        )?;
        let published = PUBLISHED_BANK..PUBLISHED_BANK + 1;
        Ok(StateSpaceStateCase {
            projection: inputs.activation(
                self.activation,
                &[rows, shape.projection_width()],
                seed + 1,
            )?,
            convolution: inputs.weight(scope, WeightKind::RecurrentConvolution)?,
            convolution_bias: inputs.weight(scope, WeightKind::RecurrentConvolutionBias)?,
            rate: inputs.weight(scope, WeightKind::RecurrentDecay)?,
            time_bias: inputs.weight(scope, WeightKind::RecurrentTimeBias)?,
            skip: inputs.weight(scope, WeightKind::StateSpaceSkip)?,
            segments: inputs.i32s(&[banks + 1, 2], &segments)?,
            stop: inputs.i32s(&[banks], &stop)?,
            previous_bank: inputs.i32s(&[banks], &tables.bank[..slots])?,
            previous_tape: inputs.i32s(&[banks], &vec![0; slots])?,
            following_bank: inputs.i32s(&[banks], &tables.following_bank[..slots])?,
            window: inputs.slab_state(window, TUNING_BANKS, published.clone())?,
            state: inputs.slab_state(state, TUNING_BANKS, published.clone())?,
            tape: inputs.slab_state(tape, TUNING_BANKS, published)?,
            slab_banks: TUNING_BANKS as u32,
        })
    }
}

/// The two state entries differ only in the rows they serve; they share one
/// contract and argument set.
macro_rules! state_entry {
    ($tuning:ident, $module:ident, $serves:expr) => {
        impl EntryTuning for $tuning {
            type Entry = $module::Entry;
            type Case = StateSpaceStateCase;

            fn launches(&self) -> usize {
                self.0.scopes.len()
            }

            fn bindings(&self) -> String {
                format!("A={}", self.0.activation.name())
            }

            fn statics(
                &self,
                _inputs: &ModelInputs<'_>,
            ) -> Result<Vec<(&'static str, u64)>, String> {
                Ok(self.0.shape.state_statics().to_vec())
            }

            fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
                served_row_points(limits.max_rows, $serves)
            }

            fn rotation(
                &self,
                inputs: &mut TuningInputs<'_, '_>,
                point: &PointShape,
            ) -> Result<Vec<Self::Case>, String> {
                TuningInputs::rotation_scopes(&self.0.scopes, point)
                    .into_iter()
                    .enumerate()
                    .map(|(index, scope)| self.0.case(inputs, scope, point.rows, 4 * index as u64))
                    .collect()
            }

            fn args<'a>(case: &'a mut Self::Case) -> $module::Args<'a> {
                $module::Args {
                    projection: &case.projection,
                    convolution: &case.convolution,
                    convolution_bias: &case.convolution_bias,
                    rate: &case.rate,
                    time_bias: &case.time_bias,
                    skip: &case.skip,
                    segments: &case.segments,
                    stop: &case.stop,
                    previous_bank: &case.previous_bank,
                    previous_tape: &case.previous_tape,
                    following_bank: &case.following_bank,
                    window: case.window.tensor_mut(),
                    state: case.state.tensor_mut(),
                    tape: case.tape.tensor_mut(),
                    slab_banks: case.slab_banks,
                }
            }

            fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
                vec![
                    ("window", &case.window),
                    ("state", &case.state),
                    ("tape", &case.tape),
                ]
            }

            generated_entry!($module, this => $module::Elements { A: this.0.activation });
        }
    };
}

state_entry!(StateSpaceStepTuning, state_space_step, |rows| rows
    < CHUNKED_ROWS);
state_entry!(StateSpaceChunkTuning, state_space_chunk, |rows| rows
    >= CHUNKED_ROWS);
