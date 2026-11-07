//! Tuning cases of the state-space block (`operators::state_space`): the
//! normed projection row (`attention_project`, query segment only), the
//! in-place state advance (`state_space_step` below `CHUNKED_ROWS` rows,
//! `state_space_chunk` from there) and the output projection
//! (`attention_output` over `heads` heads). `state_space_gate` declares no
//! parameters.
//!
//! The state entries write the layer's window, state and tape in place.
//! Each argument set owns small arenas of three banks (the zero seed, the
//! bank the slot reads, the bank it publishes to), filled with pseudo-random
//! state; the published bank is the case state restored before each
//! validation run.

use super::cases::{packing_statics, projection_shape, scale_extent};
use super::recurrent::{PUBLISHED_BANK, SlotTables, TUNING_BANKS};
use super::{CaseState, EntryTuning, ModelInputs, PointShape, TuningInputs};
use crate::operators::InputForm;
use crate::operators::gated_delta::graph::chunked;
use crate::{StateSpaceBinding, StateSpaceShape};
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::{attention_output, attention_project, state_space_chunk, state_space_step};
use seismic::{BuildError, Element, Tensor};

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
    scale: Tensor,
    bound: Tensor,
    /// The absent operand of every other segment's scale and bound.
    absent: Tensor,
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
        let mut statics = statics.to_vec();
        statics.extend([
            (
                "QS",
                scale_extent(inputs, &self.scopes, WeightKind::StateSpaceProjection)?,
            ),
            ("GS", 0),
            ("KS", 0),
            ("VS", 0),
        ]);
        statics.extend(packing_statics(
            inputs,
            &self.scopes,
            &[
                ("QB", Some(WeightKind::StateSpaceProjection)),
                ("GB", None),
                ("KB", None),
                ("VB", None),
            ],
        )?);
        statics.extend(InputForm::SINGLE.dimensions(shape.hidden));
        Ok(statics)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
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
                    scale: inputs.unit_scale(
                        inputs.scale_extent(scope, WeightKind::StateSpaceProjection)?,
                    )?,
                    bound: inputs.bound(scope, WeightKind::StateSpaceProjection)?,
                    absent: inputs.unit_scale(0)?,
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
            query_scale: &case.scale,
            gate_scale: &case.absent,
            key_scale: &case.absent,
            value_scale: &case.absent,
            query_bound: &case.bound,
            gate_bound: &case.absent,
            key_bound: &case.absent,
            value_bound: &case.absent,
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
    inject: Tensor,
    gated: Tensor,
    output: Tensor,
    scale: Tensor,
    bound: Tensor,
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
        let mut statics = statics.to_vec();
        statics.push((
            "OS",
            scale_extent(inputs, &self.scopes, WeightKind::RecurrentOutput)?,
        ));
        statics.extend(packing_statics(
            inputs,
            &self.scopes,
            &[("OB", Some(WeightKind::RecurrentOutput))],
        )?);
        Ok(statics)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, BuildError> {
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
                    inject: inputs.activation(Element::f32(), &[point.rows, 0], seed + 3)?,
                    gated: inputs.activation(
                        self.binding.activation,
                        &[point.rows, shape.heads, shape.head_width],
                        seed + 2,
                    )?,
                    output: inputs.weight(scope, WeightKind::RecurrentOutput)?,
                    scale: inputs
                        .unit_scale(inputs.scale_extent(scope, WeightKind::RecurrentOutput)?)?,
                    bound: inputs.bound(scope, WeightKind::RecurrentOutput)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> attention_output::Args<'a> {
        attention_output::Args {
            hidden: &case.hidden,
            inject: &case.inject,
            gated: &case.gated,
            output_weight: &case.output,
            output_scale: &case.scale,
            output_bound: &case.bound,
        }
    }

    generated_entry!(attention_output, this => attention_output::Elements {
        OW: this.binding.output,
        A: this.binding.activation,
    });
}

/// What both state entries tune over: the advance of one request's rows from
/// the bank it reads to the bank it publishes.
#[derive(Clone)]
pub(crate) struct StateSpaceState {
    pub activation: Element,
    pub shape: StateSpaceShape,
    pub scopes: Vec<WeightScope>,
}

/// `state_space_step`, for row classes below `CHUNKED_ROWS`. Its
/// parameters are mappings: every configuration is bit-exact.
#[derive(Clone)]
pub(crate) struct StateSpaceStepTuning(pub StateSpaceState);

/// `state_space_chunk`, for row classes of `CHUNKED_ROWS` and more.
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
    tables: SlotTables,
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
    ) -> Result<StateSpaceStateCase, BuildError> {
        let shape = &self.shape;
        let tables = SlotTables::new(inputs, rows)?;
        let window = inputs.activation(
            self.activation,
            &[
                TUNING_BANKS,
                shape.convolution_width - 1 + tables.tape_rows,
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
            &[TUNING_BANKS, tables.tape_rows, shape.tape_width()],
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
            tables,
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

            fn serves(&self, rows: u64) -> bool {
                ($serves)(rows)
            }

            fn rotation(
                &self,
                inputs: &mut TuningInputs<'_, '_>,
                point: &PointShape,
            ) -> Result<Vec<Self::Case>, BuildError> {
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
                    segments: &case.tables.segments,
                    stop: &case.tables.stop,
                    previous_bank: &case.tables.previous_bank,
                    previous_tape: &case.tables.previous_tape,
                    following_bank: &case.tables.following_bank,
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

state_entry!(StateSpaceStepTuning, state_space_step, |rows| !chunked(
    rows
));
state_entry!(StateSpaceChunkTuning, state_space_chunk, chunked);
