//! Tuning cases of the recurrent block: the normed segmented input
//! projection, the in-place state advance (`gated_delta_step` below
//! [`CHUNKED_ROWS`] rows, `gated_delta_chunk` from there), and the gated
//! output projection.
//!
//! The state entries write the layer's window and delta arenas in place. Each
//! argument set owns small arenas of three banks (the zero seed, the bank the
//! slot reads, the bank it publishes to), filled with pseudo-random state; the
//! published bank is the case state restored before each validation run.

use super::cases::projection_shape;
use super::{
    cpu_projection_screening, row_points, served_row_points, CaseState, EntryTuning, PointShape,
    TuningInputs, TuningLimits,
};
use crate::programs::graph::recurrent::CHUNKED_ROWS;
use magnitude_model_contracts::{MixerGeometry, RecurrentHeadMapping, WeightKind, WeightScope};
use magnitude_model_kernels::{
    gated_delta_chunk, gated_delta_output, gated_delta_project, gated_delta_step,
};
use seismic::{Device, Element, ScreeningPoint, Tensor};

/// Banks of a tuning arena: the zero seed, the bank the slot reads and the
/// bank it publishes to.
const TUNING_BANKS: u64 = 3;
/// The bank each tuning slot publishes to; the only bank a state entry writes.
const PUBLISHED_BANK: u64 = 2;

/// The recurrent geometry a specialization fixes, and every layer sharing it.
#[derive(Clone)]
pub(crate) struct RecurrentShape {
    pub key_heads: u64,
    pub value_heads: u64,
    pub width: u64,
    pub convolution_width: u64,
    pub scopes: Vec<WeightScope>,
}

impl RecurrentShape {
    fn channels(&self) -> u64 {
        (2 * self.key_heads + self.value_heads) * self.width
    }

    fn projection_width(&self) -> u64 {
        self.channels() + self.value_heads * self.width + 2 * self.value_heads
    }

    /// `H`, `NK`, `NV`, `W`, with the hidden width read from every layer's
    /// `kind` weight (`[rows, H]` or `[H, columns]` per `hidden_axis`).
    fn projection_statics(
        &self,
        inputs: &TuningInputs<'_, '_>,
        kind: WeightKind,
        hidden_axis: usize,
    ) -> Result<Vec<(&'static str, u64)>, String> {
        let (rows, columns) = projection_shape(inputs, &self.scopes, kind)?;
        let hidden = if hidden_axis == 0 { rows } else { columns };
        Ok(vec![
            ("H", hidden),
            ("NK", self.key_heads),
            ("NV", self.value_heads),
            ("W", self.width),
        ])
    }

    fn state_statics(&self) -> Vec<(&'static str, u64)> {
        vec![
            ("NK", self.key_heads),
            ("NV", self.value_heads),
            ("W", self.width),
            ("C", self.convolution_width),
        ]
    }
}

/// `gated_delta_project`: RMS prologue, segmented qkv | z | alpha | beta.
pub(crate) struct RecurrentProjectTuning {
    pub norm: Element,
    pub qkv: Element,
    pub gate: Element,
    pub alpha: Element,
    pub beta: Element,
    pub activation: Element,
    pub shape: RecurrentShape,
    pub epsilon: f32,
}

pub(crate) struct RecurrentProjectCase {
    hidden: Tensor,
    norm: Tensor,
    qkv: Tensor,
    gate: Tensor,
    alpha: Tensor,
    beta: Tensor,
    epsilon: f32,
}

impl RecurrentProjectTuning {
    fn elements(&self) -> gated_delta_project::Elements {
        gated_delta_project::Elements {
            NW: self.norm,
            QW: self.qkv,
            GW: self.gate,
            AW: self.alpha,
            BW: self.beta,
            A: self.activation,
        }
    }
}

impl EntryTuning for RecurrentProjectTuning {
    type Entry = gated_delta_project::Entry;
    type Case = RecurrentProjectCase;

    fn bindings(&self) -> String {
        format!(
            "NW={},QW={},GW={},AW={},BW={},A={}",
            self.norm.name(),
            self.qkv.name(),
            self.gate.name(),
            self.alpha.name(),
            self.beta.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        self.shape
            .projection_statics(inputs, WeightKind::RecurrentQueryKeyValue, 1)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn screening(&self, device: &Device, points: &[PointShape]) -> Vec<ScreeningPoint> {
        cpu_projection_screening(device, points)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        TuningInputs::rotation_scopes(&self.shape.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let qkv = inputs.weight(scope, WeightKind::RecurrentQueryKeyValue)?;
                let hidden = qkv.extents()[1];
                Ok(RecurrentProjectCase {
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, hidden],
                        index as u64 + 1,
                    )?,
                    norm: inputs.weight(scope, WeightKind::InputNorm)?,
                    gate: inputs.weight(scope, WeightKind::RecurrentGate)?,
                    alpha: inputs.weight(scope, WeightKind::RecurrentAlpha)?,
                    beta: inputs.weight(scope, WeightKind::RecurrentBeta)?,
                    qkv,
                    epsilon: self.epsilon,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> gated_delta_project::Args<'a> {
        gated_delta_project::Args {
            hidden: &case.hidden,
            input_norm: &case.norm,
            qkv_weight: &case.qkv,
            gate_weight: &case.gate,
            alpha_weight: &case.alpha,
            beta_weight: &case.beta,
            epsilon: case.epsilon,
        }
    }

    generated_entry!(gated_delta_project, this => this.elements());
}

/// `gated_delta_output`: gated per-head RMS · SiLU(z) prologue, output
/// projection, residual.
pub(crate) struct RecurrentOutputTuning {
    pub recurrent_norm: Element,
    pub output: Element,
    pub activation: Element,
    pub shape: RecurrentShape,
    pub epsilon: f32,
}

pub(crate) struct RecurrentOutputCase {
    hidden: Tensor,
    mixed: Tensor,
    projection: Tensor,
    recurrent_norm: Tensor,
    output: Tensor,
    epsilon: f32,
}

impl RecurrentOutputTuning {
    fn elements(&self) -> gated_delta_output::Elements {
        gated_delta_output::Elements {
            RN: self.recurrent_norm,
            OW: self.output,
            A: self.activation,
        }
    }
}

impl EntryTuning for RecurrentOutputTuning {
    type Entry = gated_delta_output::Entry;
    type Case = RecurrentOutputCase;

    fn bindings(&self) -> String {
        format!(
            "RN={},OW={},A={}",
            self.recurrent_norm.name(),
            self.output.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        self.shape
            .projection_statics(inputs, WeightKind::RecurrentOutput, 0)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn screening(&self, device: &Device, points: &[PointShape]) -> Vec<ScreeningPoint> {
        cpu_projection_screening(device, points)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        TuningInputs::rotation_scopes(&self.shape.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let output = inputs.weight(scope, WeightKind::RecurrentOutput)?;
                let hidden = output.extents()[0];
                let seed = 3 * index as u64;
                Ok(RecurrentOutputCase {
                    hidden: inputs.activation(Element::f32(), &[point.rows, hidden], seed + 1)?,
                    mixed: inputs.activation(
                        self.activation,
                        &[point.rows, self.shape.value_heads, self.shape.width],
                        seed + 2,
                    )?,
                    projection: inputs.activation(
                        self.activation,
                        &[point.rows, self.shape.projection_width()],
                        seed + 3,
                    )?,
                    recurrent_norm: inputs.weight(scope, WeightKind::RecurrentNorm)?,
                    output,
                    epsilon: self.epsilon,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> gated_delta_output::Args<'a> {
        gated_delta_output::Args {
            hidden: &case.hidden,
            mixed: &case.mixed,
            projection: &case.projection,
            recurrent_norm: &case.recurrent_norm,
            output_weight: &case.output,
            epsilon: case.epsilon,
        }
    }

    generated_entry!(gated_delta_output, this => this.elements());
}

/// What both state entries tune over: the gated delta advance of one
/// request's rows from the bank it reads to the bank it publishes.
pub(crate) struct RecurrentState {
    pub activation: Element,
    pub shape: RecurrentShape,
    pub epsilon: f32,
}

/// `gated_delta_step`, for row classes below [`CHUNKED_ROWS`]. Its
/// parameters are mappings: every configuration is bit-exact.
pub(crate) struct RecurrentStepTuning(pub RecurrentState);

/// `gated_delta_chunk`, for row classes of [`CHUNKED_ROWS`] and more.
pub(crate) struct RecurrentChunkTuning(pub RecurrentState);

/// One argument set of either state entry; they share one contract.
pub(crate) struct RecurrentStateCase {
    projection: Tensor,
    convolution: Tensor,
    rate: Tensor,
    time_bias: Tensor,
    segments: Tensor,
    stop: Tensor,
    previous_bank: Tensor,
    previous_tape: Tensor,
    following_bank: Tensor,
    window: CaseState,
    delta: CaseState,
    tape: CaseState,
    norm_epsilon: f32,
    grouped: bool,
}

/// Tape rows of a tuning case's banks: one, the least a store holds. Cases
/// publish after every row (stop = rows), so the tape is never written and
/// its size does not change the cost.
const TUNING_TAPE_ROWS: u64 = 1;

impl RecurrentState {
    fn bindings(&self) -> String {
        format!("A={}", self.activation.name())
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<RecurrentStateCase>, String> {
        let grouped = self.grouped(inputs)?;
        TuningInputs::rotation_scopes(&self.shape.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| self.case(inputs, scope, point.rows, 4 * index as u64, grouped))
            .collect()
    }

    /// Whether the model's q/k heads map to value heads in groups.
    fn grouped(&self, inputs: &TuningInputs<'_, '_>) -> Result<bool, String> {
        let scope = *self
            .shape
            .scopes
            .first()
            .ok_or("a tuning case needs at least one layer")?;
        let WeightScope::TargetBlock(index) = scope else {
            return Err(format!("recurrent layers are target blocks, not {scope:?}"));
        };
        match inputs
            .definition
            .geometry
            .blocks
            .get(index as usize)
            .map(|block| &block.mixer)
        {
            Some(MixerGeometry::Recurrent(geometry)) => Ok(matches!(
                geometry.head_mapping,
                RecurrentHeadMapping::Grouped
            )),
            _ => Err(format!("block {index} has no recurrent mixer")),
        }
    }

    fn case(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        scope: WeightScope,
        rows: u64,
        seed: u64,
        grouped: bool,
    ) -> Result<RecurrentStateCase, String> {
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
        let delta = inputs.activation(
            Element::f32(),
            &[TUNING_BANKS, shape.value_heads, shape.width, shape.width],
            seed + 3,
        )?;
        let tape = inputs.activation(
            Element::f32(),
            &[
                TUNING_BANKS,
                TUNING_TAPE_ROWS,
                (shape.value_heads + shape.key_heads) * shape.width + shape.value_heads,
            ],
            seed + 4,
        )?;
        let published = PUBLISHED_BANK..PUBLISHED_BANK + 1;
        Ok(RecurrentStateCase {
            projection: inputs.activation(
                self.activation,
                &[rows, shape.projection_width()],
                seed + 1,
            )?,
            convolution: inputs.weight(scope, WeightKind::RecurrentConvolution)?,
            rate: inputs.weight(scope, WeightKind::RecurrentDecay)?,
            time_bias: inputs.weight(scope, WeightKind::RecurrentTimeBias)?,
            segments: inputs.i32s(&[banks + 1, 2], &segments)?,
            stop: inputs.i32s(&[banks], &stop)?,
            previous_bank: inputs.i32s(&[banks], &tables.bank[..slots])?,
            previous_tape: inputs.i32s(&[banks], &vec![0; slots])?,
            following_bank: inputs.i32s(&[banks], &tables.following_bank[..slots])?,
            window: inputs.state(window, published.clone())?,
            delta: inputs.state(delta, published.clone())?,
            tape: inputs.state(tape, published)?,
            norm_epsilon: self.epsilon * shape.width as f32,
            grouped,
        })
    }
}

/// The two state entries differ only in the rows they serve; they share one
/// contract and argument set.
macro_rules! state_entry {
    ($tuning:ident, $module:ident, $serves:expr) => {
        impl EntryTuning for $tuning {
            type Entry = $module::Entry;
            type Case = RecurrentStateCase;

            fn bindings(&self) -> String {
                self.0.bindings()
            }

            fn statics(
                &self,
                _inputs: &TuningInputs<'_, '_>,
            ) -> Result<Vec<(&'static str, u64)>, String> {
                Ok(self.0.shape.state_statics())
            }

            fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
                served_row_points(limits.max_rows, $serves)
            }

            fn rotation(
                &self,
                inputs: &mut TuningInputs<'_, '_>,
                point: &PointShape,
            ) -> Result<Vec<Self::Case>, String> {
                self.0.rotation(inputs, point)
            }

            fn args<'a>(case: &'a mut Self::Case) -> $module::Args<'a> {
                $module::Args {
                    projection: &case.projection,
                    convolution: &case.convolution,
                    rate: &case.rate,
                    time_bias: &case.time_bias,
                    segments: &case.segments,
                    stop: &case.stop,
                    previous_bank: &case.previous_bank,
                    previous_tape: &case.previous_tape,
                    following_bank: &case.following_bank,
                    window: case.window.tensor_mut(),
                    delta: case.delta.tensor_mut(),
                    tape: case.tape.tensor_mut(),
                    norm_epsilon: case.norm_epsilon,
                    grouped: case.grouped,
                }
            }

            fn state(case: &Self::Case) -> Vec<&CaseState> {
                vec![&case.window, &case.delta, &case.tape]
            }

            generated_entry!($module, this => $module::Elements { A: this.0.activation });
        }
    };
}

state_entry!(RecurrentStepTuning, gated_delta_step, |rows| rows
    < CHUNKED_ROWS);
state_entry!(RecurrentChunkTuning, gated_delta_chunk, |rows| rows
    >= CHUNKED_ROWS);
