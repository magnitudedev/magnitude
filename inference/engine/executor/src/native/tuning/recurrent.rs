//! Tuning cases of the recurrent block: the normed segmented input
//! projection, the in-place state advance publishing the gated rows
//! (`gated_delta_step` below [`CHUNKED_ROWS`] rows, `gated_delta_chunk` from
//! there), and the plain residual output projection (`attention_output`); on
//! a backend of the convolved step form, `gated_delta_project_convolved` and
//! `gated_delta_step_convolved` below [`CHUNKED_ROWS`] rows.
//!
//! The state entries write the layer's window and delta arenas in place. Each
//! argument set owns small arenas of three banks (the zero seed, the bank the
//! slot reads, the bank it publishes to), filled with pseudo-random state; the
//! published bank is the case state restored before each validation run.

use super::cases::projection_shape;
use super::{
    row_points, served_row_points, CaseState, EntryTuning, ModelInputs, PointShape, TuningInputs,
    TuningLimits,
};
use crate::operators::gated_delta::graph::CHUNKED_ROWS;
use magnitude_family_contracts::{Operator, RecurrentHeadMapping, WeightKind, WeightScope};
use magnitude_kernels::{
    attention_output, gated_delta_chunk, gated_delta_project, gated_delta_project_convolved,
    gated_delta_step, gated_delta_step_convolved,
};
use seismic::{Element, Tensor};

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
    /// qkv weight (`[rows, H]`).
    fn projection_statics(
        &self,
        inputs: &ModelInputs<'_>,
    ) -> Result<Vec<(&'static str, u64)>, String> {
        let (_, hidden) =
            projection_shape(inputs, &self.scopes, WeightKind::RecurrentQueryKeyValue)?;
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

    fn launches(&self) -> usize {
        self.shape.scopes.len()
    }

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

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        self.shape.projection_statics(inputs)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
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

/// The recurrent output projection: `attention_output` over the gated rows
/// [M, NV, W] the state entries publish, plus the residual.
pub(crate) struct RecurrentOutputTuning {
    pub output: Element,
    pub activation: Element,
    pub shape: RecurrentShape,
}

pub(crate) struct RecurrentOutputCase {
    hidden: Tensor,
    gated: Tensor,
    output: Tensor,
}

impl RecurrentOutputTuning {
    fn elements(&self) -> attention_output::Elements {
        attention_output::Elements {
            OW: self.output,
            A: self.activation,
        }
    }
}

impl EntryTuning for RecurrentOutputTuning {
    type Entry = attention_output::Entry;
    type Case = RecurrentOutputCase;

    fn launches(&self) -> usize {
        self.shape.scopes.len()
    }

    fn bindings(&self) -> String {
        format!("OW={},A={}", self.output.name(), self.activation.name())
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (hidden, columns) =
            projection_shape(inputs, &self.shape.scopes, WeightKind::RecurrentOutput)?;
        if columns != self.shape.value_heads * self.shape.width {
            return Err("the recurrent output projection disagrees with the binding".into());
        }
        Ok(vec![
            ("D", hidden),
            ("Q", self.shape.value_heads),
            ("W", self.shape.width),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
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
                let seed = 2 * index as u64;
                Ok(RecurrentOutputCase {
                    hidden: inputs.activation(Element::f32(), &[point.rows, hidden], seed + 1)?,
                    gated: inputs.activation(
                        self.activation,
                        &[point.rows, self.shape.value_heads, self.shape.width],
                        seed + 2,
                    )?,
                    output,
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

    generated_entry!(attention_output, this => this.elements());
}

/// What both state entries tune over: the gated delta advance of one
/// request's rows from the bank it reads to the bank it publishes, and the
/// gating of its outputs.
pub(crate) struct RecurrentState {
    pub recurrent_norm: Element,
    pub activation: Element,
    pub shape: RecurrentShape,
    pub epsilon: f32,
}

/// `gated_delta_step`, for row classes below [`CHUNKED_ROWS`]. Its
/// parameters are mappings: every configuration is bit-exact.
pub(crate) struct RecurrentStepTuning(pub RecurrentState);

/// `gated_delta_chunk`, for row classes of [`CHUNKED_ROWS`] and more.
pub(crate) struct RecurrentChunkTuning(pub RecurrentState);

/// One request slot's tables for a case of `rows` rows: it reads bank 1 with
/// no tape rows and publishes to the case's published bank after every row.
pub(crate) struct SlotTables {
    segments: Tensor,
    stop: Tensor,
    previous_bank: Tensor,
    previous_tape: Tensor,
    following_bank: Tensor,
}

impl SlotTables {
    fn new(inputs: &mut TuningInputs<'_, '_>, rows: u64) -> Result<Self, String> {
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
        Ok(Self {
            segments: inputs.i32s(&[banks + 1, 2], &segments)?,
            stop: inputs.i32s(&[banks], &stop)?,
            previous_bank: inputs.i32s(&[banks], &tables.bank[..slots])?,
            previous_tape: inputs.i32s(&[banks], &vec![0; slots])?,
            following_bank: inputs.i32s(&[banks], &tables.following_bank[..slots])?,
        })
    }
}

/// A tuning arena's window banks of pseudo-random activations.
fn window_state(
    inputs: &mut TuningInputs<'_, '_>,
    shape: &RecurrentShape,
    activation: Element,
    seed: u64,
) -> Result<CaseState, String> {
    let window = inputs.activation(
        activation,
        &[
            TUNING_BANKS,
            shape.convolution_width - 1 + TUNING_TAPE_ROWS,
            shape.channels(),
        ],
        seed,
    )?;
    inputs.slab_state(window, TUNING_BANKS, PUBLISHED_BANK..PUBLISHED_BANK + 1)
}

/// A tuning arena's delta and tape banks of pseudo-random state (seeds
/// `seed + 3` and `seed + 4`).
fn delta_tape_states(
    inputs: &mut TuningInputs<'_, '_>,
    shape: &RecurrentShape,
    seed: u64,
) -> Result<(CaseState, CaseState), String> {
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
    Ok((
        inputs.slab_state(delta, TUNING_BANKS, published.clone())?,
        inputs.slab_state(tape, TUNING_BANKS, published)?,
    ))
}

/// One argument set of either state entry; they share one contract.
pub(crate) struct RecurrentStateCase {
    projection: Tensor,
    convolution: Tensor,
    rate: Tensor,
    time_bias: Tensor,
    recurrent_norm: Tensor,
    tables: SlotTables,
    window: CaseState,
    delta: CaseState,
    tape: CaseState,
    slab_banks: u32,
    norm_epsilon: f32,
    epsilon: f32,
    grouped: bool,
}

/// Tape rows of a tuning case's banks: one, the least a store holds. Cases
/// publish after every row (stop = rows), so the tape is never written and
/// its size does not change the cost.
const TUNING_TAPE_ROWS: u64 = 1;

impl RecurrentState {
    fn bindings(&self) -> String {
        format!(
            "RN={},A={}",
            self.recurrent_norm.name(),
            self.activation.name()
        )
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
        match inputs.operator(&self.shape.scopes)? {
            Operator::GatedDelta(delta) => {
                Ok(matches!(delta.head_mapping, RecurrentHeadMapping::Grouped))
            }
            other => Err(format!("a {} layer has no recurrent mixer", other.name())),
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
        let tables = SlotTables::new(inputs, rows)?;
        let window = window_state(inputs, shape, self.activation, seed + 2)?;
        let (delta, tape) = delta_tape_states(inputs, shape, seed)?;
        Ok(RecurrentStateCase {
            projection: inputs.activation(
                self.activation,
                &[rows, shape.projection_width()],
                seed + 1,
            )?,
            convolution: inputs.weight(scope, WeightKind::RecurrentConvolution)?,
            rate: inputs.weight(scope, WeightKind::RecurrentDecay)?,
            time_bias: inputs.weight(scope, WeightKind::RecurrentTimeBias)?,
            recurrent_norm: inputs.weight(scope, WeightKind::RecurrentNorm)?,
            tables,
            window,
            delta,
            tape,
            slab_banks: TUNING_BANKS as u32,
            norm_epsilon: self.epsilon * shape.width as f32,
            epsilon: self.epsilon,
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

            fn launches(&self) -> usize {
                self.0.shape.scopes.len()
            }

            fn bindings(&self) -> String {
                self.0.bindings()
            }

            fn statics(
                &self,
                _inputs: &ModelInputs<'_>,
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
                    recurrent_norm: &case.recurrent_norm,
                    segments: &case.tables.segments,
                    stop: &case.tables.stop,
                    previous_bank: &case.tables.previous_bank,
                    previous_tape: &case.tables.previous_tape,
                    following_bank: &case.tables.following_bank,
                    window: case.window.tensor_mut(),
                    delta: case.delta.tensor_mut(),
                    tape: case.tape.tensor_mut(),
                    slab_banks: case.slab_banks,
                    norm_epsilon: case.norm_epsilon,
                    epsilon: case.epsilon,
                    grouped: case.grouped,
                }
            }

            fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
                vec![
                    ("window", &case.window),
                    ("delta", &case.delta),
                    ("tape", &case.tape),
                ]
            }

            generated_entry!($module, this => $module::Elements {
                RN: this.0.recurrent_norm,
                A: this.0.activation,
            });
        }
    };
}

state_entry!(RecurrentStepTuning, gated_delta_step, |rows| rows
    < CHUNKED_ROWS);
state_entry!(RecurrentChunkTuning, gated_delta_chunk, |rows| rows
    >= CHUNKED_ROWS);

/// `gated_delta_project_convolved`, the convolved step form's projection,
/// for row classes below [`CHUNKED_ROWS`]: `gated_delta_project`'s
/// parameters, and the windows its launch publishes.
pub(crate) struct RecurrentProjectConvolvedTuning(pub RecurrentProjectTuning);

pub(crate) struct RecurrentProjectConvolvedCase {
    project: RecurrentProjectCase,
    convolution: Tensor,
    tables: SlotTables,
    window: CaseState,
    slab_banks: u32,
}

impl EntryTuning for RecurrentProjectConvolvedTuning {
    type Entry = gated_delta_project_convolved::Entry;
    type Case = RecurrentProjectConvolvedCase;

    fn launches(&self) -> usize {
        self.0.shape.scopes.len()
    }

    fn bindings(&self) -> String {
        self.0.bindings()
    }

    fn statics(&self, inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let mut statics = self.0.shape.projection_statics(inputs)?;
        statics.push(("C", self.0.shape.convolution_width));
        Ok(statics)
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        served_row_points(limits.max_rows, |rows| rows < CHUNKED_ROWS)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let shape = &self.0.shape;
        let projections = self.0.rotation(inputs, point)?;
        TuningInputs::rotation_scopes(&shape.scopes, point)
            .into_iter()
            .zip(projections)
            .enumerate()
            .map(|(index, (scope, project))| {
                Ok(RecurrentProjectConvolvedCase {
                    project,
                    convolution: inputs.weight(scope, WeightKind::RecurrentConvolution)?,
                    tables: SlotTables::new(inputs, point.rows)?,
                    window: window_state(inputs, shape, self.0.activation, 4 * index as u64 + 2)?,
                    slab_banks: TUNING_BANKS as u32,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> gated_delta_project_convolved::Args<'a> {
        gated_delta_project_convolved::Args {
            hidden: &case.project.hidden,
            input_norm: &case.project.norm,
            qkv_weight: &case.project.qkv,
            gate_weight: &case.project.gate,
            alpha_weight: &case.project.alpha,
            beta_weight: &case.project.beta,
            convolution: &case.convolution,
            segments: &case.tables.segments,
            stop: &case.tables.stop,
            previous_bank: &case.tables.previous_bank,
            previous_tape: &case.tables.previous_tape,
            following_bank: &case.tables.following_bank,
            window: case.window.tensor_mut(),
            epsilon: case.project.epsilon,
            slab_banks: case.slab_banks,
        }
    }

    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        vec![("window", &case.window)]
    }

    generated_entry!(gated_delta_project_convolved, this => gated_delta_project_convolved::Elements {
        NW: this.0.norm,
        QW: this.0.qkv,
        GW: this.0.gate,
        AW: this.0.alpha,
        BW: this.0.beta,
        A: this.0.activation,
    });
}

/// `gated_delta_step_convolved`, the convolved step form's state advance,
/// for row classes below [`CHUNKED_ROWS`]. Its parameters are mappings:
/// every configuration is bit-exact.
pub(crate) struct RecurrentStepConvolvedTuning(pub RecurrentState);

pub(crate) struct RecurrentStepConvolvedCase {
    projection: Tensor,
    convolved: Tensor,
    rate: Tensor,
    time_bias: Tensor,
    recurrent_norm: Tensor,
    tables: SlotTables,
    delta: CaseState,
    tape: CaseState,
    slab_banks: u32,
    norm_epsilon: f32,
    epsilon: f32,
    grouped: bool,
}

impl EntryTuning for RecurrentStepConvolvedTuning {
    type Entry = gated_delta_step_convolved::Entry;
    type Case = RecurrentStepConvolvedCase;

    fn launches(&self) -> usize {
        self.0.shape.scopes.len()
    }

    fn bindings(&self) -> String {
        self.0.bindings()
    }

    fn statics(&self, _inputs: &ModelInputs<'_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = &self.0.shape;
        Ok(vec![
            ("NK", shape.key_heads),
            ("NV", shape.value_heads),
            ("W", shape.width),
        ])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        served_row_points(limits.max_rows, |rows| rows < CHUNKED_ROWS)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let state = &self.0;
        let shape = &state.shape;
        let grouped = state.grouped(inputs)?;
        TuningInputs::rotation_scopes(&shape.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 4 * index as u64;
                let (delta, tape) = delta_tape_states(inputs, shape, seed)?;
                Ok(RecurrentStepConvolvedCase {
                    projection: inputs.activation(
                        state.activation,
                        &[point.rows, shape.projection_width()],
                        seed + 1,
                    )?,
                    convolved: inputs.activation(
                        Element::f32(),
                        &[point.rows, shape.channels()],
                        seed + 2,
                    )?,
                    rate: inputs.weight(scope, WeightKind::RecurrentDecay)?,
                    time_bias: inputs.weight(scope, WeightKind::RecurrentTimeBias)?,
                    recurrent_norm: inputs.weight(scope, WeightKind::RecurrentNorm)?,
                    tables: SlotTables::new(inputs, point.rows)?,
                    delta,
                    tape,
                    slab_banks: TUNING_BANKS as u32,
                    norm_epsilon: state.epsilon * shape.width as f32,
                    epsilon: state.epsilon,
                    grouped,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> gated_delta_step_convolved::Args<'a> {
        gated_delta_step_convolved::Args {
            projection: &case.projection,
            convolved: &case.convolved,
            rate: &case.rate,
            time_bias: &case.time_bias,
            recurrent_norm: &case.recurrent_norm,
            segments: &case.tables.segments,
            stop: &case.tables.stop,
            previous_bank: &case.tables.previous_bank,
            previous_tape: &case.tables.previous_tape,
            following_bank: &case.tables.following_bank,
            delta: case.delta.tensor_mut(),
            tape: case.tape.tensor_mut(),
            slab_banks: case.slab_banks,
            norm_epsilon: case.norm_epsilon,
            epsilon: case.epsilon,
            grouped: case.grouped,
        }
    }

    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        vec![("delta", &case.delta), ("tape", &case.tape)]
    }

    generated_entry!(gated_delta_step_convolved, this => gated_delta_step_convolved::Elements {
        RN: this.0.recurrent_norm,
        A: this.0.activation,
    });
}
