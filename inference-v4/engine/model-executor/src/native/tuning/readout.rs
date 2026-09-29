//! Tuning cases of the readout: the final-norm vocabulary projections
//! (`readout_head_rows`, `readout_selected_rows`, the MTP head's
//! `head_logits_rows`) and token selection (`shape_rows`, `sample_rows`).
//!
//! Readout points are the projected-row counts the engine serves (bounded by
//! its projected-row limit, not its batch rows). The vocabulary projection
//! weight is one tensor far larger than any device cache, so each point has
//! a single argument set.

use super::cases::projection_shape;
use super::{
    row_points, served_row_points, CaseState, EntryTuning, PointShape, TuningInputs, TuningLimits,
};
use crate::native::draft_vocabulary;
use magnitude_model_batching::{HISTORY_WIDTH, SHAPING_WIDTH};
use magnitude_model_contracts::{WeightKind, WeightScope};
use magnitude_model_kernels::{
    draft_rows, head_logits_rows, readout_head_rows, readout_selected_rows, sample_rows, shape_rows,
};
use seismic::{Element, Tensor};

/// Candidate tokens of a `readout_selected_rows` tuning point.
const SELECTED_TOKENS: u64 = 256;

/// The MTP input's fused embedding, two RMS norms and combine projection.
pub(crate) struct DraftRowsTuning {
    pub embedding: Element,
    pub embedding_norm: Element,
    pub hidden_norm: Element,
    pub combine: Element,
    pub activation: Element,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct DraftRowsCase {
    tokens: Tensor,
    table: Tensor,
    conditioning: Tensor,
    embedding_norm: Tensor,
    hidden_norm: Tensor,
    combine: Tensor,
    epsilon: f32,
}

impl EntryTuning for DraftRowsTuning {
    type Entry = draft_rows::Entry;
    type Case = DraftRowsCase;

    fn bindings(&self) -> String {
        format!(
            "EW={},EN={},HN={},CW={},A={}",
            self.embedding.name(),
            self.embedding_norm.name(),
            self.hidden_norm.name(),
            self.combine.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (hidden, joined) = projection_shape(inputs, &self.scopes, WeightKind::HeadCombine)?;
        if joined != 2 * hidden {
            return Err("the draft combine weight does not have two hidden inputs".into());
        }
        Ok(vec![("D", hidden)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let table = inputs.weight(WeightScope::Target, WeightKind::Embedding)?;
        let [vocabulary, hidden] = table.extents()[..] else {
            return Err("the draft embedding table is not a matrix".into());
        };
        if vocabulary == 0 {
            return Err("the draft embedding table is empty".into());
        }
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let tokens = (0..point.rows)
                    .map(|row| {
                        i32::try_from((row + index as u64) % vocabulary)
                            .map(|token| [token, 0])
                            .map_err(|_| "the draft vocabulary exceeds i32")
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                Ok(DraftRowsCase {
                    tokens: inputs.i32s(&[point.rows, 2], &tokens)?,
                    table: table.clone(),
                    conditioning: inputs.activation(
                        self.activation,
                        &[point.rows, hidden],
                        index as u64 + 1,
                    )?,
                    embedding_norm: inputs.weight(scope, WeightKind::HeadEmbeddingNorm)?,
                    hidden_norm: inputs.weight(scope, WeightKind::HeadHiddenNorm)?,
                    combine: inputs.weight(scope, WeightKind::HeadCombine)?,
                    epsilon: self.epsilon,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> draft_rows::Args<'a> {
        draft_rows::Args {
            tokens: &case.tokens,
            table: &case.table,
            conditioning: &case.conditioning,
            embedding_norm: &case.embedding_norm,
            hidden_norm: &case.hidden_norm,
            combine: &case.combine,
            epsilon: case.epsilon,
        }
    }

    generated_entry!(draft_rows, this => draft_rows::Elements {
        EW: this.embedding,
        A: this.activation,
        EN: this.embedding_norm,
        HN: this.hidden_norm,
        CW: this.combine,
    });
}

/// The projected-row points of the readout.
fn projected_points(limits: TuningLimits) -> Vec<PointShape> {
    served_row_points(limits.max_projected_rows, |_| true)
}

/// `[V, D]` of the output projection.
fn vocabulary_shape(inputs: &TuningInputs<'_, '_>) -> Result<(u64, u64), String> {
    projection_shape(inputs, &[WeightScope::Target], WeightKind::Output)
}

/// `readout_head_rows`: final RMS norm and vocabulary projection of the rows
/// `out_rows` selects.
pub(crate) struct HeadRowsTuning {
    pub norm: Element,
    pub weight: Element,
    pub activation: Element,
    pub epsilon: f32,
}

pub(crate) struct HeadRowsCase {
    hidden: Tensor,
    norm: Tensor,
    weight: Tensor,
    out_rows: Tensor,
    epsilon: f32,
}

impl HeadRowsTuning {
    fn elements(&self) -> readout_head_rows::Elements {
        readout_head_rows::Elements {
            NW: self.norm,
            OW: self.weight,
            A: self.activation,
        }
    }
}

impl EntryTuning for HeadRowsTuning {
    type Entry = readout_head_rows::Entry;
    type Case = HeadRowsCase;

    fn bindings(&self) -> String {
        format!(
            "NW={},OW={},A={}",
            self.norm.name(),
            self.weight.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (vocabulary, hidden) = vocabulary_shape(inputs)?;
        Ok(vec![("V", vocabulary), ("D", hidden)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        projected_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let weight = inputs.weight(WeightScope::Target, WeightKind::Output)?;
        let hidden = weight.extents()[1];
        Ok(vec![HeadRowsCase {
            hidden: inputs.activation(Element::f32(), &[point.rows, hidden], 1)?,
            norm: inputs.weight(WeightScope::Target, WeightKind::OutputNorm)?,
            out_rows: inputs.every_row(point.rows)?,
            weight,
            epsilon: self.epsilon,
        }])
    }

    fn args<'a>(case: &'a mut Self::Case) -> readout_head_rows::Args<'a> {
        readout_head_rows::Args {
            hidden: &case.hidden,
            norm: &case.norm,
            weight: &case.weight,
            out_rows: &case.out_rows,
            epsilon: case.epsilon,
        }
    }

    generated_entry!(readout_head_rows, this => this.elements());
}

/// `readout_selected_rows`: final RMS norm and the logits of a candidate token
/// set.
pub(crate) struct SelectedRowsTuning {
    pub norm: Element,
    pub weight: Element,
    pub activation: Element,
    pub epsilon: f32,
}

pub(crate) struct SelectedRowsCase {
    head: HeadRowsCase,
    selected: Tensor,
}

impl SelectedRowsTuning {
    fn elements(&self) -> readout_selected_rows::Elements {
        readout_selected_rows::Elements {
            NW: self.norm,
            OW: self.weight,
            A: self.activation,
        }
    }
}

impl EntryTuning for SelectedRowsTuning {
    type Entry = readout_selected_rows::Entry;
    type Case = SelectedRowsCase;

    fn bindings(&self) -> String {
        format!(
            "NW={},OW={},A={}",
            self.norm.name(),
            self.weight.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (vocabulary, hidden) = vocabulary_shape(inputs)?;
        Ok(vec![("V", vocabulary), ("D", hidden)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        projected_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let head = HeadRowsTuning {
            norm: self.norm,
            weight: self.weight,
            activation: self.activation,
            epsilon: self.epsilon,
        }
        .rotation(inputs, point)?;
        let (vocabulary, _) = vocabulary_shape(inputs)?;
        // Candidates spread over the whole vocabulary, as a constrained or
        // speculative candidate set is.
        let stride = vocabulary / SELECTED_TOKENS;
        let tokens = (0..SELECTED_TOKENS)
            .map(|index| i32::try_from(index * stride).map_err(|_| "vocabulary exceeds i32"))
            .collect::<Result<Vec<_>, _>>()?;
        head.into_iter()
            .map(|head| {
                Ok(SelectedRowsCase {
                    head,
                    selected: inputs.i32s(&[SELECTED_TOKENS], &tokens)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> readout_selected_rows::Args<'a> {
        readout_selected_rows::Args {
            hidden: &case.head.hidden,
            norm: &case.head.norm,
            weight: &case.head.weight,
            out_rows: &case.head.out_rows,
            selected: &case.selected,
            epsilon: case.head.epsilon,
        }
    }

    generated_entry!(readout_selected_rows, this => this.elements());
}

/// `head_logits_rows`: the MTP head's vocabulary projection of its normed
/// feature rows.
pub(crate) struct HeadLogitsTuning {
    pub weight: Element,
    pub activation: Element,
}

pub(crate) struct HeadLogitsCase {
    features: Tensor,
    weight: Tensor,
}

impl HeadLogitsTuning {
    fn elements(&self) -> head_logits_rows::Elements {
        head_logits_rows::Elements {
            OW: self.weight,
            A: self.activation,
        }
    }
}

impl EntryTuning for HeadLogitsTuning {
    type Entry = head_logits_rows::Entry;
    type Case = HeadLogitsCase;

    fn bindings(&self) -> String {
        format!("OW={},A={}", self.weight.name(), self.activation.name())
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (vocabulary, hidden) = vocabulary_shape(inputs)?;
        Ok(vec![("V", draft_vocabulary(vocabulary)), ("D", hidden)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        projected_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let output = inputs.weight(WeightScope::Target, WeightKind::Output)?;
        let [vocabulary, hidden] = output.extents()[..] else {
            return Err("the output weight is not a matrix".into());
        };
        let weight = output
            .slice_leading(0, draft_vocabulary(vocabulary))
            .map_err(|error| error.to_string())?;
        Ok(vec![HeadLogitsCase {
            features: inputs.activation(self.activation, &[point.rows, hidden], 1)?,
            weight,
        }])
    }

    fn args<'a>(case: &'a mut Self::Case) -> head_logits_rows::Args<'a> {
        head_logits_rows::Args {
            features: &case.features,
            weight: &case.weight,
        }
    }

    generated_entry!(head_logits_rows, this => this.elements());
}

/// Pseudo-random logits of `rows` rows over `vocabulary` tokens.
fn logits(
    inputs: &TuningInputs<'_, '_>,
    rows: u64,
    vocabulary: u64,
    seed: u64,
) -> Result<Tensor, String> {
    inputs.activation(Element::f32(), &[rows, vocabulary], seed)
}

/// `shape_rows`: penalties, temperature, top-k, min-p and top-p of each
/// selected row over `vocabulary` tokens (the target's, or the draft
/// head's). Its parameters partition the vocabulary without changing any
/// result.
pub(crate) struct ShapeRowsTuning {
    pub vocabulary: u64,
}

pub(crate) struct ShapeRowsCase {
    logits: Tensor,
    params: Tensor,
    history: Tensor,
    out: CaseState,
}

impl EntryTuning for ShapeRowsTuning {
    type Entry = shape_rows::Entry;
    type Case = ShapeRowsCase;

    fn bindings(&self) -> String {
        "fixed".into()
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(vec![("V", self.vocabulary), ("Hn", HISTORY_WIDTH as u64)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        projected_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let rows = point.rows;
        let vocabulary = self.vocabulary;
        // A typical sampling configuration with every stage active:
        // temperature, top-k, top-p, min-p, repetition, presence, frequency.
        let shaping: [f32; SHAPING_WIDTH] = [0.7, 20.0, 0.95, 0.05, 1.1, 0.2, 0.1, 0.0];
        let params = (0..rows).flat_map(|_| shaping).collect::<Vec<_>>();
        let history = (0..rows * HISTORY_WIDTH as u64)
            .map(|index| {
                i32::try_from(index * 7919 % vocabulary).map_err(|_| "vocabulary exceeds i32")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let out = inputs.scratch(Element::f32(), &[rows, vocabulary])?;
        Ok(vec![ShapeRowsCase {
            logits: logits(inputs, rows, vocabulary, 1)?,
            params: inputs.f32s(&[rows, SHAPING_WIDTH as u64], &params)?,
            history: inputs.i32s(&[rows, HISTORY_WIDTH as u64], &history)?,
            out: inputs.state(out, 0..rows)?,
        }])
    }

    fn args<'a>(case: &'a mut Self::Case) -> shape_rows::Args<'a> {
        shape_rows::Args {
            logits: &case.logits,
            params: &case.params,
            history: &case.history,
            out: case.out.tensor_mut(),
        }
    }

    fn state(case: &Self::Case) -> Vec<&CaseState> {
        vec![&case.out]
    }

    generated_entry!(shape_rows);
}

/// `sample_rows`: one draw per selected row over its shaped logits of
/// `vocabulary` tokens. Its parameters partition the vocabulary without
/// changing any draw.
pub(crate) struct SampleRowsTuning {
    pub vocabulary: u64,
}

pub(crate) struct SampleRowsCase {
    logits: Tensor,
    mask: Tensor,
    constrained: Tensor,
    draws: Tensor,
    result: CaseState,
}

impl EntryTuning for SampleRowsTuning {
    type Entry = sample_rows::Entry;
    type Case = SampleRowsCase;

    fn bindings(&self) -> String {
        "fixed".into()
    }

    fn statics(&self, _: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        Ok(vec![("V", self.vocabulary)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        projected_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let rows = point.rows;
        let words = self.vocabulary.div_ceil(32);
        let draws = (0..rows * 6)
            .map(|index| (index as u32).wrapping_mul(0x9e37_79b9))
            .collect::<Vec<_>>();
        let result = inputs.scratch(Element::i32(), &[rows, 2])?;
        Ok(vec![SampleRowsCase {
            logits: logits(inputs, rows, self.vocabulary, 2)?,
            mask: inputs.u32s(&[rows, words], &vec![u32::MAX; (rows * words) as usize])?,
            // Every other row constrained: both the masked and the free
            // path are measured.
            constrained: inputs.i32s(
                &[rows],
                &(0..rows).map(|row| (row % 2) as i32).collect::<Vec<_>>(),
            )?,
            draws: inputs.u32s(&[rows, 6], &draws)?,
            result: inputs.state(result, 0..rows)?,
        }])
    }

    fn args<'a>(case: &'a mut Self::Case) -> sample_rows::Args<'a> {
        sample_rows::Args {
            logits: &case.logits,
            mask: &case.mask,
            constrained: &case.constrained,
            draws: &case.draws,
            result: case.result.tensor_mut(),
        }
    }

    fn state(case: &Self::Case) -> Vec<&CaseState> {
        vec![&case.result]
    }

    generated_entry!(sample_rows);
}
