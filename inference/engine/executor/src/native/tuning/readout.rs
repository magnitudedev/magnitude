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
use magnitude_batching::{HISTORY_WIDTH, SHAPING_WIDTH};
use magnitude_family_contracts::{ProgressivePlane, WeightKind, WeightScope};
use magnitude_kernels::{
    draft_rows, head_logits_rows, readout_exact_rows, readout_head_rows, readout_planes_rows,
    readout_refine_rows, readout_selected_rows, readout_top_rows, sample_rows, shape_rows,
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

    fn launches(&self) -> usize {
        self.scopes.len()
    }

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
    /// The output weight's leading rows it projects onto (a draft readout's
    /// `draft_vocabulary`); `None` projects onto every row.
    pub rows: Option<u64>,
}

pub(crate) struct HeadRowsCase {
    hidden: Tensor,
    norm: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
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

    fn launches(&self) -> usize {
        1
    }

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
        Ok(vec![
            (
                "V",
                self.rows.map_or(vocabulary, |rows| rows.min(vocabulary)),
            ),
            ("D", hidden),
            (
                "WS",
                inputs.scale_extent(WeightScope::Target, WeightKind::Output)?,
            ),
        ])
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
        let [vocabulary, hidden] = weight.extents()[..] else {
            return Err("the output weight is not a matrix".into());
        };
        let weight = weight
            .slice_leading(0, self.rows.map_or(vocabulary, |rows| rows.min(vocabulary)))
            .map_err(|error| error.to_string())?;
        Ok(vec![HeadRowsCase {
            hidden: inputs.activation(Element::f32(), &[point.rows, hidden], 1)?,
            norm: inputs.weight(WeightScope::Target, WeightKind::OutputNorm)?,
            out_rows: inputs.every_row(point.rows)?,
            weight,
            weight_scale: inputs
                .unit_scale(inputs.scale_extent(WeightScope::Target, WeightKind::Output)?)?,
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
            softcap: 0.0,
            weight_scale: &case.weight_scale,
        }
    }

    generated_entry!(readout_head_rows, this => this.elements());
}

/// The certified levels serve selection rows up to the backend's certified
/// row bound.
fn certified_points(limits: TuningLimits, rows: u64) -> Vec<PointShape> {
    served_row_points(limits.max_projected_rows.min(rows), |_| true)
}

/// The bindings every progressive head entry shares.
#[derive(Clone, Copy)]
pub(crate) struct ProgressiveTuning {
    pub norm: Element,
    pub activation: Element,
    pub epsilon: f32,
    /// The most selected rows the certified levels serve
    /// (`readout::certified_rows`).
    pub certified_rows: u64,
    /// The planes' leading rows it projects onto (a draft head's
    /// `draft_vocabulary`); `None` projects onto every row.
    pub rows: Option<u64>,
}

impl ProgressiveTuning {
    /// `[V, D]` it projects onto.
    fn shape(&self, inputs: &TuningInputs<'_, '_>) -> Result<(u64, u64), String> {
        let (vocabulary, hidden) = planes_shape(inputs)?;
        Ok((self.rows.map_or(vocabulary, |rows| rows.min(vocabulary)), hidden))
    }

    /// The planes' rows it projects onto, in `ProgressivePlane::ALL` order.
    fn planes(&self, inputs: &mut TuningInputs<'_, '_>) -> Result<[Tensor; 5], String> {
        let (vocabulary, _) = self.shape(inputs)?;
        let [top, bit3, rest, scales, radius] = planes(inputs)?;
        let rows = |plane: Tensor| {
            plane
                .slice_leading(0, vocabulary)
                .map_err(|error| error.to_string())
        };
        Ok([rows(top)?, rows(bit3)?, rows(rest)?, rows(scales)?, rows(radius)?])
    }
}

/// `[V, D]` of a progressive head, from its top plane (`[V, D / 8]`).
fn planes_shape(inputs: &TuningInputs<'_, '_>) -> Result<(u64, u64), String> {
    let [vocabulary, words] = inputs.weight_shape(
        WeightScope::Target,
        WeightKind::OutputPlane(ProgressivePlane::Top),
    )?[..] else {
        return Err("the top plane is not a matrix".into());
    };
    Ok((vocabulary, words * 8))
}

fn plane(inputs: &mut TuningInputs<'_, '_>, plane: ProgressivePlane) -> Result<Tensor, String> {
    inputs.weight(WeightScope::Target, WeightKind::OutputPlane(plane))
}

/// The planes of a progressive head, in `ProgressivePlane::ALL` order.
fn planes(inputs: &mut TuningInputs<'_, '_>) -> Result<[Tensor; 5], String> {
    Ok([
        plane(inputs, ProgressivePlane::Top)?,
        plane(inputs, ProgressivePlane::Bit3)?,
        plane(inputs, ProgressivePlane::Rest)?,
        plane(inputs, ProgressivePlane::Scales)?,
        plane(inputs, ProgressivePlane::Radius)?,
    ])
}

/// A certified level's selection inputs over `rows` rows: sampling rows at
/// temperature 1 and unconstrained, so the levels' scores carry the
/// sampler's noise.
struct Selecting {
    draws: Tensor,
    temperature: Tensor,
    mask: Tensor,
    constrained: Tensor,
}

impl Selecting {
    fn new(inputs: &TuningInputs<'_, '_>, rows: u64, vocabulary: u64) -> Result<Self, String> {
        let draws = (0..rows * 6)
            .map(|index| {
                if index % 6 == 0 {
                    1
                } else {
                    (index as u32).wrapping_mul(0x9e37_79b9)
                }
            })
            .collect::<Vec<_>>();
        let words = vocabulary.div_ceil(32);
        Ok(Self {
            draws: inputs.u32s(&[rows, 6], &draws)?,
            temperature: inputs.f32s(&[rows], &vec![1.0; rows as usize])?,
            mask: inputs.u32s(&[rows, words], &vec![u32::MAX; (rows * words) as usize])?,
            constrained: inputs.i32s(&[rows], &vec![0; rows as usize])?,
        })
    }
}

/// `readout_top_rows`: the final norm and the 4-bit view of every
/// vocabulary row, with each row's threshold.
pub(crate) struct TopRowsTuning(pub ProgressiveTuning);

pub(crate) struct TopRowsCase {
    hidden: Tensor,
    norm: Tensor,
    planes: [Tensor; 5],
    out_rows: Tensor,
    selecting: Selecting,
    epsilon: f32,
}

impl EntryTuning for TopRowsTuning {
    type Entry = readout_top_rows::Entry;
    type Case = TopRowsCase;

    fn launches(&self) -> usize {
        1
    }

    fn bindings(&self) -> String {
        format!("NW={},A={}", self.0.norm.name(), self.0.activation.name())
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (vocabulary, hidden) = self.0.shape(inputs)?;
        Ok(vec![("V", vocabulary), ("D", hidden)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        certified_points(limits, self.0.certified_rows)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let (vocabulary, hidden) = self.0.shape(inputs)?;
        Ok(vec![TopRowsCase {
            hidden: inputs.activation(Element::f32(), &[point.rows, hidden], 1)?,
            norm: inputs.weight(WeightScope::Target, WeightKind::OutputNorm)?,
            planes: self.0.planes(inputs)?,
            out_rows: inputs.every_row(point.rows)?,
            selecting: Selecting::new(inputs, point.rows, vocabulary)?,
            epsilon: self.0.epsilon,
        }])
    }

    fn args<'a>(case: &'a mut Self::Case) -> readout_top_rows::Args<'a> {
        let [top, bit3, rest, scales, radius] = &case.planes;
        readout_top_rows::Args {
            hidden: &case.hidden,
            norm: &case.norm,
            top,
            bit3,
            rest,
            scales,
            radius,
            out_rows: &case.out_rows,
            draws: &case.selecting.draws,
            temperature: &case.selecting.temperature,
            mask: &case.selecting.mask,
            constrained: &case.selecting.constrained,
            epsilon: case.epsilon,
        }
    }

    generated_entry!(readout_top_rows, this => readout_top_rows::Elements {
        NW: this.0.norm,
        A: this.0.activation,
    });
}

/// A later certified level's previous logits: finite at one vocabulary row
/// in `step` (the survivors a real selection keeps after the 4-bit view are
/// about one in twelve), -inf elsewhere, under a threshold every finite row
/// reaches.
fn survivors(
    inputs: &TuningInputs<'_, '_>,
    rows: u64,
    vocabulary: u64,
    step: u64,
) -> Result<(Tensor, Tensor), String> {
    let previous = (0..rows * vocabulary)
        .map(|index| {
            if (index % vocabulary) % step == 0 {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect::<Vec<_>>();
    Ok((
        inputs.f32s(&[rows, vocabulary], &previous)?,
        inputs.f32s(&[rows], &vec![f32::NEG_INFINITY; rows as usize])?,
    ))
}

/// A later level's operands.
pub(crate) struct LevelCase {
    features: Tensor,
    planes: [Tensor; 5],
    previous: Tensor,
    floor: Tensor,
    length: Tensor,
    selecting: Selecting,
}

impl LevelCase {
    fn new(
        inputs: &mut TuningInputs<'_, '_>,
        tuning: &ProgressiveTuning,
        rows: u64,
        step: u64,
    ) -> Result<Self, String> {
        let (vocabulary, hidden) = tuning.shape(inputs)?;
        let (previous, floor) = survivors(inputs, rows, vocabulary, step)?;
        Ok(Self {
            features: inputs.activation(tuning.activation, &[rows, hidden], 2)?,
            planes: tuning.planes(inputs)?,
            previous,
            floor,
            length: inputs.f32s(&[rows], &vec![1.0; rows as usize])?,
            selecting: Selecting::new(inputs, rows, vocabulary)?,
        })
    }
}

/// `readout_refine_rows`: the 5-bit view of the rows level one keeps.
pub(crate) struct RefineRowsTuning(pub ProgressiveTuning);

impl EntryTuning for RefineRowsTuning {
    type Entry = readout_refine_rows::Entry;
    type Case = LevelCase;

    fn launches(&self) -> usize {
        1
    }

    fn bindings(&self) -> String {
        format!("A={}", self.0.activation.name())
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (vocabulary, hidden) = self.0.shape(inputs)?;
        Ok(vec![("V", vocabulary), ("D", hidden)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        certified_points(limits, self.0.certified_rows)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        Ok(vec![LevelCase::new(
            inputs,
            &self.0,
            point.rows,
            12,
        )?])
    }

    fn args<'a>(case: &'a mut Self::Case) -> readout_refine_rows::Args<'a> {
        let [top, bit3, rest, scales, radius] = &case.planes;
        readout_refine_rows::Args {
            features: &case.features,
            top,
            bit3,
            rest,
            scales,
            radius,
            coarse: &case.previous,
            floor: &case.floor,
            length: &case.length,
            draws: &case.selecting.draws,
            temperature: &case.selecting.temperature,
            mask: &case.selecting.mask,
            constrained: &case.selecting.constrained,
        }
    }

    generated_entry!(readout_refine_rows, this => readout_refine_rows::Elements {
        A: this.0.activation,
    });
}

/// `readout_exact_rows`: the exact logits of the rows level two keeps (a few
/// hundred of a large vocabulary).
pub(crate) struct ExactRowsTuning(pub ProgressiveTuning);

impl EntryTuning for ExactRowsTuning {
    type Entry = readout_exact_rows::Entry;
    type Case = LevelCase;

    fn launches(&self) -> usize {
        1
    }

    fn bindings(&self) -> String {
        format!("A={}", self.0.activation.name())
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (vocabulary, hidden) = self.0.shape(inputs)?;
        Ok(vec![("V", vocabulary), ("D", hidden)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        certified_points(limits, self.0.certified_rows)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        Ok(vec![LevelCase::new(
            inputs,
            &self.0,
            point.rows,
            1000,
        )?])
    }

    fn args<'a>(case: &'a mut Self::Case) -> readout_exact_rows::Args<'a> {
        let [top, bit3, rest, scales, radius] = &case.planes;
        readout_exact_rows::Args {
            features: &case.features,
            top,
            bit3,
            rest,
            scales,
            radius,
            fine: &case.previous,
            floor: &case.floor,
            length: &case.length,
            draws: &case.selecting.draws,
            temperature: &case.selecting.temperature,
            mask: &case.selecting.mask,
            constrained: &case.selecting.constrained,
        }
    }

    generated_entry!(readout_exact_rows, this => readout_exact_rows::Elements {
        A: this.0.activation,
    });
}

/// `readout_planes_rows`: the final norm and the exact logits of every
/// vocabulary row from the planes.
pub(crate) struct PlanesRowsTuning(pub ProgressiveTuning);

pub(crate) struct PlanesRowsCase {
    hidden: Tensor,
    norm: Tensor,
    planes: [Tensor; 5],
    out_rows: Tensor,
    epsilon: f32,
}

impl EntryTuning for PlanesRowsTuning {
    type Entry = readout_planes_rows::Entry;
    type Case = PlanesRowsCase;

    fn launches(&self) -> usize {
        1
    }

    fn bindings(&self) -> String {
        format!("NW={},A={}", self.0.norm.name(), self.0.activation.name())
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let (vocabulary, hidden) = self.0.shape(inputs)?;
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
        let (_, hidden) = self.0.shape(inputs)?;
        Ok(vec![PlanesRowsCase {
            hidden: inputs.activation(Element::f32(), &[point.rows, hidden], 1)?,
            norm: inputs.weight(WeightScope::Target, WeightKind::OutputNorm)?,
            planes: self.0.planes(inputs)?,
            out_rows: inputs.every_row(point.rows)?,
            epsilon: self.0.epsilon,
        }])
    }

    fn args<'a>(case: &'a mut Self::Case) -> readout_planes_rows::Args<'a> {
        let [top, bit3, rest, scales, _] = &case.planes;
        readout_planes_rows::Args {
            hidden: &case.hidden,
            norm: &case.norm,
            top,
            bit3,
            rest,
            scales,
            out_rows: &case.out_rows,
            epsilon: case.epsilon,
        }
    }

    generated_entry!(readout_planes_rows, this => readout_planes_rows::Elements {
        NW: this.0.norm,
        A: this.0.activation,
    });
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

    fn launches(&self) -> usize {
        1
    }

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
            rows: None,
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
            softcap: 0.0,
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

    fn launches(&self) -> usize {
        1
    }

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

/// `shape_rows`: constraint mask, penalties, temperature, top-k, min-p and top-p of each
/// selected row over `vocabulary` tokens (the target's, or the draft
/// head's). Its parameters partition the vocabulary without changing any
/// result.
pub(crate) struct ShapeRowsTuning {
    pub vocabulary: u64,
}

pub(crate) struct ShapeRowsCase {
    /// Shaped in place, so every run restores the raw logits.
    logits: CaseState,
    mask: Tensor,
    constrained: Tensor,
    params: Tensor,
    history: Tensor,
}

impl EntryTuning for ShapeRowsTuning {
    type Entry = shape_rows::Entry;
    type Case = ShapeRowsCase;

    fn launches(&self) -> usize {
        1
    }

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
        let words = vocabulary.div_ceil(32);
        Ok(vec![ShapeRowsCase {
            logits: inputs.state(logits(inputs, rows, vocabulary, 1)?, 0..rows)?,
            // Every other row constrained: both the masked and the free
            // path are measured.
            mask: inputs.u32s(&[rows, words], &vec![u32::MAX; (rows * words) as usize])?,
            constrained: inputs.i32s(
                &[rows],
                &(0..rows).map(|row| (row % 2) as i32).collect::<Vec<_>>(),
            )?,
            params: inputs.f32s(&[rows, SHAPING_WIDTH as u64], &params)?,
            history: inputs.i32s(&[rows, HISTORY_WIDTH as u64], &history)?,
        }])
    }

    fn args<'a>(case: &'a mut Self::Case) -> shape_rows::Args<'a> {
        shape_rows::Args {
            logits: case.logits.tensor_mut(),
            mask: &case.mask,
            constrained: &case.constrained,
            params: &case.params,
            history: &case.history,
        }
    }

    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        vec![("logits", &case.logits)]
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

    fn launches(&self) -> usize {
        1
    }

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

    fn state(case: &Self::Case) -> Vec<(&'static str, &CaseState)> {
        vec![("result", &case.result)]
    }

    generated_entry!(sample_rows);
}
