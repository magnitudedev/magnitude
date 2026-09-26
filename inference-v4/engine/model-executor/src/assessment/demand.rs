//! Header-derived decode demand: every launch of one plain target decode
//! step, keyed by the measured class it costs against, with how often it
//! launches and how many bytes it streams. Derived from the same program and
//! load plans native preparation uses, and from the state layout the memory
//! terms use for history bytes.

use super::basis::{ClassMeasurement, CostShape, MeasurementBasis, MeasurementKey};
use super::AssessmentError;
use crate::{FeedForwardProgramSlot, MixerProgramSlot, ModelLoadPlan, WeightPlan};
use magnitude_model_contracts::{ModelDefinition, WeightKind, WeightScope};
use magnitude_model_state::{KvCodec, LayerRef, ModelStateLayout};
use seismic::Element;

/// One measured class's share of a plain decode step. Streamed bytes grow
/// linearly with context depth only for history-reading classes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemandTerm {
    pub key: MeasurementKey,
    pub launches: u64,
    /// Bytes streamed per step independent of depth, summed over launches.
    pub bytes: u64,
    /// Additional bytes streamed per step for each token of context.
    pub bytes_per_context_token: u64,
    /// For a weight-streaming (`Curve`) class, the bytes each entry call
    /// this term belongs to streams in total, over all its segments: the
    /// size its cost is read at. Zero for other classes.
    pub launch_bytes: u64,
}

/// Every launch of one plain target decode step, grouped by measured class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeDemand {
    pub terms: Vec<DemandTerm>,
}

fn demand_error(message: impl Into<String>) -> AssessmentError {
    AssessmentError::Demand(message.into())
}

/// Terms in first-launch order; terms of equal keys and launch sizes
/// accumulate.
#[derive(Default)]
struct Terms(Vec<DemandTerm>);

impl Terms {
    /// Launches that are whole entry calls: a weight-streaming call's
    /// launch size is its own bytes per launch.
    fn add(
        &mut self,
        key: MeasurementKey,
        launches: u64,
        bytes: u64,
        bytes_per_context_token: u64,
    ) -> Result<(), AssessmentError> {
        let launch_bytes = match key.class.cost_shape() {
            CostShape::Curve => bytes / launches,
            CostShape::PerLaunch | CostShape::Linear => 0,
        };
        self.add_in_launch(key, launches, bytes, bytes_per_context_token, launch_bytes)
    }

    fn add_in_launch(
        &mut self,
        key: MeasurementKey,
        launches: u64,
        bytes: u64,
        bytes_per_context_token: u64,
        launch_bytes: u64,
    ) -> Result<(), AssessmentError> {
        let overflow = || demand_error(format!("{} demand overflows", key.class.name()));
        match self
            .0
            .iter_mut()
            .find(|term| term.key == key && term.launch_bytes == launch_bytes)
        {
            Some(term) => {
                term.launches = term.launches.checked_add(launches).ok_or_else(overflow)?;
                term.bytes = term.bytes.checked_add(bytes).ok_or_else(overflow)?;
                term.bytes_per_context_token = term
                    .bytes_per_context_token
                    .checked_add(bytes_per_context_token)
                    .ok_or_else(overflow)?;
            }
            None => self.0.push(DemandTerm {
                key,
                launches,
                bytes,
                bytes_per_context_token,
                launch_bytes,
            }),
        }
        Ok(())
    }

    /// One launch of a segmented weight-streaming entry: each segment's
    /// bytes cost against its own representation's curve at the whole
    /// launch's size, and the launch is counted once, on the first segment.
    fn segmented(
        &mut self,
        segments: &[(&WeightPlan, u64)],
        key: impl Fn(Element) -> MeasurementKey,
    ) -> Result<(), AssessmentError> {
        let launch_bytes = segments
            .iter()
            .try_fold(0u64, |total, (_, bytes)| total.checked_add(*bytes))
            .ok_or_else(|| demand_error("launch bytes overflow"))?;
        for (index, (weight, bytes)) in segments.iter().enumerate() {
            self.add_in_launch(
                key(weight.resident),
                u64::from(index == 0),
                *bytes,
                0,
                launch_bytes,
            )?;
        }
        Ok(())
    }
}

/// The planned weight of `kind` in `scope`, checked against the element the
/// program slot binds.
fn planned<'a>(
    load: &'a ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    bound: Element,
) -> Result<&'a WeightPlan, AssessmentError> {
    let weight = load
        .target()
        .iter()
        .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
        .ok_or_else(|| demand_error(format!("{kind:?} weight is absent for {scope:?}")))?;
    if weight.resident != bound {
        return Err(demand_error(format!(
            "{kind:?} binding for {scope:?} disagrees with the resident load plan"
        )));
    }
    Ok(weight)
}

/// The resident bytes of `selected` of an expert tensor's `experts` equal
/// slabs: a decode row streams only its selected experts.
fn selected_experts(
    weight: &WeightPlan,
    experts: u64,
    selected: u64,
) -> Result<u64, AssessmentError> {
    if experts == 0 || selected > experts || weight.resident_bytes % experts != 0 {
        return Err(demand_error(format!(
            "expert tensor {:?} does not split into {experts} slabs",
            weight.descriptor.name
        )));
    }
    (weight.resident_bytes / experts)
        .checked_mul(selected)
        .ok_or_else(|| demand_error("selected expert bytes overflow"))
}

fn usize_bytes(bytes: usize) -> Result<u64, AssessmentError> {
    u64::try_from(bytes).map_err(|_| demand_error("state bytes exceed u64"))
}

impl DecodeDemand {
    /// Every launch one plain decode step of one row makes, from the model's
    /// program plan, resident load plan and state layout.
    pub fn from_model(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
    ) -> Result<Self, AssessmentError> {
        let program = load
            .program_plan(definition, codec)
            .map_err(|error| demand_error(error.to_string()))?;
        let layout =
            ModelStateLayout::derive(&definition.geometry, 0, codec, 0).map_err(demand_error)?;
        let target = program.target();
        let mut terms = Terms::default();

        let embedding = target.embedding();
        let table = planned(
            load,
            WeightScope::Target,
            WeightKind::Embedding,
            embedding.table,
        )?;
        let vocabulary = definition.geometry.vocabulary;
        terms.add(
            MeasurementKey::embedding_rows(embedding.table, embedding.activation),
            1,
            table.resident_bytes / vocabulary,
            0,
        )?;

        let mut recurrent_layer = 0usize;
        for (index, block) in target.blocks().iter().enumerate() {
            let layer =
                u32::try_from(index).map_err(|_| demand_error("block index exceeds u32"))?;
            let scope = WeightScope::TargetBlock(layer);
            match block.mixer() {
                MixerProgramSlot::Attention(binding) => {
                    let weight = |kind, bound| planned(load, scope, kind, bound);
                    planned(load, scope, WeightKind::InputNorm, binding.norm)?;
                    let segments = [
                        weight(WeightKind::QueryGate, binding.query_gate)?,
                        weight(WeightKind::Key, binding.key)?,
                        weight(WeightKind::Value, binding.value)?,
                    ];
                    terms.segmented(
                        &segments.map(|weight| (weight, weight.resident_bytes)),
                        |element| {
                            MeasurementKey::attention_project(
                                binding.norm,
                                element,
                                binding.activation,
                            )
                        },
                    )?;
                    let affine = match binding.history {
                        KvCodec::Dense => false,
                        KvCodec::AffineK8V4 => true,
                        KvCodec::RotatedK4V4 => {
                            return Err(demand_error(
                                "rotated K4/V4 history has no native attention entry",
                            ))
                        }
                    };
                    let history = layout
                        .target_history
                        .iter()
                        .find(|component| component.layer == LayerRef::Target(layer))
                        .ok_or_else(|| {
                            demand_error(format!("attention block {index} has no history"))
                        })?
                        .planes()
                        .iter()
                        .try_fold(0u64, |total, plane| {
                            total
                                .checked_add(usize_bytes(plane.row_bytes)?)
                                .ok_or_else(|| demand_error("history row bytes overflow"))
                        })?;
                    terms.add(
                        MeasurementKey::attention_decode(affine, binding.shape, binding.activation),
                        1,
                        0,
                        history,
                    )?;
                    let output = weight(WeightKind::AttentionOutput, binding.output)?;
                    terms.add(
                        MeasurementKey::attention_output(binding.output, binding.activation),
                        1,
                        output.resident_bytes,
                        0,
                    )?;
                }
                MixerProgramSlot::Recurrent(binding) => {
                    let weight = |kind, bound| planned(load, scope, kind, bound);
                    planned(load, scope, WeightKind::InputNorm, binding.norm)?;
                    let segments = [
                        weight(WeightKind::RecurrentQueryKeyValue, binding.qkv)?,
                        weight(WeightKind::RecurrentGate, binding.gate)?,
                        weight(WeightKind::RecurrentAlpha, binding.alpha)?,
                        weight(WeightKind::RecurrentBeta, binding.beta)?,
                    ];
                    terms.segmented(
                        &segments.map(|weight| (weight, weight.resident_bytes)),
                        |element| {
                            MeasurementKey::delta_project(binding.norm, element, binding.activation)
                        },
                    )?;
                    // The step reads and publishes the layer's recurrent bank:
                    // window, delta and tape components.
                    let first = recurrent_layer * 3;
                    let state = layout
                        .target_recurrent
                        .get(first..first + 3)
                        .ok_or_else(|| {
                            demand_error(format!("recurrent block {index} has no state"))
                        })?
                        .iter()
                        .try_fold(0u64, |total, component| {
                            total
                                .checked_add(usize_bytes(component.bytes().map_err(demand_error)?)?)
                                .ok_or_else(|| demand_error("recurrent state bytes overflow"))
                        })?;
                    recurrent_layer += 1;
                    terms.add(
                        MeasurementKey::delta_step(
                            binding.key_heads,
                            binding.value_heads,
                            binding.width,
                            binding.convolution_width,
                            binding.activation,
                        ),
                        1,
                        state,
                        0,
                    )?;
                    planned(
                        load,
                        scope,
                        WeightKind::RecurrentNorm,
                        binding.recurrent_norm,
                    )?;
                    let output = weight(WeightKind::RecurrentOutput, binding.output)?;
                    terms.add(
                        MeasurementKey::delta_output(
                            binding.recurrent_norm,
                            binding.output,
                            binding.activation,
                        ),
                        1,
                        output.resident_bytes,
                        0,
                    )?;
                }
            }
            match block.feed_forward() {
                FeedForwardProgramSlot::Dense(binding) => {
                    let weight = |kind, bound| planned(load, scope, kind, bound);
                    planned(load, scope, WeightKind::FeedForwardNorm, binding.norm)?;
                    let segments = [
                        weight(WeightKind::DenseGate, binding.gate)?,
                        weight(WeightKind::DenseUp, binding.up)?,
                    ];
                    terms.segmented(
                        &segments.map(|weight| (weight, weight.resident_bytes)),
                        |element| {
                            MeasurementKey::dense_expand(binding.norm, element, binding.activation)
                        },
                    )?;
                    let down = weight(WeightKind::DenseDown, binding.down)?;
                    terms.add(
                        MeasurementKey::dense_output(binding.down, binding.activation),
                        1,
                        down.resident_bytes,
                        0,
                    )?;
                }
                FeedForwardProgramSlot::Routed(binding) => {
                    let weight = |kind, bound| planned(load, scope, kind, bound);
                    planned(load, scope, WeightKind::FeedForwardNorm, binding.norm)?;
                    let router = weight(WeightKind::Router, binding.router)?;
                    terms.add(
                        MeasurementKey::routed_route(
                            binding.norm,
                            binding.router,
                            binding.activation,
                            binding.hidden,
                            binding.experts,
                            binding.selected,
                        ),
                        1,
                        router.resident_bytes,
                        0,
                    )?;
                    let chosen = |weight: &WeightPlan| {
                        selected_experts(weight, binding.experts, binding.selected)
                    };
                    let expert_gate = weight(WeightKind::ExpertGate, binding.expert_gate)?;
                    let expert_up = weight(WeightKind::ExpertUp, binding.expert_up)?;
                    let shared_gate = weight(WeightKind::SharedGate, binding.shared_gate)?;
                    let shared_up = weight(WeightKind::SharedUp, binding.shared_up)?;
                    terms.segmented(
                        &[
                            (expert_gate, chosen(expert_gate)?),
                            (expert_up, chosen(expert_up)?),
                            (shared_gate, shared_gate.resident_bytes),
                            (shared_up, shared_up.resident_bytes),
                        ],
                        |element| MeasurementKey::routed_expand(element, binding.activation),
                    )?;
                    let expert_down = weight(WeightKind::ExpertDown, binding.expert_down)?;
                    let shared_down = weight(WeightKind::SharedDown, binding.shared_down)?;
                    terms.segmented(
                        &[
                            (expert_down, chosen(expert_down)?),
                            (shared_down, shared_down.resident_bytes),
                        ],
                        |element| MeasurementKey::routed_output(element, binding.activation),
                    )?;
                }
            }
        }

        // The selection readout graph: final-norm features, the vocabulary
        // projection (its physical matrix, even when tied to the embedding),
        // and sampling of the one F32 logits row.
        let readout = target.readout();
        let norm = planned(
            load,
            WeightScope::Target,
            WeightKind::OutputNorm,
            readout.norm,
        )?;
        terms.add(
            MeasurementKey::readout_features(readout.norm, readout.activation),
            1,
            norm.resident_bytes,
            0,
        )?;
        let output = planned(
            load,
            WeightScope::Target,
            WeightKind::Output,
            readout.weight,
        )?;
        terms.add(
            MeasurementKey::readout_head(readout.norm, readout.weight, readout.activation),
            1,
            output.resident_bytes,
            0,
        )?;
        terms.add(
            MeasurementKey::sample_rows(),
            1,
            vocabulary
                .checked_mul(4)
                .ok_or_else(|| demand_error("logits bytes overflow"))?,
            0,
        )?;
        // Every entry call of the step depends on the one before it (the
        // step's graph runs are one dependency chain), and the step is
        // submitted once and waited on for its selection.
        let calls = terms
            .0
            .iter()
            .try_fold(0u64, |calls, term| calls.checked_add(term.launches))
            .ok_or_else(|| demand_error("entry call count overflows"))?;
        terms.add(MeasurementKey::launch_dependency(), calls, 0, 0)?;
        terms.add(MeasurementKey::step_submission(), 1, 0, 0)?;
        Ok(Self { terms: terms.0 })
    }

    /// Classes this demand needs that the basis did not measure: absent from
    /// the basis or recorded unsupported. A nonempty result means the model
    /// is incompatible with the basis's device.
    pub fn unmeasured<'a>(
        &'a self,
        basis: &'a MeasurementBasis,
    ) -> Vec<(&'a MeasurementKey, Option<&'a str>)> {
        self.terms
            .iter()
            .filter_map(|term| match basis.get(&term.key) {
                Some(ClassMeasurement::Measured { .. }) => None,
                Some(ClassMeasurement::Unsupported { reason }) => {
                    Some((&term.key, Some(reason.as_str())))
                }
                None => Some((&term.key, None)),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::basis::OperationClass;
    use seismic::Layout;

    fn fixture() -> (ModelDefinition, ModelLoadPlan) {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        (definition, load)
    }

    fn resident(load: &ModelLoadPlan, kind: WeightKind) -> &WeightPlan {
        load.target()
            .iter()
            .find(|weight| weight.role.kind == kind)
            .unwrap()
    }

    fn term(demand: &DecodeDemand, class: OperationClass) -> &DemandTerm {
        let [term] = demand
            .terms
            .iter()
            .filter(|term| term.key.class == class)
            .collect::<Vec<_>>()[..]
        else {
            panic!("fixture has one {} term", class.name());
        };
        term
    }

    #[test]
    fn every_plain_decode_launch_is_a_term_with_planned_bytes() {
        let (definition, load) = fixture();
        let demand = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        let classes = demand
            .terms
            .iter()
            .map(|term| term.key.class)
            .collect::<Vec<_>>();
        assert_eq!(
            classes,
            [
                OperationClass::EmbeddingRows,
                OperationClass::AttentionProject,
                OperationClass::AttentionDecode,
                OperationClass::AttentionOutput,
                OperationClass::DenseExpand,
                OperationClass::DenseOutput,
                OperationClass::ReadoutFeatures,
                OperationClass::ReadoutHead,
                OperationClass::SampleRows,
                OperationClass::LaunchDependency,
                OperationClass::StepSubmission,
            ]
        );
        // Nine entry calls, each depending on the previous; one step.
        assert_eq!(term(&demand, OperationClass::LaunchDependency).launches, 9);
        assert!(demand
            .terms
            .iter()
            .filter(|term| term.key.class != OperationClass::LaunchDependency)
            .all(|term| term.launches == 1));
        let bytes = |kinds: &[WeightKind]| {
            kinds
                .iter()
                .map(|kind| resident(&load, *kind).resident_bytes)
                .sum::<u64>()
        };
        // The fixture's weights share one representation, so each segmented
        // entry is one term.
        assert_eq!(
            term(&demand, OperationClass::AttentionProject).bytes,
            bytes(&[WeightKind::QueryGate, WeightKind::Key, WeightKind::Value])
        );
        assert_eq!(
            term(&demand, OperationClass::DenseExpand).bytes,
            bytes(&[WeightKind::DenseGate, WeightKind::DenseUp])
        );
        assert_eq!(
            term(&demand, OperationClass::DenseOutput).bytes,
            bytes(&[WeightKind::DenseDown])
        );
        assert_eq!(
            term(&demand, OperationClass::ReadoutHead).bytes,
            bytes(&[WeightKind::Output])
        );
        assert_eq!(
            term(&demand, OperationClass::SampleRows).bytes,
            definition.geometry.vocabulary * 4
        );
        let key = &term(&demand, OperationClass::DenseOutput).key;
        assert_eq!(
            *key,
            MeasurementKey::dense_output(
                resident(&load, WeightKind::DenseDown).resident,
                Element::bf16()
            )
        );
    }

    #[test]
    fn history_bytes_per_token_follow_the_state_layout_and_codec() {
        let (definition, load) = fixture();
        // One kv head of width 64: dense bf16 keys and values, 2 × 64 × 2.
        let dense = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        let decode = term(&dense, OperationClass::AttentionDecode);
        assert_eq!((decode.bytes, decode.bytes_per_context_token), (0, 256));
        assert!(dense
            .terms
            .iter()
            .filter(|term| term.key.class != OperationClass::AttentionDecode)
            .all(|term| term.bytes_per_context_token == 0));
        // Affine: 64 key bytes, 32 value bytes and two (scale, zero) f16
        // pairs per 32-value group for each.
        let affine = DecodeDemand::from_model(&definition, &load, KvCodec::AffineK8V4).unwrap();
        let decode = term(&affine, OperationClass::AttentionDecodeK8V4);
        assert_eq!(decode.bytes_per_context_token, 64 + 32 + 2 * (2 * 2 * 2));
        assert!(matches!(
            DecodeDemand::from_model(&definition, &load, KvCodec::RotatedK4V4),
            Err(AssessmentError::Demand(_))
        ));
    }

    #[test]
    fn segments_cost_by_representation_at_their_launch_size() {
        let mut terms = Terms::default();
        let (_, load) = fixture();
        let query = resident(&load, WeightKind::QueryGate);
        let mut value = resident(&load, WeightKind::Value).clone();
        value.resident = Element::stored("q6k", Layout::Rows16).unwrap();
        let key =
            |element| MeasurementKey::attention_project(Element::bf16(), element, Element::bf16());
        terms
            .segmented(&[(query, 10), (query, 20), (&value, 30)], key)
            .unwrap();
        terms
            .segmented(&[(query, 1), (query, 2), (&value, 3)], key)
            .unwrap();
        // A second launch of the first size accumulates with it.
        terms
            .segmented(&[(query, 10), (query, 20), (&value, 30)], key)
            .unwrap();
        assert_eq!(
            terms.0,
            [
                DemandTerm {
                    key: key(query.resident),
                    launches: 2,
                    bytes: 60,
                    bytes_per_context_token: 0,
                    launch_bytes: 60,
                },
                DemandTerm {
                    key: key(value.resident),
                    launches: 0,
                    bytes: 60,
                    bytes_per_context_token: 0,
                    launch_bytes: 60,
                },
                DemandTerm {
                    key: key(query.resident),
                    launches: 1,
                    bytes: 3,
                    bytes_per_context_token: 0,
                    launch_bytes: 6,
                },
                DemandTerm {
                    key: key(value.resident),
                    launches: 0,
                    bytes: 3,
                    bytes_per_context_token: 0,
                    launch_bytes: 6,
                },
            ]
        );
    }
}
