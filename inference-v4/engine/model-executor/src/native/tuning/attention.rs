//! Tuning cases of the gated attention block: the normed Q/K/V projection,
//! the fused attention entries (`qwen_attention_decode` up to
//! [`DECODE_ROWS`] rows, `qwen_attention_prefill` beyond) and the output
//! projection.
//!
//! Attention points cross the rows an entry serves with the served history
//! lengths. Each point is one request: its rows see `context` accepted
//! history rows and append their keys and values right after them, as a
//! decode step, a speculative verification or a prefill chunk does. The
//! points of one history length read one shared pair of pseudo-random
//! history arenas through views that end at their appended rows; those rows
//! are the case state restored before each validation run. No point reads
//! rows another point appends, so every configuration sees the same inputs
//! whatever the point order.

use super::cases::projection_shape;
use super::{
    row_points, served_row_points, with_contexts, CaseState, EntryTuning, PointShape,
    TuningInputs, TuningLimits,
};
use crate::programs::graph::attention::{rotary_components, rotary_frequencies, DECODE_ROWS};
use crate::AttentionShape;
use magnitude_model_contracts::{MixerGeometry, RotarySemantics, WeightKind, WeightScope};
use magnitude_model_kernels::{
    qwen_attention_decode, qwen_attention_output, qwen_attention_prefill, qwen_attention_project,
};
use seismic::{Element, Tensor};

/// `qwen_attention_project`: RMS prologue, fused query+gate | key | value
/// projection.
pub(crate) struct AttentionProjectTuning {
    pub norm: Element,
    pub query_gate: Element,
    pub key: Element,
    pub value: Element,
    pub activation: Element,
    pub shape: AttentionShape,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

pub(crate) struct AttentionProjectCase {
    hidden: Tensor,
    input_norm: Tensor,
    query_norm: Tensor,
    query_gate: Tensor,
    key: Tensor,
    value: Tensor,
    epsilon: f32,
}

impl AttentionProjectTuning {
    fn elements(&self) -> qwen_attention_project::Elements {
        qwen_attention_project::Elements {
            NW: self.norm,
            QW: self.query_gate,
            KW: self.key,
            VW: self.value,
            A: self.activation,
        }
    }
}

impl EntryTuning for AttentionProjectTuning {
    type Entry = qwen_attention_project::Entry;
    type Case = AttentionProjectCase;

    fn bindings(&self) -> String {
        format!(
            "NW={},QW={},KW={},VW={},A={}",
            self.norm.name(),
            self.query_gate.name(),
            self.key.name(),
            self.value.name(),
            self.activation.name()
        )
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        let (_, hidden) = projection_shape(inputs, &self.scopes, WeightKind::QueryGate)?;
        if hidden != shape.hidden {
            return Err(format!(
                "the query projection is {hidden} wide, the binding {}",
                shape.hidden
            ));
        }
        Ok(vec![
            ("D", shape.hidden),
            ("KV", shape.kv_heads),
            ("G", shape.group),
            ("W", shape.width),
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
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                Ok(AttentionProjectCase {
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, self.shape.hidden],
                        index as u64 + 1,
                    )?,
                    input_norm: inputs.weight(scope, WeightKind::InputNorm)?,
                    query_norm: inputs.weight(scope, WeightKind::QueryNorm)?,
                    query_gate: inputs.weight(scope, WeightKind::QueryGate)?,
                    key: inputs.weight(scope, WeightKind::Key)?,
                    value: inputs.weight(scope, WeightKind::Value)?,
                    epsilon: self.epsilon,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> qwen_attention_project::Args<'a> {
        qwen_attention_project::Args {
            hidden: &case.hidden,
            input_norm: &case.input_norm,
            query_norm: &case.query_norm,
            query_gate_weight: &case.query_gate,
            key_weight: &case.key,
            value_weight: &case.value,
            epsilon: case.epsilon,
        }
    }

    generated_entry!(qwen_attention_project, this => this.elements());
}

/// `qwen_attention_output`: output projection plus residual.
pub(crate) struct AttentionOutputTuning {
    pub output: Element,
    pub activation: Element,
    pub shape: AttentionShape,
    pub scopes: Vec<WeightScope>,
}

pub(crate) struct AttentionOutputCase {
    hidden: Tensor,
    gated: Tensor,
    output: Tensor,
}

impl AttentionOutputTuning {
    fn elements(&self) -> qwen_attention_output::Elements {
        qwen_attention_output::Elements {
            OW: self.output,
            A: self.activation,
        }
    }
}

impl EntryTuning for AttentionOutputTuning {
    type Entry = qwen_attention_output::Entry;
    type Case = AttentionOutputCase;

    fn bindings(&self) -> String {
        format!("OW={},A={}", self.output.name(), self.activation.name())
    }

    fn statics(&self, inputs: &TuningInputs<'_, '_>) -> Result<Vec<(&'static str, u64)>, String> {
        let shape = self.shape;
        let heads = shape.kv_heads * shape.group;
        if projection_shape(inputs, &self.scopes, WeightKind::AttentionOutput)?
            != (shape.hidden, heads * shape.width)
        {
            return Err("the attention output projection disagrees with the binding".into());
        }
        Ok(vec![("D", shape.hidden), ("Q", heads), ("W", shape.width)])
    }

    fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
        row_points(limits)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
    ) -> Result<Vec<Self::Case>, String> {
        let shape = self.shape;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 2 * index as u64;
                Ok(AttentionOutputCase {
                    hidden: inputs.activation(
                        Element::f32(),
                        &[point.rows, shape.hidden],
                        seed + 1,
                    )?,
                    gated: inputs.activation(
                        self.activation,
                        &[point.rows, shape.kv_heads * shape.group, shape.width],
                        seed + 2,
                    )?,
                    output: inputs.weight(scope, WeightKind::AttentionOutput)?,
                })
            })
            .collect()
    }

    fn args<'a>(case: &'a mut Self::Case) -> qwen_attention_output::Args<'a> {
        qwen_attention_output::Args {
            hidden: &case.hidden,
            gated: &case.gated,
            output_weight: &case.output,
        }
    }

    generated_entry!(qwen_attention_output, this => this.elements());
}

/// What both fused attention entries tune over.
pub(crate) struct AttentionMix {
    pub activation: Element,
    pub shape: AttentionShape,
    pub scopes: Vec<WeightScope>,
    pub epsilon: f32,
}

/// `qwen_attention_decode`, for row classes up to [`DECODE_ROWS`].
pub(crate) struct AttentionDecodeTuning(pub AttentionMix);

/// `qwen_attention_prefill`, for row classes beyond [`DECODE_ROWS`].
pub(crate) struct AttentionPrefillTuning(pub AttentionMix);

/// One argument set of either fused entry; they share one contract.
pub(crate) struct AttentionMixCase {
    query_gate: Tensor,
    key: Tensor,
    value: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    rotary_components: Tensor,
    rotary_frequencies: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    fresh: Tensor,
    destinations: Tensor,
    history_key: CaseState,
    history_value: CaseState,
    epsilon: f32,
    scale: f32,
}

/// The model's rotary embedding: every attention block shares it.
fn rotary(inputs: &TuningInputs<'_, '_>) -> Result<RotarySemantics, String> {
    inputs
        .definition
        .geometry
        .blocks
        .iter()
        .find_map(|block| match &block.mixer {
            MixerGeometry::Attention(attention) => Some(attention.rotary.clone()),
            MixerGeometry::Recurrent(_) => None,
        })
        .ok_or_else(|| "the model has no attention block".to_owned())
}

impl AttentionMix {
    fn bindings(&self) -> String {
        format!("A={}", self.activation.name())
    }

    fn statics(&self) -> Vec<(&'static str, u64)> {
        let shape = self.shape;
        vec![
            ("KV", shape.kv_heads),
            ("G", shape.group),
            ("P", shape.rotary_pairs),
            ("S", shape.width - 2 * shape.rotary_pairs),
        ]
    }

    /// The history rows the arena of points with `context` history rows
    /// holds: the history plus the most rows any such point appends.
    fn history_rows(points: &[PointShape], context: u64) -> u64 {
        context
            + points
                .iter()
                .filter(|point| point.context == Some(context))
                .map(|point| point.rows)
                .max()
                .unwrap_or(0)
    }

    fn rotation(
        &self,
        inputs: &mut TuningInputs<'_, '_>,
        point: &PointShape,
        points: Vec<PointShape>,
    ) -> Result<Vec<AttentionMixCase>, String> {
        let shape = self.shape;
        let rows = point.rows;
        let context = point
            .context
            .ok_or("attention tuning points carry a history length")?;
        // Points of one history length share an arena: they all read its
        // first `context` rows, which nothing writes, and append after them.
        // A longer history has its own arena, so no point ever reads rows
        // another point appended.
        let history_rows = Self::history_rows(&points, context);
        let arena = |name: &str, seed: u64, inputs: &mut TuningInputs<'_, '_>| {
            inputs.shared(format!("{name}-c{context}"), |inputs| {
                inputs.activation(
                    self.activation,
                    &[history_rows, shape.kv_heads, shape.width],
                    seed,
                )
            })
        };
        let keys = arena("history_key", 0x6b, inputs)?;
        let values = arena("history_value", 0x76, inputs)?;
        let view = context + rows;
        let appended = context..view;
        let history_key = inputs.state(
            keys.slice_leading(0, view).map_err(|error| error.to_string())?,
            appended.clone(),
        )?;
        let history_value = inputs.state(
            values
                .slice_leading(0, view)
                .map_err(|error| error.to_string())?,
            appended,
        )?;
        let tables = inputs.batch(rows, context, 1, context)?;
        let segments = tables.class.segments() as u64;
        let rotary = rotary(inputs)?;
        let components = rotary_components(&rotary)?;
        let frequencies = rotary_frequencies(&rotary);
        let pairs = components.len() as u64;
        let coordinates = tables.coordinates.iter().flatten().copied().collect::<Vec<_>>();
        let visible = tables
            .visible
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let fresh = tables.fresh.iter().flatten().copied().collect::<Vec<_>>();
        let heads = shape.kv_heads * shape.group;
        TuningInputs::rotation_scopes(&self.scopes, point)
            .into_iter()
            .enumerate()
            .map(|(index, scope)| {
                let seed = 3 * index as u64;
                Ok(AttentionMixCase {
                    query_gate: inputs.activation(
                        self.activation,
                        &[rows, heads * 2 * shape.width],
                        seed + 1,
                    )?,
                    key: inputs.activation(
                        self.activation,
                        &[rows, shape.kv_heads * shape.width],
                        seed + 2,
                    )?,
                    value: inputs.activation(
                        self.activation,
                        &[rows, shape.kv_heads * shape.width],
                        seed + 3,
                    )?,
                    query_norm: inputs.weight(scope, WeightKind::QueryNorm)?,
                    key_norm: inputs.weight(scope, WeightKind::KeyNorm)?,
                    rotary_components: inputs.i32s(&[pairs], &components)?,
                    rotary_frequencies: inputs.f32s(&[pairs], &frequencies)?,
                    coordinates: inputs.i32s(&[rows, 4], &coordinates)?,
                    visible: inputs.i32s(&[rows, segments, 2], &visible)?,
                    fresh: inputs.i32s(&[rows, 2], &fresh)?,
                    destinations: inputs.i32s(&[rows], &tables.destinations)?,
                    history_key: history_key.share(),
                    history_value: history_value.share(),
                    epsilon: self.epsilon,
                    scale: 1.0 / (shape.width as f32).sqrt(),
                })
            })
            .collect()
    }
}

/// The two fused entries differ only in the rows they serve; they share one
/// contract and argument set.
macro_rules! mix_entry {
    ($tuning:ident, $module:ident, $serves:expr) => {
        impl $tuning {
            fn served(limits: TuningLimits) -> Vec<PointShape> {
                with_contexts(limits, served_row_points(limits.max_rows, $serves))
            }
        }

        impl EntryTuning for $tuning {
            type Entry = $module::Entry;
            type Case = AttentionMixCase;

            fn bindings(&self) -> String {
                self.0.bindings()
            }

            fn statics(
                &self,
                _inputs: &TuningInputs<'_, '_>,
            ) -> Result<Vec<(&'static str, u64)>, String> {
                Ok(self.0.statics())
            }

            fn points(&self, limits: TuningLimits) -> Vec<PointShape> {
                Self::served(limits)
            }

            fn rotation(
                &self,
                inputs: &mut TuningInputs<'_, '_>,
                point: &PointShape,
            ) -> Result<Vec<Self::Case>, String> {
                let points = Self::served(inputs.limits);
                self.0.rotation(inputs, point, points)
            }

            fn args<'a>(case: &'a mut Self::Case) -> $module::Args<'a> {
                $module::Args {
                    query_gate: &case.query_gate,
                    key: &case.key,
                    value: &case.value,
                    query_norm: &case.query_norm,
                    key_norm: &case.key_norm,
                    rotary_components: &case.rotary_components,
                    rotary_frequencies: &case.rotary_frequencies,
                    coordinates: &case.coordinates,
                    visible: &case.visible,
                    fresh: &case.fresh,
                    destinations: &case.destinations,
                    history_key: case.history_key.tensor_mut(),
                    history_value: case.history_value.tensor_mut(),
                    epsilon: case.epsilon,
                    scale: case.scale,
                }
            }

            fn state(case: &Self::Case) -> Vec<&CaseState> {
                vec![&case.history_key, &case.history_value]
            }

            generated_entry!($module, this => $module::Elements { A: this.0.activation });
        }
    };
}

mix_entry!(
    AttentionDecodeTuning,
    qwen_attention_decode,
    |rows| rows <= DECODE_ROWS
);
mix_entry!(
    AttentionPrefillTuning,
    qwen_attention_prefill,
    |rows| rows > DECODE_ROWS
);
