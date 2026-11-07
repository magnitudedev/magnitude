use super::{AttentionKernels, DenseKernels};
use crate::{AttentionBinding, DenseBinding, ScalableWeight};
use magnitude_family_contracts::{
    AffinePart, CorrectorSite, CorrectorWeight, SublayerIndex, WeightKind, WeightScope,
};
use magnitude_kernels::{
    corrector_attention, corrector_factors, corrector_features, corrector_nodes, corrector_path,
    corrector_rows, dense_output, draft_convolve_input, draft_convolve_residual, draft_gated_rows,
    draft_path_step, draft_top_k, embedding_rows, project_rows, readout_features_rows, sample_rows,
    shape_rows, widen_rows,
};
use seismic::{Element, NativeKernel};
use std::collections::HashMap;

/// Immutable specializations of a separate draft (DFlash, DSpark, DFlash2,
/// LiLiCorr), one per distinct binding. Token selection runs over the full
/// target vocabulary.
#[derive(Debug, Default)]
pub struct DraftKernels {
    /// The layers' block attention and their context injection (the same
    /// attention with the fusion norm as its input norm).
    pub(super) attention: HashMap<AttentionBinding, AttentionKernels>,
    pub(super) dense: HashMap<DenseBinding, DenseKernels>,
    /// Embeds the block's tokens (raw rows), and a corrector's candidates.
    pub(super) embedding: Option<NativeKernel<embedding_rows::Entry>>,
    /// The output norm and the target's vocabulary projection of the
    /// proposing rows.
    pub(super) head: Option<super::ProjectionKernels>,
    /// Token selection over the readout vocabulary.
    pub(super) shape: Option<NativeKernel<shape_rows::Entry>>,
    pub(super) sample: Option<NativeKernel<sample_rows::Entry>>,
    /// Widens a device-conditioned entry's target features to F32.
    pub(super) widen: Option<NativeKernel<widen_rows::Entry>>,
    pub(super) markov: Option<MarkovKernels>,
    /// The unfused stages' projections and norms (`stage_entries`), shared
    /// by the convolved layers, the selector and the corrector.
    pub(super) projections: DraftProjections,
    pub(super) norms: DraftNorms,
    pub(super) convolved: Option<ConvolvedKernels>,
    pub(super) selector: Option<SelectorKernels>,
    pub(super) corrector: Option<CorrectorKernels>,
}

/// An unfused draft projection's specialization: its weight kind, weight
/// element and published element. Every scope sharing it has one
/// accumulator-scale extent.
pub type DraftProjectionKey = (WeightKind, Element, Element);

/// A plain `project_rows` per [`DraftProjectionKey`], prepared once for
/// every scope that shares it.
pub type DraftProjections = HashMap<DraftProjectionKey, NativeKernel<project_rows::Entry>>;

/// A norm over rows (`readout_features_rows`): its weight element and width.
pub type DraftNormKey = (Element, u64);

pub type DraftNorms = HashMap<DraftNormKey, NativeKernel<readout_features_rows::Entry>>;

/// A convolved backbone's entries beside its projections and norms: the two
/// convolution halves and the feed-forward's gated product.
#[derive(Clone, Debug)]
pub struct ConvolvedKernels {
    pub convolve_input: NativeKernel<draft_convolve_input::Entry>,
    pub convolve_residual: NativeKernel<draft_convolve_residual::Entry>,
    pub gated: NativeKernel<draft_gated_rows::Entry>,
}

/// DFlash2's candidate path: the top-k, the codebook gathers (at the
/// selector rank) and the ordered path step.
#[derive(Clone, Debug)]
pub struct SelectorKernels {
    pub top_k: NativeKernel<draft_top_k::Entry>,
    pub predecessor: NativeKernel<embedding_rows::Entry>,
    pub successor: NativeKernel<embedding_rows::Entry>,
    pub path: NativeKernel<draft_path_step::Entry>,
}

/// A `corrector_rows` specialization: its row width, residual rows and
/// published element.
pub type CorrectorRowsKey = (u64, u64, Element);

/// LiLiCorr's corrector entries beside its projections and norms (its
/// candidates' embedding rows are the block embedding's entry).
#[derive(Clone, Debug)]
pub struct CorrectorKernels {
    pub top_k: NativeKernel<draft_top_k::Entry>,
    pub features: NativeKernel<corrector_features::Entry>,
    pub rows: HashMap<CorrectorRowsKey, NativeKernel<corrector_rows::Entry>>,
    pub nodes: NativeKernel<corrector_nodes::Entry>,
    pub attention: NativeKernel<corrector_attention::Entry>,
    pub factors: NativeKernel<corrector_factors::Entry>,
    pub path: NativeKernel<corrector_path::Entry>,
}

/// One draft layer's entries.
#[derive(Clone, Debug)]
pub struct DraftBlockKernels {
    pub attention: AttentionKernels,
    pub injection: AttentionKernels,
    pub dense: DenseKernels,
}

/// DSpark's chain: the Markov memory of the previous token, its projection
/// onto the slot logits. The fixed zero confidence threshold needs no kernel.
#[derive(Clone, Debug)]
pub struct MarkovKernels {
    pub embedding: NativeKernel<embedding_rows::Entry>,
    pub projection: NativeKernel<dense_output::Entry>,
}

/// One convolved draft layer's unfused entries, in graph order: the
/// attention norm, its coefficient projection, the query, key, value and
/// output projections; the feed-forward norm, its coefficient projection,
/// and the gate, up and down projections.
#[derive(Clone, Debug)]
pub struct ConvolvedLayerKernels {
    pub attention_norm: NativeKernel<readout_features_rows::Entry>,
    pub attention_coefficients: NativeKernel<project_rows::Entry>,
    pub query: NativeKernel<project_rows::Entry>,
    pub key: NativeKernel<project_rows::Entry>,
    pub value: NativeKernel<project_rows::Entry>,
    pub output: NativeKernel<project_rows::Entry>,
    pub feed_forward_norm: NativeKernel<readout_features_rows::Entry>,
    pub feed_forward_coefficients: NativeKernel<project_rows::Entry>,
    pub gate: NativeKernel<project_rows::Entry>,
    pub up: NativeKernel<project_rows::Entry>,
    pub down: NativeKernel<project_rows::Entry>,
}

/// A convolved backbone's entries resolved in draft order.
#[derive(Clone, Debug)]
pub struct AttestedConvolved {
    pub layers: Vec<ConvolvedLayerKernels>,
    pub convolve_input: NativeKernel<draft_convolve_input::Entry>,
    pub convolve_residual: NativeKernel<draft_convolve_residual::Entry>,
    pub gated: NativeKernel<draft_gated_rows::Entry>,
}

/// DFlash2's candidate path resolved: the output norm of the proposing rows
/// the selector projects, its hidden projection and its path entries.
#[derive(Clone, Debug)]
pub struct AttestedSelector {
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub hidden: NativeKernel<project_rows::Entry>,
    pub top_k: NativeKernel<draft_top_k::Entry>,
    pub predecessor: NativeKernel<embedding_rows::Entry>,
    pub successor: NativeKernel<embedding_rows::Entry>,
    pub path: NativeKernel<draft_path_step::Entry>,
}

/// One corrector layer's entries resolved.
#[derive(Clone, Debug)]
pub struct CorrectorLayerKernels {
    pub attention_norm: NativeKernel<readout_features_rows::Entry>,
    pub attention_input: NativeKernel<project_rows::Entry>,
    pub attention_output: NativeKernel<project_rows::Entry>,
    pub attention_residual: NativeKernel<corrector_rows::Entry>,
    pub mlp_norm: NativeKernel<readout_features_rows::Entry>,
    pub up: NativeKernel<project_rows::Entry>,
    pub activated: NativeKernel<corrector_rows::Entry>,
    pub down: NativeKernel<project_rows::Entry>,
    pub down_residual: NativeKernel<corrector_rows::Entry>,
}

/// LiLiCorr's corrector resolved: the output norm of the block's mask rows
/// (its pass rows) and the fusion norm of the anchors' context rows, then
/// every stage of the corrector in graph order.
#[derive(Clone, Debug)]
pub struct AttestedCorrector {
    pub pass_norm: NativeKernel<readout_features_rows::Entry>,
    pub anchor_rows: NativeKernel<readout_features_rows::Entry>,
    pub top_k: NativeKernel<draft_top_k::Entry>,
    pub features: NativeKernel<corrector_features::Entry>,
    pub feature_down: NativeKernel<project_rows::Entry>,
    pub token: Option<NativeKernel<project_rows::Entry>>,
    pub pass: NativeKernel<project_rows::Entry>,
    pub nodes: NativeKernel<corrector_nodes::Entry>,
    pub context: NativeKernel<project_rows::Entry>,
    pub context_bias: NativeKernel<corrector_rows::Entry>,
    pub anchor_norm: NativeKernel<readout_features_rows::Entry>,
    pub attention: NativeKernel<corrector_attention::Entry>,
    pub layers: Vec<CorrectorLayerKernels>,
    pub output_norm: NativeKernel<readout_features_rows::Entry>,
    pub factors: NativeKernel<corrector_factors::Entry>,
    pub factor_input: NativeKernel<project_rows::Entry>,
    pub factor_activated: NativeKernel<corrector_rows::Entry>,
    pub outgoing: NativeKernel<project_rows::Entry>,
    pub incoming: NativeKernel<project_rows::Entry>,
    pub anchor: NativeKernel<project_rows::Entry>,
    pub unit: NativeKernel<corrector_rows::Entry>,
    pub path: NativeKernel<corrector_path::Entry>,
}

/// The weight kind of a corrector map's matrix at `site`.
pub(crate) fn corrector_kind(site: CorrectorSite) -> WeightKind {
    WeightKind::Corrector(CorrectorWeight::Affine {
        site,
        part: AffinePart::Weight,
    })
}

/// The `corrector_rows` specializations a corrector runs (`(N, R, Y)`): its
/// residual joins (attention and down projections), its activated rows (the
/// layers' up projections and the factor input), its biased anchor context
/// and its unit-length heads.
pub(crate) fn corrector_rows_keys(
    binding: &crate::CorrectorBinding,
    activation: Element,
) -> Vec<CorrectorRowsKey> {
    let f32 = Element::f32();
    let keys = [
        (binding.width, 1, f32),
        (binding.mlp, 0, activation),
        (binding.width, 0, f32),
        (binding.width, 0, activation),
        (binding.factor, 0, f32),
    ];
    let mut distinct = Vec::new();
    for key in keys {
        if !distinct.contains(&key) {
            distinct.push(key);
        }
    }
    distinct
}

impl DraftKernels {
    /// Resolve the stage catalog for `plan`'s convolved layers; an error
    /// names the first missing entry.
    pub(crate) fn attest_convolved(
        &self,
        plan: &crate::DraftProgramPlan,
        binding: &crate::ConvolutionBinding,
    ) -> Result<AttestedConvolved, String> {
        let kernels = self.convolved.as_ref().ok_or("convolved draft stages")?;
        let (activation, f32) = (plan.activation(), Element::f32());
        let hidden = plan.hidden();
        let layers = binding
            .layers
            .iter()
            .map(|layer| {
                Ok(ConvolvedLayerKernels {
                    attention_norm: self.norm(layer.attention_norm, hidden)?,
                    attention_coefficients: self.projection(
                        WeightKind::ConvolutionProjection,
                        layer.attention_coefficients,
                        f32,
                    )?,
                    query: self.projection(WeightKind::Query, layer.query, activation)?,
                    key: self.projection(WeightKind::Key, layer.key, activation)?,
                    value: self.projection(WeightKind::Value, layer.value, activation)?,
                    output: self.projection(WeightKind::AttentionOutput, layer.output, f32)?,
                    feed_forward_norm: self.norm(layer.feed_forward_norm, hidden)?,
                    feed_forward_coefficients: self.projection(
                        WeightKind::ConvolutionProjection,
                        layer.feed_forward_coefficients,
                        f32,
                    )?,
                    gate: self.projection(WeightKind::DenseGate, layer.gate, activation)?,
                    up: self.projection(WeightKind::DenseUp, layer.up, activation)?,
                    down: self.projection(WeightKind::DenseDown, layer.down, f32)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(AttestedConvolved {
            layers,
            convolve_input: kernels.convolve_input.clone(),
            convolve_residual: kernels.convolve_residual.clone(),
            gated: kernels.gated.clone(),
        })
    }

    /// Resolve DFlash2's candidate path.
    pub(crate) fn attest_selector(
        &self,
        plan: &crate::DraftProgramPlan,
        binding: crate::SelectorBinding,
    ) -> Result<AttestedSelector, String> {
        let kernels = self.selector.as_ref().ok_or("selector draft stages")?;
        Ok(AttestedSelector {
            features: self.norm(plan.output_norm(), plan.hidden())?,
            hidden: self.projection(
                WeightKind::SelectorHidden,
                ScalableWeight {
                    element: binding.hidden,
                    scale: 0,
                },
                plan.activation(),
            )?,
            top_k: kernels.top_k.clone(),
            predecessor: kernels.predecessor.clone(),
            successor: kernels.successor.clone(),
            path: kernels.path.clone(),
        })
    }

    /// Resolve LiLiCorr's corrector.
    pub(crate) fn attest_corrector(
        &self,
        plan: &crate::DraftProgramPlan,
        binding: &crate::CorrectorBinding,
    ) -> Result<AttestedCorrector, String> {
        let kernels = self.corrector.as_ref().ok_or("corrector draft stages")?;
        let (activation, f32) = (plan.activation(), Element::f32());
        let hidden = plan.hidden();
        let width = binding.width;
        let projection =
            |site, weight| self.projection(corrector_kind(site), weight, f32);
        let rows = |key: CorrectorRowsKey| {
            kernels.rows.get(&key).cloned().ok_or_else(|| {
                format!(
                    "corrector_rows N={} R={} Y={}",
                    key.0,
                    key.1,
                    key.2.name()
                )
            })
        };
        let layers = binding
            .layers
            .iter()
            .map(|layer| {
                Ok(CorrectorLayerKernels {
                    attention_norm: self.norm(layer.attention_norm, width)?,
                    attention_input: projection(
                        CorrectorSite::AttentionInput,
                        layer.attention_input,
                    )?,
                    attention_output: projection(
                        CorrectorSite::AttentionOutput,
                        layer.attention_output,
                    )?,
                    attention_residual: rows((width, 1, f32))?,
                    mlp_norm: self.norm(layer.mlp_norm, width)?,
                    up: projection(CorrectorSite::Up, layer.up)?,
                    activated: rows((binding.mlp, 0, activation))?,
                    down: projection(CorrectorSite::Down, layer.down)?,
                    down_residual: rows((width, 1, f32))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(AttestedCorrector {
            pass_norm: self.norm(plan.output_norm(), hidden)?,
            anchor_rows: self.norm(plan.fusion_norm(), hidden)?,
            top_k: kernels.top_k.clone(),
            features: kernels.features.clone(),
            feature_down: projection(CorrectorSite::FeatureDown, binding.feature_down)?,
            token: binding
                .token
                .map(|token| projection(CorrectorSite::Token, token))
                .transpose()?,
            pass: projection(CorrectorSite::Pass, binding.pass)?,
            nodes: kernels.nodes.clone(),
            context: projection(CorrectorSite::Context, binding.context)?,
            context_bias: rows((width, 0, f32))?,
            anchor_norm: self.norm(binding.anchor_norm, width)?,
            attention: kernels.attention.clone(),
            layers,
            output_norm: self.norm(binding.output_norm, width)?,
            factors: kernels.factors.clone(),
            factor_input: projection(CorrectorSite::FactorInput, binding.factor_input)?,
            factor_activated: rows((width, 0, activation))?,
            outgoing: projection(CorrectorSite::Outgoing, binding.outgoing)?,
            incoming: projection(CorrectorSite::Incoming, binding.incoming)?,
            anchor: projection(CorrectorSite::Anchor, binding.anchor)?,
            unit: rows((binding.factor, 0, f32))?,
            path: kernels.path.clone(),
        })
    }

    fn norm(
        &self,
        element: Element,
        width: u64,
    ) -> Result<NativeKernel<readout_features_rows::Entry>, String> {
        self.norms
            .get(&(element, width))
            .cloned()
            .ok_or_else(|| format!("readout_features_rows NW={} D={width}", element.name()))
    }

    fn projection(
        &self,
        kind: WeightKind,
        weight: ScalableWeight,
        output: Element,
    ) -> Result<NativeKernel<project_rows::Entry>, String> {
        self.projections
            .get(&(kind, weight.element, output))
            .cloned()
            .ok_or_else(|| {
                format!(
                    "project_rows {kind:?} W={} Y={}",
                    weight.element.name(),
                    output.name()
                )
            })
    }
}

/// The unfused stages' distinct projections, each with the scopes that
/// share it, and their distinct norms (element and width), in plan order:
/// the convolved layers', the selector's and the corrector's. Preparation,
/// the planned workspace charge and the attested census all derive these
/// stages' entries from here.
pub(crate) fn stage_entries(
    plan: &crate::DraftProgramPlan,
) -> (Vec<(DraftProjectionKey, Vec<WeightScope>)>, Vec<DraftNormKey>) {
    let (activation, hidden) = (plan.activation(), plan.hidden());
    let f32 = Element::f32();
    let scope = |index: usize, sublayer: u32| {
        WeightScope::DraftSublayer(SublayerIndex {
            block: u32::try_from(index).expect("draft block count fits u32"),
            sublayer,
        })
    };
    let mut projections: Vec<(DraftProjectionKey, Vec<WeightScope>)> = Vec::new();
    let mut add = |kind: WeightKind, weight: ScalableWeight, output: Element, scope: WeightScope| {
        let key = (kind, weight.element, output);
        match projections.iter_mut().find(|(existing, _)| *existing == key) {
            Some((_, scopes)) => scopes.push(scope),
            None => projections.push((key, vec![scope])),
        }
    };
    let mut norms: Vec<DraftNormKey> = Vec::new();
    let mut norm = |key: DraftNormKey| {
        if !norms.contains(&key) {
            norms.push(key);
        }
    };
    if let Some(convolutions) = plan.convolutions() {
        for (index, layer) in convolutions.layers.iter().enumerate() {
            let (mixer, feed_forward) = (scope(index, 0), scope(index, 1));
            norm((layer.attention_norm, hidden));
            norm((layer.feed_forward_norm, hidden));
            add(WeightKind::Query, layer.query, activation, mixer);
            add(WeightKind::Key, layer.key, activation, mixer);
            add(WeightKind::Value, layer.value, activation, mixer);
            add(WeightKind::AttentionOutput, layer.output, f32, mixer);
            add(
                WeightKind::ConvolutionProjection,
                layer.attention_coefficients,
                f32,
                mixer,
            );
            add(
                WeightKind::ConvolutionProjection,
                layer.feed_forward_coefficients,
                f32,
                feed_forward,
            );
            add(WeightKind::DenseGate, layer.gate, activation, feed_forward);
            add(WeightKind::DenseUp, layer.up, activation, feed_forward);
            add(WeightKind::DenseDown, layer.down, f32, feed_forward);
        }
    }
    if let Some(selector) = plan.selector() {
        norm((plan.output_norm(), hidden));
        add(
            WeightKind::SelectorHidden,
            ScalableWeight {
                element: selector.hidden,
                scale: 0,
            },
            activation,
            WeightScope::Draft,
        );
    }
    if let Some(corrector) = plan.corrector() {
        let draft = WeightScope::Draft;
        norm((plan.output_norm(), hidden));
        norm((plan.fusion_norm(), hidden));
        norm((corrector.anchor_norm, corrector.width));
        norm((corrector.output_norm, corrector.width));
        let mut site = |site, weight, scope| add(corrector_kind(site), weight, f32, scope);
        site(CorrectorSite::FeatureDown, corrector.feature_down, draft);
        if let Some(token) = corrector.token {
            site(CorrectorSite::Token, token, draft);
        }
        site(CorrectorSite::Pass, corrector.pass, draft);
        site(CorrectorSite::Context, corrector.context, draft);
        for (index, layer) in corrector.layers.iter().enumerate() {
            let scope = WeightScope::CorrectorLayer(
                u32::try_from(index).expect("corrector layer count fits u32"),
            );
            site(CorrectorSite::AttentionInput, layer.attention_input, scope);
            site(CorrectorSite::AttentionOutput, layer.attention_output, scope);
            site(CorrectorSite::Up, layer.up, scope);
            site(CorrectorSite::Down, layer.down, scope);
        }
        site(CorrectorSite::FactorInput, corrector.factor_input, draft);
        site(CorrectorSite::Outgoing, corrector.outgoing, draft);
        site(CorrectorSite::Incoming, corrector.incoming, draft);
        site(CorrectorSite::Anchor, corrector.anchor, draft);
        for layer in &corrector.layers {
            norm((layer.attention_norm, corrector.width));
            norm((layer.mlp_norm, corrector.width));
        }
    }
    (projections, norms)
}
