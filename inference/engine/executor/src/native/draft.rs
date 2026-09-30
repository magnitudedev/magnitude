use super::{AttentionKernels, DenseKernels};
use crate::{AttentionBinding, DenseBinding};
use magnitude_family_contracts::WeightKind;
use magnitude_kernels::{
    dense_output, draft_confidence, draft_convolve_input, draft_convolve_residual,
    draft_gated_rows, draft_path_step, draft_top_k, embedding_rows, project_rows,
    readout_features_rows, readout_head_rows, sample_rows, shape_rows, widen_rows,
};
use seismic::{Element, NativeKernel};
use std::collections::HashMap;

/// The vocabulary a separate draft's readout projects onto and selects over.
/// Like the MTP head (`draft_vocabulary`), it scores the frequency-ordered
/// leading rows: verification still selects over the whole vocabulary, so
/// this bounds which tokens can be proposed, never which are emitted.
/// DSpark's Markov bias is prepared over the whole vocabulary, so a draft
/// with a Markov chain reads it all.
pub(crate) fn draft_readout_vocabulary(markov: bool, vocabulary: u64) -> u64 {
    if markov {
        vocabulary
    } else {
        super::draft_vocabulary(vocabulary)
    }
}

/// Immutable specializations of a separate draft (DFlash, DSpark, DFlash2), one per
/// distinct binding. Token selection runs over the draft's readout
/// vocabulary (`draft_readout_vocabulary`).
#[derive(Debug, Default)]
pub struct DraftKernels {
    /// The layers' block attention and their context injection (the same
    /// attention with the fusion norm as its input norm).
    pub(super) attention: HashMap<AttentionBinding, AttentionKernels>,
    pub(super) dense: HashMap<DenseBinding, DenseKernels>,
    /// Embeds the block's tokens (raw rows).
    pub(super) embedding: Option<NativeKernel<embedding_rows::Entry>>,
    /// The output norm and the target's vocabulary projection of the
    /// proposing rows.
    pub(super) head: Option<NativeKernel<readout_head_rows::Entry>>,
    /// Token selection over the readout vocabulary.
    pub(super) shape: Option<NativeKernel<shape_rows::Entry>>,
    pub(super) sample: Option<NativeKernel<sample_rows::Entry>>,
    /// Widens a device-conditioned entry's target features to F32.
    pub(super) widen: Option<NativeKernel<widen_rows::Entry>>,
    pub(super) markov: Option<MarkovKernels>,
    pub(super) dflash2: Option<Dflash2Kernels>,
}

/// The unfused projections of DFlash2's block pass: a plain `project_rows`
/// per (weight kind, weight element, published element), prepared once for
/// every draft layer that shares it.
pub type Dflash2Projections = HashMap<Dflash2ProjectionKey, NativeKernel<project_rows::Entry>>;

/// DFlash2's block pass and candidate path: each sublayer's normed rows (the
/// layer norm's `readout_features_rows`, keyed by the norm's element), its
/// two convolution halves, the feed-forward's gated product, and the
/// selector's top-k, codebook gathers (at the selector rank) and ordered
/// path step.
#[derive(Clone, Debug)]
pub struct Dflash2Kernels {
    pub norms: HashMap<Element, NativeKernel<readout_features_rows::Entry>>,
    pub projections: Dflash2Projections,
    pub convolve_input: NativeKernel<draft_convolve_input::Entry>,
    pub convolve_residual: NativeKernel<draft_convolve_residual::Entry>,
    pub gated: NativeKernel<draft_gated_rows::Entry>,
    pub top_k: NativeKernel<draft_top_k::Entry>,
    pub predecessor: NativeKernel<embedding_rows::Entry>,
    pub successor: NativeKernel<embedding_rows::Entry>,
    pub path: NativeKernel<draft_path_step::Entry>,
}

/// One draft layer's entries.
#[derive(Clone, Debug)]
pub struct DraftBlockKernels {
    pub attention: AttentionKernels,
    pub injection: AttentionKernels,
    pub dense: DenseKernels,
}

/// DSpark's chain: the Markov memory of the previous token, its projection
/// onto the slot logits, and the slot's confidence over its output-normed
/// row.
#[derive(Clone, Debug)]
pub struct MarkovKernels {
    pub embedding: NativeKernel<embedding_rows::Entry>,
    pub projection: NativeKernel<dense_output::Entry>,
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub confidence: NativeKernel<draft_confidence::Entry>,
}

/// One DFlash2 draft layer's unfused entries, in graph order: the attention
/// norm, its coefficient projection, the query, key, value and output
/// projections; the feed-forward norm, its coefficient projection, and the
/// gate, up and down projections.
#[derive(Clone, Debug)]
pub struct Dflash2LayerKernels {
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

/// DFlash2's entries resolved in draft order.
#[derive(Clone, Debug)]
pub struct AttestedDflash2 {
    pub layers: Vec<Dflash2LayerKernels>,
    /// The output norm of the proposing rows the selector projects.
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub hidden: NativeKernel<project_rows::Entry>,
    pub convolve_input: NativeKernel<draft_convolve_input::Entry>,
    pub convolve_residual: NativeKernel<draft_convolve_residual::Entry>,
    pub gated: NativeKernel<draft_gated_rows::Entry>,
    pub top_k: NativeKernel<draft_top_k::Entry>,
    pub predecessor: NativeKernel<embedding_rows::Entry>,
    pub successor: NativeKernel<embedding_rows::Entry>,
    pub path: NativeKernel<draft_path_step::Entry>,
}

impl Dflash2Kernels {
    /// Resolve the catalog for `plan`'s layers; an error names the first
    /// missing entry.
    pub(crate) fn attest(
        &self,
        plan: &crate::DraftProgramPlan,
        binding: &crate::Dflash2Binding,
    ) -> Result<AttestedDflash2, String> {
        let activation = plan.activation();
        let f32 = Element::f32();
        let norm = |element: Element| {
            self.norms
                .get(&element)
                .cloned()
                .ok_or_else(|| format!("readout_features_rows NW={}", element.name()))
        };
        let projection = |kind: WeightKind, weight: Element, output: Element| {
            self.projections
                .get(&(kind, weight, output))
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "project_rows {kind:?} W={} Y={}",
                        weight.name(),
                        output.name()
                    )
                })
        };
        let layers = plan
            .blocks()
            .iter()
            .zip(&binding.convolutions)
            .map(|(block, [attention_coefficients, dense_coefficients])| {
                let (attention, dense) = (block.attention, block.feed_forward);
                Ok(Dflash2LayerKernels {
                    attention_norm: norm(attention.norm)?,
                    attention_coefficients: projection(
                        WeightKind::ConvolutionProjection,
                        *attention_coefficients,
                        f32,
                    )?,
                    query: projection(WeightKind::Query, attention.query, activation)?,
                    key: projection(WeightKind::Key, attention.key, activation)?,
                    value: projection(WeightKind::Value, attention.value, activation)?,
                    output: projection(WeightKind::AttentionOutput, attention.output, f32)?,
                    feed_forward_norm: norm(dense.norm)?,
                    feed_forward_coefficients: projection(
                        WeightKind::ConvolutionProjection,
                        *dense_coefficients,
                        f32,
                    )?,
                    gate: projection(WeightKind::DenseGate, dense.gate, activation)?,
                    up: projection(WeightKind::DenseUp, dense.up, activation)?,
                    down: projection(WeightKind::DenseDown, dense.down, f32)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(AttestedDflash2 {
            layers,
            features: norm(plan.output_norm())?,
            hidden: projection(
                WeightKind::SelectorHidden,
                binding.selector.hidden,
                activation,
            )?,
            convolve_input: self.convolve_input.clone(),
            convolve_residual: self.convolve_residual.clone(),
            gated: self.gated.clone(),
            top_k: self.top_k.clone(),
            predecessor: self.predecessor.clone(),
            successor: self.successor.clone(),
            path: self.path.clone(),
        })
    }
}

/// A DFlash2 unfused projection's specialization: its weight kind, weight
/// element and published element.
pub type Dflash2ProjectionKey = (WeightKind, Element, Element);

/// DFlash2's distinct unfused projections, each with the draft scopes that
/// share it, and its distinct norm elements (the layers' and the output
/// norm's), in plan order. Preparation, the planned workspace charge and
/// the attested census all derive DFlash2's entries from here.
pub(crate) fn dflash2_entries(
    plan: &crate::DraftProgramPlan,
    binding: &crate::Dflash2Binding,
) -> (
    Vec<(
        Dflash2ProjectionKey,
        Vec<magnitude_family_contracts::WeightScope>,
    )>,
    Vec<Element>,
) {
    use magnitude_family_contracts::{SublayerIndex, WeightScope};
    let activation = plan.activation();
    let f32 = Element::f32();
    let scope = |index: usize, sublayer: u32| {
        WeightScope::DraftSublayer(SublayerIndex {
            block: u32::try_from(index).expect("draft block count fits u32"),
            sublayer,
        })
    };
    let mut projections: Vec<(Dflash2ProjectionKey, Vec<WeightScope>)> = Vec::new();
    let mut add = |key: Dflash2ProjectionKey, scope: WeightScope| match projections
        .iter_mut()
        .find(|(existing, _)| *existing == key)
    {
        Some((_, scopes)) => scopes.push(scope),
        None => projections.push((key, vec![scope])),
    };
    let mut norms = vec![plan.output_norm()];
    for (index, (block, [attention_coefficients, dense_coefficients])) in
        plan.blocks().iter().zip(&binding.convolutions).enumerate()
    {
        let (mixer, feed_forward) = (scope(index, 0), scope(index, 1));
        let (attention, dense) = (block.attention, block.feed_forward);
        for norm in [attention.norm, dense.norm] {
            if !norms.contains(&norm) {
                norms.push(norm);
            }
        }
        add((WeightKind::Query, attention.query, activation), mixer);
        add((WeightKind::Key, attention.key, activation), mixer);
        add((WeightKind::Value, attention.value, activation), mixer);
        add((WeightKind::AttentionOutput, attention.output, f32), mixer);
        add(
            (
                WeightKind::ConvolutionProjection,
                *attention_coefficients,
                f32,
            ),
            mixer,
        );
        add(
            (WeightKind::ConvolutionProjection, *dense_coefficients, f32),
            feed_forward,
        );
        add(
            (WeightKind::DenseGate, dense.gate, activation),
            feed_forward,
        );
        add((WeightKind::DenseUp, dense.up, activation), feed_forward);
        add((WeightKind::DenseDown, dense.down, f32), feed_forward);
    }
    add(
        (
            WeightKind::SelectorHidden,
            binding.selector.hidden,
            activation,
        ),
        WeightScope::Draft,
    );
    (projections, norms)
}
