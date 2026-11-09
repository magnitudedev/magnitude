//! Softmax attention over history spans (`Operator::Attention`): the forms
//! the attention entries implement, their static axes, weight roles and
//! program binding.

pub(crate) mod graph;

use super::{PlanError, WeightPush, tail};
use crate::{
    AttentionBinding, AttentionShape, AttentionValues, GroupedOutputBinding, GroupedOutputShape,
    LowRankQueryBinding, ScalableWeight, SelectionBinding, SelectionShape,
};
use magnitude_family_contracts::{
    Attention, AttentionGate, Decoder, GateFunction, GateGranularity, HeadNorm, HistoryDomain,
    HistoryReads, KeyValue, MediaRowAttention, Operator, OutputForm, OutputProjection,
    OutputRotary, QueryProjection, Rotary, RotaryLayout, RotaryPair, RotaryPairing,
    RotaryPlacement, ValueNorm, ValueSource, WeightKind,
};
use magnitude_state::{CodecSpec, HistoryEncoding, HistoryValue, KvCodec};
use seismic::Element;

/// The attention forms the attention entries implement: a full query, or a
/// low-rank one composed of two projections (`graph::attention`), a full or
/// grouped low-rank output projection (its groups' products by
/// `project_rows`, then the output projection over them), any output gate, a
/// query norm and a key norm each present or absent (`NQ`, `N`), values
/// projected, from the raw key, or the stored key row (`VK`, dense
/// histories only), an optional unweighted value norm, one norm epsilon,
/// owned histories in a token or window domain or a Shared layer's source
/// history, and any rotary table in half-split or adjacent pairs over the
/// leading or trailing dimensions (`RA`, `RT`) whose stored divisors
/// (checked at load, `check_rotary_divisors`) cover its pairs, with
/// the attended row of an ungated layer optionally turned back by the
/// inverse rotation (`RO`) ([`shape`] names each form it has no axis for).
/// Media rows attend
/// causally or bidirectionally (`operators::media_rows_bidirectional`). A
/// history is read whole or selected by an indexer (`HistoryReads`), whose
/// rotary is half-split leading.
pub(crate) fn admit(attention: &Attention, input_epsilon: f64) -> Result<(), PlanError> {
    // A Shared layer reads its source's history (`HistoryReads::read`)
    // and projects no keys or values; its source's domain is admitted with
    // the source.
    if let KeyValue::Owned { domain, .. } = &attention.key_value {
        if !matches!(domain, HistoryDomain::Token | HistoryDomain::Window { .. }) {
            return Err(PlanError::Unsupported("attention history domain"));
        }
    }
    head_norm_epsilon(attention, input_epsilon)?;
    if let AttentionGate::Interleaved {
        function: GateFunction::Silu,
    }
    | AttentionGate::Separate {
        function: GateFunction::Silu,
        ..
    } = &attention.gate
    {
        return Err(PlanError::Unsupported("SiLU attention gate"));
    }
    if let Rotary::Table {
        pairs,
        divisors: Some(divisors),
        ..
    } = &attention.rotary
    {
        if divisors.bases.len() != pairs.len() {
            return Err(PlanError::Topology(
                "rotary divisor bases disagree with the table's pairs",
            ));
        }
    }
    // A selected read's indexer orders keys by position (`indexer_select`):
    // its rows attend causally, and its norms share one epsilon.
    if let HistoryReads::Selected { indexer, .. } = &attention.reads {
        if attention.media_rows != MediaRowAttention::Causal {
            return Err(PlanError::Unsupported(
                "selected attention attending media rows bidirectionally",
            ));
        }
        if indexer.query_norm.epsilon != indexer.key_norm.epsilon {
            return Err(PlanError::Unsupported(
                "indexer norms with differing epsilons",
            ));
        }
        // The indexer rotates its heads by the attention's table in
        // half-split leading pairs (`indexer_select`).
        if matches!(
            &attention.rotary,
            Rotary::Table { layout, .. } if *layout != RotaryLayout::HALF_SPLIT_LEADING
        ) {
            return Err(PlanError::Unsupported(
                "selected attention with rotary pairs other than half-split leading",
            ));
        }
    }
    let shape = shape(0, attention)?;
    if shape.width < 2 * shape.rotary_pairs {
        return Err(PlanError::Topology("rotary pairs exceed the head width"));
    }
    // The entries turn the published attended row back, which a gate
    // would have already scaled (`attention.seismic`'s `output_rotary`).
    if shape.output_rotary > 0 && (shape.interleaved_gate > 0 || shape.separate_gate > 0) {
        return Err(PlanError::Unsupported(
            "gated attention with an inverse output rotary",
        ));
    }
    Ok(())
}

/// How the decoder's attention layers attend media rows, per history
/// domain: a launch row reads one fresh span per history read, so the owned
/// layers of one domain attend media rows alike, and a Shared layer like its
/// source (Gemma: bidirectional on its window layers, causal on its full
/// ones). Without a vision component there are no media rows to attend.
pub(crate) fn admit_media(decoder: &Decoder, vision: bool) -> Result<(), PlanError> {
    if !vision {
        return Ok(());
    }
    let mut domains: Vec<(&HistoryDomain, MediaRowAttention)> = Vec::new();
    for (index, sublayer) in decoder.sublayers() {
        let Operator::Attention(attention) = &sublayer.op else {
            continue;
        };
        let differ = || {
            PlanError::Unsupported(
                "attention layers of one history attending media rows differently",
            )
        };
        match &attention.key_value {
            KeyValue::Owned { domain, .. } => {
                match domains.iter().find(|(other, _)| *other == domain) {
                    Some((_, form)) if *form != attention.media_rows => return Err(differ()),
                    Some(_) => {}
                    None => domains.push((domain, attention.media_rows)),
                }
            }
            KeyValue::Shared { source } => {
                let Some(Operator::Attention(owner)) = decoder
                    .sublayers()
                    .find(|(candidate, _)| candidate == source)
                    .map(|(_, sublayer)| &sublayer.op)
                else {
                    return Err(PlanError::Topology(
                        "a Shared layer's source is not attention",
                    ));
                };
                if owner.media_rows != attention.media_rows || *source >= index {
                    return Err(differ());
                }
            }
        }
    }
    Ok(())
}

/// Check stored per-pair rotary divisors (`rope_freqs`) against the table the
/// family derived from headers: pair `p`'s frequency is `bases[p] / d_p`,
/// rounded to f32 where nonzero, and at most 1e-20 where the table leaves the
/// pair unrotated. The artifact is rejected otherwise.
pub(crate) fn check_rotary_divisors(
    pairs: &[RotaryPair],
    bases: &[f64],
    stored: &[f32],
) -> Result<(), String> {
    if stored.len() != pairs.len() || bases.len() != pairs.len() {
        return Err(format!(
            "{} stored rotary divisors for {} pairs",
            stored.len(),
            pairs.len()
        ));
    }
    for (index, ((pair, base), divisor)) in pairs.iter().zip(bases).zip(stored).enumerate() {
        let frequency = base / f64::from(*divisor);
        let matches = if pair.frequency == 0.0 {
            frequency <= 1e-20
        } else {
            frequency as f32 == pair.frequency as f32
        };
        if !matches {
            return Err(format!(
                "rotary pair {index}: the stored divisor {divisor} gives frequency {frequency}, \
                 the table {}",
                pair.frequency
            ));
        }
    }
    Ok(())
}

/// The attention entries' static axes of an attention operator over a
/// `hidden`-wide residual.
pub(crate) fn shape(hidden: u64, attention: &Attention) -> Result<AttentionShape, PlanError> {
    if attention.kv_heads == 0 || attention.heads % attention.kv_heads != 0 {
        return Err(PlanError::Topology(
            "attention heads are not a multiple of its key/value heads",
        ));
    }
    let width = attention.width;
    let (interleaved_gate, separate_gate) = match &attention.gate {
        AttentionGate::None => (0, 0),
        AttentionGate::Interleaved { .. } => (width, 0),
        AttentionGate::Separate { granularity, .. } => match granularity {
            GateGranularity::Element => (0, width),
            GateGranularity::Head => (0, 1),
        },
    };
    // The rotary layout and output rotary (`RA`, `RT`, `RO`).
    let (rotary_adjacent, rotary_trailing, output_rotary) = match &attention.rotary {
        Rotary::None | Rotary::Interleaved { .. } => (0, 0, 0),
        // The inverse of no rotation is none.
        Rotary::Table {
            pairs,
            layout,
            output,
            ..
        } => (
            u64::from(layout.pairs == RotaryPairing::Adjacent),
            u64::from(layout.placement == RotaryPlacement::Trailing),
            u64::from(*output == OutputRotary::Inverse && !pairs.is_empty()),
        ),
    };
    let query_norm = u64::from(matches!(attention.query_norm, HeadNorm::Rms(_)));
    let (fresh, key_norm, value_norm, values) = match &attention.key_value {
        KeyValue::Owned {
            key_norm,
            value,
            value_norm,
            ..
        } => (
            1,
            u64::from(matches!(key_norm, HeadNorm::Rms(_))),
            u64::from(matches!(value_norm, ValueNorm::RmsUnweighted(_))),
            match value {
                ValueSource::Projected(_) => AttentionValues::Projected,
                ValueSource::RawKey => AttentionValues::RawKey,
                ValueSource::StoredKey => AttentionValues::StoredKey,
            },
        ),
        // A Shared layer projects no keys: its unread `N` is its `NQ`.
        KeyValue::Shared { .. } => (0, query_norm, 0, AttentionValues::Projected),
    };
    Ok(AttentionShape {
        hidden,
        kv_heads: attention.kv_heads,
        group: attention.heads / attention.kv_heads,
        rotary_pairs: match &attention.rotary {
            Rotary::None => 0,
            Rotary::Interleaved { width, .. } => width / 2,
            Rotary::Table { pairs, .. } => pairs.len() as u64,
        },
        width,
        interleaved_gate,
        separate_gate,
        fresh,
        query_norm,
        key_norm,
        value_norm,
        values,
        sinks: u64::from(attention.sinks.is_some()),
        rotary_adjacent,
        rotary_trailing,
        output_rotary,
    })
}

/// The attention entries' `gate_function` code: 0 sigmoid, 1 softplus (and
/// 0 for an ungated operator, whose gate is never read). [`admit`] rejects
/// every other gate.
pub(crate) fn gate_function(attention: &Attention) -> i32 {
    let function = match &attention.gate {
        AttentionGate::None => return 0,
        AttentionGate::Interleaved { function } | AttentionGate::Separate { function, .. } => {
            function
        }
    };
    match function {
        GateFunction::Sigmoid => 0,
        GateFunction::Softplus => 1,
        GateFunction::Silu => unreachable!("admission rejects SiLU attention gates"),
    }
}

/// The one epsilon of an attention operator's query, key and value norms,
/// which the attention entries take as one argument. An operator without
/// head norms uses its input norm's (never read).
pub(crate) fn head_norm_epsilon(
    attention: &Attention,
    input_epsilon: f64,
) -> Result<f64, PlanError> {
    let mut epsilons = Vec::new();
    if let HeadNorm::Rms(norm) = &attention.query_norm {
        epsilons.push(norm.epsilon);
    }
    if let KeyValue::Owned {
        key_norm,
        value_norm,
        ..
    } = &attention.key_value
    {
        if let HeadNorm::Rms(norm) = key_norm {
            epsilons.push(norm.epsilon);
        }
        if let ValueNorm::RmsUnweighted(norm) = value_norm {
            epsilons.push(norm.epsilon);
        }
    }
    match epsilons.split_first() {
        None => Ok(input_epsilon),
        Some((first, rest)) if rest.iter().all(|epsilon| epsilon == first) => Ok(*first),
        Some(_) => Err(PlanError::Unsupported(
            "attention head norms with differing epsilons",
        )),
    }
}

/// The weight kind of an attention operator's query projection: fused with
/// the gate rows when the gate is interleaved.
pub(crate) fn query_kind(attention: &Attention) -> WeightKind {
    match attention.gate {
        AttentionGate::Interleaved { .. } => WeightKind::QueryGate,
        AttentionGate::None | AttentionGate::Separate { .. } => WeightKind::Query,
    }
}

/// The weight kind the attention projection's query segment reads: the
/// query's ([`query_kind`]), or a low-rank query's `down`.
pub(crate) fn query_segment_kind(attention: &Attention) -> WeightKind {
    match &attention.query {
        QueryProjection::Full(_) => query_kind(attention),
        QueryProjection::LowRank { .. } => WeightKind::QueryDown,
    }
}

/// A low-rank query's rank and the epsilon of its `qr` norm; none for a
/// full query. The attention projection's query segment projects `rank`
/// rows, which the up projection normalizes with that epsilon.
pub(crate) fn low_rank_query(attention: &Attention) -> Option<(u64, f64)> {
    match &attention.query {
        QueryProjection::Full(_) => None,
        // `Decoder::validate` holds `down` to `[rank, hidden]`.
        QueryProjection::LowRank { down, norm, .. } => Some((down.shape[0], norm.epsilon)),
    }
}

/// The weights the operator binds, in the order its entries consume them.
pub(super) fn weights<'a>(attention: &'a Attention, push: &mut WeightPush<'_, 'a>) {
    match &attention.query {
        QueryProjection::Full(query) => push(query_kind(attention), query),
        QueryProjection::LowRank { down, norm, up } => {
            push(WeightKind::QueryDown, down);
            push(WeightKind::QueryDownNorm, &norm.weight);
            push(query_kind(attention), up);
        }
    }
    if let AttentionGate::Separate { weight, .. } = &attention.gate {
        push(WeightKind::AttentionGate, weight);
    }
    if let KeyValue::Owned { key, value, .. } = &attention.key_value {
        push(WeightKind::Key, key);
        if let ValueSource::Projected(value) = value {
            push(WeightKind::Value, value);
        }
    }
    if let HeadNorm::Rms(norm) = &attention.query_norm {
        push(WeightKind::QueryNorm, &norm.weight);
    }
    if let Rotary::Table {
        divisors: Some(divisors),
        ..
    } = &attention.rotary
    {
        push(WeightKind::RotaryDivisors, &divisors.weight);
    }
    if let KeyValue::Owned {
        key_norm: HeadNorm::Rms(norm),
        ..
    } = &attention.key_value
    {
        push(WeightKind::KeyNorm, &norm.weight);
    }
    if let HistoryReads::Selected { indexer, .. } = &attention.reads {
        push(WeightKind::IndexerQuery, &indexer.query);
        push(WeightKind::IndexerKey, &indexer.key);
        push(WeightKind::IndexerQueryNorm, &indexer.query_norm.weight);
        push(WeightKind::IndexerKeyNorm, &indexer.key_norm.weight);
    }
    if let Some(sinks) = &attention.sinks {
        push(WeightKind::AttentionSinks, sinks);
    }
    match &attention.output {
        OutputProjection::Full(output) => push(WeightKind::AttentionOutput, output),
        OutputProjection::GroupedLowRank { a, b, .. } => {
            push(WeightKind::AttentionOutputGrouped, a);
            push(WeightKind::AttentionOutput, b);
        }
    }
}

/// `project_rows`' axes of a grouped output's first product
/// (`OutputProjection::GroupedLowRank`'s block-diagonal `a`): each of its
/// groups projects a consecutive run of the attended row to `rank`; none
/// for a full output projection.
pub(crate) fn grouped_output(attention: &Attention) -> Option<GroupedOutputShape> {
    match &attention.output {
        OutputProjection::Full(_) => None,
        OutputProjection::GroupedLowRank { groups, rank, .. } => Some(GroupedOutputShape {
            groups: *groups,
            inputs: attention.heads * attention.width / groups,
            rank: *rank,
        }),
    }
}

/// The static axes of a selected read in a decoder serving `context`
/// tokens; none for a read of every visible row.
pub(crate) fn selection_shape(reads: &HistoryReads, context: u64) -> Option<SelectionShape> {
    match reads {
        HistoryReads::Visible => None,
        HistoryReads::Selected {
            block,
            blocks,
            indexer,
        } => Some(SelectionShape {
            heads: indexer.heads,
            width: indexer.width,
            block: *block,
            blocks: *blocks,
            // A row at position t < context reads blocks up to its tail's,
            // t / block: the bitmap covers them all.
            words: context / block / 32 + 1,
        }),
    }
}

/// A selected read's numerical parameters apart from its weights: block,
/// blocks, indexer heads, width and norm epsilon.
fn selection_key(reads: &HistoryReads) -> Option<(u64, u64, u64, u64, f64)> {
    match reads {
        HistoryReads::Visible => None,
        HistoryReads::Selected {
            block,
            blocks,
            indexer,
        } => Some((
            *block,
            *blocks,
            indexer.heads,
            indexer.width,
            indexer.query_norm.epsilon,
        )),
    }
}

/// Every numerical parameter of the operator a sealed graph holds apart from
/// its weights.
pub(super) fn shape_key(attention: &Attention, epsilon: f64) -> Result<String, PlanError> {
    // A full query or output adds nothing, so every other form's key is
    // unchanged.
    let low_rank = low_rank_query(attention)
        .map(|(rank, epsilon)| format!(" low-rank query {rank} {epsilon:?}"))
        .unwrap_or_default();
    let grouped = grouped_output(attention)
        .map(|grouped| format!(" grouped output {grouped:?}"))
        .unwrap_or_default();
    Ok(format!(
        "attention {:?} {:?} {:?} {:?} {} {:?}{low_rank}{grouped}",
        shape(0, attention)?,
        gate_function(attention),
        head_norm_epsilon(attention, epsilon)?,
        attention.rotary,
        attention.scale,
        selection_key(&attention.reads),
    ))
}

/// The program binding of an attention sublayer with `output` in a decoder
/// serving `context` tokens; `lookup` resolves the planned element of a role
/// in its scope. An absent projection segment binds a zero-row view of the
/// query weight.
pub(super) fn binding(
    attention: &Attention,
    output: &OutputForm,
    hidden: u64,
    context: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightKind) -> Result<ScalableWeight, PlanError>,
    activation: Element,
    history: KvCodec,
) -> Result<AttentionBinding, PlanError> {
    let shape = shape(hidden, attention)?;
    // Every projection binds its second-level scale at its accumulator-scale
    // port (`attention_project`'s QS, GS, KS, VS, `attention_output`'s OS).
    let projection = |kind| scalable(kind).map(|weight| weight.element);
    let query = projection(query_kind(attention))?;
    let segment = |present: bool, kind| if present { projection(kind) } else { Ok(query) };
    let selection = selection_shape(&attention.reads, context)
        .map(|shape| {
            Ok::<_, PlanError>(SelectionBinding {
                shape,
                query: projection(WeightKind::IndexerQuery)?,
                key: projection(WeightKind::IndexerKey)?,
            })
        })
        .transpose()?;
    let low_rank_query = low_rank_query(attention)
        .map(|(rank, _)| {
            Ok::<_, PlanError>(LowRankQueryBinding {
                rank,
                down: projection(WeightKind::QueryDown)?,
                norm: lookup(WeightKind::QueryDownNorm)?,
            })
        })
        .transpose()?;
    Ok(AttentionBinding {
        selection,
        key_value_only: false,
        shape,
        norm: lookup(WeightKind::InputNorm)?,
        query,
        low_rank_query,
        gate: segment(shape.gate_rows() > 0, WeightKind::AttentionGate)?,
        key: segment(shape.key_rows() > 0, WeightKind::Key)?,
        value: segment(shape.value_rows() > 0, WeightKind::Value)?,
        output: projection(WeightKind::AttentionOutput)?,
        grouped: grouped_output(attention)
            .map(|shape| {
                Ok::<_, PlanError>(GroupedOutputBinding {
                    shape,
                    element: projection(WeightKind::AttentionOutputGrouped)?,
                })
            })
            .transpose()?,
        activation,
        history: history_codec(attention, history, activation)?,
        tail: tail(output, &lookup)?,
    })
}

/// The history codec `attention` reads under the storage policy `history`
/// (`magnitude_state::history_codec`, which state allocation shares). The
/// K8/V4 entries hold no stored-key form (`VK = 0`).
fn history_codec(
    attention: &Attention,
    history: KvCodec,
    activation: Element,
) -> Result<CodecSpec, PlanError> {
    let codec = magnitude_state::history_codec(
        attention,
        history,
        activation
            .dtype()
            .ok_or(PlanError::Unsupported("attention activation dtype"))?,
    )
    .map_err(PlanError::InvalidDefinition)?;
    if codec.value == HistoryValue::Key && codec.encoding() != HistoryEncoding::Dense {
        return Err(PlanError::Unsupported(
            "stored-key values in a K8/V4 history",
        ));
    }
    Ok(codec)
}

/// The executor admits no Shared layer whose source's values are its stored
/// keys: a Shared layer's history codec is its own-value form
/// (`magnitude_state::history_codec`).
pub(crate) fn admit_shared(decoder: &Decoder) -> Result<(), PlanError> {
    for (_, sublayer) in decoder.sublayers() {
        let Operator::Attention(attention) = &sublayer.op else {
            continue;
        };
        let KeyValue::Shared { source } = &attention.key_value else {
            continue;
        };
        let stored = decoder
            .sublayers()
            .find(|(candidate, _)| candidate == source)
            .is_some_and(|(_, sublayer)| {
                matches!(
                    &sublayer.op,
                    Operator::Attention(owner) if matches!(
                        owner.key_value,
                        KeyValue::Owned { value: ValueSource::StoredKey, .. }
                    )
                )
            });
        if stored {
            return Err(PlanError::Unsupported(
                "a Shared attention layer reading stored-key values",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(frequency: f64) -> RotaryPair {
        RotaryPair {
            frequency,
            amplitude: 1.0,
        }
    }

    mod selected {
        use crate::programs::native_target_graph::checked_target_family_storage;
        use crate::{
            ComponentSelection, ExecutionPath, MixerProgramSlot, ModelLoadPlan, PlannedMethod,
            ResourceCapacity, ResourceLimits, ResourcePlanner, SelectionShape, resident_layout,
            source_element,
        };
        use magnitude_artifacts::{
            ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
            gguf::{Encoding, TensorDescriptor},
        };
        use magnitude_family_contracts::{
            ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder,
            DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm, FamilyId, FeedForwardUp,
            HeadNorm, HistoryDomain, HistoryReads, Indexer, InputNorm, InputSemantics, KeyValue,
            MediaRowAttention, ModelDefinition, Operator, OutputForm, ResidualForm, RmsNorm,
            OutputProjection, OutputRotary, QueryProjection, Rotary, RotaryLayout, RotaryPair,
            Sublayer, ValueNorm, ValueSource, WeightDescriptor,
        };
        use magnitude_state::{HistoryDomainKind, KvCodec, LayerRef};
        use seismic::BackendName;

        const HIDDEN: u64 = 256;
        const CONTEXT: u64 = 4096;
        const LOW_RANK: u64 = 64;

        /// A decoder of two blocks, each a selected attention (64-wide heads,
        /// fully rotated; an indexer of two 64-wide heads over blocks of four
        /// rows, two selected) and a dense feed-forward. With `low_rank`,
        /// each attention's query is low-rank, of rank 64. A `grouped` output
        /// projects the attended row through two groups of rank 32 (`a`
        /// `[64, 128]`), then `b` `[HIDDEN, 64]`.
        fn selected(low_rank: bool, grouped: bool) -> (ModelDefinition, PackageManifest) {
            let mut tensors = Vec::new();
            let mut offset = 0;
            let mut tensor = |name: String, shape: Vec<u64>| {
                let encoding = if shape.len() > 1 {
                    Encoding::BF16
                } else {
                    Encoding::F32
                };
                let nbytes = source_element(encoding, *shape.last().unwrap())
                    .unwrap()
                    .canonical_byte_len(&shape)
                    .unwrap();
                tensors.push(TensorDescriptor {
                    name: name.clone(),
                    shape: shape.clone(),
                    encoding,
                    offset,
                    nbytes,
                });
                offset += nbytes;
                WeightDescriptor::stored(name, shape)
            };
            let rms = |weight| RmsNorm {
                weight,
                epsilon: 1e-6,
            };
            let mut block = |layer: usize| {
                let name = |suffix: &str| format!("blk.{layer}.{suffix}");
                let query = if low_rank {
                    QueryProjection::LowRank {
                        down: tensor(name("attn_q_a"), vec![LOW_RANK, HIDDEN]),
                        norm: RmsNorm {
                            weight: tensor(name("attn_q_a_norm"), vec![LOW_RANK]),
                            epsilon: 1e-20,
                        },
                        up: tensor(name("attn_q_b"), vec![HIDDEN, LOW_RANK]),
                    }
                } else {
                    QueryProjection::Full(tensor(name("attn_q"), vec![HIDDEN, HIDDEN]))
                };
                let attention = Sublayer {
                    input: InputNorm::Rms(rms(tensor(name("attn_norm"), vec![HIDDEN]))),
                    op: Operator::Attention(Box::new(Attention {
                        heads: 4,
                        kv_heads: 2,
                        width: 64,
                        query,
                        gate: AttentionGate::None,
                        query_norm: HeadNorm::Rms(rms(tensor(name("attn_q_norm"), vec![64]))),
                        key_value: KeyValue::Owned {
                            key: tensor(name("attn_k"), vec![128, HIDDEN]),
                            value: ValueSource::Projected(tensor(
                                name("attn_v"),
                                vec![128, HIDDEN],
                            )),
                            key_norm: HeadNorm::Rms(rms(tensor(name("attn_k_norm"), vec![64]))),
                            value_norm: ValueNorm::None,
                            domain: HistoryDomain::Token,
                        },
                        rotary: Rotary::Table {
                            pairs: (0..32)
                                .map(|pair| RotaryPair {
                                    frequency: 1.0e4f64.powf(-(2 * pair) as f64 / 64.0),
                                    amplitude: 1.0,
                                })
                                .collect(),
                            layout: RotaryLayout::HALF_SPLIT_LEADING,
                            divisors: None,
                            output: OutputRotary::None,
                        },
                        scale: 1.0 / 8.0,
                        reads: HistoryReads::Selected {
                            block: 4,
                            blocks: 2,
                            indexer: Indexer {
                                heads: 2,
                                width: 64,
                                query: tensor(name("indexer.q_proj"), vec![128, HIDDEN]),
                                key: tensor(name("indexer.k_proj"), vec![64, HIDDEN]),
                                query_norm: rms(tensor(name("indexer.q_norm"), vec![64])),
                                key_norm: rms(tensor(name("indexer.k_norm"), vec![64])),
                            },
                        },
                        media_rows: MediaRowAttention::Causal,
                        sinks: None,
                        output: if grouped {
                            OutputProjection::GroupedLowRank {
                                groups: 2,
                                rank: 32,
                                a: tensor(name("attn_output_a"), vec![64, 128]),
                                b: tensor(name("attn_output_b"), vec![HIDDEN, 64]),
                            }
                        } else {
                            OutputProjection::Full(tensor(
                                name("attn_output"),
                                vec![HIDDEN, HIDDEN],
                            ))
                        },
                    })),
                    output: OutputForm::Residual,
                };
                let dense = Sublayer {
                    input: InputNorm::Rms(rms(tensor(name("ffn_norm"), vec![HIDDEN]))),
                    op: Operator::DenseFfn(Box::new(DenseFfn {
                        intermediate: 512,
                        up: FeedForwardUp::Gated {
                            activation: ActivationFunction::Silu,
                            limit: None,
                            gate: tensor(name("ffn_gate"), vec![512, HIDDEN]),
                            up: tensor(name("ffn_up"), vec![512, HIDDEN]),
                        },
                        down: tensor(name("ffn_down"), vec![HIDDEN, 512]),
                    })),
                    output: OutputForm::Residual,
                };
                Block {
                    sublayers: vec![attention, dense],
                }
            };
            let blocks = vec![block(0), block(1)];
            let embedding = tensor("token_embd".into(), vec![512, HIDDEN]);
            let output_norm = tensor("output_norm".into(), vec![HIDDEN]);
            let identity = PackageIdentity {
                target: ArtifactIdentity([7; 32]),
                projector: None,
            };
            let definition = ModelDefinition {
                family: FamilyId("selected-attention".into()),
                artifact_identity: identity,
                inputs: InputSemantics { coordinate_axes: 1 },
                decoder: Decoder {
                    activation_dtype: ActivationDType::BF16,
                    hidden: HIDDEN,
                    vocabulary: 512,
                    context_limit: CONTEXT,
                    residual: ResidualForm::Single,
                    entry: EntryForm {
                        embedding: embedding.clone(),
                        scale: EmbeddingScale::Unit,
                        norm: None,
                        per_layer: None,
                    },
                    blocks,
                    exit: ExitForm {
                        norm: ExitNorm::Rms(rms(output_norm)),
                        output: embedding,
                        softcap: None,
                    },
                },
                head: None,
                vision: None,
                draft: None,
            };
            let manifest = PackageManifest {
                identity,
                target: ComponentManifest {
                    files: vec![ComponentFile {
                        path: "selected.gguf".into(),
                        size: offset,
                    }],
                    identity: identity.target,
                    tensors,
                },
                projector: None,
                draft: None,
            };
            (definition, manifest)
        }

        fn limits() -> ResourceLimits {
            ResourceLimits {
                max_launch_rows: 64,
                max_launch_slots: 8,
                max_selected_rows: 64,
                max_drafting_slots: 8,
                exported_logits_rows: 0,
                max_images_per_request: 1,
                max_image_cells: 0,
                lookahead: false,
                chain_capacity: std::num::NonZeroUsize::MIN,
                conversations: std::num::NonZeroUsize::MIN,
            }
        }

        /// A grouped low-rank output plans as its grouped `a` product
        /// (`project_rows` over two groups of 128 attended columns, rank 32)
        /// then `b` as the output projection over those products, and its
        /// graphs certify on the CPU and Metal at every row class. CUDA's
        /// and Vulkan's `project_rows` take one group: their graphs refuse
        /// the call by its entry and statics.
        #[test]
        fn grouped_output_plans_its_grouped_product_and_certifies_where_natives_take_groups() {
            let (definition, manifest) = selected(false, true);
            definition.validate().unwrap();
            let limits = limits();
            for backend in BackendName::ALL {
                let codec = KvCodec::Dense;
                let load = ModelLoadPlan::derive(
                    &manifest,
                    &definition,
                    ComponentSelection {
                        head: false,
                        vision: false,
                    },
                    resident_layout(ExecutionPath::Native, backend),
                    crate::PlannedMethod::Plain,
                    crate::planning::fixtures::LIMITS,
                )
                .unwrap();
                let plan = load.program_plan(&definition, codec).unwrap();
                let MixerProgramSlot::Attention(binding) = plan.target().blocks()[0].mixer() else {
                    panic!("an attention block");
                };
                assert_eq!(
                    binding.grouped.map(|grouped| grouped.shape),
                    Some(crate::GroupedOutputShape {
                        groups: 2,
                        inputs: 128,
                        rank: 32,
                    })
                );
                assert_eq!(binding.output_input(), (2, 32));
                let state = ResourcePlanner::state_plan(
                    &definition,
                    &load,
                    PlannedMethod::Plain,
                    codec,
                    limits,
                    ResourceCapacity {
                        domain_bytes: 16 * 1024 * 1024 * 1024,
                        tensor_operations: crate::TensorOperations::Absent,
                    },
                )
                .unwrap();
                let certified = checked_target_family_storage(
                    backend,
                    &load,
                    &definition.decoder,
                    &state,
                    plan.target(),
                    limits,
                );
                match backend {
                    BackendName::Cpu | BackendName::Metal => {
                        certified.unwrap_or_else(|error| panic!("{backend:?}: {error}"));
                    }
                    BackendName::Cuda | BackendName::Vulkan => match certified {
                        Err(crate::programs::graph::GraphError::KernelDomain(violation)) => {
                            assert_eq!(violation.entry, "project_rows");
                            assert!(violation.statics.contains(&("G".to_owned(), 2)));
                        }
                        Err(error) => panic!("{backend:?}: {error}"),
                        Ok(_) => panic!("{backend:?} certified a grouped output"),
                    },
                }
            }
        }

        /// `selected(false, false)` with V4.1's window history: each layer
        /// reads every visible row of a 128-token window, its values the
        /// stored key row.
        fn stored_key_window() -> (ModelDefinition, PackageManifest) {
            let (mut definition, manifest) = selected(false, false);
            for block in &mut definition.decoder.blocks {
                let Operator::Attention(attention) = &mut block.sublayers[0].op else {
                    unreachable!("each block opens with attention")
                };
                attention.reads = HistoryReads::Visible;
                let KeyValue::Owned { value, domain, .. } = &mut attention.key_value else {
                    unreachable!("each layer owns its history")
                };
                *value = ValueSource::StoredKey;
                *domain = HistoryDomain::Window { tokens: 128 };
            }
            definition.validate().unwrap();
            (definition, manifest)
        }

        /// A stored-key window layer's state holds the key plane alone (two
        /// kv heads of 64 bf16 values: 256 bytes a row, half of K + V), and
        /// its graphs bind that one plane and certify on the CPU and Metal at
        /// every row class. CUDA's and Vulkan's attention entries refuse
        /// stored-key values (`VK`) by declaration.
        #[test]
        fn stored_key_history_plans_one_plane_and_certifies_where_natives_read_it() {
            let (definition, manifest) = stored_key_window();
            let limits = limits();
            for backend in BackendName::ALL {
                let codec = KvCodec::Dense;
                let load = ModelLoadPlan::derive(
                    &manifest,
                    &definition,
                    ComponentSelection {
                        head: false,
                        vision: false,
                    },
                    resident_layout(ExecutionPath::Native, backend),
                    crate::PlannedMethod::Plain,
                    crate::planning::fixtures::LIMITS,
                )
                .unwrap();
                let plan = load.program_plan(&definition, codec).unwrap();
                let MixerProgramSlot::Attention(binding) = plan.target().blocks()[0].mixer() else {
                    panic!("an attention block");
                };
                assert_eq!(binding.shape.stored_key(), 1);
                assert_eq!(binding.shape.value_rows(), 0);
                let state = ResourcePlanner::state_plan(
                    &definition,
                    &load,
                    PlannedMethod::Plain,
                    codec,
                    limits,
                    ResourceCapacity {
                        domain_bytes: 16 * 1024 * 1024 * 1024,
                        tensor_operations: crate::TensorOperations::Absent,
                    },
                )
                .unwrap();
                for domain in &state.target_state().domains {
                    for component in &domain.components {
                        assert_eq!(component.planes().len(), 1, "{:?}", component.layer);
                        assert_eq!(component.row_bytes().unwrap(), 2 * 64 * 2);
                    }
                }
                let certified = checked_target_family_storage(
                    backend,
                    &load,
                    &definition.decoder,
                    &state,
                    plan.target(),
                    limits,
                );
                match backend {
                    BackendName::Cpu | BackendName::Metal => {
                        certified.unwrap_or_else(|error| panic!("{backend:?}: {error}"));
                    }
                    BackendName::Cuda | BackendName::Vulkan => match certified {
                        Err(crate::programs::graph::GraphError::KernelDomain(violation)) => {
                            assert!(
                                violation.entry.starts_with("attention_"),
                                "{backend:?}: {}",
                                violation.entry
                            );
                            assert!(violation.statics.contains(&("VK".to_owned(), 1)));
                        }
                        Err(error) => panic!("{backend:?}: {error}"),
                        Ok(_) => panic!("{backend:?} certified stored-key values"),
                    },
                }
            }
            // A K8/V4 history has no stored-key entry.
            let load = ModelLoadPlan::derive(
                &manifest,
                &definition,
                ComponentSelection {
                    head: false,
                    vision: false,
                },
                resident_layout(ExecutionPath::Native, BackendName::Cpu),
                crate::PlannedMethod::Plain,
                crate::planning::fixtures::LIMITS,
            )
            .unwrap();
            match load.program_plan(&definition, KvCodec::AffineK8V4) {
                Err(error) => assert!(
                    error.to_string().contains("stored-key values in a K8/V4 history"),
                    "{error}"
                ),
                Ok(_) => panic!("planned stored-key values in a K8/V4 history"),
            }
        }

        /// A stored-key window layer's graphs prepare on this host's device
        /// with the one-plane storage the checked family certifies.
        #[test]
        fn stored_key_history_prepares_on_the_selected_device() {
            let catalog = seismic::Devices::discover(&[]).unwrap();
            let chosen = crate::platform::select_device(
                &catalog,
                ExecutionPath::Native,
                crate::platform::DeviceRequest::Automatic,
                &crate::platform::MemoryReserves::of_machine(None),
            )
            .unwrap();
            let device = catalog
                .open(catalog.resolve(chosen.info.selector()).unwrap())
                .unwrap();
            if !matches!(device.backend(), BackendName::Cpu | BackendName::Metal) {
                return;
            }
            let (definition, manifest) = stored_key_window();
            let limits = limits();
            let draft = crate::ExecutionPlanner::prepare(
                &chosen,
                &manifest,
                &definition,
                ComponentSelection {
                    head: false,
                    vision: false,
                },
                ExecutionPath::Native,
                PlannedMethod::Plain,
                KvCodec::Dense,
                limits,
            )
            .unwrap();
            let state = ResourcePlanner::state_plan(
                &definition,
                draft.load(),
                PlannedMethod::Plain,
                KvCodec::Dense,
                limits,
                ResourceCapacity {
                    domain_bytes: chosen.assessment_capacity_bytes.min(1 << 30),
                    tensor_operations: crate::TensorOperations::of(
                        chosen.info.forms_tensor_operations(),
                    ),
                },
            )
            .unwrap();
            let mut programs = crate::AttestedPrograms::prepare_draft(
                &draft,
                &device,
                crate::TuningContext {
                    definition: &definition,
                    weights: &crate::ZeroTuningWeights,
                    observer: &crate::UnreportedTuning,
                    cache: None,
                    numerical_profile: crate::ResolvedNumericalProfile::standard(),
                    settings: crate::TuningSettings::STANDARD,
                },
                crate::UnpreparedUnits::ServeDefaults,
            )
            .unwrap();
            programs
                .prepare_target_graphs(
                    &device,
                    draft.load(),
                    &definition.decoder,
                    &state,
                    draft.programs().target(),
                    limits,
                    &seismic::Workers::new(1),
                )
                .unwrap();
        }

        /// The grouped output's kernels prepare at their defaults on this
        /// host's device (the grouped `project_rows`, then `attention_output`
        /// over its products), and its graphs prepare with the storage the
        /// checked family certifies.
        #[test]
        fn grouped_output_prepares_on_the_selected_device() {
            let catalog = seismic::Devices::discover(&[]).unwrap();
            let chosen = crate::platform::select_device(
                &catalog,
                ExecutionPath::Native,
                crate::platform::DeviceRequest::Automatic,
                &crate::platform::MemoryReserves::of_machine(None),
            )
            .unwrap();
            let device = catalog
                .open(catalog.resolve(chosen.info.selector()).unwrap())
                .unwrap();
            if !matches!(device.backend(), BackendName::Cpu | BackendName::Metal) {
                return;
            }
            let (definition, manifest) = selected(false, true);
            let limits = limits();
            let draft = crate::ExecutionPlanner::prepare(
                &chosen,
                &manifest,
                &definition,
                ComponentSelection {
                    head: false,
                    vision: false,
                },
                ExecutionPath::Native,
                PlannedMethod::Plain,
                KvCodec::Dense,
                limits,
            )
            .unwrap();
            let state = ResourcePlanner::state_plan(
                &definition,
                draft.load(),
                PlannedMethod::Plain,
                KvCodec::Dense,
                limits,
                ResourceCapacity {
                    domain_bytes: chosen.assessment_capacity_bytes.min(1 << 30),
                    tensor_operations: crate::TensorOperations::of(
                        chosen.info.forms_tensor_operations(),
                    ),
                },
            )
            .unwrap();
            let mut programs = crate::AttestedPrograms::prepare_draft(
                &draft,
                &device,
                crate::TuningContext {
                    definition: &definition,
                    weights: &crate::ZeroTuningWeights,
                    observer: &crate::UnreportedTuning,
                    cache: None,
                    numerical_profile: crate::ResolvedNumericalProfile::standard(),
                    settings: crate::TuningSettings::STANDARD,
                },
                crate::UnpreparedUnits::ServeDefaults,
            )
            .unwrap();
            let target = programs
                .prepare_target_graphs(
                    &device,
                    draft.load(),
                    &definition.decoder,
                    &state,
                    draft.programs().target(),
                    limits,
                    &seismic::Workers::new(1),
                )
                .unwrap();
            let checked = checked_target_family_storage(
                device.backend(),
                draft.load(),
                &definition.decoder,
                &state,
                draft.programs().target(),
                limits,
            )
            .unwrap();
            assert_eq!(target.family().workspace_bytes(), checked.storage.workspace);
        }

        /// A selected read binds its indexer at the served context, keeps
        /// its raw keys and block keys in their own domains, and its graphs
        /// (the checked metadata route of the production fragment: the
        /// indexer, the Segments decode and the WorkList prefill) certify on
        /// every backend at every row class, with a full and with a
        /// low-rank query (its `qr` rows widened and projected up).
        #[test]
        fn selected_attention_plans_its_indexer_and_certifies_on_every_backend() {
            for low_rank in [false, true] {
                plans_and_certifies(low_rank);
            }
        }

        fn plans_and_certifies(low_rank: bool) {
            let (definition, manifest) = selected(low_rank, false);
            definition.validate().unwrap();
            assert_eq!(definition.deferred_forms(), []);
            let limits = limits();
            for backend in [
                BackendName::Cpu,
                BackendName::Metal,
                BackendName::Cuda,
                BackendName::Vulkan,
            ] {
                for codec in [KvCodec::Dense, KvCodec::AffineK8V4] {
                    let load = ModelLoadPlan::derive(
                        &manifest,
                        &definition,
                        ComponentSelection {
                            head: false,
                            vision: false,
                        },
                        resident_layout(ExecutionPath::Native, backend),
                        crate::PlannedMethod::Plain,
                        crate::planning::fixtures::LIMITS,
                    )
                    .unwrap();
                    let plan = load.program_plan(&definition, codec).unwrap();
                    let MixerProgramSlot::Attention(binding) = plan.target().blocks()[0].mixer()
                    else {
                        panic!("an attention block");
                    };
                    assert_eq!(
                        binding.low_rank_query.map(|low_rank| low_rank.rank),
                        low_rank.then_some(LOW_RANK)
                    );
                    assert_eq!(
                        binding.selection.map(|selection| selection.shape),
                        Some(SelectionShape {
                            heads: 2,
                            width: 64,
                            block: 4,
                            blocks: 2,
                            // Blocks [0, 1024] of a 4096-token context.
                            words: 33,
                        })
                    );
                    let state = ResourcePlanner::state_plan(
                        &definition,
                        &load,
                        PlannedMethod::Plain,
                        codec,
                        limits,
                        ResourceCapacity {
                            domain_bytes: 16 * 1024 * 1024 * 1024,
                            tensor_operations: crate::TensorOperations::Absent,
                        },
                    )
                    .unwrap();
                    // The Token domain, then each indexer's raw keys and
                    // block keys.
                    let kinds = state
                        .target_state()
                        .domains
                        .iter()
                        .map(|domain| domain.kind)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        kinds,
                        [
                            HistoryDomainKind::Token,
                            HistoryDomainKind::Window { rows: 3 },
                            HistoryDomainKind::Block { rate: 4 },
                        ]
                    );
                    for block in 0..2 {
                        let (window, blocks) =
                            state.target_state().indexer_history(block).unwrap();
                        assert!(
                            window
                                .components
                                .iter()
                                .any(|component| component.layer == LayerRef::Indexer(block))
                        );
                        assert!(blocks.rows as u64 >= CONTEXT / 4);
                    }
                    checked_target_family_storage(
                        backend,
                        &load,
                        &definition.decoder,
                        &state,
                        plan.target(),
                        limits,
                    )
                    .unwrap_or_else(|error| {
                        panic!("{backend:?} {codec:?} low-rank {low_rank}: {error}")
                    });
                }
            }
        }
    }

    /// Each of DeepSeek V4.1's window-attention forms the entries have no
    /// axis for is refused by name, never planned as another form; its
    /// low-rank query is admitted.
    #[test]
    fn forms_without_an_entry_axis_are_refused_by_name() {
        use magnitude_family_contracts::{
            OutputProjection, OutputRotary, QueryProjection, RmsNorm, RotaryLayout, RotaryPairing,
            RotaryPlacement, WeightDescriptor,
        };
        let weight = |name: &str, shape: &[u64]| WeightDescriptor::stored(name, shape);
        let base = Attention {
            heads: 2,
            kv_heads: 1,
            width: 64,
            query: QueryProjection::Full(weight("q", &[128, 32])),
            gate: AttentionGate::None,
            query_norm: HeadNorm::None,
            key_value: KeyValue::Owned {
                key: weight("k", &[64, 32]),
                value: ValueSource::RawKey,
                key_norm: HeadNorm::None,
                value_norm: ValueNorm::None,
                domain: HistoryDomain::Window { tokens: 128 },
            },
            rotary: Rotary::Table {
                pairs: vec![pair(1.0); 8],
                layout: RotaryLayout::HALF_SPLIT_LEADING,
                divisors: None,
                output: OutputRotary::None,
            },
            scale: 0.125,
            reads: HistoryReads::Visible,
            media_rows: MediaRowAttention::Causal,
            sinks: Some(weight("sinks", &[2])),
            output: OutputProjection::Full(weight("o", &[32, 128])),
        };
        admit(&base, 1e-6).unwrap();
        // The low-rank query composes from the projection entries.
        let mut low_rank = base.clone();
        low_rank.query = QueryProjection::LowRank {
            down: weight("q_a", &[16, 32]),
            norm: RmsNorm {
                weight: weight("q_a_norm", &[16]),
                epsilon: 1e-20,
            },
            up: weight("q_b", &[128, 16]),
        };
        admit(&low_rank, 1e-6).unwrap();
        assert_eq!(low_rank_query(&low_rank), Some((16, 1e-20)));
        // DeepSeek V4.1's form: values the stored key row, adjacent pairs
        // over the trailing dimensions, the attended row turned back, a key
        // norm without a query norm.
        let rms = |name: &str| {
            HeadNorm::Rms(RmsNorm {
                weight: weight(name, &[64]),
                epsilon: 1e-6,
            })
        };
        let mut v41 = base.clone();
        let Rotary::Table { layout, output, .. } = &mut v41.rotary else {
            unreachable!()
        };
        *layout = RotaryLayout {
            pairs: RotaryPairing::Adjacent,
            placement: RotaryPlacement::Trailing,
        };
        *output = OutputRotary::Inverse;
        let KeyValue::Owned {
            key_norm, value, ..
        } = &mut v41.key_value
        else {
            unreachable!()
        };
        *key_norm = rms("k_norm");
        *value = ValueSource::StoredKey;
        admit(&v41, 1e-6).unwrap();
        let axes = shape(0, &v41).unwrap();
        assert_eq!(
            (
                axes.query_norm,
                axes.key_norm,
                axes.rotary_adjacent,
                axes.rotary_trailing,
                axes.output_rotary,
                axes.stored_key(),
                axes.value_rows(),
            ),
            (0, 1, 1, 1, 1, 1, 0)
        );
        // Its history is the key plane alone, dense only.
        let dense = history_codec(&v41, KvCodec::Dense, Element::bf16()).unwrap();
        assert_eq!(dense.value, HistoryValue::Key);
        assert_eq!(dense.planes().unwrap().len(), 1);
        match history_codec(&v41, KvCodec::AffineK8V4, Element::bf16()) {
            Err(PlanError::Unsupported(reason)) => {
                assert_eq!(reason, "stored-key values in a K8/V4 history")
            }
            other => panic!("planned as {other:?}"),
        }
        let mut gated = v41.clone();
        gated.gate = AttentionGate::Separate {
            weight: weight("g", &[2, 32]),
            function: GateFunction::Sigmoid,
            granularity: GateGranularity::Head,
        };
        match admit(&gated, 1e-6) {
            Err(PlanError::Unsupported(reason)) => {
                assert_eq!(reason, "gated attention with an inverse output rotary")
            }
            other => panic!("admitted as {other:?}"),
        }
        let mut selected = v41.clone();
        selected.reads = HistoryReads::Selected {
            block: 4,
            blocks: 2,
            indexer: magnitude_family_contracts::Indexer {
                heads: 2,
                width: 64,
                query: weight("indexer.q", &[128, 32]),
                key: weight("indexer.k", &[64, 32]),
                query_norm: RmsNorm {
                    weight: weight("indexer.q_norm", &[64]),
                    epsilon: 1e-6,
                },
                key_norm: RmsNorm {
                    weight: weight("indexer.k_norm", &[64]),
                    epsilon: 1e-6,
                },
            },
        };
        match admit(&selected, 1e-6) {
            Err(PlanError::Unsupported(reason)) => assert_eq!(
                reason,
                "selected attention with rotary pairs other than half-split leading"
            ),
            other => panic!("admitted as {other:?}"),
        }
    }

    /// The decode demand of a stored-key layer reads half the history bytes
    /// of K + V: assessment sums its component's planes
    /// (`assessment::demand`, read as it is), which the history codec
    /// derives. The fixture's one kv head of 64 bf16 values: 256 bytes per
    /// token over two planes, 128 over the key plane.
    #[test]
    fn stored_key_demand_reads_half_the_history_bytes() {
        let read = |stored: bool| {
            let mut definition = crate::planning::fixtures::fixture_definition();
            for sublayer in definition
                .decoder
                .blocks
                .iter_mut()
                .flat_map(|block| &mut block.sublayers)
                .filter(|_| stored)
            {
                let Operator::Attention(attention) = &mut sublayer.op else {
                    continue;
                };
                if let KeyValue::Owned { value, .. } = &mut attention.key_value {
                    *value = ValueSource::StoredKey;
                }
            }
            let manifest = crate::planning::fixtures::fixture_manifest(&definition);
            let load = crate::ModelLoadPlan::derive(
                &manifest,
                &definition,
                crate::ComponentSelection {
                    head: false,
                    vision: false,
                },
                seismic::Layout::Rows8,
                crate::PlannedMethod::Plain,
                crate::planning::fixtures::LIMITS,
            )
            .unwrap();
            crate::assessment::demand::tests::demand(&definition, &load, KvCodec::Dense)
                .unwrap()
                .calls
                .iter()
                .filter_map(|call| call.work.history)
                .map(|history| history.bytes_per_token)
                .collect::<Vec<_>>()
        };
        assert_eq!(read(false), [256]);
        assert_eq!(read(true), [128]);
    }

    #[test]
    fn stored_rotary_divisors_must_reproduce_the_table() {
        let bases = [1.0, 0.5, 0.25];
        let pairs = [pair(1.0), pair(0.25), pair(0.0)];
        // 1/1, 0.5/2, 0.25/1e30 (unrotated).
        check_rotary_divisors(&pairs, &bases, &[1.0, 2.0, 1e30]).unwrap();
        assert!(check_rotary_divisors(&pairs, &bases, &[1.0, 4.0, 1e30]).is_err());
        assert!(check_rotary_divisors(&pairs, &bases, &[1.0, 2.0, 1.0]).is_err());
        assert!(check_rotary_divisors(&pairs, &bases, &[1.0, 2.0]).is_err());
    }
}
