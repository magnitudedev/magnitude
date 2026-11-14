//! Softmax attention over history spans (`Operator::Attention`): the forms
//! the attention entries implement, their static axes, weight roles and
//! program binding.

pub(crate) mod graph;
pub(crate) mod pooled;
pub(crate) mod selection;
pub(crate) mod split;

use super::{PlanError, WeightPush, tail};
use crate::planning::{AttentionForm, AttentionWidths, GateForm, SeparateGate};
use crate::{
    AbsorbedQueryBinding, AbsorbedQueryShape, AttentionBinding, AttentionShape, AttentionValues,
    CandidatePool, CandidatesBinding,
    GroupedOutputBinding, GroupedOutputShape, IndexerBinding, LowRankQueryBinding, PickedBinding,
    PooledBinding, ScalableWeight, SelectionBinding, SelectionKeysBinding, SelectionShape,
};
use magnitude_family_contracts::{
    Attention, AttentionGate, Candidates, Decoder, GateFunction, GateGranularity, HeadNorm, HeadWeights,
    HistoryReads, HistoryRows, Indexer, IndexerInput, IndexerKeys, MediaRowAttention,
    Operator, OutputForm, OutputProjection, OutputRotary, Pool, PoolNorm, PooledRows, ProjectedRows,
    QueryKeys, QueryProjection, RmsNorm, Rotary, RotaryPair, RotaryPairing,
    RotaryPlacement, RowProducer, RowRounding, Selection, SublayerIndex, TokenDomain, ValueNorm,
    ValueSource, WeightKind,
};
use magnitude_state::{CodecSpec, HistoryEncoding, KvCodec};
use seismic::{DType, Element};

/// The rows of an attention's first source, the one its fresh rows join:
/// its own projected rows, or those of the earlier layer it shares.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FreshRows<'a> {
    Own(&'a ProjectedRows),
    Shared(SublayerIndex),
}

/// [`FreshRows`] of `attention`. `Decoder::validate` keeps a pooled source
/// out of the first place.
pub(crate) fn fresh_rows(attention: &Attention) -> FreshRows<'_> {
    match &attention.fresh().rows {
        HistoryRows::Own(RowProducer::Projected(rows)) => FreshRows::Own(rows),
        HistoryRows::Shared { source, .. } => FreshRows::Shared(*source),
        HistoryRows::Own(RowProducer::Pooled(_)) => {
            panic!("a validated attention's first source is projected")
        }
    }
}

/// The norm of a pooled source's rows the entries run: an RMS norm of the
/// pooled row; a per-token norm of a pooled source's rows (no family's)
/// refused by name.
pub(crate) fn pooled_norm(rows: &PooledRows) -> Result<&RmsNorm, PlanError> {
    match &rows.norm {
        PoolNorm::Pooled(norm) => Ok(norm),
        PoolNorm::Token { .. } => Err(PlanError::Unsupported(
            "a pooled source's rows normed per token",
        )),
    }
}

/// [`pooled_norm`] of pooled rows of an admitted attention, which
/// [`admit`] holds to that form.
pub(crate) fn admitted_pooled_norm(rows: &PooledRows) -> &RmsNorm {
    pooled_norm(rows).expect("admission refuses every other pooled norm")
}

/// The rows of `attention`'s pooled sources (an indexer's pooled keys,
/// which `selected_read` admits, are not among them).
fn pooled_sources(attention: &Attention) -> impl Iterator<Item = &PooledRows> {
    attention.history.iter().filter_map(|source| match &source.rows {
        HistoryRows::Own(RowProducer::Pooled(rows)) => Some(rows),
        HistoryRows::Own(RowProducer::Projected(_)) | HistoryRows::Shared { .. } => None,
    })
}

/// A selected read over the layer's token rows (its first source's
/// `HistoryReads::Selected`): its indexer, scoring units of `keys.rate`
/// rows, and those pooled keys, which `pool_rows` makes into the keys'
/// Window and Block (`graph::select`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SelectedRead<'a> {
    pub indexer: &'a Indexer,
    pub keys: &'a PooledRows,
}

/// The selected read of `reads` over token rows: none for a read of every
/// visible row, and every form the entries don't implement refused by
/// name.
pub(crate) fn selected_read(reads: &HistoryReads) -> Result<Option<SelectedRead<'_>>, PlanError> {
    let indexer = match reads {
        HistoryReads::Visible => return Ok(None),
        HistoryReads::Selected(Selection::Reuse { .. }) => {
            return Err(PlanError::Unsupported("a reused selection over token rows"));
        }
        HistoryReads::Selected(Selection::Own(indexer)) => indexer,
    };
    let IndexerKeys::Pooled(keys) = &indexer.keys else {
        return Err(PlanError::Unsupported(
            "indexer keys over token rows other than pooled keys",
        ));
    };
    if !matches!(keys.rounding, RowRounding::None) {
        return Err(PlanError::Unsupported("rounded pooled indexer keys"));
    }
    // Validation admits no pooled indexer keys of one row.
    assert!(!matches!(keys.pool, Pool::Identity), "validated pooled indexer keys pool at least two rows");
    if !matches!(indexer.candidates, Candidates::All) {
        return Err(PlanError::Unsupported("a candidate pool over token rows"));
    }
    Ok(Some(SelectedRead { indexer, keys }))
}

/// [`selected_read`] of an admitted attention's first source, whose
/// selection [`admit`] holds to that form.
pub(crate) fn admitted_selection(attention: &Attention) -> Option<SelectedRead<'_>> {
    selected_read(&attention.fresh().reads)
        .expect("admission refuses every other selected read")
}

/// A layer's second source, read as an index list into its rows
/// (`attention_decode`/`_prefill`'s `selected_rows` over `selected`, `KS`
/// picks per row) in the softmax over its first.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PickedSource<'a> {
    /// Its own pooled rows, picked by its own indexer (a Full layer):
    /// `pool_rows` makes them (at rate 1, each token's own row, which
    /// `attention_append_dense` stores), the indexer's entries key and pick
    /// them.
    Own {
        part: u32,
        rows: &'a PooledRows,
        indexer: &'a Indexer,
    },
    /// Part `part` of the earlier layer `source`'s pooled rows, picked by
    /// its own indexer over that layer's index keys (a Reindex layer): it
    /// makes no rows or keys, and its picks replace that layer's for the
    /// later layers reusing them.
    Reindex {
        source: SublayerIndex,
        part: u32,
        indexer: &'a Indexer,
    },
    /// Part `part` of the earlier layer `source`'s pooled rows, through the
    /// picks of the indexer of part `picker_part` of layer `picker` (a Reuse
    /// layer): it makes and picks nothing.
    Reused {
        source: SublayerIndex,
        part: u32,
        picker: SublayerIndex,
        picker_part: u64,
    },
}

impl PickedSource<'_> {
    /// The pooled source whose planes the second source of block `block`'s
    /// attention reads: part `part` of its own rows, or of its source
    /// layer's. A graph shared by blocks of one shape names no block; each
    /// block keeps its own (`PreparedTargetGraphs::pooled_source`).
    pub(crate) fn pooled(&self, block: u32) -> pooled::PooledSource {
        match *self {
            Self::Own { part, .. } => pooled::PooledSource { block, part },
            Self::Reindex { source, part, .. } | Self::Reused { source, part, .. } => {
                pooled::PooledSource { block: source.block, part }
            }
        }
    }
}

/// An attention sublayer at `index` of `decoder`: what its second source
/// resolves of the decoder's other layers (the indexer it reuses, the
/// pooled rows it reads, the candidate pool it produces or restricts to).
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecoderSite<'a> {
    pub decoder: &'a Decoder,
    pub index: SublayerIndex,
}

/// The second source of `attention`, if it has one, in a form the entries
/// implement: its own pooled rows under an indexer over the low-rank query
/// row with latent keys, another layer's pooled rows under its own such
/// indexer over that layer's keys, or another layer's pooled rows through
/// an earlier layer's picks. Every other form is refused by name.
pub(crate) fn picked_source(attention: &Attention) -> Result<Option<PickedSource<'_>>, PlanError> {
    let second = match attention.history.as_slice() {
        [_] => return Ok(None),
        [_, second] => second,
        _ => {
            return Err(PlanError::Unsupported(
                "attention over more than two history sources",
            ));
        }
    };
    let pooled_part =
        |part: u64| u32::try_from(part).map_err(|_| PlanError::Unsupported("a pooled part past u32"));
    // The entries' indexer: its query over the low-rank query row.
    let admitted = |indexer: &Indexer| match indexer.query.input {
        IndexerInput::QueryRank => Ok(()),
        IndexerInput::Hidden => Err(PlanError::Unsupported(
            "an indexer over the hidden row of a pooled source",
        )),
    };
    match (&second.rows, &second.reads) {
        (_, HistoryReads::Visible) => Err(PlanError::Unsupported(
            "a pooled history source read whole",
        )),
        (
            HistoryRows::Own(RowProducer::Pooled(rows)),
            HistoryReads::Selected(Selection::Own(indexer)),
        ) => {
            admitted(indexer.as_ref())?;
            match (&indexer.keys, &indexer.candidates) {
                (IndexerKeys::Latent { .. }, Candidates::All | Candidates::Produce { .. }) => {}
                (IndexerKeys::Pooled(_), _) => {
                    return Err(PlanError::Unsupported(
                        "pooled indexer keys over a pooled source",
                    ));
                }
                // An indexer owning its rows has no earlier one over them.
                (IndexerKeys::Shared { .. }, _) | (_, Candidates::Restrict { .. }) => {
                    return Err(PlanError::Topology(
                        "a validated indexer over its own rows keeps their keys and restricts to no pool",
                    ));
                }
            }
            Ok(Some(PickedSource::Own {
                part: 1,
                rows,
                indexer,
            }))
        }
        (
            HistoryRows::Shared { source, part: rows_part },
            HistoryReads::Selected(Selection::Own(indexer)),
        ) => {
            admitted(indexer.as_ref())?;
            // Contract validation holds shared keys and a restriction to
            // the indexer keeping its own over the same rows: the rows'
            // owner, which a Shared source names.
            let owner = (*source, *rows_part);
            match (&indexer.keys, &indexer.candidates) {
                (IndexerKeys::Shared { source, part }, Candidates::All)
                    if (*source, *part) == owner => {}
                (
                    IndexerKeys::Shared { source, part },
                    Candidates::Restrict { source: producer, part: producer_part },
                ) if (*source, *part) == owner && (*producer, *producer_part) == owner => {}
                (IndexerKeys::Shared { .. }, _) => {
                    return Err(PlanError::Topology(
                        "a validated Reindex indexer shares the keys and pool of the rows' owner",
                    ));
                }
                (IndexerKeys::Pooled(_) | IndexerKeys::Latent { .. }, _) => {
                    return Err(PlanError::Unsupported(
                        "an indexer keeping its own keys over another layer's pooled rows",
                    ));
                }
            }
            Ok(Some(PickedSource::Reindex {
                source: *source,
                part: pooled_part(*rows_part)?,
                indexer,
            }))
        }
        (
            HistoryRows::Shared { source, part: rows_part },
            HistoryReads::Selected(Selection::Reuse { source: picker, part: picker_part }),
        ) => {
            // Contract validation holds the reused picks to the same rows.
            Ok(Some(PickedSource::Reused {
                source: *source,
                part: pooled_part(*rows_part)?,
                picker: *picker,
                picker_part: *picker_part,
            }))
        }
        (HistoryRows::Own(_), HistoryReads::Selected(Selection::Reuse { .. }))
        | (HistoryRows::Own(RowProducer::Projected(_)), _) => Err(PlanError::Topology(
            "a validated second source is pooled and reads its own picks or reuses another's",
        )),
    }
}

/// The own indexer of part `part` of the attention sublayer `index` of
/// `decoder`.
fn indexer_at(decoder: &Decoder, index: SublayerIndex, part: u64) -> Option<&Indexer> {
    match &decoder.attention(index)?.history.get(part as usize)?.reads {
        HistoryReads::Selected(Selection::Own(indexer)) => Some(indexer.as_ref()),
        _ => None,
    }
}

/// The picks a row reads from `attention`'s second source (`KS`): its own
/// indexer's `top`, or for a reused selection the top of the indexer it
/// reuses, which only its decoder (`site`) knows.
pub(crate) fn picks(attention: &Attention, site: Option<DecoderSite<'_>>) -> Result<u64, PlanError> {
    match picked_source(attention)? {
        None => Ok(0),
        Some(PickedSource::Own { indexer, .. } | PickedSource::Reindex { indexer, .. }) => {
            Ok(indexer.top)
        }
        Some(PickedSource::Reused { picker, picker_part, .. }) => {
            let site = site.ok_or(PlanError::Topology(
                "a reused selection's picks without its decoder",
            ))?;
            indexer_at(site.decoder, picker, picker_part)
                .map(|indexer| indexer.top)
                .ok_or(PlanError::Topology("a reused selection names no earlier indexer"))
        }
    }
}

/// The pooled rows a Reindex source reads: part `part` of the earlier
/// layer `source`'s own.
pub(crate) fn shared_rows(decoder: &Decoder, source: SublayerIndex, part: u32) -> Result<&PooledRows, PlanError> {
    match decoder
        .attention(source)
        .and_then(|owner| owner.history.get(part as usize))
        .map(|owner| &owner.rows)
    {
        Some(HistoryRows::Own(RowProducer::Pooled(rows))) => Ok(rows),
        _ => Err(PlanError::Topology(
            "a Reindex source names no earlier layer's own pooled rows",
        )),
    }
}

/// The candidates the indexer of the second source (part `part`) of the
/// attention sublayer at `site` scores: every unit; the pool it produces,
/// listed for the top of the indexers restricted to it (one top, or the
/// pool is refused by name; a pool none restricts to is not made); or the
/// pool it restricts to.
pub(crate) fn candidates(
    site: Option<DecoderSite<'_>>,
    part: u64,
    indexer: &Indexer,
) -> Result<CandidatesBinding, PlanError> {
    let site = || site.ok_or(PlanError::Topology("a candidate pool without its decoder"));
    match indexer.candidates {
        Candidates::All => Ok(CandidatesBinding::All),
        Candidates::Produce { blocks, block_units } => {
            let site = site()?;
            let mut tops = site
                .decoder
                .sublayers()
                .filter_map(|(index, _)| site.decoder.attention(index))
                .flat_map(|attention| &attention.history)
                .filter_map(|source| match &source.reads {
                    HistoryReads::Selected(Selection::Own(restricted)) => Some(restricted),
                    _ => None,
                })
                .filter(|restricted| {
                    restricted.candidates
                        == Candidates::Restrict {
                            source: site.index,
                            part,
                        }
                })
                .map(|restricted| restricted.top);
            let Some(top) = tops.next() else {
                return Ok(CandidatesBinding::All);
            };
            if tops.any(|other| other != top) {
                return Err(PlanError::Unsupported(
                    "indexers restricted to one candidate pool with differing tops",
                ));
            }
            Ok(CandidatesBinding::Produce(CandidatePool { blocks, block_units, top }))
        }
        Candidates::Restrict { source, part: producer_part } => {
            let site = site()?;
            match indexer_at(site.decoder, source, producer_part).map(|producer| &producer.candidates) {
                Some(Candidates::Produce { blocks, block_units }) => {
                    Ok(CandidatesBinding::Restrict(CandidatePool {
                        blocks: *blocks,
                        block_units: *block_units,
                        top: indexer.top,
                    }))
                }
                _ => Err(PlanError::Topology(
                    "a restricted indexer names no earlier candidate pool",
                )),
            }
        }
    }
}

/// The scores an indexer row of a pooled source of `rate` keeps (`N`) under
/// a `context` limit: a row at t < context lists the (t + 1) / rate rows
/// complete before it.
pub(crate) fn list_scores(context: u64, rate: u64) -> u64 {
    context.div_ceil(rate).max(1)
}

/// The trained roundings of an index-list source's own indexer as its
/// entries take them: its query rows' (`indexer_score`) and those of the
/// keys it writes (`indexer_keys`), none for keys it shares, which their
/// owner writes. The one reading of them for its binding
/// (`IndexerBinding`) and its graphs (`pooled::IndexBlock`).
pub(crate) fn indexer_roundings(indexer: &Indexer) -> (crate::planning::RoundingArgs, crate::planning::RoundingArgs) {
    let keys = match &indexer.keys {
        IndexerKeys::Latent { rounding, .. } => crate::planning::rounding_args(*rounding),
        IndexerKeys::Shared { .. } => crate::planning::RoundingArgs::NONE,
        // A selected read's pooled keys, which it admits unrounded.
        IndexerKeys::Pooled(keys) => crate::planning::rounding_args(keys.rounding),
    };
    (crate::planning::rounding_args(indexer.query.rounding), keys)
}

/// The key rounding the dense entries apply to `attention`'s prepared keys
/// (`KR` and its four scalars): its own projected rows' trained rounding,
/// or none.
fn key_rounding(attention: &Attention) -> RowRounding {
    match fresh_rows(attention) {
        FreshRows::Own(rows) => rows.rounding,
        FreshRows::Shared(_) => RowRounding::None,
    }
}

/// The attention forms the attention entries implement: a full query, or a
/// low-rank one composed of two projections (`graph::attention`), either
/// meeting the keys directly or absorbed per head by `project_rows`, a full or
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
    // The entries read the first source by its visible spans and a second,
    // pooled one by its picks ([`picked_source`]); a Shared layer's first
    // source is its source's, admitted with the source. A rounded first
    // source holds stored keys (contract validation), which the dense
    // entries round as they prepare them (`KR`).
    for rows in pooled_sources(attention) {
        pooled_norm(rows)?;
        // `pool_rows` gates a pooled source's rows by their scores alone.
        if matches!(rows.pool, Pool::SoftmaxGated { position: Some(_), .. }) {
            return Err(PlanError::Unsupported("a pooled source's rows gated by place"));
        }
    }
    let picked = picked_source(attention)?;
    if picked.is_some() && !matches!(attention.fresh().reads, HistoryReads::Visible) {
        return Err(PlanError::Unsupported(
            "a selected first source beside a pooled source",
        ));
    }
    // A pooled plane holds one row per group, which every key head reads.
    if picked.is_some() && attention.kv_heads != 1 {
        return Err(PlanError::Unsupported(
            "a pooled source beside more than one key head",
        ));
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
    // A selected read's indexer orders keys by position: its rows attend
    // causally.
    if selected_read(&attention.fresh().reads)?.is_some()
        && attention.media_rows != MediaRowAttention::Causal
    {
        return Err(PlanError::Unsupported(
            "selected attention attending media rows bidirectionally",
        ));
    }
    let shape = geometry(0, attention)?;
    if shape.widths.width < 2 * shape.widths.rotary_pairs {
        return Err(PlanError::Topology("rotary pairs exceed the head width"));
    }
    // The entries turn the published attended row back, which a gate
    // would have already scaled (`attention.seismic`'s `output_rotary`).
    if shape.form.output_rotary && shape.form.gate != GateForm::None {
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
    let mut domains: Vec<(TokenDomain, MediaRowAttention)> = Vec::new();
    for (index, sublayer) in decoder.sublayers() {
        let Operator::Attention(attention) = &sublayer.op else {
            continue;
        };
        let differ = || {
            PlanError::Unsupported(
                "attention layers of one history attending media rows differently",
            )
        };
        match fresh_rows(attention) {
            FreshRows::Own(rows) => {
                match domains.iter().find(|(other, _)| *other == rows.domain) {
                    Some((_, form)) if *form != attention.media_rows => return Err(differ()),
                    Some(_) => {}
                    None => domains.push((rows.domain, attention.media_rows)),
                }
            }
            FreshRows::Shared(source) => {
                let Some(Operator::Attention(owner)) = decoder
                    .sublayers()
                    .find(|(candidate, _)| *candidate == source)
                    .map(|(_, sublayer)| &sublayer.op)
                else {
                    return Err(PlanError::Topology(
                        "a Shared layer's source is not attention",
                    ));
                };
                if owner.media_rows != attention.media_rows || source >= index {
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
/// `hidden`-wide residual at `site`, its picks ([`picks`]) included: a
/// reused selection's are the indexer's it reuses, which only its decoder
/// names. Admission, which has no decoder, checks [`geometry`].
pub(crate) fn shape(
    hidden: u64,
    attention: &Attention,
    site: Option<DecoderSite<'_>>,
) -> Result<AttentionShape, PlanError> {
    let geometry = geometry(hidden, attention)?;
    Ok(AttentionShape {
        widths: AttentionWidths {
            picks: picks(attention, site)?,
            ..geometry.widths
        },
        ..geometry
    })
}

/// [`shape`]'s axes that the operator alone decides: every one but the picks
/// of a reused selection, which only its decoder knows (`picks` 0 here).
fn geometry(hidden: u64, attention: &Attention) -> Result<AttentionShape, PlanError> {
    if attention.kv_heads == 0 || attention.heads % attention.kv_heads != 0 {
        return Err(PlanError::Topology(
            "attention heads are not a multiple of its key/value heads",
        ));
    }
    let width = attention.width;
    let gate = match &attention.gate {
        AttentionGate::None => GateForm::None,
        AttentionGate::Interleaved { .. } => GateForm::Interleaved,
        AttentionGate::Separate { granularity, .. } => GateForm::Separate(match granularity {
            GateGranularity::Element => SeparateGate::Element,
            GateGranularity::Head => SeparateGate::Head,
        }),
    };
    // The rotary layout and output rotary (`RA`, `RT`, `RO`).
    let (rotary_adjacent, rotary_trailing, output_rotary) = match &attention.rotary {
        Rotary::None | Rotary::Interleaved { .. } => (false, false, false),
        // The inverse of no rotation is none.
        Rotary::Table {
            pairs,
            layout,
            output,
            ..
        } => (
            layout.pairs == RotaryPairing::Adjacent,
            layout.placement == RotaryPlacement::Trailing,
            *output == OutputRotary::Inverse && !pairs.is_empty(),
        ),
    };
    let query_norm = matches!(attention.query_norm, HeadNorm::Rms(_));
    let (fresh, key_norm, value_norm, values) = match fresh_rows(attention) {
        FreshRows::Own(rows) => (
            true,
            matches!(rows.key_norm, HeadNorm::Rms(_)),
            matches!(rows.value_norm, ValueNorm::RmsUnweighted(_)),
            match rows.value {
                ValueSource::Projected(_) => AttentionValues::Projected,
                ValueSource::RawKey => AttentionValues::RawKey,
                ValueSource::StoredKey => AttentionValues::StoredKey,
            },
        ),
        // A Shared layer projects no keys: its unread `N` is its `NQ`.
        FreshRows::Shared(_) => (false, query_norm, false, AttentionValues::Projected),
    };
    Ok(AttentionShape {
        form: AttentionForm {
            gate,
            fresh,
            query_norm,
            key_norm,
            value_norm,
            values,
            sinks: attention.sinks.is_some(),
            rotary_adjacent,
            rotary_trailing,
            output_rotary,
            absorbed_query: matches!(attention.query_keys, QueryKeys::Absorbed { .. }),
        },
        widths: AttentionWidths {
            hidden,
            kv_heads: attention.kv_heads,
            group: attention.heads / attention.kv_heads,
            rotary_pairs: match &attention.rotary {
                Rotary::None => 0,
                Rotary::Interleaved { width, .. } => width / 2,
                Rotary::Table { pairs, .. } => pairs.len() as u64,
            },
            width,
            picks: 0,
        },
        key_rounding: crate::planning::rounding_args(key_rounding(attention)),
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
    if let FreshRows::Own(rows) = fresh_rows(attention) {
        if let HeadNorm::Rms(norm) = &rows.key_norm {
            epsilons.push(norm.epsilon);
        }
        if let ValueNorm::RmsUnweighted(norm) = rows.value_norm {
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
    if let QueryKeys::Absorbed { weight, .. } = &attention.query_keys {
        push(WeightKind::QueryKeys, weight);
    }
    if let AttentionGate::Separate { weight, .. } = &attention.gate {
        push(WeightKind::AttentionGate, weight);
    }
    let own = match fresh_rows(attention) {
        FreshRows::Own(rows) => Some(rows),
        FreshRows::Shared(_) => None,
    };
    if let Some(rows) = own {
        push(WeightKind::Key, &rows.key);
        if let ValueSource::Projected(value) = &rows.value {
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
    if let Some(ProjectedRows {
        key_norm: HeadNorm::Rms(norm),
        ..
    }) = own
    {
        push(WeightKind::KeyNorm, &norm.weight);
    }
    // Every later source's own pooled rows, then each source's own indexer.
    for source in &attention.history {
        if let HistoryRows::Own(RowProducer::Pooled(rows)) = &source.rows {
            push(WeightKind::PooledProjection, &rows.projection);
            if let Pool::SoftmaxGated { gate, position } = &rows.pool {
                push(WeightKind::PooledGate, gate);
                if let Some(position) = position {
                    push(WeightKind::PooledGatePosition, position);
                }
            }
            match &rows.norm {
                PoolNorm::Pooled(norm) => push(WeightKind::PooledNorm, &norm.weight),
                PoolNorm::Token { norm, bias } => {
                    push(WeightKind::PooledNorm, &norm.weight);
                    push(WeightKind::PooledNormBias, bias);
                }
            }
        }
    }
    for source in &attention.history {
        let HistoryReads::Selected(Selection::Own(indexer)) = &source.reads else {
            continue;
        };
        push(WeightKind::IndexerQuery, &indexer.query.projection);
        // Shared keys are their owner's, as are their weights.
        // The key norm's weight, and its bias when it has one.
        let key_norm = match &indexer.keys {
            IndexerKeys::Pooled(keys) => {
                push(WeightKind::IndexerKey, &keys.projection);
                if let Pool::SoftmaxGated { gate, position } = &keys.pool {
                    push(WeightKind::IndexerKeyGate, gate);
                    if let Some(position) = position {
                        push(WeightKind::IndexerKeyGatePosition, position);
                    }
                }
                Some(match &keys.norm {
                    PoolNorm::Pooled(norm) => (&norm.weight, None),
                    PoolNorm::Token { norm, bias } => (&norm.weight, Some(bias)),
                })
            }
            IndexerKeys::Latent {
                projection, norm, ..
            } => {
                push(WeightKind::IndexerKey, projection);
                Some((&norm.weight, None))
            }
            IndexerKeys::Shared { .. } => None,
        };
        if let HeadNorm::Rms(norm) = &indexer.query.norm {
            push(WeightKind::IndexerQueryNorm, &norm.weight);
        }
        if let Some((weight, bias)) = key_norm {
            push(WeightKind::IndexerKeyNorm, weight);
            if let Some(bias) = bias {
                push(WeightKind::IndexerKeyNormBias, bias);
            }
        }
        if let HeadWeights::Projected(weight) = &indexer.head_weights {
            push(WeightKind::IndexerHeadWeights, weight);
        }
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
            inputs: magnitude_family_contracts::grouped_output_columns(
                attention.heads * attention.width,
                *groups,
            )
            .unwrap_or_else(|error| unreachable!("validation splits a grouped output evenly: {error}")),
            rank: *rank,
        }),
    }
}

/// `project_rows`' axes of an absorbed query (`QueryKeys::Absorbed`): each
/// head's projected row through its own block of the weight into the
/// `W`-wide query the entries attend with; none for direct queries.
pub(crate) fn absorbed_query(attention: &Attention) -> Option<AbsorbedQueryShape> {
    match &attention.query_keys {
        QueryKeys::Direct => None,
        QueryKeys::Absorbed { width, .. } => Some(AbsorbedQueryShape {
            heads: attention.heads,
            width: *width,
            keys: attention.width,
        }),
    }
}

/// The static axes of a selected read in a decoder serving `context`
/// tokens; none for a read of every visible row.
pub(crate) fn selection_shape(attention: &Attention, context: u64) -> Option<SelectionShape> {
    admitted_selection(attention).map(|read| read_shape(read, context))
}

fn read_shape(read: SelectedRead<'_>, context: u64) -> SelectionShape {
    SelectionShape {
        block: read.keys.rate,
        // A row at position t < context reads units up to its tail's,
        // t / block: the bitmap covers them all.
        words: context / read.keys.rate / 32 + 1,
    }
}

/// A selected read's numerical parameters apart from its weights: its
/// keys' pooling and norm, and its indexer's.
fn selection_key(attention: &Attention) -> Result<Option<String>, PlanError> {
    selected_read(&attention.fresh().reads)?
        .map(|read| {
            let norm = match &read.keys.norm {
                PoolNorm::Pooled(norm) => format!("pooled {:?}", norm.epsilon),
                PoolNorm::Token { norm, .. } => format!("token {:?}", norm.epsilon),
            };
            let pool = match &read.keys.pool {
                Pool::Mean => "mean".to_string(),
                Pool::SoftmaxGated { position, .. } => format!("gated {}", position.is_some()),
                Pool::Identity => unreachable!("selected_read admits pooling of two rows or more"),
            };
            let query_norm = match &read.indexer.query.norm {
                HeadNorm::Rms(norm) => Some(norm.epsilon),
                HeadNorm::None => None,
            };
            Ok(format!(
                "selected {} {pool} {norm} {:?} {query_norm:?} {}",
                read.keys.rate,
                read.indexer.query.input,
                indexer_key(None, 0, read.indexer)?,
            ))
        })
        .transpose()
}

/// An own indexer's numerical parameters apart from its weights and keys:
/// its heads, picks, scale, query form and the candidates it scores
/// ([`candidates`], resolved at `site`).
fn indexer_key(
    site: Option<DecoderSite<'_>>,
    part: u64,
    indexer: &Indexer,
) -> Result<String, PlanError> {
    Ok(format!(
        "{} {} {} {} {:?} {:?} {:?} {:?}",
        indexer.heads,
        indexer.width,
        indexer.top,
        indexer.scale,
        indexer.query.rounding,
        matches!(indexer.query.norm, HeadNorm::Rms(_)),
        matches!(indexer.head_weights, HeadWeights::Projected(_)),
        candidates(site, part, indexer)?,
    ))
}

/// A second source's numerical parameters apart from its weights: an own
/// source's pooling, rounding, keys and indexer; a Reindex source's rows
/// (whose keys are their owner's) and indexer; or the source a reused one
/// reads (which fixes its rows and picks).
fn picked_key(attention: &Attention, site: Option<DecoderSite<'_>>) -> Result<Option<String>, PlanError> {
    picked_source(attention)?
        .map(|source| {
            Ok(match source {
                PickedSource::Own { part, rows, indexer } => format!(
                    "own {} {} {:?} {:?} {:?} {}",
                    rows.rate,
                    pooled::gated(rows),
                    pooled_norm(rows)?.epsilon,
                    rows.rounding,
                    // [`picked_source`] admits latent keys alone.
                    match &indexer.keys {
                        IndexerKeys::Latent { norm, rounding, .. } => (norm.epsilon, *rounding),
                        IndexerKeys::Pooled(_) | IndexerKeys::Shared { .. } => {
                            unreachable!("picked_source admits an own source's latent keys alone")
                        }
                    },
                    indexer_key(site, u64::from(part), indexer)?,
                ),
                PickedSource::Reindex { source, part, indexer } => format!(
                    "reindex {source:?} {part} {}",
                    indexer_key(site, u64::from(part), indexer)?
                ),
                PickedSource::Reused { source, part, .. } => format!("reused {source:?} {part}"),
            })
        })
        .transpose()
}

/// Every numerical parameter of the operator a sealed graph holds apart from
/// its weights; its second source's candidate pool is resolved at `site`.
pub(super) fn shape_key(
    attention: &Attention,
    epsilon: f64,
    site: Option<DecoderSite<'_>>,
) -> Result<String, PlanError> {
    // A full query or output and a direct query add nothing, so every other
    // form's key is unchanged.
    let low_rank = low_rank_query(attention)
        .map(|(rank, epsilon)| format!(" low-rank query {rank} {epsilon:?}"))
        .unwrap_or_default();
    let grouped = grouped_output(attention)
        .map(|grouped| format!(" grouped output {grouped:?}"))
        .unwrap_or_default();
    let absorbed = absorbed_query(attention)
        .map(|absorbed| format!(" absorbed query {absorbed:?}"))
        .unwrap_or_default();
    Ok(format!(
        "attention {:?} {:?} {:?} {:?} {} {:?} {:?}{low_rank}{grouped}{absorbed}",
        geometry(0, attention)?,
        gate_function(attention),
        head_norm_epsilon(attention, epsilon)?,
        attention.rotary,
        attention.scale,
        selection_key(attention)?,
        picked_key(attention, site)?,
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
    site: Option<DecoderSite<'_>>,
) -> Result<AttentionBinding, PlanError> {
    let shape = shape(hidden, attention, site)?;
    // Every projection binds its second-level scale at its accumulator-scale
    // port (`attention_project`'s QS, GS, KS, VS, `attention_output`'s OS).
    let projection = |kind| scalable(kind).map(|weight| weight.element);
    // Rounded rows are stored in BF16 (`magnitude_state::history_codec`),
    // which the entries read as their activation element.
    if (shape.key_rounding.rounds() || shape.widths.picks > 0)
        && activation != Element::dense(DType::BF16)
    {
        return Err(PlanError::Unsupported(
            "rounded or pooled history rows under an activation other than BF16",
        ));
    }
    // An absorbed query and the latent rows it reads are held in the
    // activation element (precision.md, "Absorbed latent attention"): F16
    // would bound them by its range.
    let absorbed = absorbed_query(attention);
    if absorbed.is_some() && activation == Element::dense(DType::F16) {
        return Err(PlanError::Unsupported(
            "absorbed query keys under an F16 activation",
        ));
    }
    // An own indexer's binding: its query, head weights and candidates.
    let indexer_binding = |part: u32, indexer: &Indexer| {
        let (query_rounding, key_rounding) = indexer_roundings(indexer);
        Ok::<_, PlanError>(IndexerBinding {
            input: indexer.query.input,
            heads: indexer.heads,
            width: indexer.width,
            top: indexer.top,
            query_norm: u64::from(matches!(indexer.query.norm, HeadNorm::Rms(_))),
            query: projection(WeightKind::IndexerQuery)?,
            head_weights: match indexer.head_weights {
                HeadWeights::Uniform => None,
                HeadWeights::Projected(_) => Some(projection(WeightKind::IndexerHeadWeights)?),
            },
            candidates: candidates(site, u64::from(part), indexer)?,
            query_rounding,
            key_rounding,
        })
    };
    let picked = match picked_source(attention)? {
        None => None,
        Some(PickedSource::Reused { .. }) => Some(PickedBinding::Reused),
        Some(PickedSource::Reindex { source, part, indexer }) => Some(PickedBinding::Reindex {
            source: source.block,
            part,
            indexer: indexer_binding(part, indexer)?,
        }),
        Some(PickedSource::Own { part, rows, indexer }) => {
            Some(PickedBinding::Own(PooledBinding {
                part,
                rate: rows.rate,
                projection: projection(WeightKind::PooledProjection)?,
                gate: match pooled::gated(rows) {
                    true => Some(projection(WeightKind::PooledGate)?),
                    false => None,
                },
                norm: lookup(WeightKind::PooledNorm)?,
                rounding: crate::planning::rounding_args(rows.rounding),
                key: projection(WeightKind::IndexerKey)?,
                key_norm: lookup(WeightKind::IndexerKeyNorm)?,
                indexer: indexer_binding(part, indexer)?,
            }))
        }
    };
    let query = projection(query_kind(attention))?;
    let segment = |present: bool, kind| if present { projection(kind) } else { Ok(query) };
    let selection = selected_read(&attention.fresh().reads)?
        .map(|read| {
            let gate = match read.keys.pool {
                Pool::SoftmaxGated { .. } => Some(projection(WeightKind::IndexerKeyGate)?),
                Pool::Mean | Pool::Identity => None,
            };
            Ok::<_, PlanError>(SelectionBinding {
                shape: read_shape(read, context),
                keys: SelectionKeysBinding {
                    projection: projection(WeightKind::IndexerKey)?,
                    gate,
                    place: matches!(read.keys.pool, Pool::SoftmaxGated { position: Some(_), .. }),
                    norm: lookup(WeightKind::IndexerKeyNorm)?,
                    token_norm: matches!(read.keys.norm, PoolNorm::Token { .. }),
                },
                indexer: indexer_binding(0, read.indexer)?,
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
    // The dense entries alone round keys and read an index-list source,
    // and such rows are stored dense whatever the policy (`history_codec`).
    let history = history_codec(attention, history, activation)?;
    if (shape.key_rounding.rounds() || shape.widths.picks > 0)
        && history.encoding() != HistoryEncoding::Dense
    {
        return Err(PlanError::Topology(
            "a rounded or index-list layer's history planned other than dense",
        ));
    }
    Ok(AttentionBinding {
        selection,
        picked,
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
        absorbed: absorbed
            .map(|shape| {
                Ok::<_, PlanError>(AbsorbedQueryBinding {
                    shape,
                    element: projection(WeightKind::QueryKeys)?,
                })
            })
            .transpose()?,
        activation,
        history,
        tail: tail(output, &lookup)?,
    })
}

/// The history codec `attention` reads under the storage policy `history`
/// (`magnitude_state::history_codec`, which state allocation shares).
fn history_codec(
    attention: &Attention,
    history: KvCodec,
    activation: Element,
) -> Result<CodecSpec, PlanError> {
    magnitude_state::history_codec(
        attention,
        history,
        activation
            .dtype()
            .ok_or(PlanError::Unsupported("attention activation dtype"))?,
    )
    .map_err(PlanError::InvalidDefinition)
}

/// The executor admits no Shared layer whose source's values are its stored
/// keys: a Shared layer's history codec is its own-value form
/// (`magnitude_state::history_codec`).
pub(crate) fn admit_shared(decoder: &Decoder) -> Result<(), PlanError> {
    for (_, sublayer) in decoder.sublayers() {
        let Operator::Attention(attention) = &sublayer.op else {
            continue;
        };
        let FreshRows::Shared(source) = fresh_rows(attention) else {
            continue;
        };
        let stored = decoder
            .sublayers()
            .find(|(candidate, _)| *candidate == source)
            .is_some_and(|(_, sublayer)| {
                matches!(
                    &sublayer.op,
                    Operator::Attention(owner) if matches!(
                        fresh_rows(owner),
                        FreshRows::Own(ProjectedRows { value: ValueSource::StoredKey, .. })
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
pub(crate) mod tests {
    use super::*;

    fn pair(frequency: f64) -> RotaryPair {
        RotaryPair {
            frequency,
            amplitude: 1.0,
        }
    }

    pub(crate) mod selected {
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
            ActivationDType, ActivationFunction, Attention, AttentionGate, Block, BlockElement,
            BlockRounding, BlockScale, ImportTransform, SublayerIndex, Decoder,
            DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm, FamilyId, FeedForwardUp,
            HeadNorm, HeadWeights, HistoryReads, HistoryRows, HistorySource, Indexer,
            IndexerInput, IndexerKeys, IndexerQuery, InputNorm, InputSemantics, MediaRowAttention,
            ModelDefinition, Operator, OutputForm, OutputProjection, Pool, PoolNorm, PooledRows,
            ProjectedRows, ResidualForm, RmsNorm, RowProducer, RowRounding, Selection,
            TokenDomain, OutputRotary, QueryKeys, QueryProjection, Rotary, RotaryLayout, RotaryPair,
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
                        query_keys: QueryKeys::Direct,
                        gate: AttentionGate::None,
                        query_norm: HeadNorm::Rms(rms(tensor(name("attn_q_norm"), vec![64]))),
                        history: vec![HistorySource {
                            rows: HistoryRows::Own(RowProducer::Projected(ProjectedRows {
                                key: tensor(name("attn_k"), vec![128, HIDDEN]),
                                value: ValueSource::Projected(tensor(
                                    name("attn_v"),
                                    vec![128, HIDDEN],
                                )),
                                key_norm: HeadNorm::Rms(rms(tensor(
                                    name("attn_k_norm"),
                                    vec![64],
                                ))),
                                value_norm: ValueNorm::None,
                                domain: TokenDomain::Token,
                                rounding: RowRounding::None,
                            })),
                            reads: HistoryReads::Selected(Selection::Own(Box::new(Indexer {
                                heads: 2,
                                width: 64,
                                query: IndexerQuery {
                                    input: IndexerInput::Hidden,
                                    projection: tensor(name("indexer.q_proj"), vec![128, HIDDEN]),
                                    norm: HeadNorm::Rms(rms(tensor(
                                        name("indexer.q_norm"),
                                        vec![64],
                                    ))),
                                    rounding: RowRounding::None,
                                },
                                keys: IndexerKeys::Pooled(PooledRows {
                                    rate: 4,
                                    projection: tensor(name("indexer.k_proj"), vec![64, HIDDEN]),
                                    pool: Pool::Mean,
                                    norm: PoolNorm::Pooled(rms(tensor(name("indexer.k_norm"), vec![64]))),
                                    rounding: RowRounding::None,
                                }),
                                head_weights: HeadWeights::Uniform,
                                scale: 1.0 / 8.0,
                                top: 2,
                                candidates: magnitude_family_contracts::Candidates::All,
                            }))),
                        }],
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
        /// graphs certify on every backend at every row class.
        #[test]
        fn grouped_output_plans_its_grouped_product_and_certifies_on_every_backend() {
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
                certified.unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            }
        }

        /// `selected(true, false)` with GLM's absorbed latent attention in
        /// each block: 4 query heads 32 wide off its low-rank `qr`, each
        /// absorbed into the 64-wide latent row the layer stores once as key
        /// and value, read whole without positions, and a grouped output
        /// whose blocks are the heads' value projections. The per-head
        /// projections are stored as GLM stores them, `attn_k_b` `[4, 64,
        /// 32]` and `attn_v_b` `[4, 32, 64]`, and read with their leading
        /// axes merged.
        fn absorbed() -> (ModelDefinition, PackageManifest) {
            let (mut definition, mut manifest) = selected(true, false);
            let mut offset = manifest.target.files[0].size;
            let tensors = &mut manifest.target.tensors;
            let mut tensor = |name: String, shape: Vec<u64>| {
                let encoding = if shape.len() > 1 { Encoding::BF16 } else { Encoding::F32 };
                let nbytes = source_element(encoding, *shape.last().unwrap())
                    .unwrap()
                    .canonical_byte_len(&shape)
                    .unwrap();
                tensors.push(TensorDescriptor { name: name.clone(), shape: shape.clone(), encoding, offset, nbytes });
                offset += nbytes;
                WeightDescriptor::stored(name, shape)
            };
            fn merged(stack: WeightDescriptor) -> WeightDescriptor {
                WeightDescriptor {
                    shape: vec![stack.shape[0] * stack.shape[1], stack.shape[2]],
                    transforms: vec![ImportTransform::MergeLeadingAxes],
                    ..stack
                }
            }
            for (layer, block) in definition.decoder.blocks.iter_mut().enumerate() {
                let name = |suffix: &str| format!("blk.{layer}.{suffix}");
                let Operator::Attention(attention) = &mut block.sublayers[0].op else {
                    unreachable!("each block opens with attention")
                };
                let QueryProjection::LowRank { down, norm, .. } = attention.query.clone() else {
                    unreachable!("a low-rank query")
                };
                **attention = Attention {
                    heads: 4,
                    kv_heads: 1,
                    width: 128,
                    query: QueryProjection::LowRank {
                        down,
                        norm,
                        up: tensor(name("attn_q_b.heads"), vec![4 * 64, LOW_RANK]),
                    },
                    query_keys: QueryKeys::Absorbed {
                        weight: merged(tensor(name("attn_k_b"), vec![4, 128, 64])),
                        width: 64,
                    },
                    gate: AttentionGate::None,
                    query_norm: HeadNorm::None,
                    history: vec![HistorySource::visible(HistoryRows::Own(RowProducer::Projected(
                        ProjectedRows {
                            key: tensor(name("attn_kv_a_mqa"), vec![128, HIDDEN]),
                            value: ValueSource::StoredKey,
                            key_norm: HeadNorm::Rms(RmsNorm {
                                weight: tensor(name("attn_kv_a_norm"), vec![128]),
                                epsilon: 1e-6,
                            }),
                            value_norm: ValueNorm::None,
                            domain: TokenDomain::Token,
                            rounding: RowRounding::None,
                        },
                    )))],
                    rotary: Rotary::None,
                    scale: 1.0 / 8.0,
                    media_rows: MediaRowAttention::Causal,
                    sinks: None,
                    output: OutputProjection::GroupedLowRank {
                        groups: 4,
                        rank: 64,
                        a: merged(tensor(name("attn_v_b"), vec![4, 64, 128])),
                        b: tensor(name("attn_output.heads"), vec![HIDDEN, 4 * 64]),
                    },
                };
            }
            manifest.target.files[0].size = offset;
            definition.validate().unwrap();
            (definition, manifest)
        }

        /// Absorbed latent attention plans as its query heads at their own
        /// width (`Q` = 4·64 off the rank-64 `qr`), absorbed per head by
        /// `project_rows` into the 128-wide queries its history entries
        /// attend with over one stored-key latent plane, and the grouped
        /// value products. Its graphs certify on every backend at every row
        /// class; on the GPUs its decode classes attend with the prefill
        /// entries. Head width 64 and latent width 128 lie
        /// inside every backend's projection domain (CUDA's projections
        /// reduce over multiples of 64), and the latent is wider than the
        /// head so a transposition of the two shows.
        #[test]
        fn absorbed_latent_attention_plans_and_certifies() {
            let (definition, manifest) = absorbed();
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
                    binding.absorbed.map(|absorbed| absorbed.shape),
                    Some(crate::AbsorbedQueryShape { heads: 4, width: 64, keys: 128 })
                );
                assert_eq!((binding.query_rows(), binding.query_segment_rows()), (4 * 64, LOW_RANK));
                assert_eq!(
                    binding.grouped.map(|grouped| grouped.shape),
                    Some(crate::GroupedOutputShape { groups: 4, inputs: 128, rank: 64 })
                );
                assert_eq!((binding.shape.widths.width, binding.shape.stored_key()), (128, 1));
                assert_eq!(
                    crate::operators::attention::graph::dense_decodes(backend, binding).unwrap(),
                    // CUDA's decode holds no stored key; the others' vector
                    // forms (F32 products) hold this narrow a group.
                    backend != BackendName::Cuda,
                    "{backend:?}"
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
                let certified = checked_target_family_storage(
                    backend,
                    &load,
                    &definition.decoder,
                    &state,
                    plan.target(),
                    limits,
                );
                certified.unwrap_or_else(|error| panic!("{backend:?}: {error}"));
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
                let source = &mut attention.history[0];
                source.reads = HistoryReads::Visible;
                let HistoryRows::Own(RowProducer::Projected(rows)) = &mut source.rows else {
                    unreachable!("each layer owns its history")
                };
                rows.value = ValueSource::StoredKey;
                rows.domain = TokenDomain::Window { tokens: 128 };
            }
            definition.validate().unwrap();
            (definition, manifest)
        }

        /// A stored-key window layer's state holds the key plane alone (two
        /// kv heads of 64 bf16 values: 256 bytes a row, half of K + V), and
        /// its graphs bind that one plane and certify on every backend at
        /// every row class.
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
                certified.unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            }
            // Stored-key rows are the model's own dense store: a K8/V4 host
            // codec leaves them dense.
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
            load.program_plan(&definition, KvCodec::AffineK8V4)
                .expect("stored-key rows plan under a K8/V4 host codec, in their own dense store");
        }

        /// The compressed layer forms of [`compressed`].
        #[derive(Clone, Copy, Debug)]
        pub(crate) enum CompressedForm {
            /// V4.1 slice C's: block 0 a Full layer (its rows pooled per 2
            /// tokens by a softmax gate), block 1 a Reuse layer (block 0's
            /// pooled rows through block 0's picks), each picking 2 rows.
            FullAndReuse,
            /// Block 0's rows are its tokens' own (rate 1) and its indexer
            /// produces a pool of `blocks` blocks of `block_units` units;
            /// block 1 is a Reindex layer (block 0's rows and index keys under
            /// its own query and head weights, restricted to that pool). Each
            /// picks `top` rows.
            Reindexed { blocks: u64, block_units: u64, top: u64 },
            /// V4.1's layers 2 and 8: both blocks are Full layers, each
            /// pooling its own rows per 2 tokens under its own indexer, so the
            /// two sources share the rate's Window and Block domains. Each
            /// picks 32 rows, every block of a context of up to 64 tokens, so
            /// no selection hinges on a near-tie of its scores. Block 0's pool
            /// is softmax-gated (pending rows of twice the width); block 1's is
            /// too when `second_gated`, else a mean (pending rows of the
            /// width): one domain of equal, or of mixed, component widths.
            TwoFull { second_gated: bool },
        }

        /// A Reindexed form whose pool (2 blocks of 2 units) and picks (2)
        /// bind from a few rows on.
        pub(crate) const SMALL_POOL: CompressedForm =
            CompressedForm::Reindexed { blocks: 2, block_units: 2, top: 2 };

        /// V4.1's compressed layer forms (`form`) over `selected(true,
        /// false)`: block 0 a Full layer (its rounded stored-key window of
        /// one key head, then its pooled rows, picked by its own indexer over
        /// `qr` and the pooled latents), block 1 a layer reading block 0's
        /// pooled rows after its own rounded window.
        pub(crate) fn compressed(form: CompressedForm) -> (ModelDefinition, PackageManifest) {
            let (rate, top, produced) = match form {
                CompressedForm::FullAndReuse => (2, 2, None),
                CompressedForm::TwoFull { .. } => (2, 32, None),
                CompressedForm::Reindexed { blocks, block_units, top } => (
                    1,
                    top,
                    Some(magnitude_family_contracts::Candidates::Produce { blocks, block_units }),
                ),
            };
            let (mut definition, mut manifest) = selected(true, false);
            // The selected layers' own key, value and key-norm tensors give
            // way to the window's.
            manifest.target.tensors.retain(|tensor| {
                !["attn_k", "attn_v", "attn_k_norm", "indexer.q_proj", "indexer.q_norm",
                    "indexer.k_proj", "indexer.k_norm"]
                    .iter()
                    .any(|suffix| tensor.name.ends_with(&format!(".{suffix}")))
            });
            let mut offset = manifest.target.files[0].size;
            let tensors = &mut manifest.target.tensors;
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
            let full = SublayerIndex {
                block: 0,
                sublayer: 0,
            };
            for (index, block) in definition.decoder.blocks.iter_mut().enumerate() {
                let name = |suffix: &str| format!("blk.{index}.{suffix}");
                let Operator::Attention(attention) = &mut block.sublayers[0].op else {
                    unreachable!("each block opens with attention")
                };
                attention.kv_heads = 1;
                let window = HistorySource::visible(HistoryRows::Own(RowProducer::Projected(
                    ProjectedRows {
                        key: tensor(name("attn_kv"), vec![64, HIDDEN]),
                        value: ValueSource::StoredKey,
                        key_norm: HeadNorm::Rms(rms(tensor(name("attn_kv_norm"), vec![64]))),
                        value_norm: ValueNorm::None,
                        domain: TokenDomain::Window { tokens: 128 },
                        rounding: RowRounding::Block(BlockRounding {
                            element: BlockElement::E4M3,
                            block: 64,
                            scale: BlockScale::Ue8m0,
                            amax_floor: 1e-4,
                        }),
                    },
                )));
                let pooled = if index == 0 || matches!(form, CompressedForm::TwoFull { .. }) {
                    HistorySource {
                        rows: HistoryRows::Own(RowProducer::Pooled(PooledRows {
                            rate,
                            projection: tensor(name("compressor.kv"), vec![64, HIDDEN]),
                            pool: match form {
                                CompressedForm::Reindexed { .. } => Pool::Identity,
                                CompressedForm::TwoFull { second_gated: false } if index == 1 => Pool::Mean,
                                CompressedForm::FullAndReuse | CompressedForm::TwoFull { .. } => Pool::SoftmaxGated {
                                    gate: tensor(name("compressor.gate"), vec![64, HIDDEN]),
                                    position: None,
                                },
                            },
                            norm: PoolNorm::Pooled(rms(tensor(name("compressor.norm"), vec![64]))),
                            rounding: RowRounding::None,
                        })),
                        reads: HistoryReads::Selected(Selection::Own(Box::new(Indexer {
                            heads: 2,
                            width: 64,
                            query: IndexerQuery {
                                input: IndexerInput::QueryRank,
                                projection: tensor(name("indexer.q_b"), vec![128, LOW_RANK]),
                                norm: HeadNorm::None,
                                rounding: RowRounding::None,
                            },
                            keys: IndexerKeys::Latent {
                                projection: tensor(name("indexer.k"), vec![64, 64]),
                                norm: rms(tensor(name("indexer.k_norm"), vec![64])),
                                rounding: RowRounding::None,
                            },
                            head_weights: HeadWeights::Projected(tensor(
                                name("indexer.w"),
                                vec![2, HIDDEN],
                            )),
                            scale: 1.0 / 8.0,
                            top,
                            candidates: produced
                                .clone()
                                .unwrap_or(magnitude_family_contracts::Candidates::All),
                        }))),
                    }
                } else if matches!(form, CompressedForm::Reindexed { .. }) {
                    HistorySource {
                        rows: HistoryRows::Shared {
                            source: full,
                            part: 1,
                        },
                        reads: HistoryReads::Selected(Selection::Own(Box::new(Indexer {
                            heads: 2,
                            width: 64,
                            query: IndexerQuery {
                                input: IndexerInput::QueryRank,
                                projection: tensor(name("indexer.q_b"), vec![128, LOW_RANK]),
                                norm: HeadNorm::None,
                                rounding: RowRounding::None,
                            },
                            keys: IndexerKeys::Shared { source: full, part: 1 },
                            head_weights: HeadWeights::Projected(tensor(
                                name("indexer.w"),
                                vec![2, HIDDEN],
                            )),
                            scale: 1.0 / 8.0,
                            top,
                            candidates: magnitude_family_contracts::Candidates::Restrict {
                                source: full,
                                part: 1,
                            },
                        }))),
                    }
                } else {
                    HistorySource {
                        rows: HistoryRows::Shared {
                            source: full,
                            part: 1,
                        },
                        reads: HistoryReads::Selected(Selection::Reuse {
                            source: full,
                            part: 1,
                        }),
                    }
                };
                attention.history = vec![window, pooled];
            }
            manifest.target.files[0].size = offset;
            definition.validate().unwrap();
            (definition, manifest)
        }

        /// A pooled source's rows gated by place are refused by name: the
        /// entries pool a source's rows by their gate scores alone, a place
        /// term only an indexer's pooled keys (GLM-5.3's).
        #[test]
        fn a_pooled_source_gated_by_place_is_refused() {
            use crate::{PlanError, operators::attention::admit};
            let (definition, _) = compressed(CompressedForm::FullAndReuse);
            let Operator::Attention(attention) = &definition.decoder.blocks[0].sublayers[0].op else {
                unreachable!("each block opens with attention")
            };
            let mut placed = (**attention).clone();
            let HistoryRows::Own(RowProducer::Pooled(rows)) = &mut placed.history[1].rows else {
                unreachable!("the Full layer pools its own rows")
            };
            let Pool::SoftmaxGated { position, .. } = &mut rows.pool else {
                unreachable!("the Full layer's pool is gated")
            };
            *position = Some(WeightDescriptor::stored("blk.0.compressor.ape", vec![rows.rate, 64]));
            admit(attention, 1e-6).unwrap();
            match admit(&placed, 1e-6) {
                Err(PlanError::Unsupported(reason)) => assert_eq!(reason, "a pooled source's rows gated by place"),
                other => panic!("admitted as {other:?}"),
            }
        }

        /// Slice C's Full and Reuse layers plan their rounded keys (`KR`)
        /// and two picks a row (`KS`), the Full layer its own pooled source
        /// and the Reuse layer its source's, and their graphs (the pooling,
        /// the indexer, the shared picks table) certify on the CPU, Metal
        /// and CUDA at every row class.
        #[test]
        fn compressed_full_and_reuse_layers_plan_and_certify_on_cpu_metal_and_cuda() {
            let (definition, manifest) = compressed(CompressedForm::FullAndReuse);
            let limits = limits();
            for backend in [BackendName::Cpu, BackendName::Metal, BackendName::Cuda] {
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
                let bindings = plan
                    .target()
                    .blocks()
                    .iter()
                    .map(|block| match block.mixer() {
                        MixerProgramSlot::Attention(binding) => binding,
                        _ => panic!("an attention block"),
                    })
                    .collect::<Vec<_>>();
                for binding in &bindings {
                    assert!(binding.shape.key_rounding.rounds());
                    assert_eq!(binding.shape.widths.picks, 2);
                    assert_eq!(binding.shape.stored_key(), 1);
                }
                assert!(matches!(
                    bindings[0].picked,
                    Some(crate::PickedBinding::Own(crate::PooledBinding {
                        part: 1,
                        rate: 2,
                        gate: Some(_),
                        ..
                    }))
                ));
                assert_eq!(bindings[1].picked, Some(crate::PickedBinding::Reused));
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
                let (window, blocks) = state.target_state().pooled_history(0, 1).unwrap();
                assert!(window.is_some());
                assert!(blocks.rows as u64 >= CONTEXT / 2);
                assert!(state.target_state().pooled_history(1, 1).is_none());
                checked_target_family_storage(
                    backend,
                    &load,
                    &definition.decoder,
                    &state,
                    plan.target(),
                    limits,
                )
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            }
        }

        /// A rate-1 Full layer producing a candidate pool and a Reindex
        /// layer restricted to it bind their indexers (the Full layer its
        /// own ungated rows and keys, the pool listed for the Reindex
        /// layer's top; the Reindex layer block 0's rows and keys), the
        /// rate-1 rows keep no Window domain, and their graphs (the
        /// identity pooling without a window, the pool, the restricted
        /// indexer over its table) certify on the CPU, Metal and CUDA.
        #[test]
        fn identity_rows_and_a_reindex_layer_plan_and_certify_on_cpu_metal_and_cuda() {
            let (definition, manifest) = compressed(SMALL_POOL);
            let limits = limits();
            let pool = crate::CandidatePool {
                blocks: 2,
                block_units: 2,
                top: 2,
            };
            for backend in [BackendName::Cpu, BackendName::Metal, BackendName::Cuda] {
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
                let bindings = plan
                    .target()
                    .blocks()
                    .iter()
                    .map(|block| match block.mixer() {
                        MixerProgramSlot::Attention(binding) => binding,
                        _ => panic!("an attention block"),
                    })
                    .collect::<Vec<_>>();
                assert!(bindings.iter().all(|binding| binding.shape.widths.picks == 2));
                assert!(matches!(
                    bindings[0].picked,
                    Some(crate::PickedBinding::Own(crate::PooledBinding {
                        part: 1,
                        rate: 1,
                        gate: None,
                        indexer: crate::IndexerBinding {
                            candidates: crate::CandidatesBinding::Produce(produced),
                            ..
                        },
                        ..
                    })) if produced == pool
                ));
                assert!(matches!(
                    bindings[1].picked,
                    Some(crate::PickedBinding::Reindex {
                        source: 0,
                        part: 1,
                        indexer: crate::IndexerBinding {
                            top: 2,
                            candidates: crate::CandidatesBinding::Restrict(restricted),
                            ..
                        },
                    }) if restricted == pool
                ));
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
                let (window, blocks) = state.target_state().pooled_history(0, 1).unwrap();
                assert!(window.is_none());
                assert!(blocks.rows as u64 >= CONTEXT);
                assert!(state.target_state().pooled_history(1, 1).is_none());
                checked_target_family_storage(
                    backend,
                    &load,
                    &definition.decoder,
                    &state,
                    plan.target(),
                    limits,
                )
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
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
            let programs = crate::AttestedPrograms::prepare_draft(
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
            let programs = crate::AttestedPrograms::prepare_draft(
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
                            block: 4,
                            // Units [0, 1024] of a 4096-token context.
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
                                .any(|component| {
                                    component.layer == LayerRef::IndexKeys { block, part: 0 }
                                })
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
            HistorySource, OutputProjection, OutputRotary, QueryKeys, QueryProjection, RmsNorm, RotaryLayout,
            RotaryPairing, RotaryPlacement, WeightDescriptor,
        };
        let weight = |name: &str, shape: &[u64]| WeightDescriptor::stored(name, shape);
        let base = Attention {
            heads: 2,
            kv_heads: 1,
            width: 64,
            query: QueryProjection::Full(weight("q", &[128, 32])),
            query_keys: QueryKeys::Direct,
            gate: AttentionGate::None,
            query_norm: HeadNorm::None,
            history: vec![HistorySource::visible(HistoryRows::Own(RowProducer::Projected(
                ProjectedRows {
                    key: weight("k", &[64, 32]),
                    value: ValueSource::RawKey,
                    key_norm: HeadNorm::None,
                    value_norm: ValueNorm::None,
                    domain: TokenDomain::Window { tokens: 128 },
                    rounding: RowRounding::None,
                },
            )))],
            rotary: Rotary::Table {
                pairs: vec![pair(1.0); 8],
                layout: RotaryLayout::HALF_SPLIT_LEADING,
                divisors: None,
                output: OutputRotary::None,
            },
            scale: 0.125,
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
        let HistoryRows::Own(RowProducer::Projected(rows)) = &mut v41.history[0].rows else {
            unreachable!()
        };
        rows.key_norm = rms("k_norm");
        rows.value = ValueSource::StoredKey;
        admit(&v41, 1e-6).unwrap();
        let axes = geometry(0, &v41).unwrap();
        assert_eq!(
            (
                axes.form.query_norm,
                axes.form.key_norm,
                axes.form.rotary_adjacent,
                axes.form.rotary_trailing,
                axes.form.output_rotary,
                axes.form.stored_key(),
                axes.value_rows(),
            ),
            (false, true, true, true, true, true, 0)
        );
        // Its history is the key plane alone, dense only.
        let dense = history_codec(&v41, KvCodec::Dense, Element::bf16()).unwrap();
        assert_eq!(dense.value, magnitude_state::HistoryValue::Key);
        assert_eq!(dense.planes().unwrap().len(), 1);
        assert_eq!(history_codec(&v41, KvCodec::AffineK8V4, Element::bf16()).unwrap(), dense);
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
        selected.history[0].reads = HistoryReads::Selected(Selection::Own(Box::new(
            magnitude_family_contracts::Indexer {
                heads: 2,
                width: 64,
                query: magnitude_family_contracts::IndexerQuery {
                    input: IndexerInput::Hidden,
                    projection: weight("indexer.q", &[128, 32]),
                    norm: HeadNorm::Rms(RmsNorm {
                        weight: weight("indexer.q_norm", &[64]),
                        epsilon: 1e-6,
                    }),
                    rounding: RowRounding::None,
                },
                keys: IndexerKeys::Pooled(PooledRows {
                    rate: 4,
                    projection: weight("indexer.k", &[64, 32]),
                    pool: Pool::Mean,
                    norm: PoolNorm::Pooled(RmsNorm {
                        weight: weight("indexer.k_norm", &[64]),
                        epsilon: 1e-6,
                    }),
                    rounding: RowRounding::None,
                }),
                head_weights: HeadWeights::Uniform,
                scale: 0.125,
                top: 2,
                candidates: magnitude_family_contracts::Candidates::All,
            },
        )));
        // The indexer rotates by the attention's table in its layout
        // (`pool_rows`, `indexer_score`).
        admit(&selected, 1e-6).unwrap();
        // GLM-5.3-Flash's keys, normed per token with a bias and gated by
        // place, are admitted; a candidate pool over token rows and
        // rounded keys are refused by name.
        fn indexer(selected: &mut Attention) -> &mut Indexer {
            let HistoryReads::Selected(Selection::Own(indexer)) = &mut selected.history[0].reads else {
                unreachable!()
            };
            indexer
        }
        fn keys(selected: &mut Attention) -> &mut PooledRows {
            let IndexerKeys::Pooled(keys) = &mut indexer(selected).keys else { unreachable!() };
            keys
        }
        let mut glm = selected.clone();
        keys(&mut glm).norm = PoolNorm::Token {
            norm: magnitude_family_contracts::LayerNorm {
                weight: weight("indexer.k_norm", &[64]),
                epsilon: 1e-6,
            },
            bias: weight("indexer.k_norm_bias", &[64]),
        };
        keys(&mut glm).pool = Pool::SoftmaxGated {
            gate: weight("indexer.gate", &[64, 32]),
            position: Some(weight("indexer.ape", &[4, 64])),
        };
        admit(&glm, 1e-6).unwrap();
        let mut pooled = selected.clone();
        indexer(&mut pooled).candidates = magnitude_family_contracts::Candidates::Produce {
            blocks: 2,
            block_units: 2,
        };
        let mut rounded = selected.clone();
        keys(&mut rounded).rounding = magnitude_family_contracts::RowRounding::Block(
            magnitude_family_contracts::BlockRounding {
                element: magnitude_family_contracts::BlockElement::E2M1,
                block: 16,
                scale: magnitude_family_contracts::BlockScale::E4M3,
                amax_floor: 6.0 * 2f64.powi(-9),
            },
        );
        for (refused, expected) in [
            (pooled, "a candidate pool over token rows"),
            (rounded, "rounded pooled indexer keys"),
        ] {
            match admit(&refused, 1e-6) {
                Err(PlanError::Unsupported(reason)) => assert_eq!(reason, expected),
                other => panic!("admitted as {other:?}"),
            }
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
                if let HistoryRows::Own(RowProducer::Projected(rows)) =
                    &mut attention.history[0].rows
                {
                    rows.value = ValueSource::StoredKey;
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
