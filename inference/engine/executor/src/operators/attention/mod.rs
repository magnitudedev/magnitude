//! Softmax attention over history spans (`Operator::Attention`): the forms
//! the attention entries implement, their static axes, weight roles and
//! program binding.

pub(crate) mod graph;

use super::{tail, PlanError, WeightPush};
use crate::{AttentionBinding, AttentionShape};
use magnitude_family_contracts::{
    Attention, AttentionGate, Decoder, GateFunction, GateGranularity, HeadNorm, HistoryDomain,
    HistoryReads, KeyValue, MediaRowAttention, Operator, OutputForm, Rotary, RotaryPair,
    ValueNorm, ValueSource, WeightKind,
};
use magnitude_state::KvCodec;
use seismic::Element;

/// The attention forms the attention entries implement: any output gate,
/// query and key norms both present or both absent, projected values or
/// values from the raw key, an optional unweighted value norm, one norm
/// epsilon, owned histories in a token or window domain or a Shared layer's
/// source history, and any rotary table whose stored divisors (checked at
/// load, `check_rotary_divisors`) cover its pairs. Media rows attend
/// causally or bidirectionally (`operators::media_rows_bidirectional`).
pub(crate) fn admit(attention: &Attention, input_epsilon: f64) -> Result<(), PlanError> {
    // A Shared layer reads its source's history (`StateStore::history_read`)
    // and projects no keys or values; its source's domain is admitted with
    // the source.
    if let KeyValue::Owned {
        key_norm, domain, ..
    } = &attention.key_value
    {
        if matches!(attention.query_norm, HeadNorm::None) != matches!(key_norm, HeadNorm::None) {
            return Err(PlanError::Unsupported(
                "attention with only one of query and key norms",
            ));
        }
        if !matches!(domain, HistoryDomain::Token | HistoryDomain::Window { .. }) {
            return Err(PlanError::Unsupported("attention history domain"));
        }
    }
    head_norm_epsilon(attention, input_epsilon)?;
    if let Rotary::Table {
        pairs,
        divisors: Some(divisors),
    } = &attention.rotary
    {
        if divisors.bases.len() != pairs.len() {
            return Err(PlanError::Topology(
                "rotary divisor bases disagree with the table's pairs",
            ));
        }
    }
    if attention.reads != HistoryReads::Visible {
        return Err(PlanError::Unsupported("attention history reads"));
    }
    let shape = shape(0, attention)?;
    if shape.width < 2 * shape.rotary_pairs {
        return Err(PlanError::Topology("rotary pairs exceed the head width"));
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
            PlanError::Unsupported("attention layers of one history attending media rows differently")
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
                    return Err(PlanError::Topology("a Shared layer's source is not attention"));
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
    let (fresh, value_norm, projected_value) = match &attention.key_value {
        KeyValue::Owned {
            value, value_norm, ..
        } => (
            1,
            u64::from(matches!(value_norm, ValueNorm::RmsUnweighted(_))),
            matches!(value, ValueSource::Projected(_)),
        ),
        KeyValue::Shared { .. } => (0, 0, false),
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
        head_norm: u64::from(matches!(attention.query_norm, HeadNorm::Rms(_))),
        value_norm,
        projected_value,
    })
}

/// The attention entries' `gate_function` code: 0 sigmoid, 1 softplus (and
/// 0 for an ungated operator, whose gate is never read).
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
    }
}

/// The one epsilon of an attention operator's query, key and value norms,
/// which the attention entries take as one argument. An operator without
/// head norms uses its input norm's (never read).
pub(crate) fn head_norm_epsilon(attention: &Attention, input_epsilon: f64) -> Result<f64, PlanError> {
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

/// The weights the operator binds, in the order its entries consume them.
pub(super) fn weights<'a>(attention: &'a Attention, push: &mut WeightPush<'_, 'a>) {
    push(query_kind(attention), &attention.query);
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
    push(WeightKind::AttentionOutput, &attention.output);
}

/// Every numerical parameter of the operator a sealed graph holds apart from
/// its weights.
pub(super) fn shape_key(attention: &Attention, epsilon: f64) -> Result<String, PlanError> {
    Ok(format!(
        "attention {:?} {:?} {:?} {:?} {}",
        shape(0, attention)?,
        gate_function(attention),
        head_norm_epsilon(attention, epsilon)?,
        attention.rotary,
        attention.scale
    ))
}

/// The program binding of an attention sublayer with `output`; `lookup`
/// resolves the planned element of a role in its scope. An absent
/// projection segment binds a zero-row view of the query weight.
pub(super) fn binding(
    attention: &Attention,
    output: &OutputForm,
    hidden: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    activation: Element,
    history: KvCodec,
) -> Result<AttentionBinding, PlanError> {
    let shape = shape(hidden, attention)?;
    let query = lookup(query_kind(attention))?;
    let segment = |present: bool, kind| if present { lookup(kind) } else { Ok(query) };
    Ok(AttentionBinding {
        shape,
        norm: lookup(WeightKind::InputNorm)?,
        query,
        gate: segment(shape.gate_rows() > 0, WeightKind::AttentionGate)?,
        key: segment(shape.key_rows() > 0, WeightKind::Key)?,
        value: segment(shape.value_rows() > 0, WeightKind::Value)?,
        output: lookup(WeightKind::AttentionOutput)?,
        activation,
        history,
        tail: tail(output, &lookup)?,
    })
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
