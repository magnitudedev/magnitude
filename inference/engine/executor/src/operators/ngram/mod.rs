//! The n-gram input sublayer (`Operator::NGramInput`, qwen4exp and
//! DeepSeek V4.1): rows of a hashed n-gram table gathered per position and
//! gated into each residual stream, then added (`ngram_add`) or convolved
//! over a banked window (`ngram_inject`) (`ngram.seismic`). The device
//! evaluates the row rule (`ngram_rows`) from the step's token rows and the
//! token window it carries in its bank, and gathers the rows from the
//! table, a gathered table the plan holds whole or streams
//! (`gathered_row`). The host evaluates the same rule (`NGramHash::rows`)
//! over the tokens it knows to bring a streamed table's rows to their slots
//! before a step reads them.

pub(crate) mod graph;

use super::PlanError;
use super::streams::StreamsGeometry;
use crate::planning::NGramRowRule;
use crate::{NGramBinding, NGramGateForm, NGramTaps};
use magnitude_family_contracts::{
    Decoder, InputNorm, NGramGate, NGramInput, NGramMedia, OutputForm, Sublayer, WeightKind,
};
use seismic::Element;

/// The n-gram input sublayer of a [`super::PairedBlock`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct NGramSublayer<'a> {
    pub op: &'a NGramInput,
}

impl NGramSublayer<'_> {
    /// Its bank components, after its block's mixer's: under a convolution
    /// its window of convolution inputs, then the token window.
    pub fn bank_components(&self) -> usize {
        1 + usize::from(self.op.mixing.is_some())
    }
}

/// The form `ngram_rows`, `ngram_gate`, `ngram_add` and `ngram_inject`
/// implement: an unnormalized input adding into the streams, whose table
/// rows, fill and boundary are `i32` words (the device's row and token
/// rows).
pub(super) fn admit(sublayer: &Sublayer) -> Result<NGramSublayer<'_>, PlanError> {
    let magnitude_family_contracts::Operator::NGramInput(op) = &sublayer.op else {
        return Err(super::unsupported_operator(&sublayer.op, "n-gram input"));
    };
    let hash = &op.hash;
    let words = op
        .table
        .shape
        .first()
        .is_some_and(|rows| i32::try_from(*rows).is_ok())
        && i32::try_from(hash.fill).is_ok()
        && hash.boundary.is_none_or(|boundary| i32::try_from(boundary).is_ok());
    if sublayer.input != InputNorm::None || sublayer.output != OutputForm::Residual || !words {
        return Err(PlanError::Unsupported("n-gram input sublayer form"));
    }
    Ok(NGramSublayer { op })
}

/// Until vision rows reach the row rule, an n-gram input whose media rows
/// block (`NGramMedia::Blocking`) admits a text-only decoder and refuses one
/// with a vision component by name.
pub(crate) fn admit_media(decoder: &Decoder, vision: bool) -> Result<(), PlanError> {
    let blocking = decoder.sublayers().any(|(_, sublayer)| {
        matches!(
            &sublayer.op,
            magnitude_family_contracts::Operator::NGramInput(input)
                if input.hash.media == NGramMedia::Blocking
        )
    });
    match vision && blocking {
        true => Err(PlanError::Unsupported("media rows blocking an n-gram lookback")),
        false => Ok(()),
    }
}

/// Every numerical parameter of the sublayer a sealed graph holds apart
/// from its weights, the row rule's constants among them.
pub(super) fn shape_key(sublayer: NGramSublayer<'_>) -> String {
    let op = sublayer.op;
    let hash = &op.hash;
    let mixing = op
        .mixing
        .as_ref()
        .map(|mixing| (mixing.kernel, mixing.dilation));
    let gate = match op.gate {
        NGramGate::Norms { .. } => "norms",
        NGramGate::Product(_) => "product",
    };
    // A vocabulary-long map keys by its digest.
    let map = hash.token_map.as_ref().map(|map| {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(map.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<_>>());
        digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>()
    });
    format!(
        "n-gram {} {} {} {mixing:?} {gate} {} {:?} {:?} {:?} {:?} {} {:?} {:?}",
        hash.context,
        hash.heads_per_order,
        op.width,
        op.epsilon,
        hash.multipliers,
        hash.sizes,
        hash.offsets,
        map,
        hash.fill,
        hash.boundary,
        hash.media,
    )
}

/// The program binding of an n-gram input sublayer of `decoder`: `lookup`
/// resolves the planned element of a role in its scope, `table` the
/// resident element of the stored table's slots.
pub(crate) fn binding(
    sublayer: NGramSublayer<'_>,
    decoder: &Decoder,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    table: Element,
    activation: Element,
) -> Result<NGramBinding, PlanError> {
    let op = sublayer.op;
    let streams = StreamsGeometry::of(decoder)?.ok_or(PlanError::Topology(
        "an n-gram input without residual streams",
    ))?;
    // The gate's weights and the convolution's norm share one element.
    let (gate, norm) = match &op.gate {
        NGramGate::Norms { .. } => {
            let norm = lookup(WeightKind::NGramKeyNorm)?;
            if lookup(WeightKind::NGramQueryNorm)? != norm {
                return Err(PlanError::Unsupported("n-gram norms of differing representations"));
            }
            (NGramGateForm::Norms, norm)
        }
        NGramGate::Product(_) => (NGramGateForm::Product, lookup(WeightKind::NGramGate)?),
    };
    let convolution = match &op.mixing {
        None => None,
        Some(mixing) => {
            if lookup(WeightKind::NGramConvolutionNorm)? != norm
                || lookup(WeightKind::NGramConvolution)? != Element::f32()
            {
                return Err(PlanError::Unsupported(
                    "n-gram norms of differing representations or a non-F32 convolution",
                ));
            }
            Some(NGramTaps {
                kernel: mixing.kernel,
                dilation: mixing.dilation,
            })
        }
    };
    Ok(NGramBinding {
        hidden: decoder.hidden,
        streams: streams.streams,
        rule: NGramRowRule::of(&op.hash),
        width: op.width,
        gate,
        convolution,
        table,
        key: lookup(WeightKind::NGramKey)?,
        value: lookup(WeightKind::NGramValue)?,
        norm,
        activation,
    })
}
