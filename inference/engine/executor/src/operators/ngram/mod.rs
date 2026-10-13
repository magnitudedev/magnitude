//! The n-gram input sublayer (`Operator::NGramInput`, qwen4exp): rows of a
//! hashed n-gram table gathered per position, gated into each residual
//! stream and convolved over a banked window (`ngram.seismic`). The device
//! evaluates the row rule (`ngram_rows`) from the step's token rows and the
//! token window it carries in its bank, and gathers the rows from the table,
//! a gathered table the plan holds whole or streams (`gathered_row`). The
//! host evaluates the same rule (`NGramHash::rows`) over the tokens it knows
//! to bring a streamed table's rows to their slots before a step reads them.

pub(crate) mod graph;

use super::PlanError;
use super::streams::StreamsGeometry;
use crate::NGramBinding;
use magnitude_family_contracts::{
    Decoder, InputNorm, NGramInput, OutputForm, Sublayer, WeightKind,
};
use seismic::Element;

/// The n-gram input sublayer of a [`super::PairedBlock`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct NGramSublayer<'a> {
    pub op: &'a NGramInput,
}

impl NGramSublayer<'_> {
    /// Its bank components, after its block's mixer's: the window of
    /// convolution inputs, then the token window.
    pub fn bank_components(&self) -> usize {
        2
    }
}

/// The form `ngram_rows`, `ngram_gate` and `ngram_inject` implement: an
/// unnormalized input adding into the streams, with one norm epsilon, whose
/// table rows and reset token are `i32` words (the device's row and token
/// rows).
pub(super) fn admit(sublayer: &Sublayer) -> Result<NGramSublayer<'_>, PlanError> {
    let magnitude_family_contracts::Operator::NGramInput(op) = &sublayer.op else {
        return Err(super::unsupported_operator(&sublayer.op, "n-gram input"));
    };
    let words = op
        .table
        .shape
        .first()
        .is_some_and(|rows| i32::try_from(*rows).is_ok())
        && i32::try_from(op.hash.reset).is_ok();
    if sublayer.input != InputNorm::None
        || sublayer.output != OutputForm::Residual
        || op.query_norm.epsilon != op.key_norm.epsilon
        || op.convolution_norm.epsilon != op.key_norm.epsilon
        || !words
    {
        return Err(PlanError::Unsupported("n-gram input sublayer form"));
    }
    Ok(NGramSublayer { op })
}

/// Every numerical parameter of the sublayer a sealed graph holds apart
/// from its weights, the row rule's constants among them.
pub(super) fn shape_key(sublayer: NGramSublayer<'_>) -> String {
    let op = sublayer.op;
    let hash = &op.hash;
    format!(
        "n-gram {} {} {} {} {} {} {:?} {:?} {:?} {}",
        hash.context,
        hash.heads_per_order,
        op.width,
        op.kernel,
        op.dilation,
        op.key_norm.epsilon,
        hash.multipliers,
        hash.sizes,
        hash.offsets,
        hash.reset
    )
}

/// The program binding of an n-gram input sublayer of `decoder`: `lookup`
/// resolves the planned element of a role in its scope, `table` the
/// streamed table's stored element and the element its reads decode as.
pub(crate) fn binding(
    sublayer: NGramSublayer<'_>,
    decoder: &Decoder,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    table: (Element, Element),
    activation: Element,
) -> Result<NGramBinding, PlanError> {
    let op = sublayer.op;
    let streams = StreamsGeometry::of(decoder)?.ok_or(PlanError::Topology(
        "an n-gram input without residual streams",
    ))?;
    let norm = lookup(WeightKind::NGramKeyNorm)?;
    if lookup(WeightKind::NGramQueryNorm)? != norm
        || lookup(WeightKind::NGramConvolutionNorm)? != norm
        || lookup(WeightKind::NGramConvolution)? != Element::f32()
    {
        return Err(PlanError::Unsupported(
            "n-gram norms of differing representations or a non-F32 convolution",
        ));
    }
    Ok(NGramBinding {
        hidden: decoder.hidden,
        streams: streams.streams,
        context: op.hash.context,
        heads_per_order: op.hash.heads_per_order,
        width: op.width,
        kernel: op.kernel,
        dilation: op.dilation,
        table_source: table.0,
        table: table.1,
        key: lookup(WeightKind::NGramKey)?,
        value: lookup(WeightKind::NGramValue)?,
        norm,
        activation,
    })
}
