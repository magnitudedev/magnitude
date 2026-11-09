//! Rotary tables: plain rotary and YaRN fold exactly into each pair's
//! frequency and amplitude. A table is laid out in the checkpoint's own
//! pairing ([`RotaryLayout`]); the half-split (NEOX) pairing over leading
//! dimensions, pair `p` rotating dimensions `(p, p + P)`, is the one most
//! families state, and adjacent checkpoints of those families are permuted
//! into it at import (`half_split_rows`).

use crate::{HeaderError, Metadata};
use magnitude_family_contracts::{
    ImportTransform, OutputRotary, Rotary, RotaryDivisors, RotaryLayout, RotaryPair,
    WeightDescriptor,
};
use std::f64::consts::PI;

/// Rotary over the first `rotated` dimensions of a `width`-wide head in
/// half-split pairs: pair `p` turns at `base^(-2p/rotated)` per position.
pub fn table(rotated: u64, width: u64, base: f64) -> Result<Rotary, HeaderError> {
    laid_out(
        rotated,
        width,
        base,
        RotaryLayout::HALF_SPLIT_LEADING,
        OutputRotary::None,
    )
}

/// [`table`] over the `rotated` dimensions `layout` names, turning the
/// attended row back by `output`.
pub fn laid_out(
    rotated: u64,
    width: u64,
    base: f64,
    layout: RotaryLayout,
    output: OutputRotary,
) -> Result<Rotary, HeaderError> {
    pairs(rotated, width, base, layout, output, |_, frequency| {
        RotaryPair {
            frequency,
            amplitude: 1.0,
        }
    })
}

/// Rotary over a whole `width`-wide head whose first `rotated` pairs turn at
/// the frequencies of the whole width, `base^(-2p/width)`, and whose other
/// pairs have frequency zero (exact pass-through). The checkpoint stores the
/// split only as the per-pair `divisors` (1 for rotated pairs, 1e30 for the
/// rest), `[width/2]` F32, which the importer checks against this table.
pub fn zero_tail(
    width: u64,
    base: f64,
    rotated: u64,
    divisors: WeightDescriptor,
) -> Result<Rotary, HeaderError> {
    let pairs = width / 2;
    if !width.is_multiple_of(2) || rotated == 0 || rotated > pairs || base <= 1.0 {
        return Err(HeaderError::Rotary {
            rotated: 2 * rotated,
            width,
            base,
        });
    }
    let bases: Vec<f64> = (0..pairs)
        .map(|pair| base.powf(-(pair as f64) / pairs as f64))
        .collect();
    Ok(Rotary::Table {
        pairs: bases
            .iter()
            .enumerate()
            .map(|(pair, base)| RotaryPair {
                frequency: if (pair as u64) < rotated { *base } else { 0.0 },
                amplitude: 1.0,
            })
            .collect(),
        layout: RotaryLayout::HALF_SPLIT_LEADING,
        divisors: Some(RotaryDivisors {
            weight: divisors,
            bases,
        }),
        output: OutputRotary::None,
    })
}

/// The one table builder: pair `p` of the `rotated / 2` pairs takes
/// `pair(p, base^(-2p/rotated))`.
fn pairs(
    rotated: u64,
    width: u64,
    base: f64,
    layout: RotaryLayout,
    output: OutputRotary,
    pair: impl Fn(u64, f64) -> RotaryPair,
) -> Result<Rotary, HeaderError> {
    if rotated == 0 || !rotated.is_multiple_of(2) || rotated > width || base <= 1.0 {
        return Err(HeaderError::Rotary {
            rotated,
            width,
            base,
        });
    }
    Ok(Rotary::Table {
        pairs: (0..rotated / 2)
            .map(|p| pair(p, base.powf(-2.0 * p as f64 / rotated as f64)))
            .collect(),
        layout,
        divisors: None,
        output,
    })
}

/// The row permutation that turns rows stored in adjacent rotary pairs
/// `(2i, 2i + 1)` over the first `rotated` of each `width` rows into the
/// pairs `(i, i + rotated/2)`; the remaining rows keep their place.
pub fn half_split_rows(rotated: u64, width: u64) -> ImportTransform {
    let half = rotated / 2;
    ImportTransform::PermuteRows {
        order: (0..half)
            .map(|i| 2 * i)
            .chain((0..half).map(|i| 2 * i + 1))
            .chain(rotated..width)
            .collect(),
    }
}

/// YaRN context extension (Peng et al.) as the model definitions state it:
/// pairs turning slower than `beta_slow` rotations over the original context
/// are interpolated by `1/factor`, pairs faster than `beta_fast` keep their
/// frequency, pairs between blend linearly, and every rotated pair's cosine
/// and sine are scaled by the model's [`YarnAmplitude`].
pub struct Yarn {
    pub factor: f64,
    pub original_context: u64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub amplitude: YarnAmplitude,
}

/// How a YaRN table scales its rotated pairs' cosine and sine: a property
/// of the model's definition, which the header does not state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum YarnAmplitude {
    /// YaRN's magnitude scale, `attention_factor · (1 + 0.1·ln factor)`.
    /// GGUF semantics (coordinator decision, references lane finding 1):
    /// `rope.scaling.yarn_attn_factor` is the `attention_factor`.
    Mscale { attention_factor: f64 },
    /// Unit amplitude: the table rotates without scaling (DeepSeek V4.1,
    /// whose YaRN has no mscale).
    Unit,
}

impl YarnAmplitude {
    /// YaRN's magnitude scale with the header's attention factor
    /// (`rope.scaling.yarn_attn_factor`, 1 when absent).
    pub fn mscale(m: &Metadata) -> Result<Self, HeaderError> {
        let attention_factor = m
            .optional_number("rope.scaling.yarn_attn_factor")?
            .unwrap_or(1.0);
        if attention_factor <= 0.0 {
            return Err(HeaderError::Yarn);
        }
        Ok(Self::Mscale { attention_factor })
    }

    fn of(self, factor: f64) -> f64 {
        match self {
            Self::Mscale { attention_factor } => attention_factor * (1.0 + 0.1 * factor.ln()),
            Self::Unit => 1.0,
        }
    }
}

impl Yarn {
    /// Reads the `rope.scaling.*` frequency keys, the released
    /// configurations' defaults holding for the optional betas, under the
    /// model's `amplitude`.
    pub fn read(m: &Metadata, amplitude: YarnAmplitude) -> Result<Self, HeaderError> {
        let yarn = Self {
            factor: m.number("rope.scaling.factor")?,
            original_context: m.integer("rope.scaling.original_context_length")?,
            beta_fast: m
                .optional_number("rope.scaling.yarn_beta_fast")?
                .unwrap_or(32.0),
            beta_slow: m
                .optional_number("rope.scaling.yarn_beta_slow")?
                .unwrap_or(1.0),
            amplitude,
        };
        if yarn.factor <= 1.0 || yarn.beta_slow <= 0.0 || yarn.beta_fast <= yarn.beta_slow {
            return Err(HeaderError::Yarn);
        }
        Ok(yarn)
    }

    /// The YaRN-scaled [`laid_out`] table.
    pub fn table(
        &self,
        rotated: u64,
        width: u64,
        base: f64,
        layout: RotaryLayout,
        output: OutputRotary,
    ) -> Result<Rotary, HeaderError> {
        let dimensions = rotated as f64;
        // The pair index at which a pair completes `rotations` turns over the
        // original context.
        let correction = |rotations: f64| {
            dimensions * (self.original_context as f64 / (rotations * 2.0 * PI)).ln()
                / (2.0 * base.ln())
        };
        let low = correction(self.beta_fast).floor().max(0.0);
        let high = correction(self.beta_slow).ceil().min(dimensions - 1.0);
        let amplitude = self.amplitude.of(self.factor);
        pairs(rotated, width, base, layout, output, |p, extrapolated| {
            let kept = 1.0 - ((p as f64 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
            RotaryPair {
                frequency: extrapolated / self.factor * (1.0 - kept) + extrapolated * kept,
                amplitude,
            }
        })
    }
}
