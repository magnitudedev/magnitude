//! Rotary tables in the engine's half-split (NEOX) pairing: pair `p` rotates
//! dimensions `(p, p + P)`. Plain rotary and YaRN fold exactly into each
//! pair's frequency and amplitude; a checkpoint that pairs adjacent
//! dimensions is permuted into this pairing at import.

use crate::{HeaderError, Metadata};
use magnitude_family_contracts::{ImportTransform, Rotary, RotaryPair};
use std::f64::consts::PI;

/// Rotary over the first `rotated` dimensions of a `width`-wide head: pair
/// `p` turns at `base^(-2p/rotated)` per position.
pub fn table(rotated: u64, width: u64, base: f64) -> Result<Rotary, HeaderError> {
    pairs(rotated, width, base, |_, frequency| RotaryPair {
        frequency,
        amplitude: 1.0,
    })
}

fn pairs(
    rotated: u64,
    width: u64,
    base: f64,
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
        divisors: None,
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
/// and sine are scaled by `attention_factor · (1 + 0.1·ln factor)`.
///
/// GGUF semantics (coordinator decision, references lane finding 1):
/// `rope.scaling.yarn_attn_factor` multiplies the YaRN magnitude scale.
pub struct Yarn {
    pub factor: f64,
    pub original_context: u64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub attention_factor: f64,
}

impl Yarn {
    /// Reads the `rope.scaling.*` keys; the released configurations'
    /// defaults hold for the optional terms.
    pub fn read(m: &Metadata) -> Result<Self, HeaderError> {
        let yarn = Self {
            factor: m.number("rope.scaling.factor")?,
            original_context: m.integer("rope.scaling.original_context_length")?,
            beta_fast: m
                .optional_number("rope.scaling.yarn_beta_fast")?
                .unwrap_or(32.0),
            beta_slow: m
                .optional_number("rope.scaling.yarn_beta_slow")?
                .unwrap_or(1.0),
            attention_factor: m
                .optional_number("rope.scaling.yarn_attn_factor")?
                .unwrap_or(1.0),
        };
        if yarn.factor <= 1.0
            || yarn.beta_slow <= 0.0
            || yarn.beta_fast <= yarn.beta_slow
            || yarn.attention_factor <= 0.0
        {
            return Err(HeaderError::Yarn);
        }
        Ok(yarn)
    }

    /// The YaRN-scaled [`table`].
    pub fn table(&self, rotated: u64, width: u64, base: f64) -> Result<Rotary, HeaderError> {
        let dimensions = rotated as f64;
        // The pair index at which a pair completes `rotations` turns over the
        // original context.
        let correction = |rotations: f64| {
            dimensions * (self.original_context as f64 / (rotations * 2.0 * PI)).ln()
                / (2.0 * base.ln())
        };
        let low = correction(self.beta_fast).floor().max(0.0);
        let high = correction(self.beta_slow).ceil().min(dimensions - 1.0);
        let amplitude = self.attention_factor * (1.0 + 0.1 * self.factor.ln());
        pairs(rotated, width, base, |p, extrapolated| {
            let kept = 1.0 - ((p as f64 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
            RotaryPair {
                frequency: extrapolated / self.factor * (1.0 - kept) + extrapolated * kept,
                amplitude,
            }
        })
    }
}
