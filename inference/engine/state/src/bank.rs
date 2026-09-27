//! Typed recurrent bank components. A bank holds one conversation's (or
//! checkpoint's) recurrent-layer state: the state published after its
//! committed rows, and a tape recording the rows after it. A speculative
//! prefix of at least the committed rows commits as the version (bank, tape
//! rows): a reader replays the tape rows onto the published state. A bank
//! holds at least one tape row, so its tape is a real component; plain
//! advances never write it.

use crate::{ComponentSpec, Error};
use seismic::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankComponent {
    /// A causal convolution's input window: the `width - 1` input rows
    /// before the published state, then the input rows of the tape,
    /// `[width - 1 + tape, channels]`. Replaying the tape shifts the window.
    /// The gated delta window stores raw projection rows in activation
    /// precision; LFM2's short convolution stores `u = B⊙X` in F32 so the
    /// product is not rounded before its sum.
    ConvWindow {
        width: usize,
        channels: usize,
        dtype: DType,
    },
    /// The gated delta rule state `[heads, width, width]`, F32.
    DeltaState { heads: usize, width: usize },
    /// The gated delta tape: per row the innovations `[value_heads, width]`,
    /// keys `[key_heads, width]` and decays `[value_heads]`, F32. Replay
    /// applies the delta rule row by row.
    DeltaTape {
        value_heads: usize,
        key_heads: usize,
        width: usize,
    },
    /// The Mamba-2 state `[heads, head_width, state_width]`, F32: it
    /// accumulates, and its reference keeps it F32.
    SsmState {
        heads: usize,
        head_width: usize,
        state_width: usize,
    },
    /// The tape of Mamba-2's additive rule `h ← exp(dt·A)·h + dt·x⊗B`: per
    /// row the inputs `x [heads, head_width]`, `B [groups, state_width]` and
    /// the steps `dt [heads]`, F32. `A` is a weight, so replay needs nothing
    /// else.
    SsmTape {
        heads: usize,
        head_width: usize,
        groups: usize,
        state_width: usize,
    },
    /// Compressor rings indexed by position mod `rate` (deferred families).
    /// Interface only: speculation needs a snapshot/commit-after-verify rule
    /// before a store can hold it.
    PositionRing { rate: usize, width: usize },
}

impl BankComponent {
    /// The physical component of a bank that records `tape_rows` rows after
    /// its published state (at least one).
    pub fn spec(self, tape_rows: usize) -> Result<ComponentSpec, Error> {
        let tape = tape_rows.max(1);
        let overflow = || Error::Request("bank component shape overflows".into());
        let (shape, dtype) = match self {
            Self::ConvWindow {
                width,
                channels,
                dtype,
            } => {
                let history = width
                    .checked_sub(1)
                    .ok_or_else(|| Error::Request("convolution width must be positive".into()))?;
                (
                    vec![history.checked_add(tape).ok_or_else(overflow)?, channels],
                    dtype,
                )
            }
            Self::DeltaState { heads, width } => (vec![heads, width, width], DType::F32),
            Self::DeltaTape {
                value_heads,
                key_heads,
                width,
            } => (
                vec![
                    tape,
                    value_heads
                        .checked_add(key_heads)
                        .and_then(|heads| heads.checked_mul(width))
                        .and_then(|vectors| vectors.checked_add(value_heads))
                        .ok_or_else(overflow)?,
                ],
                DType::F32,
            ),
            Self::SsmState {
                heads,
                head_width,
                state_width,
            } => (vec![heads, head_width, state_width], DType::F32),
            Self::SsmTape {
                heads,
                head_width,
                groups,
                state_width,
            } => (
                vec![
                    tape,
                    heads
                        .checked_mul(head_width)
                        .and_then(|x| x.checked_add(groups.checked_mul(state_width)?))
                        .and_then(|row| row.checked_add(heads))
                        .ok_or_else(overflow)?,
                ],
                DType::F32,
            ),
            Self::PositionRing { .. } => return Err(Error::UnsupportedBankComponent(self)),
        };
        let spec = ComponentSpec { shape, dtype };
        spec.bytes().map_err(Error::Request)?;
        Ok(spec)
    }

    /// Bytes of this component in one bank.
    pub fn bytes(self, tape_rows: usize) -> Result<u64, Error> {
        u64::try_from(self.spec(tape_rows)?.bytes().map_err(Error::Request)?)
            .map_err(|_| Error::Request("bank component bytes exceed u64".into()))
    }
}

/// Bytes of one bank: the sum of its components. A bank slab holds as many
/// banks as fit its byte target, at least one.
pub fn recurrent_bank_bytes(components: &[ComponentSpec]) -> Result<u64, Error> {
    components.iter().try_fold(0_u64, |total, spec| {
        total
            .checked_add(
                u64::try_from(spec.bytes().map_err(Error::Request)?)
                    .map_err(|_| Error::Request("recurrent bank bytes exceed u64".into()))?,
            )
            .ok_or_else(|| Error::Request("recurrent bank byte count overflow".into()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_derive_their_shapes_and_bytes() {
        let window = BankComponent::ConvWindow {
            width: 4,
            channels: 64,
            dtype: DType::BF16,
        };
        assert_eq!(window.spec(3).unwrap().shape, [6, 64]);
        // A bank records at least one tape row.
        assert_eq!(window.spec(0).unwrap().shape, [4, 64]);
        let short = BankComponent::ConvWindow {
            width: 3,
            channels: 2048,
            dtype: DType::F32,
        };
        assert_eq!(short.bytes(0).unwrap(), 3 * 2048 * 4);
        assert_eq!(
            BankComponent::DeltaTape {
                value_heads: 4,
                key_heads: 2,
                width: 8,
            }
            .spec(3)
            .unwrap()
            .shape,
            [3, 52]
        );
        let ssm = BankComponent::SsmState {
            heads: 64,
            head_width: 64,
            state_width: 128,
        };
        assert_eq!(ssm.spec(0).unwrap().dtype, DType::F32);
        assert_eq!(ssm.bytes(0).unwrap(), 64 * 64 * 128 * 4);
        // x [4, 16] | B [2, 32] | dt [4] per tape row.
        assert_eq!(
            BankComponent::SsmTape {
                heads: 4,
                head_width: 16,
                groups: 2,
                state_width: 32,
            }
            .spec(2)
            .unwrap()
            .shape,
            [2, 64 + 64 + 4]
        );
        let specs = [window.spec(3).unwrap(), ssm.spec(3).unwrap()];
        assert_eq!(
            recurrent_bank_bytes(&specs).unwrap(),
            window.bytes(3).unwrap() + ssm.bytes(3).unwrap()
        );
    }

    #[test]
    fn position_rings_are_interface_only() {
        let ring = BankComponent::PositionRing { rate: 4, width: 8 };
        assert!(matches!(
            ring.spec(1),
            Err(Error::UnsupportedBankComponent(component)) if component == ring
        ));
    }
}
