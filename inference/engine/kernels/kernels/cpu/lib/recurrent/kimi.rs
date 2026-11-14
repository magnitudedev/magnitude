// Kimi Delta Attention's rule for the CPU delta-rule library (`recurrent`;
// entries `kimi_delta_step` and `kimi_delta_chunk`, contracts and the
// portable `kimi_delta_rows` in recurrent.seismic): beta, z and the decay
// inputs are their own operands, and each key column of a value head has its
// own bounded decay factor. Everything else is the library's.

use super::recurrent::{Recurrence, Rule};
use seismic::cpu::{math, Dense, Tensor};

/// The operands of one KDA call, shared by both entries: the library's
/// operands over the q | k | v rows, and the per-channel inputs.
pub struct Kimi<'a, A: Dense> {
    /// `projection` holds the q | k | v rows [M, CH]; `time_bias` is [NV * W].
    pub recurrence: Recurrence<'a, A>,
    /// beta's input, [M, NV].
    pub beta: Tensor<'a, A, 2>,
    /// z rows, [M, NV * W].
    pub gate: Tensor<'a, A, 2>,
    /// Decay inputs, [M, NV * W].
    pub decay: Tensor<'a, A, 2>,
    /// The bounded-sigmoid decay's lower bound.
    pub lower: f32,
}

/// The `Kimi` a KDA entry binds from its parameters (`recurrent::recurrence`'s
/// counterpart). Expands where `recurrent` and `kimi` name the library's
/// modules.
macro_rules! kimi_rule {
    ($cx:expr) => {
        kimi::Kimi {
            recurrence: recurrent::recurrence!($cx),
            beta: $cx.arg_beta(),
            gate: $cx.arg_gate(),
            decay: $cx.arg_decay(),
            lower: $cx.arg_decay_lower(),
        }
    };
}
pub(crate) use kimi_rule;

impl<'a, A: Dense> Rule<'a> for Kimi<'a, A> {
    type Act = A;

    #[inline(always)]
    fn recurrence(&self) -> &Recurrence<'a, A> {
        &self.recurrence
    }

    #[inline(always)]
    fn decays(&self) -> usize {
        self.recurrence.width
    }

    /// beta = sigmoid(b) of value head `head` at `row`, with its W decay
    /// factors exp(lower * sigmoid(-rate * (a + time_bias))) in `decay`.
    #[inline(always)]
    fn gates(&self, row: usize, head: usize, decay: &mut [f32]) -> f32 {
        let r = &self.recurrence;
        let beta = math::sigmoid(self.beta.get([row, head]));
        let rate = r.rate.get([head]);
        let first = head * r.width;
        for (channel, factor) in (first..first + r.width).zip(decay.iter_mut()) {
            let shifted = self.decay.get([row, channel]) + r.time_bias.get([channel]);
            *factor = math::exp(self.lower / (1.0 + math::exp(rate * shifted)));
        }
        beta
    }

    #[inline(always)]
    fn gate_input(&self, row: usize, head: usize) -> &'a [A::Storage] {
        let w = self.recurrence.width;
        self.gate.span([row, head * w], w)
    }
}
