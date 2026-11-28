//! Block Attention Residuals (`ResidualForm::BlockAttention`) static schedule
//! derivation (`BlockSchedule`, `BlockAttnResPlan`).
//!
//! All residual sources (`blocks`, `partial`, `prelude`, `bank`) stay in F32
//! to prevent correlated rounding error accumulation across looped core passes.
//! Live execution is owned by `programs::native_target_graph::block_attn_res`.

use crate::error::PlanError;
use magnitude_family_contracts::{
    BlockAttentionInput, BlockTransition, CrossPassInput, Decoder, InputNorm, OutputForm,
    ResidualForm,
};

/// Static per-block source-stack schedule derived from `decoder.blocks`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockSchedule {
    /// Sources `[blocks..., partial]` entering this block (`>= 1`).
    pub sources_in: usize,
    /// Sources `[blocks..., partial]` after this block's attention (`sources_in + 1` on `BlockCommit`).
    pub sources_mid: usize,
    /// Cross-pass bank source count (0 when `cross_pass` is `None`).
    pub bank_sources: usize,
    /// Optional prelude/bank transition at the start of this block.
    pub transition: Option<BlockTransition>,
}

/// Complete static schedule for a `ResidualForm::BlockAttention` decoder.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BlockAttnResPlan {
    pub block_size: u64,
    pub normalize_values: bool,
    pub norm_epsilon: f64,
    pub blocks: Vec<BlockSchedule>,
    pub max_sources: usize,
    pub max_bank: usize,
    pub final_sources: usize,
}

impl BlockAttnResPlan {
    /// Derive and validate the static source-stack schedule from `decoder`.
    pub(crate) fn of(decoder: &Decoder) -> Result<Self, PlanError> {
        let ResidualForm::BlockAttention {
            block_size,
            normalize_values,
            norm_epsilon,
        } = decoder.residual
        else {
            return Err(PlanError::Topology(
                "BlockAttnResPlan requires ResidualForm::BlockAttention",
            ));
        };
        let mut sources = 1usize;
        let mut prelude_sources: Option<usize> = None;
        let mut bank_sources = 0usize;
        let mut max_sources = 1usize;
        let mut max_bank = 0usize;
        let mut blocks = Vec::with_capacity(decoder.blocks.len());

        for block in &decoder.blocks {
            match block.transition {
                None => {}
                Some(BlockTransition::SnapshotPrelude) => {
                    prelude_sources = Some(sources);
                }
                Some(BlockTransition::ResetToPreludeAndBank) => {
                    let prev_prelude = prelude_sources.ok_or(PlanError::Topology(
                        "ResetToPreludeAndBank without a prior SnapshotPrelude",
                    ))?;
                    let core_blocks = sources.checked_sub(prev_prelude).ok_or(
                        PlanError::Topology("ResetToPreludeAndBank has no core blocks for bank"),
                    )?;
                    if core_blocks == 0 {
                        return Err(PlanError::Topology(
                            "ResetToPreludeAndBank produced an empty cross-pass bank",
                        ));
                    }
                    bank_sources = core_blocks;
                    max_bank = max_bank.max(bank_sources);
                    sources = prev_prelude;
                }
            }

            let sources_in = sources;
            let attn = block
                .sublayers
                .first()
                .ok_or(PlanError::Topology("block-attention block has no sublayers"))?;
            let sources_mid = match attn.output {
                OutputForm::BlockCommit => sources_in + 1,
                OutputForm::Residual => sources_in,
                _ => {
                    return Err(PlanError::Unsupported(
                        "block-attention attention output other than Residual or BlockCommit",
                    ));
                }
            };
            max_sources = max_sources.max(sources_mid);

            let uses_cross_pass = block.sublayers.iter().any(|s| {
                matches!(
                    &s.input,
                    InputNorm::BlockAttention(BlockAttentionInput {
                        cross_pass: CrossPassInput::Single(_) | CrossPassInput::SplitQkv { .. },
                        ..
                    })
                )
            });
            if uses_cross_pass && bank_sources == 0 {
                return Err(PlanError::Topology(
                    "CrossPassInput used before ResetToPreludeAndBank built a bank",
                ));
            }

            blocks.push(BlockSchedule {
                sources_in,
                sources_mid,
                bank_sources: if uses_cross_pass { bank_sources } else { 0 },
                transition: block.transition,
            });
            sources = sources_mid;
        }

        Ok(Self {
            block_size,
            normalize_values,
            norm_epsilon,
            blocks,
            max_sources,
            max_bank,
            final_sources: sources,
        })
    }
}
