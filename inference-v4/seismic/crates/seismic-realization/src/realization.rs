//! Shared physical realization contracts.
//!
//! The old `InvocationConditions` alias reconstruction is deleted: the root
//! `ResolvedAbi` carries the alias rules, and the runtime validates actual
//! byte ranges against them. This module owns only the encoding helpers the
//! runtime and backends share; it contains no second execution IR.

pub mod dispatch;
pub mod executable;
pub mod numerics;

pub use numerics::{
    compose, compose_all, default_tolerance, satisfies_policy, unit_roundoff,
    AssignmentFingerprint, CapabilitySignatureId, CountExpr, EvidenceKey, NumericalEvidence,
    NumericalTransfer, PolicyDecision, ReductionTopology, ToolchainId, WorkloadFingerprint,
};

pub use executable::{
    AliasRule, BufferBinding, BufferBindingId, ExecutableDialect, ResolvedAbi, ResolvedPlan,
    ResultBinding,
};

use seismic_lang::abi::{ScalarLayout, ScalarParameter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathFunction {
    Exp,
    ExpFast,
    Log,
    Sin,
    Cos,
}

impl MathFunction {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Exp | Self::ExpFast => "seismic_exp",
            Self::Log => "seismic_log",
            Self::Sin => "seismic_sin",
            Self::Cos => "seismic_cos",
        }
    }
}

pub fn encode_scalars(schema: &[ScalarParameter], scalars: &[f64]) -> Result<Vec<u64>, String> {
    let bytes = ScalarLayout::words(schema)?.encode(scalars)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}

/// Validate the actual byte ranges of one invocation against the root ABI
/// alias rules: `MayOverlap` pairs may overlap only when both are shared-read
/// ranges; every `MustDisjoint` pair must be disjoint in allocation identity
/// or byte range. Empty ranges never conflict.
pub fn validate_alias_rules<D: ExecutableDialect>(
    abi: &ResolvedAbi<D>,
    locate: impl Fn(BufferBindingId) -> Result<Option<(u64, u64, u64)>, String>,
) -> Result<(), String> {
    let buffers = &abi.buffers;
    let find = |binding: BufferBindingId| -> Option<&BufferBinding> {
        buffers.iter().find(|buffer| buffer.binding == binding)
    };
    for rule in &abi.alias_rules {
        let (left, right) = match rule {
            AliasRule::MayOverlap { left, right } => (*left, *right),
            AliasRule::MustDisjoint { left, right } => (*left, *right),
        };
        let (left_buffer, right_buffer) = (find(left), find(right));
        let (Some(left_buffer), Some(right_buffer)) = (left_buffer, right_buffer) else {
            return Err("an alias rule names an absent buffer binding".into());
        };
        if left_buffer.bytes == 0 || right_buffer.bytes == 0 {
            continue;
        }
        // A binding without a location is runtime-allocated (a result
        // buffer): distinct from every caller range by construction.
        let (left_location, right_location) = (locate(left)?, locate(right)?);
        let overlapping = match (left_location, right_location) {
            (Some((left_id, left_offset, _)), Some((right_id, right_offset, _))) => {
                left_id == right_id
                    && left_offset < right_offset.saturating_add(right_buffer.bytes)
                    && right_offset < left_offset.saturating_add(left_buffer.bytes)
            }
            _ => false,
        };
        match rule {
            AliasRule::MayOverlap { .. } => {
                // Shared-read ranges may overlap; nothing to reject.
            }
            AliasRule::MustDisjoint { .. } => {
                if overlapping {
                    return Err(format!(
                        "buffers `{}` and `{}` overlap but must be disjoint",
                        left_buffer.path, right_buffer.path
                    ));
                }
            }
        }
    }
    Ok(())
}
