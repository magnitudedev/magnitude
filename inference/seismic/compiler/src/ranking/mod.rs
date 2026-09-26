//! Ranking: an explicit phase, run at actual recorded invocations, that
//! measures optional members and publishes measured regions. Trials never
//! compare outcomes and never admit anything.

use crate::executable::ExecutableVariant;
use seismic_lang::expr::compiled::InvocationValues;
use seismic_native_target::TargetFamily;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct RankingOptions {
    /// Wall time for all optional work of one kernel.
    pub time: Duration,
    /// Recorded buckets measured.
    pub points: u32,
    /// Seed of synthetic content and proposals.
    pub seed: u64,
}

impl Default for RankingOptions {
    fn default() -> Self {
        Self {
            time: Duration::from_secs(10),
            points: 12,
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrialProtocol {
    pub warmup: u32,
    pub trials: u32,
}

impl TrialProtocol {
    pub const SCREEN: Self = Self {
        warmup: 1,
        trials: 3,
    };
}

/// Synthetic content of one trial point. Integer/bool elements:
/// `Uniform01` = uniform over {0, 1} (bool: false/true), `Ones` = all 1/true,
/// `Zeros` = all 0/false. Float elements: `Uniform01` = uniform over [-1, 1],
/// `Ones` = 1.0, `Zeros` = 0.0, each converted with
/// `reference_math::conversion::convert_bits(F32, dtype, _)`. Packed and
/// external elements: random packet bytes from the draw's seed in every draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentDraw {
    Uniform01 { seed: u64 },
    Ones,
    Zeros,
}

/// `[Uniform01(s0), Ones, Zeros, Uniform01(s1)]`.
pub const CONTENT_DRAWS: usize = 4;

pub struct TrialRequest<'a, T: TargetFamily, H, S> {
    pub variant: Arc<ExecutableVariant<T, H>>,
    pub values: &'a InvocationValues,
    pub sample: &'a S,
    pub content: ContentDraw,
    pub protocol: TrialProtocol,
}

#[derive(Clone, Debug)]
pub enum TrialOutcome {
    Completed { samples: Vec<Duration> },
    SourceFailure,
}

pub const ARCHIVE_CANDIDATES: usize = 4096;
/// Consecutive proposals of already-realized coordinates.
pub const STALL_PROPOSALS: usize = 256;
/// A challenger's screened median must be at least 5% faster.
pub const PROMOTION_MARGIN: f64 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_bound_exploration_only() {
        let options = RankingOptions::default();
        assert_eq!(options.time, Duration::from_secs(10));
        assert_eq!(options.points, 12);
        assert_eq!(options.seed, 0);
        assert_eq!(TrialProtocol::SCREEN.warmup, 1);
        assert_eq!(TrialProtocol::SCREEN.trials, 3);
    }
}
