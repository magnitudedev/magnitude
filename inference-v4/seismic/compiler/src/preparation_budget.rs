//! Separate construction and planning budgets.
//!
//! [`PreparationBudget`] limits structural refinement and demand-driven native
//! realization. [`PlanningBudget`] limits evaluator search and retained-policy
//! materialization. Keeping them distinct prevents structural/native work from
//! silently spending solver and policy-retention resources.

use seismic_target::NativeArtifactMetrics;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparationBudget {
    /// Source construction steps available to optional evaluator exploration.
    /// Pausing preserves the actual constructor state; it does not remove definitions.
    pub construction_work_units: u64,
    /// Total optional source-construction time available to the evaluator.
    pub construction_wall_time: Duration,
    pub native_compile_wall_time: Duration,
    pub native_code_bytes: u64,
    pub metadata_bytes: u64,
    pub required_native_compile_wall_time: Duration,
    pub required_native_code_bytes: u64,
    pub required_retained_metadata_bytes: u64,
}

impl Default for PreparationBudget {
    fn default() -> Self {
        Self {
            construction_work_units: 100_000,
            construction_wall_time: Duration::from_secs(2),
            native_compile_wall_time: Duration::from_secs(2),
            native_code_bytes: 256 * 1024 * 1024,
            metadata_bytes: 64 * 1024 * 1024,
            required_native_compile_wall_time: Duration::from_secs(60),
            required_native_code_bytes: 256 * 1024 * 1024,
            required_retained_metadata_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Resources available only to planning an already evaluated domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanningBudget {
    pub solver_work: u64,
    pub solver_memory_bytes: u64,
    pub optimized_assignments: u64,
    pub executable_variants: u64,
    pub retained_metadata_bytes: u64,
    /// Hard ceiling for the mandatory universal executable. Crossing this is
    /// a typed planning failure rather than a partial-coverage result.
    pub required_retained_metadata_bytes: u64,
}

impl Default for PlanningBudget {
    fn default() -> Self {
        Self {
            solver_work: 200_000,
            solver_memory_bytes: 64 * 1024 * 1024,
            optimized_assignments: 256,
            executable_variants: 256,
            retained_metadata_bytes: 64 * 1024 * 1024,
            required_retained_metadata_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparationBudgetTracker {
    limit: PreparationBudget,
    native_compile_wall_time: Duration,
    native_code_bytes: u64,
    metadata_bytes: u64,
    required_native_compile_wall_time: Duration,
    required_native_code_bytes: u64,
    required_metadata_bytes: u64,
}

impl PreparationBudgetTracker {
    pub(crate) fn new(limit: PreparationBudget) -> Self {
        Self {
            limit,
            native_compile_wall_time: Duration::ZERO,
            native_code_bytes: 0,
            metadata_bytes: 0,
            required_native_compile_wall_time: Duration::ZERO,
            required_native_code_bytes: 0,
            required_metadata_bytes: 0,
        }
    }

    fn charge_required_metadata(
        &mut self,
        bytes: u64,
    ) -> Result<(), crate::errors::PreparationError> {
        let total = self
            .required_metadata_bytes
            .checked_add(bytes)
            .ok_or_else(|| {
                crate::errors::PreparationError::UniversalClosure(
                    "required retained metadata accounting overflowed".into(),
                )
            })?;
        if total > self.limit.required_retained_metadata_bytes {
            return Err(crate::errors::PreparationError::UniversalClosure(
                "required retained metadata exceeds its configured hard ceiling".into(),
            ));
        }
        self.required_metadata_bytes = total;
        Ok(())
    }

    pub(crate) fn record_required_native_artifact(
        &mut self,
        metrics: NativeArtifactMetrics,
    ) -> Result<bool, crate::errors::PreparationError> {
        let compilation = Duration::from_nanos(metrics.compilation_ns);
        let required_wall_time = self
            .required_native_compile_wall_time
            .checked_add(compilation)
            .ok_or_else(|| {
                crate::errors::PreparationError::UniversalClosure(
                    "required native compile-time accounting overflowed".into(),
                )
            })?;
        let required_code_bytes = self
            .required_native_code_bytes
            .checked_add(metrics.code_bytes)
            .ok_or_else(|| {
                crate::errors::PreparationError::UniversalClosure(
                    "required native code-size accounting overflowed".into(),
                )
            })?;
        if required_wall_time > self.limit.required_native_compile_wall_time
            || required_code_bytes > self.limit.required_native_code_bytes
        {
            return Err(crate::errors::PreparationError::UniversalClosure(
                "cumulative required native artifacts exceed a configured hard ceiling".into(),
            ));
        }
        self.charge_required_metadata(metrics.metadata_bytes)?;
        self.required_native_compile_wall_time = required_wall_time;
        self.required_native_code_bytes = required_code_bytes;
        Ok(self.record_native_artifact(metrics))
    }

    /// Records exact post-compilation measurements. The completed artifact is
    /// retained even if it crosses an optional limit; `false` prevents the
    /// next optional native attempt. This never hides a toolchain failure by
    /// dropping an artifact after compilation was attempted.
    pub(crate) fn record_native_artifact(&mut self, metrics: NativeArtifactMetrics) -> bool {
        let wall_time = self
            .native_compile_wall_time
            .saturating_add(Duration::from_nanos(metrics.compilation_ns));
        let code_bytes = self.native_code_bytes.saturating_add(metrics.code_bytes);
        let metadata_bytes = self.metadata_bytes.saturating_add(metrics.metadata_bytes);
        self.native_compile_wall_time = wall_time;
        self.native_code_bytes = code_bytes;
        self.metadata_bytes = metadata_bytes;
        self.native_attempt_available()
    }

    pub(crate) fn native_usage(&self) -> (u64, u64) {
        (self.native_code_bytes, self.metadata_bytes)
    }

    pub(crate) fn native_attempt_available(&self) -> bool {
        self.native_compile_wall_time <= self.limit.native_compile_wall_time
            && self.native_code_bytes <= self.limit.native_code_bytes
            && self.metadata_bytes <= self.limit.metadata_bytes
    }

    pub(crate) fn extend_native_time(&mut self, additional: Duration) {
        self.limit.native_compile_wall_time = self
            .limit
            .native_compile_wall_time
            .saturating_add(additional);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuation_extends_time_but_cannot_erase_byte_exhaustion() {
        let mut timed = PreparationBudgetTracker::new(PreparationBudget {
            native_compile_wall_time: Duration::ZERO,
            ..Default::default()
        });
        assert!(!timed.record_native_artifact(NativeArtifactMetrics {
            compilation_ns: 1,
            code_bytes: 1,
            metadata_bytes: 1
        }));
        timed.extend_native_time(Duration::from_secs(1));
        assert!(timed.native_attempt_available());
        let mut full = PreparationBudgetTracker::new(PreparationBudget {
            native_code_bytes: 0,
            ..Default::default()
        });
        assert!(!full.record_native_artifact(NativeArtifactMetrics {
            compilation_ns: 1,
            code_bytes: 1,
            metadata_bytes: 1
        }));
        full.extend_native_time(Duration::from_secs(100));
        assert!(!full.native_attempt_available());
    }

    #[test]
    fn completed_native_artifact_is_recorded_before_exhaustion() {
        let mut tracker = PreparationBudgetTracker::new(PreparationBudget {
            native_code_bytes: 4,
            ..PreparationBudget::default()
        });
        assert!(!tracker.record_native_artifact(NativeArtifactMetrics {
            compilation_ns: 1,
            code_bytes: 5,
            metadata_bytes: 2,
        }));
        assert!(!tracker.record_native_artifact(NativeArtifactMetrics {
            compilation_ns: 1,
            code_bytes: 0,
            metadata_bytes: 0,
        }));
    }

    #[test]
    fn mandatory_work_is_retained_and_stops_optional_work() {
        let mut tracker = PreparationBudgetTracker::new(PreparationBudget {
            metadata_bytes: 0,
            ..PreparationBudget::default()
        });
        assert!(!tracker
            .record_required_native_artifact(NativeArtifactMetrics {
                compilation_ns: 0,
                code_bytes: 0,
                metadata_bytes: 1,
            })
            .unwrap());
    }

    #[test]
    fn planning_memory_is_always_a_finite_allowance() {
        let budget = PlanningBudget {
            solver_memory_bytes: 0,
            ..PlanningBudget::default()
        };
        assert_eq!(budget.solver_memory_bytes, 0);
    }

    #[test]
    fn mandatory_native_hard_ceilings_are_cumulative() {
        let mut tracker = PreparationBudgetTracker::new(PreparationBudget {
            required_native_compile_wall_time: Duration::from_nanos(3),
            required_native_code_bytes: 3,
            ..PreparationBudget::default()
        });
        assert!(tracker
            .record_required_native_artifact(NativeArtifactMetrics {
                compilation_ns: 2,
                code_bytes: 2,
                metadata_bytes: 0,
            })
            .is_ok());
        assert!(matches!(
            tracker.record_required_native_artifact(NativeArtifactMetrics {
                compilation_ns: 2,
                code_bytes: 2,
                metadata_bytes: 0,
            }),
            Err(crate::errors::PreparationError::UniversalClosure(_))
        ));
    }
}
