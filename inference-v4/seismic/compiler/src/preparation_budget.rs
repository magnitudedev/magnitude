//! Separate construction and planning budgets.
//!
//! [`PreparationBudget`] limits refinement and native realization.  It is
//! consumed before a [`CandidateDomain`](crate::candidate_domain::CandidateDomain)
//! is sealed. [`PlanningBudget`] limits only the subsequent search and
//! portfolio materialization over an already evaluated domain. Keeping the
//! two values distinct prevents candidate construction from silently spending
//! (or inheriting) planner resources.

use seismic_target::NativeArtifactMetrics;
use std::collections::HashSet;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparationBudget {
    /// Total non-universal alternatives refinement may construct across root
    /// families and nested calls.
    pub refinement_constructed_alternatives: u64,
    /// Monotonic elapsed-time ceiling for the complete refinement session.
    pub refinement_construction_wall_time: Duration,
    pub unique_native_templates: u64,
    pub native_compile_wall_time: Duration,
    pub native_code_bytes: u64,
    pub metadata_bytes: u64,
    pub required_native_templates: u64,
    pub required_native_compile_wall_time: Duration,
    pub required_native_code_bytes: u64,
    pub required_retained_metadata_bytes: u64,
}

impl Default for PreparationBudget {
    fn default() -> Self {
        Self {
            refinement_constructed_alternatives: 256,
            refinement_construction_wall_time: Duration::from_secs(2),
            unique_native_templates: 256,
            native_compile_wall_time: Duration::from_secs(2),
            native_code_bytes: 256 * 1024 * 1024,
            metadata_bytes: 64 * 1024 * 1024,
            required_native_templates: 4_096,
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
    native_templates: HashSet<[u8; 32]>,
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
            native_templates: HashSet::new(),
            native_compile_wall_time: Duration::ZERO,
            native_code_bytes: 0,
            metadata_bytes: 0,
            required_native_compile_wall_time: Duration::ZERO,
            required_native_code_bytes: 0,
            required_metadata_bytes: 0,
        }
    }

    /// Records every previously unseen template in one already-admitted
    /// implementation. The implementation remains retained if this exact
    /// charge crosses the optional ceiling; `false` stops later admission.
    pub(crate) fn record_native_templates(&mut self, identities: &[[u8; 32]]) -> bool {
        let new = identities
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|identity| !self.native_templates.contains(identity))
            .collect::<Vec<_>>();
        self.native_templates.extend(new);
        (self.native_templates.len() as u64) <= self.limit.unique_native_templates
    }

    /// Records mandatory universal templates even when their size alone
    /// crosses an optional optimization limit. Coverage is never refused by
    /// an optimization budget; `false` prevents subsequent optional work.
    pub(crate) fn record_required_native_templates(
        &mut self,
        identities: &[[u8; 32]],
    ) -> Result<bool, crate::errors::PreparationError> {
        self.native_templates.extend(identities.iter().copied());
        if self.native_templates.len() as u64 > self.limit.required_native_templates {
            return Err(crate::errors::PreparationError::UniversalClosure(
                "native template count exceeds the fixed universal ceiling".into(),
            ));
        }
        Ok(self.native_templates.len() as u64 <= self.limit.unique_native_templates)
    }

    pub(crate) fn charge_metadata(&mut self, bytes: u64) -> bool {
        let total = self.metadata_bytes.saturating_add(bytes);
        self.metadata_bytes = total;
        total <= self.limit.metadata_bytes
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

    pub(crate) fn record_required_metadata(
        &mut self,
        bytes: u64,
    ) -> Result<bool, crate::errors::PreparationError> {
        self.charge_required_metadata(bytes)?;
        Ok(self.charge_metadata(bytes))
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
        if wall_time > self.limit.native_compile_wall_time
            || code_bytes > self.limit.native_code_bytes
            || metadata_bytes > self.limit.metadata_bytes
        {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_recording_is_deduplicated_and_retained() {
        let mut tracker = PreparationBudgetTracker::new(PreparationBudget {
            unique_native_templates: 2,
            ..PreparationBudget::default()
        });
        let a = [1; 32];
        let b = [2; 32];
        let c = [3; 32];
        assert!(tracker.record_native_templates(&[a, a, b]));
        assert!(!tracker.record_native_templates(&[a, c]));
        assert!(!tracker.record_native_templates(&[a, b]));
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
            unique_native_templates: 0,
            metadata_bytes: 0,
            ..PreparationBudget::default()
        });
        assert!(!tracker
            .record_required_native_templates(&[[7; 32]])
            .unwrap());
        assert!(!tracker.record_required_metadata(1).unwrap());
        assert!(!tracker.record_native_templates(&[[8; 32]]));
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
    fn mandatory_hard_ceiling_is_a_typed_failure() {
        let mut tracker = PreparationBudgetTracker::new(PreparationBudget {
            required_native_templates: 0,
            ..PreparationBudget::default()
        });
        assert!(matches!(
            tracker.record_required_native_templates(&[[1; 32]]),
            Err(crate::errors::PreparationError::UniversalClosure(_))
        ));
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
