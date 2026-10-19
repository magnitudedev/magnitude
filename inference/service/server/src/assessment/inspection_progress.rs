//! An inspection's progress: the phase it is in, how much of that phase is done, and each
//! version's verdict as soon as it is known, reported to the inspection's sink after every step.

use std::sync::Mutex;

use magnitude_service_contracts::models::{
    CandidateInspectionPhase, CandidateInspectionProgress, CandidateInspectionProgressSink,
    CandidateVersionProgress, ModelAssessment,
};

pub(super) struct InspectionProgress {
    sink: CandidateInspectionProgressSink,
    state: Mutex<CandidateInspectionProgress>,
}

impl InspectionProgress {
    pub(super) fn new(sink: CandidateInspectionProgressSink) -> Self {
        Self {
            sink,
            state: Mutex::new(CandidateInspectionProgress {
                phase: CandidateInspectionPhase::ReadingHeaders {
                    completed: 0,
                    total: 0,
                },
                versions: Vec::new(),
            }),
        }
    }

    /// Progress nobody observes, for the drafter lists and assessments outside an inspection.
    pub(super) fn silent() -> Self {
        Self::new(CandidateInspectionProgressSink::none())
    }

    /// Reading headers: `rows` are the files named so far, `completed` of `total` headers known.
    pub(super) fn reading_headers(&self, rows: Vec<(String, u64)>, completed: usize, total: usize) {
        self.update(|state| {
            state.phase = CandidateInspectionPhase::ReadingHeaders {
                completed: count(completed),
                total: count(total),
            };
            state.versions = rows.into_iter().map(unchecked).collect();
        });
    }

    pub(super) fn header_read(&self) {
        self.update(|state| {
            if let CandidateInspectionPhase::ReadingHeaders { completed, total } = &mut state.phase
            {
                *completed = (*completed + 1).min(*total);
            }
        });
    }

    /// Checking versions: `rows` are the versions the inspection returns, none checked yet.
    pub(super) fn checking_versions(&self, rows: Vec<(String, u64)>) {
        self.update(|state| {
            state.phase = CandidateInspectionPhase::CheckingVersions {
                completed: 0,
                total: count(rows.len()),
            };
            state.versions = rows.into_iter().map(unchecked).collect();
        });
    }

    /// The version at `index` is checked, with its plain verdict when it has one.
    pub(super) fn version_checked(&self, index: usize, assessment: Option<ModelAssessment>) {
        self.update(|state| {
            if let Some(version) = state.versions.get_mut(index) {
                version.assessment = assessment;
            }
            if let CandidateInspectionPhase::CheckingVersions { completed, total } =
                &mut state.phase
            {
                *completed = (*completed + 1).min(*total);
            }
        });
    }

    /// Checking the drafters of the version at `index`, the one the inspection recommends.
    pub(super) fn checking_drafters(&self, index: usize) {
        self.update(|state| {
            let version = state
                .versions
                .get(index)
                .map(|version| version.label.clone())
                .unwrap_or_default();
            state.phase = CandidateInspectionPhase::CheckingDrafters {
                version,
                completed: 0,
                total: 0,
            };
        });
    }

    /// `planned` more drafter checks are about to run.
    pub(super) fn drafters_planned(&self, planned: usize) {
        if planned == 0 {
            return;
        }
        self.update(|state| {
            if let CandidateInspectionPhase::CheckingDrafters { total, .. } = &mut state.phase {
                *total = total.saturating_add(count(planned));
            }
        });
    }

    pub(super) fn drafter_checked(&self) {
        self.update(|state| {
            if let CandidateInspectionPhase::CheckingDrafters {
                completed, total, ..
            } = &mut state.phase
            {
                *completed = (*completed + 1).min(*total);
            }
        });
    }

    fn update(&self, change: impl FnOnce(&mut CandidateInspectionProgress)) {
        if !self.sink.is_reporting() {
            return;
        }
        let snapshot = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            change(&mut state);
            state.clone()
        };
        self.sink.report(snapshot);
    }
}

fn unchecked((label, size_bytes): (String, u64)) -> CandidateVersionProgress {
    CandidateVersionProgress {
        label,
        size_bytes,
        assessment: None,
    }
}

fn count(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}
