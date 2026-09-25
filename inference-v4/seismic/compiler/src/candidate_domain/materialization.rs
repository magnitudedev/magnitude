//! Owned, bounded construction. This module has no traversal policy: a caller
//! chooses a source path and supplies the work allowance for this advance.

use super::*;
use crate::implementation::ConstructionContext;
use super::SourceEntryBorrow;
use crate::portable::construction::{ConstructionStep, SourceConstruction};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct ConstructionAllowance {
    pub work_units: u64,
    pub wall_time: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstructionPending {
    WorkAllowance,
    Capacity(crate::portable::capacity::CapacityPending),
    Initialization(seismic_lang::initialization::InitializationFailure),
    /// A stored logical slice still needs its registered representation map.
    RepresentationView {
        value: seismic_lang::ids::SemanticValueId,
        representation: seismic_lang::ids::RepresentationId,
    },
}

#[derive(Clone, Debug)]
pub enum ConstructionExclusion {
    UnknownBody,
    UnsupportedMapping,
    InvalidCallSelection,
}

#[derive(Clone, Debug)]
pub enum Materialization {
    Choice(BodyChoice),
    /// A closed physical member. Numerical admission and native realization
    /// remain separate operations owned by the evaluation session.
    Ready(ConstructionCoordinate),
    Excluded(ConstructionExclusion),
    Pending(ConstructionPending),
}

#[derive(Clone, Debug)]
pub struct ConstructionAdvance {
    pub state: Materialization,
    pub work_units: u64,
    pub elapsed: Duration,
}

/// A bounded immutable read of constructed physical data in its owning arena.
pub struct MaterializedRead<'a, B: seismic_native_target::TargetFamily> {
    candidate: &'a DomainCandidate<B>,
    arena: RwLockReadGuard<'a, ExprArena>,
}
impl<B: seismic_native_target::TargetFamily> MaterializedRead<'_, B> {
    pub fn arena(&self) -> &ExprArena {
        &self.arena
    }
    pub fn candidate(&self) -> &ConstructedCandidate<B> {
        &self.candidate.family
    }
    pub fn executable(&self) -> crate::evaluation::TargetClosedExecutableView<'_, B> {
        crate::evaluation::TargetClosedExecutableView::new(
            &self.candidate.family,
            self.candidate.constraints.conjuncts(),
        )
    }
}

pub(super) struct Suspended<B: seismic_native_target::TargetFamily> {
    source: SourceConstruction<B>,
    selected: Vec<(CallPath, BodySelection)>,
}
pub(super) enum Progress<B: seismic_native_target::TargetFamily> {
    Suspended(Suspended<B>),
    Complete(ConstructionCoordinate),
}
impl<B: seismic_native_target::TargetFamily> std::fmt::Debug for Progress<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Suspended(_) => f.write_str("Suspended source construction"),
            Self::Complete(identity) => f.debug_tuple("Complete").field(identity).finish(),
        }
    }
}

impl<B: seismic_native_target::TargetFamily> CandidateDomain<'_, B> {
    pub fn read_materialized(
        &self,
        coordinate: &ConstructionCoordinate,
    ) -> Result<MaterializedRead<'_, B>, CoordinateError> {
        Ok(MaterializedRead {
            candidate: self.family(coordinate)?,
            arena: self.arena.read().expect("domain expression owner poisoned"),
        })
    }

    pub fn root_selections(&self) -> Vec<BodySelection> {
        use seismic_lang::entry::CandidateKind;
        let root = self.source_program.family(self.source_program.root());
        self.families
            .iter()
            .flat_map(|family| {
                let body = root
                    .candidates()
                    .iter()
                    .find(|body| body.function == family.body)
                    .unwrap();
                let mappings: &[BodyMapping] = match body.kind {
                    CandidateKind::Portable => &[BodyMapping::Sequential, BodyMapping::Independent],
                    _ => &[BodyMapping::Authored],
                };
                mappings.iter().map(move |mapping| BodySelection {
                    subject: self.source_program.subject().clone(),
                    body: family.identity,
                    source_definition: self
                        .source_program
                        .function(family.body)
                        .source_definition(),
                    mapping: *mapping,
                })
            })
            .collect()
    }

    pub fn general_construction(&self) -> ConstructionCoordinate {
        self.materialized.first().construction.clone()
    }

    pub fn advance(
        &mut self,
        coordinate: &ConstructionCoordinate,
        allowance: ConstructionAllowance,
    ) -> ConstructionAdvance {
        let started = Instant::now();
        let mut work_units = 0;
        let result = self.advance_inner(coordinate, allowance, started, &mut work_units);
        ConstructionAdvance {
            state: result,
            work_units,
            elapsed: started.elapsed(),
        }
    }

    fn advance_inner(
        &mut self,
        coordinate: &ConstructionCoordinate,
        allowance: ConstructionAllowance,
        started: Instant,
        work_units: &mut u64,
    ) -> Materialization {
        if coordinate == &self.general_construction() {
            return Materialization::Ready(self.general_construction());
        }
        let Some(_) = self
            .families
            .iter()
            .find(|family| {
                let function = self.source_program.function(family.body);
                family.identity == coordinate.root.body
                    && function.source_definition() == coordinate.root.source_definition
                    && self.source_program.subject() == &coordinate.root.subject
            })
        else {
            return Materialization::Excluded(ConstructionExclusion::UnknownBody);
        };
        if !self.root_selections().contains(&coordinate.root) {
            return Materialization::Excluded(ConstructionExclusion::UnsupportedMapping);
        }
        let mut progress = self.materializations.remove(coordinate);
        // A selected successor consumes the very same parent continuation. A
        // later sibling reconstructs its prefix under its own charged allowance.
        if progress.is_none() && !coordinate.calls.is_empty() {
            let mut prefix = coordinate.clone();
            prefix.calls.pop();
            progress = self
                .materializations
                .remove(&prefix)
                .and_then(|progress| match progress {
                    Progress::Suspended(progress) => Some(Progress::Suspended(progress)),
                    complete => {
                        self.materializations.insert(prefix, complete);
                        None
                    }
                });
        }
        if let Some(Progress::Complete(canonical)) = progress {
            if self.family(&canonical).is_ok() {
                self.materializations
                    .insert(coordinate.clone(), Progress::Complete(canonical.clone()));
                return Materialization::Ready(canonical);
            }
            progress = None;
        }
        let mut progress = match progress {
            Some(Progress::Suspended(progress)) => Some(progress),
            None => None,
            Some(Progress::Complete(_)) => unreachable!(),
        };
        loop {
            if *work_units >= allowance.work_units || started.elapsed() >= allowance.wall_time {
                if let Some(progress) = progress {
                    self.materializations
                        .insert(coordinate.clone(), Progress::Suspended(progress));
                }
                return Materialization::Pending(ConstructionPending::WorkAllowance);
            }
            *work_units += 1;
            let mut arena = self
                .arena
                .write()
                .expect("domain expression owner poisoned");
            let entry = SourceEntryBorrow::new(&mut arena, &self.source_program, &self.schema);
            let (current, mut context) = if let Some(progress) = progress.take() {
                (
                    progress,
                    ConstructionContext::for_resume(
                        entry,
                        self.target,
                        self.registry,
                        &self.constants,
                        &self.precision,
                    ),
                )
            } else {
                let (source, context) = crate::implementation::begin_source_construction(
                    entry,
                    coordinate.root.clone(),
                    self.target,
                    self.registry,
                    &self.constants,
                    &self.precision,
                );
                (
                    Suspended {
                        source,
                        selected: Vec::new(),
                    },
                    context,
                )
            };
            let Suspended { source, selected } = current;
            match source.advance(&mut context) {
                ConstructionStep::Pending(source) => {
                    progress = Some(Suspended { source, selected })
                }
                ConstructionStep::Choice(mut next, choice) => {
                    let current_choices = selected;
                    if let Some((_, selected)) = coordinate
                        .calls
                        .iter()
                        .find(|(path, _)| path == &choice.path)
                    {
                        if !choice.alternatives.contains(selected) {
                            return Materialization::Excluded(
                                ConstructionExclusion::InvalidCallSelection,
                            );
                        }
                        next.select(selected.clone());
                        let selection = (choice.path.clone(), selected.clone());
                        // The selected path is retained beside the actual
                        // continuation in the order choices were reached.
                        let mut choices = current_choices;
                        choices.push(selection);
                        progress = Some(Suspended {
                            source: next,
                            selected: choices,
                        });
                    } else {
                        self.materializations.insert(
                            coordinate.clone(),
                            Progress::Suspended(Suspended {
                                source: next,
                                selected: current_choices,
                            }),
                        );
                        return Materialization::Choice(choice);
                    }
                }
                ConstructionStep::Unresolved(next, reason) => {
                    self.materializations.insert(coordinate.clone(), Progress::Suspended(Suspended { source: next, selected }));
                    return Materialization::Pending(reason);
                }
                ConstructionStep::Complete(candidate) => {
                    if selected.len() != coordinate.calls.len()
                        || !coordinate
                            .calls
                            .iter()
                            .all(|selection| selected.contains(selection))
                    {
                        return Materialization::Excluded(
                            ConstructionExclusion::InvalidCallSelection,
                        );
                    }
                    let canonical = ConstructionCoordinate {
                        root: coordinate.root.clone(),
                        calls: selected,
                    };
                    if canonical != self.general_construction()
                        && !self
                            .materialized
                            .iter()
                            .any(|cached| cached.construction == canonical)
                    {
                        let candidate = internals::structural_candidate(
                            &mut arena,
                            candidate,
                            canonical.clone(),
                            self.target_domain.predicate().node(),
                            &self.precision,
                        );
                        self.materialized.push(candidate);
                    }
                    self.materializations
                        .insert(coordinate.clone(), Progress::Complete(canonical.clone()));
                    return Materialization::Ready(canonical);
                }
            }
        }
    }
}
