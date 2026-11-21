//! Owned progress through the source-directed physical constructor.
//!
//! Each step borrows the domain context temporarily. Its continuation retains
//! the actual builder, bound values and lexical source position. Child calls
//! and branch arms resume that state rather than reconstructing a prefix.

use super::*;
use crate::implementation::{BuilderState, CallConstruction, ConstructionContext, ScheduleBranch};

struct ConstructionState<B: seismic_native_target::TargetFamily> {
    /// Boxed: most of a construction's bytes, which every step that moves
    /// the construction on (`advance`, each a new [`ConstructionStep`])
    /// would otherwise copy through its frame.
    builder: Box<BuilderState<B>>,
    values: SemanticBindings,
    mode: SemanticMode,
    computed_producers: BTreeSet<SemanticValueId>,
    regions: Vec<RegionReturn>,
}

pub(crate) struct SourceConstruction<B: seismic_native_target::TargetFamily> {
    state: ConstructionState<B>,
    next: SourceStep<B>,
}

#[derive(Clone, Copy)]
struct RegionCursor {
    region: RegionId,
    next_node: usize,
}

impl RegionCursor {
    fn begin(region: RegionId) -> Self {
        Self {
            region,
            next_node: 0,
        }
    }
    fn next(self) -> Self {
        Self {
            next_node: self.next_node + 1,
            ..self
        }
    }
}

enum LowerStep<B: seismic_native_target::TargetFamily> {
    Body(RegionCursor),
    Node {
        node: NodeId,
        resume: Continuation,
    },
    SelectedArm,
    Segment {
        resume: Continuation,
        progress: SegmentConstruction,
    },
    Publish,
    CallNext {
        resume: Continuation,
        progress: CallConstruction<B>,
        bounds: Vec<Bound>,
    },
}

enum SourceStep<B: seismic_native_target::TargetFamily> {
    Lower(LowerStep<B>),
    CallChild {
        resume: Continuation,
        progress: CallConstruction<B>,
        bounds: Vec<Bound>,
        child: Box<SourceConstruction<B>>,
    },
    Close,
}

#[derive(Clone, Copy)]
enum Continuation {
    Body(RegionCursor),
    SelectedArm,
    Close,
}
impl Continuation {
    fn step<B: seismic_native_target::TargetFamily>(self) -> SourceStep<B> {
        match self {
            Self::Body(cursor) => SourceStep::Lower(LowerStep::Body(cursor)),
            Self::SelectedArm => SourceStep::Lower(LowerStep::SelectedArm),
            Self::Close => SourceStep::Close,
        }
    }
}

#[derive(Clone, Copy)]
enum SelectedAction {
    Node(NodeId),
    Publish,
}
impl SelectedAction {
    fn step<B: seismic_native_target::TargetFamily>(self) -> SourceStep<B> {
        match self {
            Self::Node(node) => SourceStep::Lower(LowerStep::Node {
                node,
                resume: Continuation::SelectedArm,
            }),
            Self::Publish => SourceStep::Lower(LowerStep::Publish),
        }
    }
}

enum RegionReturn {
    Loop {
        resume: Continuation,
        progress: LoopConstruction,
    },
    Selected {
        resume: Continuation,
        action: SelectedAction,
        progress: SelectedConstruction,
    },
    Then {
        resume: Continuation,
        branch: IfConstruction,
    },
    Otherwise {
        resume: Continuation,
        branch: IfConstruction,
    },
}

pub(super) struct IfConstruction {
    pub(super) schedule: ScheduleBranch,
    pub(super) condition: seismic_lang::expr::BoolExpr,
    pub(super) captures: Vec<BindingId>,
    pub(super) outputs: Vec<SemanticValueId>,
    pub(super) then_region: RegionId,
    pub(super) else_region: RegionId,
    pub(super) parent: SemanticBindings,
    pub(super) then_products: Vec<Bound>,
    pub(super) then_contents: Option<StorageContents>,
}

pub(super) struct SelectedConstruction {
    pub(super) selector: BindingSelector,
    pub(super) schedule: ScheduleBranch,
    pub(super) inherited: SemanticBindings,
    pub(super) selected: i64,
    pub(super) outcomes: Vec<(i64, SemanticBindings)>,
}

pub(super) struct SegmentCompletion {
    pub(super) checks: SourceChecks,
    pub(super) check_statuses: Vec<crate::implementation::PortableSourceStatus>,
    pub(super) start: NatExpr,
    pub(super) end: NatExpr,
    pub(super) body: RegionId,
    pub(super) captures: Vec<SemanticValueId>,
}

pub(super) struct SegmentConstruction {
    pub(super) cursor: seismic_ir::kernel::dynamic::PortableCursor,
    pub(super) environment: source_control::SegmentEnvironment,
    pub(super) branch: seismic_ir::kernel::dynamic::PortableBranch,
    pub(super) frames: Vec<source_control::SegmentFrame>,
    pub(super) helpers: BTreeMap<FamilyId, FunctionId>,
    pub(super) checks: SourceStatuses,
    pub(super) cohort: Option<cohort::Cohort>,
    pub(super) domain: SegmentLaunchDomain,
    pub(super) logical_base: Option<seismic_ir::kernel::dynamic::LogicalIndexBinding>,
    pub(super) completion: SegmentCompletion,
}

pub(super) enum SegmentStep {
    Pending(SegmentConstruction),
    Unresolved(SegmentConstruction, capacity::CapacityPending),
    Complete {
        kernel: seismic_ir::kernel::KernelId,
        domain: SegmentLaunchDomain,
        logical_base: Option<seismic_ir::kernel::dynamic::LogicalIndexBinding>,
        completion: SegmentCompletion,
    },
}

impl SegmentConstruction {
    pub(super) fn advance<B: seismic_native_target::TargetFamily>(
        self,
        builder: &mut ImplementationBuilder<'_, B>,
    ) -> SegmentStep {
        let Self {
            cursor,
            environment,
            branch,
            mut frames,
            helpers,
            checks,
            cohort,
            domain,
            logical_base,
            completion,
        } = self;
        let program = builder.portable_program_ref();
        let target = builder.portable_target_ref();
        let registry = builder.portable_registry_ref();
        let mut kernel = builder.resume_portable_kernel(cursor);
        if frames.is_empty() {
            kernel.begin_otherwise(&branch);
            kernel.finish_branch(branch, Vec::new(), Vec::new());
            return SegmentStep::Complete {
                kernel: kernel.close(),
                domain,
                logical_base,
                completion,
            };
        }
        let helper_refs = helpers
            .iter()
            .map(|(family, function)| (*family, program.function(*function)))
            .collect();
        let mut segment = SegmentLowerer {
            function: program.function(environment.function),
            program,
            kernel: &mut kernel,
            values: environment.values,
            helpers: &helper_refs,
            checks: &checks,
            target,
            registry,
            domain,
            alive: environment.alive,
            cohort: cohort.as_ref(),
            successful: environment.successful,
            quantity_bounds: environment.quantity_bounds,
            lexical: environment.lexical,
        };
        let result = segment.lower_frame(&mut frames);
        let environment = segment.into_environment();
        let next = Self {
            cursor: kernel.suspend(),
            environment,
            branch,
            frames,
            helpers,
            checks,
            cohort,
            domain,
            logical_base,
            completion,
        };
        match result {
            Ok(()) => SegmentStep::Pending(next),
            Err(reason) => SegmentStep::Unresolved(next, reason),
        }
    }
}

pub(crate) enum ConstructionStep<B: seismic_native_target::TargetFamily> {
    Pending(SourceConstruction<B>),
    Choice(SourceConstruction<B>, crate::candidate_domain::BodyChoice),
    Unresolved(
        SourceConstruction<B>,
        crate::candidate_domain::ConstructionPending,
    ),
    Complete(ConstructedCandidate<B>),
}

enum ReadyStep<B: seismic_native_target::TargetFamily> {
    Pending(SourceConstruction<B>),
    Unresolved(
        SourceConstruction<B>,
        crate::candidate_domain::ConstructionPending,
    ),
    Complete(ConstructedCandidate<B>),
}

impl<B: seismic_native_target::TargetFamily> From<ReadyStep<B>> for ConstructionStep<B> {
    fn from(step: ReadyStep<B>) -> Self {
        match step {
            ReadyStep::Pending(next) => Self::Pending(next),
            ReadyStep::Unresolved(next, reason) => Self::Unresolved(next, reason),
            ReadyStep::Complete(candidate) => Self::Complete(candidate),
        }
    }
}

impl<B: seismic_native_target::TargetFamily> SourceConstruction<B> {
    pub(crate) fn begin<'a>(
        mut builder: ImplementationBuilder<'a, B>,
        mode: SemanticMode,
    ) -> (Self, ConstructionContext<'a, B>) {
        let function = builder.portable_function_ref();
        let values = Lowerer::new(function, &mut builder, mode).values;
        let computed_producers = if mode == SemanticMode::AuthoredBackend {
            BTreeSet::new()
        } else {
            streaming::values(function)
        };
        let root = function.root();
        let (builder, context) = builder.into_parts();
        (
            Self {
                state: ConstructionState {
                    builder: Box::new(builder),
                    values,
                    mode,
                    computed_producers,
                    regions: Vec::new(),
                },
                next: SourceStep::Lower(LowerStep::Body(RegionCursor::begin(root))),
            },
            context,
        )
    }

    fn next_choice(&self) -> Option<crate::candidate_domain::BodyChoice> {
        match &self.next {
            SourceStep::CallChild {
                progress, child, ..
            } => {
                let mut choice = child.next_choice()?;
                choice.path.0.insert(0, progress.location().clone());
                Some(choice)
            }
            SourceStep::Lower(LowerStep::CallNext { progress, .. }) => progress.choice(),
            _ => None,
        }
    }

    fn initialization_pending(
        &self,
    ) -> Option<seismic_lang::initialization::InitializationFailure> {
        match &self.next {
            SourceStep::CallChild { child, .. } => child.initialization_pending(),
            SourceStep::Lower(LowerStep::CallNext { progress, .. })
                if progress.choice().is_none() =>
            {
                progress.initialization_pending()
            }
            _ => None,
        }
    }

    pub(crate) fn select(&mut self, selection: crate::candidate_domain::BodySelection) {
        match &mut self.next {
            SourceStep::CallChild { child, .. } => child.select(selection),
            SourceStep::Lower(LowerStep::CallNext { progress, .. }) => progress.select(selection),
            _ => panic!("construction is not paused at a body choice"),
        }
    }

    pub(crate) fn advance(self, context: &mut ConstructionContext<'_, B>) -> ConstructionStep<B> {
        if let Some(choice) = self.next_choice() {
            return ConstructionStep::Choice(self, choice);
        }
        if let Some(reason) = self.initialization_pending() {
            return ConstructionStep::Unresolved(
                self,
                crate::candidate_domain::ConstructionPending::Initialization(reason),
            );
        }
        self.advance_ready(context).into()
    }

    fn advance_ready(self, context: &mut ConstructionContext<'_, B>) -> ReadyStep<B> {
        let Self { state, next } = self;
        match next {
            // The parent's physical owner is suspended while its child borrows the
            // same expression owner. Completing a child consumes it exactly once.
            SourceStep::CallChild {
                resume,
                progress,
                bounds,
                child,
            } => Self::advance_child(state, resume, progress, bounds, child, context),
            SourceStep::Close => {
                assert!(
                    state.regions.is_empty(),
                    "source closure still has an unfinished region"
                );
                let builder = ImplementationBuilder::resume(context, *state.builder);
                ReadyStep::Complete(builder.close(state.mode))
            }
            SourceStep::Lower(step) => Self::lower(state, step, context),
        }
    }

    /// [`Self::advance`] of a construction paused in a call: its child
    /// advances, and the call completes once the child does. Each half of
    /// an advance is a frame of its own, as each holds a resumed builder
    /// and the parts of the construction it rebuilds.
    fn advance_child(
        mut state: ConstructionState<B>,
        resume: Continuation,
        mut progress: CallConstruction<B>,
        bounds: Vec<Bound>,
        child: Box<SourceConstruction<B>>,
        context: &mut ConstructionContext<'_, B>,
    ) -> ReadyStep<B> {
        match child.advance_ready(context) {
            ReadyStep::Pending(child) => ReadyStep::Pending(Self {
                state,
                next: SourceStep::CallChild {
                    resume,
                    progress,
                    bounds,
                    child: Box::new(child),
                },
            }),
            ReadyStep::Unresolved(child, reason) => ReadyStep::Unresolved(
                Self {
                    state,
                    next: SourceStep::CallChild {
                        resume,
                        progress,
                        bounds,
                        child: Box::new(child),
                    },
                },
                reason,
            ),
            ReadyStep::Complete(candidate) => {
                let mut parent = ImplementationBuilder::resume(context, *state.builder);
                parent.complete_call_child(&mut progress, candidate);
                state.builder = Box::new(parent.suspend());
                ReadyStep::Pending(Self {
                    state,
                    next: SourceStep::Lower(LowerStep::CallNext {
                        resume,
                        progress,
                        bounds,
                    }),
                })
            }
        }
    }

    /// [`Self::advance`] of a construction at its next step of lowering.
    fn lower(
        state: ConstructionState<B>,
        step: LowerStep<B>,
        context: &mut ConstructionContext<'_, B>,
    ) -> ReadyStep<B> {
        let ConstructionState {
            builder,
            values,
            mode,
            computed_producers,
            mut regions,
        } = state;
        let function = context.program().function(builder.function());
        let mut builder = ImplementationBuilder::resume(context, *builder);
        let mut lowerer = Lowerer {
            function,
            builder: &mut builder,
            values,
            mode,
            computed_producers,
        };
        let mut unresolved = None;
        let next = match step {
            LowerStep::Body(cursor) => {
                if let Some((node, _)) = function.nodes(cursor.region).nth(cursor.next_node) {
                    SourceStep::Lower(LowerStep::Node {
                        node,
                        resume: Continuation::Body(cursor.next()),
                    })
                } else {
                    match regions.pop() {
                        Some(RegionReturn::Loop { resume, progress }) => {
                            lowerer.finish_loop(progress);
                            resume.step()
                        }
                        Some(RegionReturn::Then { resume, mut branch }) => {
                            lowerer.next_if(&mut branch);
                            let next = RegionCursor::begin(branch.else_region);
                            regions.push(RegionReturn::Otherwise { resume, branch });
                            SourceStep::Lower(LowerStep::Body(next))
                        }
                        Some(RegionReturn::Otherwise { resume, branch }) => {
                            lowerer.finish_if(branch);
                            resume.step()
                        }
                        Some(RegionReturn::Selected { .. }) => {
                            unreachable!("a selected action has its own completion step")
                        }
                        None => SourceStep::Lower(LowerStep::Publish),
                    }
                }
            }
            LowerStep::Node { node, resume } => {
                node_step(&mut lowerer, &mut regions, &mut unresolved, node, resume)
            }
            LowerStep::Segment { resume, progress } => match progress.advance(lowerer.builder) {
                SegmentStep::Pending(progress) => {
                    SourceStep::Lower(LowerStep::Segment { resume, progress })
                }
                SegmentStep::Unresolved(progress, reason) => {
                    unresolved = Some(crate::candidate_domain::ConstructionPending::Capacity(
                        reason,
                    ));
                    SourceStep::Lower(LowerStep::Segment { resume, progress })
                }
                SegmentStep::Complete {
                    kernel,
                    domain,
                    logical_base,
                    completion,
                } => {
                    lowerer.finish_parallel_segment(kernel, domain, logical_base, completion);
                    resume.step()
                }
            },
            LowerStep::SelectedArm => {
                let Some(RegionReturn::Selected {
                    resume,
                    action,
                    progress,
                }) = regions.pop()
                else {
                    panic!("selected source action lost its lexical continuation")
                };
                if let Some(progress) = lowerer.next_selected_binding(progress) {
                    regions.push(RegionReturn::Selected {
                        resume,
                        action,
                        progress,
                    });
                    action.step()
                } else {
                    resume.step()
                }
            }
            LowerStep::Publish => {
                if let Some(selector) = lowerer.publication_selection() {
                    let progress = lowerer.begin_selected_binding(selector);
                    let action = SelectedAction::Publish;
                    let resume = if matches!(regions.last(), Some(RegionReturn::Selected { .. })) {
                        Continuation::SelectedArm
                    } else {
                        Continuation::Close
                    };
                    regions.push(RegionReturn::Selected {
                        resume,
                        action,
                        progress,
                    });
                    action.step()
                } else {
                    lowerer.publish_results();
                    if matches!(regions.last(), Some(RegionReturn::Selected { .. })) {
                        SourceStep::Lower(LowerStep::SelectedArm)
                    } else {
                        SourceStep::Close
                    }
                }
            }
            LowerStep::CallNext {
                resume,
                mut progress,
                bounds,
            } => {
                let child = lowerer.builder.begin_call_child(
                    &mut progress,
                    &lowerer.values.selections,
                    &lowerer.values.contents,
                    &lowerer.values.binders,
                );
                if let Some(child) = child {
                    SourceStep::CallChild {
                        resume,
                        progress,
                        bounds,
                        child: Box::new(child),
                    }
                } else {
                    lowerer.finish_call(progress, &bounds);
                    resume.step()
                }
            }
        };
        let Lowerer {
            values,
            computed_producers,
            ..
        } = lowerer;
        let next = Self {
            state: ConstructionState {
                builder: Box::new(builder.suspend()),
                values,
                mode,
                computed_producers,
                regions,
            },
            next,
        };
        match unresolved {
            Some(reason) => ReadyStep::Unresolved(next, reason),
            None => ReadyStep::Pending(next),
        }
    }
}

/// The step after source node `node`: [`SourceConstruction::lower`]'s
/// largest arm, a frame of its own for the steps it builds.
fn node_step<B: seismic_native_target::TargetFamily>(
    lowerer: &mut Lowerer<'_, '_, B>,
    regions: &mut Vec<RegionReturn>,
    unresolved: &mut Option<crate::candidate_domain::ConstructionPending>,
    node: NodeId,
    resume: Continuation,
) -> SourceStep<B> {
    let (function, mode) = (lowerer.function, lowerer.mode);
    if let Some(selector) = lowerer.node_selection(node) {
        let progress = lowerer.begin_selected_binding(selector);
        let action = SelectedAction::Node(node);
        regions.push(RegionReturn::Selected {
            resume,
            action,
            progress,
        });
        action.step()
    } else if let Some((value, representation)) = lowerer.pending_representation_view(node) {
        *unresolved = Some(
            crate::candidate_domain::ConstructionPending::RepresentationView {
                value,
                representation,
            },
        );
        SourceStep::Lower(LowerStep::Node { node, resume })
    } else {
        let data = function.node(node);
        match data.view() {
            SemanticNodeView::Call { inputs, .. } => {
                lowerer.snapshot_before_write(node);
                let (progress, bounds) = lowerer.prepare_call(node, inputs);
                SourceStep::Lower(LowerStep::CallNext {
                    resume,
                    progress,
                    bounds,
                })
            }
            SemanticNodeView::Loop {
                kind: LoopKind::Parallel,
                start,
                end,
                captures,
                body,
                carries,
                ..
            } if mode == SemanticMode::Portable => {
                lowerer.snapshot_before_write(node);
                for dependency in data.dependencies() {
                    lowerer.materialize_tensor(dependency);
                }
                // Checked independent visits may execute in any order. The
                // required mapping uses the ordinary source cursor so exact
                // host quantities and reached failures keep their source
                // operations instead of entering a native-only segment. A
                // visit's failure ends only that participant.
                let progress = lowerer.begin_loop(
                    start,
                    end,
                    captures,
                    body,
                    carries,
                    seismic_ir::schedule::RepeatVisits::Independent,
                );
                regions.push(RegionReturn::Loop { resume, progress });
                SourceStep::Lower(LowerStep::Body(RegionCursor::begin(body)))
            }
            SemanticNodeView::Loop {
                kind: LoopKind::Parallel,
                start,
                end,
                captures,
                body,
                carries,
                ..
            } => {
                lowerer.snapshot_before_write(node);
                for dependency in data.dependencies() {
                    lowerer.materialize_tensor(dependency);
                }
                let progress =
                    lowerer.begin_parallel_segment(start, end, captures, body, carries, &[]);
                SourceStep::Lower(LowerStep::Segment { resume, progress })
            }
            SemanticNodeView::Loop {
                start,
                end,
                captures,
                body,
                carries,
                ..
            } => {
                lowerer.snapshot_before_write(node);
                for dependency in data.dependencies() {
                    lowerer.materialize_tensor(dependency);
                }
                let progress = lowerer.begin_loop(
                    start,
                    end,
                    captures,
                    body,
                    carries,
                    seismic_ir::schedule::RepeatVisits::Ordered,
                );
                regions.push(RegionReturn::Loop { resume, progress });
                SourceStep::Lower(LowerStep::Body(RegionCursor::begin(body)))
            }
            SemanticNodeView::If {
                condition,
                captures,
                outputs,
                then,
                otherwise,
            } => {
                lowerer.snapshot_before_write(node);
                for dependency in data.dependencies() {
                    lowerer.materialize_tensor(dependency);
                }
                let branch = lowerer.begin_if(condition, captures, outputs, then, otherwise);
                regions.push(RegionReturn::Then { resume, branch });
                SourceStep::Lower(LowerStep::Body(RegionCursor::begin(then)))
            }
            _ => {
                lowerer.lower_node_resolved(node);
                resume.step()
            }
        }
    }
}
