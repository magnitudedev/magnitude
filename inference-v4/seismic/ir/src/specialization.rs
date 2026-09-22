//! Exact pre-native specialization of a closed executable.
//!
//! Candidate decisions may choose schedule regions, but native compilers must
//! never be asked to compile kernels from regions the candidate cannot enter.
//! This module owns the exhaustive traversal from a closed schedule plus an
//! exact decision assignment to the launches and kernels that can execute.

use crate::execution::ClosedExecutableIr;
use crate::kernel::{Kernel, KernelId};
use crate::schedule::{Launch, LaunchId, ScheduleStep};
use crate::storage::LaunchLocalLayout;
use crate::target::KernelDialect;
use seismic_lang::expr::{DecisionId, ExprArena, PartialAssignment, SymbolValue};

/// Why an exact candidate assignment cannot specialize a closed executable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecializationError {
    /// A schedule choice has no value in the candidate assignment.
    MissingDecision(DecisionId),
    /// A decision symbol was bound with a value of the wrong expression sort.
    NonIntegerDecision(DecisionId),
    /// The assignment selected a value outside the schedule's closed options.
    AbsentChoiceValue { decision: DecisionId, value: i64 },
}

impl std::fmt::Display for SpecializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDecision(decision) => {
                write!(f, "native specialization is missing {decision:?}")
            }
            Self::NonIntegerDecision(decision) => {
                write!(
                    f,
                    "native specialization has a non-integer value for {decision:?}"
                )
            }
            Self::AbsentChoiceValue { decision, value } => write!(
                f,
                "native specialization selected absent value {value} for {decision:?}"
            ),
        }
    }
}

impl std::error::Error for SpecializationError {}

/// One active launch together with the exact closed kernel and local layout it
/// references. All values retain their original IR ordinals.
#[derive(Clone, Copy)]
pub struct SpecializedLaunch<'a, T: KernelDialect> {
    pub id: LaunchId,
    pub launch: &'a Launch,
    pub layout: &'a LaunchLocalLayout,
    pub kernel: &'a Kernel<T>,
}

/// The exact native-facing closure of one candidate.
///
/// Construction is available only through
/// [`ClosedExecutableIr::specialize_for_native`].
/// The traversal is exhaustive over [`ScheduleStep`]: runtime branches retain
/// both arms, loops retain their body, and every compile-time choice retains
/// exactly its assigned arm. Consequently a native realizer can compile
/// `kernels()` and cannot observe an inactive kernel through this value.
pub struct NativeSpecialization<'a, T: KernelDialect> {
    executable: &'a ClosedExecutableIr<T>,
    launches: Vec<LaunchId>,
    kernels: Vec<KernelId>,
}

impl<'a, T: KernelDialect> NativeSpecialization<'a, T> {
    /// Active launches in deterministic schedule traversal order.
    pub fn launches(&self) -> impl ExactSizeIterator<Item = SpecializedLaunch<'a, T>> + '_ {
        self.launches.iter().copied().map(|id| {
            let launch = self.executable.schedule().launch(id);
            let layout = &self.executable.launch_layouts()[id.index() as usize];
            let kernel = self.executable.kernels().kernel(launch.kernel);
            SpecializedLaunch {
                id,
                launch,
                layout,
                kernel,
            }
        })
    }

    /// Active kernels in first-launch order, with duplicates removed.
    pub fn kernels(&self) -> impl ExactSizeIterator<Item = (KernelId, &'a Kernel<T>)> + '_ {
        self.kernels
            .iter()
            .copied()
            .map(|id| (id, self.executable.kernels().kernel(id)))
    }

    /// Dense native-artifact index for an active original kernel identity.
    pub fn native_kernel_index(&self, id: KernelId) -> Option<u32> {
        self.kernels
            .iter()
            .position(|candidate| *candidate == id)
            .map(|index| u32::try_from(index).expect("active kernel count exceeds u32"))
    }
}

impl<T: KernelDialect> ClosedExecutableIr<T> {
    /// Resolve every compile-time schedule choice before native compilation.
    pub fn specialize_for_native<'a>(
        &'a self,
        arena: &ExprArena,
        assignment: &PartialAssignment,
    ) -> Result<NativeSpecialization<'a, T>, SpecializationError> {
        let mut launches = Vec::new();
        collect_active_launches(
            self.schedule().steps(),
            self.schedule(),
            arena,
            assignment,
            &mut launches,
        )?;

        let mut kernels = Vec::new();
        for id in launches.iter().copied() {
            let kernel = self.schedule().launch(id).kernel;
            if !kernels.contains(&kernel) {
                kernels.push(kernel);
            }
        }
        Ok(NativeSpecialization {
            executable: self,
            launches,
            kernels,
        })
    }
}

fn collect_active_launches(
    steps: &[ScheduleStep],
    schedule: &crate::schedule::ParametricSchedule,
    arena: &ExprArena,
    assignment: &PartialAssignment,
    launches: &mut Vec<LaunchId>,
) -> Result<(), SpecializationError> {
    for step in steps {
        match step {
            ScheduleStep::Launch(id) => {
                // Touch the launch here so the closed schedule remains the
                // authority for identity validity.
                schedule.launch(*id);
                launches.push(*id);
            }
            ScheduleStep::Copy(_)
            | ScheduleStep::Fill(_)
            | ScheduleStep::ScalarMove(_)
            | ScheduleStep::ScalarRead(_)
            | ScheduleStep::Check(_) => {}
            ScheduleStep::If {
                then_steps,
                else_steps,
                ..
            } => {
                collect_active_launches(then_steps, schedule, arena, assignment, launches)?;
                collect_active_launches(else_steps, schedule, arena, assignment, launches)?;
            }
            ScheduleStep::Repeat { body, .. } => {
                collect_active_launches(body, schedule, arena, assignment, launches)?;
            }
            ScheduleStep::Choose { decision, options } => {
                let value = match assignment.get(arena.decision_symbol(*decision)) {
                    Some(SymbolValue::Int(value)) => value,
                    Some(_) => return Err(SpecializationError::NonIntegerDecision(*decision)),
                    None => return Err(SpecializationError::MissingDecision(*decision)),
                };
                let selected = options
                    .iter()
                    .find_map(|(candidate, body)| (*candidate == value).then_some(body))
                    .ok_or(SpecializationError::AbsentChoiceValue {
                        decision: *decision,
                        value,
                    })?;
                collect_active_launches(selected, schedule, arena, assignment, launches)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::{AllocationPlan, Construction};
    use crate::target::{
        IntrinsicIdentityBuilder, IntrinsicNumericalSemantics, KernelDialect, VectorSupport,
    };
    use seismic_lang::expr::{FiniteDomain, SymbolValue};

    #[derive(Debug)]
    struct Dialect;
    #[derive(Clone, Debug)]
    enum NoIntrinsic {}

    impl KernelDialect for Dialect {
        const NAME: seismic_lang::registry::BackendName = seismic_lang::registry::BackendName::Cpu;
        type Facts = ();
        type Intrinsic = NoIntrinsic;

        fn write_intrinsic_identity(op: &NoIntrinsic, _: &mut IntrinsicIdentityBuilder) {
            match *op {}
        }
        fn intrinsic_numerics(
            _: &(),
            _: &seismic_lang::registry::IntrinsicSignature,
            op: &NoIntrinsic,
        ) -> IntrinsicNumericalSemantics {
            match *op {}
        }
        fn intrinsic_addressable_resources(
            op: &NoIntrinsic,
        ) -> Vec<crate::kernel::ops::AddressableResourceHandle> {
            match *op {}
        }
    }

    fn choice_executable(
        arena: &mut ExprArena,
    ) -> (ClosedExecutableIr<Dialect>, DecisionId, [KernelId; 3]) {
        let mut construction = Construction::<Dialect>::new(arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let first = construction
            .portable_kernel(arena, &(), &[], &vectors)
            .close();
        let second = construction
            .portable_kernel(arena, &(), &[], &vectors)
            .close();
        let shared = construction
            .portable_kernel(arena, &(), &[], &vectors)
            .close();
        let decision = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let condition = arena.bool(false);
        let mut schedule = construction.schedule(arena, 0);
        schedule.choose(decision, |choice| {
            choice.option(0, |selected| {
                selected.launch_sequential(first);
                selected.launch_sequential(shared);
            });
            choice.option(1, |selected| {
                selected.branch(
                    condition,
                    |then_steps| {
                        then_steps.launch_sequential(second);
                    },
                    |else_steps| {
                        else_steps.launch_sequential(shared);
                    },
                );
            });
        });
        let token = schedule.close();
        let executable = construction
            .close(token)
            .analyze_allocations()
            .apply_allocation_plan(arena, AllocationPlan::distinct())
            .finish()
            .close_execution(arena);
        (executable, decision, [first, second, shared])
    }

    #[test]
    fn specialization_selects_one_choice_and_preserves_both_runtime_arms() {
        let mut arena = ExprArena::default();
        let (executable, decision, [first, second, shared]) = choice_executable(&mut arena);

        let mut zero = PartialAssignment::new();
        zero.bind(arena.decision_symbol(decision), SymbolValue::Int(0));
        let zero = executable.specialize_for_native(&arena, &zero).unwrap();
        assert_eq!(
            zero.kernels().map(|(id, _)| id).collect::<Vec<_>>(),
            vec![first, shared]
        );
        assert_eq!(zero.launches().len(), 2);
        assert_eq!(zero.native_kernel_index(first), Some(0));
        assert_eq!(zero.native_kernel_index(shared), Some(1));
        assert_eq!(zero.native_kernel_index(second), None);

        let mut one = PartialAssignment::new();
        one.bind(arena.decision_symbol(decision), SymbolValue::Int(1));
        let one = executable.specialize_for_native(&arena, &one).unwrap();
        assert_eq!(
            one.kernels().map(|(id, _)| id).collect::<Vec<_>>(),
            vec![second, shared]
        );
        assert_eq!(one.launches().len(), 2);
    }

    #[test]
    fn specialization_rejects_every_non_exact_choice_binding() {
        let mut arena = ExprArena::default();
        let (executable, decision, _) = choice_executable(&mut arena);
        assert_eq!(
            executable
                .specialize_for_native(&arena, &PartialAssignment::new())
                .err(),
            Some(SpecializationError::MissingDecision(decision))
        );

        let mut wrong_sort = PartialAssignment::new();
        wrong_sort.bind(arena.decision_symbol(decision), SymbolValue::Bool(false));
        assert_eq!(
            executable.specialize_for_native(&arena, &wrong_sort).err(),
            Some(SpecializationError::NonIntegerDecision(decision))
        );

        let mut absent = PartialAssignment::new();
        absent.bind(arena.decision_symbol(decision), SymbolValue::Int(7));
        assert_eq!(
            executable.specialize_for_native(&arena, &absent).err(),
            Some(SpecializationError::AbsentChoiceValue { decision, value: 7 })
        );
    }
}
