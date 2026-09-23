//! Read-only prediction over the authoritative kernel and schedule structures.
//! Model implementations supply immutable target rules and service facts.
//! The traversal has no native compiler, executor, or artifact handle input.

use crate::*;
use seismic_ir::kernel::{ops::ClosedOpView, Kernel};
use seismic_ir::schedule::Launch;
use seismic_ir::storage::LaunchLocalLayout;
use seismic_ir::target::{KernelEmissionLayout, PhysicalDialect};
use seismic_lang::expr::ExprArena;

/// Pure backend analytical vocabulary. It declares the complete backend service
/// set and exhaustive cost transfer without owning device observations.
pub trait AnalyticalService: Copy + Send + Sync + 'static {
    const ALL: &'static [Self];
    fn stable_name(self) -> &'static str;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalyticalServiceState {
    Available,
    Unsupported { reason: &'static str },
}

/// Declares a finite analytical service vocabulary from one variant list.
/// The enum, exhaustive `ALL`, and semantic identity match are generated from
/// the same input so no variant can be omitted from profile certification.
#[macro_export]
macro_rules! analytical_services {
    ($vis:vis enum $name:ident { $($variant:ident => $stable_name:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, PartialEq, Eq)]
        $vis enum $name { $($variant),+ }

        impl $crate::AnalyticalService for $name {
            const ALL: &'static [Self] = &[$(Self::$variant),+];

            fn stable_name(self) -> &'static str {
                match self { $(Self::$variant => $stable_name),+ }
            }
        }

    };
}

pub trait AnalyticalModelDefinition<T: PhysicalDialect>: Send + Sync + 'static {
    type Service: AnalyticalService + PartialEq;

    fn model_revision(&self) -> &'static str;

    fn service_state(
        &self,
        _facts: &T::Facts,
        _supported_intrinsics: &std::collections::BTreeSet<seismic_lang::ids::IntrinsicId>,
        _service: Self::Service,
    ) -> AnalyticalServiceState {
        AnalyticalServiceState::Available
    }

    fn operation_cost(
        &self,
        facts: &T::Facts,
        supported_intrinsics: &std::collections::BTreeSet<seismic_lang::ids::IntrinsicId>,
        arena: &mut ExprArena,
        kernel: &Kernel<T>,
        emission: &KernelEmissionLayout,
        launch: &Launch<T>,
        locals: &LaunchLocalLayout,
        op: ClosedOpView<'_, T>,
    ) -> Result<OperationCost<Self::Service>, ModelLimitation>;
}

pub trait ExecutionModel<B: PhysicalDialect>: ServiceModel {
    fn emission_layout(&self, kernel: &Kernel<B>) -> KernelEmissionLayout;
    /// Exhaustive analytical cost for one authoritative closed operation.
    /// Implementations may use only the immutable model and supplied IR
    /// views. The return type cannot represent an accidentally empty cost.
    fn operation_cost(
        &self,
        arena: &mut ExprArena,
        kernel: &Kernel<B>,
        emission: &KernelEmissionLayout,
        launch: &Launch<B>,
        locals: &LaunchLocalLayout,
        op: ClosedOpView<'_, B>,
    ) -> Result<OperationCost<ServiceClassId>, ModelLimitation>;
}

pub fn estimate<B: PhysicalDialect, M: ExecutionModel<B> + ?Sized>(
    profile: &M,
    arena: &mut ExprArena,
    executable: seismic_ir::execution::ClosedExecutionView<'_, B>,
) -> Result<TotalPerformanceModel, ModelLimitation> {
    fn core(service: CoreService) -> ServiceClassId {
        ServiceClassId::new(service.stable_name())
    }
    #[derive(Clone)]
    struct Fragment {
        duration: seismic_lang::expr::DurationExpr,
        contributions: Vec<ModeledContribution>,
    }

    impl Fragment {
        fn empty(arena: &mut ExprArena) -> Self {
            Self {
                duration: arena.duration(&[]),
                contributions: Vec::new(),
            }
        }

        fn service<M: ServiceModel + ?Sized>(
            profile: &M,
            arena: &mut ExprArena,
            service: ServiceClassId,
            units: seismic_lang::expr::NatExpr,
            mode: DemandMode,
            invocation: InvocationProvenance,
        ) -> Self {
            let contribution =
                service_contribution(profile, arena, service, units, mode, invocation);
            Self {
                duration: contribution.duration,
                contributions: vec![contribution],
            }
        }

        fn append(&mut self, arena: &mut ExprArena, other: Self) {
            self.duration = arena.duration_add(self.duration, other.duration);
            self.contributions.extend(other.contributions);
        }

        fn guard_evidence(&mut self, arena: &mut ExprArena, guard: seismic_lang::expr::BoolExpr) {
            let zero_units = arena.nat(0);
            let zero_duration = arena.duration(&[]);
            for contribution in &mut self.contributions {
                contribution.region.guards.push(guard);
                contribution.units = arena.nat_select(guard, contribution.units, zero_units);
                contribution.duration =
                    arena.duration_select(guard, contribution.duration, zero_duration);
            }
        }
    }

    let one = arena.nat(1);
    let mut launches = Vec::with_capacity(executable.launches().len());
    for (launch_ordinal, launch, local_layout, kernel) in executable.launches() {
        let emission = profile.emission_layout(kernel);
        let grid = arena.nat_product(&launch.grid);
        let workgroup = arena.nat_product(&launch.workgroup);
        let participants = arena.nat_mul(grid, workgroup);
        let invocation = InvocationProvenance::KernelLaunch {
            ordinal: launch_ordinal as u32,
        };
        let mut launch_assessment = Fragment::service(
            profile,
            arena,
            core(CoreService::Submission),
            one,
            DemandMode::DependencyLatency,
            invocation,
        );
        let data = kernel;
        for (block, multiplicity) in data.blocks().iter().zip(data.block_multiplicity()) {
            for op in &block.ops {
                let closed = kernel.closed_op(op, &emission);
                let cost =
                    profile.operation_cost(arena, kernel, &emission, launch, local_layout, closed)?;
                let demands = match cost {
                    OperationCost::Demands(demands) => Some(demands),
                    OperationCost::Elided(_) => None,
                };
                for demand in demands.into_iter().flat_map(OperationDemands::into_iter) {
                    let mut units = match demand.scope {
                        DemandScope::PerParticipant => arena.nat_mul(demand.units, participants),
                        DemandScope::PerLaunch => demand.units,
                    };
                    if let Some(multiplicity) = multiplicity {
                        units = arena.nat_mul(units, *multiplicity);
                    }
                    let contribution = Fragment::service(
                        profile,
                        arena,
                        demand.class,
                        units,
                        demand.mode,
                        invocation,
                    );
                    launch_assessment.append(arena, contribution);
                }
            }
        }
        launches.push(launch_assessment);
    }

    fn sequence<M: ServiceModel + ?Sized>(
        profile: &M,
        arena: &mut ExprArena,
        launches: &[Fragment],
        steps: &[seismic_ir::schedule::ScheduleStep],
    ) -> Result<Fragment, ModelLimitation> {
        let one = arena.nat(1);
        let mut result = Fragment::empty(arena);
        for step in steps {
            let item = match step {
                seismic_ir::schedule::ScheduleStep::Imported { body, .. } => sequence(profile, arena, launches, body)?,
                seismic_ir::schedule::ScheduleStep::BeginAllocationInstance { .. } => Fragment::service(
                    profile, arena, core(CoreService::AllocationInstance), one,
                    DemandMode::DependencyLatency, InvocationProvenance::HostSchedule,
                ),
                seismic_ir::schedule::ScheduleStep::PublishTensor { .. } | seismic_ir::schedule::ScheduleStep::BindArgumentTensor { .. } => Fragment::service(
                    profile, arena, core(CoreService::TensorPublication), one,
                    DemandMode::DependencyLatency, InvocationProvenance::HostSchedule,
                ),
                seismic_ir::schedule::ScheduleStep::Launch(id) => {
                    launches[id.index() as usize].clone()
                }
                seismic_ir::schedule::ScheduleStep::Copy(copy) => Fragment::service(
                    profile,
                    arena,
                    core(CoreService::Copy),
                    copy.bytes,
                    DemandMode::SaturatedCapacity,
                    InvocationProvenance::HostSchedule,
                ),
                seismic_ir::schedule::ScheduleStep::Fill(fill) => Fragment::service(
                    profile,
                    arena,
                    core(CoreService::Fill),
                    fill.bytes,
                    DemandMode::SaturatedCapacity,
                    InvocationProvenance::HostSchedule,
                ),
                seismic_ir::schedule::ScheduleStep::ScalarRead(_) => Fragment::service(
                    profile,
                    arena,
                    core(CoreService::ScalarRead),
                    one,
                    DemandMode::DependencyLatency,
                    InvocationProvenance::HostSchedule,
                ),
                seismic_ir::schedule::ScheduleStep::ScalarMove(_) => Fragment::service(
                    profile,
                    arena,
                    core(CoreService::ScalarMove),
                    one,
                    DemandMode::DependencyLatency,
                    InvocationProvenance::HostSchedule,
                ),
                seismic_ir::schedule::ScheduleStep::EvaluateHost(_) => return Err(ModelLimitation::HostQuantityWidth),
                seismic_ir::schedule::ScheduleStep::Check(_) => Fragment::service(
                    profile,
                    arena,
                    core(CoreService::DataCheck),
                    one,
                    DemandMode::DependencyLatency,
                    InvocationProvenance::HostSchedule,
                ),
                seismic_ir::schedule::ScheduleStep::If {
                    condition,
                    then_steps,
                    else_steps,
                    ..
                } => {
                    let mut then_part = sequence(profile, arena, launches, then_steps)?;
                    let mut else_part = sequence(profile, arena, launches, else_steps)?;
                    let otherwise = arena.not(*condition);
                    then_part.guard_evidence(arena, *condition);
                    else_part.guard_evidence(arena, otherwise);
                    let duration =
                        arena.duration_select(*condition, then_part.duration, else_part.duration);
                    then_part.duration = duration;
                    then_part.contributions.extend(else_part.contributions);
                    then_part
                }
                seismic_ir::schedule::ScheduleStep::Repeat {
                    binder,
                    symbol,
                    start,
                    end,
                    body,
                    ..
                } => {
                    let mut body = sequence(profile, arena, launches, body)?;
                    let upper = arena.nat_max(*end, *start);
                    let trips = arena.nat_sub(upper, *start);
                    for contribution in &mut body.contributions {
                        if arena
                            .free_symbols(seismic_lang::expr::AnyExpr::Nat(contribution.units))
                            .contains(symbol)
                        {
                            contribution.units = arena.nat_fold_range(
                                seismic_lang::expr::FoldOp::Sum,
                                *binder,
                                *start,
                                trips,
                                contribution.units,
                            );
                        } else {
                            contribution.units = arena.nat_mul(contribution.units, trips);
                        }
                        if arena
                            .free_symbols(seismic_lang::expr::AnyExpr::Duration(
                                contribution.duration,
                            ))
                            .contains(symbol)
                        {
                            contribution.duration = arena.duration_sum_range(
                                *binder,
                                *start,
                                trips,
                                contribution.duration,
                            );
                        } else {
                            contribution.duration =
                                arena.duration_scale(contribution.duration, trips);
                        }
                    }
                    if arena
                        .free_symbols(seismic_lang::expr::AnyExpr::Duration(body.duration))
                        .contains(symbol)
                    {
                        body.duration =
                            arena.duration_sum_range(*binder, *start, trips, body.duration);
                    } else {
                        body.duration = arena.duration_scale(body.duration, trips);
                    }
                    body
                }
            };
            result.append(arena, item);
        }
        Ok(result)
    }

    let result = sequence(profile, arena, &launches, executable.schedule().steps())?;
    Ok(TotalPerformanceModel {
        estimate: result.duration,
        contributions: result.contributions,
        maximum_relative_error_basis_points: profile.maximum_relative_error_basis_points(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use seismic_ir::construction::{AllocationPlan, Construction};
    use seismic_ir::kernel::ops::{ConstantValue, ValueType};
    use seismic_ir::schedule::LaunchParticipation;
    use seismic_ir::target::{
        IntrinsicIdentityBuilder, IntrinsicNumericalSemantics, KernelWordLayout, VectorSupport,
    };
    use seismic_lang::expr::Assignment;
    use seismic_lang::types::DType;

    #[derive(Debug)]
    struct Dialect;
    #[derive(Clone, Debug, PartialEq)]
    struct FixtureAbi;
    impl seismic_ir::target::KernelAbiModel<Dialect> for FixtureAbi {
        fn layout(
            &self,
            _: &seismic_ir::kernel::Kernel<Dialect>,
        ) -> seismic_ir::target::KernelAbiLayout {
            seismic_ir::target::KernelAbiLayout {
                footprint: seismic_ir::target::KernelAbiFootprint {
                    bytes: 0,
                    alignment: 1,
                },
                allocations: vec![],
            }
        }
    }
    fn local_policy() -> seismic_ir::target::LocalRealizationPolicy {
        seismic_ir::target::LocalRealizationPolicy {
            workgroup: seismic_ir::target::LocalRealization::NativeDynamic,
            participant: seismic_ir::target::LocalRealization::NativeStatic,
            register: seismic_ir::target::LocalRealization::NativeStatic,
        }
    }

    #[derive(Clone, Debug)]
    enum NoIntrinsic {}
    // Deliberately no Backend implementation, native handle, or executor.
    impl PhysicalDialect for Dialect {
        type LaunchDescriptor = ();
        fn ordinary_launch() -> Self::LaunchDescriptor {
            ()
        }

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
        ) -> Vec<seismic_ir::kernel::ops::AddressableResourceHandle> {
            match *op {}
        }
    }
    struct Model(ServiceDefinition);
    impl ServiceModel for Model {
        fn service(&self, class: ServiceClassId) -> &ServiceDefinition {
            assert_eq!(class, self.0.class);
            &self.0
        }
        fn maximum_relative_error_basis_points(&self) -> u16 {
            0
        }
    }
    impl ExecutionModel<Dialect> for Model {
        fn emission_layout(&self, kernel: &Kernel<Dialect>) -> KernelEmissionLayout {
            KernelEmissionLayout {
                words: KernelWordLayout::for_kernel(kernel),
                bindings: vec![],
                locals: vec![],
                addressable_resources: vec![],
                scalar_args: vec![],
                result_types: vec![],
            }
        }
        fn operation_cost(
            &self,
            _: &mut ExprArena,
            _: &Kernel<Dialect>,
            _: &KernelEmissionLayout,
            _: &Launch<Dialect>,
            _: &LaunchLocalLayout,
            _: ClosedOpView<'_, Dialect>,
        ) -> Result<OperationCost, ModelLimitation> {
            Ok(OperationCost::Elided(ProvenElision::CompileTimeOnly))
        }
    }

    #[test]
    fn real_builders_and_prediction_require_no_native_backend() {
        let model = Model(ServiceDefinition {
            class: ServiceClassId::new("core.submission"),
            correlation: ServiceCorrelationId::new("fixture"),
            qualification: ServiceQualificationDomain {
                minimum_units: 1,
                maximum_units: 1,
                maximum_concurrent_uses: 1,
            },
            accuracy: ServiceAccuracyClass::Compute,
            topology: ResourceTopology {
                resources: 1,
                max_concurrency: 1,
            },
            dependency_latency: DurationInterval::new(7, 7, 1),
            saturated_capacity: ServiceCurve {
                setup: DurationInterval::new(0, 0, 1),
                regimes: vec![],
            },
            provenance: FactProvenance::Derived {
                rule: "exact fixture",
                inputs: Box::new([]),
            },
        });
        for trips in 0..8 {
            let mut arena = ExprArena::default();
            let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
            let vectors = VectorSupport::default();
            let kernel = construction
                .portable_kernel(&mut arena, &(), &[], &vectors)
                .close();
            let one = arena.nat(1);
            let start = arena.nat(2);
            let end = arena.nat(2 + trips);
            let outer_start = arena.nat(0);
            let outer_end = arena.nat(2);
            let empty = arena.bool(false);
            let mut builder = construction.schedule(&mut arena, 0);
            builder.repeat(outer_start, outer_end, |outer, _| {
                outer.repeat(start, end, |body, _| {
                    let id = body.launch(Launch {
                        kernel,
                        descriptor: (),
                        grid: [one; 3],
                        workgroup: [one; 3],
                        empty,
                        parallel_extent: None,
                        logical_base: None,
                    });
                    body.step_launch(id);
                });
            });
            let token = builder.close();
            let analyzed = construction
                .close(token)
                .normalize_launches(&mut arena, u64::MAX, 64)
                .unwrap()
                .analyze_allocations();
            let planned = analyzed.apply_allocation_plan(&mut arena, AllocationPlan::distinct());
            let ir = planned.finish();
            let executable = ir.close_execution(&mut arena, local_policy(), &FixtureAbi);
            let prediction = estimate(&model, &mut arena, executable.view()).unwrap();
            let duration = prediction.estimate();
            assert_eq!(prediction.contributions().len(), 1);
            let contribution = &prediction.contributions()[0];
            assert_eq!(
                contribution.region().invocation(),
                InvocationProvenance::KernelLaunch { ordinal: 0 }
            );
            assert_eq!(
                arena
                    .eval_nat(contribution.units(), &Assignment::new())
                    .unwrap(),
                (trips * 2).into()
            );
            assert!(matches!(
                contribution.evidence(),
                FactProvenance::Derived { .. }
            ));
            let value = arena.eval_duration(duration, &Assignment::new()).unwrap();
            assert_eq!(
                value.upper().numerator(),
                u128::from(trips * 14) * u128::from(value.upper().denominator())
            );
        }
    }

    #[test]
    fn guarded_repeat_is_total_when_an_operation_is_proven_elided() {
        let model = Model(ServiceDefinition {
            class: ServiceClassId::new("core.submission"),
            correlation: ServiceCorrelationId::new("fixture"),
            qualification: ServiceQualificationDomain {
                minimum_units: 1,
                maximum_units: 1,
                maximum_concurrent_uses: 1,
            },
            accuracy: ServiceAccuracyClass::Compute,
            topology: ResourceTopology {
                resources: 1,
                max_concurrency: 1,
            },
            dependency_latency: DurationInterval::new(7, 7, 1),
            saturated_capacity: ServiceCurve {
                setup: DurationInterval::new(0, 0, 1),
                regimes: Vec::new(),
            },
            provenance: FactProvenance::Derived {
                rule: "exact fixture",
                inputs: Box::new([]),
            },
        });
        let mut arena = ExprArena::default();
        let (_, branch_symbol) = arena.target_constant(seismic_lang::expr::SymbolSort::Nat);
        let branch_value = arena.nat_symbol(branch_symbol);
        let zero = arena.nat(0);
        let condition = arena.nat_cmp(seismic_lang::expr::CmpOp::Gt, branch_value, zero);

        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let complete_kernel = construction
            .portable_kernel(&mut arena, &(), &[], &vectors)
            .close();
        let mut incomplete_kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        incomplete_kernel.constant(ConstantValue::U32(1), ValueType::Scalar(DType::U32));
        let incomplete_kernel = incomplete_kernel.close();

        let one = arena.nat(1);
        let two = arena.nat(2);
        let empty = arena.bool(false);
        let launch = |kernel| Launch {
            kernel,
            descriptor: (),
            grid: [one; 3],
            workgroup: [one; 3],
            empty,
            parallel_extent: None,
            logical_base: None,
        };
        let mut schedule = construction.schedule(&mut arena, 0);
        let incomplete = schedule.launch(launch(incomplete_kernel));
        let complete_a = schedule.launch(launch(complete_kernel));
        let complete_b = schedule.launch(launch(complete_kernel));
        schedule.repeat(zero, two, |body, _| {
            body.branch(
                condition,
                |then| {
                    then.step_launch(incomplete);
                },
                |otherwise| {
                    otherwise.step_launch(complete_a);
                    otherwise.step_launch(complete_b);
                },
            );
        });
        let token = schedule.close();
        let analyzed = construction
            .close(token)
            .normalize_launches(&mut arena, u64::MAX, 64)
            .unwrap()
            .analyze_allocations();
        let planned = analyzed.apply_allocation_plan(&mut arena, AllocationPlan::distinct());
        let ir = planned.finish();
        let executable = ir.close_execution(&mut arena, local_policy(), &FixtureAbi);
        let assessment = estimate(&model, &mut arena, executable.view()).unwrap();

        let mut complete_assignment = Assignment::new();
        complete_assignment.bind(branch_symbol, seismic_lang::expr::SymbolValue::Nat(0u32.into()));
        let complete_complement = arena
            .eval_duration(assessment.estimate(), &complete_assignment)
            .unwrap();
        assert_eq!(
            complete_complement.upper().numerator(),
            u128::from(28 * complete_complement.upper().denominator())
        );

        let mut incomplete_assignment = Assignment::new();
        incomplete_assignment.bind(branch_symbol, seismic_lang::expr::SymbolValue::Nat(1u32.into()));
        let total = arena
            .eval_duration(assessment.estimate(), &incomplete_assignment)
            .unwrap();
        assert_eq!(
            total.upper().numerator(),
            u128::from(14 * total.upper().denominator())
        );
    }

    #[test]
    fn explicit_operation_elision_still_produces_a_total_model() {
        let definition = ServiceDefinition {
            class: ServiceClassId::new("core.submission"),
            correlation: ServiceCorrelationId::new("fixture"),
            qualification: ServiceQualificationDomain {
                minimum_units: 1,
                maximum_units: 1,
                maximum_concurrent_uses: 1,
            },
            accuracy: ServiceAccuracyClass::Compute,
            topology: ResourceTopology {
                resources: 1,
                max_concurrency: 1,
            },
            dependency_latency: DurationInterval::new(1, 1, 1),
            saturated_capacity: ServiceCurve {
                setup: DurationInterval::new(0, 0, 1),
                regimes: Vec::new(),
            },
            provenance: FactProvenance::Derived {
                rule: "fixture",
                inputs: Box::new([]),
            },
        };
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        kernel.constant(ConstantValue::U32(1), ValueType::Scalar(DType::U32));
        let kernel = kernel.close();
        let one = arena.nat(1);
        let empty = arena.bool(false);
        let mut schedule = construction.schedule(&mut arena, 0);
        let launch = schedule.launch(Launch {
            kernel,
            descriptor: (),
            grid: [one; 3],
            workgroup: [one; 3],
            empty,
            parallel_extent: None,
            logical_base: None,
        });
        schedule.step_launch(launch);
        let token = schedule.close();
        let analyzed = construction
            .close(token)
            .normalize_launches(&mut arena, u64::MAX, 64)
            .unwrap()
            .analyze_allocations();
        let planned = analyzed.apply_allocation_plan(&mut arena, AllocationPlan::distinct());
        let ir = planned.finish();
        let executable = ir.close_execution(&mut arena, local_policy(), &FixtureAbi);
        let assessment = estimate(&Model(definition), &mut arena, executable.view()).unwrap();
        let duration = arena
            .eval_duration(assessment.estimate(), &Assignment::new())
            .unwrap();
        assert_eq!(
            duration.lower().numerator(),
            duration.lower().denominator() as u128
        );
    }
}
