// Included only in controller's test module. The production controller owns all
// proposals, compilation ordering, observations, confirmation and publication.
fn coupled_domain<'ctx>(
    device: &'ctx seismic_native_target::DeviceDescription<FakeTarget>,
    registry: &'ctx crate::target::CompilerRegistry<FakeTarget>,
) -> crate::candidate_domain::CandidateDomain<'ctx, FakeTarget> {
    use crate::refinement::{ChoiceDeclaration, ConstructedCandidate};
    use seismic_ir::construction::AllocationPlan;
    use seismic_ir::construction::Construction;
    use seismic_lang::expr::FiniteDomain;
    let (domain, _) = domain_with_optional(device, registry);
    let mut parts = domain.into_parts();
    let mut materialized = parts.materialized.into_vec();
    let original = materialized.pop().unwrap();
    materialized.truncate(1);
    let mut family = std::sync::Arc::try_unwrap(original.family)
        .unwrap()
        .test_into_parts();
    let arena = &mut parts.arena;
    let mut construction = Construction::<FakeTarget>::new(arena, vec![], false, 0);
    let vectors = seismic_ir::physical_target::VectorSupport::default();
    let kernel = construction
        .portable_kernel(arena, &(), &[], &vectors)
        .close();
    let decisions = (0..6)
        .map(|_| arena.decision(FiniteDomain::new(vec![0, 1]).unwrap()))
        .collect::<Vec<_>>();
    let mut schedule = construction.schedule(arena, 0);
    schedule.launch_sequential(kernel);
    let token = schedule.close();
    family.executable = construction
        .close(token)
        .normalize_launches(arena, u64::MAX, 64)
        .unwrap()
        .analyze_allocations()
        .apply_allocation_plan(arena, AllocationPlan::distinct())
        .finish()
        .close_execution(arena, device.local_realization(), device.kernel_abi());
    let always = arena.bool(true);
    family.choices = decisions
        .iter()
        .map(|decision| ChoiceDeclaration {
            kind: crate::refinement::ChoiceKind::WorkgroupSize,
            decision: *decision,
            meaning: "coupled axis",
            active_when: always,
        })
        .collect();
    family.identity.structure = [61; 32];
    materialized.push(crate::candidate_domain::DomainCandidate {
        construction: original.construction,
        family: std::sync::Arc::new(ConstructedCandidate::test_from_parts(family)),
        constraints: original.constraints,
        numerical: original.numerical,
    });
    parts.materialized = crate::candidate_domain::NonEmpty::new(materialized).unwrap();
    crate::candidate_domain::CandidateDomain::from_parts(parts)
}

fn coupled_latency(candidate: Option<u8>, point: usize) -> u64 {
    let Some(candidate) = candidate else {
        return 1100;
    };
    let target = [0b111111, 0b010110, 0b101001][point];
    let pairs = (0..3)
        .filter(|pair| (candidate >> (pair * 2)) & 3 == (target >> (pair * 2)) & 3)
        .count() as u64;
    1000 - pairs * 100 - if pairs == 3 { 200 } else { 0 }
}

struct CostlyCompiler(CountingCompiler);
impl NativeCompiler<FakeTarget> for CostlyCompiler {
    type Context = ();
    type Candidate = (usize, Duration);
    type Handle = usize;
    fn form(
        &self,
        context: &(),
        target: &seismic_native_target::DeviceDescription<FakeTarget>,
        kernel: &seismic_ir::kernel::Kernel<FakeTarget>,
        layout: &seismic_ir::physical_target::KernelEmissionLayout,
    ) -> Result<Self::Candidate, seismic_native_target::NativeCompilationError> {
        let started = Instant::now();
        let candidate = self.0.form(context, target, kernel, layout)?;
        std::thread::sleep(Duration::from_micros(if candidate % 3 == 0 {
            2500
        } else {
            300
        }));
        Ok((candidate, started.elapsed()))
    }
    fn reflect(
        &self,
        target: &seismic_native_target::DeviceDescription<FakeTarget>,
        kernel: &seismic_ir::kernel::Kernel<FakeTarget>,
        layout: &seismic_ir::physical_target::KernelEmissionLayout,
        candidate: Self::Candidate,
    ) -> Result<
        seismic_native_target::NativeKernelReflection<FakeTarget, usize>,
        seismic_native_target::NativeCompilationError,
    > {
        let reflection = self.0.reflect(target, kernel, layout, candidate.0)?;
        let mut metrics = reflection.metrics();
        metrics.compilation_ns = candidate.1.as_nanos() as u64;
        Ok(seismic_native_target::NativeKernelReflection::new(
            candidate.0,
            reflection.description().clone(),
            metrics,
        ))
    }
}
struct CoupledObserver {
    labels: HashMap<[u8; 32], Option<u8>>,
}
impl<H> ControlledObserver<FakeTarget, H> for CoupledObserver {
    fn environment(&mut self) -> Result<[u8; 32], ObservationError> {
        Ok([19; 32])
    }
    fn observe(
        &mut self,
        request: ObservationRequest<'_, FakeTarget, H>,
    ) -> Result<Observation, ObservationError> {
        let super::super::CaseArgument::Scalar(SymbolValue::F32(value)) = request.case.arguments[0]
        else {
            panic!("scalar case")
        };
        let point = (value.to_bits() - 1.0f32.to_bits()) as usize;
        let candidate = self.labels[&request.executable.identity().assignment];
        let setup = Instant::now();
        std::thread::sleep(Duration::from_micros(if point == 1 { 150 } else { 20 }));
        let setup_time = setup.elapsed();
        std::thread::sleep(Duration::from_micros(30 * request.protocol.trials as u64));
        Ok(Observation {
            support: Default::default(),
            samples: vec![
                Duration::from_nanos(coupled_latency(candidate, point));
                request.protocol.trials
            ],
            setup_time,
            checking_time: Duration::ZERO,
            environment: [19; 32],
        })
    }
}

#[test]
fn coupled_fixture_requires_compound_edits_and_can_combine_donors() {
    use super::super::evolution::{propose, Operator};
    let device = device();
    let registry = registry();
    let domain = coupled_domain(&device, &registry);
    let family = domain
        .constructed()
        .find(|family| family.choices().len() == 6)
        .unwrap();
    let coordinate = |mask: u8| {
        domain
            .canonicalize(
                domain.proposal_for_decisions(
                    family.identity().clone(),
                    family
                        .choices()
                        .iter()
                        .enumerate()
                        .map(|(axis, choice)| (choice.decision(), i64::from(mask >> axis & 1)))
                        .collect(),
                ),
            )
            .unwrap()
    };
    let mask = |coordinate: &CandidateCoordinate| {
        coordinate
            .choices()
            .iter()
            .enumerate()
            .fold(0u8, |mask, (axis, (_, value))| {
                mask | ((*value as u8) << axis)
            })
    };
    let parent = coordinate(0);
    let donor_a = coordinate(0b111100);
    let donor_b = coordinate(0b001111);
    let mut random = Random(42);
    let mut compound_gains = 0;
    let mut crossover_optima = 0;
    for _ in 0..128 {
        let single = propose(&domain, Some(&parent), None, Operator::Mutate, &mut random).unwrap();
        assert_eq!(
            coupled_latency(Some(mask(&single)), 0),
            coupled_latency(Some(0), 0)
        );
        let compound = propose(
            &domain,
            Some(&parent),
            None,
            Operator::Compound,
            &mut random,
        )
        .unwrap();
        compound_gains +=
            usize::from(coupled_latency(Some(mask(&compound)), 0) < coupled_latency(Some(0), 0));
        let crossed = propose(
            &domain,
            Some(&donor_a),
            Some(&donor_b),
            Operator::Crossover,
            &mut random,
        )
        .unwrap();
        crossover_optima += usize::from(mask(&crossed) == 63);
    }
    assert!(compound_gains > 0 && crossover_optima > 0);
    assert_eq!(finite_coordinates(&domain, 65).unwrap().len(), 65);
    assert!(finite_coordinates(&domain, 64).is_none());
}

#[test]
#[ignore = "actual-wall-time coupled-controller qualification; run explicitly with --nocapture"]
fn coupled_cold_cache_controller_qualification() {
    for seed in [1, 7, 29] {
        for (strategy, cost_ordering) in [
            (TestStrategy::Evolution, true),
            (TestStrategy::Fresh, true),
            (TestStrategy::Enumeration, true),
            (TestStrategy::Evolution, false),
        ] {
            let device = device();
            let registry = registry();
            // A separate probe labels semantic assignments. None of these native
            // resources enter the measured campaign's compiler or realizer.
            let probe_compiler = CountingCompiler::new();
            let probe_domain = coupled_domain(&device, &registry);
            let mut probe = EvaluationSession::from_domain(
                probe_domain,
                &registry,
                &probe_compiler,
                &(),
                &device,
                &PreparationBudget::default(),
                &PlanningBudget::default(),
            )
            .unwrap();
            let coordinates = finite_coordinates(probe.domain(), 65).unwrap();
            let mut labels = HashMap::new();
            for coordinate in coordinates {
                let label = (!coordinate.choices().is_empty()).then(|| {
                    coordinate
                        .choices()
                        .iter()
                        .enumerate()
                        .fold(0u8, |mask, (axis, (_, value))| {
                            mask | ((*value as u8) << axis)
                        })
                });
                let RealizationAdmission::Ready(id) =
                    probe.realize_checked(&coordinate).unwrap()
                else {
                    panic!("probe candidate rejected")
                };
                labels.insert(probe.executable(id).unwrap().identity().assignment, label);
            }
            drop(probe);
            let domain = coupled_domain(&device, &registry);
            let symbol = match domain.schema().parameters()[0].kind {
                seismic_lang::entry::ParameterKind::Scalar { symbol, .. } => symbol,
                _ => panic!(),
            };
            let mut scope = InvocationScope::for_entry(domain.entry());
            scope.constrain(
                InvocationParameter::Scalar(0),
                SymbolValue::F32(1.0),
                SymbolValue::F32(f32::from_bits(1.0f32.to_bits() + 2)),
            );
            let compiler = CostlyCompiler(CountingCompiler::new());
            let mut session = EvaluationSession::from_domain(
                domain,
                &registry,
                &compiler,
                &(),
                &device,
                &PreparationBudget::default(),
                &PlanningBudget::default(),
            )
            .unwrap();
            assert_eq!(
                compiler.0.form_count(),
                0,
                "cold campaign must start with no compiled artifact"
            );
            let options = FeedbackOptions {
                seed,
                search_time: Duration::from_millis(120),
                optimize_for: Some(scope),
                ..Default::default()
            };
            let mut evaluator = FeedbackEvaluator::new(
                &session,
                CoupledObserver { labels },
                options,
                Instant::now(),
            )
            .unwrap();
            evaluator.strategy = strategy;
            evaluator.cost_ordering = cost_ordering;
            for checkpoint in [120, 240] {
                let policy = evaluator.evaluate(&mut session).unwrap();
                let finalize_started = Instant::now();
                let kernel = session.finalize(policy.0, policy.1, policy.2).unwrap();
                evaluator.finalized(finalize_started.elapsed(), Some(kernel.planning_report()));
                evaluator.suspend();
                let mut latencies = Vec::new();
                for point in 0..3 {
                    let mut values = InvocationValues::new();
                    values.bind(
                        symbol,
                        SymbolValue::F32(f32::from_bits(1.0f32.to_bits() + point)),
                    );
                    let selected = kernel
                        .variants()
                        .iter()
                        .nth(kernel.select(&values).as_usize())
                        .unwrap();
                    let candidate = evaluator.observer.labels[&selected.identity().assignment];
                    let latency = coupled_latency(candidate, point as usize);
                    assert!(latency <= 1100);
                    latencies.push(latency);
                }
                assert!(
                    compiler.0.form_count() <= 14,
                    "shared native artifacts must compile only once"
                );
                println!("coupled_feedback_v1 seed={seed} strategy={strategy:?} cost_ordering={cost_ordering} cache=cold-start checkpoint_ms={checkpoint} elapsed_us={} candidates={} confirmed={} native_forms={} native_hits={} native_misses={} native_code_bytes={} native_metadata_bytes={} retained_variants={} retained_metadata_bytes={} overrun_us={} point_latencies_ns={latencies:?}",evaluator.report.elapsed.as_micros(),evaluator.report.prepared_candidates,evaluator.report.confirmed_points,compiler.0.form_count(),evaluator.report.native_hits,evaluator.report.native_misses,evaluator.report.native_code_bytes,evaluator.report.native_metadata_bytes,evaluator.report.retained_variants,evaluator.report.retained_metadata_bytes,evaluator.report.elapsed.as_micros().saturating_sub(checkpoint as u128*1000));
                if checkpoint == 120 {
                    evaluator.resume(Duration::from_millis(120));
                }
            }
        }
    }
}

#[test]
#[ignore = "matched-observation invocation and measurement ablations; run explicitly with --nocapture"]
fn invocation_sampling_and_screen_precision_ablations() {
    struct IslandObserver {
        labels: HashMap<crate::implementation::ImplementationIdentity, u8>,
        executions: usize,
    }
    fn truth(candidate: u8, point: u32) -> u64 {
        match candidate {
            1 if (700..=1700).contains(&point) || point >= 6000 => 400,
            2 if (4111..=4113).contains(&point) => 250,
            0 => 1000,
            _ => 1400,
        }
    }
    impl<H> ControlledObserver<FakeTarget, H> for IslandObserver {
        fn environment(&mut self) -> Result<[u8; 32], ObservationError> {
            Ok([31; 32])
        }
        fn observe(
            &mut self,
            request: ObservationRequest<'_, FakeTarget, H>,
        ) -> Result<Observation, ObservationError> {
            self.executions += request.protocol.warmup + request.protocol.trials;
            let super::super::CaseArgument::Scalar(SymbolValue::F32(value)) =
                request.case.arguments[0]
            else {
                panic!("scalar fixture")
            };
            let point = value.to_bits() - 1.0f32.to_bits();
            let candidate = self.labels[&request.executable.identity().implementation];
            Ok(Observation {
                support: Default::default(),
                samples: vec![
                    Duration::from_nanos(truth(candidate, point));
                    request.protocol.trials
                ],
                setup_time: Duration::ZERO,
                checking_time: Duration::ZERO,
                environment: [31; 32],
            })
        }
    }
    const ALLOWANCE: usize = 2048;
    for seed in [1, 7, 29] {
        for (fixed, screen_trials) in [(false, 3), (true, 3), (false, 31)] {
            let device = device();
            let registry = registry();
            let compiler = CountingCompiler::new();
            let (domain, _) = domain_with_optional(&device, &registry);
            let symbol = match domain.schema().parameters()[0].kind {
                seismic_lang::entry::ParameterKind::Scalar { symbol, .. } => symbol,
                _ => panic!(),
            };
            let mut scope = InvocationScope::for_entry(domain.entry());
            scope.constrain(
                InvocationParameter::Scalar(0),
                SymbolValue::F32(1.0),
                SymbolValue::F32(f32::from_bits(1.0f32.to_bits() + 8192)),
            );
            let mut session = EvaluationSession::from_domain(
                domain,
                &registry,
                &compiler,
                &(),
                &device,
                &PreparationBudget::default(),
                &PlanningBudget::default(),
            )
            .unwrap();
            let coordinates = finite_coordinates(session.domain(), 8).unwrap();
            let mut labels = HashMap::new();
            for coordinate in coordinates {
                let label = coordinate
                    .choices()
                    .first()
                    .map_or(0, |(_, value)| 1 + *value as u8);
                let RealizationAdmission::Ready(id) =
                    session.realize_checked(&coordinate).unwrap()
                else {
                    panic!("fixture candidate rejected")
                };
                labels.insert(
                    session
                        .executable(id)
                        .unwrap()
                        .identity()
                        .implementation
                        .clone(),
                    label,
                );
            }
            let options = FeedbackOptions {
                seed,
                search_time: Duration::from_secs(5),
                optimize_for: Some(scope),
                ..Default::default()
            };
            let mut evaluator = FeedbackEvaluator::new(
                &session,
                IslandObserver {
                    labels,
                    executions: 0,
                },
                options,
                Instant::now(),
            )
            .unwrap();
            evaluator.observation_policy.allowance = Some(ALLOWANCE);
            evaluator.observation_policy.screen_trials = screen_trials;
            if fixed {
                // A grid fixed before any performance observation, deliberately
                // independent of the fixture's winner boundaries.
                evaluator.observation_policy.fixed_points = Some(
                    (0..=8192)
                        .step_by(64)
                        .map(|offset| {
                            Point(vec![(1.0f32.to_bits() as u64 + (1u64 << 31)) + offset])
                        })
                        .collect(),
                );
            }
            let policy = evaluator.evaluate(&mut session).unwrap();
            let kernel = session.finalize(policy.0, policy.1, policy.2).unwrap();
            evaluator.suspend();
            assert_eq!(
                evaluator.observer.executions,
                evaluator.observation_policy.spent
            );
            assert!(evaluator.observation_policy.spent <= ALLOWANCE);
            assert!(
                evaluator.observation_policy.exhausted,
                "comparison must hit observation allowance before wall-time safety limit"
            );
            // Evaluate both independently selected semantic probes and every
            // explored point. Exact-point dispatch must not generalize a win.
            let mut points = vec![
                0, 699, 700, 1000, 1700, 1701, 4096, 4111, 4112, 4113, 4114, 5999, 6000, 7000, 8192,
            ];
            points.extend(evaluator.points.iter().map(|population| {
                (population.point.0[0] - (1.0f32.to_bits() as u64 + (1u64 << 31))) as u32
            }));
            points.sort_unstable();
            points.dedup();
            let mut outcomes = Vec::new();
            let mut unseen = 0;
            for point in points {
                let rank = 1.0f32.to_bits() as u64 + (1u64 << 31) + point as u64;
                let sampled = evaluator
                    .points
                    .iter()
                    .any(|population| population.point.0 == vec![rank]);
                let mut values = InvocationValues::new();
                values.bind(
                    symbol,
                    SymbolValue::F32(f32::from_bits(1.0f32.to_bits() + point)),
                );
                let selected = kernel
                    .variants()
                    .iter()
                    .nth(kernel.select(&values).as_usize())
                    .unwrap();
                let candidate = evaluator.observer.labels[&selected.identity().implementation];
                let latency = truth(candidate, point);
                assert!(
                    latency <= 1000,
                    "delivered policy regressed at evaluation point"
                );
                if !sampled {
                    unseen += 1;
                    assert_eq!(candidate, 0, "unseen invocation must retain general policy");
                }
                outcomes.push(format!(
                    "{point}:{}:{latency}",
                    if sampled { "seen" } else { "unseen" }
                ));
            }
            assert!(unseen > 0);
            println!("invocation_precision_v1 seed={seed} fixed={fixed} screen_trials={screen_trials} allowance={ALLOWANCE} spent={} elapsed_us={} attempted={} measured={} confirmed={} unseen={} outcomes={}",evaluator.observation_policy.spent,evaluator.report.elapsed.as_micros(),evaluator.report.attempted_points,evaluator.report.measured_points,evaluator.report.confirmed_points,unseen,outcomes.join(","));
        }
    }
}
