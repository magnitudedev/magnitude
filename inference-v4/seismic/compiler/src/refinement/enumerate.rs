//! Deterministic enumeration of the structural candidate-family domain.

use super::{CandidateFamily, ConstructionAuthority, RefinementBudget, RefinementRequest};
use crate::{
    errors::PreparationError,
    implementation::{
        Applicability, CallSite, FunctionContract, ImplementationBuilder, ImplementationFactory,
    },
    portable::{AuthoredSemanticFactory, PortableFactory, PortableParallelFactory},
};
use seismic_lang::{entry::CandidateKind, expr::ExprArena};

pub(super) struct EnumeratedCandidateFamilies<B: seismic_target::TargetFamily> {
    pub universal: CandidateFamily<B>,
    pub optimized: Vec<CandidateFamily<B>>,
    pub registered_factory_traversal_complete: bool,
}

/// Constructs the complete family domain allowed by `budget`.  This layer
/// performs no native formation, estimation, measurement, solving, or
/// retention callback; its output is an ordinary inspectable value.
pub(super) fn enumerate_candidate_families<B: seismic_target::TargetFamily>(
    arena: &mut ExprArena,
    request: RefinementRequest<'_, B>,
    budget: RefinementBudget,
) -> Result<EnumeratedCandidateFamilies<B>, PreparationError> {
    let RefinementRequest {
        program,
        schema,
        target,
        registry,
        constants,
        precision,
    } = request;
    let family = program.family(program.root());
    let reference = family.reference();
    let reference_candidate = reference.candidate();
    let reference_function = program.function(reference.function());
    let reference_contract = FunctionContract::derive(reference_function);
    let reference_factory = PortableFactory;
    let reference_request = crate::implementation::FactoryRequest {
        function: reference_function,
        program,
        contract: &reference_contract,
        target,
        constants,
        precision,
        candidate_kind: reference_candidate.kind,
        numerical_role: reference_candidate.numerical,
        semantic_coverage: reference_candidate.applicability,
        site: CallSite::Root,
    };
    if let Applicability::NotApplicable { reason } =
        reference_factory.applicable(&reference_request)
    {
        panic!("checked reference portable body declined construction: {reason}");
    }
    let reference_builder = ImplementationBuilder::new_universal(
        arena,
        program,
        reference_function,
        target,
        registry,
        constants,
        precision,
        reference_candidate.applicability,
        <PortableFactory as ImplementationFactory<B>>::identity(&reference_factory),
        reference_candidate.numerical,
        CallSite::Root,
        Some(schema),
        budget.clone(),
    );
    let universal = reference_factory.construct(&reference_request, reference_builder);
    let mut optimized = Vec::new();
    let mut traversal_complete = true;

    'candidates: for candidate in std::iter::once(reference_candidate).chain(family.alternatives())
    {
        let is_reference = std::ptr::eq(candidate, reference_candidate);
        let backend_match = match candidate.kind {
            CandidateKind::Portable => true,
            CandidateKind::Lowering { backend } | CandidateKind::Helper { backend } => {
                backend == B::NAME
            }
        };
        if !backend_match
            || candidate
                .requires
                .iter()
                .any(|capability| !target.supports_capability(*capability))
        {
            continue;
        }
        let function = program.function(candidate.function);
        let contract = FunctionContract::derive(function);
        let site = CallSite::Root;
        if !is_reference {
            if !budget.borrow_mut().admit_optional_implementation() {
                traversal_complete = false;
                break;
            }
            let factory: &dyn ImplementationFactory<B> = match candidate.kind {
                CandidateKind::Portable => &PortableFactory,
                CandidateKind::Lowering { .. } | CandidateKind::Helper { .. } => {
                    &AuthoredSemanticFactory
                }
            };
            let request = crate::implementation::FactoryRequest {
                function,
                program,
                contract: &contract,
                target,
                constants,
                precision,
                candidate_kind: candidate.kind,
                numerical_role: candidate.numerical,
                semantic_coverage: candidate.applicability,
                site,
            };
            match factory.applicable(&request) {
                Applicability::Applicable => {
                    let started = std::time::Instant::now();
                    let builder = ImplementationBuilder::new(
                        arena,
                        program,
                        function,
                        target,
                        registry,
                        constants,
                        precision,
                        candidate.applicability,
                        factory.identity(),
                        ConstructionAuthority::Optimized,
                        candidate.numerical,
                        site,
                        Some(schema),
                        budget.clone(),
                    );
                    let family = factory.construct(&request, builder);
                    optimized.push(family);
                    if !budget
                        .borrow_mut()
                        .record_implementation_construction(started.elapsed())
                    {
                        traversal_complete = false;
                        break;
                    }
                }
                Applicability::NotApplicable { reason } => {
                    panic!("checked portable alternative declined construction: {reason}")
                }
            }
        }
        if candidate.kind == CandidateKind::Portable {
            if !budget.borrow_mut().admit_optional_implementation() {
                traversal_complete = false;
                break;
            }
            let factory = PortableParallelFactory;
            let request = crate::implementation::FactoryRequest {
                function,
                program,
                contract: &contract,
                target,
                constants,
                precision,
                candidate_kind: candidate.kind,
                numerical_role: candidate.numerical,
                semantic_coverage: candidate.applicability,
                site,
            };
            match factory.applicable(&request) {
                Applicability::Applicable => {
                    let started = std::time::Instant::now();
                    let builder = ImplementationBuilder::new(
                        arena,
                        program,
                        function,
                        target,
                        registry,
                        constants,
                        precision,
                        candidate.applicability,
                        <PortableParallelFactory as ImplementationFactory<B>>::identity(&factory),
                        ConstructionAuthority::Optimized,
                        candidate.numerical,
                        site,
                        Some(schema),
                        budget.clone(),
                    );
                    let family = factory.construct(&request, builder);
                    optimized.push(family);
                    if !budget
                        .borrow_mut()
                        .record_implementation_construction(started.elapsed())
                    {
                        traversal_complete = false;
                        break;
                    }
                }
                Applicability::NotApplicable { reason } => {
                    panic!("checked portable body declined parallel construction: {reason}")
                }
            }
        }
        // Backend structural policies are additive alternatives to portable
        // semantics. They never replace an authored lowering/helper.
        if candidate.kind != CandidateKind::Portable {
            continue;
        }
        for factory in registry.factories() {
            let request = crate::implementation::FactoryRequest {
                function,
                program,
                contract: &contract,
                target,
                constants,
                precision,
                candidate_kind: candidate.kind,
                numerical_role: candidate.numerical,
                semantic_coverage: candidate.applicability,
                site,
            };
            match factory.applicable(&request) {
                Applicability::Applicable => {
                    if !budget.borrow_mut().admit_optional_implementation() {
                        traversal_complete = false;
                        break 'candidates;
                    }
                    let started = std::time::Instant::now();
                    let builder = ImplementationBuilder::new(
                        arena,
                        program,
                        function,
                        target,
                        registry,
                        constants,
                        precision,
                        candidate.applicability,
                        factory.identity(),
                        ConstructionAuthority::Optimized,
                        candidate.numerical,
                        site,
                        Some(schema),
                        budget.clone(),
                    );
                    let family = factory.construct(&request, builder);
                    optimized.push(family);
                    if !budget
                        .borrow_mut()
                        .record_implementation_construction(started.elapsed())
                    {
                        traversal_complete = false;
                        break 'candidates;
                    }
                }
                Applicability::NotApplicable { .. } => {}
            }
        }
    }

    Ok(EnumeratedCandidateFamilies {
        universal,
        optimized,
        registered_factory_traversal_complete: traversal_complete,
    })
}
