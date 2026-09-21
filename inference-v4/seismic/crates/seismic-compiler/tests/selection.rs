//! Strategy selection over a large parallel domain: the solver must not
//! execute a large `parallel for` on the single-participant serial witness.
//! Millisecond loop for selection bugs (no engine launch required).

use seismic_lang::logical::specialization::{ShapeBinding, SpecializationDomain};
use seismic_lang::logical::{construct, EffectiveTargetIdentity};
use seismic_lang::program::{compile, SourceFile};
use seismic_lang::types::TensorType;
use seismic_realization::kernel::{
    ExecutableDialect, IntrinsicCatalog, IntrinsicConsequences, IntrinsicOperand,
    IntrinsicReferences, IntrinsicResult,
};
use seismic_realization::plan_space::{form_plan_space, NumericalContext};
use seismic_realization::strategy::{MappingCatalog, MappingRule};
use seismic_realization::target::{EffectiveTargetProfile, TargetLimits};
use seismic_lang::sir::IntrinsicUse;
use seismic_realization::kernel::CostUnit;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
struct NoIntrinsic;

struct SelectionDialect;

impl seismic_realization::kernel::sealed::Sealed for SelectionDialect {}

impl ExecutableDialect for SelectionDialect {
    type Intrinsic = NoIntrinsic;
    type LayoutTemplate = ();
    type ResolvedLayout = ();

    fn intrinsic_references(_: &NoIntrinsic) -> IntrinsicReferences {
        IntrinsicReferences::default()
    }
    fn intrinsic_consequences(_: &NoIntrinsic) -> IntrinsicConsequences {
        unreachable!("the test target authorizes no capability")
    }
    fn public_layout(_: &TensorType, _: seismic_realization::kernel::PlaneRef) {}
    fn internal_layout(_: &TensorType, _: seismic_realization::kernel::PlaneRef) {}
    fn resolve_layout(_: &(), _: &seismic_realization::plan_space::SolvedValues) {}
}

struct NoCatalog;

impl IntrinsicCatalog<SelectionDialect> for NoCatalog {
    fn lower(
        &self,
        _: &IntrinsicUse,
        _: &[IntrinsicOperand],
        _: IntrinsicResult,
    ) -> NoIntrinsic {
        unreachable!("the test target authorizes no capability")
    }
}

struct UnitCost;

impl seismic_realization::strategy::CostModel for UnitCost {
    fn launch_overhead_ns(&self) -> u64 {
        200
    }
    fn point_cost_ns(&self, _: &CostUnit) -> u64 {
        2
    }
}

struct NoRules;

impl MappingCatalog<SelectionDialect> for NoRules {
    fn rules(&self) -> &[Box<dyn MappingRule>] {
        &[]
    }
    fn intrinsics(&self) -> &dyn IntrinsicCatalog<SelectionDialect> {
        &NoCatalog
    }
    fn cost_model(&self) -> &dyn seismic_realization::strategy::CostModel {
        &UnitCost
    }
    fn native_fact_domains(&self) -> &[seismic_realization::strategy::NativeFactDomain] {
        &[]
    }
    fn limits(&self) -> &TargetLimits {
        unreachable!("the universal rules consult no backend limits")
    }
}

fn profile() -> EffectiveTargetProfile {
    EffectiveTargetProfile {
        backend: "cpu".to_string(),
        capability_fingerprint: "test".into(),
        toolchain_fingerprint: "test".into(),
        effective_signatures: BTreeSet::new(),
        limits: TargetLimits {
            max_participants: 1024,
            max_workgroups_axis: [65535; 3],
            max_workgroup_bytes: 32768,
            max_explicit_private_bytes: 4096,
            cooperative_grid: None,
            max_device_bytes: 1 << 40,
            max_direct_bindings: 31,
        },
    }
}

#[test]
fn large_parallel_work_selects_a_parallel_strategy() {
    let source = "fn import[B](src: &tensor[B * 48] u32, out: &mut tensor[B * 48] u32):\n    parallel for b in 0..B:\n        out[b * 48 : (b + 1) * 48] = src[b * 48 : (b + 1) * 48]\n";
    let program = compile(&[SourceFile {
        path: "test.seismic".into(),
        text: source.into(),
    }])
    .expect("source checks");
    let shapes: BTreeMap<String, ShapeBinding> =
        [("B".to_string(), ShapeBinding::Exact(100_000))]
            .into_iter()
            .collect();
    let domain = SpecializationDomain::new(&program, "import", shapes, BTreeMap::new())
        .expect("domain binds");
    let target = EffectiveTargetIdentity {
        backend: "cpu".into(),
        capability_fingerprint: "test".into(),
    };
    let supports = |_: &IntrinsicUse| Ok(());
    let logical = construct(&program, &target, &supports, &domain).expect("logical");
    let space = form_plan_space(&logical, &profile(), &NoRules).expect("plan space");
    let policy = seismic_lang::precision::PrecisionPolicy::Exact;
    let numerics = NumericalContext {
        precision: &policy,
        evidence: &[],
    };
    let model = space.solver_model(&numerics);
    for fact in model.facts() {
        for (_, block) in fact.consequences.blocks.entries() {
            println!(
                "CANDIDATE occurrence={} strategy={} participants={:?} cost={}",
                fact.occurrence.0, fact.strategy.0, block.participants, fact.consequences.cost,
            );
        }
    }
    let budget = seismic_compiler::planning::Budget::default();
    let assignment =
        seismic_compiler::planning::plan(&space, &numerics, budget).expect("planning succeeds");
    let mut selected_participants = Vec::new();
    for (occurrence, selection) in assignment.selections() {
        let Some(strategy) = *selection else {
            panic!("the entry occurrence is never inactive");
        };
        let fact = model
            .facts()
            .iter()
            .find(|fact| fact.occurrence == *occurrence && fact.strategy == strategy)
            .expect("the selected strategy is a modeled fact");
        for (_, block) in fact.consequences.blocks.entries() {
            selected_participants.push(block.participants.clone());
        }
    }
    println!("SELECTED participants={selected_participants:?}");
    assert!(
        selected_participants.iter().any(|p| {
            p.as_constant().map(|c| c > 1).unwrap_or(false)
                || p.atoms().iter().any(|a| {
                    matches!(a, seismic_lang::sym::Atom::Param(name) if name.starts_with("participants"))
                })
        }),
        "a large parallel domain must not be executed by a single participant: {selected_participants:?}"
    );
}
