//! Millisecond repro: logical construction of `qwen_embedding_rows` with the
//! engine's exact element bindings (a packed representation table). The
//! engine hit "a result leaf on storage N is not fully initialized" here;
//! this test constructs the logical program alone — no device, no weights.

use magnitude_engine::models::qwen35::program::program;
use seismic_lang::logical::specialization::{ShapeBinding, SpecializationDomain};
use seismic_lang::logical::{construct, EffectiveTargetIdentity};
use seismic_lang::types::Elem;
use std::collections::BTreeMap;

#[test]
fn qwen_embedding_rows_constructs_with_a_packed_table() {
    let program = program().expect("the model program checks");
    let shapes: BTreeMap<String, ShapeBinding> = [
        ("M".to_string(), ShapeBinding::Exact(1)),
        ("V".to_string(), ShapeBinding::Exact(248_320)),
        ("D".to_string(), ShapeBinding::Exact(2_560)),
    ]
    .into_iter()
    .collect();
    // The engine binds the table's element to the imported q6k resident
    // representation and `A` to the activation dtype.
    let elems: BTreeMap<String, Elem> = [
        ("EW".to_string(), Elem::Repr("q6k".into())),
        ("A".to_string(), Elem::Dtype(seismic_lang::types::DType::BF16)),
    ]
    .into_iter()
    .collect();
    let domain = SpecializationDomain::new(&program, "qwen_embedding_rows", shapes, elems)
        .expect("the domain binds");
    let target = EffectiveTargetIdentity {
        backend: "metal".into(),
        capability_fingerprint: "test".into(),
    };
    let supports = |_: &seismic_lang::sir::IntrinsicUse| Ok::<(), String>(());
    match construct(&program, &target, &supports, &domain) {
        Ok(_) => {}
        Err(error) => panic!("construction failed: {error:?}"),
    }
}

mod formation {
    use super::*;
    use magnitude_engine::preparation::Program;
    use seismic_lang::types::TensorType;
    use seismic_realization::kernel::{
        ExecutableDialect, IntrinsicCatalog, IntrinsicConsequences, IntrinsicOperand,
        IntrinsicReferences, IntrinsicResult, PlaneRef,
    };
    use seismic_realization::plan_space::{form_plan_space, NumericalContext};
    use seismic_realization::strategy::{MappingCatalog, MappingRule};
    use seismic_realization::target::{EffectiveTargetProfile, TargetLimits};
    use seismic_lang::sir::IntrinsicUse;
    use seismic_realization::kernel::CostUnit;
    use std::collections::{BTreeMap, BTreeSet};

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct NoIntrinsic;

    struct Dialect;

    impl seismic_realization::kernel::sealed::Sealed for Dialect {}

    impl ExecutableDialect for Dialect {
        type Intrinsic = NoIntrinsic;
        type LayoutTemplate = ();
        type ResolvedLayout = ();
        fn intrinsic_references(_: &NoIntrinsic) -> IntrinsicReferences {
            IntrinsicReferences::default()
        }
        fn intrinsic_consequences(_: &NoIntrinsic) -> IntrinsicConsequences {
            unreachable!("no capability")
        }
        fn public_layout(_: &TensorType, _: PlaneRef) {}
        fn internal_layout(_: &TensorType, _: PlaneRef) {}
        fn resolve_layout(_: &(), _: &seismic_realization::plan_space::SolvedValues) {}
    }

    struct NoCatalog;
    impl IntrinsicCatalog<Dialect> for NoCatalog {
        fn lower(&self, _: &IntrinsicUse, _: &[IntrinsicOperand], _: IntrinsicResult) -> NoIntrinsic {
            unreachable!("no capability")
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
    impl MappingCatalog<Dialect> for NoRules {
        fn rules(&self) -> &[Box<dyn MappingRule>] {
            &[]
        }
        fn intrinsics(&self) -> &dyn IntrinsicCatalog<Dialect> {
            &NoCatalog
        }
        fn cost_model(&self) -> &dyn seismic_realization::strategy::CostModel {
            &UnitCost
        }
        fn native_fact_domains(&self) -> &[seismic_realization::strategy::NativeFactDomain] {
            &[]
        }
        fn limits(&self) -> &TargetLimits {
            unreachable!("no limits")
        }
    }

    fn profile() -> EffectiveTargetProfile {
        EffectiveTargetProfile {
            backend: "cpu".into(),
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

    fn logical() -> seismic_lang::logical::LogicalProgram {
        let program: Program = program().expect("program");
        let shapes: BTreeMap<String, ShapeBinding> = [
            ("M".to_string(), ShapeBinding::Exact(1)),
            ("V".to_string(), ShapeBinding::Exact(248_320)),
            ("D".to_string(), ShapeBinding::Exact(2_560)),
        ]
        .into_iter()
        .collect();
        let elems: BTreeMap<String, Elem> = [
            ("EW".to_string(), Elem::Repr("q6k".into())),
            ("A".to_string(), Elem::Dtype(seismic_lang::types::DType::BF16)),
        ]
        .into_iter()
        .collect();
        let domain = SpecializationDomain::new(&program, "qwen_embedding_rows", shapes, elems)
            .expect("domain");
        let target = EffectiveTargetIdentity {
            backend: "cpu".into(),
            capability_fingerprint: "test".into(),
        };
        let supports = |_: &IntrinsicUse| Ok::<(), String>(());
        construct(&program, &target, &supports, &domain).expect("logical")
    }

    #[test]
    fn qwen_embedding_rows_forms_completely() {
        let logical = logical();
        let space = form_plan_space(&logical, &profile(), &NoRules);
        match space {
            Ok(_) => {}
            Err(error) => panic!("formation failed: {error}"),
        }
    }
}
