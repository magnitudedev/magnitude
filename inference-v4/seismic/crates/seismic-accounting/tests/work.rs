use seismic_accounting::{
    quantity::Count,
    work::{self, WorkKind},
};
use seismic_lang::{
    hir::Builtin,
    program::{collect_files, compile, SourceFile},
    types::DType,
    Scope,
};
use std::{collections::HashMap, path::PathBuf};

fn standard() -> seismic_lang::program::Program {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../seismic-std/lib");
    compile(
        &collect_files(&[root]).unwrap(),
        &["metal".into(), "cpu".into()],
    )
    .unwrap_or_else(|e| {
        panic!(
            "{}",
            e.iter().map(|e| e.render()).collect::<Vec<_>>().join("\n")
        )
    })
}

#[test]
fn packed_projection_derives_fma_and_decode_without_a_kernel_formula() {
    let p = standard();
    let shapes = HashMap::from([("N".into(), 128), ("K".into(), 256)]);
    let a = work::derive(&p, "projection", &shapes).unwrap();
    assert!(a.is_exact(), "{a:?}");
    let fma = a
        .terms
        .iter()
        .filter(|t| {
            matches!(
                t.kind,
                WorkKind::Builtin {
                    operation: Builtin::Fma,
                    dtype: DType::F32
                }
            )
        })
        .fold(Count::Exact(0), |n, t| n.add(&t.count));
    assert_eq!(fma, Count::Exact(128 * 256));
    let decode = a
        .terms
        .iter()
        .filter(|t| matches!(t.kind, WorkKind::Decode { .. }))
        .fold(Count::Exact(0), |n, t| n.add(&t.count));
    assert_eq!(decode, Count::Exact(128 * 256));
}

#[test]
fn rsqrt_remains_its_own_operation_class() {
    let p = standard();
    let shapes = HashMap::from([("R".into(), 3), ("W".into(), 64)]);
    let elements = ["T", "U", "V"]
        .into_iter()
        .map(|p| (p.into(), seismic_lang::types::Elem::Dtype(DType::BF16)))
        .collect();
    let a = work::derive_specialized(&p, "rms_norm", &shapes, &elements).unwrap();
    assert!(a.is_exact(), "{a:?}");
    let rsqrt = a
        .terms
        .iter()
        .find(|t| {
            matches!(
                t.kind,
                WorkKind::Builtin {
                    operation: Builtin::Rsqrt,
                    ..
                }
            )
        })
        .unwrap();
    // This counts the portable algorithm as written. Hoisting one rsqrt per row is
    // a realization transformation and must not rewrite the source-derived account.
    assert_eq!(rsqrt.count, Count::Exact(3 * 64));
    let reduction = a
        .terms
        .iter()
        .find(|t| matches!(t.kind, WorkKind::Reduction { .. }))
        .unwrap();
    assert_eq!(reduction.count, Count::Exact(3));
    assert!(matches!(
        reduction.kind,
        WorkKind::Reduction {
            extent: Count::Exact(64),
            ..
        }
    ));
}

#[test]
fn runtime_branches_remain_conditional() {
    let source = SourceFile {
        path: "conditional.seismic.portable".into(),
        scope: Scope::Portable,
        text: "fn choose(x: f32):\n  if x > 0.0:\n    y = exp(x)\n  else:\n    y = x * x\n".into(),
    };
    let p = compile(&[source], &[]).unwrap_or_else(|e| {
        panic!(
            "{}",
            e.iter().map(|e| e.render()).collect::<Vec<_>>().join("\n")
        )
    });
    let a = work::derive(&p, "choose", &HashMap::new()).unwrap();
    assert!(!a.is_exact());
    let exp = a
        .terms
        .iter()
        .find(|t| {
            matches!(
                t.kind,
                WorkKind::Builtin {
                    operation: Builtin::Exp,
                    ..
                }
            )
        })
        .unwrap();
    assert_eq!(exp.conditions.len(), 1);
}

#[test]
fn missing_shapes_fail_instead_of_zero_work() {
    assert!(work::derive(&standard(), "projection", &HashMap::new()).is_err());
}

#[test]
fn publication_and_generic_operand_conversions_are_not_free() {
    let a = work::derive(
        &standard(),
        "projection",
        &HashMap::from([("N".into(), 128), ("K".into(), 256)]),
    )
    .unwrap();
    let count = |from, to| {
        a.terms
            .iter()
            .filter(|t| t.kind == WorkKind::Conversion { from, to })
            .fold(Count::Exact(0), |n, t| n.add(&t.count))
    };
    assert_eq!(count(DType::F32, DType::BF16), Count::Exact(128));
    assert_eq!(count(DType::BF16, DType::F32), Count::Exact(128 * 256));
}

#[test]
fn reduction_account_retains_order_permission() {
    for ordered in [false, true] {
        let p=compile(&[SourceFile{path:"ordered.seismic.portable".into(),scope:Scope::Portable,text:format!("fn ordered(x: tensor[65] f32):\n  t = load(x)\n  s = reduce(t,0,sum,ordered={ordered})\n")}],&[]).unwrap();
        let account = work::derive(&p, "ordered", &HashMap::new()).unwrap();
        let term = account
            .terms
            .iter()
            .find(|t| matches!(t.kind, WorkKind::Reduction { .. }))
            .unwrap();
        assert!(
            matches!(term.kind,WorkKind::Reduction{ordered:permission,extent:Count::Exact(65),..} if permission==ordered)
        );
    }
}

#[test]
fn generic_local_publication_counts_rounding_and_widening() {
    let program = compile(&[SourceFile {path:"local.seismic.portable".into(),scope:Scope::Portable,text:"fn local[N](x: tensor[N] ACTIVATION, out: tensor[N] f32):\n  a = load(x)\n  compact = tile[N] ACTIVATION\n  for i in owned(compact): compact[i] = f32(a[i]) * 1.003\n  y = tile[N] f32\n  for i in owned(y): y[i] = f32(compact[i]) * 1031.0\n  store(y,out)\n".into()}],&[]).unwrap();
    let account = work::derive_specialized(&program,"local",&HashMap::from([("N".into(),4)]),&HashMap::from([("ACTIVATION".into(),seismic_lang::types::Elem::Dtype(DType::BF16))])).unwrap();
    assert!(account.is_exact());
    let count=|from,to|account.terms.iter().filter(|t|t.kind==WorkKind::Conversion{from,to}).fold(Count::Exact(0),|n,t|n.add(&t.count));
    assert_eq!(count(DType::F32,DType::BF16),Count::Exact(4));
    assert_eq!(count(DType::BF16,DType::F32),Count::Exact(8));
}
