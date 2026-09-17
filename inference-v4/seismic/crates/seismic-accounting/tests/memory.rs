use seismic_accounting::memory::{self, Backing, Bindings};
use seismic_lang::{
    program::{collect_files, compile, Program, SourceFile},
    Scope,
};
use std::{collections::HashMap, path::PathBuf};
fn standard() -> Program {
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
fn source(text: &str) -> Program {
    compile(
        &[SourceFile {
            path: "test.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap_or_else(|e| {
        panic!(
            "{}",
            e.iter().map(|e| e.render()).collect::<Vec<_>>().join("\n")
        )
    })
}
fn bytes(account: &memory::MemoryAccount, name: &str) -> (u64, u64) {
    let a = account
        .accesses
        .iter()
        .find(|(id, _)| *id == name)
        .unwrap()
        .1;
    (a.reads.bytes(), a.writes.bytes())
}
#[test]
fn projection_counts_reuse_and_packed_planes_exactly() {
    let a = memory::derive(
        &standard(),
        "projection",
        &HashMap::from([("N".into(), 128), ("K".into(), 256)]),
        &Bindings::new(),
        100_000,
    )
    .unwrap();
    assert!(a.is_exact(), "{:?}", a.unavailable);
    assert_eq!(bytes(&a, "x"), (512, 0));
    assert_eq!(bytes(&a, "w.words"), (16384, 0));
    assert_eq!(bytes(&a, "w.scale"), (1024, 0));
    assert_eq!(bytes(&a, "w.bias"), (1024, 0));
    assert_eq!(bytes(&a, "out"), (0, 256));
}
#[test]
fn row_norm_reuses_weights_and_tracks_only_actual_rows() {
    let a = memory::derive_specialized(
        &standard(),
        "rms_norm",
        &HashMap::from([("R".into(), 3), ("W".into(), 64)]),
        &HashMap::from([
            (
                "T".into(),
                seismic_lang::types::Elem::Dtype(seismic_lang::types::DType::BF16),
            ),
            (
                "U".into(),
                seismic_lang::types::Elem::Dtype(seismic_lang::types::DType::BF16),
            ),
            (
                "V".into(),
                seismic_lang::types::Elem::Dtype(seismic_lang::types::DType::BF16),
            ),
        ]),
        &Bindings::new(),
        100_000,
    )
    .unwrap();
    assert!(a.is_exact(), "{:?}", a.unavailable);
    assert_eq!(bytes(&a, "x"), (384, 0));
    assert_eq!(bytes(&a, "weight"), (128, 0));
    assert_eq!(bytes(&a, "out"), (0, 384));
}
#[test]
fn budget_exhaustion_does_not_claim_complete_account() {
    let a = memory::derive(
        &standard(),
        "projection",
        &HashMap::from([("N".into(), 128), ("K".into(), 256)]),
        &Bindings::new(),
        1,
    )
    .unwrap();
    assert!(!a.is_exact());
    assert!(a.unavailable.iter().any(|e| e.contains("budget")));
}
#[test]
fn repeated_slice_is_not_capped_touches_and_call_views_preserve_offset() {
    let p=source("fn inner(x: tensor[4] f32):\n  t = load(x)\nfn outer(x: tensor[32] f32):\n  for i in range(20):\n    inner(x[8:12])\n");
    let a = memory::derive(&p, "outer", &HashMap::new(), &Bindings::new(), 10000).unwrap();
    assert!(a.is_exact(), "{:?}", a.unavailable);
    assert_eq!(bytes(&a, "x"), (16, 0));
    let access = a.accesses.iter().next().unwrap().1;
    assert_eq!(access.reads.ranges().collect::<Vec<_>>(), vec![32..48]);
}
#[test]
fn aliased_parameters_union_by_backing_identity() {
    let p =
        source("fn aliases(x: tensor[8] f32, y: tensor[8] f32):\n  a = load(x)\n  b = load(y)\n");
    let bindings = Bindings::from([
        (
            ("x".into(), "".into()),
            Backing {
                identity: "shared".into(),
                byte_offset: 0,
            },
        ),
        (
            ("y".into(), "".into()),
            Backing {
                identity: "shared".into(),
                byte_offset: 16,
            },
        ),
    ]);
    let a = memory::derive(&p, "aliases", &HashMap::new(), &bindings, 1000).unwrap();
    assert!(a.is_exact());
    assert_eq!(bytes(&a, "shared"), (48, 0));
}
#[test]
fn data_dependent_visibility_never_becomes_zero_or_capacity() {
    let p=source("fn visible(x: tensor[32] f32, visible: tensor[1] i32):\n  n = visible[0]\n  t = load(x[0:n])\n");
    let a = memory::derive(&p, "visible", &HashMap::new(), &Bindings::new(), 1000).unwrap();
    assert!(!a.is_exact());
    assert_eq!(bytes(&a, "visible"), (4, 0));
    assert!(a.unavailable.iter().any(|e| e.contains("runtime index")));
}

#[test]
fn budget_boundary_cannot_skip_later_accesses_silently() {
    let p = source("fn pair(x: tensor[8] f32, y: tensor[8] f32):\n  a = load(x)\n  b = load(y)\n");
    let a = memory::derive(&p, "pair", &HashMap::new(), &Bindings::new(), 2).unwrap();
    assert!(!a.is_exact());
    assert_eq!(bytes(&a, "x"), (32, 0));
    assert!(a.unavailable[0].contains("budget"));
}

#[test]
fn partial_packed_groups_need_an_explicit_representation_contract() {
    let result = memory::derive(
        &standard(),
        "projection",
        &HashMap::from([("N".into(), 128), ("K".into(), 65)]),
        &Bindings::new(),
        1000,
    );
    assert!(result.unwrap_err().contains("complete groups"));
}

#[test]
fn affine_contiguous_union_cost_is_independent_of_iteration_count() {
    let p=source("fn copy[N](x: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    store(t, out[row:row+1])\n");
    let mut steps = None;
    for n in [17, 65_536, 1_000_000_000] {
        let account = memory::derive(
            &p,
            "copy",
            &HashMap::from([("N".into(), n)]),
            &Bindings::new(),
            100,
        )
        .unwrap();
        assert!(account.is_exact(), "{:?}", account.unavailable);
        assert_eq!(bytes(&account, "x"), (n as u64 * 4, 0));
        assert_eq!(bytes(&account, "out"), (0, n as u64 * 4));
        if let Some(steps) = steps {
            assert_eq!(account.analysis_steps, steps)
        } else {
            steps = Some(account.analysis_steps)
        }
    }
}

#[test]
fn affine_union_matches_independent_bitmap_including_reverse_overlap_and_holes() {
    for step in [0, 1, 2, 3, 5] {
        for width in [1, 2, 4] {
            for reversed in [false, true] {
                let n = 8;
                let capacity = n * step + width;
                let start = if reversed {
                    format!("{capacity} - row * {step} - {width}")
                } else {
                    format!("row * {step}")
                };
                let end = if reversed {
                    format!("{capacity} - row * {step}")
                } else {
                    format!("row * {step} + {width}")
                };
                let p=source(&format!("fn access(x: tensor[{capacity}] f32, out: tensor[{n}, {width}] f32):\n  for row in parallel:\n    t = load(x[{start}:{end}])\n    store(t,out[row])\n"));
                let account =
                    memory::derive(&p, "access", &HashMap::new(), &Bindings::new(), 10_000)
                        .unwrap();
                assert!(account.is_exact(), "{:?}", account.unavailable);
                let mut bitmap = vec![false; capacity];
                for row in 0..n {
                    let begin = if reversed {
                        capacity - row * step - width
                    } else {
                        row * step
                    };
                    for bit in &mut bitmap[begin..begin + width] {
                        *bit = true
                    }
                }
                assert_eq!(
                    bytes(&account, "x").0,
                    bitmap.iter().filter(|v| **v).count() as u64 * 4,
                    "step {step} width {width} reverse {reversed}"
                );
                assert_eq!(bytes(&account, "out").1, (n * width * 4) as u64);
            }
        }
    }
}

#[test]
fn nonlinear_and_conditional_domains_are_not_extrapolated_from_endpoints() {
    let p=source("fn modulo(x: tensor[4] f32, out: tensor[9] f32):\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = x[row % 4]\n    store(y,out[row:row+1])\n\nfn conditional(x: tensor[8] f32, out: tensor[8] f32):\n  for row in parallel:\n    if row < 4:\n      t = load(x[row:row+1])\n      store(t,out[row:row+1])\n");
    let a = memory::derive(&p, "modulo", &HashMap::new(), &Bindings::new(), 10_000).unwrap();
    assert!(a.is_exact(), "{:?}", a.unavailable);
    assert_eq!(bytes(&a, "x"), (16, 0));
    let b = memory::derive(&p, "conditional", &HashMap::new(), &Bindings::new(), 10_000).unwrap();
    assert!(b.is_exact(), "{:?}", b.unavailable);
    assert_eq!(bytes(&b, "x"), (16, 0));
    assert_eq!(bytes(&b, "out"), (0, 16));
}
