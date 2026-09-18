//! Checker behaviour: programs that must pass, and programs that must fail with the named rule.

use seismic_lang::program::{compile, SourceFile};
use seismic_lang::Scope;
use std::path::PathBuf;

fn file(name: &str, text: &str) -> SourceFile {
    let scope = if name.ends_with(".portable") { Scope::Portable } else { Scope::Backend(name.rsplit('.').next().unwrap().to_string()) };
    SourceFile { path: PathBuf::from(name), text: text.to_string(), scope ,
    }
}

fn errors(files: &[SourceFile], backends: &[&str]) -> Vec<String> {
    let backends: Vec<String> = backends.iter().map(|s| s.to_string()).collect();
    match compile(files, &backends) {
        Ok(_) => Vec::new(),
        Err(es) => es.into_iter().map(|e| e.diagnostic.message).collect(),
    }
}

fn assert_error(files: &[SourceFile], needle: &str) {
    let es = errors(files, &["metal"]);
    assert!(es.iter().any(|m| m.contains(needle)), "expected an error containing {needle:?}, got {es:#?}");
}

const MATMUL: &str = "construct matmul[M, N, K](A: tile[M, K] T, B: tile[N, K] U, acc: tile[M, N] f32):
  for i, j in owned(acc):
    acc[i, j] += reduce(A[i, :] * B[j, :], 0, sum)
";

#[test]
fn passing_program() {
    let files = [
        file("m.seismic.portable", MATMUL),
        file("m.seismic.metal", "lower matmul: portable\n"),
        file("k.seismic.portable", "fn scale[R, W](x: tensor[R, W] f32, out: tensor[R, W] f32):\n  for r in parallel:\n    t = load(x[r])\n    y = tile[W] f32\n    for i in owned(y): y[i] = t[i] * 2.0\n    store(y, out[r])\n"),
    ];
    assert_eq!(errors(&files, &["metal"]), Vec::<String>::new());
}

#[test]
fn index_out_of_bounds_is_rejected() {
    let files = [file("k.seismic.portable", "fn f[R, W](x: tensor[R, W] f32, out: tensor[R, W] f32):\n  for r in parallel:\n    t = load(x[r + 1])\n    store(t, out[r])\n")];
    assert_error(&files, "index may exceed extent `R`");
}

#[test]
fn read_before_assignment_is_rejected() {
    let files = [file("k.seismic.portable", "fn f[W](out: tensor[W] f32):\n  for r in parallel:\n    y = tile[W] f32\n    store(y, out)\n")];
    assert_error(&files, "read before it is assigned");
}

#[test]
fn partial_elementwise_write_is_rejected() {
    let files = [file("k.seismic.portable", "fn f[W](out: tensor[W] f32):\n  for r in parallel:\n    y = tile[W] f32\n    y[0] = 1.0\n    store(y, out)\n")];
    assert_error(&files, "assign every element through");
}

#[test]
fn dependent_work_items_are_rejected() {
    let files = [file("k.seismic.portable", "fn f[R, W](x: tensor[R, W] f32, out: tensor[W] f32):\n  for r in parallel:\n    t = load(x[r])\n    store(t, out)\n")];
    assert_error(&files, "cannot prove work items are independent");
}

#[test]
fn implicit_narrowing_is_rejected() {
    let files = [file("k.seismic.portable", "fn f[W](x: tensor[W] f32, out: tensor[W] f32):\n  for r in parallel:\n    t = load(x)\n    n = tile[W] i32\n    for i in owned(n): n[i] = t[i]\n    store(n, out)\n")];
    assert_error(&files, "cast explicitly");
}

#[test]
fn construct_row_is_required() {
    let files = [file("m.seismic.portable", MATMUL)];
    assert_error(&files, "has no lowering for backend `metal`");
}

#[test]
fn coverage_is_required() {
    let files = [
        file("m.seismic.portable", MATMUL),
        file("m.seismic.metal", "lower matmul[M, N, K](A: tile[M, K] T, B: tile[N, K] U, acc: tile[M, N] f32):\n  for i in range(M / 8):\n    for j in range(N / 8):\n      c = simdgroup_matrix(f32)\n      simdgroup_load(c, acc, i * 8, j * 8)\n      simdgroup_store(c, acc, i * 8, j * 8)\n"),
    ];
    assert_error(&files, "no lowering covers the whole domain");
}

#[test]
fn kernels_may_not_have_lowerings() {
    let files = [
        file("k.seismic.portable", "fn f[W](out: tensor[W] f32):\n  for r in parallel:\n    y = tile[W] f32\n    for i in owned(y): y[i] = 0.0\n    store(y, out)\n"),
        file("k.seismic.metal", "lower f: portable\n"),
    ];
    assert_error(&files, "only constructs have lowerings");
}

#[test]
fn intrinsics_are_scoped() {
    let files = [file("k.seismic.portable", "fn f[W](out: tensor[W] f32):\n  for r in parallel:\n    x = simd_sum(1.0)\n    y = tile[W] f32\n    for i in owned(y): y[i] = x\n    store(y, out)\n")];
    assert_error(&files, "not available in this file's scope");
}

#[test]
fn duplicate_names_are_rejected() {
    let body = "fn f[W](out: tensor[W] f32):\n  for r in parallel:\n    y = tile[W] f32\n    for i in owned(y): y[i] = 0.0\n    store(y, out)\n";
    let files = [file("a.seismic.portable", body), file("b.seismic.portable", body),
    ];
    assert_error(&files, "already declared");
}

#[test]
fn packed_tiles_cannot_be_transposed() {
    let files = [file("k.seismic.portable", "fn f[N, K](w: tensor[N, K] q4g64, out: tensor[1, N] bf16):\n  for r in parallel:\n    t = load(w).T\n    y = tile[1, N] f32\n    for i, j in owned(y): y[i, j] = 0.0\n    store(y, out)\n")];
    assert_error(&files, "cannot be transposed");
}

#[test]
fn reduction_order_is_a_static_permission_and_argmax_needs_nonempty_domain() {
    let source="fn f(flag: bool, x: tensor[4] f32):\n  t = load(x)\n  s = reduce(t, 0, sum, ordered=true)\n";
    assert!(errors(&[file("ordered.seismic.portable",source)],&[]).is_empty());
    assert_error(&[file("ordered.seismic.portable",&source.replace("ordered=true","ordered=flag"),
        )],"boolean literal",
    );
    assert_error(&[file("ordered.seismic.portable",&source.replace("ordered=true","true"),
        )],"named `ordered`",
    );
    assert_error(&[file("empty.seismic.portable","fn f(x: tensor[4] f32):\n  t = load(x[0:0])\n  at = reduce(t, 0, argmax)\n",
        )],"nonempty axis",
    );
}

#[test]
fn shape_dependent_argmax_domain_is_validated_at_specialization() {
    let p=compile(&[file("domain.seismic.portable","fn f[N, B](x: tensor[N / B] f32):\n  t = load(x)\n  at = reduce(t,0,argmax)\n",
        )],&[],
    ).unwrap();
    assert!(seismic_lang::lower::lower(&p,"f","cpu",&[("N".into(),8),("B".into(),4)].into()).is_ok());
    assert!(seismic_lang::lower::lower(&p,"f","cpu",&[("N".into(),2),("B".into(),4)].into()).unwrap_err().contains("nonempty axis"));
}
#[test]
fn complete_tile_outputs_compose_without_spurious_initialization() {
    let helper="fn fill[N](out: tile[N] f32, flag: bool):\n  for i in owned(out):\n    if flag: out[i] = 1.0\n    else: out[i] = 2.0\n";
    let caller="fn entry(out: tensor[4] f32, flag: bool):\n  tile_out = tile[4] f32\n  fill(tile_out,flag)\n  store(tile_out,out)\n";
    assert!(errors(
        &[file(
            "output.seismic.portable",
            &format!("{helper}\n{caller}")
        )],
        &[]
    )
    .is_empty());
    let conditional="fn entry(out: tensor[4] f32, flag: bool):\n  tile_out = tile[4] f32\n  if flag: fill(tile_out,flag)\n  store(tile_out,out)\n";
    assert_error(
        &[file(
            "output.seismic.portable",
            &format!("{helper}\n{conditional}"),
        )],
        "read before",
    );
    let both="fn entry(out: tensor[4] f32, flag: bool):\n  tile_out = tile[4] f32\n  if flag: fill(tile_out,true)\n  else: fill(tile_out,false)\n  store(tile_out,out)\n";
    assert!(errors(
        &[file(
            "output.seismic.portable",
            &format!("{helper}\n{both}")
        )],
        &[]
    )
    .is_empty());
    for helper in [
        "fn fill[N](out: tile[N] f32, flag: bool):\n  for i in owned(out):\n    if flag: out[i] = 1.0\n",
        "fn fill[N](out: tile[N] f32, flag: bool):\n  if flag:\n    for i in owned(out): out[i] = 1.0\n",
        "fn fill[N](out: tile[N] f32, flag: bool):\n  old = out[0]\n  for i in owned(out): out[i] = old\n",
    ] {
        assert_error(&[file("output.seismic.portable",&format!("{helper}\n{caller}"))],"read before");
    }
}

#[test]
fn output_proof_does_not_initialize_partial_or_aliased_inputs() {
    let helper="fn copy[N](input: tile[N] f32, out: tile[N] f32):\n  for i in owned(out): out[i] = input[i]\n";
    let alias = "fn entry(out: tensor[4] f32):\n  t = tile[4] f32\n  copy(t,t)\n  store(t,out)\n";
    assert_error(
        &[file(
            "alias.seismic.portable",
            &format!("{helper}\n{alias}"),
        )],
        "read before",
    );
    let partial="fn fill[N](out: tile[N] f32):\n  for i in owned(out): out[i] = 1.0\n\nfn entry(out: tensor[4] f32):\n  t = tile[4] f32\n  fill(t[0:2])\n  store(t,out)\n";
    assert_error(&[file("partial.seismic.portable", partial)], "read before");
}

#[test]
fn fragment_store_initializes_only_a_complete_definite_tile() {
    let probe = |extent: usize, write: &str| format!(
        "fn probe(x: tile[8,8] f32, enabled: bool):\n  tmp = tile[{extent},8] f32\n  a = simdgroup_matrix(f32)\n  simdgroup_load(a,x,0,0)\n{write}\n  b = simdgroup_matrix(f32)\n  simdgroup_load(b,tmp,0,0)\n"
    );
    let full = probe(8, "  simdgroup_store(a,tmp,0,0)");
    let es = errors(&[file("write.seismic.metal", &full)], &["metal"]);
    assert!(es.is_empty(), "{es:?}");
    let partial = probe(16, "  simdgroup_store(a,tmp,0,0)");
    assert_error(&[file("partial.seismic.metal", &partial)], "read before");
    let conditional = probe(8, "  if enabled:\n    simdgroup_store(a,tmp,0,0)");
    assert_error(&[file("conditional.seismic.metal", &conditional)], "read before");
    let both = probe(8, "  if enabled:\n    simdgroup_store(a,tmp,0,0)\n  else:\n    simdgroup_store(a,tmp,0,0)");
    let es = errors(&[file("both.seismic.metal", &both)], &["metal"]);
    assert!(es.is_empty(), "{es:?}");
    let invalid = probe(8, "  simdgroup_store(a,tmp,1,0)");
    assert_error(&[file("bounds.seismic.metal", &invalid)], "block may exceed");
}

#[test]
fn definite_assignment_uses_typed_coordinates_and_requires_the_first_write_to_cover() {
    let valid="fn fill[N](out:tensor[N] f32):\n  t = tile[N] f32\n  for i in owned(t):\n    j = i + 0\n    t[j] = 1.0\n  store(t,out)\n";
    assert!(errors(&[file("assignment.seismic.portable",valid)],&[]).is_empty());
    let invalid="fn fill[N](out:tensor[N] f32):\n  t = tile[N] f32\n  for i in owned(t):\n    t[0] = 1.0\n    t[i] = t[i] + 1.0\n  store(t,out)\n";
    assert_error(&[file("assignment.seismic.portable",invalid)],"first owned write must cover");
}

#[test]
fn logical_loads_cannot_be_used_as_source_partition_iterators() {
    assert_error(&[file("logical.seismic.portable", "fn f(x:tensor[8] f32):\n  for t in load(x,over=0):\n    n = extent(t,0)\n")], "logical value, not an iterator");
}

#[test]
fn repeated_logical_windows_share_shape_only_while_their_bounds_are_unchanged() {
    let helper="fn pair[N](a:tile[N] f32,b:tile[N] f32):\n  for i in owned(a): a[i] = a[i] + b[i]\n";
    let text=format!("{helper}fn f(x:tensor[8] f32,bounds:tensor[2] i32):\n  a = load(x[bounds[0]:bounds[1]])\n  b = load(x[bounds[0]:bounds[1]])\n  pair(a,b)\n");
    assert!(errors(&[file("logical.seismic.portable",&text)],&[]).is_empty());
    let changed=text.replace("  b = load", "  next = tile[2] i32\n  for i in owned(next): next[i] = i\n  store(next,bounds)\n  b = load");
    assert!(!errors(&[file("logical.seismic.portable",&changed)],&[]).is_empty(),"changing a bound must not reuse the old window's shape identity");
}

#[test]
fn slice_endpoints_reject_named_non_i32_values() {
    for dtype in ["f32", "bool", "u32"] {
        for (parameters, binding) in [
            (format!("bound:{dtype}"), String::new()),
            (format!("bounds:tensor[1] {dtype}"), "  bound = bounds[0]\n".into()),
        ] {
            for (slice, endpoint) in [
                ("bound:", "start"),
                ("bound:8", "start"),
                (":bound", "end"),
                ("0:bound", "end"),
            ] {
                let source = format!(
                    "fn f(x:tensor[8] f32,{parameters}):\n{binding}  t = load(x[{slice}])\n"
                );
                let es = errors(&[file("endpoints.seismic.portable", &source)], &[]);
                let expected = format!("slice {endpoint} must be i32, found {dtype}");
                assert!(es.iter().any(|e| e == &expected), "{source}\n{es:#?}");
            }
        }
    }
}

#[test]
fn dynamic_i32_slice_endpoints_preserve_clamped_windows() {
    use seismic_lang::interp::{Arg, Interpreter, TensorData};
    use seismic_lang::types::DType;
    use std::collections::HashMap;

    for (parameters, bindings) in [
        ("start:i32,end:i32", ""),
        ("bounds:tensor[2] i32", "  start = bounds[0]\n  end = bounds[1]\n"),
    ] {
        let source = format!(
            "fn f(x:tensor[8] f32,{parameters},out:tensor[6] i32):
{bindings}  window = load(x[start:end])
  tail = load(x[start:])
  head = load(x[:end])
  result = tile[6] i32
  for i in owned(result): result[i] = 0
  result[0] = extent(window,0)
  result[1] = i32(reduce(window,0,sum))
  result[2] = extent(tail,0)
  result[3] = i32(reduce(tail,0,sum))
  result[4] = extent(head,0)
  result[5] = i32(reduce(head,0,sum))
  store(result,out)
"
        );
        let program = compile(&[file("endpoints.seismic.portable", &source)], &[])
            .unwrap_or_else(|es| panic!("{source}\n{es:#?}"));
        let mut interpreter = Interpreter::new(&program);
        let x = interpreter.add_tensor(TensorData::dense(
            DType::F32, vec![8], (1..=8).map(f64::from).collect(),
        ));
        let out = interpreter.add_tensor(TensorData::dense(DType::I32, vec![6], vec![0.0; 6]));
        for (start, end, expected) in [
            (2, 6, [4, 18, 6, 33, 6, 21]),
            (-4, 3, [3, 6, 8, 36, 3, 6]),
            (4, 99, [4, 26, 4, 26, 8, 36]),
            (99, 3, [0, 0, 0, 0, 3, 6]),
            (4, -9, [0, 0, 4, 26, 0, 0]),
            (i32::MIN, i32::MAX, [8, 36, 8, 36, 8, 36]),
            (i32::MAX, i32::MIN, [0; 6]),
        ] {
            let mut args = vec![Arg::Tensor(x)];
            if bindings.is_empty() {
                args.extend([Arg::Scalar(f64::from(start)), Arg::Scalar(f64::from(end))]);
            } else {
                let bounds = interpreter.add_tensor(TensorData::dense(
                    DType::I32, vec![2], vec![f64::from(start), f64::from(end)],
                ));
                args.push(Arg::Tensor(bounds));
            }
            args.push(Arg::Tensor(out));
            interpreter.run("f", &args, &HashMap::new()).unwrap();
            let actual: Vec<_> = (0..6).map(|i| interpreter.tensors[out].get(i)).collect();
            assert_eq!(actual, expected.map(f64::from), "{parameters}: [{start}:{end}]");
        }
    }
}

#[test]
fn symbolic_slice_endpoints_keep_static_bounds_obligations() {
    for (slice, diagnostic) in [
        ("-1:4", "slice start may be negative"),
        ("0:N+1", "slice end may exceed extent `N`"),
    ] {
        let source = format!("fn f[N](x:tensor[N] f32):\n  t = load(x[{slice}])\n");
        assert_error(&[file("endpoints.seismic.portable", &source)], diagnostic);
    }
    let source = "fn f[N](x:tensor[N] f32):\n  full = load(x[:])\n  empty = load(x[N:N])\n";
    let es = errors(&[file("endpoints.seismic.portable", source)], &[]);
    assert!(es.is_empty(), "{es:#?}");
}

#[test]
fn affine_tail_guards_bound_the_access_in_their_own_branch() {
    let source = "fn copy[M](x:tensor[M] f32,out:tensor[M] f32):\n  a=load(x)\n  y=tile[M] f32\n  for p in owned(y): y[p]=0.0\n  for block in range((M+7)/8):\n    for lane in range(8):\n      if block*8+lane<M: y[block*8+lane]=a[block*8+lane]\n  store(y,out)\n";
    assert!(errors(&[file("guard.seismic.portable", source)], &[]).is_empty());
    for wrong in [source.replace("block*8+lane<M", "block*8+lane<=M"), source.replace("block*8+lane<M", "lane<M"), source.replace("if block*8+lane<M: ", "")] {
        let errors = errors(&[file("guard.seismic.portable", &wrong)], &[]);
        assert!(errors.iter().any(|error| error.contains("index may exceed extent")), "{errors:?}");
    }
    let after = source.replace("  store(y,out)", "      y[block*8+lane]=a[block*8+lane]\n  store(y,out)");
    assert!(errors(&[file("guard.seismic.portable", &after)], &[]).iter().any(|error| error.contains("index may exceed extent")), "a completed guard cannot constrain the next access");
}
