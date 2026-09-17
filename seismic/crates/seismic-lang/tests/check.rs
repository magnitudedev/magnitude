//! Checker behaviour: programs that must pass, and programs that must fail with the named rule.

use seismic_lang::program::{compile, SourceFile};
use seismic_lang::Scope;
use std::path::PathBuf;

fn file(name: &str, text: &str) -> SourceFile {
    let scope = if name.ends_with(".portable") { Scope::Portable } else { Scope::Backend(name.rsplit('.').next().unwrap().to_string()) };
    SourceFile { path: PathBuf::from(name), text: text.to_string(), scope }
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
    let files = [file("a.seismic.portable", body), file("b.seismic.portable", body)];
    assert_error(&files, "already declared");
}

#[test]
fn packed_tiles_cannot_be_transposed() {
    let files = [file("k.seismic.portable", "fn f[N, K](w: tensor[N, K] q4g64, out: tensor[1, N] bf16):\n  for r in parallel:\n    t = load(w).T\n    y = tile[1, N] f32\n    for i, j in owned(y): y[i, j] = 0.0\n    store(y, out)\n")];
    assert_error(&files, "cannot be transposed");
}
