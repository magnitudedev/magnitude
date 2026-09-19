//! Reference interpreter over `sir`: the semantic oracle. Regions execute under a
//! caller-supplied partitioning so partition independence can be exercised.
//!
//! Scalars carry their dtype and every operation rounds once at its dtype, so generic
//! bodies evaluate at the element types actually bound. Shaped values are zero-based
//! strided selections; the semantic coordinate of a structural axis' first position is its
//! slice's `lo`, taken from the static axis type, so callees (whose axes are semantic) see
//! plain zero-based extents. Parallel visits run sequentially in lexicographic order.
use super::family::Workload;
use super::sir::{DefId, Program};
use super::types::{RegionId, SliceId};
pub use tensor::{round_to, Rng, TensorData};

mod eval;
mod exec;
mod native;
mod scalar;
mod tensor;
mod value;

#[derive(Clone, Debug)]
pub enum Arg {
    /// Index into the interpreter's tensor table.
    Tensor(usize),
    Scalar(f64),
}

/// Chooses the slice width for each static binder. `extent` is the runtime extent of the
/// partitioned domain for this visit. Must return a value in `1..=max(extent, 1)`.
pub trait Partitioner {
    fn width(&self, definition: &str, region: RegionId, slice: SliceId, extent: i64) -> i64;
}

/// Every binder gets width `w` (clamped to the extent).
pub struct Uniform(pub i64);

impl Partitioner for Uniform {
    fn width(&self, _: &str, _: RegionId, _: SliceId, extent: i64) -> i64 {
        self.0.clamp(1, extent.max(1))
    }
}

/// Forces the body of a call or entry: given the family name and every applicable
/// definition with a body (ascending `DefId`), `Some(id)` selects it, `None` defers.
pub type Choice<'a> = Box<dyn Fn(&str, &[DefId]) -> Option<DefId> + 'a>;

pub struct Interpreter<'a> {
    pub program: &'a Program,
    pub tensors: Vec<TensorData>,
    pub partitioner: Box<dyn Partitioner + 'a>,
    /// Target whose target-specific functions and lowerings may be interpreted; `None`
    /// considers portable functions only.
    pub target: Option<String>,
    /// Overrides the deterministic first-applicable definition choice.
    pub choice: Option<Choice<'a>>,
}

impl<'a> Interpreter<'a> {
    pub fn new(program: &'a Program) -> Interpreter<'a> {
        Interpreter {
            program,
            tensors: Vec::new(),
            partitioner: Box::new(Uniform(1)),
            target: None,
            choice: None,
        }
    }

    pub fn with_choice(
        mut self,
        choice: impl Fn(&str, &[DefId]) -> Option<DefId> + 'a,
    ) -> Interpreter<'a> {
        self.choice = Some(Box::new(choice));
        self
    }

    pub fn add_tensor(&mut self, t: TensorData) -> usize {
        self.tensors.push(t);
        self.tensors.len() - 1
    }

    /// Run linked function `name` for `workload`.
    pub fn run(&mut self, name: &str, args: &[Arg], workload: &Workload) -> Result<(), String> {
        self.run_entry(name, args, workload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::bf16_round;
    use crate::program::{compile, SourceFile};
    use crate::types::{DType, Elem};

    const MATMUL: &str = "\
fn matmul[M, N, K](a: tile[M, K] T, b: tile[N, K] U, inout acc: tile[M, N] f32):
    for i, j in owned(acc):
        let mut s = acc[i, j]
        for k in axis(a, 1):
            s = fma(f32(a[i, k]), f32(b[j, k]), s)
        acc[i, j] = s
";

    const LINEAR: &str = "\
fn linear[M, N, K](x: tensor[M, K] T, weight: tensor[N, K] U, out y: tensor[M, N] V):
    parallel [rows, cols] in (0..M, 0..N):
        let mut acc = zeros_like(y[rows, cols], dtype=f32)
        ordered [k] in 0..K:
            matmul(load(x[rows, k]), load(weight[cols, k]), into=acc)
        publish acc to y[rows, cols]
";

    const RMS_NORM: &str = "\
fn rms_norm[R, W](x: tensor[R, W] T, weight: tensor[W] U, out y: tensor[R, W] V, eps: f32):
    parallel [rows] in 0..R:
        let w = f32(weight)
        for row in rows:
            let t = f32(x[row])
            let ss = reduce(t * t, 0, sum)
            publish t * rsqrt(ss / f32(W) + eps) * w to y[row]
";

    const SUM_SQUARES: &str = "\
fn sum_squares[N](x: tensor[N] f32, out y: tensor[1] f32):
    stage prepare:
        let partials = parallel [p] in 0..N:
            let values = f32(x[p])
            yield reduce(values * values, 0, sum)
        yield partials
    stage finish(partials):
        let mut total = f32(0.0)
        ordered [p] in partials:
            total = total + partials[p]
        publish total to y[0]
";

    const MERGE_SUM: &str = "\
admit fn merge_sum[K](x: tensor[K] f32, out y: tensor[1] f32):
    let total = parallel [part] in 0..K:
        yield reduce(f32(x[part]), 0, sum)
    merge (left, right) identity f32(0.0):
        yield left + right
    publish total to y[0]
";

    const SCAN: &str = "\
admit fn scan[N](x: tensor[N] f32, out before: tensor[N] f32, out after: tensor[N] f32, out last: tensor[2] f32):
    let partials = parallel [p] in 0..N:
        yield reduce(f32(x[p]), 0, sum)
    let mut (m, s) = (f32(-inf), f32(0.0))
    let checkpoints = ordered [p] in partials:
        let previous = s
        (m, s) = (max(m, partials[p]), s + partials[p])
        yield previous, s
    parallel [p] in checkpoints:
        let (b, a) = checkpoints[p]
        publish zeros_like(before[p], dtype=f32) + b to before[p]
        publish zeros_like(after[p], dtype=f32) + a to after[p]
    publish m to last[0]
    publish s to last[1]
";

    fn program(sources: &[&str]) -> Program {
        let files: Vec<SourceFile> = sources
            .iter()
            .enumerate()
            .map(|(i, text)| SourceFile {
                path: format!("test{i}.seismic"),
                text: text.to_string(),
            })
            .collect();
        compile(&files).unwrap_or_else(|d| {
            panic!(
                "{}",
                d.iter().map(|d| d.render()).collect::<Vec<_>>().join("\n")
            )
        })
    }

    fn workload(shapes: &[(&str, i64)], elems: &[(&str, DType)]) -> Workload {
        Workload {
            shapes: shapes.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
            elems: elems
                .iter()
                .map(|(n, d)| (n.to_string(), Elem::Dtype(*d)))
                .collect(),
            ..Workload::default()
        }
    }

    /// Run `name` over `tensors` under uniform width `width`; returns the tensors afterwards.
    fn run(
        program: &Program,
        name: &str,
        tensors: &[TensorData],
        scalars: &[f64],
        workload: &Workload,
        width: i64,
    ) -> Vec<Vec<u64>> {
        let mut interp = Interpreter::new(program);
        interp.partitioner = Box::new(Uniform(width));
        let mut args: Vec<Arg> = tensors
            .iter()
            .map(|t| Arg::Tensor(interp.add_tensor(t.clone())))
            .collect();
        args.extend(scalars.iter().map(|s| Arg::Scalar(*s)));
        interp
            .run(name, &args, workload)
            .unwrap_or_else(|e| panic!("{name} at width {width}: {e}"));
        interp
            .tensors
            .iter()
            .map(|t| match t {
                TensorData::Dense { data, .. } => data.iter().map(|v| v.to_bits()).collect(),
                TensorData::Packed { .. } => Vec::new(),
            })
            .collect()
    }

    fn values(t: &TensorData) -> Vec<f32> {
        match t {
            TensorData::Dense { data, .. } => data.iter().map(|v| *v as f32).collect(),
            TensorData::Packed { shape, .. } => (0..shape.iter().product())
                .map(|i| t.get(i) as f32)
                .collect(),
        }
    }

    fn bits(expected: &[f32]) -> Vec<u64> {
        expected.iter().map(|v| f64::from(*v).to_bits()).collect()
    }

    /// Piece sums of `x` under uniform width `w`, each an ascending f32 accumulation.
    fn piece_sums(x: &[f32], w: usize, square: bool) -> Vec<f32> {
        x.chunks(w)
            .map(|c| {
                c.iter()
                    .fold(0f32, |acc, v| acc + if square { v * v } else { *v })
            })
            .collect()
    }

    #[test]
    fn linear_is_bit_identical_across_widths() {
        let p = program(&[MATMUL, LINEAR]);
        let (m, n, k) = (5usize, 9usize, 32usize);
        let mut rng = Rng(0x5eed);
        let x = TensorData::random_dense(&mut rng, DType::BF16, vec![m, k]);
        let dense = TensorData::random_dense(&mut rng, DType::F32, vec![n, k]);
        let packed = TensorData::random_packed(
            &mut rng,
            crate::repr::lookup("q4g32").expect("registered representation"),
            vec![n, k],
        );
        for weight in [dense, packed] {
            let y = TensorData::dense(DType::BF16, vec![m, n], vec![0.0; m * n]);
            let mut w = workload(
                &[("M", m as i64), ("N", n as i64), ("K", k as i64)],
                &[("T", DType::BF16), ("V", DType::BF16)],
            );
            w.elems.insert(
                "U".into(),
                match &weight {
                    TensorData::Dense { dtype, .. } => Elem::Dtype(*dtype),
                    TensorData::Packed { repr, .. } => Elem::Repr(repr.name.into()),
                },
            );
            let tensors = [x.clone(), weight.clone(), y];
            let narrow = run(&p, "linear", &tensors, &[], &w, 1);
            let wide = run(&p, "linear", &tensors, &[], &w, 7);
            assert_eq!(narrow, wide);
            let (xs, ws) = (values(&x), values(&weight));
            let expected: Vec<f32> = (0..m * n)
                .map(|o| {
                    bf16_round((0..k).fold(0f32, |acc, c| {
                        xs[o / n * k + c].mul_add(ws[o % n * k + c], acc)
                    }))
                })
                .collect();
            assert_eq!(wide[2], bits(&expected));
        }
    }

    #[test]
    fn rms_norm_matches_reference() {
        let p = program(&[RMS_NORM]);
        let (r, w) = (6usize, 11usize);
        let mut rng = Rng(0xabcdef);
        let x = TensorData::random_dense(&mut rng, DType::BF16, vec![r, w]);
        let weight = TensorData::random_dense(&mut rng, DType::F32, vec![w]);
        let y = TensorData::dense(DType::BF16, vec![r, w], vec![0.0; r * w]);
        let load = workload(
            &[("R", r as i64), ("W", w as i64)],
            &[("T", DType::BF16), ("U", DType::F32), ("V", DType::BF16)],
        );
        let eps = 1e-5f32;
        let tensors = [x.clone(), weight.clone(), y];
        let narrow = run(&p, "rms_norm", &tensors, &[f64::from(eps)], &load, 1);
        let wide = run(&p, "rms_norm", &tensors, &[f64::from(eps)], &load, 7);
        assert_eq!(narrow, wide);
        let (xs, ws) = (values(&x), values(&weight));
        let mut expected = Vec::new();
        for row in xs.chunks(w) {
            let ss = row.iter().fold(0f32, |acc, v| acc + v * v);
            let scale = (1.0 / f64::from(ss / w as f32 + eps).sqrt()) as f32;
            expected.extend(row.iter().zip(&ws).map(|(t, g)| bf16_round(t * scale * g)));
        }
        assert_eq!(wide[2], bits(&expected));
    }

    #[test]
    fn ordered_traversal_reuses_the_parallel_partition() {
        let p = program(&[SUM_SQUARES]);
        let n = 20usize;
        let x = TensorData::random_dense(&mut Rng(77), DType::F32, vec![n]);
        let y = TensorData::dense(DType::F32, vec![1], vec![0.0]);
        let load = workload(&[("N", n as i64)], &[]);
        for width in [1usize, 7, 20] {
            let out = run(
                &p,
                "sum_squares",
                &[x.clone(), y.clone()],
                &[],
                &load,
                width as i64,
            );
            let total = piece_sums(&values(&x), width, true)
                .into_iter()
                .fold(0f32, |acc, v| acc + v);
            assert_eq!(out[1], bits(&[total]), "width {width}");
        }
    }

    #[test]
    fn merge_combines_a_near_equal_partition_pairwise() {
        let p = program(&[MERGE_SUM]);
        let k = 10usize;
        let x = TensorData::random_dense(&mut Rng(4242), DType::F32, vec![k]);
        let y = TensorData::dense(DType::F32, vec![1], vec![0.0]);
        let xs = values(&x);
        let sum = |r: std::ops::Range<usize>| xs[r].iter().fold(0f32, |acc, v| acc + v);
        // Width 3 over 10 gives 4 parts of 3, 3, 2, 2; width 4 gives 3 parts of 4, 3, 3 with
        // the odd part forwarded; width 10 gives the single partial.
        let cases = [
            (3, (sum(0..3) + sum(3..6)) + (sum(6..8) + sum(8..10))),
            (4, (sum(0..4) + sum(4..7)) + sum(7..10)),
            (10, sum(0..10)),
        ];
        for (width, expected) in cases {
            let out = run(
                &p,
                "merge_sum",
                &[x.clone(), y.clone()],
                &[],
                &workload(&[("K", k as i64)], &[]),
                width,
            );
            assert_eq!(out[1], bits(&[expected]), "width {width}");
        }
    }

    #[test]
    fn ordered_scan_keeps_tuple_state_and_checkpoints() {
        let p = program(&[SCAN]);
        let n = 13usize;
        let x = TensorData::random_dense(&mut Rng(99), DType::F32, vec![n]);
        let zeros = |len: usize| TensorData::dense(DType::F32, vec![len], vec![0.0; len]);
        for width in [1usize, 5] {
            let out = run(
                &p,
                "scan",
                &[x.clone(), zeros(n), zeros(n), zeros(2)],
                &[],
                &workload(&[("N", n as i64)], &[]),
                width as i64,
            );
            let (mut m, mut s) = (f32::NEG_INFINITY, 0f32);
            let (mut before, mut after) = (Vec::new(), Vec::new());
            for (piece, partial) in piece_sums(&values(&x), width, false)
                .into_iter()
                .enumerate()
            {
                let members = width.min(n - piece * width);
                before.extend(std::iter::repeat_n(s, members));
                (m, s) = (m.max(partial), s + partial);
                after.extend(std::iter::repeat_n(s, members));
            }
            assert_eq!(out[1], bits(&before), "width {width}");
            assert_eq!(out[2], bits(&after), "width {width}");
            assert_eq!(out[3], bits(&[m, s]), "width {width}");
        }
    }

    #[test]
    fn target_lowering_is_an_optional_interpreter_candidate() {
        let p = program(&["\
fn choose(out y: tensor[1] f32):
    publish f32(1.0) to y[0]

lower choose(out y: tensor[1] f32) for cpu:
    publish f32(2.0) to y[0]
"]);
        let lowering = p.family("choose").unwrap().lowerings[0];
        let output = || TensorData::dense(DType::F32, vec![1], vec![0.0]);

        let mut ordinary = Interpreter::new(&p);
        ordinary.target = Some("cpu".into());
        let y = ordinary.add_tensor(output());
        ordinary
            .run("choose", &[Arg::Tensor(y)], &Workload::default())
            .unwrap();
        assert_eq!(ordinary.tensors[y].get(0), 1.0);

        let mut forced = Interpreter::new(&p).with_choice(move |name, candidates| {
            assert_eq!(name, "choose");
            assert_eq!(candidates.len(), 2);
            Some(lowering)
        });
        forced.target = Some("cpu".into());
        let y = forced.add_tensor(output());
        forced
            .run("choose", &[Arg::Tensor(y)], &Workload::default())
            .unwrap();
        assert_eq!(forced.tensors[y].get(0), 2.0);
    }
}
