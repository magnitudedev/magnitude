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
    /// Logical half-open range value `(start, end)`.
    Range(i64, i64),
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
    use crate::program::{compile, SourceFile};
    use crate::types::DType;

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

    fn values(t: &TensorData) -> Vec<f32> {
        match t {
            TensorData::Dense { data, .. } => data.iter().map(|v| *v as f32).collect(),
            TensorData::Packed { shape, .. } => (0..shape.iter().product())
                .map(|i| t.get(i) as f32)
                .collect(),
        }
    }

    #[test]
    fn logical_matrix_intrinsics_have_owned_reference_results() {
        let p = program(&["\
fn product(a: &tensor[2, 3] f32, b: &tensor[3, 2] f32, y: &mut tensor[2, 2] f32) for metal requires metal.matrix:
    let value = metal.matrix.matmul(a, b, accumulation=f32)
    for i in 0..2:
        for j in 0..2:
            y[i, j] = value[i, j]

fn product_add(a: &tensor[2, 3] f32, b: &tensor[3, 2] f32, c: &tensor[2, 2] f32, y: &mut tensor[2, 2] f32) for metal requires metal.matrix:
    let value = metal.matrix.matmul_add(a, b, c)
    for i in 0..2:
        for j in 0..2:
            y[i, j] = value[i, j]
"]);
        let matrix = |shape, values| TensorData::dense(DType::F32, shape, values);
        let a = matrix(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = matrix(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        for (entry, add, expected) in [
            ("product", None, vec![58.0, 64.0, 139.0, 154.0]),
            (
                "product_add",
                Some(vec![1.0, 2.0, 3.0, 4.0]),
                vec![59.0, 66.0, 142.0, 158.0],
            ),
        ] {
            let mut interpreter = Interpreter::new(&p);
            interpreter.target = Some("metal".into());
            let a_id = interpreter.add_tensor(a.clone());
            let b_id = interpreter.add_tensor(b.clone());
            let mut args = vec![Arg::Tensor(a_id), Arg::Tensor(b_id)];
            if let Some(values) = add {
                let c_id = interpreter.add_tensor(matrix(vec![2, 2], values));
                args.push(Arg::Tensor(c_id));
            }
            let output = interpreter.add_tensor(matrix(vec![2, 2], vec![0.0; 4]));
            args.push(Arg::Tensor(output));
            interpreter.run(entry, &args, &Workload::default()).unwrap();
            assert_eq!(values(&interpreter.tensors[output]), expected);
        }
    }
}
