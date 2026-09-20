//! Reference interpreter over the checked semantic program: the semantic
//! oracle. Every function family is evaluated through its reference body and
//! the intrinsic registry.
//!
//! Scalars carry their dtype and every operation rounds once at its dtype, so
//! generic bodies evaluate at the element types actually bound. Shaped values
//! are zero-based strided selections. Independent (`parallel for`) visits run
//! sequentially in ascending coordinate order — the deterministic reference
//! order. Nothing is ever clamped: out-of-bounds selections are errors.
use super::sir::Program;
use super::types::Elem;
pub use tensor::{round_to, Rng, TensorData};
pub use value::Value;

mod eval;
mod exec;
mod scalar;
mod tensor;
pub mod value;

use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub enum Arg {
    /// Index into the interpreter's tensor table.
    Tensor(usize),
    Scalar(f64),
    /// Logical half-open range value `(start, end)`.
    Range(i64, i64),
}

/// Concrete invocation bindings for shape and element parameters.
#[derive(Clone, Debug, Default)]
pub struct Bindings {
    pub shapes: BTreeMap<String, i64>,
    pub elems: BTreeMap<String, Elem>,
}

pub struct Interpreter<'a> {
    pub program: &'a Program,
    pub tensors: Vec<TensorData>,
    /// Target whose backend-specific helper bodies may be interpreted; `None`
    /// considers portable functions only.
    pub target: Option<String>,
}

impl<'a> Interpreter<'a> {
    pub fn new(program: &'a Program) -> Interpreter<'a> {
        Interpreter {
            program,
            tensors: Vec::new(),
            target: None,
        }
    }

    pub fn add_tensor(&mut self, t: TensorData) -> usize {
        self.tensors.push(t);
        self.tensors.len() - 1
    }

    /// Run linked function `name` for the given argument bindings. Tensor
    /// arguments are updated in place; the returned value is the function's
    /// result.
    pub fn run(
        &mut self,
        name: &str,
        args: &[Arg],
        bindings: &Bindings,
    ) -> Result<value::Value, String> {
        self.run_entry(name, args, bindings)
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
    fn atomic_updates_combine_in_visit_order() {
        let p = program(&["\
fn histogram[N, E](routes: &tensor[N] i32, counts: &mut tensor[E] i32):
    parallel for i in 0..N:
        atomic(add, counts[routes[i]], 1)

fn best[N, E](values: &tensor[N] f32, routes: &tensor[N] i32, top: &mut tensor[E] f32):
    parallel for i in 0..N:
        atomic(max, top[routes[i]], f32(values[i]))

fn least[N, E](values: &tensor[N] f32, routes: &tensor[N] i32, low: &mut tensor[E] f32):
    parallel for i in 0..N:
        atomic(min, low[routes[i]], f32(values[i]))
"]);
        let bindings = Bindings {
            shapes: [("N".to_string(), 6), ("E".to_string(), 3)]
                .into_iter()
                .collect(),
            ..Bindings::default()
        };
        let routes = TensorData::dense(DType::I32, vec![6], vec![0.0, 2.0, 2.0, 1.0, 2.0, 0.0]);
        let nan = f64::NAN;
        let samples = TensorData::dense(DType::F32, vec![6], vec![1.0, 5.0, nan, -3.0, 7.0, 2.0]);

        let mut interpreter = Interpreter::new(&p);
        let r = interpreter.add_tensor(routes.clone());
        let counts = interpreter.add_tensor(TensorData::dense(DType::I32, vec![3], vec![0.0; 3]));
        interpreter
            .run(
                "histogram",
                &[Arg::Tensor(r), Arg::Tensor(counts)],
                &bindings,
            )
            .unwrap();
        assert_eq!(values(&interpreter.tensors[counts]), vec![2.0, 1.0, 3.0]);

        // `max`/`min` ignore a NaN operand, as the reference reductions do.
        let mut interpreter = Interpreter::new(&p);
        let v = interpreter.add_tensor(samples.clone());
        let r = interpreter.add_tensor(routes.clone());
        let top = interpreter.add_tensor(TensorData::dense(
            DType::F32,
            vec![3],
            vec![f64::NEG_INFINITY; 3],
        ));
        interpreter
            .run(
                "best",
                &[Arg::Tensor(v), Arg::Tensor(r), Arg::Tensor(top)],
                &bindings,
            )
            .unwrap();
        assert_eq!(values(&interpreter.tensors[top]), vec![2.0, -3.0, 7.0]);

        let mut interpreter = Interpreter::new(&p);
        let v = interpreter.add_tensor(samples);
        let r = interpreter.add_tensor(routes);
        let low = interpreter.add_tensor(TensorData::dense(
            DType::F32,
            vec![3],
            vec![f64::INFINITY; 3],
        ));
        interpreter
            .run(
                "least",
                &[Arg::Tensor(v), Arg::Tensor(r), Arg::Tensor(low)],
                &bindings,
            )
            .unwrap();
        assert_eq!(values(&interpreter.tensors[low]), vec![1.0, -3.0, 5.0]);
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
            interpreter.run(entry, &args, &Bindings::default()).unwrap();
            assert_eq!(values(&interpreter.tensors[output]), expected);
        }
    }

    #[test]
    fn reference_bodies_evaluate_with_registry_semantics() {
        let p = program(&["\
fn add[N](x: &tensor[N] f32, y: &tensor[N] f32) -> tensor[N] f32:
    let a = load(x)
    let b = load(y)
    return a + b

fn sum[N](x: &tensor[N] f32) -> f32:
    let v = load(x)
    return reduce(v, 0, sum)
"]);
        let mut interpreter = Interpreter::new(&p);
        let x = interpreter.add_tensor(TensorData::dense(
            DType::F32,
            vec![4],
            vec![1.0, 2.0, 3.0, 4.0],
        ));
        let y = interpreter.add_tensor(TensorData::dense(
            DType::F32,
            vec![4],
            vec![10.0, 20.0, 30.0, 40.0],
        ));
        let bindings = Bindings {
            shapes: [("N".to_string(), 4)].into_iter().collect(),
            ..Bindings::default()
        };
        let result = interpreter
            .run("add", &[Arg::Tensor(x), Arg::Tensor(y)], &bindings)
            .unwrap();
        match result {
            value::Value::Tensor(s) => {
                let data = interpreter.gather_for_test(&s);
                assert_eq!(data, vec![11.0, 22.0, 33.0, 44.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }
        let total = interpreter
            .run("sum", &[Arg::Tensor(x)], &bindings)
            .unwrap();
        assert!(matches!(total, Value::Scalar(DType::F32, v) if v == 10.0f32 as f64));
    }

    #[test]
    fn packed_values_are_readable_but_not_writable() {
        let p = program(&["\
fn read[N](x: &tensor[N] q4g64) -> f32:
    let v = decode(x)
    return reduce(v, 0, sum)
"]);
        let mut interpreter = Interpreter::new(&p);
        let packed = TensorData::random_packed(
            &mut Rng(0x1234_5678_9abc_def0),
            crate::repr::lookup("q4g64").unwrap(),
            vec![128],
        );
        let id = interpreter.add_tensor(packed);
        let bindings = Bindings {
            shapes: [("N".to_string(), 128)].into_iter().collect(),
            ..Bindings::default()
        };
        let result = interpreter.run("read", &[Arg::Tensor(id)], &bindings);
        assert!(result.is_ok(), "{result:?}");
    }
}
