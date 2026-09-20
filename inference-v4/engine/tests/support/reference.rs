//! Numerical fixture harness. The structured interpreter is the semantic oracle and
//! runs every fixture under two partitionings; Metal goes through joint selection and
//! the ordinary runtime. Neither path accepts an implementation choice from a test.
#![allow(dead_code)]
pub use seismic_lang::family::Workload;
pub use seismic_lang::interp::{Arg, Interpreter, TensorData};
use seismic_lang::{
    interp::Uniform,
    sir::{Definition, Program},
    types::{DType, Elem, Extent, Shaped, Ty},
};
use seismic_runtime::{
    plan::{Bindings, PlanCompiler},
    Buffer,
};
use std::collections::HashMap;

/// Slice widths every reference fixture must agree under.
pub const WIDTHS: [i64; 2] = [1, 7];

pub fn interpreter(program: &Program, width: i64) -> Interpreter<'_> {
    let mut vm = Interpreter::new(program);
    vm.partitioner = Box::new(Uniform(width));
    vm
}
/// The definition stating an entry's ABI: the family's first implementation.
pub fn entry<'a>(program: &'a Program, name: &str) -> &'a Definition {
    let family = program
        .resolve_family(name)
        .unwrap_or_else(|error| panic!("{error}"));
    program.definition(
        *family
            .bodies
            .iter()
            .chain(&family.lowerings)
            .next()
            .unwrap(),
    )
}
pub fn extents(tensor: &Shaped, shapes: &HashMap<String, i64>) -> Vec<usize> {
    tensor
        .axes
        .iter()
        .map(|axis| match axis {
            Extent::Semantic(extent) => extent.eval(&|p| shapes.get(p).copied()).unwrap() as usize,
            Extent::Structural(_) => panic!("entry tensor has a structural extent"),
        })
        .collect()
}
pub fn element(tensor: &TensorData) -> Elem {
    match tensor {
        TensorData::Dense { dtype, .. } => Elem::Dtype(*dtype),
        TensorData::Packed { repr, .. } => Elem::Repr(repr.name.into()),
    }
}
/// Element parameters are whatever the bound tensors are; disagreement is a test bug.
fn elements<'t>(
    definition: &Definition,
    tensor: impl Fn(&str, usize) -> &'t TensorData,
) -> HashMap<String, Elem> {
    let mut elements = HashMap::new();
    for (ordinal, param) in definition.params.iter().enumerate() {
        if let Ty::Tensor(Shaped {
            elem: Elem::Param(p),
            ..
        })
        | Ty::View(Shaped {
            elem: Elem::Param(p),
            ..
        }) = &param.ty
        {
            let bound = element(tensor(&param.name, ordinal));
            if let Some(previous) = elements.insert(p.clone(), bound.clone()) {
                assert_eq!(
                    previous, bound,
                    "{}: element parameter {p}",
                    definition.name
                );
            }
        }
    }
    elements
}
pub fn workload(shapes: &HashMap<String, i64>, elements: &HashMap<String, Elem>) -> Workload {
    Workload {
        shapes: shapes.iter().map(|(n, v)| (n.clone(), *v)).collect(),
        elems: elements
            .iter()
            .map(|(n, e)| (n.clone(), e.clone()))
            .collect(),
        ..Workload::default()
    }
}
/// Positional interpreter call; tensors may alias by naming one id twice.
pub fn run(vm: &mut Interpreter<'_>, name: &str, args: &[Arg], shapes: &HashMap<String, i64>) {
    let tensors = &vm.tensors;
    let elements = elements(entry(vm.program, name), |parameter, ordinal| {
        match &args[ordinal] {
            Arg::Tensor(id) => &tensors[*id],
            Arg::Scalar(_) | Arg::Range(..) => panic!("{name}.{parameter} is a tensor"),
        }
    });
    vm.run(name, args, &workload(shapes, &elements))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
}
/// Zeroed dense storage for every tensor parameter; `dtype` resolves element parameters.
pub fn allocate(
    program: &Program,
    name: &str,
    shapes: &HashMap<String, i64>,
    dtype: impl Fn(&str) -> DType,
) -> HashMap<String, TensorData> {
    entry(program, name)
        .params
        .iter()
        .filter_map(|param| {
            let (Ty::Tensor(tensor) | Ty::View(tensor)) = &param.ty else {
                return None;
            };
            let dtype = match &tensor.elem {
                Elem::Dtype(d) => *d,
                Elem::Param(p) => dtype(p),
                Elem::Repr(_) => panic!("{name}.{} is packed", param.name),
            };
            let shape = extents(tensor, shapes);
            let count = shape.iter().product();
            Some((
                param.name.clone(),
                TensorData::dense(dtype, shape, vec![0.; count]),
            ))
        })
        .collect()
}
/// Publish host values at the tensor's dtype.
pub fn fill(tensor: &mut TensorData, values: impl IntoIterator<Item = f64>) {
    let mut count = 0;
    for (flat, value) in values.into_iter().enumerate() {
        tensor.set(flat, value);
        count = flat + 1;
    }
    assert_eq!(count, tensor.shape().iter().product::<usize>());
}
pub fn values(tensor: &TensorData) -> Vec<f32> {
    (0..tensor.shape().iter().product())
        .map(|flat| tensor.get(flat) as f32)
        .collect()
}

struct Bound<'a> {
    buffers: HashMap<String, HashMap<String, Buffer>>,
    scalars: &'a HashMap<String, f64>,
}
impl Bindings for Bound<'_> {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        self.buffers.get(root)?.get(plane)
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        self.scalars.get(name).copied()
    }
}
pub enum Backend<'a> {
    Interpreter(&'a Program, i64),
    Metal(PlanCompiler<'a>),
}
impl Backend<'_> {
    pub fn program(&self) -> &Program {
        match self {
            Backend::Interpreter(program, _) => program,
            Backend::Metal(compiler) => compiler.program(),
        }
    }
    /// Run one entry over named tensors, leaving every dense tensor as the entry left it.
    pub fn run(
        &mut self,
        name: &str,
        shapes: &HashMap<String, i64>,
        tensors: &mut HashMap<String, TensorData>,
        scalars: &HashMap<String, f64>,
    ) -> Option<(seismic_runtime::Selection, Workload)> {
        let definition = entry(self.program(), name).clone();
        let elements = elements(&definition, |parameter, _| &tensors[parameter]);
        match self {
            Backend::Interpreter(program, width) => {
                let mut vm = interpreter(program, *width);
                let mut ids = Vec::new();
                let args = definition
                    .params
                    .iter()
                    .map(|param| match &param.ty {
                        Ty::Tensor(_) | Ty::View(_) => {
                            let id = vm.add_tensor(tensors[&param.name].clone());
                            ids.push((param.name.clone(), id));
                            Arg::Tensor(id)
                        }
                        _ => Arg::Scalar(scalars[&param.name]),
                    })
                    .collect::<Vec<_>>();
                vm.run(name, &args, &workload(shapes, &elements))
                    .unwrap_or_else(|e| panic!("{name} at width {width}: {e}"));
                for (parameter, id) in ids {
                    tensors.insert(parameter, vm.tensors[id].clone());
                }
                None
            }
            Backend::Metal(compiler) => {
                let mut plan = compiler.compile_entry(name, shapes, &elements).unwrap();
                let device = compiler.device();
                let mut bound = Bound {
                    buffers: HashMap::new(),
                    scalars,
                };
                for (parameter, tensor) in tensors.iter() {
                    let planes = match tensor {
                        TensorData::Dense { .. } => vec![""],
                        TensorData::Packed { repr, .. } => {
                            repr.planes().iter().map(|p| p.name).collect()
                        }
                    };
                    let buffers = planes
                        .into_iter()
                        .zip(tensor.device_bytes())
                        .map(|(plane, bytes)| {
                            (plane.to_string(), device.buffer_from(&bytes).unwrap())
                        })
                        .collect();
                    bound.buffers.insert(parameter.clone(), buffers);
                }
                plan.execute(&bound)
                    .unwrap_or_else(|e| panic!("{name} on metal: {e}"));
                for (parameter, tensor) in tensors.iter_mut() {
                    if matches!(tensor, TensorData::Dense { .. }) {
                        let buffer = &bound.buffers[parameter][""];
                        let mut bytes = vec![0; buffer.len()];
                        buffer.read(&mut bytes).unwrap();
                        tensor.load_device_bytes(&bytes);
                    }
                }
                let kernel = plan.kernel().unwrap();
                let selection = kernel.borrow().selection().clone();
                Some((
                    selection,
                    Workload {
                        precision: compiler.settings().precision,
                        ..workload(shapes, &elements)
                    },
                ))
            }
        }
    }
}
