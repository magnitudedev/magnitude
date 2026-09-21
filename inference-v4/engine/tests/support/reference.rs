//! Numerical fixture harness. The structured interpreter is the semantic oracle and
//! runs every fixture under two partitionings; Metal goes through the sealed
//! compiler pipeline and submission. Neither path accepts an implementation
//! choice from a test.
#![allow(dead_code)]
use seismic_lang::interp::value::Backing;
pub use seismic_lang::interp::value::Value as InterpValue;
pub use seismic_lang::interp::{Arg, Bindings as InterpBindings, Interpreter, TensorData};
use seismic_lang::{
    logical::specialization::{ShapeBinding, SpecializationDomain},
    sir::{Definition, Program},
    types::{DType, Elem, ExtentExpr, TensorType, ValueType},
};
use seismic_runtime::{
    invocation::Bindings,
    plan::{InvocationResults, PlanCompiler},
    submission::Submission,
    Buffer,
};
use std::collections::{BTreeMap, HashMap};

/// The reference interpreter over a checked program.
pub fn interpreter(program: &Program) -> Interpreter<'_> {
    Interpreter::new(program)
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
pub fn extents(tensor: &TensorType, shapes: &HashMap<String, i64>) -> Vec<usize> {
    tensor
        .axes
        .iter()
        .map(|axis| match axis {
            ExtentExpr::Sym(extent) => extent.eval(&|p| shapes.get(p).copied()).unwrap() as usize,
            ExtentExpr::Static(value) => *value as usize,
            ExtentExpr::Runtime(_) => panic!("entry tensor extent is runtime-dependent"),
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
        if let ValueType::Tensor(TensorType {
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

/// Interpreter shape/element bindings of one geometry.
fn interp_bindings(
    shapes: &HashMap<String, i64>,
    elements: &HashMap<String, Elem>,
) -> InterpBindings {
    InterpBindings {
        shapes: shapes.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        elems: elements.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
    }
}

/// The complete specialization domain of one entry geometry: every shape
/// bound exactly and every element parameter concretized by its tensor.
fn domain(
    program: &Program,
    name: &str,
    shapes: &HashMap<String, i64>,
    elements: &HashMap<String, Elem>,
) -> SpecializationDomain {
    SpecializationDomain::new(
        program,
        name,
        shapes
            .iter()
            .map(|(n, v)| {
                let value = u64::try_from(*v)
                    .unwrap_or_else(|_| panic!("shape {n}={v} is not a valid extent"));
                (n.clone(), ShapeBinding::Exact(value))
            })
            .collect(),
        elements
            .iter()
            .map(|(n, e)| (n.clone(), e.clone()))
            .collect(),
    )
    .unwrap_or_else(|error| panic!("{name}: {error}"))
}

/// The specialization domain of one entry over bound tensors: element
/// parameters concretized by their tensors, every shape bound exactly.
pub fn tensor_domain(
    program: &Program,
    name: &str,
    shapes: &HashMap<String, i64>,
    tensors: &HashMap<String, TensorData>,
) -> SpecializationDomain {
    let definition = entry(program, name);
    let bound = elements(definition, |parameter, _| &tensors[parameter]);
    domain(program, name, shapes, &bound)
}

/// Positional interpreter call; tensors may alias by naming one id twice.
/// Returns the entry's result value.
pub fn run(
    vm: &mut Interpreter<'_>,
    name: &str,
    args: &[Arg],
    shapes: &HashMap<String, i64>,
) -> InterpValue {
    run_with_elements(vm, name, args, shapes, &HashMap::new())
}

/// Positional interpreter call with default element bindings for element
/// parameters no tensor argument carries.
pub fn run_with_elements(
    vm: &mut Interpreter<'_>,
    name: &str,
    args: &[Arg],
    shapes: &HashMap<String, i64>,
    extra: &HashMap<String, Elem>,
) -> InterpValue {
    let tensors = &vm.tensors;
    let mut bound = elements(entry(vm.program, name), |parameter, ordinal| {
        match &args[ordinal] {
            Arg::Tensor(id) => &tensors[*id],
            Arg::Scalar(_) | Arg::Range(..) => panic!("{name}.{parameter} is a tensor"),
        }
    });
    for (name, element) in extra {
        bound
            .entry(name.clone())
            .or_insert_with(|| element.clone());
    }
    vm.run(name, args, &interp_bindings(shapes, &bound))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
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
            let ValueType::Tensor(tensor) = &param.ty else {
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
        tensor.set(flat, value).unwrap();
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
    shapes: &'a HashMap<String, i64>,
}
impl Bindings for Bound<'_> {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        self.buffers.get(root)?.get(plane)
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        self.scalars.get(name).copied()
    }
    fn shape(&self, name: &str) -> Option<u64> {
        self.shapes.get(name).and_then(|v| u64::try_from(*v).ok())
    }
}
/// Tensor result leaves by canonical path.
fn tensor_leaf_paths(ty: &ValueType) -> HashMap<Vec<u32>, &ValueType> {
    fn walk<'a>(
        ty: &'a ValueType,
        prefix: &mut Vec<u32>,
        out: &mut HashMap<Vec<u32>, &'a ValueType>,
    ) {
        match ty {
            ValueType::Tensor(_) => {
                out.insert(prefix.clone(), ty);
            }
            ValueType::Tuple(children) => {
                for (ordinal, child) in children.iter().enumerate() {
                    prefix.push(ordinal as u32);
                    walk(child, prefix, out);
                    prefix.pop();
                }
            }
            _ => (),
        }
    }
    let mut out = HashMap::new();
    walk(ty, &mut Vec::new(), &mut out);
    out
}

/// Publish owned result leaves under `result{leaf}` names: the leaf's
/// outer tuple ordinal (a single result is `result0`).
fn publish_interpreted(value: &InterpValue, tensors: &mut HashMap<String, TensorData>) {
    fn walk(
        value: &InterpValue,
        ordinal: usize,
        tensors: &mut HashMap<String, TensorData>,
    ) -> usize {
        match value {
            InterpValue::Tuple(items) => {
                let mut leaves = 0;
                for (index, item) in items.iter().enumerate() {
                    leaves += walk(item, ordinal + index, tensors);
                }
                leaves
            }
            InterpValue::Tensor(shaped) => {
                if let Backing::Owned(dense) = &shaped.backing {
                    tensors.insert(
                        format!("result{ordinal}"),
                        TensorData::dense(
                            dense.borrow().dtype,
                            shaped.shape.clone(),
                            dense.borrow().data.clone(),
                        ),
                    );
                }
                1
            }
            _ => 1,
        }
    }
    walk(value, 0, tensors);
}

/// Publish compiler-allocated result planes under `result{leaf}` names.
fn publish_compiled(
    results: &InvocationResults,
    definition: &Definition,
    shapes: &HashMap<String, i64>,
    elements: &HashMap<String, Elem>,
    tensors: &mut HashMap<String, TensorData>,
) {
    let leaves = tensor_leaf_paths(&definition.result);
    for plane in &results.planes {
        if !plane.plane.is_empty() {
            continue;
        }
        let Some(ty) = leaves.get(&plane.path) else {
            continue;
        };
        let Some(path) = plane.path.first().copied() else {
            continue;
        };
        let ValueType::Tensor(shaped) = ty else {
            continue;
        };
        let element = match &shaped.elem {
            Elem::Param(p) => elements.get(p.as_str()).unwrap_or(&shaped.elem),
            concrete => concrete,
        };
        let Elem::Dtype(dtype) = element else {
            continue;
        };
        let shape = extents(shaped, shapes);
        let count = shape.iter().product::<usize>().max(1);
        let mut bytes = vec![0u8; plane.buffer.len()];
        plane.buffer.read(&mut bytes).unwrap();
        let mut tensor = TensorData::dense(*dtype, shape, vec![0.; count]);
        tensor.load_device_bytes(&bytes);
        tensors.insert(format!("result{path}"), tensor);
    }
}

pub enum Backend<'a> {
    /// Element bindings beyond what tensor arguments imply (activation
    /// dtypes that no parameter carries).
    Interpreter(&'a Program, HashMap<String, Elem>),
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
    ) {
        let definition = entry(self.program(), name).clone();
        let bound_elements = elements(&definition, |parameter, _| &tensors[parameter]);
        match self {
            Backend::Interpreter(program, extra_elements) => {
                let mut vm = interpreter(program);
                let mut ids = Vec::new();
                let args = definition
                    .params
                    .iter()
                    .map(|param| match &param.ty {
                        ValueType::Tensor(_) => {
                            let id = vm.add_tensor(tensors[&param.name].clone());
                            ids.push((param.name.clone(), id));
                            Arg::Tensor(id)
                        }
                        _ => Arg::Scalar(scalars[&param.name]),
                    })
                    .collect::<Vec<_>>();
                let mut elements = bound_elements.clone();
                for (name, element) in extra_elements {
                    elements.entry(name.clone()).or_insert_with(|| element.clone());
                }
                let value = vm
                    .run(name, &args, &interp_bindings(shapes, &elements))
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                for (parameter, id) in ids {
                    tensors.insert(parameter, vm.tensors[id].clone());
                }
                publish_interpreted(&value, tensors);
            }
            Backend::Metal(compiler) => {
                let domain = domain(compiler.program(), name, shapes, &bound_elements);
                let plan = compiler
                    .compile_entry(&domain)
                    .unwrap_or_else(|e| panic!("{name} compile: {e}"));
                let device = compiler.device();
                let mut bound = Bound {
                    buffers: HashMap::new(),
                    scalars,
                    shapes,
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
                let invocation = plan
                    .prepare(&bound)
                    .unwrap_or_else(|e| panic!("{name} prepare on metal: {e}"));
                let results = Submission::single(invocation)
                    .execute()
                    .unwrap_or_else(|e| panic!("{name} execute on metal: {e}"))
                    .remove(0);
                for (parameter, tensor) in tensors.iter_mut() {
                    if matches!(tensor, TensorData::Dense { .. }) {
                        let buffer = &bound.buffers[parameter][""];
                        let mut bytes = vec![0; buffer.len()];
                        buffer.read(&mut bytes).unwrap();
                        tensor.load_device_bytes(&bytes);
                    }
                }
                publish_compiled(&results, &definition, shapes, &bound_elements, tensors);
            }
        }
    }
}
