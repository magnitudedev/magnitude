use seismic_lang::{
    interp::{Arg, Interpreter, TensorData},
    ir::Function,
    lower::{lower_selected, Options},
    lowered_ir::{Alternative, DecisionKind},
    program::{compile, Program, SourceFile},
    reduction::structured::{self, Tree},
    types::DType,
    Scope,
};
use std::collections::HashMap;

fn run(
    p: &Program,
    dtype: DType,
    shape: &[usize],
    output_shape: &[usize],
    output_dtype: DType,
    data: &[f64],
) -> Vec<u8> {
    let mut vm = Interpreter::new(p);
    let x = vm.add_tensor(TensorData::dense(dtype, shape.into(), data.into()));
    let out = vm.add_tensor(TensorData::dense(
        output_dtype,
        output_shape.into(),
        vec![0.0; output_shape.iter().product()],
    ));
    vm.run(
        "entry",
        &[Arg::Tensor(x), Arg::Tensor(out)],
        &HashMap::new(),
    )
    .unwrap();
    vm.tensors[out].device_bytes().remove(0)
}
fn check(
    dtype: DType,
    name: &str,
    operation: &str,
    shape: &[usize],
    axis: usize,
    data: &[f64],
    ordered: bool,
) {
    let mut result_shape = shape.to_vec();
    result_shape.remove(axis);
    let scalar = result_shape.is_empty();
    if scalar {
        result_shape.push(1);
    }
    let output_dtype = if operation == "argmax" {
        DType::I32
    } else {
        dtype
    };
    let output_name = if operation == "argmax" { "i32" } else { name };
    let dimensions = |shape: &[usize]| {
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    };
    let result = if scalar {
        format!("  y = tile[1] {output_name}\n  for i in owned(y): y[i] = r\n  store(y,out)\n")
    } else {
        "  store(r,out)\n".into()
    };
    let text=format!("fn entry(x:tensor[{}] {name},out:tensor[{}] {output_name}):\n  t = load(x)\n  r = reduce(t,{axis},{operation},ordered={ordered})\n{result}",dimensions(shape),dimensions(&result_shape));
    let p = compile(
        &[SourceFile {
            path: "primitive.seismic.portable".into(),
            text,
            scope: Scope::Portable,
        }],
        &[],
    )
    .unwrap();
    let reference = run(&p, dtype, shape, &result_shape, output_dtype, data);
    for tree in if ordered {
        vec![Tree::Ordered]
    } else {
        vec![Tree::Ordered, Tree::Pairwise, Tree::Explicit, Tree::SeedThenPairwise]
    } {
        let lowered = lower_selected(
            &p,
            "entry",
            "cpu",
            &HashMap::new(),
            &HashMap::new(),
            &Options::default(),
            &mut |d| {
                Ok(match d.kind {
                    DecisionKind::Reduction { .. }
                        if d.alternatives.contains(&Alternative::ReductionTree(tree)) =>
                    {
                        Alternative::ReductionTree(tree)
                    }
                    DecisionKind::ReductionSegments { extent } => {
                        Alternative::ReductionSegment(extent.min(2))
                    }
                    _ => d.alternatives.get(0).unwrap(),
                })
            },
        )
        .unwrap();
        for lowered in [lowered.clone(), structured::materialize(&lowered).unwrap()] {
            let selected = Program {
                functions: vec![Function {
                    name: lowered.name,
                    is_construct: false,
                    shape_params: vec![],
                    elem_params: vec![],
                    params: lowered.params,
                    index_params: lowered.index_params,
                    vars: lowered.vars,
                    body: lowered.body,
                }],
                lowerings: vec![],
                signatures: HashMap::new(),
            };
            assert_eq!(
                run(&selected, dtype, shape, &result_shape, output_dtype, data),
                reference,
                "{name} {operation} {shape:?} axis{axis} {tree:?}"
            );
        }
    }
}
#[test]
fn ordinary_reductions_preserve_scalar_tile_dtype_and_empty_contracts() {
    for (dtype, name, data) in [
        (DType::F32, "f32", vec![-7.0, 3.0, 9.0, -2.0, 1.0, 4.0]),
        (DType::F16, "f16", vec![-7.0, 3.0, 9.0, -2.0, 1.0, 4.0]),
        (DType::BF16, "bf16", vec![-7.0, 3.0, 9.0, -2.0, 1.0, 4.0]),
        (
            DType::I32,
            "i32",
            vec![i32::MAX as f64, 1.0, -3.0, i32::MIN as f64, -1.0, 9.0],
        ),
        (
            DType::U32,
            "u32",
            vec![u32::MAX as f64, 1.0, 3.0, 7.0, 2.0, 9.0],
        ),
        (DType::Bool, "bool", vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
    ] {
        for operation in ["sum", "max", "min", "argmax"] {
            for (shape, axis) in [(vec![6], 0), (vec![2, 3], 0), (vec![2, 3], 1)] {
                check(dtype, name, operation, &shape, axis, &data, true);
            }
            if operation != "argmax" {
                check(dtype, name, operation, &[0], 0, &[], true);
            }
        }
    }
}
#[test]
fn regrouped_extrema_keep_first_index_ties_and_nan_contract() {
    for data in [
        vec![f64::NAN; 7],
        vec![f64::NEG_INFINITY; 7],
        vec![2.0, 5.0, 5.0, f64::NAN, 4.0, 5.0, 1.0],
    ] {
        for operation in ["max", "min", "argmax"] {
            check(DType::F32, "f32", operation, &[7], 0, &data, false);
        }
    }
}
