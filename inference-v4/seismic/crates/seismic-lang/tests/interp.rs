//! The interpreter computes what the portable bodies say.

use seismic_lang::interp::{Arg, Interpreter, TensorData};
use seismic_lang::program::{collect_files, compile};
use seismic_lang::repr;
use seismic_lang::types::DType;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn std_program() -> seismic_lang::program::Program {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../seismic-std/lib");
    let files = collect_files(&[PathBuf::from(root)]).unwrap();
    compile(&files, &["metal".to_string(), "cpu".to_string()]).unwrap_or_else(|es| panic!("{}", es.iter().map(|e| e.render()).collect::<Vec<_>>().join("\n")))
}

use seismic_lang::interp::Rng;

/// Pack a [n, k] matrix of q4 codes with per-group scale and bias; returns the tensor and the decoded values.
fn packed_q4g64(rng: &mut Rng, n: usize, k: usize) -> (TensorData, Vec<f64>) {
    let rep = repr::lookup("q4g64").unwrap();
    let tensor = TensorData::random_packed(rng, rep, vec![n, k]);
    let decoded = (0..n * k).map(|i| tensor.get(i)).collect();
    (tensor, decoded)
}

#[test]
fn projection_matches_direct_dot_products() {
    let program = std_program();
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (n, k) = (6, 128);
    let x: Vec<f64> = (0..k).map(|_| seismic_lang::numeric::bf16_round((rng.unit() - 0.5) as f32) as f64).collect();
    let (w, decoded) = packed_q4g64(&mut rng, n, k);
    let mut interp = Interpreter::new(&program);
    let xt = interp.add_tensor(TensorData::dense(DType::BF16, vec![1, k], x.clone()));
    let wt = interp.add_tensor(w);
    let out = interp.add_tensor(TensorData::dense(DType::BF16, vec![1, n], vec![0.0; n]));
    let shapes = HashMap::from([("N".to_string(), n as i64), ("K".to_string(), k as i64)]);
    interp.run("projection", &[Arg::Tensor(xt), Arg::Tensor(wt), Arg::Tensor(out)], &shapes).unwrap();
    for col in 0..n {
        let expect: f64 = (0..k).map(|i| x[i] * decoded[col * k + i]).sum();
        let got = interp.tensors[out].get(col);
        let tol = expect.abs() * 1e-2 + 1e-2; // bf16 output rounding
        assert!((got - expect).abs() <= tol, "col {col}: got {got}, expected {expect}");
    }
}

#[test]
fn rms_norm_matches_definition() {
    let program = std_program();
    let mut rng = Rng(7);
    let (r, w) = (3, 64);
    let x: Vec<f64> = (0..r * w).map(|_| seismic_lang::numeric::bf16_round((rng.unit() * 2.0 - 1.0) as f32) as f64).collect();
    let weight: Vec<f64> = (0..w).map(|_| seismic_lang::numeric::bf16_round((rng.unit() + 0.5) as f32) as f64).collect();
    let mut interp = Interpreter::new(&program);
    let xt = interp.add_tensor(TensorData::dense(DType::BF16, vec![r, w], x.clone()));
    let wt = interp.add_tensor(TensorData::dense(DType::BF16, vec![w], weight.clone()));
    let out = interp.add_tensor(TensorData::dense(DType::BF16, vec![r, w], vec![0.0; r * w]));
    let shapes = HashMap::from([("R".to_string(), r as i64), ("W".to_string(), w as i64)]);
    interp.run("rms_norm", &[Arg::Tensor(xt), Arg::Tensor(wt), Arg::Tensor(out), Arg::Scalar(1e-6)], &shapes).unwrap();
    for row in 0..r {
        let ss: f64 = (0..w).map(|i| x[row * w + i] * x[row * w + i]).sum();
        let inv = 1.0 / (ss / w as f64 + 1e-6).sqrt();
        for i in 0..w {
            let expect = x[row * w + i] * inv * weight[i];
            let got = interp.tensors[out].get(row * w + i);
            assert!((got - expect).abs() <= expect.abs() * 1e-2 + 1e-2, "({row},{i}): got {got}, expected {expect}");
        }
    }
}

#[test]
fn integer_division_is_euclidean_and_invalid_inputs_return_errors() {
    use seismic_lang::{program::SourceFile, Scope};
    let program=compile(&[SourceFile{path:"division.seismic.portable".into(),scope:Scope::Portable,text:"fn division(a: i32, b: i32, out: tensor[2] i32):\n  y = tile[2] i32\n  for i in owned(y):\n    if i == 0: y[i] = a / b\n    else: y[i] = a % b\n  store(y,out)\n".into()}],&[]).unwrap();
    let mut interpreter=Interpreter::new(&program);
    let output=interpreter.add_tensor(TensorData::dense(DType::I32,vec![2],vec![0.0;2]));
    for (a,b,q,r) in [(-17,3,-6,1),(-17,-3,6,1),(17,-3,-5,2)] {
        interpreter.run("division",&[Arg::Scalar(f64::from(a)),Arg::Scalar(f64::from(b)),Arg::Tensor(output)],&HashMap::new()).unwrap();
        assert_eq!(interpreter.tensors[output].get(0),f64::from(q));assert_eq!(interpreter.tensors[output].get(1),f64::from(r));
    }
    for (a,b) in [(1,0),(i32::MIN,-1)] {
        assert!(interpreter.run("division",&[Arg::Scalar(f64::from(a)),Arg::Scalar(f64::from(b)),Arg::Tensor(output)],&HashMap::new()).is_err());
    }
}

#[test]
fn integer_arithmetic_and_bit_casts_preserve_32bit_wrapping() {
    use seismic_lang::{program::SourceFile,Scope};
    let p=compile(&[SourceFile{path:"wrapping.seismic.portable".into(),scope:Scope::Portable,text:r#"
fn wrapping(a:i32,b:i32,out:tensor[11] i32):
  t=tile[11] i32
  for i in owned(t): t[i]=0
  t[0]=a+b
  t[1]=a-b
  t[2]=a*b
  t[3]=-a
  t[4]=abs(a)
  t[5]=i32(u32(a))
  t[6]=i32(~u32(a))
  t[7]=i32(u32(a)+u32(b))
  c=a
  c+=b
  t[8]=c
  t[9]=a
  t[9]-=b
  t[10]=a
  t[10]*=b
  store(t,out)
"#.into()}],&[]).unwrap();
    let mut vm=Interpreter::new(&p);
    let out=vm.add_tensor(TensorData::dense(DType::I32,vec![11],vec![0.0;11]));
    for (a,b) in [(i32::MAX,1),(i32::MIN,-1),(-1,i32::MAX),(65536,65536)] {
        vm.run("wrapping",&[Arg::Scalar(a as f64),Arg::Scalar(b as f64),Arg::Tensor(out)],&HashMap::new()).unwrap();
        for (i,want) in [a.wrapping_add(b),a.wrapping_sub(b),a.wrapping_mul(b),a.wrapping_neg(),a.wrapping_abs(),a,!a,a.wrapping_add(b),a.wrapping_add(b),a.wrapping_sub(b),a.wrapping_mul(b)].into_iter().enumerate() {assert_eq!(vm.tensors[out].get(i),want as f64);}
    }
}

#[test]
fn tile_expressions_share_scalar_order_wrapping_rounding_and_errors() {
    use seismic_lang::{program::SourceFile, Scope};
    for dtype in [DType::I32, DType::U32, DType::F16, DType::F32] {
        let expressions = ["a+b", "a-b", "a*b", "s-a", "s/a", "a/s", "a%s", "s%a"];
        let vector = expressions.iter().enumerate().map(|(i,e)| {
            format!("  v{i} = {e}\n  store(v{i},out[{i},:])\n")
        }).collect::<String>();
        let scalar = expressions.iter().enumerate().map(|(i,e)| {
            format!("    if i == {i}: r[i,j] = {}\n",e.replace('a',"a[j]").replace('b',"b[j]"))
        }).collect::<String>();
        let name = dtype.name();
        let text = format!("fn vector(x:tensor[4] {name},y:tensor[4] {name},s:{name},out:tensor[8,4] {name}):\n  a=load(x)\n  b=load(y)\n{vector}\nfn scalar(x:tensor[4] {name},y:tensor[4] {name},s:{name},out:tensor[8,4] {name}):\n  a=load(x)\n  b=load(y)\n  r=tile[8,4] {name}\n  for i,j in owned(r):\n    r[i,j] = 0\n{scalar}  store(r,out)\n");
        let p=compile(&[SourceFile{path:"elementwise.seismic.portable".into(),scope:Scope::Portable,text}],&[]).unwrap();
        let mut vm=Interpreter::new(&p);
        let values = match dtype {
            DType::I32 => vec![i32::MAX as f64,i32::MIN as f64,-7.0,3.0],
            DType::U32 => vec![u32::MAX as f64,2147483648.0,7.0,3.0],
            _ => vec![65504.0,-0.25,-7.0,3.0],
        };
        let x=vm.add_tensor(TensorData::dense(dtype,vec![4],values));
        let y=vm.add_tensor(TensorData::dense(dtype,vec![4],vec![2.0,1.0,6.0,3.0]));
        let vector=vm.add_tensor(TensorData::dense(dtype,vec![8,4],vec![0.0;32]));
        let scalar=vm.add_tensor(TensorData::dense(dtype,vec![8,4],vec![0.0;32]));
        for (entry,out) in [("vector",vector),("scalar",scalar)] {
            vm.run(entry,&[Arg::Tensor(x),Arg::Tensor(y),Arg::Scalar(7.0),Arg::Tensor(out)],&HashMap::new()).unwrap();
        }
        assert_eq!(vm.tensors[vector].device_bytes(),vm.tensors[scalar].device_bytes(),"{name}");
        if dtype.is_int() {
            for (entry,out) in [("vector",vector),("scalar",scalar)] {
                assert!(vm.run(entry,&[Arg::Tensor(x),Arg::Tensor(y),Arg::Scalar(0.0),Arg::Tensor(out)],&HashMap::new()).is_err(),"{name} {entry} division by zero");
            }
        }
    }
}
