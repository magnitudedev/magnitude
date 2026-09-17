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
    let cpw = rep.codes_per_word() as usize;
    let mut words = vec![0u32; n * k / cpw];
    let groups = n * k / rep.group as usize;
    let mut scale = Vec::with_capacity(groups);
    let mut bias = Vec::with_capacity(groups);
    for _ in 0..groups {
        scale.push((rng.unit() * 0.1 + 0.01) as f32);
        bias.push((rng.unit() - 0.5) as f32);
    }
    let mut decoded = vec![0f64; n * k];
    for row in 0..n {
        for col in 0..k {
            let code = (rng.next() % 16) as u32;
            let w = row * (k / cpw) + col / cpw;
            words[w] |= code << ((col % cpw) as u32 * rep.bits);
            let g = row * (k / rep.group as usize) + col / rep.group as usize;
            decoded[row * k + col] = (scale[g] as f64 * code as f64 + bias[g] as f64) as f32 as f64;
        }
    }
    (TensorData::Packed { repr: rep, shape: vec![n, k], words, scale, bias }, decoded)
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
    // `projection` computes RPI output rows per work item; one row per item here.
    let shapes = HashMap::from([("N".to_string(), n as i64), ("K".to_string(), k as i64), ("RPI".to_string(), 1)]);
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
