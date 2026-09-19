//! Standard streaming attention compared to an independent f64 softmax reference.
use seismic_lang::{
    interp::TensorData,
    lower::Options,
    lowered_ir::LoweredIr,
    program::{compile, SourceFile},
    types::DType,
};
use std::collections::HashMap;
pub fn exercise(
    mut run: impl FnMut(&LoweredIr, &mut [Vec<u8>], &[f64]) -> Result<(), String>,
    backend: &str,
) {
    let portable = [
        (
            "attention.seismic",
            include_str!("../../seismic-std/lib/kernels/attention.seismic"),
        ),
        (
            "matmul.seismic",
            include_str!("../../seismic-std/lib/constructs/matmul.seismic"),
        ),
    ];
    let mut files = portable
        .into_iter()
        .map(|(name, text)| SourceFile {
            path: name.into(),
            text: text.into(),
        })
        .collect::<Vec<_>>();
    let lowering = match backend {
        "cpu" => include_str!("../../seismic-std/lib/constructs/matmul-cpu.seismic"),
        "cuda" => include_str!("../../seismic-std/lib/constructs/matmul-cuda.seismic"),
        "metal" => include_str!("../../seismic-std/lib/constructs/matmul-metal.seismic"),
        _ => panic!("unsupported test backend"),
    };
    files.push(SourceFile {
        path: format!("matmul-{backend}.seismic").into(),
        text: lowering.into(),
    });
    let p = compile(&files).unwrap_or_else(|e| panic!("{e:?}"));
    let shapes = HashMap::from([
        ("Q".into(), 3),
        ("T".into(), 19),
        ("H".into(), 4),
        ("KV".into(), 2),
        ("W".into(), 8),
    ]);
    let tensor = |shape: Vec<usize>, seed: usize| {
        let n = shape.iter().product();
        TensorData::dense(
            DType::BF16,
            shape,
            (0..n)
                .map(|i| (((i * 17 + seed) % 47) as f64 - 23.0) / 16.0)
                .collect(),
        )
    };
    let q = tensor(vec![3, 4, 8], 3);
    let k = tensor(vec![19, 2, 8], 7);
    let v = tensor(vec![19, 2, 8], 11);
    let ranges = [(0usize, 19usize), (3, 17), (18, 19)];
    let scale = 0.25f64;
    let mut expected = Vec::new();
    for (row, (start, end)) in ranges.iter().copied().enumerate() {
        for head in 0..4 {
            let scores = (start..end)
                .map(|t| {
                    (0..8)
                        .map(|w| {
                            q.get((row * 4 + head) * 8 + w) * k.get((t * 2 + head / 2) * 8 + w)
                        })
                        .sum::<f64>()
                        * scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|x| (x - maximum).exp())
                .collect::<Vec<_>>();
            let denominator: f64 = weights.iter().sum();
            for w in 0..8 {
                expected.push(
                    weights
                        .iter()
                        .enumerate()
                        .map(|(i, weight)| weight * v.get(((start + i) * 2 + head / 2) * 8 + w))
                        .sum::<f64>()
                        / denominator,
                );
            }
        }
    }
    for piece in [None, Some(1), Some(7), Some(32)] {
        let l = seismic_lang::lower::lower_specialized(
            &p,
            "attention",
            backend,
            &shapes,
            &HashMap::from([(
                "A".into(),
                seismic_lang::types::Elem::Dtype(seismic_lang::types::DType::BF16),
            )]),
            &Options {
                piece,
                ..Default::default()
            },
        )
        .unwrap();
        let mut buffers = [
            q.device_bytes()[0].clone(),
            k.device_bytes()[0].clone(),
            v.device_bytes()[0].clone(),
            ranges
                .into_iter()
                .flat_map(|(s, e)| [s as i32, e as i32])
                .flat_map(i32::to_le_bytes)
                .collect(),
            vec![0; 3 * 4 * 8 * 2],
        ];
        run(&l, &mut buffers, &[scale]).unwrap_or_else(|e| panic!("piece={piece:?}: {e}"));
        let mut output = tensor(vec![3, 4, 8], 0);
        output.load_device_bytes(&buffers[4]);
        for (i, expected) in expected.iter().enumerate() {
            let got = output.get(i);
            assert!(
                (got - expected).abs() <= 0.006 + 0.006 * expected.abs(),
                "piece={piece:?} output[{i}]={got}, expected {expected}"
            );
        }
    }
}
