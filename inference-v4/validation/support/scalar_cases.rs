//! Independent small semantic cases shared by backend qualification tests.
use seismic_lang::{
    lowered_ir::LoweredIr,
    program::{compile, SourceFile},
};
use std::collections::HashMap;
pub fn exercise(mut run: impl FnMut(&LoweredIr, &mut [Vec<u8>], &[f64]), backend: &str) {
    let program = compile(&[SourceFile {
        path: "scalar-semantics.seismic".into(),
        text: include_str!("../programs/scalar-semantics.seismic").into(),
    }])
    .unwrap_or_else(|e| panic!("{e:?}"));
    let lower =
        |name| seismic_lang::lower::lower(&program, name, backend, &HashMap::new()).unwrap();
    let bytes = |values: Vec<f32>| {
        values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let mut empty = [bytes(vec![7.0]), bytes(vec![9.0])];
    run(&lower("empty_window"), &mut empty, &[]);
    assert_eq!(
        empty[1],
        bytes(vec![0.0]),
        "an empty stream executes no pieces"
    );
    let mut values = [
        bytes(vec![
            f32::from_bits(0x3f800001),
            f32::from_bits(0x3f7ffffe),
            -1.0,
        ]),
        vec![0; 8],
    ];
    run(&lower("explicit_fma"), &mut values, &[]);
    assert_eq!(
        values[1],
        bytes(vec![0.0, f32::from_bits(0xa8800000)]),
        "only explicit fma may contract multiplication and addition"
    );
    let mut row = vec![-8.0f32; 65];
    row[0] = f32::NAN;
    row[7] = 19.0;
    row[64] = 19.0;
    let input = [row.clone(), vec![f32::NEG_INFINITY; 65], vec![f32::NAN; 65]].concat();
    let mut values = [bytes(input), vec![0; 12], vec![0; 12], vec![0; 12]];
    run(&lower("reduction_extrema"), &mut values, &[]);
    assert_eq!(
        values[1],
        bytes(vec![-8.0, f32::NEG_INFINITY, f32::INFINITY]),
        "minimum ignores NaNs"
    );
    assert_eq!(
        values[2],
        bytes(vec![19.0, f32::NEG_INFINITY, f32::NEG_INFINITY]),
        "maximum ignores NaNs"
    );
    assert_eq!(
        values[3],
        [7i32, 0, 0]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>(),
        "argmax tie and no-improvement behavior"
    );
    for (input, expected) in [(row, 7), (vec![f32::NEG_INFINITY; 65], 0)] {
        let target = input[expected];
        let mut values = [bytes(input), vec![0; 4], vec![0; 4]];
        run(&lower("argmax_lookup"), &mut values, &[]);
        assert_eq!(values[1], target.to_le_bytes());
        assert_eq!(values[2], (expected as i32).to_le_bytes());
    }
    let bf = [0x3f80u16, 0x3b80, 0x3b80, 0x3b80]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let hf = [0x3c00u16, 0x1000, 0x1000, 0x1000]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let mut values = [bf, hf, vec![0; 8]];
    run(&lower("narrow_sum"), &mut values, &[]);
    assert_eq!(
        values[2],
        bytes(vec![1.0, 1.0]),
        "narrow accumulation publishes every addition"
    );
    let mut floats = vec![0.0; 65];
    floats[..4].copy_from_slice(&[1e20, 1.0, -1e20, 1.0]);
    let narrow = |first: u16, rest: u16| {
        std::iter::once(first)
            .chain(std::iter::repeat_n(rest, 64))
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let mut values = [
        bytes(floats),
        narrow(0x3f80, 0x3b80),
        narrow(0x3c00, 0x1000),
        vec![0; 12],
    ];
    run(&lower("ordered_large_sum"), &mut values, &[]);
    assert_eq!(
        values[3],
        bytes(vec![1.0, 1.0, 1.0]),
        "ordered reductions preserve axis order and narrow publication across lanes"
    );
    let mut integers = vec![i32::MIN; 66];
    integers[33 + 8] = 17;
    integers[33 + 32] = 17;
    let mut values = [
        integers.into_iter().flat_map(i32::to_le_bytes).collect(),
        vec![0; 8],
        vec![0; 8],
        vec![0; 8],
    ];
    run(&lower("integer_extrema"), &mut values, &[]);
    for (got, expected) in values[1..]
        .iter()
        .zip([[i32::MIN, i32::MIN], [i32::MIN, 17], [0, 8]])
    {
        assert_eq!(
            *got,
            expected
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }
    let float = |bytes: &[u8]| f32::from_le_bytes(bytes.try_into().unwrap());
    let input = [1.0f32, -7.0, 3.25, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    for position in 0..4 {
        let mut buffers = [input.clone(), vec![0; 4]];
        run(
            &seismic_lang::lower::lower(&program, "bounded_index", backend, &HashMap::new())
                .unwrap(),
            &mut buffers,
            &[position as f64],
        );
        assert_eq!(buffers[1], input[position * 4..position * 4 + 4]);
    }
    let mut buffers = [input.clone(), vec![0; 16]];
    run(
        &seismic_lang::lower::lower(&program, "tile_copy", backend, &HashMap::new()).unwrap(),
        &mut buffers,
        &[],
    );
    assert_eq!(buffers[1], input, "tile copy must not alias source");
    let mut buffers = [input.clone(), vec![0; 16]];
    run(
        &seismic_lang::lower::lower(&program, "changed_source", backend, &HashMap::new()).unwrap(),
        &mut buffers,
        &[],
    );
    let expected = [3.0f32, -13.0, 7.5, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(
        buffers[1], expected,
        "producer must retain original source values"
    );
    let mut buffers = [vec![0; 4]];
    run(
        &seismic_lang::lower::lower(&program, "loop_carried", backend, &HashMap::new()).unwrap(),
        &mut buffers,
        &[],
    );
    assert_eq!(float(&buffers[0]), 7.0, "loop-carried tile value");
    for (choose, expected) in [(0.0, 11.0), (1.0, 7.0)] {
        let mut buffers = [vec![0; 4]];
        run(
            &seismic_lang::lower::lower(&program, "branch_merge", backend, &HashMap::new())
                .unwrap(),
            &mut buffers,
            &[choose],
        );
        assert_eq!(float(&buffers[0]), expected, "branch merge");
    }
    let mut buffers = [0x3f81u16.to_le_bytes().to_vec(), vec![0; 4]];
    run(
        &seismic_lang::lower::lower(&program, "bf16_rounding", backend, &HashMap::new()).unwrap(),
        &mut buffers,
        &[],
    );
    assert_eq!(
        float(&buffers[1]),
        1.015625,
        "BF16 multiply rounds before widening"
    );
    let mut buffers = [vec![0; 4]];
    run(
        &seismic_lang::lower::lower(&program, "mixed_scalars", backend, &HashMap::new()).unwrap(),
        &mut buffers,
        &[1.0, 2.0, 4.0, 0.0],
    );
    assert_eq!(float(&buffers[0]), 16.0, "mixed-width scalar ABI");

    for name in ["large_snapshot", "snapshot_reduction"] {
        let input = (0..65).map(|i| i as f32 - 31.0).collect::<Vec<_>>();
        let expected = if name == "snapshot_reduction" {
            input.iter().sum::<f32>().to_le_bytes().to_vec()
        } else {
            input
                .iter()
                .rev()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let mut buffers = [
            input
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
            vec![0; expected.len()],
        ];
        run(
            &seismic_lang::lower::lower(&program, name, backend, &HashMap::new()).unwrap(),
            &mut buffers,
            &[],
        );
        assert_eq!(buffers[0], vec![0; 260], "{name}: backing overwritten");
        assert_eq!(buffers[1], expected, "{name}: original snapshot retained");
    }
    let input = (0..65)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let mut buffers = [input.clone(), vec![255; 4]];
    run(
        &seismic_lang::lower::lower(&program, "loaded_mutation", backend, &HashMap::new()).unwrap(),
        &mut buffers,
        &[],
    );
    assert_eq!(
        buffers[0], input,
        "loaded tile mutation must not affect tensor backing"
    );
    assert_eq!(buffers[1], vec![0; 4]);
    let mut buffers = [0x3c01u16.to_le_bytes().to_vec(), vec![0; 4]];
    run(
        &seismic_lang::lower::lower(&program, "half_rounding", backend, &HashMap::new()).unwrap(),
        &mut buffers,
        &[1.0009765625],
    );
    assert_eq!(
        float(&buffers[1]),
        f32::from_bits(0x3f80_4000),
        "half multiply publishes before widening, with typed half scalar argument"
    );
}
