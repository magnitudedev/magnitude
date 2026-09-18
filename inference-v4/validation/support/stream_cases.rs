//! Runtime domains, tails and loop-carried state with independent integer-exact sums.
use seismic_lang::{
    lowered_ir::LoweredIr, lower::Options,
    program::{compile, SourceFile},
    Scope,
};
use std::collections::HashMap;
pub fn exercise(
    mut run: impl FnMut(&LoweredIr, &mut [Vec<u8>]) -> Result<(), String>,
    backend: &str,
) {
    let p = compile(&[SourceFile {
        path: "stream.seismic.portable".into(), scope: Scope::Portable,
        text: "fn stream[T](x: tensor[T] f32, visible: tensor[2] i32, out: tensor[1] f32):\n  acc = tile[1] f32\n  for i in owned(acc): acc[i] = 0.0\n  for t in load(x[visible[0]:visible[1]], over=0):\n    acc[0] += reduce(t, 0, sum)\n  store(acc, out)\n".into(),
    }], &[]).unwrap();
    for piece in [None, Some(1), Some(17), Some(64), Some(200)] {
        let l = seismic_lang::lower::lower_with(
            &p,
            "stream",
            backend,
            &HashMap::from([("T".into(), 137)]),
            &Options { piece },
        )
        .unwrap();
        for (start, end) in [
            (0i32, 137i32),
            (3, 132),
            (16, 34),
            (71, 72),
            (137, 137),
            (0, 0),
            (1, 0),
            (-1, 2),
            (0, 138),
        ] {
            let input: Vec<f32> = (0..137).map(|i| i as f32 - 68.0).collect();
            let mut buffers = [
                input.iter().flat_map(|v| v.to_le_bytes()).collect(),
                [start, end]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect(),
                vec![0; 4],
            ];
            let result = run(&l, &mut buffers);
            if start < 0 || end < start || end > 137 {
                assert!(result.is_err(), "invalid domain [{start},{end}) admitted");
            } else {
                result.unwrap_or_else(|e| panic!("piece={piece:?} [{start},{end}): {e}"));
                let expected: f32 = input[start as usize..end as usize].iter().sum();
                assert_eq!(
                    f32::from_le_bytes(buffers[2].as_slice().try_into().unwrap()),
                    expected,
                    "piece={piece:?} [{start},{end})"
                );
            }
        }
    }
}
