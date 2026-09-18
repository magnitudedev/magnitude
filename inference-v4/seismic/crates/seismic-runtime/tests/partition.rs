#![cfg(target_os = "macos")]
use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
#[test]
#[ignore = "requires Metal hardware"]
fn metal_independent_partition() {
    let source="fn pointwise(x: tensor[2, 256] f32, out: tensor[2, 256] f32):\n  for row in parallel:\n    a = load(x[row])\n    y = tile[256] f32\n    for i in owned(y): y[i] = a[i] * 2.0 + 1.0\n    store(y,out[row])\n";
    let program = compile(
        &[SourceFile {
            path: "partition.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = lower(&program, "pointwise", "metal", &HashMap::new()).unwrap();
    let device = Device::metal().unwrap();
    let bytes = (0..512)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let expected = (0..512)
        .flat_map(|i| (i as f32 * 2. + 1.).to_le_bytes())
        .collect::<Vec<_>>();
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        let tile_count = if loads == seismic_realization::LoadStrategy::Materialize {
            2
        } else {
            1
        };
        for piece in [1, 16, 32, 64, 256] {
            let emitted = seismic_metal::msl::emit_with(
                &lowered,
                seismic_metal::execution::Config {
                    loads,
                    tile_piece: Some(piece),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                emitted.launches[0].threadgroups,
                ((2 * 256 / piece + 3) / 4) as u64
            );
            let mut kernel = device
                .compile(
                    &lowered,
                    Candidate::Metal(seismic_metal::execution::Config {
                        loads,
                        tile_piece: Some(piece),
                        ..Default::default()
                    }),
                )
                .unwrap();
            let facts = kernel.metal_pipeline_facts().unwrap();
            assert_eq!(facts.len(), 1);
            assert_eq!(facts[0].execution_width, 32);
            assert!(facts[0].max_threads_per_group >= emitted.launches[0].threads_per_threadgroup);
            assert_eq!(emitted.launches[0].declared_threadgroup_bytes, 0);
            assert_eq!(emitted.launches[0].tiles.len(), tile_count);
            let dispatch = emitted.launches[0].dispatch.as_ref().unwrap();
            let bytes_per_lane = emitted.launches[0]
                .tiles
                .iter()
                .map(|t| t.bytes(dispatch).unwrap().0)
                .sum::<u64>();
            assert_eq!(
                bytes_per_lane,
                if piece <= 32 {
                    tile_count as u64 * piece as u64 * 4
                } else {
                    tile_count as u64 * (piece as u64).div_ceil(32) * 4
                }
            );
            let input = device.buffer_from(&bytes).unwrap();
            let output = device.buffer_from(&vec![0xa5; 2056]).unwrap();
            let view = output.view(4..2052).unwrap();
            kernel.execute(&[input.clone(), view.clone()], &[]).unwrap();
            let mut actual = vec![0; 2056];
            output.read(&mut actual).unwrap();
            assert_eq!(&actual[4..2052], expected);
            assert_eq!(&actual[..4], &[0xa5; 4]);
            assert_eq!(&actual[2052..], &[0xa5; 4]);
            // Exact in-place mapping is safe; a shifted overlapping mapping is not.
            kernel
                .execute(&[input.clone(), input.clone()], &[])
                .unwrap();
            input.read(&mut actual[..2048]).unwrap();
            assert_eq!(&actual[..2048], expected);
            output.write(&vec![0xa5; 2056]).unwrap();
            assert!(kernel
                .execute(&[output.view(0..2048).unwrap(), view], &[])
                .unwrap_err()
                .contains("overlapping"));
            output.read(&mut actual).unwrap();
            assert_eq!(actual, vec![0xa5; 2056]);
        }
    }
    assert!(seismic_metal::msl::emit_with(
        &lowered,
        seismic_metal::execution::Config {
            tile_piece: Some(3),
            ..Default::default()
        }
    )
    .is_err());
    for expression in ["a[255-i]", "f32(i)"] {
        let text = source.replace("a[i] * 2.0 + 1.0", expression);
        let p = compile(
            &[SourceFile {
                path: "reject.seismic.portable".into(),
                scope: Scope::Portable,
                text,
            }],
            &[],
        )
        .unwrap();
        let l = lower(&p, "pointwise", "metal", &HashMap::new()).unwrap();
        assert!(seismic_lang::partition::pointwise(&l, 32).is_err());
    }
}
