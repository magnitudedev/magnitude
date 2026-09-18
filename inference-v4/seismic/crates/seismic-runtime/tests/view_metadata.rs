//! Logical shape queries evaluate view metadata without requiring its data.
use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{Candidate, Device};

fn encode(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|n| n.to_le_bytes()).collect()
}

fn extents(device: &Device, candidate: &Candidate) {
    for snapshot in [false, true] {
        let preparation = if snapshot {
            "  snapshot = load(x[bounds[0]:bounds[1]])\n  reset = tile[2] i32\n  for i in owned(reset): reset[i] = 0\n  store(reset,bounds)\n"
        } else {
            ""
        };
        let view = if snapshot {
            "snapshot"
        } else {
            "x[bounds[0]:bounds[1]]"
        };
        let program = compile(&[SourceFile {
            path: "view_metadata.seismic.portable".into(),
            scope: Scope::Portable,
            text: format!("fn evaluate(x:tensor[8] i32,bounds:tensor[2] i32,out:tensor[1] i32):\n{preparation}  result = tile[1] i32\n  for i in owned(result): result[i] = extent({view},0) + 1\n  store(result,out)\n"),
        }], &[]).unwrap();
        let function = lower(&program, "evaluate", device.backend(), &Default::default()).unwrap();
        let mut kernel = device.compile(&function, candidate.clone()).unwrap();
        let input = device.buffer_from(&encode(&[1; 8])).unwrap();
        let bounds = device.buffer(8).unwrap();
        let output = device.buffer(4).unwrap();
        for (start, end) in [
            (2, 7),
            (-5, 3),
            (3, 99),
            (7, 2),
            (0, 0),
            (i32::MIN, i32::MAX),
        ] {
            bounds.write(&encode(&[start, end])).unwrap();
            kernel
                .execute(&[input.clone(), bounds.clone(), output.clone()], &[])
                .unwrap();
            let mut bytes = [0; 4];
            output.read(&mut bytes).unwrap();
            let end = end.clamp(0, 8);
            let expected = end - start.clamp(0, end) + 1;
            assert_eq!(
                i32::from_le_bytes(bytes),
                expected,
                "snapshot={snapshot}, start={start}, end={end}"
            );
        }
    }
}

fn point_guards(device: &Device, candidate: &Candidate) {
    for query in [
        "extent(x[index[0],:],0)",
        "extent(x[0,:extent(x[index[0],:],0)],0)",
        "extent(x[0,:extent(x[index[0],:],0)+0],0)",
        "extent(x[0,:],extent(x[index[0],:],0)-4)",
    ] {
        let program = compile(&[SourceFile {
        path: "view_metadata_guard.seismic.portable".into(),
        scope: Scope::Portable,
        text: format!("fn evaluate(x:tensor[2,4] i32,index:tensor[1] i32,out:tensor[1] i32):\n  result = tile[1] i32\n  for i in owned(result): result[i] = {query}\n  store(result,out)\n"),
    }], &[]).unwrap();
        let function = lower(&program, "evaluate", device.backend(), &Default::default()).unwrap();
        let mut kernel = device.compile(&function, candidate.clone()).unwrap();
        let input = device.buffer_from(&encode(&[1; 8])).unwrap();
        let index = device.buffer(4).unwrap();
        let output = device.buffer(4).unwrap();
        for coordinate in [0, -1, 1, 2, 0] {
            index.write(&encode(&[coordinate])).unwrap();
            let result = kernel.execute(&[input.clone(), index.clone(), output.clone()], &[]);
            assert_eq!(
                result.is_ok(),
                (0..2).contains(&coordinate),
                "{query}, coordinate={coordinate}: {result:?}"
            );
            if result.is_ok() {
                let mut bytes = [0; 4];
                output.read(&mut bytes).unwrap();
                assert_eq!(i32::from_le_bytes(bytes), 4);
            }
        }
    }
}

fn branches(device: &Device, candidate: &Candidate) {
    let program = compile(&[SourceFile {
        path: "view_metadata_branches.seismic.portable".into(),
        scope: Scope::Portable,
        text: "fn evaluate(x:tensor[8] i32,bounds:tensor[2] i32,out:tensor[1] i32):\n  result = tile[1] i32\n  for i in owned(result):\n    if bounds[0] > 0:\n      result[i] = extent(x[bounds[0]:bounds[1]],0)\n    else:\n      result[i] = extent(x[bounds[0]:bounds[1]],0) + 1\n  store(result,out)\n".into(),
    }], &[]).unwrap();
    let function = lower(&program, "evaluate", device.backend(), &Default::default()).unwrap();
    let mut kernel = device.compile(&function, candidate.clone()).unwrap();
    let input = device.buffer_from(&encode(&[1; 8])).unwrap();
    let bounds = device.buffer(8).unwrap();
    let output = device.buffer(4).unwrap();
    for (start, end) in [(2, 7), (-5, 3), (7, 2), (0, 0), (1, 99)] {
        bounds.write(&encode(&[start, end])).unwrap();
        kernel
            .execute(&[input.clone(), bounds.clone(), output.clone()], &[])
            .unwrap();
        let mut bytes = [0; 4];
        output.read(&mut bytes).unwrap();
        let end = end.clamp(0, 8);
        assert_eq!(
            i32::from_le_bytes(bytes),
            end - start.clamp(0, end) + i32::from(start <= 0)
        );
    }
}

fn exercise(device: Device, candidate: Candidate) {
    extents(&device, &candidate);
    point_guards(&device, &candidate);
    branches(&device, &candidate);
}

#[test]
fn cpu_view_metadata_preserves_clamping_snapshots_and_guards() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn metal_view_metadata_preserves_clamping_snapshots_and_guards() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_view_metadata_preserves_clamping_snapshots_and_guards() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::Sequential,
                loads: LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    );
}
