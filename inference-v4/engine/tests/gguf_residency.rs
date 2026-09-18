use seismic_engine::weights::{
    descriptor::{Stored, Transform, WeightDescriptor},
    gguf::Encoding,
    residency::Importer,
    source::FileSource,
};
use seismic_lang::{
    lower::{lower_specialized, Options},
    types::{DType, Elem},
};
use seismic_runtime::{Candidate, Device};
use serde_json::Value;
use std::{collections::HashMap, rc::Rc, sync::Arc};
fn exercise(device: Device, candidate: Candidate) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/gguf-codec-reference.json"
    ))
    .unwrap();
    let program = seismic_std::program().unwrap();
    let device = Rc::new(device);
    let mut importer = Importer::new(device.clone(), candidate.clone()).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let encoding = Encoding::try_from(case["encoding"].as_u64().unwrap() as u32).unwrap();
        let shape = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap())
            .collect::<Vec<_>>();
        let count = shape.iter().product::<u64>() as usize;
        let hex = case["source_hex"].as_str().unwrap();
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect::<Vec<_>>();
        let path = std::env::temp_dir().join(format!(
            "seismic-gguf-codec-{}-{}-{}",
            std::process::id(),
            device.backend(),
            encoding as u32
        ));
        std::fs::write(&path, &bytes).unwrap();
        let source = Arc::new(FileSource::open(&path).unwrap());
        let descriptor = WeightDescriptor {
            name: "weight".into(),
            shape: shape.clone(),
            transform: Transform::Identity,
        };
        let stored = Stored::GgmlBlocks {
            source: source.clone(),
            offset: 0,
            nbytes: bytes.len() as u64,
            shape: shape.clone(),
            encoding,
        };
        let resident = importer.import(&descriptor, &stored, DType::BF16).unwrap();
        let Elem::Repr(name) = resident.element() else { panic!("expected packed resident"); };
        let representation = seismic_lang::repr::lookup(name).unwrap();
        let payload_bytes = representation.planes().iter().map(|p| p.bytes(count as u64).unwrap()).sum::<u64>();
        if matches!(encoding, Encoding::Q4K | Encoding::Q5K | Encoding::Q6K) {
            assert_eq!(payload_bytes, bytes.len() as u64, "compact {:?} payload", encoding);
            assert!(resident.plane("scale").is_none());
            assert!(resident.plane("bias").is_none());
        }
        // Representation storage is compact; the independent V3 fixture below
        // supplies the decoded values, rather than the former widened planes.
        let mut actual = Vec::new();
        let lowered = lower_specialized(
            &program,
            "import_weight",
            device.backend(),
            &HashMap::from([("N".into(), count as i64)]),
            &HashMap::from([
                ("T".into(), resident.element().clone()),
                ("U".into(), Elem::Dtype(DType::F32)),
            ]),
            &Options::default(),
        )
        .unwrap();
        let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
        let out = device.buffer(count * 4).unwrap();
        let buffers = kernel
            .buffers()
            .iter()
            .map(|slot| {
                if slot.parameter == "source" {
                    resident.plane(&slot.plane).unwrap().clone()
                } else {
                    out.clone()
                }
            })
            .collect::<Vec<_>>();
        kernel.execute(&buffers, &[0.]).unwrap();
        actual.resize(count * 4, 0);
        out.read(&mut actual).unwrap();
        for (i, (b, e)) in actual
            .chunks_exact(4)
            .zip(case["values"].as_array().unwrap())
            .enumerate()
        {
            let a = f32::from_le_bytes(b.try_into().unwrap());
            let e = e.as_f64().unwrap() as f32;
            assert!(
                (a - e).abs() <= 1e-7 + e.abs() * 1e-6,
                "{:?} element {i}: {a} != {e}",
                encoding
            );
        }
        let bad = Stored::GgmlBlocks {
            source,
            offset: 0,
            nbytes: bytes.len() as u64 + 1,
            shape,
            encoding,
        };
        assert!(importer.import(&descriptor, &bad, DType::F32).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
#[test]
fn cpu_gguf_residency() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::Materialize,
        },
    );
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_gguf_residency() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_gguf_residency() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    );
}
