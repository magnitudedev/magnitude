#[path = "support/reference.rs"]
mod reference;
use reference::{Arg, TensorData};
use seismic_engine::weights::{
    descriptor::{Stored, Transform, WeightDescriptor},
    gguf::Encoding,
    residency::{block_import, Importer},
    source::FileSource,
};
use seismic_lang::types::{DType, Elem, ValueType};
use serde_json::Value;
use std::{collections::HashMap, rc::Rc, sync::Arc};
struct Case {
    encoding: Encoding,
    shape: Vec<u64>,
    bytes: Vec<u8>,
    values: Vec<f32>,
}
fn cases() -> Vec<Case> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../validation/results/fixtures/gguf-codec-reference.json"
    ))
    .unwrap();
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let hex = case["source_hex"].as_str().unwrap();
            Case {
                encoding: Encoding::try_from(case["encoding"].as_u64().unwrap() as u32).unwrap(),
                shape: case["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap())
                    .collect(),
                bytes: (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                    .collect(),
                values: case["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect(),
            }
        })
        .collect()
}
fn check(case: &Case, actual: &[f32]) {
    assert_eq!(actual.len(), case.values.len());
    for (i, (a, e)) in actual.iter().zip(&case.values).enumerate() {
        assert!(
            (a - e).abs() <= 1e-7 + e.abs() * 1e-6,
            "{:?} element {i}: {a} != {e}",
            case.encoding
        );
    }
}
/// The import entries and the packed decode, without any device: imported planes are
/// read back as the resident representation and widened by `import_weight`.
#[test]
fn reference_gguf_import() {
    let program = seismic_std::program().unwrap();
    {
        for case in cases() {
            let (entry, representation) = block_import(case.encoding).unwrap();
            let count = case.shape.iter().product::<u64>() as usize;
            let shapes = HashMap::from([(
                "B".to_string(),
                (count as u64 / case.encoding.block_elements()) as i64,
            )]);
            let mut padded = case.bytes.clone();
            padded.resize(padded.len().div_ceil(4) * 4, 0);
            let mut vm = reference::interpreter(&program);
            let mut planes = Vec::new();
            let args = reference::entry(&program, entry)
                .params
                .iter()
                .map(|param| {
                    let ValueType::Tensor(tensor) = &param.ty else {
                        panic!("{entry}.{} is not a tensor", param.name)
                    };
                    let Elem::Dtype(dtype) = tensor.elem else {
                        panic!("{entry}.{} is not dense", param.name)
                    };
                    let shape = reference::extents(tensor, &shapes);
                    let size = shape.iter().product();
                    let mut data = TensorData::dense(dtype, shape, vec![0.; size]);
                    match param.name.as_str() {
                        "data" => data.load_device_bytes(&padded),
                        "halves" => data.load_device_bytes(&case.bytes),
                        _ => planes.push(vm.tensors.len()),
                    }
                    Arg::Tensor(vm.add_tensor(data))
                })
                .collect::<Vec<_>>();
            reference::run(&mut vm, entry, &args, &shapes);
            let planes = planes
                .into_iter()
                .map(|id| vm.tensors[id].device_bytes().remove(0))
                .collect();
            let source = vm.add_tensor(TensorData::Packed {
                repr: seismic_lang::repr::lookup(representation).unwrap(),
                shape: vec![count],
                planes,
            });
            let out = vm.add_tensor(TensorData::dense(DType::F32, vec![count], vec![0.; count]));
            reference::run(
                &mut vm,
                "import_weight",
                &[Arg::Tensor(source), Arg::Tensor(out)],
                &HashMap::from([("N".to_string(), count as i64)]),
            );
            check(&case, &reference::values(&vm.tensors[out]));
        }
    }
}
#[test]
#[ignore = "requires a Metal device"]
fn metal_gguf_residency() {
    use seismic_runtime::{
        plan::{Bindings, PlanCompiler, Settings},
        Buffer, Device,
    };
    struct Decode<'a> {
        source: &'a seismic_engine::weights::residency::ResidentWeight,
        out: &'a Buffer,
    }
    impl Bindings for Decode<'_> {
        fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
            if root == "source" {
                self.source.plane(plane)
            } else {
                Some(self.out)
            }
        }
        fn scalar(&self, _: &str) -> Option<f64> {
            Some(0.)
        }
    }
    let program = seismic_std::program().unwrap();
    let device = Rc::new(Device::metal().unwrap());
    let mut importer = Importer::new(device.clone(), Settings::default()).unwrap();
    let mut compiler = PlanCompiler::new(&device, &program, Settings::default());
    for case in cases() {
        let count = case.shape.iter().product::<u64>() as usize;
        let path = std::env::temp_dir().join(format!(
            "seismic-gguf-codec-{}-{}",
            std::process::id(),
            case.encoding as u32
        ));
        std::fs::write(&path, &case.bytes).unwrap();
        let source = Arc::new(FileSource::open(&path).unwrap());
        let descriptor = WeightDescriptor {
            name: "weight".into(),
            shape: case.shape.clone(),
            transform: Transform::Identity,
        };
        let stored = |nbytes| Stored::GgmlBlocks {
            source: source.clone(),
            offset: 0,
            nbytes,
            shape: case.shape.clone(),
            encoding: case.encoding,
        };
        let resident = importer
            .import(&descriptor, &stored(case.bytes.len() as u64), DType::BF16)
            .unwrap();
        let Elem::Repr(name) = resident.element() else {
            panic!("expected packed resident");
        };
        let representation = seismic_lang::repr::lookup(name).unwrap();
        let payload_bytes = representation
            .planes()
            .iter()
            .map(|p| p.bytes(count as u64).unwrap())
            .sum::<u64>();
        if matches!(case.encoding, Encoding::Q4K | Encoding::Q5K | Encoding::Q6K) {
            assert_eq!(
                payload_bytes,
                case.bytes.len() as u64,
                "compact {:?} payload",
                case.encoding
            );
            assert!(resident.plane("scale").is_none());
            assert!(resident.plane("bias").is_none());
        }
        // Representation storage is compact; the independent V3 fixture supplies the
        // decoded values, widened here through the ordinary selected runtime.
        let out = device.buffer(count * 4).unwrap();
        compiler
            .compile_entry(
                "import_weight",
                &HashMap::from([("N".into(), count as i64)]),
                &HashMap::from([
                    ("T".into(), resident.element().clone()),
                    ("U".into(), Elem::Dtype(DType::F32)),
                ]),
            )
            .unwrap()
            .execute(&Decode {
                source: &resident,
                out: &out,
            })
            .unwrap();
        let mut actual = vec![0; count * 4];
        out.read(&mut actual).unwrap();
        check(
            &case,
            &actual
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(),
        );
        assert!(importer
            .import(
                &descriptor,
                &stored(case.bytes.len() as u64 + 1),
                DType::F32
            )
            .is_err());
        std::fs::remove_file(path).unwrap();
    }
}
