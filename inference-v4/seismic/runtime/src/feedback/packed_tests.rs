use super::*;
use seismic_compiler::feedback::FeedbackOptions;
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;

fn packed_reads(backend: registry::BackendName) {
    let catalog = crate::api::catalog::Catalog::discover().unwrap();
    let info = catalog
        .devices()
        .iter()
        .find(|d| d.backend == backend)
        .unwrap();
    let device = catalog.open(info.id).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "packed-field-read.seismic".into(),
        text: "fn probe[N](x: &tensor[N] Q) -> tensor[N] f32:\n    let mut result = tensor[N] f32\n    parallel for i in 0..N:\n        result[i] = f32(x[i])\n    return result\n".into(),
    }])).unwrap();
    for name in ["q4g64", "q4k", "q5k", "q6k", "iq4g32", "nvfp4_e2m1_block16"] {
        let representation = registry::representation(name).unwrap();
        let registry::RepresentationKind::Packed(layout) =
            &registry::representation_info(representation).kind
        else {
            unreachable!()
        };
        let bindings = ElementBindings::new().bind("Q", representation);
        let entry_id = module.entry_named("probe").unwrap();
        let entry = module.entry(entry_id, &bindings).unwrap();
        let mut bytes = vec![0; layout.packet_size as usize];
        for (ordinal, plane) in layout.planes.iter().enumerate() {
            let storage =
                &mut bytes[plane.offset as usize..(plane.offset + plane.bytes_per_group) as usize];
            match plane.encoding {
                registry::PlaneEncoding::Dense(dtype) => {
                    for (i, slot) in storage.chunks_exact_mut(dtype.bytes() as usize).enumerate() {
                        let scalar = seismic_lang::reference_math::float_literal(
                            dtype,
                            if (i + ordinal) % 2 == 0 {
                                1.0078125
                            } else {
                                -0.03125
                            },
                        );
                        slot.copy_from_slice(
                            &scalar.bits().to_le_bytes()[..dtype.bytes() as usize],
                        );
                    }
                }
                _ => {
                    // Different adjacent bytes force five/six-bit fields to use
                    // both sides of byte and word boundaries.
                    for (i, byte) in storage.iter_mut().enumerate() {
                        *byte = (i as u8).wrapping_mul(73).wrapping_add(0x95);
                    }
                }
            }
        }
        let mut interpreter = Interpreter::new(&entry);
        let tensor = interpreter.add_tensor(
            TensorData::encoded(representation, vec![layout.group as usize], bytes.clone())
                .unwrap(),
        );
        let outcome = interpreter.run(&[Arg::Tensor(tensor)]).unwrap();
        let reference = outcome.results().next().unwrap();
        let seismic_lang::interp::OutcomeValue::Tensor(reference) = reference.value() else {
            panic!()
        };
        let expected = reference.canonical_bytes().unwrap().unwrap();
        let prepared = crate::api::kernel::prepare(
            &module,
            entry_id,
            bindings,
            &device,
            PreparationOptions::feedback(
                PrecisionPolicy::Exact,
                FeedbackOptions {
                    search_time: Duration::ZERO,
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        let input = Arc::new(
            TensorInner::from_host(&device, representation, &[u64::from(layout.group)], &bytes)
                .unwrap(),
        );
        let mut args = EncodedArgs::new();
        args.push_tensor(input.clone());
        let mut actual = crate::api::kernel::call(&Arc::new(prepared), args).unwrap();
        assert_eq!(
            actual.take_tensor().read_to_host().unwrap(),
            expected,
            "{backend:?}/{name} packed fields and arithmetic"
        );
        assert_eq!(input.read_to_host().unwrap(), bytes);
    }
}

#[test]
fn cpu_packed_fields_use_the_shared_scalar_recipe() {
    packed_reads(registry::BackendName::Cpu);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_packed_fields_use_the_shared_scalar_recipe() {
    packed_reads(registry::BackendName::Metal);
}
