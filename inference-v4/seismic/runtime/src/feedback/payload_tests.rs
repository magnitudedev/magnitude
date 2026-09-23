use super::*;
use crate::api::kernel::EncodedScalar;
use seismic_compiler::feedback::FeedbackOptions;
use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;

fn scalar_argument(dtype: DType, bits: u32) -> EncodedScalar {
    match dtype {
        DType::F16 => EncodedScalar::F16(bits as u16),
        DType::BF16 => EncodedScalar::BF16(bits as u16),
        DType::F32 => EncodedScalar::F32Bits(bits),
        _ => unreachable!(),
    }
}

fn result_bits(result: ArgumentValue) -> u32 {
    match result {
        ArgumentValue::F16(bits) | ArgumentValue::BF16(bits) => u32::from(bits),
        ArgumentValue::F32(value) => value.to_bits(),
        other => panic!("unexpected payload result: {other:?}"),
    }
}

fn payload_transport(backend: registry::BackendName) {
    let catalog = crate::api::catalog::Catalog::discover().unwrap();
    let info = catalog
        .devices()
        .iter()
        .find(|device| device.backend == backend)
        .unwrap();
    let device = catalog.open(info.id).unwrap();
    for (dtype, words) in [
        (
            DType::F16,
            [0x7c01u32, 0xfc35, 0x7e01, 0xfeab, 0, 0x8000, 1, 0x8001],
        ),
        (
            DType::BF16,
            [0x7f81, 0xffa3, 0x7fc1, 0xffe3, 0, 0x8000, 1, 0x8001],
        ),
        (
            DType::F32,
            [
                0x7f80_0001,
                0xff80_0035,
                0x7fc0_0001,
                0xffe0_00ab,
                0,
                0x8000_0000,
                1,
                0x8000_0001,
            ],
        ),
    ] {
        let text = format!(
            r#"fn identity(value: {dtype}) -> {dtype}:
    return value

fn scalar(value: {dtype}, alternate: {dtype}, choose: bool, times: range[3]) -> {dtype}:
    let mut result = identity(value)
    if choose:
        result = identity(alternate)
    for iteration in times:
        result = identity(result)
    return result

fn tensor(x: &tensor[8] {dtype}, y: &tensor[8] {dtype}, choose: bool, times: range[3]) -> (tensor[8] {dtype}, {dtype}):
    let mut result = zeros_like(x)
    parallel for i in 0..8:
        let mut value = identity(x[i])
        if choose:
            value = identity(y[i])
        for iteration in times:
            value = identity(value)
        result[i] = value
    return (result, identity(result[0]))
"#
        );
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "payload-transport.seismic".into(),
            text,
        }]))
        .unwrap();
        let prepare = |name| {
            Arc::new(
                crate::api::kernel::prepare(
                    &module,
                    module.entry_named(name).unwrap(),
                    ElementBindings::default(),
                    &device,
                    PreparationOptions::feedback(
                        PrecisionPolicy::Exact,
                        FeedbackOptions {
                            search_time: Duration::ZERO,
                            ..Default::default()
                        },
                    ),
                )
                .unwrap(),
            )
        };
        let scalar = prepare("scalar");
        let tensor = prepare("tensor");
        let bytes = |values: &[u32]| {
            values
                .iter()
                .flat_map(|bits| bits.to_le_bytes()[..dtype.bytes() as usize].to_vec())
                .collect::<Vec<_>>()
        };
        let alternate = words.iter().copied().rev().collect::<Vec<_>>();
        let input_bytes = bytes(&words);
        let alternate_bytes = bytes(&alternate);
        let x = Arc::new(
            TensorInner::from_host(&device, registry::dense(dtype), &[8], &input_bytes).unwrap(),
        );
        let y = Arc::new(
            TensorInner::from_host(&device, registry::dense(dtype), &[8], &alternate_bytes)
                .unwrap(),
        );
        for (choose, repetitions) in [(false, 0), (true, 0), (false, 2), (true, 2)] {
            for (index, &word) in words.iter().enumerate() {
                let mut args = EncodedArgs::new();
                args.push_scalar(scalar_argument(dtype, word));
                args.push_scalar(scalar_argument(dtype, alternate[index]));
                args.push_scalar(EncodedScalar::Bool(choose));
                args.push_scalar(EncodedScalar::Range {
                    start: 0_u64.into(),
                    end: u64::try_from(repetitions).unwrap().into(),
                });
                let actual = crate::api::kernel::call(&scalar, args)
                    .unwrap()
                    .take_scalar();
                assert_eq!(
                    result_bits(actual),
                    if choose { alternate[index] } else { word },
                    "{backend:?}/{dtype} scalar branch/carry payload"
                );
            }
            let mut args = EncodedArgs::new();
            args.push_tensor(x.clone());
            args.push_tensor(y.clone());
            args.push_scalar(EncodedScalar::Bool(choose));
            args.push_scalar(EncodedScalar::Range {
                start: 0_u64.into(),
                end: u64::try_from(repetitions).unwrap().into(),
            });
            let mut result = crate::api::kernel::call(&tensor, args).unwrap();
            assert_eq!(
                result.take_tensor().read_to_host().unwrap(),
                if choose {
                    &alternate_bytes
                } else {
                    &input_bytes
                }
                .as_slice(),
                "{backend:?}/{dtype} load/join/store payload"
            );
            assert_eq!(
                result_bits(result.take_scalar()),
                if choose { alternate[0] } else { words[0] },
                "{backend:?}/{dtype} loaded scalar result"
            );
            assert_eq!(x.read_to_host().unwrap(), input_bytes);
            assert_eq!(y.read_to_host().unwrap(), alternate_bytes);
        }
    }
}

#[test]
fn cpu_payloads_survive_load_helper_join_carry_store_and_result() {
    payload_transport(registry::BackendName::Cpu);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_payloads_survive_load_helper_join_carry_store_and_result() {
    payload_transport(registry::BackendName::Metal);
}

fn arithmetic_payloads(backend: registry::BackendName) {
    let catalog = crate::api::catalog::Catalog::discover().unwrap();
    let info = catalog
        .devices()
        .iter()
        .find(|device| device.backend == backend)
        .unwrap();
    let device = catalog.open(info.id).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "arithmetic-payload.seismic".into(),
        text: r#"fn fused(a: bf16, b: bf16, c: bf16) -> bf16:
    return fma(a, b, c)

fn remainder(a: f32, b: f32) -> f32:
    return a % b

fn absolute(a: f16) -> f16:
    return abs(a)
"#
        .into(),
    }]))
    .unwrap();
    for (name, dtype, arguments, expected) in [
        ("fused", DType::BF16, vec![0x3fc0, 0x3f81, 0xb280], 0x3fc1),
        (
            "remainder",
            DType::F32,
            vec![7.0f32.to_bits(), 4.0f32.to_bits()],
            3.0f32.to_bits(),
        ),
        ("absolute", DType::F16, vec![0xfc01], 0x7c01),
    ] {
        let kernel = Arc::new(
            crate::api::kernel::prepare(
                &module,
                module.entry_named(name).unwrap(),
                ElementBindings::default(),
                &device,
                PreparationOptions::feedback(
                    PrecisionPolicy::Exact,
                    FeedbackOptions {
                        search_time: Duration::ZERO,
                        ..Default::default()
                    },
                ),
            )
            .unwrap(),
        );
        let mut args = EncodedArgs::new();
        for bits in arguments {
            args.push_scalar(scalar_argument(dtype, bits));
        }
        let actual = crate::api::kernel::call(&kernel, args)
            .unwrap()
            .take_scalar();
        assert_eq!(
            result_bits(actual),
            expected,
            "{backend:?}/{name} arithmetic recipe payload"
        );
    }
}

#[test]
fn cpu_arithmetic_uses_typed_scalar_recipes() {
    arithmetic_payloads(registry::BackendName::Cpu);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_arithmetic_uses_typed_scalar_recipes() {
    arithmetic_payloads(registry::BackendName::Metal);
}

fn computed_tensor_recipes(backend: registry::BackendName) {
    let catalog = crate::api::catalog::Catalog::discover().unwrap();
    let info = catalog
        .devices()
        .iter()
        .find(|device| device.backend == backend)
        .unwrap();
    let device = catalog.open(info.id).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "computed-tensor-recipes.seismic".into(),
        text: r#"fn fused(a: &tensor[1] bf16, b: &tensor[1] bf16, c: &tensor[1] bf16) -> tensor[1] bf16:
    return fma(a, b, c)

fn maximum(a: &tensor[4] f32, b: &tensor[4] f32) -> tensor[4] f32:
    return max(a, b)
"#.into(),
    }])).unwrap();
    for (name, dtype, arguments, expected) in [
        (
            "fused",
            DType::BF16,
            vec![vec![0x3fc0], vec![0x3f81], vec![0xb280]],
            vec![0x3fc1],
        ),
        (
            "maximum",
            DType::F32,
            vec![
                vec![0x8000_0000, 0, 0xff80_0001, 0x7fc0_0001],
                vec![0, 0x8000_0000, 1.0f32.to_bits(), 0xffc0_0002],
            ],
            vec![0, 0, 1.0f32.to_bits(), 0x7fc0_0000],
        ),
    ] {
        let kernel = Arc::new(
            crate::api::kernel::prepare(
                &module,
                module.entry_named(name).unwrap(),
                ElementBindings::default(),
                &device,
                PreparationOptions::feedback(
                    PrecisionPolicy::Exact,
                    FeedbackOptions {
                        search_time: Duration::ZERO,
                        ..Default::default()
                    },
                ),
            )
            .unwrap(),
        );
        let encode = |words: &[u32]| {
            words
                .iter()
                .flat_map(|bits| bits.to_le_bytes()[..dtype.bytes() as usize].to_vec())
                .collect::<Vec<_>>()
        };
        let mut args = EncodedArgs::new();
        let mut inputs = Vec::new();
        for words in arguments {
            let bytes = encode(&words);
            let input = Arc::new(
                TensorInner::from_host(
                    &device,
                    registry::dense(dtype),
                    &[words.len() as u64],
                    &bytes,
                )
                .unwrap(),
            );
            args.push_tensor(input.clone());
            inputs.push((input, bytes));
        }
        let output = crate::api::kernel::call(&kernel, args)
            .unwrap()
            .take_tensor();
        assert_eq!(
            output.read_to_host().unwrap(),
            encode(&expected),
            "{backend:?}/{name} computed tensor recipe"
        );
        for (input, bytes) in inputs {
            assert_eq!(
                input.read_to_host().unwrap(),
                bytes,
                "{backend:?}/{name} immutable payload"
            );
        }
    }
}

#[test]
fn cpu_computed_tensor_recipes_preserve_fma_rounding_and_maximum_bits() {
    computed_tensor_recipes(registry::BackendName::Cpu);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_computed_tensor_recipes_preserve_fma_rounding_and_maximum_bits() {
    computed_tensor_recipes(registry::BackendName::Metal);
}

fn packed_reference_boundary(backend: registry::BackendName) {
    let catalog = crate::api::catalog::Catalog::discover().unwrap();
    let info = catalog
        .devices()
        .iter()
        .find(|device| device.backend == backend)
        .unwrap();
    let device = catalog.open(info.id).unwrap();
    let module=check_source(SourceSet::new(vec![SourceFile{path:"packed-outcome-boundary.seismic".into(),text:
        "fn probe(a: &tensor[128] bf16, w: &tensor[128] q4g64) -> f32:\n    let first = fma(f32(a[63]), w[63], 0.0)\n    return fma(f32(a[64]), w[64], first)\n".into()}])).unwrap();
    let kernel = Arc::new(
        crate::api::kernel::prepare(
            &module,
            module.entry_named("probe").unwrap(),
            ElementBindings::default(),
            &device,
            PreparationOptions::feedback(
                PrecisionPolicy::Exact,
                FeedbackOptions {
                    search_time: Duration::ZERO,
                    ..Default::default()
                },
            ),
        )
        .unwrap(),
    );
    let mut activation = vec![0; 256];
    activation[126..128].copy_from_slice(&0x3fa0u16.to_le_bytes()); // 1.25
    activation[128..130].copy_from_slice(&0xc000u16.to_le_bytes()); // -2
                                                                    // Two canonical q4g64 packets: code bytes, BF16 scale, BF16 bias.
                                                                    // Values 63 and 64 occupy different code, scale and bias locations.
    let mut weight = vec![0; 72];
    weight[31] = 0x90;
    weight[32..34].copy_from_slice(&0x3fc0u16.to_le_bytes()); // 1.5
    weight[34..36].copy_from_slice(&0xc000u16.to_le_bytes()); // -2
    weight[36] = 0x05;
    weight[68..70].copy_from_slice(&0xbf00u16.to_le_bytes()); // -0.5
    weight[70..72].copy_from_slice(&0x4040u16.to_le_bytes()); // 3
    let a = Arc::new(
        TensorInner::from_host(&device, registry::dense(DType::BF16), &[128], &activation).unwrap(),
    );
    let w = Arc::new(
        TensorInner::from_host(
            &device,
            registry::representation("q4g64").unwrap(),
            &[128],
            &weight,
        )
        .unwrap(),
    );
    let mut arguments = EncodedArgs::new();
    arguments.push_tensor(a.clone());
    arguments.push_tensor(w.clone());
    let actual = crate::api::kernel::call(&kernel, arguments)
        .unwrap()
        .take_scalar();
    // ((9 * 1.5 - 2) * 1.25) + ((5 * -0.5 + 3) * -2)
    assert_eq!(
        result_bits(actual),
        13.375f32.to_bits(),
        "{backend:?} actual packet boundary FMA"
    );
    assert_eq!(a.read_to_host().unwrap(), activation);
    assert_eq!(w.read_to_host().unwrap(), weight);
}

#[test]
fn cpu_packed_reference_crosses_packet_boundary_with_signed_coefficients() {
    packed_reference_boundary(registry::BackendName::Cpu);
}
#[test]
#[ignore = "requires Metal device"]
fn metal_packed_reference_crosses_packet_boundary_with_signed_coefficients() {
    packed_reference_boundary(registry::BackendName::Metal);
}

fn source_cast_payloads(backend: registry::BackendName, store: bool) {
    use seismic_lang::reference_math::{evaluate, scalar_recipe, ReferenceScalar, ScalarOp};
    let catalog = crate::api::catalog::Catalog::discover().unwrap();
    let info = catalog
        .devices()
        .iter()
        .find(|device| device.backend == backend)
        .unwrap();
    let device = catalog.open(info.id).unwrap();
    for (from, to, samples) in [
        (
            DType::F16,
            DType::F32,
            vec![
                0, 0x8000, 1, 0x8001, 0x3ff, 0x400, 0x7bff, 0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7e55,
                0xfe55,
            ],
        ),
        (
            DType::BF16,
            DType::F32,
            vec![
                0, 0x8000, 1, 0x8001, 0x7f, 0x80, 0x7f7f, 0x7f80, 0xff80, 0x7f81, 0xff81, 0x7fc5,
                0xffc5,
            ],
        ),
        (
            DType::F32,
            DType::F16,
            vec![
                0, 0x80000000, 1, 0x80000001, 0x33000000, 0x33000001, 0x33800000, 0x387fc000,
                0x38800000, 0x3f801000, 0x3f803000, 0x477fe000, 0x477ff000, 0x7f800000, 0xff800000,
                0x7f800001, 0xff800001, 0x7fc12345, 0xffc12345,
            ],
        ),
        (
            DType::F32,
            DType::BF16,
            vec![
                0, 0x80000000, 1, 0x80000001, 0x00008000, 0x00008001, 0x00010000, 0x007f8000,
                0x00800000, 0x3f808000, 0x3f818000, 0x7f7f0000, 0x7f7f8000, 0x7f800000, 0xff800000,
                0x7f800001, 0xff800001, 0x7fc12345, 0xffc12345,
            ],
        ),
    ] {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "source-cast-payload.seismic".into(),
            text: if store { format!("fn cast_value(a: {}, dst: &mut tensor[1] {}) -> {}:\n    let value = {}(a)\n    dst[0] = value\n    return value\n", from.name(), to.name(), to.name(), to.name()) } else { format!("fn cast_value(a: {}) -> {}:\n    return {}(a)\n",from.name(),to.name(),to.name()) },
        }])).unwrap();
        let kernel = Arc::new(
            crate::api::kernel::prepare(
                &module,
                module.entry_named("cast_value").unwrap(),
                ElementBindings::default(),
                &device,
                PreparationOptions::feedback(
                    PrecisionPolicy::Exact,
                    FeedbackOptions {
                        search_time: Duration::ZERO,
                        ..Default::default()
                    },
                ),
            )
            .unwrap(),
        );
        let recipe = scalar_recipe(ScalarOp::Cast(to), &[from]);
        let destination = Arc::new(
            TensorInner::from_host(
                &device,
                registry::dense(to),
                &[1],
                &vec![0; to.bytes() as usize],
            )
            .unwrap(),
        );
        for bits in samples {
            let expected = evaluate(&recipe, &[ReferenceScalar::from_bits(from, bits)])
                .unwrap()
                .bits();
            let mut args = EncodedArgs::new();
            args.push_scalar(scalar_argument(from, bits));
            if store {
                args.push_tensor(destination.clone());
            }
            let actual = crate::api::kernel::call(&kernel, args)
                .unwrap()
                .take_scalar();
            assert_eq!(
                result_bits(actual),
                expected,
                "{backend:?} {from:?}->{to:?} input {bits:#010x}"
            );
            if store {
                assert_eq!(
                    destination.read_to_host().unwrap(),
                    expected.to_le_bytes()[..to.bytes() as usize],
                    "{backend:?} dense write {from:?}->{to:?} input {bits:#010x}"
                );
            }
        }
    }
}

#[test]
fn cpu_source_casts_match_recipe_exceptional_and_rounding_bits() {
    source_cast_payloads(registry::BackendName::Cpu, false);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_source_casts_match_recipe_exceptional_and_rounding_bits() {
    source_cast_payloads(registry::BackendName::Metal, false);
}

#[test]
fn cpu_source_cast_dense_writes_match_recipe_bits() {
    source_cast_payloads(registry::BackendName::Cpu, true);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_source_cast_dense_writes_match_recipe_bits() {
    source_cast_payloads(registry::BackendName::Metal, true);
}
