use magnitude_model_kernels::{import_dense, repack_weight};

fn dense_bytes(name: &str, values: &[f32]) -> Vec<u8> {
    match name {
        "f32" => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        "f16" => [0xc280u16, 0x8000, 0x3000, 0x3e00, 0x4cc0]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
        "bf16" => values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    }
}

fn read_bits(bytes: &[u8], bit: usize, width: usize) -> u32 {
    (0..width).fold(0, |value, offset| {
        value | (u32::from((bytes[(bit + offset) / 8] >> ((bit + offset) % 8)) & 1) << offset)
    })
}

fn write_bits(bytes: &mut [u8], bit: usize, width: usize, value: u32) {
    for offset in 0..width {
        let target = bit + offset;
        let mask = 1u8 << (target % 8);
        bytes[target / 8] =
            (bytes[target / 8] & !mask) | (((value >> offset) as u8 & 1) << (target % 8));
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let converted = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let shift = mantissa.leading_zeros() - 21;
            sign | ((127 - 15 - shift + 1) << 23) | ((mantissa << (shift + 1) & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(converted)
}

fn host_repack(source_name: &str, input: &[u8]) -> Vec<u8> {
    match source_name {
        "gguf_q8_0" => [&input[2..34], &input[0..2]].concat(),
        "gguf_q4_k" => {
            let mut output = vec![0; 144];
            for position in 0..256 {
                let source_byte = 16 + (position / 64) * 32 + position % 32;
                let code = (input[source_byte] >> ((position % 64 / 32) * 4)) & 15;
                write_bits(&mut output[0..128], position * 4, 4, u32::from(code));
            }
            for group in 0..8 {
                let index = group % 4;
                let scale_low = input[4 + index];
                let bias_low = input[8 + index];
                let high = input[12 + index];
                let scale = if group < 4 {
                    scale_low & 63
                } else {
                    (high & 15) | ((scale_low >> 6) << 4)
                };
                let bias = if group < 4 {
                    bias_low & 63
                } else {
                    (high >> 4) | ((bias_low >> 6) << 4)
                };
                write_bits(&mut output[128..140], group * 12, 6, u32::from(scale));
                write_bits(&mut output[128..140], group * 12 + 6, 6, u32::from(bias));
            }
            output[140..142].copy_from_slice(&input[0..2]);
            output[142..144].copy_from_slice(&input[2..4]);
            output
        }
        "gguf_q5_k" => {
            let mut output = vec![0; 176];
            for position in 0..256 {
                let low_byte = 48 + (position / 64) * 32 + position % 32;
                let low = (input[low_byte] >> ((position % 64 / 32) * 4)) & 15;
                let high = (input[16 + position % 32] >> (position / 32)) & 1;
                write_bits(
                    &mut output[0..160],
                    position * 5,
                    5,
                    u32::from(low | (high << 4)),
                );
            }
            for group in 0..8 {
                let index = group % 4;
                let scale_low = input[4 + index];
                let bias_low = input[8 + index];
                let high = input[12 + index];
                let scale = if group < 4 {
                    scale_low & 63
                } else {
                    (high & 15) | ((scale_low >> 6) << 4)
                };
                let bias = if group < 4 {
                    bias_low & 63
                } else {
                    (high >> 4) | ((bias_low >> 6) << 4)
                };
                write_bits(&mut output[160..172], group * 12, 6, u32::from(scale));
                write_bits(&mut output[160..172], group * 12 + 6, 6, u32::from(bias));
            }
            output[172..174].copy_from_slice(&input[0..2]);
            output[174..176].copy_from_slice(&input[2..4]);
            output
        }
        "gguf_q6_k" => {
            let mut output = vec![0; 210];
            for position in 0..256 {
                let low_byte = (position / 128) * 64 + position % 64;
                let low = (input[low_byte] >> ((position % 128 / 64) * 4)) & 15;
                let high_byte = 128 + (position / 128) * 32 + position % 32;
                let high = (input[high_byte] >> ((position % 128 / 32) * 2)) & 3;
                write_bits(
                    &mut output[0..192],
                    position * 6,
                    6,
                    u32::from(low | (high << 4)),
                );
            }
            output[192..210].copy_from_slice(&input[192..210]);
            output
        }
        "gguf_iq4_xs" => {
            let mut output = vec![0; 160];
            for position in 0..256 {
                let source_byte = 8 + (position / 32) * 16 + position % 16;
                let code = (input[source_byte] >> ((position % 32 / 16) * 4)) & 15;
                write_bits(&mut output[0..128], position * 4, 4, u32::from(code));
            }
            let base = f16_to_f32(u16::from_le_bytes(input[0..2].try_into().unwrap()));
            for group in 0..8 {
                let low = read_bits(input, (4 + group / 2) * 8 + (group % 2) * 4, 4);
                let high = read_bits(input, 16 + 2 * group, 2);
                let scale = base * ((low | (high << 4)) as i32 - 32) as f32;
                output[128 + group * 4..132 + group * 4].copy_from_slice(&scale.to_le_bytes());
            }
            output
        }
        _ => unreachable!(),
    }
}

#[test]
fn k_quant_repack_preserves_independent_gguf_affine_values() {
    // These coefficient bytes exercise both GGUF's low six-bit groups and
    // its split high-bit groups. A raw 12-byte copy gives different values.
    let expected_scales = [1u32, 2, 3, 4, 48, 34, 20, 6];
    let expected_biases = [5u32, 6, 7, 8, 33, 51, 5, 23];
    for (name, size, code, coefficient_offset) in [
        ("gguf_q4_k", 144, 15.0f32, 128),
        ("gguf_q5_k", 176, 31.0f32, 160),
    ] {
        let mut input = vec![0xffu8; size];
        input[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // scale = 1
        input[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // minimum = 0.5
        input[4..8].copy_from_slice(&[0xc1, 0x82, 0x43, 0x04]);
        input[8..12].copy_from_slice(&[0x85, 0xc6, 0x07, 0x48]);
        input[12..16].copy_from_slice(&[0x10, 0x32, 0x54, 0x76]);
        let resident = host_repack(name, &input);
        for group in 0..8 {
            let scale = read_bits(&resident[coefficient_offset..], group * 12, 6);
            let bias = read_bits(&resident[coefficient_offset..], group * 12 + 6, 6);
            assert_eq!(scale, expected_scales[group], "{name} scale group {group}");
            assert_eq!(bias, expected_biases[group], "{name} minimum group {group}");
            let value = code * scale as f32 - 0.5 * bias as f32;
            let expected =
                code * expected_scales[group] as f32 - 0.5 * expected_biases[group] as f32;
            assert_eq!(value, expected, "{name} value group {group}");
        }
    }
}

#[test]
fn generated_surface_is_one_dense_import_and_one_exact_repack() {
    let _ = import_dense::for_device_with;
    let _ = import_dense::native_for_device_with;
    let _ = repack_weight::for_device_with;
    let _ = repack_weight::native_for_device_with;
    let source = include_str!("../kernels/import.seismic");
    assert!(!source.contains("copy_to_f32"));
    assert!(!source.contains("NegativeExp"));
}

#[cfg(target_os = "macos")]
#[test]
fn native_dense_import_matches_host_for_all_nine_pairs() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let values = [-3.25f32, -0.0, 0.125, 1.5, 19.0];
    for source_name in ["f32", "f16", "bf16"] {
        for destination_name in ["f32", "f16", "bf16"] {
            let source_element = seismic::Element::named(source_name).unwrap();
            let destination_element = seismic::Element::named(destination_name).unwrap();
            let source = seismic::Tensor::from_host(
                &device,
                source_element,
                &[values.len() as u64],
                &dense_bytes(source_name, &values),
            )
            .unwrap();
            let elements = import_dense::Elements {
                E: source_element,
                U: destination_element,
            };
            let native = import_dense::native_for_device_with(
                &device,
                elements,
                &seismic::NativeSpecialization::new(),
            )
            .unwrap()
            .call(import_dense::Args { source: &source })
            .unwrap()
            .value;
            assert_eq!(
                native.read_to_host().unwrap(),
                dense_bytes(destination_name, &values),
                "{source_name}->{destination_name}"
            );
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn native_repack_matches_host_recipe_for_all_five_pairs() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let pairs = [
        ("gguf_q8_0", "q8g32s", 32usize, 34usize),
        ("gguf_q4_k", "q4k", 256, 144),
        ("gguf_q5_k", "q5k", 256, 176),
        ("gguf_q6_k", "q6k", 256, 210),
        ("gguf_iq4_xs", "iq4g32", 256, 136),
    ];
    for (source_name, destination_name, logical, packet_bytes) in pairs {
        let source_element = seismic::Element::named(source_name).unwrap();
        let destination_element = seismic::Element::named(destination_name).unwrap();
        let bytes = (0..packet_bytes)
            .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
            .collect::<Vec<_>>();
        let source =
            seismic::Tensor::from_host(&device, source_element, &[logical as u64], &bytes).unwrap();
        let native = repack_weight::native_for_device_with(
            &device,
            repack_weight::Elements {
                E: source_element,
                U: destination_element,
            },
            &seismic::NativeSpecialization::new(),
        )
        .unwrap()
        .call(repack_weight::Args { source: &source })
        .unwrap()
        .value;
        let actual = native.read_to_host().unwrap();
        let mut expected = host_repack(source_name, &bytes);
        expected.resize(actual.len(), 0);
        assert_eq!(actual, expected, "{source_name}->{destination_name}");
    }
}
