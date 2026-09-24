use magnitude_model_kernels::head_logits_rows;

#[test]
fn generated_head_surface_exposes_checked_and_native_routes() {
    let _ = head_logits_rows::for_device_with;
    let _ = head_logits_rows::native_for_device_with;
}

#[cfg(target_os = "macos")]
#[test]
fn native_head_logits_matches_host_with_bf16_features_and_q8_weights() {
    use magnitude_model_kernels::repack_weight;

    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let external = seismic::Element::named("gguf_q8_0").unwrap();
    let resident = seismic::Element::named("q8g32s").unwrap();
    let mut bytes = Vec::new();
    for group in 0..2 {
        bytes.extend_from_slice(&seismic_lang::registry::f16_bits(0.5).to_le_bytes());
        bytes.extend((0..32).map(|lane| (group * 32 + lane + 1) as u8));
    }
    let source = seismic::Tensor::from_host(&device, external, &[64], &bytes).unwrap();
    let weight = repack_weight::native_for_device_with(
        &device,
        repack_weight::Elements {
            E: external,
            U: resident,
        },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(repack_weight::Args { source: &source })
    .unwrap()
    .value
    .reshape(&[2, 32])
    .unwrap();
    let feature_values = (0..64)
        .map(|index| (index as f32 - 16.0) / 8.0)
        .collect::<Vec<_>>();
    let feature_bytes = feature_values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    let features =
        seismic::Tensor::from_host(&device, seismic::Element::bf16(), &[2, 32], &feature_bytes)
            .unwrap();
    let actual = head_logits_rows::native_for_device_with(
        &device,
        head_logits_rows::Elements {
            A: seismic::Element::bf16(),
            OW: resident,
        },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(head_logits_rows::Args {
        features: &features,
        weight: &weight,
    })
    .unwrap()
    .value
    .read_to_host()
    .unwrap()
    .chunks_exact(4)
    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
    .collect::<Vec<_>>();
    let rounded = feature_values
        .iter()
        .map(|value| f32::from_bits(value.to_bits() & 0xffff_0000))
        .collect::<Vec<_>>();
    let expected = (0..2)
        .flat_map(|row| {
            let rounded = &rounded;
            (0..2).map(move |output| {
                (0..32)
                    .map(|column| {
                        rounded[row * 32 + column] * (output * 32 + column + 1) as f32 * 0.5
                    })
                    .sum::<f32>()
            })
        })
        .collect::<Vec<_>>();
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= expected.abs() * 1e-5 + 1e-4,
            "logit {index}: {actual} != {expected}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn native_head_logits_decodes_q4k_and_q5k_affine_packets() {
    fn write_bits(bytes: &mut [u8], bit: usize, width: usize, value: u32) {
        for offset in 0..width {
            let target = bit + offset;
            let mask = 1u8 << (target % 8);
            bytes[target / 8] =
                (bytes[target / 8] & !mask) | ((((value >> offset) & 1) as u8) << (target % 8));
        }
    }

    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let feature_values = (0..512)
        .map(|index| ((index % 13) as f32 - 6.0) / 8.0)
        .collect::<Vec<_>>();
    let feature_bytes = feature_values
        .iter()
        .flat_map(|value| seismic_lang::registry::f16_bits(*value).to_le_bytes())
        .collect::<Vec<_>>();
    let features =
        seismic::Tensor::from_host(&device, seismic::Element::f16(), &[2, 256], &feature_bytes)
            .unwrap();

    for (name, code_bits, packet_bytes, coefficient_offset) in
        [("q4k", 4, 144, 128), ("q5k", 5, 176, 160)]
    {
        let mut packets = vec![0u8; 2 * packet_bytes];
        for output in 0..2 {
            let packet = &mut packets[output * packet_bytes..(output + 1) * packet_bytes];
            for position in 0..256 {
                let code = ((position * 7 + output * 3) % (1 << code_bits)) as u32;
                write_bits(packet, position * code_bits, code_bits, code);
            }
            for group in 0..8 {
                write_bits(
                    &mut packet[coefficient_offset..coefficient_offset + 12],
                    group * 12,
                    6,
                    (group + output + 1) as u32,
                );
                write_bits(
                    &mut packet[coefficient_offset..coefficient_offset + 12],
                    group * 12 + 6,
                    6,
                    (group + 2) as u32,
                );
            }
            packet[packet_bytes - 4..packet_bytes - 2]
                .copy_from_slice(&seismic_lang::registry::f16_bits(0.5).to_le_bytes());
            packet[packet_bytes - 2..packet_bytes]
                .copy_from_slice(&seismic_lang::registry::f16_bits(0.25).to_le_bytes());
        }
        let resident = seismic::Element::named(name).unwrap();
        let weight = seismic::Tensor::from_host(&device, resident, &[2, 256], &packets).unwrap();
        let actual = head_logits_rows::native_for_device_with(
            &device,
            head_logits_rows::Elements {
                A: seismic::Element::f16(),
                OW: resident,
            },
            &seismic::NativeSpecialization::new(),
        )
        .unwrap()
        .call(head_logits_rows::Args {
            features: &features,
            weight: &weight,
        })
        .unwrap()
        .value
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
        for row in 0..2 {
            for output in 0..2 {
                let expected = (0..256)
                    .map(|column| {
                        let group = column / 32;
                        let code = ((column * 7 + output * 3) % (1 << code_bits)) as f32;
                        let weight =
                            0.5 * (group + output + 1) as f32 * code - 0.25 * (group + 2) as f32;
                        feature_values[row * 256 + column] * weight
                    })
                    .sum::<f32>();
                let value = actual[row * 2 + output];
                assert!(
                    (value - expected).abs() <= expected.abs() * 1e-5 + 1e-3,
                    "{name} row {row} output {output}: {value} != {expected}"
                );
            }
        }
    }
}
