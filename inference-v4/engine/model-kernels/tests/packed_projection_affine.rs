use magnitude_model_kernels::{
    qwen_attention_output, qwen_attention_project, qwen_recurrent_output, qwen_recurrent_project,
};

#[cfg(target_os = "macos")]
#[test]
fn native_projection_stages_decode_q4k_and_q5k_affine_packets() {
    use seismic::{BackendName, Device, DeviceCatalog, Element, Tensor};

    fn write_bits(bytes: &mut [u8], bit: usize, width: usize, value: u32) {
        for offset in 0..width {
            let target = bit + offset;
            let mask = 1u8 << (target % 8);
            bytes[target / 8] =
                (bytes[target / 8] & !mask) | ((((value >> offset) & 1) as u8) << (target % 8));
        }
    }

    fn tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Tensor::from_host(device, Element::f32(), shape, &bytes).unwrap()
    }

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor
            .read_to_host()
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn close(actual: &[f32], expected: &[f32], stage: &str) {
        assert_eq!(actual.len(), expected.len(), "{stage} length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= expected.abs() * 2e-4 + 2e-3,
                "{stage}[{index}]: Metal {actual}, host {expected}"
            );
        }
    }

    fn packed_weight(device: &Device, name: &str, outputs: usize) -> (Tensor, Vec<f32>) {
        let (code_bits, packet_bytes, coefficient_offset) = match name {
            "q4k" => (4, 144, 128),
            "q5k" => (5, 176, 160),
            _ => unreachable!(),
        };
        let mut packets = vec![0u8; outputs * packet_bytes];
        let mut decoded = vec![0.0; outputs * 256];
        for output in 0..outputs {
            let packet = &mut packets[output * packet_bytes..(output + 1) * packet_bytes];
            for position in 0..256 {
                let code = ((position * 7 + output * 3) % (1 << code_bits)) as u32;
                write_bits(packet, position * code_bits, code_bits, code);
                let group = position / 32;
                decoded[output * 256 + position] =
                    0.5 * (group + output + 1) as f32 * code as f32 - 0.25 * (group + 2) as f32;
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
        let resident = Element::named(name).unwrap();
        let weight = Tensor::from_host(device, resident, &[outputs as u64, 256], &packets).unwrap();
        (weight, decoded)
    }

    fn dot(input: &[f32], weights: &[f32], outputs: usize) -> Vec<f32> {
        (0..outputs)
            .map(|output| {
                (0..256)
                    .map(|source| input[source] * weights[output * 256 + source])
                    .sum()
            })
            .collect()
    }

    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap();
    let activation_values = (0..256)
        .map(|index| ((index % 13) as f32 - 6.0) / 8.0)
        .collect::<Vec<_>>();
    let activation = tensor(&device, &[1, 256], &activation_values);
    let gated_values = (0..256)
        .map(|index| ((index % 17) as f32 - 8.0) / 16.0)
        .collect::<Vec<_>>();
    let gated = tensor(&device, &[1, 1, 256], &gated_values);
    let recurrent_gated = tensor(&device, &[1, 256], &gated_values);
    let hidden = tensor(&device, &[1, 1], &[0.375]);
    let zeros = tensor(&device, &[1, 1, 256], &[0.0; 256]);
    let value = tensor(&device, &[1, 256], &[0.0; 256]);
    let destinations =
        Tensor::from_host(&device, Element::i32(), &[1], &(-1i32).to_le_bytes()).unwrap();

    for name in ["q4k", "q5k"] {
        let element = Element::named(name).unwrap();
        let (query, query_host) = packed_weight(&device, name, 2);
        let (key, key_host) = packed_weight(&device, name, 1);
        let (value_weight, value_host) = packed_weight(&device, name, 1);
        let attention = qwen_attention_project::native_for_device_with(
            &device,
            qwen_attention_project::Elements {
                QW: element,
                KW: element,
                VW: element,
                A: Element::f32(),
            },
        )
        .unwrap()
        .call(qwen_attention_project::Args {
            normalized: &activation,
            query_norm: &tensor(&device, &[1], &[1.0]),
            query_gate_weight: &query,
            key_weight: &key,
            value_weight: &value_weight,
        })
        .unwrap();
        close(
            &values(&attention.r0),
            &dot(&activation_values, &query_host, 2),
            "attention query",
        );
        close(
            &values(&attention.r1),
            &dot(&activation_values, &key_host, 1),
            "attention key",
        );
        close(
            &values(&attention.r2),
            &dot(&activation_values, &value_host, 1),
            "attention value",
        );

        let (qkv, qkv_host) = packed_weight(&device, name, 3);
        let (gate, gate_host) = packed_weight(&device, name, 1);
        let (alpha, alpha_host) = packed_weight(&device, name, 1);
        let (beta, beta_host) = packed_weight(&device, name, 1);
        let recurrent = qwen_recurrent_project::native_for_device_with(
            &device,
            qwen_recurrent_project::Elements {
                QW: element,
                GW: element,
                AW: element,
                BW: element,
                A: Element::f32(),
            },
        )
        .unwrap()
        .call(qwen_recurrent_project::Args {
            normalized: &activation,
            qkv_weight: &qkv,
            gate_weight: &gate,
            alpha_weight: &alpha,
            beta_weight: &beta,
        })
        .unwrap()
        .value;
        let expected = [
            dot(&activation_values, &qkv_host, 3),
            dot(&activation_values, &gate_host, 1),
            dot(&activation_values, &alpha_host, 1),
            dot(&activation_values, &beta_host, 1),
        ]
        .concat();
        close(&values(&recurrent), &expected, "recurrent projection");

        let (output, output_host) = packed_weight(&device, name, 1);
        let expected_output = [0.375 + dot(&gated_values, &output_host, 1)[0]];
        let mut history_key = tensor(&device, &[1, 1, 256], &[0.0; 256]);
        let mut history_value = tensor(&device, &[1, 1, 256], &[0.0; 256]);
        let attention_output = qwen_attention_output::native_for_device_with(
            &device,
            qwen_attention_output::Elements {
                OW: element,
                A: Element::f32(),
            },
        )
        .unwrap()
        .call(qwen_attention_output::Args {
            hidden: &hidden,
            gated: &gated,
            prepared_key: &zeros,
            value: &value,
            output_weight: &output,
            destinations: &destinations,
            history_key: &mut history_key,
            history_value: &mut history_value,
        })
        .unwrap()
        .value;
        close(
            &values(&attention_output),
            &expected_output,
            "attention output",
        );

        let recurrent_output = qwen_recurrent_output::native_for_device_with(
            &device,
            qwen_recurrent_output::Elements {
                OW: element,
                A: Element::f32(),
            },
        )
        .unwrap()
        .call(qwen_recurrent_output::Args {
            hidden: &hidden,
            gated: &recurrent_gated,
            output_weight: &output,
        })
        .unwrap()
        .value;
        close(
            &values(&recurrent_output),
            &expected_output,
            "recurrent output",
        );
    }
}
