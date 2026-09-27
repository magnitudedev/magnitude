//! The separate-draft entries (`tap_rows`, `feature_rows`,
//! `draft_confidence`) against host models of their contracts. The CPU test
//! runs everywhere; the accelerator test covers every other device the host
//! has.

use magnitude_kernels::{draft_confidence, feature_rows, tap_rows};
use seismic::{BackendName, Device, Element, NativeSpecialization, Tensor};

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn values(count: usize, seed: u32) -> Vec<f32> {
    (0..count)
        .map(|index| {
            ((index as u32).wrapping_mul(2654435761) ^ seed) % 2001
        } as f32
            / 500.0
            - 2.0)
        .collect()
}

/// The entry's declared statics fixed from `extents`, its parameters at
/// their defaults.
fn specialization(
    statics: &[String],
    defaults: impl Fn(&NativeSpecialization) -> NativeSpecialization,
    extents: &[(&str, u64)],
) -> NativeSpecialization {
    defaults(&statics.iter().fold(NativeSpecialization::new(), |fixed, name| {
        let (_, value) = extents
            .iter()
            .find(|(extent, _)| extent == name)
            .unwrap_or_else(|| panic!("static {name} has no extent"));
        fixed.with_static(name.clone(), *value)
    }))
}

fn taps_write_their_column_blocks(device: &Device) {
    let implementation = tap_rows::native_implementation(device)
        .unwrap()
        .expect("tap_rows has a native implementation on every device");
    for (rows, width, taps) in [(1usize, 64usize, 2usize), (5, 96, 3), (17, 128, 5)] {
        let extents = [
            ("M", rows as u64),
            ("D", width as u64),
            ("T", taps as u64),
        ];
        let kernel = tap_rows::native_for_device_with(
            device,
            tap_rows::Elements { A: Element::bf16() },
            &specialization(
                &implementation.statics,
                |fixed| implementation.default_specialization(fixed).unwrap(),
                &extents,
            ),
        )
        .unwrap();
        let sentinel = vec![7.0f32; rows * taps * width];
        let mut buffer = Tensor::from_host(
            device,
            Element::bf16(),
            &[rows as u64, (taps * width) as u64],
            &bf16_bytes(&sentinel),
        )
        .unwrap();
        let mut expected = sentinel.clone();
        // Write every other tap, leaving the rest untouched.
        for tap in (0..taps).step_by(2) {
            let residual_values = values(rows * width, tap as u32 + 1);
            let residual = Tensor::from_host(
                device,
                Element::f32(),
                &[rows as u64, width as u64],
                &f32_bytes(&residual_values),
            )
            .unwrap();
            let index = Tensor::from_host(device, Element::i32(), &[1], &i32_bytes(&[tap as i32]))
                .unwrap();
            kernel
                .call(tap_rows::Args {
                    residual: &residual,
                    index: &index,
                    taps: &mut buffer,
                })
                .unwrap();
            for row in 0..rows {
                for column in 0..width {
                    expected[row * taps * width + tap * width + column] =
                        bf16_round(residual_values[row * width + column]);
                }
            }
        }
        assert_eq!(
            bf16_values(&buffer.read_to_host().unwrap()),
            expected,
            "{:?} tap_rows {rows}x{width}x{taps}",
            device.backend()
        );
    }
}

fn features_gather_the_published_rows(device: &Device) {
    let implementation = feature_rows::native_implementation(device)
        .unwrap()
        .expect("feature_rows has a native implementation on every device");
    for (rows, width, published) in [(1usize, 64usize, vec![0i32]), (9, 96, vec![8, 0, 3])] {
        let outputs = published.len();
        let extents = [
            ("M", rows as u64),
            ("O", outputs as u64),
            ("D", width as u64),
        ];
        let kernel = feature_rows::native_for_device_with(
            device,
            feature_rows::Elements { A: Element::bf16() },
            &specialization(
                &implementation.statics,
                |fixed| implementation.default_specialization(fixed).unwrap(),
                &extents,
            ),
        )
        .unwrap();
        let fused_values = values(rows * width, 11);
        let fused = Tensor::from_host(
            device,
            Element::f32(),
            &[rows as u64, width as u64],
            &f32_bytes(&fused_values),
        )
        .unwrap();
        let out_rows = Tensor::from_host(
            device,
            Element::i32(),
            &[outputs as u64],
            &i32_bytes(&published),
        )
        .unwrap();
        let features = kernel
            .call(feature_rows::Args {
                fused: &fused,
                out_rows: &out_rows,
            })
            .unwrap()
            .value;
        let expected = published
            .iter()
            .flat_map(|&row| {
                fused_values[row as usize * width..][..width]
                    .iter()
                    .map(|value| bf16_round(*value))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            bf16_values(&features.read_to_host().unwrap()),
            expected,
            "{:?} feature_rows {rows}x{width}",
            device.backend()
        );
    }
}

fn confidence_declines_below_the_threshold(device: &Device) {
    let implementation = draft_confidence::native_implementation(device)
        .unwrap()
        .expect("draft_confidence has a native implementation on every device");
    let (slots, width, rank) = (6usize, 128usize, 32usize);
    let extents = [
        ("S", slots as u64),
        ("D", width as u64),
        ("R", rank as u64),
    ];
    let kernel = draft_confidence::native_for_device_with(
        device,
        draft_confidence::Elements { A: Element::bf16() },
        &specialization(
            &implementation.statics,
            |fixed| implementation.default_specialization(fixed).unwrap(),
            &extents,
        ),
    )
    .unwrap();
    let feature_values = values(slots * width, 3)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    let memory_values = values(slots * rank, 5)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    let weight_values = values(width + rank, 7)
        .into_iter()
        .map(|value| value / 16.0)
        .collect::<Vec<_>>();
    let bias_value = 0.25f32;
    let confidence = |slot: usize| {
        let score = feature_values[slot * width..][..width]
            .iter()
            .zip(&weight_values[..width])
            .map(|(x, w)| x * w)
            .sum::<f32>()
            + memory_values[slot * rank..][..rank]
                .iter()
                .zip(&weight_values[width..])
                .map(|(x, w)| x * w)
                .sum::<f32>()
            + bias_value;
        1.0 / (1.0 + (-score).exp())
    };
    let confidences = (0..slots).map(confidence).collect::<Vec<_>>();
    let mut sorted = confidences.clone();
    sorted.sort_by(f32::total_cmp);
    // A threshold midway between two confidences, far from either.
    let threshold = (sorted[slots / 2 - 1] + sorted[slots / 2]) / 2.0;
    assert!(sorted[slots / 2] - sorted[slots / 2 - 1] > 1.0e-3);
    let features = Tensor::from_host(
        device,
        Element::bf16(),
        &[slots as u64, width as u64],
        &bf16_bytes(&feature_values),
    )
    .unwrap();
    let memory = Tensor::from_host(
        device,
        Element::bf16(),
        &[slots as u64, rank as u64],
        &bf16_bytes(&memory_values),
    )
    .unwrap();
    let weight = Tensor::from_host(
        device,
        Element::f32(),
        &[(width + rank) as u64],
        &f32_bytes(&weight_values),
    )
    .unwrap();
    let bias = Tensor::from_host(device, Element::f32(), &[1], &f32_bytes(&[bias_value])).unwrap();
    // Every slot proposes token 10 + slot; slot 0 already failed selection.
    let initial = (0..slots)
        .flat_map(|slot| [10 + slot as i32, i32::from(slot == 0)])
        .collect::<Vec<_>>();
    let mut selection = Tensor::from_host(
        device,
        Element::i32(),
        &[slots as u64, 2],
        &i32_bytes(&initial),
    )
    .unwrap();
    kernel
        .call(draft_confidence::Args {
            features: &features,
            memory: &memory,
            weight: &weight,
            bias: &bias,
            threshold,
            selection: &mut selection,
        })
        .unwrap();
    let actual = selection
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    let expected = (0..slots)
        .flat_map(|slot| {
            let status = if confidences[slot] < threshold {
                3
            } else {
                initial[slot * 2 + 1]
            };
            [10 + slot as i32, status]
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "{:?} draft_confidence", device.backend());
}

fn entries_match_on(device: &Device) {
    taps_write_their_column_blocks(device);
    features_gather_the_published_rows(device);
    confidence_declines_below_the_threshold(device);
}

#[test]
fn dflash_entries_match_the_host_model_on_cpu() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    entries_match_on(&catalog.open_backend(BackendName::Cpu).unwrap());
}

#[test]
fn dflash_entries_match_the_host_model_on_accelerators() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    for backend in [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        entries_match_on(&device);
    }
}
