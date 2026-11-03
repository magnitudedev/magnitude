use magnitude_kernels::readout_scores_rows;
use seismic::{BackendName, Element, NativeSpecialization, Tensor};

/// CUDA, else Metal; `SEISMIC_TEST_BACKEND=vulkan` selects Vulkan (a CUDA
/// host also has a Vulkan device). Then the CPU.
fn devices() -> Vec<seismic::Device> {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    let gpu = match std::env::var("SEISMIC_TEST_BACKEND").ok().as_deref() {
        Some("vulkan") => Some(catalog.open_backend(BackendName::Vulkan).expect("the Vulkan device opens")),
        Some(other) => panic!("SEISMIC_TEST_BACKEND={other}: only vulkan is selectable"),
        None => [BackendName::Cuda, BackendName::Metal]
            .into_iter()
            .find_map(|backend| catalog.open_backend(backend).ok()),
    };
    gpu.into_iter()
        .chain(std::iter::once(catalog.open_backend(BackendName::Cpu).unwrap()))
        .collect()
}

/// BF16 values spread over a few binades, and their bits.
fn rows(count: usize, width: usize, seed: u32) -> (Vec<f32>, Vec<u8>) {
    let values = (0..count * width)
        .map(|index| {
            let mixed = (index as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            let value = (mixed % 2001) as f32 / 250.0 - 4.0;
            f32::from_bits(value.to_bits() & 0xffff_0000)
        })
        .collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    (values, bytes)
}

/// Each score is the F32 sum of its query and key rows' products, scaled
/// once: the CPU sums them in column order, as the reference does, exactly;
/// the GPUs in another order, within F32 reassociation of the sum.
#[test]
fn scores_are_scaled_dot_products_of_feature_rows() {
    let width = 2560;
    let scale = (width as f32).sqrt().recip();
    for device in devices() {
        // The GPUs specialize the width; the CPU reads it at run time.
        let specialization = match device.backend() {
            BackendName::Cpu => NativeSpecialization::new(),
            _ => NativeSpecialization::new().with_static("D", width as u64),
        };
        let kernel = readout_scores_rows::native_for_device_with(
            &device,
            readout_scores_rows::Elements { A: Element::bf16() },
            &specialization,
        )
        .unwrap();
        for (queries, keys) in [(1usize, 1usize), (2, 7), (1, 300)] {
            let (query, query_bytes) = rows(queries, width, 3);
            let (key, key_bytes) = rows(keys, width, 11);
            let tensor = |count: usize, bytes: &[u8]| {
                Tensor::from_host(&device, Element::bf16(), &[count as u64, width as u64], bytes)
                    .unwrap()
            };
            let scores = kernel
                .call(readout_scores_rows::Args {
                    query: &tensor(queries, &query_bytes),
                    keys: &tensor(keys, &key_bytes),
                    scale,
                })
                .unwrap()
                .value
                .read_to_host()
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(scores.len(), queries * keys);
            for (index, score) in scores.iter().enumerate() {
                let (q, k) = (index / keys, index % keys);
                let products = query[q * width..][..width]
                    .iter()
                    .zip(&key[k * width..][..width]);
                let expected = products.clone().fold(0.0_f32, |sum, (a, b)| a.mul_add(*b, sum)) * scale;
                let magnitude = products.map(|(a, b)| (a * b).abs()).sum::<f32>() * scale;
                let bound = match device.backend() {
                    BackendName::Cpu => 0.0,
                    _ => magnitude * width as f32 * f32::EPSILON,
                };
                assert!(
                    (score - expected).abs() <= bound,
                    "{:?} {queries}x{keys}: score {index} is {score}, expected {expected}",
                    device.backend()
                );
            }
        }
    }
}
