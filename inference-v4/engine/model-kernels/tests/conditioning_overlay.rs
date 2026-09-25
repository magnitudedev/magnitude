use magnitude_model_kernels::conditioning_overlay;

/// The host's GPU (Metal on macOS, else Vulkan when present), then the CPU.
fn devices() -> Vec<seismic::Device> {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let gpu = if cfg!(target_os = "macos") {
        Some(catalog.open_backend(seismic::BackendName::Metal).unwrap())
    } else {
        catalog.open_backend(seismic::BackendName::Vulkan).ok()
    };
    gpu.into_iter()
        .chain(std::iter::once(catalog.open_backend(seismic::BackendName::Cpu).unwrap()))
        .collect()
}

/// The overlay publishes every conditioned value bit for bit (NaN payloads,
/// infinities and signed zero included) over rows that held other values.
#[test]
fn native_overlay_publishes_conditioned_rows_bit_exactly() {
    for device in devices() {
        for (rows, width) in [(1usize, 2usize), (5, 37), (3, 2560)] {
            let special = [0x7fc0_1234u32, 0x7f80_0000, 0xff80_0000, 0x8000_0000, 0x0000_0001];
            let input = (0..rows * width)
                .map(|index| special.get(index).copied().unwrap_or(((index as f32) * 0.37 - 11.0).to_bits()))
                .collect::<Vec<_>>();
            let bytes = |words: &[u32]| words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
            let shape = [rows as u64, width as u64];
            let source = seismic::Tensor::from_host(&device, seismic::Element::f32(), &shape, &bytes(&input)).unwrap();
            let mut out =
                seismic::Tensor::from_host(&device, seismic::Element::f32(), &shape, &bytes(&vec![0xdead_beef; rows * width]))
                    .unwrap();
            conditioning_overlay::native_for_device(&device, &seismic::NativeSpecialization::new())
                .unwrap()
                .call(conditioning_overlay::Args { input: &source, out: &mut out })
                .unwrap();
            assert_eq!(out.read_to_host().unwrap(), bytes(&input), "{:?} {rows}x{width}", device.backend());
        }
    }
}
