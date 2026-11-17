mod declared;
use magnitude_kernels::conditioning_overlay;
use seismic_lang::reference_math::conversion::canonical_nan;
use seismic_lang::types::DType;

/// The host's GPU (Metal on macOS, else Vulkan when present), then the CPU;
/// a requested backend among them alone
/// ([`seismic_native_tests::first_and_cpu`]).
fn devices() -> Vec<seismic::Device> {
    use seismic::BackendName::{Cpu, Metal, Vulkan};
    let gpu = if cfg!(target_os = "macos") { Metal } else { Vulkan };
    seismic_native_tests::first_and_cpu(&[gpu], &[gpu, Cpu])
}

/// The overlay publishes every conditioned value exactly (infinities, signed
/// zero and subnormals included) over rows that held other values, and a NaN
/// reads out as the registry's one NaN. Read from the host, a bit move and an
/// F32 store agree on every non-NaN value, so this checks the values, not that
/// the overlay moves bits.
#[test]
fn native_overlay_publishes_conditioned_rows_value_exactly() {
    for device in devices() {
        for (rows, width) in [(1usize, 2usize), (5, 37), (3, 2560)] {
            let special = [
                0x7fc0_1234u32,
                0x7f80_0000,
                0xff80_0000,
                0x8000_0000,
                0x0000_0001,
            ];
            let input = (0..rows * width)
                .map(|index| {
                    special
                        .get(index)
                        .copied()
                        .unwrap_or(((index as f32) * 0.37 - 11.0).to_bits())
                })
                .collect::<Vec<_>>();
            let bytes = |words: &[u32]| {
                words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>()
            };
            let shape = [rows as u64, width as u64];
            let source = seismic::Tensor::from_host(
                &device,
                seismic::Element::f32(),
                &shape,
                &bytes(&input),
            )
            .unwrap();
            let mut out = seismic::Tensor::from_host(
                &device,
                seismic::Element::f32(),
                &shape,
                &bytes(&vec![0xdead_beef; rows * width]),
            )
            .unwrap();
            declared::prepared(&device, &conditioning_overlay::native_entry(), &seismic::NativeSpecialization::new())
                .unwrap()
                .call(conditioning_overlay::Args {
                    input: &source,
                    out: &mut out,
                })
                .unwrap();
            let held = input
                .iter()
                .map(|&word| {
                    if f32::from_bits(word).is_nan() {
                        canonical_nan(DType::F32)
                    } else {
                        word
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(
                out.read_to_host().unwrap(),
                bytes(&held),
                "{:?} {rows}x{width}",
                device.backend()
            );
        }
    }
}
