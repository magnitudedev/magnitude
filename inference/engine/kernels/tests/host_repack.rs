//! The host's row conversions (`seismic::cpu::repack`), which land streamed
//! expert records into a host tier (`experts::landing`), against the device's
//! `repack_weight`, which imports the same weights resident: for every
//! registered conversion into the device's resident layout, the bytes a
//! landing writes, converted piece by piece as a landing converts them,
//! equal the device repack's byte for byte. An expert record of a format the
//! two convert differently would read differently streamed than resident.

mod declared;
mod random;

use magnitude_kernels::repack_weight;
use random::Rng;
use seismic::{Device, Element};

/// The source bytes a landing converts at once (`experts::landing`'s
/// `CONVERSION_PIECE`): pieces of whole stored tiles of at most this many.
const CONVERSION_PIECE: usize = 128 << 10;

/// Expert matrices `[N, K]` at served models' widths: an up projection
/// (Qwen3.6 35B-A3B's 512 features of a 2048 hidden) and its down
/// projection.
const MATRICES: [[usize; 2]; 2] = [[512, 2048], [2048, 512]];

/// An external packet format a routed expert is stored in, and where its
/// packet's float scales lie: f16 words, and UE4M3 bytes. Records are
/// random packets whose scales are finite and in range, as a quantizer
/// writes them: a random scale may be a NaN, whose payload a conversion
/// need not keep.
struct Streamed {
    source: &'static str,
    packet: usize,
    f16: &'static [usize],
    e4m3: &'static [usize],
}

/// The formats served catalog models store routed experts in (Q4_K, Q5_K,
/// Q6_K, Q8_0 and Q5_1 GGUF variants, NVFP4, IQ2_XXS), IQ2_XS, and MXFP4,
/// the withheld DeepSeek V4.1's (its E8M0 scale moves bit for bit): each
/// must have a host conversion into the resident layout that the device
/// repack matches.
const STREAMED: [Streamed; 9] = [
    Streamed { source: "gguf_q4_k", packet: 144, f16: &[0, 2], e4m3: &[] },
    Streamed { source: "gguf_q5_k", packet: 176, f16: &[0, 2], e4m3: &[] },
    Streamed { source: "gguf_q6_k", packet: 210, f16: &[208], e4m3: &[] },
    Streamed { source: "gguf_q8_0", packet: 34, f16: &[0], e4m3: &[] },
    Streamed { source: "gguf_q5_1", packet: 24, f16: &[0, 2], e4m3: &[] },
    Streamed { source: "gguf_nvfp4", packet: 36, f16: &[], e4m3: &[0, 1, 2, 3] },
    Streamed { source: "gguf_iq2_xxs", packet: 66, f16: &[0], e4m3: &[] },
    Streamed { source: "gguf_iq2_xs", packet: 74, f16: &[0], e4m3: &[] },
    Streamed { source: "gguf_mxfp4", packet: 17, f16: &[], e4m3: &[] },
];

impl Streamed {
    /// `packets` random packets, their scales finite and in range.
    fn record(&self, packets: usize, rng: &mut Rng) -> Vec<u8> {
        let mut bytes = (0..packets * self.packet).map(|_| rng.byte()).collect::<Vec<_>>();
        for packet in bytes.chunks_exact_mut(self.packet) {
            for &at in self.f16 {
                let scale = seismic_lang::registry::f16_bits(rng.uniform(1e-4, 1e-1));
                packet[at..at + 2].copy_from_slice(&scale.to_le_bytes());
            }
            for &at in self.e4m3 {
                // Exponent 1..=14, any mantissa: normal, never the NaN 0x7f.
                packet[at] = ((1 + rng.byte() % 14) << 3) | (rng.byte() & 7);
            }
        }
        bytes
    }
}

#[test]
fn host_landing_conversions_equal_the_device_repack() {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    for backend in [seismic::BackendName::Metal, seismic::BackendName::Cuda, seismic::BackendName::Vulkan] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        conversions_equal_on(&device);
    }
}

fn conversions_equal_on(device: &Device) {
    let layout = device.backend().resident_layout();
    let mut compared = Vec::new();
    let mut unreferenced = Vec::new();
    for conversion in seismic::cpu::repack::conversions() {
        let Some(streamed) = STREAMED.iter().find(|streamed| streamed.source == conversion.source()) else {
            continue;
        };
        let (Some(source), Some(target)) = (Element::named(conversion.source()), Element::named(conversion.target())) else {
            continue;
        };
        if target.layout() != layout {
            continue;
        }
        let entry = repack_weight::native_entry_with(repack_weight::Elements { E: source, U: target });
        let Ok(specialization) = declared::declared(device, &entry, |name| panic!("unexpected static {name}"), &[]) else {
            unreferenced.push(conversion.source());
            continue;
        };
        let Ok(kernel) = entry.prepare(device, &specialization) else {
            unreferenced.push(conversion.source());
            continue;
        };
        for [rows, k] in MATRICES {
            if k % conversion.group() != 0 {
                continue;
            }
            let form = format!("{:?}: {} -> {} [{rows}, {k}]", device.backend(), conversion.source(), conversion.target());
            let mut rng = Rng::new(0x5eed ^ (rows * k) as u64);
            assert_eq!(conversion.source_bytes(1, conversion.group()), streamed.packet, "{form}: packet bytes");
            let stored = streamed.record(rows * k / conversion.group(), &mut rng);
            // The host's, as a landing converts a stored record.
            let row = conversion.stored_row_bytes(k).unwrap_or_else(|| panic!("{form}: the target holds no such row"));
            let mut host = vec![0u8; rows.next_multiple_of(conversion.tile()) * row];
            let tiles = std::num::NonZeroUsize::new((CONVERSION_PIECE / conversion.source_bytes(conversion.tile(), k)).max(1)).unwrap();
            for (from, to) in conversion.pieces(k, rows, tiles).unwrap() {
                conversion.piece(k, rows, from, &stored, &mut host[to]).unwrap_or_else(|| panic!("{form}: a piece did not convert"));
            }
            // The device's.
            let tensor = seismic::Tensor::from_host(device, source, &[1, rows as u64, k as u64], &stored).unwrap();
            let device_bytes = kernel.call(repack_weight::Args { source: &tensor }).unwrap().value.read_to_host().unwrap();
            assert_eq!(device_bytes.len(), host.len(), "{form}: record bytes");
            if let Some(first) = device_bytes.iter().zip(&host).position(|(device, host)| device != host) {
                let differing = device_bytes.iter().zip(&host).filter(|(device, host)| device != host).count();
                panic!("{form}: {differing} of {} bytes differ, the first at {first} (row {})", host.len(), first / row);
            }
        }
        compared.push(conversion.source());
    }
    eprintln!("{:?}: host conversions equal the device repack for {compared:?}; no device repack for {unreferenced:?}", device.backend());
    assert!(unreferenced.is_empty(), "{:?}: no device repack to compare for {unreferenced:?}", device.backend());
    let uncovered = STREAMED.iter().map(|streamed| streamed.source).filter(|source| !compared.contains(source)).collect::<Vec<_>>();
    assert!(uncovered.is_empty(), "{:?}: no host and device repack compared for streamed formats {uncovered:?}", device.backend());
}
