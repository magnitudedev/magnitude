//! Conformance of each device backend's element owner (`owner_publish`):
//! every write operation of the owner (CUDA and Metal `element::Out`, Vulkan
//! `out_put*`) stores a NaN as a NaN and every other value as the
//! element's round to nearest even: finite values and infinities exactly.
//! The engine's boundary owns the one NaN: what the test reads back
//! (`read_to_host`) is the registry's one NaN of the element (F32
//! `0x7fc00000`, BF16 `0x7fc0`, F16 `0x7e00`) for every NaN payload class. The CPU's owner has its own unit test
//! (`seismic_native_cpu::element`).

use seismic::{BackendName, Element, NativeSpecialization, Tensor};
use seismic_native_tests::owner_publish;

/// NaNs of every class: both signs, quiet and signalling, a payload
/// (`0x1234`), all ones, the registry's own, and payloads a truncating or
/// carrying narrowing turns into an infinity (`0x7f800001`, `0x7f807fff`,
/// `0x7f800fff`, `0x7f801000`), a zero (`0xffffffff` + bf16's rounding
/// bias), or another NaN (`0x7fbfffff`).
const NANS: [u32; 13] = [
    0x7f80_0001,
    0xff80_0001,
    0x7fc0_1234,
    0xffc0_0000,
    0xffff_ffff,
    0x7fbf_ffff,
    0x7fc0_0000,
    0x7fff_ffff,
    0xff80_1234,
    0x7f80_7fff,
    0x7f80_0fff,
    0x7f80_1000,
    0x7fa0_0000,
];

/// Finite values and infinities: exact in every element, and rounding ties
/// and overflows of the 16-bit elements.
const VALUES: [u32; 16] = [
    0x0000_0000,             // +0
    0x8000_0000,             // -0
    0x3f80_0000,             // 1
    0xc020_0000,             // -2.5
    0x4380_0000,             // 256
    0xbec0_0000,             // -0.375
    0x3580_0000,             // 2^-20: an f16 subnormal
    0x477f_e000,             // 65504: f16's largest; bf16 rounds it up
    0x477f_f000,             // 65520: f16 ties to infinity
    0x3f80_8000,             // a bf16 tie (to even, down)
    0x3f80_1000,             // 1 + 2^-11: an f16 tie (to even, down)
    0x7f7f_ffff,             // f32's largest: both 16-bit elements overflow
    0x0daa_c19c,             // about 1e-30: zero in f16, normal in bf16
    0xc77f_ffff,             // -65535.996: f16 overflows to -infinity
    0x7f80_0000,             // +infinity
    0xff80_0000,             // -infinity
];

/// The write operations of each backend's kernel, result row by row.
fn operations(backend: BackendName) -> &'static [&'static str] {
    match backend {
        BackendName::Cuda => &[
            "put",
            "put2 at even indices",
            "put2 at odd indices",
            "put_span<2>",
            "put_span<4>",
            "put_span<8>",
            "offset().put_span<8>",
        ],
        BackendName::Metal => &[
            "put",
            "put2 at even indices",
            "put2 at odd indices",
            "put4 aligned",
            "put4 at an odd element",
            "put8 as one 16-byte store",
        ],
        BackendName::Vulkan => &[
            "out_put",
            "out_put2 at even indices",
            "out_put2 at odd indices",
            "out_put8 as 16-byte words",
        ],
        other => panic!("{other:?} has no owner_publish kernel"),
    }
}

/// The element's stored bits of `bits` (16-bit elements in the low half), as
/// the registry defines them.
fn stored(element: &str, bits: u32) -> u32 {
    let value = f32::from_bits(bits);
    match (element, value.is_nan()) {
        ("f32", true) => 0x7fc0_0000,
        ("f32", false) => bits,
        ("bf16", true) => 0x7fc0,
        ("bf16", false) => (bits + 0x7fff + ((bits >> 16) & 1)) >> 16,
        ("f16", true) => 0x7e00,
        ("f16", false) => u32::from(f16_bits(value)),
        _ => unreachable!("an element of the test"),
    }
}

/// Round-to-nearest-even f32 → f16 for non-NaN values.
fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00;
    }
    let biased = exponent - 127 + 15;
    if biased >= 0x1f {
        return sign | 0x7c00;
    }
    let round = |value: u32, shift: u32| {
        let half = 1u32 << (shift - 1);
        let remainder = value & ((1u32 << shift) - 1);
        let truncated = value >> shift;
        if remainder > half || (remainder == half && truncated & 1 == 1) {
            truncated + 1
        } else {
            truncated
        }
    };
    if biased <= 0 {
        if biased < -10 {
            return sign;
        }
        return sign | round(mantissa | 0x80_0000, (14 - biased) as u32) as u16;
    }
    sign | round(((biased as u32) << 23) | mantissa, 13) as u16
}

/// One row of the input: every payload at every position of an eight-element
/// span (so in both halves of every pair and every lane of every vector
/// store), `x[8a + b] = payloads[(a + b) % P]`.
fn row() -> Vec<u32> {
    let payloads = NANS.iter().chain(&VALUES).copied().collect::<Vec<_>>();
    let count = payloads.len();
    (0..count).flat_map(|a| (0..8).map(move |b| (a, b))).map(|(a, b)| payloads[(a + b) % count]).collect()
}

fn specialization(rows: usize, n: usize) -> NativeSpecialization {
    NativeSpecialization::new().with_static("R", rows as u64).with_static("N", n as u64)
}

const ELEMENTS: [&str; 3] = ["f32", "bf16", "f16"];

fn element(name: &str) -> Element {
    match name {
        "f32" => Element::f32(),
        "bf16" => Element::bf16(),
        "f16" => Element::f16(),
        _ => unreachable!("an element of the test"),
    }
}

/// Every write operation of every device backend's owner, for every element
/// and payload: a NaN is read out as the registry's one NaN, every other
/// value as its round to nearest even.
#[test]
fn every_owner_write_publishes_the_registrys_one_nan() {
    use BackendName::{Cuda, Metal, Vulkan};
    let x = row();
    let n = x.len();
    let devices = seismic_native_tests::setup(&[Metal, Cuda, Vulkan], |setup| {
        let rows = operations(setup.backend()).len();
        for name in ELEMENTS {
            setup.admit(
                format_args!("owner_publish {name}"),
                owner_publish::native_admission_with(
                    setup.backend(),
                    owner_publish::Elements { E: element(name) },
                    &specialization(rows, n),
                ),
            );
        }
    });
    let mut failures = Vec::new();
    for device in devices {
        let backend = device.backend();
        let operations = operations(backend);
        let rows = operations.len();
        let bytes = (0..rows).flat_map(|_| &x).flat_map(|bits| bits.to_le_bytes()).collect::<Vec<_>>();
        let input = Tensor::from_host(&device, Element::f32(), &[rows as u64, n as u64], &bytes).expect("host tensor");
        for name in ELEMENTS {
            let kernel = owner_publish::native_for_device_with(
                &device,
                owner_publish::Elements { E: element(name) },
                &specialization(rows, n),
            )
            .unwrap_or_else(|error| panic!("owner_publish {name} on {backend:?}: {error}"));
            let result = kernel.call(owner_publish::Args { x: &input }).expect("call");
            let published = result.value.read_to_host().expect("host read");
            let width = if name == "f32" { 4 } else { 2 };
            assert_eq!(published.len(), rows * n * width, "{backend:?} {name}");
            for (index, word) in published.chunks_exact(width).enumerate() {
                let got = word.iter().rev().fold(0u32, |bits, byte| bits << 8 | u32::from(*byte));
                let (r, column) = (index / n, index % n);
                let expected = stored(name, x[column]);
                if got != expected {
                    failures.push(format!(
                        "{backend:?} {name} {}: {:#010x} published {got:#x}, expected {expected:#x} (column {column})",
                        operations[r], x[column]
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} published elements differ:\n{}", failures.len(), failures.join("\n"));
}
