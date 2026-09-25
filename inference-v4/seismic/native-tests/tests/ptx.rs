//! The CUDA device library (the CUDA prelude's inline-PTX helpers), one
//! fixture per helper group, run on every available CUDA device against host
//! references. Hosts without a CUDA device run nothing. Each formation's
//! wall time (NVRTC compilation plus module load) is printed.

use seismic::{
    Availability, BackendName, Device, DeviceCatalog, Element, NativeKernel, NativeSpecialization,
    Tensor,
};
use seismic_native_tests::{
    ptx_approximate, ptx_cp_async, ptx_dp4a_s8, ptx_dp4a_u8s8, ptx_ldmatrix, ptx_ldmatrix_trans,
    ptx_load_nc, ptx_mma_m16n8k16, ptx_mma_m16n8k32_s8, ptx_pack, ptx_redux_s32, ptx_redux_u32,
    ptx_rounded, ptx_shuffle, ptx_unpack,
};
use std::time::Instant;

fn cuda_devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let available = catalog.topology().devices().iter().any(|device| {
        device.backend == BackendName::Cuda
            && matches!(device.availability, Availability::Available)
    });
    if !available {
        eprintln!("no available CUDA device; the CUDA device-library fixtures do not run");
        return Vec::new();
    }
    vec![catalog
        .open_backend(BackendName::Cuda)
        .unwrap_or_else(|error| panic!("available CUDA device must open: {error}"))]
}

/// Form a native kernel, reporting the formation's wall time.
fn formed<E: seismic::Entry>(
    label: &str,
    form: impl FnOnce() -> Result<NativeKernel<E>, seismic::LoadError>,
) -> NativeKernel<E> {
    let started = Instant::now();
    let kernel = form().unwrap_or_else(|error| panic!("{label}: {error}"));
    eprintln!(
        "formation {label}: {:.1} ms",
        started.elapsed().as_secs_f64() * 1e3
    );
    kernel
}

fn tensor(device: &Device, element: Element, extents: &[u64], bytes: Vec<u8>) -> Tensor {
    Tensor::from_host(device, element, extents, &bytes).expect("host tensor")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn words<const N: usize>(tensor: &Tensor) -> Vec<[u8; N]> {
    tensor
        .read_to_host()
        .expect("host read")
        .chunks_exact(N)
        .map(|word| word.try_into().expect("word"))
        .collect()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    words::<4>(tensor)
        .into_iter()
        .map(f32::from_le_bytes)
        .collect()
}

fn read_i32(tensor: &Tensor) -> Vec<i32> {
    words::<4>(tensor)
        .into_iter()
        .map(i32::from_le_bytes)
        .collect()
}

fn read_u16(tensor: &Tensor) -> Vec<u16> {
    words::<2>(tensor)
        .into_iter()
        .map(u16::from_le_bytes)
        .collect()
}

/// A deterministic sequence (SplitMix64).
struct Sequence(u64);

impl Sequence {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn between(&mut self, low: i32, high: i32) -> i32 {
        low + (self.next() % u64::from((high - low + 1) as u32)) as i32
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Round-to-nearest-even f32 → bf16 for non-NaN values.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
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
        // Subnormal result: units of 2^-24.
        return sign | round(mantissa | 0x80_0000, (14 - biased) as u32) as u16;
    }
    // A carry out of the mantissa increments the exponent (to infinity at the top).
    sign | round(((biased as u32) << 23) | mantissa, 13) as u16
}

/// Exact f16 → f32.
fn f16_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        0x1f => f32::INFINITY,
        _ => (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[test]
fn mma_m16n8k16_matches_the_f32_reference_for_f16_and_bf16() {
    for device in cuda_devices() {
        let k = 64usize;
        // Small halves: every product and partial sum is exact in f32.
        let a = (0..16 * k)
            .map(|index| ((index * 7 + index / k * 3) % 9) as f32 * 0.5 - 2.0)
            .collect::<Vec<_>>();
        let b = (0..8 * k)
            .map(|index| ((index * 5 + index / k) % 7) as f32 * 0.5 - 1.5)
            .collect::<Vec<_>>();
        let expected = (0..16 * 8)
            .map(|index| {
                let (i, j) = (index / 8, index % 8);
                (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum::<f32>()
            })
            .collect::<Vec<_>>();
        for (name, element, encode) in [
            ("f16", Element::f16(), f16_bits as fn(f32) -> u16),
            ("bf16", Element::bf16(), bf16_bits as fn(f32) -> u16),
        ] {
            let encoded =
                |values: &[f32]| u16_bytes(&values.iter().map(|v| encode(*v)).collect::<Vec<_>>());
            let a_tensor = tensor(&device, element, &[16, k as u64], encoded(&a));
            let b_tensor = tensor(&device, element, &[8, k as u64], encoded(&b));
            let kernel = formed(&format!("ptx_mma_m16n8k16 {name}"), || {
                ptx_mma_m16n8k16::native_for_device_with(
                    &device,
                    ptx_mma_m16n8k16::Elements { E: element },
                    &NativeSpecialization::new().with_static("K", k as u64),
                )
            });
            let artifact = &kernel.artifact().0;
            assert!(artifact.contains("nvrtc 13."), "{artifact}");
            assert!(artifact.contains("--fmad=false"), "{artifact}");
            let result = kernel
                .call(ptx_mma_m16n8k16::Args {
                    a: &a_tensor,
                    b: &b_tensor,
                })
                .expect("call");
            assert_eq!(read_f32(&result.value), expected, "{name}");
        }
    }
}

#[test]
fn mma_m16n8k32_s8_matches_the_integer_reference() {
    for device in cuda_devices() {
        let k = 96usize;
        let mut sequence = Sequence(7);
        let a = (0..16 * k)
            .map(|_| sequence.between(-128, 127))
            .collect::<Vec<_>>();
        let b = (0..8 * k)
            .map(|_| sequence.between(-128, 127))
            .collect::<Vec<_>>();
        let expected = (0..16 * 8)
            .map(|index| {
                let (i, j) = (index / 8, index % 8);
                (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum::<i32>()
            })
            .collect::<Vec<_>>();
        let kernel = formed("ptx_mma_m16n8k32_s8", || {
            ptx_mma_m16n8k32_s8::native_for_device(
                &device,
                &NativeSpecialization::new().with_static("K", k as u64),
            )
        });
        let result = kernel
            .call(ptx_mma_m16n8k32_s8::Args {
                a: &tensor(&device, Element::i32(), &[16, k as u64], i32_bytes(&a)),
                b: &tensor(&device, Element::i32(), &[8, k as u64], i32_bytes(&b)),
            })
            .expect("call");
        assert_eq!(read_i32(&result.value), expected);
    }
}

#[test]
fn ldmatrix_fragments_reproduce_and_transpose_every_matrix() {
    for device in cuda_devices() {
        // 256 distinct exactly representable values.
        let values = (0..256)
            .map(|index| f16_bits(index as f32))
            .collect::<Vec<_>>();
        let x = tensor(&device, Element::f16(), &[4, 8, 8], u16_bytes(&values));
        let transposed = (0..256)
            .map(|index| {
                let (m, r, c) = (index / 64, index / 8 % 8, index % 8);
                values[m * 64 + c * 8 + r]
            })
            .collect::<Vec<_>>();
        for count in [1u64, 2, 4] {
            let specialization = NativeSpecialization::new().with_param("COUNT", count);
            let elements = || ptx_ldmatrix::Elements { E: Element::f16() };
            let plain = formed(&format!("ptx_ldmatrix x{count}"), || {
                ptx_ldmatrix::native_for_device_with(&device, elements(), &specialization)
            });
            let result = plain.call(ptx_ldmatrix::Args { x: &x }).expect("call");
            assert_eq!(read_u16(&result.value), values, "x{count}");
            let trans = formed(&format!("ptx_ldmatrix_trans x{count}"), || {
                ptx_ldmatrix_trans::native_for_device_with(
                    &device,
                    ptx_ldmatrix_trans::Elements { E: Element::f16() },
                    &specialization,
                )
            });
            let result = trans
                .call(ptx_ldmatrix_trans::Args { x: &x })
                .expect("call");
            assert_eq!(read_u16(&result.value), transposed, "x{count}.trans");
        }
    }
}

#[test]
fn cp_async_pipelines_copy_with_a_zero_filled_tail() {
    for device in cuda_devices() {
        for n in [5003usize, 1000, 3] {
            let values = (0..n)
                .map(|index| index as f32 * 0.5 - 7.0)
                .collect::<Vec<_>>();
            let x = tensor(&device, Element::f32(), &[n as u64], f32_bytes(&values));
            for stages in [1u64, 2, 3] {
                let kernel = formed(&format!("ptx_cp_async stages {stages} n {n}"), || {
                    ptx_cp_async::native_for_device(
                        &device,
                        &NativeSpecialization::new().with_param("STAGES", stages),
                    )
                });
                let result = kernel.call(ptx_cp_async::Args { x: &x }).expect("call");
                let copied = read_f32(&result.value);
                assert!(
                    !copied[0].is_nan(),
                    "stages {stages}, n {n}: zero fill did not happen"
                );
                assert_eq!(copied, values, "stages {stages}, n {n}");
            }
        }
    }
}

#[test]
fn non_coherent_loads_and_prefetch_copy_every_element() {
    for device in cuda_devices() {
        for n in [5003usize, 4096, 2, 1] {
            let values = (0..n)
                .map(|index| (index as f32).sqrt())
                .collect::<Vec<_>>();
            let x = tensor(&device, Element::f32(), &[n as u64], f32_bytes(&values));
            for no_allocate in [0u64, 1] {
                let kernel = formed(
                    &format!("ptx_load_nc no_allocate {no_allocate} n {n}"),
                    || {
                        ptx_load_nc::native_for_device(
                            &device,
                            &NativeSpecialization::new().with_param("NO_ALLOCATE", no_allocate),
                        )
                    },
                );
                let result = kernel.call(ptx_load_nc::Args { x: &x }).expect("call");
                assert_eq!(
                    read_f32(&result.value),
                    values,
                    "no_allocate {no_allocate}, n {n}"
                );
            }
        }
    }
}

#[test]
fn shuffles_and_butterfly_reductions_match_the_lane_reference() {
    for device in cuda_devices() {
        let w = 3usize;
        let values = (0..w * 32)
            .map(|index| ((index * 11) % 13) as f32 * 0.25 - 1.5)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0f32; 5 * w * 32];
        for row in 0..w {
            let lane_values = &values[row * 32..row * 32 + 32];
            let total = lane_values.iter().sum::<f32>();
            let largest = lane_values.iter().copied().fold(f32::MIN, f32::max);
            for lane in 0..32 {
                let place = |plane: usize| plane * w * 32 + row * 32 + lane;
                expected[place(0)] = lane_values[31 - lane];
                expected[place(1)] = lane_values[if lane + 3 < 32 { lane + 3 } else { lane }];
                expected[place(2)] = lane_values[if lane >= 3 { lane - 3 } else { lane }];
                expected[place(3)] = total;
                expected[place(4)] = largest;
            }
        }
        let kernel = formed("ptx_shuffle", || {
            ptx_shuffle::native_for_device(&device, &NativeSpecialization::new())
        });
        let result = kernel
            .call(ptx_shuffle::Args {
                x: &tensor(&device, Element::f32(), &[w as u64, 32], f32_bytes(&values)),
            })
            .expect("call");
        assert_eq!(read_f32(&result.value), expected);
    }
}

#[test]
fn redux_matches_the_integer_reference_for_signed_and_unsigned() {
    for device in cuda_devices() {
        let w = 4usize;
        let mut sequence = Sequence(11);
        let signed = (0..w * 32)
            .map(|_| sequence.between(-1000, 1000))
            .collect::<Vec<_>>();
        let unsigned = (0..w * 32)
            .map(|_| sequence.between(0, 100_000))
            .collect::<Vec<_>>();
        let expected = |values: &[i32]| {
            values
                .chunks_exact(32)
                .flat_map(|row| {
                    [
                        row.iter().sum::<i32>(),
                        *row.iter().min().expect("row"),
                        *row.iter().max().expect("row"),
                    ]
                })
                .collect::<Vec<_>>()
        };
        let s32 = formed("ptx_redux_s32", || {
            ptx_redux_s32::native_for_device(&device, &NativeSpecialization::new())
        })
        .call(ptx_redux_s32::Args {
            x: &tensor(&device, Element::i32(), &[w as u64, 32], i32_bytes(&signed)),
        })
        .expect("call");
        assert_eq!(read_i32(&s32.value), expected(&signed));
        // The unsigned values are below 2^31, so their i32 view orders the same.
        let u32 = formed("ptx_redux_u32", || {
            ptx_redux_u32::native_for_device(&device, &NativeSpecialization::new())
        })
        .call(ptx_redux_u32::Args {
            x: &tensor(
                &device,
                Element::u32(),
                &[w as u64, 32],
                i32_bytes(&unsigned),
            ),
        })
        .expect("call");
        assert_eq!(read_i32(&u32.value), expected(&unsigned));
    }
}

#[test]
fn dp4a_matches_the_byte_dot_product_reference() {
    for device in cuda_devices() {
        let n = 1000usize;
        let mut sequence = Sequence(3);
        let signed = (0..n * 4)
            .map(|_| sequence.between(-128, 127))
            .collect::<Vec<_>>();
        let unsigned = (0..n * 4)
            .map(|_| sequence.between(0, 255))
            .collect::<Vec<_>>();
        let b = (0..n * 4)
            .map(|_| sequence.between(-128, 127))
            .collect::<Vec<_>>();
        let acc = (0..n)
            .map(|_| sequence.between(-100_000, 100_000))
            .collect::<Vec<_>>();
        let reference = |a: &[i32]| {
            (0..n)
                .map(|i| acc[i] + (0..4).map(|j| a[i * 4 + j] * b[i * 4 + j]).sum::<i32>())
                .collect::<Vec<_>>()
        };
        let b_tensor = tensor(&device, Element::i32(), &[n as u64, 4], i32_bytes(&b));
        let acc_tensor = tensor(&device, Element::i32(), &[n as u64], i32_bytes(&acc));
        let s8 = formed("ptx_dp4a_s8", || {
            ptx_dp4a_s8::native_for_device(&device, &NativeSpecialization::new())
        })
        .call(ptx_dp4a_s8::Args {
            a: &tensor(&device, Element::i32(), &[n as u64, 4], i32_bytes(&signed)),
            b: &b_tensor,
            acc: &acc_tensor,
        })
        .expect("call");
        assert_eq!(read_i32(&s8.value), reference(&signed));
        let u8s8 = formed("ptx_dp4a_u8s8", || {
            ptx_dp4a_u8s8::native_for_device(&device, &NativeSpecialization::new())
        })
        .call(ptx_dp4a_u8s8::Args {
            a: &tensor(
                &device,
                Element::i32(),
                &[n as u64, 4],
                i32_bytes(&unsigned),
            ),
            b: &b_tensor,
            acc: &acc_tensor,
        })
        .expect("call");
        assert_eq!(read_i32(&u8s8.value), reference(&unsigned));
    }
}

#[test]
fn pairs_pack_with_nearest_even_rounding_and_unpack_exactly() {
    for device in cuda_devices() {
        let mut sequence = Sequence(5);
        // Wide magnitudes (f16 overflow and subnormals included), exact ties
        // for both formats, and signed zeros.
        let mut values = (0..4096)
            .map(|_| {
                let magnitude = 2f32.powi(sequence.between(-30, 17)) * (1.0 + sequence.unit());
                if sequence.next() & 1 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect::<Vec<_>>();
        values.extend([
            f32::from_bits(0x3f80_8000),
            f32::from_bits(0x3f81_8000),
            1.0 + 2f32.powi(-11),
            1.0 + 3.0 * 2f32.powi(-11),
            0.0,
            -0.0,
        ]);
        let x = tensor(
            &device,
            Element::f32(),
            &[values.len() as u64],
            f32_bytes(&values),
        );
        for (name, element, encode) in [
            ("f16", Element::f16(), f16_bits as fn(f32) -> u16),
            ("bf16", Element::bf16(), bf16_bits as fn(f32) -> u16),
        ] {
            let kernel = formed(&format!("ptx_pack {name}"), || {
                ptx_pack::native_for_device_with(
                    &device,
                    ptx_pack::Elements { E: element },
                    &NativeSpecialization::new(),
                )
            });
            let result = kernel.call(ptx_pack::Args { x: &x }).expect("call");
            let expected = values
                .iter()
                .map(|value| encode(*value))
                .collect::<Vec<_>>();
            assert_eq!(read_u16(&result.value), expected, "{name}");
        }
        // Every non-NaN 16-bit pattern widens exactly.
        for (name, element, decode, is_nan) in [
            (
                "f16",
                Element::f16(),
                f16_value as fn(u16) -> f32,
                (|bits: u16| bits & 0x7c00 == 0x7c00 && bits & 0x3ff != 0) as fn(u16) -> bool,
            ),
            (
                "bf16",
                Element::bf16(),
                bf16_value as fn(u16) -> f32,
                (|bits: u16| bits & 0x7f80 == 0x7f80 && bits & 0x7f != 0) as fn(u16) -> bool,
            ),
        ] {
            let mut patterns = (0..=u16::MAX)
                .filter(|bits| !is_nan(*bits))
                .collect::<Vec<_>>();
            if patterns.len() % 2 == 1 {
                patterns.push(0);
            }
            let kernel = formed(&format!("ptx_unpack {name}"), || {
                ptx_unpack::native_for_device_with(
                    &device,
                    ptx_unpack::Elements { E: element },
                    &NativeSpecialization::new(),
                )
            });
            let result = kernel
                .call(ptx_unpack::Args {
                    x: &tensor(
                        &device,
                        element,
                        &[patterns.len() as u64],
                        u16_bytes(&patterns),
                    ),
                })
                .expect("call");
            let widened = read_f32(&result.value);
            for (bits, value) in patterns.iter().zip(widened) {
                assert_eq!(
                    value.to_bits(),
                    decode(*bits).to_bits(),
                    "{name} {bits:#06x}"
                );
            }
        }
    }
}

#[test]
fn rounded_arithmetic_is_correctly_rounded_including_subnormals() {
    for device in cuda_devices() {
        let n = 4096usize;
        let mut sequence = Sequence(13);
        let mut draw = || {
            let magnitude = 2f32.powi(sequence.between(-140, 60)) * (1.0 + sequence.unit());
            if sequence.next() & 1 == 0 {
                magnitude
            } else {
                -magnitude
            }
        };
        let a = (0..n).map(|_| draw()).collect::<Vec<_>>();
        let b = (0..n).map(|_| draw()).collect::<Vec<_>>();
        let c = (0..n).map(|_| draw()).collect::<Vec<_>>();
        let kernel = formed("ptx_rounded", || {
            ptx_rounded::native_for_device(&device, &NativeSpecialization::new())
        });
        let result = kernel
            .call(ptx_rounded::Args {
                a: &tensor(&device, Element::f32(), &[n as u64], f32_bytes(&a)),
                b: &tensor(&device, Element::f32(), &[n as u64], f32_bytes(&b)),
                c: &tensor(&device, Element::f32(), &[n as u64], f32_bytes(&c)),
            })
            .expect("call");
        let values = read_f32(&result.value);
        for i in 0..n {
            let expected = [a[i].mul_add(b[i], c[i]), a[i] * b[i], a[i] + b[i]];
            for (plane, expected) in expected.into_iter().enumerate() {
                assert_eq!(
                    values[plane * n + i].to_bits(),
                    expected.to_bits(),
                    "plane {plane}: {} {} {}",
                    a[i],
                    b[i],
                    c[i]
                );
            }
        }
    }
}

#[test]
fn approximate_functions_stay_within_their_documented_error() {
    for device in cuda_devices() {
        let n = 4096usize;
        let mut sequence = Sequence(17);
        let x = (0..n)
            .map(|_| 0.01 + 8.0 * sequence.unit())
            .collect::<Vec<_>>();
        let kernel = formed("ptx_approximate", || {
            ptx_approximate::native_for_device(&device, &NativeSpecialization::new())
        });
        let result = kernel
            .call(ptx_approximate::Args {
                x: &tensor(&device, Element::f32(), &[n as u64], f32_bytes(&x)),
            })
            .expect("call");
        let values = read_f32(&result.value);
        // (reference, relative bound, absolute bound) per plane.
        let checks: [(fn(f64) -> f64, f64, f64); 5] = [
            (f64::exp2, 2e-6, 0.0),
            (f64::log2, 0.0, 2e-6),
            (|v| 1.0 / v, 2e-7, 0.0),
            (|v| 1.0 / v.sqrt(), 5e-7, 0.0),
            (f64::tanh, 0.0, 1e-3),
        ];
        for (plane, (reference, relative, absolute)) in checks.into_iter().enumerate() {
            for i in 0..n {
                let expected = reference(f64::from(x[i]));
                let error = (f64::from(values[plane * n + i]) - expected).abs();
                assert!(
                    error <= relative * expected.abs() + absolute,
                    "plane {plane} at {}: {} vs {expected}",
                    x[i],
                    values[plane * n + i]
                );
            }
        }
    }
}
