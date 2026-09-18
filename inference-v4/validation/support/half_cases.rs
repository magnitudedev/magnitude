//! Independent binary16 rounding boundaries: every finite adjacent pair supplies
//! its midpoint and the immediately neighboring f32 values. This does not repeat
//! the compiler's bit-conversion algorithm.
use seismic_lang::{
    lowered_ir::LoweredIr,
    program::{compile, SourceFile},
    Scope,
};
use std::collections::HashMap;
fn half_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 10) & 31;
    let fraction = bits & 1023;
    if exp == 0 {
        sign * f32::from(fraction) * 2.0f32.powi(-24)
    } else if exp == 31 {
        if fraction == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + f32::from(fraction) / 1024.0) * 2.0f32.powi(i32::from(exp) - 15)
    }
}
pub fn exercise(mut run: impl FnMut(&LoweredIr, &mut [Vec<u8>]), backend: &str) {
    let text="fn encode[N](x: tensor[N] f32, out: tensor[N] f16):\n  for row in parallel:\n    t = load(x[row:row+1])\n    store(t,out[row:row+1])\n\nfn decode[N](x: tensor[N] f16, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    store(t,out[row:row+1])\n";
    let program = compile(
        &[SourceFile {
            path: "half.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let input = (0..=u16::MAX)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let mut buffers = [input, vec![0; 65536 * 4]];
    let lowered = seismic_lang::lower::lower(
        &program,
        "decode",
        backend,
        &HashMap::from([("N".into(), 65536)]),
    )
    .unwrap();
    run(&lowered, &mut buffers);
    for (bits, bytes) in (0..=u16::MAX).zip(buffers[1].chunks_exact(4)) {
        let expected = half_value(bits);
        let actual = f32::from_le_bytes(bytes.try_into().unwrap());
        if expected.is_nan() {
            assert!(actual.is_nan(), "decode {bits:04x}")
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "decode {bits:04x}")
        }
    }
    let mut samples = vec![
        (0.0, 0u16),
        (-0.0, 0x8000),
        (f32::MAX, 0x7c00),
        (-f32::MAX, 0xfc00),
        (f32::INFINITY, 0x7c00),
        (f32::NEG_INFINITY, 0xfc00),
    ];
    for low in 0..0x7c00u16 {
        let a = half_value(low);
        let b = if low == 0x7bff {
            65536.0
        } else {
            half_value(low + 1)
        };
        let midpoint = (a + b) * 0.5;
        let tie = if low & 1 == 0 { low } else { low + 1 };
        for (value, expected) in [
            (a, low),
            (f32::from_bits(midpoint.to_bits() - 1), low),
            (midpoint, tie),
            (f32::from_bits(midpoint.to_bits() + 1), low + 1),
        ] {
            samples.push((value, expected));
            samples.push((-value, expected | 0x8000));
        }
    }
    let mut buffers = [
        samples
            .iter()
            .flat_map(|(value, _)| value.to_le_bytes())
            .collect::<Vec<_>>(),
        vec![0; samples.len() * 2],
    ];
    let lowered = seismic_lang::lower::lower(
        &program,
        "encode",
        backend,
        &HashMap::from([("N".into(), samples.len() as i64)]),
    )
    .unwrap();
    run(&lowered, &mut buffers);
    for ((value, expected), bytes) in samples.iter().zip(buffers[1].chunks_exact(2)) {
        assert_eq!(
            u16::from_le_bytes(bytes.try_into().unwrap()),
            *expected,
            "encode {value:e} ({:08x})",
            value.to_bits()
        );
    }
    eprintln!(
        "binary16: 65536 decodes and {} exact rounding-boundary encodes",
        samples.len()
    );
    let mut buffers = [
        [0x7f80_0001u32, 0xff80_0001, 0x7fc0_0000, 0x7fff_ffff]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
        vec![0; 8],
    ];
    let lowered = seismic_lang::lower::lower(
        &program,
        "encode",
        backend,
        &HashMap::from([("N".into(), 4)]),
    )
    .unwrap();
    run(&lowered, &mut buffers);
    for bytes in buffers[1].chunks_exact(2) {
        assert!(
            u16::from_le_bytes(bytes.try_into().unwrap()) & 0x7fff > 0x7c00,
            "half NaN encoding cannot become infinity"
        );
    }
}
