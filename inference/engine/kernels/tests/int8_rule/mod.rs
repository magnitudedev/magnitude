//! The INT8 activation rule, as the backends' tests check it (the rule itself
//! is `seismic_native_cpu::quant`'s, over the scale's grid): every backend
//! quantizes a group of values by the CPU's rule (`seismic-native-cpu`
//! `quant.rs`): a scale `d` of `max|x| / 127` rounded to nearest, or where
//! that is subnormal rounded up onto the subnormal grid (so every code lies
//! in [-127, 127] and only an all-zero group has a zero scale), and each code
//! the exact quotient `x / d` rounded once, half away from zero, NaN coding 0.
//!
//! A test runs [`blocks`] through its backend's INT8 projection against
//! weights that read one column each (`probe_blocks`), so an output is its
//! column's `d * code`, and [`assert_codes`] compares them with the rule's.
//! The CPU's scale is F32; Metal stores its scales as F16, the same rule on
//! that grid ([`Grid`]).
#![allow(dead_code)]

use seismic_lang::registry::{bf16_round, f16_bits};
use seismic_native_cpu::quant::{self, Q8Block};

/// Values per group: the INT8 GEMMs' scale group.
pub const GROUP: usize = 32;

/// The format a backend stores its scales in, which the rule rounds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grid {
    F32,
    F16,
}

/// The CPU's quantization of one group (a partial block: its scale and the
/// group's codes).
pub fn cpu(values: &[f32; GROUP]) -> (f32, [i8; GROUP]) {
    let mut block = [Q8Block::ZERO];
    quant::quantize(values, &mut block);
    (block[0].scale(), block[0].codes()[..GROUP].try_into().unwrap())
}

/// The rule on `grid`: the CPU's quantizer on F32, and on F16 the same rule
/// (`quant::quantize_group`, its one owner) over F16's grid.
pub fn by_rule(values: &[f32; GROUP], grid: Grid) -> (f32, [i8; GROUP]) {
    match grid {
        Grid::F32 => cpu(values),
        Grid::F16 => {
            let mut codes = [0i8; GROUP];
            let scale = quant::quantize_group::<quant::F16Grid>(values, &mut codes);
            (scale, codes)
        }
    }
}

/// The fixture groups, every value BF16 (the activations' element): zero
/// groups and a group of NaN and signed zeros; exact ties (`127 2^k` the
/// maximum, `(n + 1/2) 2^k` the values, which half-to-even coding takes to
/// even codes), each with its BF16 neighbours; groups whose scale is
/// subnormal on F32 (maxima down to BF16's least subnormal, which the CPU
/// and CUDA keep) and on F16 (maxima below 127 2^-14, from just under it to
/// below 2^-24); and random groups of random scale.
pub fn blocks() -> Vec<[f32; GROUP]> {
    let mut groups: Vec<[f32; GROUP]> = vec![[0.0; GROUP], [-0.0; GROUP]];
    let mut mixed = [0.0f32; GROUP];
    mixed[3] = f32::NAN;
    mixed[4] = -0.0;
    mixed[9] = f32::NAN;
    groups.push(mixed);
    mixed[0] = 0.75;
    mixed[5] = -f32::from_bits(0x0001_0000);
    groups.push(mixed);

    for exponent in [-100, -12, -3, 0, 5, 10] {
        let step = 2f32.powi(exponent);
        // Every tie (2n + 1) / 2 for n < 127, in groups that hold the
        // maximum, 31 values each; alternating signs; then every tie's
        // BF16 neighbours below and above.
        let ties = (0..127).map(|n| (2 * n + 1) as f32 * 0.5 * step).collect::<Vec<_>>();
        let neighbours = ties
            .iter()
            .flat_map(|tie| [f32::from_bits(tie.to_bits() - 0x1_0000), f32::from_bits(tie.to_bits() + 0x1_0000)])
            .collect::<Vec<_>>();
        for values in [ties, neighbours] {
            for chunk in values.chunks(GROUP - 1) {
                let mut group = [0.0f32; GROUP];
                group[0] = 127.0 * step;
                for (i, value) in chunk.iter().enumerate() {
                    group[1 + i] = if (i as i32 + exponent).rem_euclid(2) == 0 { *value } else { -value };
                }
                groups.push(group);
            }
        }
    }

    let least = f32::from_bits(0x0001_0000); // BF16's least subnormal, 2^-133
    let maxima = [
        bf16_round(127.0 * f32::MIN_POSITIVE * 0.999),
        bf16_round(1.49e-36),
        2f32.powi(-122),
        2f32.powi(-130),
        least * 127.0,
        least * 3.0,
        least,
        // F16's subnormal scales: below 127 2^-14, down to under 2^-24.
        bf16_round(127.0 * 2f32.powi(-14) * 0.999),
        bf16_round(0.005),
        bf16_round(1e-3),
        2f32.powi(-14),
        bf16_round(3e-5),
        bf16_round(1e-6),
        bf16_round(2.0e-8),
    ];
    let mut state = 0x2545_f491_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for maximum in maxima {
        let mut group = [0.0f32; GROUP];
        for value in &mut group {
            let fraction = (next() % 2001) as f32 / 1000.0 - 1.0;
            *value = bf16_round(maximum * fraction);
        }
        group[0] = maximum;
        group[5] = f32::NAN;
        group[6] = -0.0;
        group[7] = -least.min(maximum);
        groups.push(group);
    }
    for _ in 0..64 {
        let scale = 2f32.powi((next() % 40) as i32 - 20);
        groups.push(std::array::from_fn(|_| {
            bf16_round(scale * ((next() % 4001) as f32 / 2000.0 - 1.0))
        }));
    }
    for group in &groups {
        assert!(group.iter().all(|value| value.is_nan() || bf16_round(*value) == *value), "BF16 values");
    }
    groups
}

/// Whether `group` has a nonzero F32 subnormal value, which a device that
/// flushes them (Metal) reads as zero.
pub fn has_subnormal(group: &[f32; GROUP]) -> bool {
    group.iter().any(|value| value.is_subnormal())
}

/// The Q8_0 blocks (GGUF bytes: an F16 scale 1.0, then 32 codes) of `rows`
/// weight rows over `k` columns whose row `n < 32` holds code 1 at column
/// `n` and every other code 0: a product with an activation row is its
/// column `n` of the first group.
pub fn probe_blocks(rows: usize, k: usize) -> Vec<u8> {
    assert!(k % GROUP == 0);
    let mut bytes = Vec::new();
    for row in 0..rows {
        for block in 0..k / GROUP {
            bytes.extend(f16_bits(1.0).to_le_bytes());
            bytes.extend((0..GROUP).map(|column| u8::from(block == 0 && row == column)));
        }
    }
    bytes
}

/// Checks that `outputs` (row `r`'s `stride` values, the first group's
/// columns `0..32` its `d * code`) are those of `groups[r]` under the rule
/// on `grid`: each output within `slack` (relative) of `d * code`, and so
/// its code, since a code one off is a scale `d` away.
pub fn assert_codes(label: &str, groups: &[[f32; GROUP]], outputs: &[f32], stride: usize, grid: Grid, slack: f64) {
    assert_eq!(outputs.len(), groups.len() * stride, "{label}: outputs");
    let mut failures = Vec::new();
    for (row, group) in groups.iter().enumerate() {
        let (scale, codes) = by_rule(group, grid);
        for (column, code) in codes.iter().enumerate() {
            let actual = f64::from(outputs[row * stride + column]);
            let expected = f64::from(scale) * f64::from(*code);
            let bound = expected.abs() * slack + 1e-300;
            // A scale `d` above or below the exact code is outside the slack
            // for every |code| <= 127 where slack < 1 / 254.
            if !((actual - expected).abs() <= bound) {
                failures.push((row, column, group[column], scale, *code, actual));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{label}: {} of {} codes differ; first (row, column, value, scale, code, output): {:?}",
        failures.len(),
        groups.len() * GROUP,
        &failures[..failures.len().min(4)]
    );
}
