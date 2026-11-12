// line_int8 on CPU (contract in line.seismic): a work item makes S int8
// multiply-adds into i32, in UNROLL independent accumulators seeded apart,
// each one vector register of its tier's quantized dots
// (`Isa::QUANTIZED_DOTS`), the instructions the INT8 form's dots issue
// (`seismic::cpu::weights::quantized_dot`: `vpdpbusd` where the tier has
// VNNI, on whole 512-bit registers where the tier has them; AVX2 byte pair
// products otherwise; NEON `sdot`; a tier with none is refused) in loop
// iterations of ROUNDS rounds of every accumulator, so the loop's own
// instructions are a small part of each. Every operand byte is twice its
// `x`, truncated; it writes its `x` back.

/// Rounds of every accumulator one loop iteration makes.
const ROUNDS: usize = 8;

#[inline(always)]
fn line_int8<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let lane = group[0] as usize;
    let x = cx.arg_x().get([lane]);
    let byte = (2.0 * x) as i8;
    let steps = cx.dim_s() as usize;
    let total = match cx.param_unroll() {
        1 => dots::<L, 1>(byte, steps),
        2 => dots::<L, 2>(byte, steps),
        4 => dots::<L, 4>(byte, steps),
        8 => dots::<L, 8>(byte, steps),
        16 => dots::<L, 16>(byte, steps),
        32 => dots::<L, 32>(byte, steps),
        other => unreachable!("UNROLL {other} is not declared"),
    };
    // SAFETY: each work item writes its own lane.
    unsafe { cx.arg_out().set([lane], x + f32::from(u8::from(total == u32::MAX))) };
}

/// The wrapped sum of C accumulators after `steps` int8 multiply-adds among
/// them.
#[inline(always)]
fn dots<L: Isa, const C: usize>(byte: i8, steps: usize) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if L::TIER.vector_bytes() == 64 && L::TIER.has("avx512vnni") {
        use std::arch::x86_64::*;
        // SAFETY: the tier has AVX-512 VNNI; the peak runs its whole
        // registers, twice the forms' 256-bit dots.
        unsafe {
            let operand = _mm512_set1_epi8(byte);
            let mut sums: [__m512i; C] = std::array::from_fn(|k| _mm512_set1_epi32(k as i32));
            for _ in 0..steps / (64 * C * ROUNDS) {
                for _ in 0..ROUNDS {
                    for sum in sums.iter_mut() {
                        *sum = seismic::cpu::weights::quantized_dot_wide::<L>(*sum, operand, operand);
                    }
                }
            }
            return sums.iter().fold(0u32, |total, sum| {
                let lanes: [i32; 16] = std::mem::transmute(*sum);
                lanes.iter().fold(total, |total, lane| total.wrapping_add(*lane as u32))
            });
        }
    }
    #[cfg(target_arch = "x86_64")]
    if L::QUANTIZED_DOTS > 0 {
        use std::arch::x86_64::*;
        // SAFETY: the tier's features are this function's, inlined into its
        // boundary; its dots are the INT8 form's.
        unsafe {
            let operand = _mm256_set1_epi8(byte);
            let mut sums: [__m256i; C] = std::array::from_fn(|k| _mm256_set1_epi32(k as i32));
            for _ in 0..steps / (32 * C * ROUNDS) {
                for _ in 0..ROUNDS {
                    for sum in sums.iter_mut() {
                        // The pair products do not depend on the accumulator,
                        // so a tier without a fused dot reads its weights
                        // opaquely, else they would be hoisted out of the
                        // loop and shared between accumulators.
                        let weights = if L::QUANTIZED_DOTS == 1 { std::hint::black_box(operand) } else { operand };
                        *sum = seismic::cpu::weights::quantized_dot::<L>(*sum, weights, operand);
                    }
                }
            }
            return sums.iter().fold(0u32, |total, sum| {
                let lanes: [i32; 8] = std::mem::transmute(*sum);
                lanes.iter().fold(total, |total, lane| total.wrapping_add(*lane as u32))
            });
        }
    }
    #[cfg(target_arch = "aarch64")]
    if L::QUANTIZED_DOTS > 0 {
        use std::arch::aarch64::*;
        // SAFETY: the tier has NEON's dot product.
        unsafe {
            let operand = vdupq_n_s8(byte);
            let mut sums: [int32x4_t; C] = std::array::from_fn(|k| vdupq_n_s32(k as i32));
            for _ in 0..steps / (16 * C * ROUNDS) {
                for _ in 0..ROUNDS {
                    for sum in sums.iter_mut() {
                        *sum = seismic::cpu::weights::quantized_dot_neon(*sum, operand, operand);
                    }
                }
            }
            return sums.iter().fold(0u32, |total, sum| total.wrapping_add(vaddvq_s32(*sum) as u32));
        }
    }
    // A tier with no quantized dots is refused before this runs: a scalar
    // loop of ours would time nothing of the hardware.
    unreachable!("the job refuses the int8 probe on a tier without quantized dots")
}
