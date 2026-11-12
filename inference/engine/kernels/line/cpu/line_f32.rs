// line_f32 on CPU (contract in line.seismic): a work item makes S f32
// multiply-adds in UNROLL independent chains, each one vector register of
// its tier (AVX-512's 16 values, AVX2's 8 or NEON's 4, fused; SSE's 4, a
// multiply and an add), in loop iterations of ROUNDS rounds of every chain,
// so the loop's own instructions are a small part of each; it writes its
// `x` back. The chains are registers by construction: plain loops over
// arrays of lanes kept them in memory, a load and a store a multiply-add.

/// Rounds of every chain one loop iteration makes.
const ROUNDS: usize = 8;

#[inline(always)]
fn line_f32<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let lane = group[0] as usize;
    let x = cx.arg_x().get([lane]);
    let steps = cx.dim_s() as usize;
    let total = match cx.param_unroll() {
        1 => chains::<L, 1>(x, steps),
        2 => chains::<L, 2>(x, steps),
        4 => chains::<L, 4>(x, steps),
        8 => chains::<L, 8>(x, steps),
        16 => chains::<L, 16>(x, steps),
        32 => chains::<L, 32>(x, steps),
        other => unreachable!("UNROLL {other} is not declared"),
    };
    // SAFETY: each work item writes its own lane.
    unsafe { cx.arg_out().set([lane], x + f32::from(u8::from(total.to_bits() == u32::MAX))) };
}

/// The sum of C chains' values after `steps` multiply-adds among them, each
/// `chain * x + x`, a chain seeded `x` plus its index.
#[inline(always)]
fn chains<L: Isa, const C: usize>(x: f32, steps: usize) -> f32 {
    let rounds = steps / (L::TIER.vector_bytes() / 4 * C * ROUNDS);
    // SAFETY: each path's features are the tier's: its vector width says
    // which.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match L::TIER.vector_bytes() {
            64 => x86::wide::<C>(x, rounds),
            32 => x86::avx2::<C>(x, rounds),
            _ => x86::sse::<C>(x, rounds),
        }
    }
    // SAFETY: every aarch64 tier has NEON.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon::fmas::<C>(x, rounds)
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::ROUNDS;
    use std::arch::x86_64::*;

    /// AVX-512's fused multiply-adds on 512-bit registers.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn wide<const C: usize>(x: f32, rounds: usize) -> f32 {
        let xs = _mm512_set1_ps(x);
        let mut chains: [__m512; C] = std::array::from_fn(|k| _mm512_set1_ps(x + k as f32));
        for _ in 0..rounds {
            for _ in 0..ROUNDS {
                for k in 0..C {
                    chains[k] = _mm512_fmadd_ps(chains[k], xs, xs);
                }
            }
        }
        chains.iter().map(|chain| _mm512_reduce_add_ps(*chain)).sum()
    }

    /// AVX2's fused multiply-adds on 256-bit registers.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn avx2<const C: usize>(x: f32, rounds: usize) -> f32 {
        let xs = _mm256_set1_ps(x);
        let mut chains: [__m256; C] = std::array::from_fn(|k| _mm256_set1_ps(x + k as f32));
        for _ in 0..rounds {
            for _ in 0..ROUNDS {
                for k in 0..C {
                    chains[k] = _mm256_fmadd_ps(chains[k], xs, xs);
                }
            }
        }
        chains.iter().map(|chain| std::mem::transmute::<__m256, [f32; 8]>(*chain).iter().sum::<f32>()).sum()
    }

    /// SSE's multiplies and adds on 128-bit registers (no FMA).
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sse<const C: usize>(x: f32, rounds: usize) -> f32 {
        let xs = _mm_set1_ps(x);
        let mut chains: [__m128; C] = std::array::from_fn(|k| _mm_set1_ps(x + k as f32));
        for _ in 0..rounds {
            for _ in 0..ROUNDS {
                for k in 0..C {
                    chains[k] = _mm_add_ps(_mm_mul_ps(chains[k], xs), xs);
                }
            }
        }
        chains.iter().map(|chain| std::mem::transmute::<__m128, [f32; 4]>(*chain).iter().sum::<f32>()).sum()
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::ROUNDS;
    use std::arch::aarch64::*;

    /// NEON's fused multiply-adds on 128-bit registers.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn fmas<const C: usize>(x: f32, rounds: usize) -> f32 {
        let xs = vdupq_n_f32(x);
        let mut chains: [float32x4_t; C] = std::array::from_fn(|k| vdupq_n_f32(x + k as f32));
        for _ in 0..rounds {
            for _ in 0..ROUNDS {
                for k in 0..C {
                    chains[k] = vfmaq_f32(xs, chains[k], xs);
                }
            }
        }
        chains.iter().map(|chain| vaddvq_f32(*chain)).sum()
    }
}
