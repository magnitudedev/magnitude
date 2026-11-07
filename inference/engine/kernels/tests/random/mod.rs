//! The kernel tests' pseudo-random source. A test binary declares this
//! module and uses the draws it needs.
#![allow(dead_code)]

/// A 64-bit LCG (Knuth's MMIX constants) yielding the high `BITS` bits of
/// its state, the bits with the longest period.
pub struct Rng(pub u64);

impl Rng {
    const MULTIPLIER: u64 = 6_364_136_223_846_793_005;
    const INCREMENT: u64 = 1_442_695_040_888_963_407;
    /// The bits a draw holds.
    const BITS: u32 = 31;

    /// The generator for `seed`, one step in, so nearby seeds start apart.
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(Self::MULTIPLIER).wrapping_add(Self::INCREMENT))
    }
    /// Uniform in [0, 2^31).
    pub fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(Self::MULTIPLIER).wrapping_add(Self::INCREMENT);
        (self.0 >> (u64::BITS - Self::BITS)) as u32
    }
    pub fn byte(&mut self) -> u8 {
        self.next() as u8
    }
    /// Uniform in [0, 1): a draw's high 24 bits, f32's precision, so the
    /// quotient is exact.
    pub fn fraction(&mut self) -> f32 {
        (self.next() >> (Self::BITS - f32::MANTISSA_DIGITS)) as f32
            / (1u32 << f32::MANTISSA_DIGITS) as f32
    }
    /// Uniform in [-1, 1).
    pub fn symmetric(&mut self) -> f32 {
        self.fraction() * 2.0 - 1.0
    }
    /// Uniform from `low` to `high`, from all of a draw's bits (their
    /// quotient rounds to f32, so `high` itself can be drawn).
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * (self.next() as f32 / (1u64 << Self::BITS) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    /// The generator whose next draw is `draw`: the state that steps to
    /// `draw` in the high bits. The multiplier is odd, so it has an inverse
    /// mod 2^64; any odd number is its own inverse to 3 bits, and each
    /// Newton step doubles that, so 5 steps reach 96 >= 64.
    fn drawing(draw: u32) -> Rng {
        let mut inverse = Rng::MULTIPLIER;
        for _ in 0..5 {
            inverse = inverse.wrapping_mul(2u64.wrapping_sub(Rng::MULTIPLIER.wrapping_mul(inverse)));
        }
        let stepped = u64::from(draw) << (u64::BITS - Rng::BITS);
        Rng(stepped.wrapping_sub(Rng::INCREMENT).wrapping_mul(inverse))
    }

    /// The draws reach both ends of their ranges and split them at the
    /// middle: `fraction` from 0 to 1 - 2^-24, through 0.5 at the draw's
    /// top bit, and `symmetric` from -1 to 1 - 2^-23, both signs.
    #[test]
    fn draws_cover_their_whole_range() {
        let top = (1u32 << Rng::BITS) - 1;
        let half = 1u32 << (Rng::BITS - 1);
        assert_eq!(drawing(top).next(), top);
        assert_eq!(drawing(0).fraction(), 0.0);
        assert_eq!(drawing(half - 1).fraction(), 0.5 - f32::EPSILON / 2.0);
        assert_eq!(drawing(half).fraction(), 0.5);
        assert_eq!(drawing(top).fraction(), 1.0 - f32::EPSILON / 2.0);
        assert_eq!(drawing(0).symmetric(), -1.0);
        assert_eq!(drawing(half).symmetric(), 0.0);
        assert_eq!(drawing(top).symmetric(), 1.0 - f32::EPSILON);
    }
}
