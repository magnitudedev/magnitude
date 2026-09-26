//! Fixed-width construction of scalar semantics. No host floating arithmetic is
//! used here: even checker-time quantization evaluates the generated bit graph.
use super::*;
use ast::{BinaryOp as SourceBinary, UnaryOp as SourceUnary};
use num_bigint::{BigInt, BigUint, Sign};

type Wide = Vec<V>;

impl Recipe {
    fn add(&self, a: V, b: V) -> V {
        self.word(WordOp::Add, a, b)
    }
    fn sub(&self, a: V, b: V) -> V {
        self.word(WordOp::Sub, a, b)
    }
    fn mask(&self, a: V, mask: u32) -> V {
        self.word(WordOp::And, a, self.u(mask))
    }
    fn xor(&self, a: V, b: V) -> V {
        self.word(WordOp::Xor, a, b)
    }
    fn union(&self, a: V, b: V) -> V {
        self.word(WordOp::Or, a, b)
    }
    fn shl(&self, a: V, n: u32) -> V {
        assert!(n < 32);
        self.word(WordOp::Shl, a, self.u(n))
    }
    fn shr(&self, a: V, n: u32) -> V {
        assert!(n < 32);
        self.word(WordOp::Shr, a, self.u(n))
    }
    fn eq(&self, a: V, b: V) -> V {
        self.word_cmp(CmpOp::Eq, a, b)
    }
    fn lt(&self, a: V, b: V) -> V {
        self.word_cmp(CmpOp::Lt, a, b)
    }
    fn nonzero(&self, a: V) -> V {
        self.word_cmp(CmpOp::Ne, a, self.u(0))
    }
    fn unsigned(&self, a: V) -> V {
        self.select(a, self.u(1), self.u(0))
    }
    fn signed_lt(&self, a: V, b: V) -> V {
        self.lt(
            self.xor(a, self.u(0x8000_0000)),
            self.xor(b, self.u(0x8000_0000)),
        )
    }
    fn max_signed(&self, a: V, b: V) -> V {
        self.select(self.signed_lt(a, b), b, a)
    }
    fn sign(&self, x: V, bit: u32) -> V {
        self.nonzero(self.mask(x, 1 << bit))
    }
    fn negate_word(&self, x: V) -> V {
        self.sub(self.u(0), x)
    }
    fn signed_magnitude(&self, x: V) -> (V, V) {
        let sign = self.sign(x, 31);
        (sign, self.select(sign, self.negate_word(x), x))
    }
}

fn wide(b: &Recipe, x: V, n: usize) -> Wide {
    assert!(n > 0);
    let mut v = vec![b.u(0); n];
    v[0] = x;
    v
}
fn wide_nonzero(b: &Recipe, a: &[V]) -> V {
    b.nonzero(a.iter().copied().fold(b.u(0), |acc, x| b.union(acc, x)))
}
fn wide_select(b: &Recipe, p: V, a: &[V], c: &[V]) -> Wide {
    assert_eq!(a.len(), c.len());
    a.iter().zip(c).map(|(&x, &y)| b.select(p, x, y)).collect()
}
fn wide_lt(b: &Recipe, a: &[V], c: &[V]) -> V {
    assert_eq!(a.len(), c.len());
    a.iter().zip(c).fold(b.boolean(false), |less, (&x, &y)| {
        b.or(b.lt(x, y), b.and(b.eq(x, y), less))
    })
}
fn wide_add(b: &Recipe, a: &[V], c: &[V]) -> Wide {
    assert_eq!(a.len(), c.len());
    let mut carry = b.u(0);
    a.iter()
        .zip(c)
        .map(|(&x, &y)| {
            let s = b.add(x, y);
            let t = b.add(s, carry);
            carry = b.unsigned(b.or(b.lt(s, x), b.lt(t, s)));
            t
        })
        .collect()
}
fn wide_sub(b: &Recipe, a: &[V], c: &[V]) -> Wide {
    assert_eq!(a.len(), c.len());
    let mut borrow = b.u(0);
    a.iter()
        .zip(c)
        .map(|(&x, &y)| {
            let s = b.sub(x, y);
            let t = b.sub(s, borrow);
            borrow = b.unsigned(b.or(b.lt(x, y), b.lt(s, borrow)));
            t
        })
        .collect()
}
/// Constant shifts can span any number of limbs. There is never a shift by 32.
fn wide_shl_const(b: &Recipe, a: &[V], count: usize) -> Wide {
    let (words, bits) = (count / 32, (count % 32) as u32);
    (0..a.len())
        .map(|i| {
            if i < words {
                return b.u(0);
            }
            let j = i - words;
            let low = b.shl(a[j], bits);
            if bits != 0 && j > 0 {
                b.union(low, b.shr(a[j - 1], 32 - bits))
            } else {
                low
            }
        })
        .collect()
}
fn wide_shr_const(b: &Recipe, a: &[V], count: usize) -> Wide {
    let (words, bits) = (count / 32, (count % 32) as u32);
    (0..a.len())
        .map(|i| {
            let j = i + words;
            if j >= a.len() {
                return b.u(0);
            }
            let low = b.shr(a[j], bits);
            if bits != 0 && j + 1 < a.len() {
                b.union(low, b.shl(a[j + 1], 32 - bits))
            } else {
                low
            }
        })
        .collect()
}
fn wide_shift(b: &Recipe, a: &[V], count: V, left: bool) -> Wide {
    let mut result = a.to_vec();
    let capacity = a.len() * 32;
    for step in (0..usize::BITS)
        .map(|bit| 1usize << bit)
        .take_while(|&x| x < capacity)
    {
        let shifted = if left {
            wide_shl_const(b, &result, step)
        } else {
            wide_shr_const(b, &result, step)
        };
        let selected = b.nonzero(b.mask(count, step as u32));
        result = wide_select(b, selected, &shifted, &result);
    }
    let in_range = b.lt(count, b.u(capacity as u32));
    wide_select(b, in_range, &result, &vec![b.u(0); a.len()])
}
/// Exact bit length, with a total result of zero for zero.
fn word_bit_length(b: &Recipe, x: V) -> V {
    let mut value = x;
    let mut n = b.u(0);
    for step in [16, 8, 4, 2, 1] {
        let high = b.shr(value, step);
        let has_high = b.nonzero(high);
        n = b.add(n, b.select(has_high, b.u(step), b.u(0)));
        value = b.select(has_high, high, value);
    }
    b.add(n, b.unsigned(b.nonzero(value)))
}
fn wide_bit_length(b: &Recipe, a: &[V]) -> V {
    a.iter().enumerate().fold(b.u(0), |n, (i, &word)| {
        b.select(
            b.nonzero(word),
            b.add(b.u((i * 32) as u32), word_bit_length(b, word)),
            n,
        )
    })
}
fn wide_bit(b: &Recipe, a: &[V], bit: usize) -> V {
    b.mask(b.shr(a[bit / 32], (bit % 32) as u32), 1)
}
/// Retain the high carry of remainder<<1 by adding an explicit limb. The
/// subtract decision therefore never mistakes a wrapped remainder for a small one.
fn wide_divide(b: &Recipe, n: &[V], d: &[V]) -> (Wide, Wide) {
    assert_eq!(n.len(), d.len());
    let width = n.len();
    let mut divisor = d.to_vec();
    divisor.push(b.u(0));
    let mut rem = vec![b.u(0); width + 1];
    let mut quotient = vec![b.u(0); width];
    for i in (0..width * 32).rev() {
        rem = wide_shl_const(b, &rem, 1);
        rem[0] = b.union(rem[0], wide_bit(b, n, i));
        let fits = b.not(wide_lt(b, &rem, &divisor));
        rem = wide_select(b, fits, &wide_sub(b, &rem, &divisor), &rem);
        quotient[i / 32] = b.union(quotient[i / 32], b.select(fits, b.u(1 << (i % 32)), b.u(0)));
    }
    rem.truncate(width);
    (quotient, rem)
}
fn multiply_words(b: &Recipe, a: V, c: V, bits: u32) -> Wide {
    wide_multiply(b, &[a], &[c], bits as usize)
}
/// Exact product of limb vectors, where only the low `bits` of `c` may be set.
/// A constant multiplier contributes only its set bits.
fn wide_multiply(b: &Recipe, a: &[V], c: &[V], bits: usize) -> Wide {
    assert!(bits <= c.len() * 32);
    let width = a.len() + c.len();
    let mut left = a.to_vec();
    left.resize(width, b.u(0));
    let mut result = vec![b.u(0); width];
    for bit in 0..bits {
        let set = b.nonzero(b.mask(c[bit / 32], 1 << (bit % 32)));
        let known = b.builder.borrow().constant(set);
        if known == Some(ReferenceScalar::Bool(false)) {
            continue;
        }
        let sum = wide_add(b, &result, &wide_shl_const(b, &left, bit));
        result = if known == Some(ReferenceScalar::Bool(true)) {
            sum
        } else {
            wide_select(b, set, &sum, &result)
        };
    }
    result
}
fn signed_add(b: &Recipe, a: &[V], sa: V, c: &[V], sc: V) -> (Wide, V) {
    let same = b.not(b.or(b.and(sa, b.not(sc)), b.and(b.not(sa), sc)));
    let less = wide_lt(b, a, c);
    let difference = wide_select(b, less, &wide_sub(b, c, a), &wide_sub(b, a, c));
    let magnitude = wide_select(b, same, &wide_add(b, a, c), &difference);
    // Equal-sign zeros retain their sign; cancellation of opposite signs is +0.
    let sign = b.select(
        same,
        sa,
        b.and(wide_nonzero(b, &difference), b.select(less, sc, sa)),
    );
    (magnitude, sign)
}

#[derive(Clone, Copy)]
struct Format {
    fraction: u32,
    emin: i32,
    emax: i32,
    width: u32,
}
impl Format {
    fn of(dtype: DType) -> Self {
        match dtype {
            DType::F16 => Self {
                fraction: 10,
                emin: -14,
                emax: 15,
                width: 16,
            },
            DType::BF16 => Self {
                fraction: 7,
                emin: -126,
                emax: 127,
                width: 16,
            },
            DType::F32 => Self {
                fraction: 23,
                emin: -126,
                emax: 127,
                width: 32,
            },
            _ => panic!("non-floating format"),
        }
    }
    fn bias(self) -> i32 {
        1 - self.emin
    }
    fn quantum(self) -> i32 {
        self.emin - self.fraction as i32
    }
    fn exponent_mask(self) -> u32 {
        ((1 << (self.width - self.fraction - 1)) - 1) << self.fraction
    }
    fn sign_mask(self) -> u32 {
        1 << (self.width - 1)
    }
    fn nan(self) -> u32 {
        self.exponent_mask() | (1 << (self.fraction - 1))
    }
    fn accumulator_words(self, product: bool) -> usize {
        let low = if product {
            2 * self.quantum()
        } else {
            self.quantum()
        };
        let high = if product {
            2 * (self.emax + 1) + 1
        } else {
            self.emax + 2
        };
        ((high - low) as usize).div_ceil(32)
    }
}
struct Float {
    bits: V,
    sign: V,
    magnitude: V,
    exponent: V,
    zero: V,
    infinity: V,
    nan: V,
}
fn decode(b: &Recipe, value: V) -> Float {
    let f = Format::of(value.ty);
    let bits = b.bits(value);
    let fraction = b.mask(bits, (1 << f.fraction) - 1);
    let field = b.shr(b.mask(bits, f.exponent_mask()), f.fraction);
    let subnormal = b.eq(field, b.u(0));
    let exceptional = b.eq(b.mask(bits, f.exponent_mask()), b.u(f.exponent_mask()));
    Float {
        bits,
        sign: b.sign(bits, f.width - 1),
        magnitude: b.select(
            exceptional,
            b.u(1),
            b.select(subnormal, fraction, b.union(fraction, b.u(1 << f.fraction))),
        ),
        exponent: b.select(
            exceptional,
            b.u(0),
            b.select(
                subnormal,
                b.u(f.quantum() as u32),
                b.sub(field, b.u((f.bias() + f.fraction as i32) as u32)),
            ),
        ),
        zero: b.eq(b.mask(bits, !f.sign_mask()), b.u(0)),
        infinity: b.and(exceptional, b.eq(fraction, b.u(0))),
        nan: b.and(exceptional, b.nonzero(fraction)),
    }
}
fn with_sign(b: &Recipe, magnitude: V, sign: V, f: Format) -> V {
    b.union(magnitude, b.select(sign, b.u(f.sign_mask()), b.u(0)))
}
fn signed_infinity(b: &Recipe, sign: V, f: Format) -> V {
    with_sign(b, b.u(f.exponent_mask()), sign, f)
}
fn canonical(b: &Recipe, invalid: V, bits: V, f: Format) -> V {
    b.select(invalid, b.u(f.nan()), bits)
}

/// Nearest-even packing of exact integer magnitude * 2^exponent. Zero and
/// exceptional callers are safe in the eager graph; bit_length(0) is total.
fn pack(b: &Recipe, n: &[V], exponent: V, sign: V, f: Format) -> V {
    let zero = b.not(wide_nonzero(b, n));
    let safe = wide_select(b, zero, &wide(b, b.u(1), n.len()), n);
    let top = b.sub(wide_bit_length(b, &safe), b.u(1));
    let quantum = b.max_signed(
        b.sub(b.add(top, exponent), b.u(f.fraction)),
        b.u(f.quantum() as u32),
    );
    let delta = b.sub(quantum, exponent);
    let right = b.signed_lt(b.u(0), delta);
    let rshift = b.select(right, delta, b.u(0));
    let lshift = b.select(right, b.u(0), b.negate_word(delta));
    let down = wide_shift(b, &safe, rshift, false);
    let up = wide_shift(b, &safe, lshift, true);
    let magnitude = b.select(right, down[0], up[0]);
    let before_guard = b.select(right, b.sub(rshift, b.u(1)), b.u(0));
    let guard_and_below = wide_shift(b, &safe, before_guard, false);
    let guard = b.and(right, b.nonzero(b.mask(guard_and_below[0], 1)));
    let restored = wide_shift(b, &guard_and_below, before_guard, true);
    let sticky = wide_nonzero(
        b,
        &safe
            .iter()
            .zip(&restored)
            .map(|(&a, &c)| b.xor(a, c))
            .collect::<Vec<_>>(),
    );
    let round = b.and(guard, b.or(sticky, b.nonzero(b.mask(magnitude, 1))));
    encode_rounded(
        b,
        b.add(magnitude, b.unsigned(round)),
        quantum,
        sign,
        zero,
        f,
    )
}
fn encode_rounded(b: &Recipe, m: V, quantum: V, sign: V, zero: V, f: Format) -> V {
    let carry = b.word_cmp(CmpOp::Ge, m, b.u(1 << (f.fraction + 1)));
    let m = b.select(carry, b.shr(m, 1), m);
    let quantum = b.add(quantum, b.unsigned(carry));
    let normal = b.word_cmp(CmpOp::Ge, m, b.u(1 << f.fraction));
    let field = b.add(quantum, b.u((f.fraction as i32 + f.bias()) as u32));
    let body = b.select(
        normal,
        b.union(b.shl(field, f.fraction), b.mask(m, (1 << f.fraction) - 1)),
        m,
    );
    let overflow = b.signed_lt(b.u(f.emax as u32), b.add(quantum, b.u(f.fraction)));
    let body = b.select(overflow, b.u(f.exponent_mask()), body);
    with_sign(b, b.select(zero, b.u(0), body), sign, f)
}

fn float_add(b: &Recipe, a: V, c: V, subtract: bool) -> V {
    let dtype = a.ty;
    let f = Format::of(dtype);
    let a = decode(b, a);
    let mut c = decode(b, c);
    if subtract {
        c.sign = b.not(c.sign);
    }
    let n = f.accumulator_words(false);
    let quantum = f.quantum();
    let left = wide_shift(
        b,
        &wide(b, a.magnitude, n),
        b.sub(a.exponent, b.u(quantum as u32)),
        true,
    );
    let right = wide_shift(
        b,
        &wide(b, c.magnitude, n),
        b.sub(c.exponent, b.u(quantum as u32)),
        true,
    );
    let (magnitude, sign) = signed_add(b, &left, a.sign, &right, c.sign);
    let result = pack(b, &magnitude, b.u(quantum as u32), sign, f);
    let opposing = b.and(
        b.and(a.infinity, c.infinity),
        b.not(b.eq(b.unsigned(a.sign), b.unsigned(c.sign))),
    );
    let result = b.select(c.infinity, signed_infinity(b, c.sign, f), result);
    let result = b.select(a.infinity, signed_infinity(b, a.sign, f), result);
    b.typed(
        canonical(b, b.or(b.or(a.nan, c.nan), opposing), result, f),
        dtype,
    )
}
fn float_multiply(b: &Recipe, a: V, c: V, addend: Option<V>) -> V {
    let dtype = a.ty;
    let f = Format::of(dtype);
    let a = decode(b, a);
    let c = decode(b, c);
    let sign = b.not(b.eq(b.unsigned(a.sign), b.unsigned(c.sign)));
    let product = multiply_words(b, a.magnitude, c.magnitude, f.fraction + 1);
    let exponent = b.add(a.exponent, c.exponent);
    let invalid = b.or(
        b.or(a.nan, c.nan),
        b.or(b.and(a.zero, c.infinity), b.and(a.infinity, c.zero)),
    );
    let infinity = b.or(a.infinity, c.infinity);
    let result = if let Some(addend) = addend {
        let z = decode(b, addend);
        let n = f.accumulator_words(true);
        let quantum = 2 * f.quantum();
        let mut product = product;
        product.resize(n, b.u(0));
        let left = wide_shift(b, &product, b.sub(exponent, b.u(quantum as u32)), true);
        let right = wide_shift(
            b,
            &wide(b, z.magnitude, n),
            b.sub(z.exponent, b.u(quantum as u32)),
            true,
        );
        let (sum, sum_sign) = signed_add(b, &left, sign, &right, z.sign);
        let result = pack(b, &sum, b.u(quantum as u32), sum_sign, f);
        let result = b.select(z.infinity, signed_infinity(b, z.sign, f), result);
        let result = b.select(infinity, signed_infinity(b, sign, f), result);
        let opposing = b.and(
            b.and(infinity, z.infinity),
            b.not(b.eq(b.unsigned(sign), b.unsigned(z.sign))),
        );
        canonical(b, b.or(b.or(invalid, z.nan), opposing), result, f)
    } else {
        let result = pack(b, &product, exponent, sign, f);
        canonical(
            b,
            invalid,
            b.select(infinity, signed_infinity(b, sign, f), result),
            f,
        )
    };
    b.typed(result, dtype)
}

fn float_divide(b: &Recipe, a: V, c: V, remainder: bool) -> V {
    let dtype = a.ty;
    let f = Format::of(dtype);
    let a = decode(b, a);
    let c = decode(b, c);
    let invalid = if remainder {
        b.or(a.infinity, c.zero)
    } else {
        b.or(b.and(a.zero, c.zero), b.and(a.infinity, c.infinity))
    };
    let invalid = b.or(b.or(a.nan, c.nan), invalid);
    let divisor = b.select(c.zero, b.u(1), c.magnitude);
    let result = if remainder {
        let n = f.accumulator_words(false);
        let quantum = f.quantum();
        let numerator = wide_shift(
            b,
            &wide(b, a.magnitude, n),
            b.sub(a.exponent, b.u(quantum as u32)),
            true,
        );
        let denominator = wide_shift(
            b,
            &wide(b, divisor, n),
            b.sub(c.exponent, b.u(quantum as u32)),
            true,
        );
        let denominator = wide_select(
            b,
            wide_nonzero(b, &denominator),
            &denominator,
            &wide(b, b.u(1), n),
        );
        let (_, rem) = wide_divide(b, &numerator, &denominator);
        b.select(
            c.infinity,
            a.bits,
            pack(b, &rem, b.u(quantum as u32), a.sign, f),
        )
    } else {
        let numerator = b.select(a.zero, b.u(1), a.magnitude);
        let exponent = b.sub(a.exponent, c.exponent);
        let delta = b.sub(word_bit_length(b, numerator), word_bit_length(b, divisor));
        let negative = b.signed_lt(delta, b.u(0));
        let left = wide_shift(
            b,
            &wide(b, numerator, 2),
            b.select(negative, b.negate_word(delta), b.u(0)),
            true,
        );
        let right = wide_shift(
            b,
            &wide(b, divisor, 2),
            b.select(negative, b.u(0), delta),
            true,
        );
        let top = b.sub(
            b.add(delta, exponent),
            b.unsigned(wide_lt(b, &left, &right)),
        );
        let quantum = b.max_signed(b.sub(top, b.u(f.fraction)), b.u(f.quantum() as u32));
        let shift = b.sub(exponent, quantum);
        let negative = b.signed_lt(shift, b.u(0));
        // Four limbs cover the full finite F32/BF16 ratio, including a shifted
        // denominator for gradual underflow: F32 needs 24+104=128 bits,
        // BF16 needs 8+120=128; F16 needs fewer. Exceptional operands are sanitized.
        let numerator = wide_shift(
            b,
            &wide(b, numerator, 4),
            b.select(negative, b.u(0), shift),
            true,
        );
        let denominator = wide_shift(
            b,
            &wide(b, divisor, 4),
            b.select(negative, b.negate_word(shift), b.u(0)),
            true,
        );
        let denominator = wide_select(
            b,
            wide_nonzero(b, &denominator),
            &denominator,
            &wide(b, b.u(1), 4),
        );
        let (quotient, rem) = wide_divide(b, &numerator, &denominator);
        let mut rem = rem;
        rem.push(b.u(0));
        let twice = wide_shl_const(b, &rem, 1);
        let mut denominator = denominator;
        denominator.push(b.u(0));
        let above = wide_lt(b, &denominator, &twice);
        let tie = b.and(b.not(above), b.not(wide_lt(b, &twice, &denominator)));
        let odd = b.nonzero(b.mask(quotient[0], 1));
        let rounded = b.add(quotient[0], b.unsigned(b.or(above, b.and(tie, odd))));
        let sign = b.not(b.eq(b.unsigned(a.sign), b.unsigned(c.sign)));
        let result = encode_rounded(b, rounded, quantum, sign, b.boolean(false), f);
        let result = b.select(
            b.or(a.zero, c.infinity),
            with_sign(b, b.u(0), sign, f),
            result,
        );
        b.select(
            b.or(a.infinity, c.zero),
            signed_infinity(b, sign, f),
            result,
        )
    };
    b.typed(canonical(b, invalid, result, f), dtype)
}

/// IEEE squareRoot: the exact integer root of the scaled significand, rounded once
/// with the remainder as sticky bit. Subnormal operands and results are exact.
fn square_root(b: &Recipe, a: V) -> V {
    let dtype = a.ty;
    let f = Format::of(dtype);
    let x = decode(b, a);
    // Scale the significand to 2(p+2) or 2(p+2)-1 bits with an even exponent, so
    // its root has p+2 bits: the precision, a round bit and one more bit.
    let root_bits = f.fraction + 3;
    let radicand_bits = 2 * root_bits;
    let scale = b.sub(b.u(radicand_bits), word_bit_length(b, x.magnitude));
    let odd = b.nonzero(b.mask(b.sub(x.exponent, scale), 1));
    let scale = b.sub(scale, b.unsigned(odd));
    let radicand = wide_shift(
        b,
        &wide(b, x.magnitude, (radicand_bits as usize).div_ceil(32)),
        scale,
        true,
    );
    // Digit-by-digit root: the remainder stays below 2*root+1 < 2^(p+3).
    let mut root = b.u(0);
    let mut remainder = b.u(0);
    for digit in (0..root_bits as usize).rev() {
        let pair = b.mask(b.shr(radicand[2 * digit / 32], (2 * digit % 32) as u32), 3);
        remainder = b.union(b.shl(remainder, 2), pair);
        let trial = b.union(b.shl(root, 2), b.u(1));
        let fits = b.not(b.lt(remainder, trial));
        remainder = b.select(fits, b.sub(remainder, trial), remainder);
        root = b.union(b.shl(root, 1), b.unsigned(fits));
    }
    let even = b.sub(x.exponent, scale);
    let half = b.union(b.shr(even, 1), b.mask(even, 0x8000_0000));
    let rounded = pack(
        b,
        &[b.union(b.shl(root, 1), b.unsigned(b.nonzero(remainder)))],
        b.sub(half, b.u(1)),
        b.boolean(false),
        f,
    );
    let result = b.select(b.or(x.zero, x.infinity), x.bits, rounded);
    let invalid = b.or(x.nan, b.and(x.sign, b.not(x.zero)));
    b.typed(canonical(b, invalid, result, f), dtype)
}
/// `rsqrt(x)` is `1 / sqrt(x)` in the operand dtype: two roundings.
fn reciprocal_square_root(b: &Recipe, a: V) -> V {
    let f = Format::of(a.ty);
    let one = b.typed(b.u((f.bias() as u32) << f.fraction), a.ty);
    float_divide(b, one, square_root(b, a), false)
}

// floor((2/pi) * 2^256), little-endian 32-bit limbs.
const TWO_OVER_PI: [u32; 8] = [
    0xdebb_c561,
    0xfe51_63ab,
    0x3c43_9041,
    0xdb62_9599,
    0xf534_ddc0,
    0xfc27_57d1,
    0x4e44_1529,
    0xa2f9_836e,
];
// round((pi/2) * 2^62)
const HALF_PI_Q62: u64 = 0x6487_ed51_10b4_611a;
const ONE_Q63: u64 = 1 << 63;
// FreeBSD __kernel_sindf and __kernel_cosdf coefficient magnitudes in Q63, from
// the msun double literals (rounded to nearest where they have finer bits).
// sin r = r(1 - z(S1 - z(S2 - z(S3 - z S4)))) and
// cos r = 1 - z(C0 - z(C1 - z(C2 - z C3))), with z = r^2 and alternating signs.
const SIN_Q63: [u64; 4] = [
    0x15_5555_54cb_ac77 << 8,          // S1 = -0x15555554cbac77.0p-55
    0x11_1110_896e_fbb2 << 4,          // S2 =  0x111110896efbb2.0p-59
    0x1a_00f9_e2ca_e774 >> 2,          // S3 = -0x1a00f9e2cae774.0p-65
    (0x16_cd87_8c3b_46a7 + 0x80) >> 8, // S4 = 0x16cd878c3b46a7.0p-71
];
const COS_Q63: [u64; 4] = [
    0x1f_ffff_fd0c_5e81 << 9, // C0 = -0x1ffffffd0c5e81.0p-54
    0x15_5553_e105_3a42 << 6, // C1 =  0x155553e1053a42.0p-57
    0x16_c087_e80f_1e27 << 1, // C2 = -0x16c087e80f1e27.0p-62
    0x19_9342_e0ee_5069 >> 5, // C3 =  0x199342e0ee5069.0p-68
];

fn wide_constant(b: &Recipe, x: u64) -> Wide {
    vec![b.u(x as u32), b.u((x >> 32) as u32)]
}
/// Unsigned Q63 product of two 64-bit fixed-point values, truncated.
fn fixed_multiply(b: &Recipe, a: &[V], c: &[V]) -> Wide {
    let product = wide_multiply(b, a, c, 64);
    wide_shr_const(b, &product, 63)[..2].to_vec()
}
/// Horner evaluation of `ONE - z(k0 - z(k1 - z(k2 - z k3)))` on magnitudes. Each
/// subtraction is positive for |r| <= pi/4, which fixes the alternating signs.
fn fixed_polynomial(b: &Recipe, z: &[V], coefficients: [u64; 4]) -> Wide {
    let mut value = wide_constant(b, coefficients[3]);
    for &k in coefficients[..3].iter().rev().chain([ONE_Q63].iter()) {
        value = wide_sub(b, &wide_constant(b, k), &fixed_multiply(b, z, &value));
    }
    value
}
fn boolean_xor(b: &Recipe, a: V, c: V) -> V {
    b.not(b.eq(b.unsigned(a), b.unsigned(c)))
}

/// F32 sine or cosine. The argument is reduced by Payne-Hanek to a quadrant and a
/// fraction of pi/2 normalized to 64 significant bits, multiplied by pi/2 in Q62,
/// and the FreeBSD kernels are evaluated in 64-bit fixed point. The only
/// rounding is the final nearest-even packing to F32.
pub(super) fn sine_or_cosine(b: &Recipe, a: V, cosine: bool) -> V {
    assert_eq!(a.ty, DType::F32);
    let f = Format::of(DType::F32);
    let x = decode(b, a);
    let finite = b.not(b.or(x.nan, x.infinity));
    // |x| <= pi/4 (0x3f490fda) is its own reduced argument.
    let direct = b.lt(b.mask(x.bits, 0x7fff_ffff), b.u(0x3f49_0fdb));

    // |x| * 2/pi = m * C * 2^(E-256). Shifting by E+24 >= 0 puts the binary point
    // at bit 280; C's truncation error stays below 2^-128 there.
    let product = wide_multiply(b, &TWO_OVER_PI.map(|w| b.u(w)), &[x.magnitude], 24);
    let aligned = wide_shift(b, &product, b.add(x.exponent, b.u(24)), true);
    let quotient = b.mask(b.shr(aligned[8], 24), 3);
    let rounds_up = b.nonzero(b.mask(aligned[8], 1 << 23));
    let mut fraction = aligned.clone();
    fraction[8] = b.mask(aligned[8], 0x00ff_ffff);
    let mut one = vec![b.u(0); 9];
    one[8] = b.u(1 << 24);
    let fraction = wide_select(b, rounds_up, &wide_sub(b, &one, &fraction), &fraction);
    let quadrant = b.mask(b.add(quotient, b.unsigned(rounds_up)), 3);
    // Over every F32 above pi/4 the fraction is at least 2^-30 (least at
    // 16367173 * 2^72, by exhaustive search), so the 64 bits from its leading one
    // lie far above the constant's truncation error.
    let length = wide_bit_length(b, &fraction);
    let normalized = wide_shift(b, &fraction, b.sub(b.u(288), length), true);
    let reduced = wide_multiply(b, &normalized[7..], &wide_constant(b, HALF_PI_Q62), 64);
    let reduced_exponent = b.sub(length, b.u(342));

    // r = R * 2^e with R in [2^61, 2^63).
    let direct_shift = b.sub(b.u(63), word_bit_length(b, x.magnitude));
    let direct_r = wide_shift(b, &wide(b, x.magnitude, 2), direct_shift, true);
    let r = wide_select(b, direct, &direct_r, &reduced[2..]);
    let exponent = b.select(direct, b.sub(x.exponent, direct_shift), reduced_exponent);
    let quadrant = b.select(direct, b.u(0), quadrant);
    let r_negative = b.and(b.not(direct), rounds_up);

    // z = r^2 in Q63.
    let square = wide_multiply(b, &r, &r, 64);
    let z_shift = b.negate_word(b.add(b.add(exponent, exponent), b.u(63)));
    let z = wide_shift(b, &square, z_shift, false)[..2].to_vec();
    let sine = wide_multiply(b, &r, &fixed_polynomial(b, &z, SIN_Q63), 64);
    let mut cosine_value = fixed_polynomial(b, &z, COS_Q63);
    cosine_value.resize(4, b.u(0));

    let odd = b.nonzero(b.mask(quadrant, 1));
    let high = b.nonzero(b.mask(quadrant, 2));
    let (uses_sine, negative) = if cosine {
        (
            odd,
            b.select(odd, boolean_xor(b, r_negative, b.not(high)), high),
        )
    } else {
        (
            b.not(odd),
            boolean_xor(
                b,
                x.sign,
                b.select(odd, high, boolean_xor(b, r_negative, high)),
            ),
        )
    };
    let magnitude = wide_select(b, uses_sine, &sine, &cosine_value);
    let scale = b.select(uses_sine, b.sub(exponent, b.u(63)), b.u((-63i32) as u32));
    let result = pack(b, &magnitude, scale, negative, f);
    b.typed(canonical(b, b.not(finite), result, f), DType::F32)
}

pub(super) fn compare(b: &Recipe, op: CmpOp, a: V, c: V) -> V {
    assert_eq!(a.ty, c.ty);
    let dtype = a.ty;
    let a_bits = b.bits(a);
    let c_bits = b.bits(c);
    if dtype.is_float() {
        let f = Format::of(dtype);
        let da = decode(b, a);
        let dc = decode(b, c);
        let nan = b.or(da.nan, dc.nan);
        let equal = b.or(b.eq(a_bits, c_bits), b.and(da.zero, dc.zero));
        let full_mask = if f.width == 32 {
            u32::MAX
        } else {
            (1 << f.width) - 1
        };
        let key = |bits, sign| b.xor(bits, b.select(sign, b.u(full_mask), b.u(f.sign_mask())));
        let less = b.and(
            b.not(equal),
            b.lt(key(a_bits, da.sign), key(c_bits, dc.sign)),
        );
        let ordered = match op {
            CmpOp::Eq => equal,
            CmpOp::Ne => b.not(equal),
            CmpOp::Lt => less,
            CmpOp::Le => b.or(less, equal),
            CmpOp::Gt => b.not(b.or(less, equal)),
            CmpOp::Ge => b.not(less),
        };
        if op == CmpOp::Ne {
            b.or(nan, ordered)
        } else {
            b.and(b.not(nan), ordered)
        }
    } else {
        let (a, c) = if dtype == DType::I32 {
            (
                b.xor(a_bits, b.u(0x8000_0000)),
                b.xor(c_bits, b.u(0x8000_0000)),
            )
        } else {
            (a_bits, c_bits)
        };
        b.word_cmp(op, a, c)
    }
}
fn minmax(b: &Recipe, a: V, c: V, max: bool) -> V {
    let dtype = a.ty;
    let selected = b.select(
        compare(b, if max { CmpOp::Gt } else { CmpOp::Lt }, a, c),
        a,
        c,
    );
    if !dtype.is_float() {
        return selected;
    }
    let f = Format::of(dtype);
    let da = decode(b, a);
    let dc = decode(b, c);
    let zero_sign = if max {
        b.and(da.sign, dc.sign)
    } else {
        b.or(da.sign, dc.sign)
    };
    let result = b.select(
        b.and(da.zero, dc.zero),
        with_sign(b, b.u(0), zero_sign, f),
        b.bits(selected),
    );
    let result = b.select(da.nan, dc.bits, b.select(dc.nan, da.bits, result));
    b.typed(canonical(b, b.and(da.nan, dc.nan), result, f), dtype)
}

pub(super) fn negate(b: &Recipe, a: V) -> V {
    let dtype = a.ty;
    let bits = b.bits(a);
    let result = if dtype.is_float() {
        b.xor(bits, b.u(Format::of(dtype).sign_mask()))
    } else {
        assert!(dtype.is_int());
        b.negate_word(bits)
    };
    b.typed(result, dtype)
}
pub(super) fn absolute(b: &Recipe, a: V) -> V {
    let dtype = a.ty;
    let bits = b.bits(a);
    let result = if dtype.is_float() {
        b.mask(bits, !Format::of(dtype).sign_mask())
    } else if dtype == DType::I32 {
        b.signed_magnitude(bits).1
    } else {
        assert_eq!(dtype, DType::U32);
        bits
    };
    b.typed(result, dtype)
}
fn integer_binary(b: &Recipe, op: SourceBinary, a: V, c: V) -> V {
    let dtype = a.ty;
    let x = b.bits(a);
    let y = b.bits(c);
    let result = match op {
        SourceBinary::Add => b.add(x, y),
        SourceBinary::Sub => b.sub(x, y),
        SourceBinary::Mul => multiply_words(b, x, y, 32)[0],
        SourceBinary::BitAnd => b.word(WordOp::And, x, y),
        SourceBinary::BitOr => b.union(x, y),
        SourceBinary::BitXor => b.xor(x, y),
        SourceBinary::Div | SourceBinary::Rem => {
            let overflow = if dtype == DType::I32 {
                b.and(b.eq(x, b.u(0x8000_0000)), b.eq(y, b.u(u32::MAX)))
            } else {
                b.boolean(false)
            };
            let zero = b.eq(y, b.u(0));
            let fail = b.or(zero, overflow);
            b.fail_if(zero, ScalarFailure::IntegerDivisionByZero);
            if dtype == DType::I32 {
                b.fail_if(overflow, ScalarFailure::SignedDivisionOverflow);
            }
            let (sx, mx) = if dtype == DType::I32 {
                b.signed_magnitude(x)
            } else {
                (b.boolean(false), x)
            };
            let (sy, my) = if dtype == DType::I32 {
                b.signed_magnitude(y)
            } else {
                (b.boolean(false), y)
            };
            let my = b.select(fail, b.u(1), my);
            let (quotient, rem) = wide_divide(b, &[mx], &[my]);
            let opposite = b.not(b.eq(b.unsigned(sx), b.unsigned(sy)));
            let quotient = b.select(opposite, b.negate_word(quotient[0]), quotient[0]);
            let adjust = b.and(sx, b.nonzero(rem[0]));
            if op == SourceBinary::Rem {
                b.select(adjust, b.sub(my, rem[0]), rem[0])
            } else {
                let adjusted = b.select(sy, b.add(quotient, b.u(1)), b.sub(quotient, b.u(1)));
                b.select(adjust, adjusted, quotient)
            }
        }
        SourceBinary::Shl | SourceBinary::Shr => {
            let valid = b.lt(y, b.u(32));
            b.fail_if(b.not(valid), ScalarFailure::ShiftCount);
            let count = b.select(valid, y, b.u(0));
            if op == SourceBinary::Shl {
                b.word(WordOp::Shl, x, count)
            } else {
                let shifted = b.word(WordOp::Shr, x, count);
                if dtype == DType::I32 {
                    // All executed counts are 0..31, including the unused sign-fill arm.
                    let nonzero = b.nonzero(count);
                    let fill_count = b.mask(b.sub(b.u(32), count), 31);
                    let fill = b.word(WordOp::Shl, b.u(u32::MAX), fill_count);
                    b.union(
                        shifted,
                        b.select(b.and(b.sign(x, 31), nonzero), fill, b.u(0)),
                    )
                } else {
                    shifted
                }
            }
        }
        _ => unreachable!("non-integer arithmetic operation"),
    };
    b.typed(result, dtype)
}
pub(super) fn binary(b: &Recipe, op: SourceBinary, a: V, c: V) -> V {
    // The source checker inserts promotions before this scalar boundary.
    assert!(a.ty == c.ty || matches!(op, SourceBinary::Shl | SourceBinary::Shr));
    let cmp = match op {
        SourceBinary::Eq => Some(CmpOp::Eq),
        SourceBinary::Ne => Some(CmpOp::Ne),
        SourceBinary::Lt => Some(CmpOp::Lt),
        SourceBinary::Le => Some(CmpOp::Le),
        SourceBinary::Gt => Some(CmpOp::Gt),
        SourceBinary::Ge => Some(CmpOp::Ge),
        _ => None,
    };
    if let Some(cmp) = cmp {
        return compare(b, cmp, a, c);
    }
    let dtype = a.ty;
    if dtype == DType::Bool {
        return match op {
            SourceBinary::And | SourceBinary::BitAnd => b.and(a, c),
            SourceBinary::Or | SourceBinary::BitOr => b.or(a, c),
            SourceBinary::BitXor => b.not(b.eq(b.bits(a), b.bits(c))),
            _ => panic!("non-Boolean operation"),
        };
    }
    if dtype.is_int() {
        return integer_binary(b, op, a, c);
    }
    match op {
        SourceBinary::Add => float_add(b, a, c, false),
        SourceBinary::Sub => float_add(b, a, c, true),
        SourceBinary::Mul => float_multiply(b, a, c, None),
        SourceBinary::Div => float_divide(b, a, c, false),
        SourceBinary::Rem => float_divide(b, a, c, true),
        _ => panic!("non-floating operation"),
    }
}

pub(super) fn cast(b: &Recipe, a: V, to: DType) -> V {
    let from = a.ty;
    if from == to {
        return a;
    }
    let bits = b.bits(a);
    if to == DType::Bool {
        let nonzero = if from.is_float() {
            b.nonzero(b.mask(bits, !Format::of(from).sign_mask()))
        } else {
            b.nonzero(bits)
        };
        return nonzero;
    }
    if from == DType::Bool {
        let word = b.unsigned(a);
        return if to.is_float() {
            b.typed(
                pack(b, &[word], b.u(0), b.boolean(false), Format::of(to)),
                to,
            )
        } else {
            b.typed(word, to)
        };
    }
    if from.is_int() && to.is_int() {
        return b.typed(bits, to);
    }
    if from.is_int() {
        let (sign, magnitude) = if from == DType::I32 {
            b.signed_magnitude(bits)
        } else {
            (b.boolean(false), bits)
        };
        return b.typed(pack(b, &[magnitude], b.u(0), sign, Format::of(to)), to);
    }
    let a = decode(b, a);
    if to.is_float() {
        let f = Format::of(to);
        let result = pack(b, &[a.magnitude], a.exponent, a.sign, f);
        let result = b.select(a.infinity, signed_infinity(b, a.sign, f), result);
        return b.typed(canonical(b, a.nan, result, f), to);
    }
    assert!(to.is_int());
    let negative_exponent = b.signed_lt(a.exponent, b.u(0));
    let shifted_left = wide_shift(
        b,
        &[a.magnitude],
        b.select(negative_exponent, b.u(0), a.exponent),
        true,
    )[0];
    let shifted_right = wide_shift(
        b,
        &[a.magnitude],
        b.select(negative_exponent, b.negate_word(a.exponent), b.u(0)),
        false,
    )[0];
    let magnitude = b.select(negative_exponent, shifted_right, shifted_left);
    let bit_length = b.add(word_bit_length(b, a.magnitude), a.exponent);
    let exceeds_word = b.signed_lt(b.u(32), bit_length);
    let limit = if to == DType::I32 {
        b.select(a.sign, b.u(0x8000_0000), b.u(0x7fff_ffff))
    } else {
        b.u(u32::MAX)
    };
    let magnitude = b.select(
        b.or(b.or(exceeds_word, a.infinity), b.lt(limit, magnitude)),
        limit,
        magnitude,
    );
    let result = if to == DType::I32 {
        b.select(a.sign, b.negate_word(magnitude), magnitude)
    } else {
        b.select(a.sign, b.u(0), magnitude)
    };
    b.typed(b.select(a.nan, b.u(0), result), to)
}

/// The word realization of `integer_to_float`: the two's-complement value
/// `high·2^32 + low` as sign and magnitude, packed by the same rounding.
fn word_integer_to_float(b: &Recipe, low: V, high: V, to: DType) -> V {
    let words = [b.bits(low), b.bits(high)];
    let sign = b.sign(words[1], 31);
    let negated = wide_sub(b, &[b.u(0), b.u(0)], &words);
    let magnitude = wide_select(b, sign, &negated, &words);
    let f = Format::of(to);
    b.typed(pack(b, &magnitude, b.u(0), sign, f), to)
}

pub(super) fn operation(b: &Recipe, op: ScalarOp, args: &[V]) -> V {
    // These are construction defects, never invocation rejection: checked
    // source has already established the operation's closed scalar signature.
    let arity = match op {
        ScalarOp::Binary(_) | ScalarOp::IntegerToFloat(_) => 2,
        ScalarOp::Math(op) => op.arity(),
        ScalarOp::Unary(_) | ScalarOp::Cast(_) => 1,
    };
    assert_eq!(args.len(), arity, "scalar recipe operand arity");
    match op {
        ScalarOp::Math(op) => {
            assert!(
                args.iter().all(|a| a.ty == args[0].ty),
                "scalar recipe requires checked promotions"
            );
            assert!(if op.numeric_operands() {
                args[0].ty.is_numeric()
            } else {
                args[0].ty.is_float()
            });
        }
        ScalarOp::Unary(SourceUnary::Not) => assert_eq!(args[0].ty, DType::Bool),
        ScalarOp::Unary(SourceUnary::BitNot) => assert!(args[0].ty.is_int()),
        ScalarOp::Unary(SourceUnary::Neg) => assert!(args[0].ty.is_numeric()),
        ScalarOp::IntegerToFloat(to) => {
            assert!(
                to.is_float(),
                "integer_to_float target must be a float dtype"
            );
            assert!(
                args.iter().all(|a| a.ty == DType::U32),
                "integer_to_float operands are U32 words"
            );
        }
        ScalarOp::Binary(_) | ScalarOp::Cast(_) => {}
    }
    match op {
        ScalarOp::Binary(op) => {
            assert_eq!(args.len(), 2);
            binary(b, op, args[0], args[1])
        }
        ScalarOp::Unary(op) => {
            assert_eq!(args.len(), 1);
            match op {
                SourceUnary::Neg => negate(b, args[0]),
                SourceUnary::Not => b.not(args[0]),
                SourceUnary::BitNot => b.typed(b.xor(b.bits(args[0]), b.u(u32::MAX)), args[0].ty),
            }
        }
        ScalarOp::Cast(to) => {
            assert_eq!(args.len(), 1);
            cast(b, args[0], to)
        }
        ScalarOp::IntegerToFloat(to) => word_integer_to_float(b, args[0], args[1], to),
        ScalarOp::Math(op) => {
            assert_eq!(args.len(), op.arity());
            match op {
                MathOp::Abs => absolute(b, args[0]),
                MathOp::Min => minmax(b, args[0], args[1], false),
                MathOp::Max => minmax(b, args[0], args[1], true),
                MathOp::Fma => float_multiply(b, args[0], args[1], Some(args[2])),
                MathOp::Sqrt => square_root(b, args[0]),
                MathOp::Rsqrt => reciprocal_square_root(b, args[0]),
                _ => transcendental(b, op, args[0]),
            }
        }
    }
}

pub(super) fn literal(dtype: DType, bits: u64) -> ReferenceScalar {
    assert!(dtype.is_float());
    let f = Format::of(dtype);
    let sign = bits >> 63 != 0;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0x7ff {
        return ReferenceScalar::from_bits(
            dtype,
            if fraction != 0 {
                f.nan()
            } else {
                f.exponent_mask() | if sign { f.sign_mask() } else { 0 }
            },
        );
    }
    let magnitude = if exponent == 0 {
        fraction
    } else {
        fraction | (1u64 << 52)
    };
    let exponent = if exponent == 0 {
        -1074
    } else {
        exponent - 1023 - 52
    };
    let b = Recipe::default();
    let words = [b.u(magnitude as u32), b.u((magnitude >> 32) as u32)];
    let bits = pack(&b, &words, b.u(exponent as u32), b.boolean(sign), f);
    let output = b.typed(bits, dtype);
    evaluate(&b.finish(output), &[]).unwrap()
}
pub(super) fn integer_literal(dtype: DType, value: i128) -> ReferenceScalar {
    if !dtype.is_float() {
        return ReferenceScalar::from_bits(dtype, value as u32);
    }
    integer_to_float(dtype, &BigInt::from(value))
}
/// The exact integer rounded once to nearest-even. Every magnitude of at least
/// 2^(emax+2) overflows, so that bound stands for all of them.
pub(super) fn integer_to_float(dtype: DType, value: &BigInt) -> ReferenceScalar {
    let f = Format::of(dtype);
    let bound = BigUint::from(1u32) << (f.emax + 2) as u32;
    let magnitude = value.magnitude().min(&bound);
    let b = Recipe::default();
    let mut words: Vec<V> = magnitude.iter_u32_digits().map(|w| b.u(w)).collect();
    if words.is_empty() {
        words.push(b.u(0));
    }
    let negative = value.sign() == Sign::Minus;
    let bits = pack(&b, &words, b.u(0), b.boolean(negative), f);
    let output = b.typed(bits, dtype);
    evaluate(&b.finish(output), &[]).unwrap()
}
