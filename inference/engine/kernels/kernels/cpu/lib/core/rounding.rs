// Trained block roundings of stored rows on CPU, evaluated exactly as
// `rounding.seismic` writes them: each run of `block` values x becomes
// code(clamp(x / s, -m, m)) * s, s the run's scale from
// a = max(max|x|, amax_floor), the quotient a division. The counterpart of
// `metal/lib/core/rounding.h`.

/// A rounding as the entries take it (`block`, `element`, `scale`,
/// `amax_floor`; `rounding.seismic`'s codes).
#[derive(Clone, Copy, Debug)]
pub struct Rounding {
    block: usize,
    element: Element,
    scale: Scale,
    amax_floor: f32,
}

#[derive(Clone, Copy, Debug)]
enum Element {
    E4M3,
    E2M1,
}

#[derive(Clone, Copy, Debug)]
enum Scale {
    Ue8m0,
    E4M3,
}

impl Element {
    /// Mantissa bits, smallest normal exponent and largest code.
    fn codes(self) -> (u32, i32, f32) {
        match self {
            Self::E4M3 => (3, -6, 448.0),
            Self::E2M1 => (1, 0, 6.0),
        }
    }
}

impl Rounding {
    /// The rounding of an entry's four scalars: `None` for `block` 0.
    /// Admission forms only the codes `rounding.seismic` defines.
    pub fn of(block: i32, element: i32, scale: i32, amax_floor: f32) -> Option<Self> {
        if block == 0 {
            return None;
        }
        let element = match element {
            0 => Element::E4M3,
            1 => Element::E2M1,
            other => panic!("rounding element code {other}"),
        };
        let scale = match scale {
            0 => Scale::Ue8m0,
            1 => Scale::E4M3,
            other => panic!("rounding scale code {other}"),
        };
        let block = usize::try_from(block).unwrap_or_else(|_| panic!("rounding block {block}"));
        Some(Self { block, element, scale, amax_floor })
    }

    /// `row` rounded in place, run by run; its length is a multiple of the
    /// run.
    pub fn apply(self, row: &mut [f32]) {
        assert_eq!(row.len() % self.block, 0, "a rounded row holds whole runs");
        let (mantissa, min_exponent, max) = self.element.codes();
        for run in row.chunks_exact_mut(self.block) {
            let amax = run.iter().fold(0.0f32, |amax, x| amax.max(x.abs()));
            let s = self.scale(amax, max);
            for x in run {
                let q = *x / s;
                let clamped = if q > max { max } else if q < -max { -max } else { q };
                *x = round_codes(clamped, mantissa, min_exponent) * s;
            }
        }
    }

    fn scale(self, amax: f32, max: f32) -> f32 {
        let a = amax.max(self.amax_floor);
        match self.scale {
            Scale::Ue8m0 => power_of_two_ceil(a * (1.0 / max)),
            Scale::E4M3 => round_codes(a / max, 3, -6),
        }
    }
}

/// `v` to the nearest code of `mantissa` bits and smallest normal exponent
/// `min_exponent`, ties to even (`round_codes`).
#[inline(always)]
fn round_codes(v: f32, mantissa: u32, min_exponent: i32) -> f32 {
    let biased = (v.to_bits() >> 23) & 255;
    let step = f32::from_bits((biased.max((min_exponent + 127) as u32) - mantissa) << 23);
    (v / step).round_ties_even() * step
}

/// The least power of two at or above positive normal `v`.
#[inline(always)]
fn power_of_two_ceil(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits(((bits >> 23) + u32::from(bits & 0x007f_ffff != 0)) << 23)
}
