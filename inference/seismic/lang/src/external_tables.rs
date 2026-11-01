//! Lookup tables conversions of external GGUF formats read (over ggml's
//! tables, `magnitude_ggml::tables`, and TQ1_0's base-3 digits); nothing
//! here is a resident representation.

/// A registered lookup table of a conversion recipe (`RepackExpr::Lookup`):
/// an id an executable form names rather than the table's address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RepackTable {
    E2m1PositiveZero,
    /// A byte's base-3 digit at one place (`DigitCodes`).
    Digit(DigitPlace),
}

impl RepackTable {
    pub fn values(self) -> &'static [i32] {
        match self {
            Self::E2m1PositiveZero => &E2M1_POSITIVE_ZERO,
            Self::Digit(DigitPlace(place)) => &DIGITS[place as usize],
        }
    }

    /// The table as one byte per entry, when it has at most 16 entries and
    /// each is a byte: a vector table instruction looks it up.
    pub fn bytes(self) -> Option<[u8; 16]> {
        let values = self.values();
        let mut bytes = [0u8; 16];
        for (byte, value) in bytes.iter_mut().zip(values) {
            *byte = u8::try_from(*value).ok()?;
        }
        (values.len() <= bytes.len()).then_some(bytes)
    }
}

/// The resident E2M1 code of a GGUF MXFP4 / NVFP4 code: ggml's
/// `kvalues_mxfp4` decodes code 8 (E2M1 -0) as +0, so a code of value 0
/// becomes code 0; every other code keeps its value.
static E2M1_POSITIVE_ZERO: [i32; 16] = {
    let mut codes = [0i32; 16];
    let mut code = 0;
    while code < 16 {
        codes[code] = if magnitude_ggml::tables::KVALUES_MXFP4[code] == 0 {
            0
        } else {
            code as i32
        };
        code += 1;
    }
    codes
};

/// Base-3 digit codes (ggml's TQ1_0): bytes each holding a few values'
/// digits as a base-3 fraction of the byte, in `runs` that cover a block's
/// values in order: a run of `bytes` bytes of `digits` digits holds the next
/// `bytes digits` values, its value `o` being digit `o / bytes` of its byte
/// `o % bytes`.
pub(crate) struct DigitCodes {
    runs: &'static [DigitRun],
}

/// A run of `bytes` bytes of `digits` digits each, at most
/// `DIGITS_PER_BYTE`.
struct DigitRun {
    bytes: u32,
    digits: u32,
}

impl DigitRun {
    const fn new(bytes: u32, digits: u32) -> Self {
        assert!(
            digits <= DIGITS_PER_BYTE,
            "a run's byte holds at most `DIGITS_PER_BYTE` digits"
        );
        Self { bytes, digits }
    }
}

/// A digit's place in its byte, below `DIGITS_PER_BYTE`
/// (`DigitCodes::place`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DigitPlace(u8);

/// The radix of digit codes.
const DIGIT_RADIX: u32 = 3;
/// Digits one byte holds at most: the most `n` with `DIGIT_RADIX^n` below
/// `2^8` (`3^5 = 243`).
const DIGITS_PER_BYTE: u32 = {
    let mut digits = 0;
    while DIGIT_RADIX.pow(digits + 1) <= 1 << u8::BITS {
        digits += 1;
    }
    digits
};

impl DigitCodes {
    /// The byte and the digit's place of value `value`.
    pub(crate) const fn place(&self, value: u32) -> (u32, DigitPlace) {
        let (mut first, mut byte) = (0, 0);
        let mut run = 0;
        while run < self.runs.len() {
            let DigitRun { bytes, digits } = self.runs[run];
            if value < first + bytes * digits {
                let offset = value - first;
                return (byte + offset % bytes, DigitPlace((offset / bytes) as u8));
            }
            first += bytes * digits;
            byte += bytes;
            run += 1;
        }
        panic!("a value lies past the digit runs")
    }
}

/// TQ1_0's digits: 32 and 16 bytes of five digits, then 4 of four.
pub(crate) static TQ1_DIGITS: DigitCodes = DigitCodes {
    runs: &[
        DigitRun::new(32, 5),
        DigitRun::new(16, 5),
        DigitRun::new(4, 4),
    ],
};

/// Digit `n` of byte `b` at each place `n`: `(((b 3^n) mod 256) 3) >> 8`
/// (ggml's `dequantize_row_tq1_0`, exact for every byte).
static DIGITS: [[i32; 256]; DIGITS_PER_BYTE as usize] = {
    let mut digits = [[0i32; 256]; DIGITS_PER_BYTE as usize];
    let mut place = 0;
    while place < DIGITS_PER_BYTE {
        let mut byte = 0;
        while byte < 256 {
            let fraction = (byte * DIGIT_RADIX.pow(place)) % 256;
            digits[place as usize][byte as usize] = ((fraction * DIGIT_RADIX) >> u8::BITS) as i32;
            byte += 1;
        }
        place += 1;
    }
    digits
};
