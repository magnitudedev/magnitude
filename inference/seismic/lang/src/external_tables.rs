//! Lookup tables conversions of external GGUF formats read (over ggml's
//! tables, `magnitude_ggml::tables`); nothing here is a resident
//! representation.

/// A registered lookup table of a conversion recipe (`RepackExpr::Lookup`):
/// an id an executable form names rather than the table's address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RepackTable {
    E2m1PositiveZero,
    Iq3sMagnitudes,
}

impl RepackTable {
    pub fn values(self) -> &'static [i32] {
        match self {
            Self::E2m1PositiveZero => &E2M1_POSITIVE_ZERO,
            Self::Iq3sMagnitudes => &IQ3S_MAGNITUDES,
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

/// The IQ3_S magnitude at index `point << 2 | j`: byte `j` of ggml's grid
/// point `point` (its sign is a recipe's `NegateIf`).
static IQ3S_MAGNITUDES: [i32; 2048] = {
    let mut values = [0i32; 2048];
    let mut index = 0;
    while index < 2048 {
        values[index] = magnitude_ggml::tables::IQ3S
            .points
            .byte(index >> 2, (index & 3) as u32) as i32;
        index += 1;
    }
    values
};
