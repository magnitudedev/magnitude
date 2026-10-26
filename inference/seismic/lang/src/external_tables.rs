//! Lookup tables conversions of external GGUF formats read (over ggml's
//! tables, `magnitude_ggml::tables`); nothing here is a resident
//! representation.

/// The resident E2M1 code of a GGUF MXFP4 / NVFP4 code: ggml's
/// `kvalues_mxfp4` decodes code 8 (E2M1 -0) as +0, so a code of value 0
/// becomes code 0; every other code keeps its value.
pub(crate) static E2M1_POSITIVE_ZERO: [i32; 16] = {
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

/// The signed IQ3_S magnitude at index `((sign << 9 | point) << 2) | j`:
/// byte `j` of grid point `point`, negated when `sign` is set.
pub(crate) static IQ3S_SIGNED_MAGNITUDES: [i32; 4096] = {
    let mut values = [0i32; 4096];
    let mut index = 0;
    while index < 4096 {
        let magnitude =
            magnitude_ggml::tables::IQ3S.points.byte((index >> 2) & 511, (index & 3) as u32) as i32;
        values[index] = if index >= 2048 { -magnitude } else { magnitude };
        index += 1;
    }
    values
};
