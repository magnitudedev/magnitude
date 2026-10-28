//! The ggml tensor types a GGUF file stores, their block geometry, and ggml's
//! lookup tables (`tables`): the one statement of them the GGUF reader, the
//! kernel registry and every backend's native library read.

use serde::{Deserialize, Serialize};

pub mod tables;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum Encoding {
    // Numeric IDs, block element counts, and serialized block sizes below are
    // pinned to ggml/llama.cpp commit
    // 5e6c0e18b6b11f109411401239b3b0ef61058dae:
    //   ggml/include/ggml.h (`enum ggml_type`)
    //   ggml/src/ggml-common.h (`QK_*` and `block_*` definitions)
    // Every type a GGUF file can store is listed, so any valid header
    // parses. Keep this container description independent from numerical
    // import support: accepting a directory entry does not imply that an
    // execution backend can decode it. Retired ids (4, 5, 31-33, 36-38) are
    // not valid in GGUF files.
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    BF16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
}

impl Encoding {
    /// Every encoding, in ggml id order.
    pub const ALL: [Self; 34] = [
        Self::F32,
        Self::F16,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Q8K,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq3Xxs,
        Self::Iq1S,
        Self::Iq4Nl,
        Self::Iq3S,
        Self::Iq2S,
        Self::Iq4Xs,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F64,
        Self::Iq1M,
        Self::BF16,
        Self::Tq1_0,
        Self::Tq2_0,
        Self::Mxfp4,
        Self::Nvfp4,
        Self::Q1_0,
    ];

    /// The encoding of ggml type id `id`; `None` for an id no GGUF file may
    /// store.
    pub fn from_id(id: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|encoding| *encoding as u32 == id)
    }

    /// The ggml type name (`ggml_type_name`), as GGUF tooling prints it.
    pub const fn ggml_name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq1S => "IQ1_S",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq2S => "IQ2_S",
            Self::Iq4Xs => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::Iq1M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::Tq1_0 => "TQ1_0",
            Self::Tq2_0 => "TQ2_0",
            Self::Mxfp4 => "MXFP4",
            Self::Nvfp4 => "NVFP4",
            Self::Q1_0 => "Q1_0",
        }
    }

    pub const fn block_elements(self) -> u64 {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::Iq4Nl
            | Self::Mxfp4 => 32,
            Self::Nvfp4 => 64,
            Self::Q1_0 => 128,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq2S
            | Self::Iq3Xxs
            | Self::Iq3S
            | Self::Iq1S
            | Self::Iq1M
            | Self::Iq4Xs
            | Self::Tq1_0
            | Self::Tq2_0 => 256,
        }
    }

    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::I8 => 1,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::Q4_0 | Self::Iq4Nl | Self::Q1_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Mxfp4 => 17,
            Self::Nvfp4 => 36,
            Self::Iq1S => 50,
            Self::Tq1_0 => 54,
            Self::Iq1M => 56,
            Self::Iq2Xxs | Self::Tq2_0 => 66,
            Self::Iq2Xs => 74,
            Self::Iq2S => 82,
            Self::Q2K => 84,
            Self::Iq3Xxs => 98,
            Self::Q3K | Self::Iq3S => 110,
            Self::Iq4Xs => 136,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }
}
