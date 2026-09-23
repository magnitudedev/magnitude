//! Immutable host preparation results. Binary numerical payloads cross the owner
//! boundary without processor objects or device handles. Multibyte values use
//! little-endian encoding; consumers must interpret them with the declared dtype.
use sha2::{Digest, Sha256};
use std::{collections::HashSet, sync::Arc};

pub const MAX_PREPARED_BYTES: usize = 512 << 20;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    I32,
    I64,
    U8,
}
impl DType {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U8 => "uint8",
        }
    }
    pub fn bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}
#[derive(Clone, Debug)]
pub struct PreparedTensor {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    data: Arc<[u8]>,
}
impl PreparedTensor {
    pub fn new(
        name: String,
        dtype: DType,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) -> Result<Self, String> {
        if name.is_empty()
            || shape.is_empty()
            || shape.len() > 5
            || shape.iter().any(|&n| n == 0 || n >= (1usize << 31))
        {
            return Err("invalid prepared tensor metadata".into());
        }
        let bytes = shape
            .iter()
            .try_fold(dtype.bytes(), |bytes, &n| bytes.checked_mul(n))
            .ok_or("prepared tensor geometry overflows")?;
        if bytes != data.len() || bytes > MAX_PREPARED_BYTES {
            return Err("prepared tensor bytes differ from geometry or exceed capacity".into());
        }
        Ok(Self {
            name,
            dtype,
            shape,
            data: data.into(),
        })
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}
#[derive(Clone, Debug)]
pub struct PreparedMedia {
    processor: String,
    tensors: Arc<[PreparedTensor]>,
    nbytes: usize,
    identity: String,
}
impl PreparedMedia {
    pub fn new(processor: String, tensors: Vec<PreparedTensor>) -> Result<Self, String> {
        if processor.len() != 64
            || !processor
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || tensors.is_empty()
            || tensors.len() > 16
        {
            return Err("prepared media requires a processor identity and bounded tensors".into());
        }
        let mut names = HashSet::new();
        let mut nbytes = 0usize;
        let mut digest = Sha256::new();
        digest.update(processor.as_bytes());
        for tensor in &tensors {
            if !names.insert(tensor.name()) {
                return Err("prepared media repeats tensor names".into());
            }
            nbytes = nbytes
                .checked_add(tensor.data.len())
                .filter(|&bytes| bytes <= MAX_PREPARED_BYTES)
                .ok_or("prepared media exceeds capacity")?;
            // V3's framed digest uses Python's tuple spelling, including the
            // singleton comma. Keep that identity stable across host languages.
            let mut shape = format!(
                "({}",
                tensor
                    .shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            if tensor.shape.len() == 1 {
                shape.push(',');
            }
            shape.push(')');
            for field in [
                tensor.name.as_bytes(),
                tensor.dtype.name().as_bytes(),
                shape.as_bytes(),
                tensor.data(),
            ] {
                digest.update((field.len() as u64).to_le_bytes());
                digest.update(field);
            }
        }
        let identity = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok(Self {
            processor,
            tensors: tensors.into(),
            nbytes,
            identity,
        })
    }
    pub fn processor(&self) -> &str {
        &self.processor
    }
    pub fn tensors(&self) -> &[PreparedTensor] {
        &self.tensors
    }
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
}
