use super::Error;
// GGUF directory validation, preserving the V3 container contract. No tensor
// bytes are uploaded or decoded while metadata and storage ranges are checked.
use std::{
    collections::HashSet,
    io::{Read, Seek, SeekFrom},
};

pub const DEFAULT_HEADER_LIMIT: u64 = 256 * 1024 * 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Encoding {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq4Xs = 23,
}
impl Encoding {
    pub fn block_elements(self) -> u64 {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q8_0 => 32,
            _ => 256,
        }
    }
    pub fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q8_0 => 34,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Iq4Xs => 136,
        }
    }
}
impl TryFrom<u32> for Encoding {
    type Error = Error;
    fn try_from(value: u32) -> Result<Self, Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            23 => Ok(Self::Iq4Xs),
            _ => Err(Error::Invalid(format!("unsupported GGUF encoding {value}"))),
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    String(String),
    Bool(bool),
    Unsigned(u64),
    Signed(i64),
    Float(f64),
}
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Scalar(Scalar),
    Array(Vec<Scalar>),
}
impl Value {
    pub fn unsigned(&self) -> Option<u64> {
        match self {
            Self::Scalar(Scalar::Unsigned(n)) => Some(*n),
            Self::Scalar(Scalar::Signed(n)) => u64::try_from(*n).ok(),
            _ => None,
        }
    }
    pub fn string(&self) -> Option<&str> {
        if let Self::Scalar(Scalar::String(s)) = self {
            Some(s)
        } else {
            None
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub name: String,
    pub value: Value,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    /// Outermost-first logical dimensions, reversing the GGML storage order.
    pub shape: Vec<u64>,
    pub encoding: Encoding,
    /// Byte offset relative to the data section.
    pub offset: u64,
    pub nbytes: u64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Directory {
    pub version: u32,
    pub byte_order: ByteOrder,
    pub alignment: u64,
    pub data_offset: u64,
    pub metadata: Vec<Metadata>,
    pub tensors: Vec<Tensor>,
}
impl Directory {
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn value(&self, name: &str) -> Option<&Value> {
        self.metadata
            .iter()
            .find(|m| m.name == name)
            .map(|m| &m.value)
    }
    /// The directory reader accepts portable endianness; encoded numerical
    /// kernels currently accept little-endian source weights, as in V3.
    pub fn require_execution_byte_order(&self) -> Result<(), Error> {
        if self.byte_order != ByteOrder::Little {
            Err(Error::Invalid(
                "encoded kernels require little-endian GGUF weights".into(),
            ))
        } else {
            Ok(())
        }
    }
}
fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
struct Reader<'a, R> {
    source: &'a mut R,
    offset: u64,
    end: u64,
    order: ByteOrder,
}
impl<R: Read> Reader<'_, R> {
    fn check(&self, size: u64) -> Result<(), Error> {
        if size > self.end.saturating_sub(self.offset) {
            Err(invalid(format!(
                "truncated or oversized GGUF header at byte {}",
                self.offset
            )))
        } else {
            Ok(())
        }
    }
    fn take<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        self.check(N as u64)?;
        let mut bytes = [0; N];
        self.source.read_exact(&mut bytes)?;
        self.offset += N as u64;
        Ok(bytes)
    }
    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16, Error> {
        let bytes = self.take()?;
        Ok(match self.order {
            ByteOrder::Little => u16::from_le_bytes(bytes),
            ByteOrder::Big => u16::from_be_bytes(bytes),
        })
    }
    fn u32(&mut self) -> Result<u32, Error> {
        let bytes = self.take()?;
        Ok(match self.order {
            ByteOrder::Little => u32::from_le_bytes(bytes),
            ByteOrder::Big => u32::from_be_bytes(bytes),
        })
    }
    fn u64(&mut self) -> Result<u64, Error> {
        let bytes = self.take()?;
        Ok(match self.order {
            ByteOrder::Little => u64::from_le_bytes(bytes),
            ByteOrder::Big => u64::from_be_bytes(bytes),
        })
    }
    fn string(&mut self) -> Result<String, Error> {
        let size = self.u64()?;
        self.check(size)?;
        let size =
            usize::try_from(size).map_err(|_| invalid("GGUF string exceeds address range"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| invalid("cannot allocate bounded GGUF string"))?;
        bytes.resize(size, 0);
        self.source.read_exact(&mut bytes)?;
        self.offset += size as u64;
        String::from_utf8(bytes).map_err(|_| invalid("invalid UTF-8 in GGUF header"))
    }
    fn scalar(&mut self, kind: u32) -> Result<Scalar, Error> {
        Ok(match kind {
            0 => Scalar::Unsigned(u64::from(self.u8()?)),
            1 => Scalar::Signed(i64::from(self.u8()? as i8)),
            2 => Scalar::Unsigned(u64::from(self.u16()?)),
            3 => Scalar::Signed(i64::from(self.u16()? as i16)),
            4 => Scalar::Unsigned(u64::from(self.u32()?)),
            5 => Scalar::Signed(i64::from(self.u32()? as i32)),
            6 => Scalar::Float(f64::from(f32::from_bits(self.u32()?))),
            7 => match self.u8()? {
                0 => Scalar::Bool(false),
                1 => Scalar::Bool(true),
                _ => return Err(invalid("invalid GGUF boolean")),
            },
            8 => Scalar::String(self.string()?),
            10 => Scalar::Unsigned(self.u64()?),
            11 => Scalar::Signed(self.u64()? as i64),
            12 => Scalar::Float(f64::from_bits(self.u64()?)),
            _ => return Err(invalid(format!("unsupported GGUF metadata type {kind}"))),
        })
    }
    fn value(&mut self) -> Result<Value, Error> {
        let kind = self.u32()?;
        if kind != 9 {
            return Ok(Value::Scalar(self.scalar(kind)?));
        }
        let element = self.u32()?;
        let count = self.u64()?;
        if element == 9 || count > self.end - self.offset {
            return Err(invalid("nested or oversized GGUF array"));
        }
        // Grow as bytes are consumed. A bounded header is not permission to
        // allocate a much larger array from an untrusted count before reading it.
        let mut values = Vec::new();
        for _ in 0..count {
            values
                .try_reserve(1)
                .map_err(|_| invalid("cannot allocate bounded GGUF array"))?;
            values.push(self.scalar(element)?)
        }
        Ok(Value::Array(values))
    }
}

pub fn read_directory<R: Read + Seek>(
    source: &mut R,
    header_limit: u64,
) -> Result<Directory, Error> {
    let size = source.seek(SeekFrom::End(0))?;
    source.seek(SeekFrom::Start(0))?;
    let mut r = Reader {
        source,
        offset: 0,
        end: size.min(header_limit),
        order: ByteOrder::Little,
    };
    if &r.take::<4>()? != b"GGUF" {
        return Err(invalid("not a GGUF container"));
    }
    let bytes = r.take::<4>()?;
    // V3 recognizes the big-endian v3 header; big-endian v2 is not admitted.
    if bytes == [0, 0, 0, 3] {
        r.order = ByteOrder::Big
    }
    let version = match r.order {
        ByteOrder::Little => u32::from_le_bytes(bytes),
        ByteOrder::Big => u32::from_be_bytes(bytes),
    };
    if !matches!(version, 2 | 3) {
        return Err(invalid(format!("unsupported GGUF version {version}")));
    }
    let tensors_count = r.u64()?;
    let metadata_count = r.u64()?;
    if tensors_count
        .checked_add(metadata_count)
        .is_none_or(|n| n > (r.end - r.offset) / 12)
    {
        return Err(invalid("GGUF entry counts exceed header bounds"));
    }
    let mut metadata = Vec::new();
    let mut names = HashSet::new();
    let mut alignment = 32;
    for _ in 0..metadata_count {
        let name = r.string()?;
        let value = r.value()?;
        if !names.insert(name.clone()) {
            return Err(invalid(format!("duplicate metadata {name:?}")));
        }
        if name == "general.alignment" {
            alignment = value
                .unsigned()
                .filter(|a| a.is_power_of_two())
                .ok_or_else(|| invalid("alignment must be a positive power of two"))?;
        }
        metadata.push(Metadata { name, value });
    }
    let mut tensors = Vec::new();
    names.clear();
    for _ in 0..tensors_count {
        let name = r.string()?;
        let rank = r.u32()?;
        if name.is_empty() || !names.insert(name.clone()) || !(1..=4).contains(&rank) {
            return Err(invalid(format!(
                "invalid or duplicate tensor directory entry {name:?}"
            )));
        }
        let mut shape = (0..rank).map(|_| r.u64()).collect::<Result<Vec<_>, _>>()?;
        let encoding = Encoding::try_from(r.u32()?)?;
        let offset = r.u64()?;
        if shape.contains(&0) || shape[0] % encoding.block_elements() != 0 {
            return Err(invalid(format!(
                "invalid block geometry on tensor {name:?}"
            )));
        }
        if offset % alignment != 0 {
            return Err(invalid(format!("misaligned tensor {name:?}")));
        }
        let elements = shape
            .iter()
            .try_fold(1u64, |a, n| a.checked_mul(*n))
            .ok_or_else(|| invalid(format!("tensor {name:?} element count overflows")))?;
        let nbytes = (elements / encoding.block_elements())
            .checked_mul(encoding.block_bytes())
            .ok_or_else(|| invalid(format!("tensor {name:?} byte size overflows")))?;
        shape.reverse();
        tensors.push(Tensor {
            name,
            shape,
            encoding,
            offset,
            nbytes,
        });
    }
    let data_offset = r
        .offset
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(|| invalid("GGUF data alignment overflows"))?;
    let mut sorted = tensors.iter().collect::<Vec<_>>();
    sorted.sort_by_key(|t| t.offset);
    let mut end = data_offset;
    for tensor in sorted {
        let start = data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| invalid("GGUF tensor offset overflows"))?;
        let next = start
            .checked_add(tensor.nbytes)
            .ok_or_else(|| invalid("GGUF tensor end overflows"))?;
        if start < end || next > size {
            return Err(invalid(format!(
                "overlapping or truncated tensor {:?}",
                tensor.name
            )));
        }
        end = next;
    }
    Ok(Directory {
        version,
        byte_order: r.order,
        alignment,
        data_offset,
        metadata,
        tensors,
    })
}

/// Open container ownership and validated stored-value descriptions. Numerical
/// interpretation remains in the residency codecs, never this directory owner.
pub struct GgufArtifact {
    directory: Directory,
    source: std::sync::Arc<super::source::FileSource>,
    identity: super::descriptor::ArtifactIdentity,
}
impl GgufArtifact {
    pub fn directory(&self) -> &Directory {
        &self.directory
    }
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        let source = std::sync::Arc::new(super::source::FileSource::open(path)?);
        let directory = read_directory(&mut source.reader(), DEFAULT_HEADER_LIMIT)?;
        directory.require_execution_byte_order()?;
        let identity = super::descriptor::ArtifactIdentity(source.digest()?);
        Ok(Self {
            directory,
            source,
            identity,
        })
    }
    pub fn identity(&self) -> super::descriptor::ArtifactIdentity {
        self.identity
    }
    pub fn stored(
        &self,
        descriptor: &super::descriptor::WeightDescriptor,
    ) -> Result<super::descriptor::Stored, Error> {
        use super::descriptor::{Stored, StoredTensor};
        use seismic_lang::types::DType;
        let tensor = self
            .directory
            .tensor(&descriptor.name)
            .ok_or_else(|| invalid(format!("missing GGUF tensor {}", descriptor.name)))?;
        let offset = self
            .directory
            .data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| invalid("GGUF absolute offset overflow"))?;
        match tensor.encoding {
            Encoding::F32 | Encoding::F16 => Ok(Stored::Dense(StoredTensor {
                source: self.source.clone(),
                offset,
                nbytes: tensor.nbytes,
                dtype: if tensor.encoding == Encoding::F32 {
                    DType::F32
                } else {
                    DType::F16
                },
                shape: tensor.shape.clone(),
            })),
            encoding => Ok(Stored::GgmlBlocks {
                source: self.source.clone(),
                offset,
                nbytes: tensor.nbytes,
                shape: tensor.shape.clone(),
                encoding,
            }),
        }
    }
}
