//! Bounded GGUF directory validation and open container ownership.
//!
//! Tensor bytes are not decoded or uploaded here. `TensorSlice` describes the
//! validated stored range; residency code chooses its numerical interpretation.

use crate::{ArtifactIdentity, Error, FileSource};
use std::{
    collections::HashSet,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
    sync::Arc,
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
    // Numeric IDs, block element counts, and serialized block sizes below are
    // pinned to ggml/llama.cpp commit
    // ff0dbb975e93a9a2899efa34bdd32d1c5cfbc183:
    //   ggml/include/ggml.h (`enum ggml_type`)
    //   ggml/src/ggml-common.h (`QK_*` and `block_*` definitions)
    // Keep this container description independent from numerical import
    // support: accepting a directory entry does not imply that an execution
    // backend can decode it.
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq4Xs = 23,
    I32 = 26,
    BF16 = 30,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
}

impl Encoding {
    pub fn block_elements(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::I32 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Iq4Nl | Self::Mxfp4 => 32,
            Self::Nvfp4 => 64,
            Self::Q1_0 => 128,
            Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Iq3S | Self::Iq4Xs => 256,
        }
    }

    pub fn block_bytes(self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 | Self::Iq4Nl => 18,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q3K | Self::Iq3S => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Iq4Xs => 136,
            Self::Mxfp4 => 17,
            Self::Nvfp4 => 36,
            Self::Q1_0 => 18,
        }
    }
}

impl TryFrom<u32> for Encoding {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            23 => Ok(Self::Iq4Xs),
            26 => Ok(Self::I32),
            30 => Ok(Self::BF16),
            39 => Ok(Self::Mxfp4),
            40 => Ok(Self::Nvfp4),
            41 => Ok(Self::Q1_0),
            _ => Err(invalid(format!("unsupported GGUF encoding {value}"))),
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
            Self::Scalar(Scalar::Unsigned(value)) => Some(*value),
            Self::Scalar(Scalar::Signed(value)) => u64::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn string(&self) -> Option<&str> {
        match self {
            Self::Scalar(Scalar::String(value)) => Some(value),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub name: String,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub name: String,
    /// Outermost-first logical dimensions, reversing GGML storage order.
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
    pub tensors: Vec<TensorDescriptor>,
}

impl Directory {
    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    pub fn value(&self, name: &str) -> Option<&Value> {
        self.metadata
            .iter()
            .find(|metadata| metadata.name == name)
            .map(|metadata| &metadata.value)
    }

    pub fn require_execution_byte_order(&self) -> Result<(), Error> {
        if self.byte_order == ByteOrder::Little {
            Ok(())
        } else {
            Err(invalid(
                "encoded kernels require little-endian GGUF weights",
            ))
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
            usize::try_from(size).map_err(|_| invalid("GGUF string exceeds host address range"))?;
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
        let mut values = Vec::new();
        for _ in 0..count {
            values
                .try_reserve(1)
                .map_err(|_| invalid("cannot allocate bounded GGUF array"))?;
            values.push(self.scalar(element)?);
        }
        Ok(Value::Array(values))
    }
}

pub fn read_directory<R: Read + Seek>(
    source: &mut R,
    header_limit: u64,
) -> Result<Directory, Error> {
    read_directory_with_payload(source, header_limit, true)
}

/// Inspect a downloaded GGUF header bundle before weight payloads exist.
/// Header bounds, tensor geometry, offsets, and non-overlap are validated;
/// only the physical presence of declared tensor bytes is deferred.
pub fn inspect_header(path: impl AsRef<Path>) -> Result<Directory, Error> {
    let source = FileSource::open(path)?;
    let directory = read_directory_with_payload(
        &mut BufReader::with_capacity(1 << 20, source.reader()),
        DEFAULT_HEADER_LIMIT,
        false,
    )?;
    directory.require_execution_byte_order()?;
    Ok(directory)
}

fn read_directory_with_payload<R: Read + Seek>(
    source: &mut R,
    header_limit: u64,
    require_payload: bool,
) -> Result<Directory, Error> {
    let size = source.seek(SeekFrom::End(0))?;
    source.seek(SeekFrom::Start(0))?;
    let mut reader = Reader {
        source,
        offset: 0,
        end: size.min(header_limit),
        order: ByteOrder::Little,
    };
    if &reader.take::<4>()? != b"GGUF" {
        return Err(invalid("not a GGUF container"));
    }
    let version_bytes = reader.take::<4>()?;
    if version_bytes == [0, 0, 0, 3] {
        reader.order = ByteOrder::Big;
    }
    let version = match reader.order {
        ByteOrder::Little => u32::from_le_bytes(version_bytes),
        ByteOrder::Big => u32::from_be_bytes(version_bytes),
    };
    if !matches!(version, 2 | 3) {
        return Err(invalid(format!("unsupported GGUF version {version}")));
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count
        .checked_add(metadata_count)
        .is_none_or(|count| count > (reader.end - reader.offset) / 12)
    {
        return Err(invalid("GGUF entry counts exceed header bounds"));
    }

    let mut metadata = Vec::new();
    let mut names = HashSet::new();
    let mut alignment = 32;
    for _ in 0..metadata_count {
        let name = reader.string()?;
        let value = reader.value()?;
        if !names.insert(name.clone()) {
            return Err(invalid(format!("duplicate metadata {name:?}")));
        }
        if name == "general.alignment" {
            alignment = value
                .unsigned()
                .filter(|alignment| alignment.is_power_of_two())
                .ok_or_else(|| invalid("alignment must be a positive power of two"))?;
        }
        metadata.push(Metadata { name, value });
    }

    let mut tensors = Vec::new();
    names.clear();
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let rank = reader.u32()?;
        if name.is_empty() || !names.insert(name.clone()) || !(1..=4).contains(&rank) {
            return Err(invalid(format!(
                "invalid or duplicate tensor directory entry {name:?}"
            )));
        }
        let mut shape = (0..rank)
            .map(|_| reader.u64())
            .collect::<Result<Vec<_>, _>>()?;
        let encoding = Encoding::try_from(reader.u32()?)?;
        let offset = reader.u64()?;
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
            .try_fold(1u64, |product, dimension| product.checked_mul(*dimension));
        let elements =
            elements.ok_or_else(|| invalid(format!("tensor {name:?} element count overflows")))?;
        let nbytes = (elements / encoding.block_elements())
            .checked_mul(encoding.block_bytes())
            .ok_or_else(|| invalid(format!("tensor {name:?} byte size overflows")))?;
        shape.reverse();
        tensors.push(TensorDescriptor {
            name,
            shape,
            encoding,
            offset,
            nbytes,
        });
    }

    let data_offset = reader
        .offset
        .checked_add(alignment - 1)
        .map(|offset| offset & !(alignment - 1))
        .ok_or_else(|| invalid("GGUF data alignment overflows"))?;
    let mut stored = tensors.iter().collect::<Vec<_>>();
    stored.sort_by_key(|tensor| tensor.offset);
    let mut end = data_offset;
    for tensor in stored {
        let start = data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| invalid("GGUF tensor offset overflows"))?;
        let next = start
            .checked_add(tensor.nbytes)
            .ok_or_else(|| invalid("GGUF tensor end overflows"))?;
        if start < end || (require_payload && next > size) {
            return Err(invalid(format!(
                "overlapping or truncated tensor {:?}",
                tensor.name
            )));
        }
        end = next;
    }

    Ok(Directory {
        version,
        byte_order: reader.order,
        alignment,
        data_offset,
        metadata,
        tensors,
    })
}

#[derive(Clone, Debug)]
pub struct TensorSlice {
    pub source: Arc<FileSource>,
    pub offset: u64,
    pub nbytes: u64,
    pub shape: Vec<u64>,
    pub encoding: Encoding,
}

impl TensorSlice {
    pub fn read(&self) -> Result<Vec<u8>, Error> {
        self.source.read(
            self.offset,
            usize::try_from(self.nbytes)
                .map_err(|_| invalid("tensor exceeds host address range"))?,
        )
    }
}

/// A validated GGUF directory retaining ownership of its open source.
#[derive(Debug)]
pub struct GgufArtifact {
    directory: Directory,
    source: Arc<FileSource>,
    identity: ArtifactIdentity,
}

impl GgufArtifact {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let source = Arc::new(FileSource::open(path)?);
        let directory = read_directory(
            &mut BufReader::with_capacity(1 << 20, source.reader()),
            DEFAULT_HEADER_LIMIT,
        )?;
        directory.require_execution_byte_order()?;
        let identity = ArtifactIdentity::for_open();
        Ok(Self {
            directory,
            source,
            identity,
        })
    }

    pub fn directory(&self) -> &Directory {
        &self.directory
    }

    pub fn source(&self) -> &Arc<FileSource> {
        &self.source
    }

    pub fn identity(&self) -> ArtifactIdentity {
        self.identity
    }

    pub fn tensor(&self, name: &str) -> Result<TensorSlice, Error> {
        let tensor = self
            .directory
            .tensor(name)
            .ok_or_else(|| invalid(format!("missing GGUF tensor {name}")))?;
        let offset = self
            .directory
            .data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| invalid("GGUF absolute tensor offset overflows"))?;
        Ok(TensorSlice {
            source: self.source.clone(),
            offset,
            nbytes: tensor.nbytes,
            shape: tensor.shape.clone(),
            encoding: tensor.encoding,
        })
    }
}
