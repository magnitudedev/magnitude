//! Bounded GGUF directory validation and open container ownership.
//!
//! Tensor bytes are not decoded or uploaded here. `TensorSlice` describes the
//! validated stored range; residency code chooses its numerical interpretation.

use crate::{ArtifactIdentity, Error, FileSource};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

pub const DEFAULT_HEADER_LIMIT: u64 = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
}

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

    pub fn block_elements(self) -> u64 {
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

    pub fn block_bytes(self) -> u64 {
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

impl TryFrom<u32> for Encoding {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::ALL
            .into_iter()
            .find(|encoding| *encoding as u32 == value)
            .ok_or_else(|| invalid(format!("unknown GGUF tensor type {value}")))
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDescriptor {
    pub name: String,
    /// Outermost-first logical dimensions, reversing GGML storage order.
    pub shape: Vec<u64>,
    pub encoding: Encoding,
    /// Byte offset relative to the data section.
    pub offset: u64,
    pub nbytes: u64,
}

/// The directory of one GGUF component. For a single file it is that
/// file's header. For a split GGUF it is the whole set: the first shard's
/// metadata, every shard's tensors, and one data section that concatenates
/// the shards' data sections in shard order (each starting at the next
/// alignment boundary after the previous shard's last tensor), so
/// `data_offset` is 0 and tensor offsets address that concatenation.
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

/// Inspect one downloaded GGUF file header before weight payloads exist.
/// Header bounds, tensor geometry, offsets, and non-overlap are validated;
/// only the physical presence of declared tensor bytes is deferred.
pub fn inspect_header(path: impl AsRef<Path>) -> Result<Directory, Error> {
    let source = FileSource::open(path)?;
    read_file_directory(&source, false)
}

/// Inspect the headers of the GGUF component `path` names: the file itself,
/// or every shard when it is the first shard of a split GGUF (see
/// [`split_paths`]). Tensor payloads need not exist. Returns the component
/// directory and its files in shard order.
pub(crate) fn inspect_component_header(path: &Path) -> Result<(Directory, Vec<FileSource>), Error> {
    let first = FileSource::open(path)?;
    let directory = read_file_directory(&first, false)?;
    match split_paths(path, &directory)? {
        None => Ok((directory, vec![first])),
        Some(paths) => {
            let mut sources = vec![first];
            let mut shards = vec![directory];
            for shard in &paths[1..] {
                let source = FileSource::open(shard)?;
                shards.push(read_file_directory(&source, false)?);
                sources.push(source);
            }
            Ok((merge_shards(shards)?.0, sources))
        }
    }
}

fn read_file_directory(source: &FileSource, require_payload: bool) -> Result<Directory, Error> {
    let directory = read_directory_with_payload(
        &mut BufReader::with_capacity(1 << 20, source.reader()),
        DEFAULT_HEADER_LIMIT,
        require_payload,
    )?;
    directory.require_execution_byte_order()?;
    Ok(directory)
}

/// Metadata every shard of a llama.cpp `gguf-split` GGUF carries.
const SPLIT_INDEX: &str = "split.no";
const SPLIT_COUNT: &str = "split.count";
const SPLIT_TENSORS: &str = "split.tensors.count";

/// This file's `(index, count)` within a split GGUF, or `None` when it is
/// not split (no split metadata, or a count of one).
fn split_position(directory: &Directory) -> Result<Option<(u64, u64)>, Error> {
    let number = |key: &str| {
        directory
            .value(key)
            .map(|value| {
                value
                    .unsigned()
                    .ok_or_else(|| invalid(format!("{key} is not an unsigned integer")))
            })
            .transpose()
    };
    match (number(SPLIT_INDEX)?, number(SPLIT_COUNT)?) {
        (None, None) | (Some(0), Some(1)) => Ok(None),
        (Some(index), Some(count)) if index < count => Ok(Some((index, count))),
        _ => Err(invalid("inconsistent GGUF split metadata")),
    }
}

/// Every file of the split GGUF whose first shard is `first`, in shard
/// order, by the split naming convention `<prefix>-NNNNN-of-MMMMM.gguf`;
/// `None` when the file is not split. Only a first shard names a split
/// component.
pub fn split_paths(first: &Path, directory: &Directory) -> Result<Option<Vec<PathBuf>>, Error> {
    let Some((index, count)) = split_position(directory)? else {
        return Ok(None);
    };
    if index != 0 {
        return Err(invalid(format!(
            "{} is shard {} of a split GGUF; open its first shard",
            first.display(),
            index + 1
        )));
    }
    let name = first
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("split GGUF shard name is not UTF-8"))?;
    let suffix = split_file_suffix(1, count);
    let prefix = name
        .strip_suffix(&suffix)
        .filter(|prefix| !prefix.is_empty())
        .ok_or_else(|| {
            invalid(format!(
                "first shard {name:?} of a {count}-file split GGUF is not named <prefix>{suffix}"
            ))
        })?;
    Ok(Some(
        (1..=count)
            .map(|number| {
                first.with_file_name(format!("{prefix}{}", split_file_suffix(number, count)))
            })
            .collect(),
    ))
}

fn split_file_suffix(number: u64, count: u64) -> String {
    format!("-{number:05}-of-{count:05}.gguf")
}

/// Merge the shard directories of one split GGUF, in shard order, with
/// where each shard's data section starts in the merged data section.
fn merge_shards(shards: Vec<Directory>) -> Result<(Directory, Vec<u64>), Error> {
    let count = u64::try_from(shards.len()).map_err(|_| invalid("split shard count overflows"))?;
    let first = shards
        .first()
        .ok_or_else(|| invalid("split GGUF has no shards"))?;
    let (version, byte_order, alignment) = (first.version, first.byte_order, first.alignment);
    let declared_tensors = first
        .value(SPLIT_TENSORS)
        .map(|value| {
            value
                .unsigned()
                .ok_or_else(|| invalid(format!("{SPLIT_TENSORS} is not an unsigned integer")))
        })
        .transpose()?;
    let mut bases = Vec::with_capacity(shards.len());
    let mut names = HashSet::new();
    let mut tensors = Vec::new();
    let mut metadata = None;
    let mut base = 0u64;
    for (index, shard) in shards.into_iter().enumerate() {
        let index = u64::try_from(index).map_err(|_| invalid("split shard index overflows"))?;
        if split_position(&shard)? != Some((index, count)) {
            return Err(invalid(format!(
                "shard {} does not declare itself shard {} of {count}",
                index + 1,
                index + 1
            )));
        }
        if (shard.version, shard.byte_order, shard.alignment) != (version, byte_order, alignment) {
            return Err(invalid(
                "split GGUF shards differ in version, byte order or alignment",
            ));
        }
        let mut end = 0u64;
        for tensor in shard.tensors {
            if !names.insert(tensor.name.clone()) {
                return Err(invalid(format!(
                    "tensor {:?} appears in more than one split shard",
                    tensor.name
                )));
            }
            end = end.max(
                tensor
                    .offset
                    .checked_add(tensor.nbytes)
                    .ok_or_else(|| invalid("GGUF tensor end overflows"))?,
            );
            tensors.push(TensorDescriptor {
                offset: base
                    .checked_add(tensor.offset)
                    .ok_or_else(|| invalid("split GGUF data offset overflows"))?,
                ..tensor
            });
        }
        bases.push(base);
        base = base
            .checked_add(end)
            .and_then(|end| end.checked_add(alignment - 1))
            .map(|end| end & !(alignment - 1))
            .ok_or_else(|| invalid("split GGUF data offset overflows"))?;
        if metadata.is_none() {
            metadata = Some(shard.metadata);
        }
    }
    if let Some(declared) = declared_tensors {
        if u64::try_from(tensors.len()).ok() != Some(declared) {
            return Err(invalid(format!(
                "split GGUF declares {declared} tensors but its shards hold {}",
                tensors.len()
            )));
        }
    }
    Ok((
        Directory {
            version,
            byte_order,
            alignment,
            data_offset: 0,
            metadata: metadata.ok_or_else(|| invalid("split GGUF has no shards"))?,
            tensors,
        },
        bases,
    ))
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

/// One open file of a GGUF component and where its data section sits in the
/// component directory's data section.
#[derive(Debug)]
struct Shard {
    source: Arc<FileSource>,
    /// The file's own data section offset.
    data_offset: u64,
    /// Start of this file's data in the directory's data section.
    base: u64,
}

/// A validated GGUF component retaining ownership of its open files: one
/// file, or every shard of a split GGUF under one directory and identity.
#[derive(Debug)]
pub struct GgufArtifact {
    directory: Directory,
    shards: Vec<Shard>,
    identity: ArtifactIdentity,
}

impl GgufArtifact {
    /// Open the GGUF component `path` names: the file, or every shard when
    /// it is the first shard of a split GGUF (see [`split_paths`]).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let first = Arc::new(FileSource::open(path)?);
        let directory = read_file_directory(&first, true)?;
        let (directory, shards) = match split_paths(path, &directory)? {
            None => {
                let data_offset = directory.data_offset;
                (
                    directory,
                    vec![Shard {
                        source: first,
                        data_offset,
                        base: 0,
                    }],
                )
            }
            Some(paths) => {
                let mut sources = vec![first];
                let mut directories = vec![directory];
                for shard in &paths[1..] {
                    let source = Arc::new(FileSource::open(shard)?);
                    directories.push(read_file_directory(&source, true)?);
                    sources.push(source);
                }
                let offsets = directories
                    .iter()
                    .map(|directory| directory.data_offset)
                    .collect::<Vec<_>>();
                let (directory, bases) = merge_shards(directories)?;
                let shards = sources
                    .into_iter()
                    .zip(offsets)
                    .zip(bases)
                    .map(|((source, data_offset), base)| Shard {
                        source,
                        data_offset,
                        base,
                    })
                    .collect();
                (directory, shards)
            }
        };
        Ok(Self {
            directory,
            shards,
            identity: ArtifactIdentity::for_open(),
        })
    }

    pub fn directory(&self) -> &Directory {
        &self.directory
    }

    /// The component's open files in shard order; one for an unsplit GGUF.
    pub fn sources(&self) -> impl ExactSizeIterator<Item = &Arc<FileSource>> {
        self.shards.iter().map(|shard| &shard.source)
    }

    pub fn identity(&self) -> ArtifactIdentity {
        self.identity
    }

    /// Carry the identity the admitting process issued for this component.
    pub(crate) fn adopt_identity(&mut self, identity: ArtifactIdentity) {
        self.identity = identity;
    }

    pub fn tensor(&self, name: &str) -> Result<TensorSlice, Error> {
        let tensor = self
            .directory
            .tensor(name)
            .ok_or_else(|| invalid(format!("missing GGUF tensor {name}")))?;
        // Shards are ordered by base and an empty shard shares its base with
        // its successor, so the last shard starting at or before the tensor
        // holds it.
        let shard = self
            .shards
            .iter()
            .rev()
            .find(|shard| shard.base <= tensor.offset)
            .ok_or_else(|| invalid(format!("GGUF tensor {name} lies before every shard")))?;
        let offset = shard
            .data_offset
            .checked_add(tensor.offset - shard.base)
            .ok_or_else(|| invalid("GGUF absolute tensor offset overflows"))?;
        Ok(TensorSlice {
            source: shard.source.clone(),
            offset,
            nbytes: tensor.nbytes,
            shape: tensor.shape.clone(),
            encoding: tensor.encoding,
        })
    }
}
