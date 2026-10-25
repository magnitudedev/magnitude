//! Bounded GGUF directory validation and open container ownership.
//!
//! Tensor bytes are not decoded or uploaded here. `TensorSlice` describes the
//! validated stored range; residency code chooses its numerical interpretation.

use crate::{ArtifactIdentity, Error, FileSource};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

mod walk;

pub use walk::{
    Entry, EntryHead, HEADER_LIMIT, Header, Preamble, Prefix, TensorInfo, Walk, read_header_bytes,
    walk,
};

/// The `general.architecture` a multimodal projector declares.
pub const PROJECTOR_ARCHITECTURE: &str = "clip";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
}

pub use magnitude_ggml::Encoding;

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
    Array(Array),
}

/// One GGUF metadata array. A GGUF array has a single element type, so its elements are held
/// unboxed in one buffer of that type rather than as a [`Scalar`] (and, for text, an allocation)
/// apiece: a tokenizer's hundreds of thousands of pieces, merges, scores and kinds then take about
/// the bytes they occupy in the file, where boxed scalars took several times that.
#[derive(Clone, Debug, PartialEq)]
pub enum Array {
    String(Strings),
    Bool(Vec<bool>),
    Unsigned(Vec<u64>),
    Signed(Vec<i64>),
    Float(Vec<f64>),
}

/// The elements of a text array, concatenated, with where each one ends.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Strings {
    text: String,
    ends: Vec<u32>,
}

/// One element of an [`Array`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Element<'a> {
    String(&'a str),
    Bool(bool),
    Unsigned(u64),
    Signed(i64),
    Float(f64),
}

impl Strings {
    fn push(&mut self, value: &str) -> Result<(), Error> {
        self.text.push_str(value);
        self.ends.push(
            u32::try_from(self.text.len())
                .map_err(|_| invalid("GGUF string array exceeds its addressable length"))?,
        );
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&str> {
        let end = *self.ends.get(index)? as usize;
        let start = index
            .checked_sub(1)
            .map_or(0, |previous| self.ends[previous] as usize);
        Some(&self.text[start..end])
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &str> + '_ {
        (0..self.len()).map(|index| self.get(index).expect("an index below the length"))
    }
}

impl<S: AsRef<str>> FromIterator<S> for Strings {
    fn from_iter<I: IntoIterator<Item = S>>(values: I) -> Self {
        let mut strings = Self::default();
        for value in values {
            strings
                .push(value.as_ref())
                .expect("an in-memory string array within its addressable length");
        }
        strings
    }
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Self::String(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::Unsigned(values) => values.len(),
            Self::Signed(values) => values.len(),
            Self::Float(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<Element<'_>> {
        Some(match self {
            Self::String(values) => Element::String(values.get(index)?),
            Self::Bool(values) => Element::Bool(*values.get(index)?),
            Self::Unsigned(values) => Element::Unsigned(*values.get(index)?),
            Self::Signed(values) => Element::Signed(*values.get(index)?),
            Self::Float(values) => Element::Float(*values.get(index)?),
        })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Element<'_>> + '_ {
        (0..self.len()).map(|index| self.get(index).expect("an index below the length"))
    }

    /// The array of `values`, which must share one element type.
    pub fn of(values: impl IntoIterator<Item = Scalar>) -> Result<Self, Error> {
        let mut values = values.into_iter().peekable();
        let mismatch = || invalid("GGUF array elements must share one type");
        Ok(match values.peek() {
            None | Some(Scalar::Unsigned(_)) => Self::Unsigned(
                values
                    .map(|value| match value {
                        Scalar::Unsigned(value) => Ok(value),
                        _ => Err(mismatch()),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            Some(Scalar::Signed(_)) => Self::Signed(
                values
                    .map(|value| match value {
                        Scalar::Signed(value) => Ok(value),
                        _ => Err(mismatch()),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            Some(Scalar::Float(_)) => Self::Float(
                values
                    .map(|value| match value {
                        Scalar::Float(value) => Ok(value),
                        _ => Err(mismatch()),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            Some(Scalar::Bool(_)) => Self::Bool(
                values
                    .map(|value| match value {
                        Scalar::Bool(value) => Ok(value),
                        _ => Err(mismatch()),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            Some(Scalar::String(_)) => {
                let mut strings = Strings::default();
                for value in values {
                    match value {
                        Scalar::String(value) => strings.push(&value)?,
                        _ => return Err(mismatch()),
                    }
                }
                Self::String(strings)
            }
        })
    }
}

impl From<Vec<bool>> for Array {
    fn from(values: Vec<bool>) -> Self {
        Self::Bool(values)
    }
}

impl From<Vec<u64>> for Array {
    fn from(values: Vec<u64>) -> Self {
        Self::Unsigned(values)
    }
}

impl From<Vec<i64>> for Array {
    fn from(values: Vec<i64>) -> Self {
        Self::Signed(values)
    }
}

impl From<Vec<f64>> for Array {
    fn from(values: Vec<f64>) -> Self {
        Self::Float(values)
    }
}

impl From<Strings> for Array {
    fn from(values: Strings) -> Self {
        Self::String(values)
    }
}

impl Element<'_> {
    pub fn unsigned(self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(value),
            Self::Signed(value) => u64::try_from(value).ok(),
            _ => None,
        }
    }

    pub fn to_scalar(self) -> Scalar {
        match self {
            Self::String(value) => Scalar::String(value.to_owned()),
            Self::Bool(value) => Scalar::Bool(value),
            Self::Unsigned(value) => Scalar::Unsigned(value),
            Self::Signed(value) => Scalar::Signed(value),
            Self::Float(value) => Scalar::Float(value),
        }
    }
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

/// A GGUF metadata value type (`gguf_type`): how a scalar, or each element of an array, is
/// stored. [`Value`] keeps only the kind of number, so a header keeps these to encode its values
/// back to their bytes ([`StoredHeader`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    U64,
    I64,
    F64,
}

impl ValueType {
    fn from_id(id: u32) -> Result<Self, Error> {
        Ok(match id {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return Err(invalid(format!("unsupported GGUF metadata type {id}"))),
        })
    }

    /// The bytes one value of this type takes; for a string, the least (its length).
    fn width(self) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::String | Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    fn id(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }

    /// The bytes of `element` stored as this type (little-endian), or an error when it is
    /// another kind of value or out of this type's range.
    fn encode(self, element: Element<'_>, out: &mut Vec<u8>) -> Result<(), Error> {
        let range = || invalid(format!("GGUF value {element:?} does not fit {self:?}"));
        match (self, element) {
            (Self::U8, Element::Unsigned(value)) => {
                out.push(u8::try_from(value).map_err(|_| range())?);
            }
            (Self::I8, Element::Signed(value)) => {
                out.extend(i8::try_from(value).map_err(|_| range())?.to_le_bytes());
            }
            (Self::U16, Element::Unsigned(value)) => {
                out.extend(u16::try_from(value).map_err(|_| range())?.to_le_bytes());
            }
            (Self::I16, Element::Signed(value)) => {
                out.extend(i16::try_from(value).map_err(|_| range())?.to_le_bytes());
            }
            (Self::U32, Element::Unsigned(value)) => {
                out.extend(u32::try_from(value).map_err(|_| range())?.to_le_bytes());
            }
            (Self::I32, Element::Signed(value)) => {
                out.extend(i32::try_from(value).map_err(|_| range())?.to_le_bytes());
            }
            (Self::F32, Element::Float(value)) => {
                let stored = value as f32;
                if f64::from(stored).to_bits() != value.to_bits() && !value.is_nan() {
                    return Err(range());
                }
                out.extend(stored.to_le_bytes());
            }
            (Self::Bool, Element::Bool(value)) => out.push(u8::from(value)),
            (Self::String, Element::String(value)) => encode_string(value, out),
            (Self::U64, Element::Unsigned(value)) => out.extend(value.to_le_bytes()),
            (Self::I64, Element::Signed(value)) => out.extend(value.to_le_bytes()),
            (Self::F64, Element::Float(value)) => out.extend(value.to_le_bytes()),
            _ => return Err(range()),
        }
        Ok(())
    }
}

/// How one metadata value is stored: a scalar, or an array of elements of one type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredType {
    Scalar(ValueType),
    Array(ValueType),
}

/// One metadata entry of a [`StoredHeader`].
#[derive(Clone, Debug, PartialEq)]
pub struct StoredMetadata {
    pub metadata: Metadata,
    pub stored: StoredType,
}

/// One little-endian GGUF file's header as it is stored: its version, its metadata with each
/// value's stored type, and its tensor directory in stored order, so it encodes back to the bytes
/// it was read from ([`StoredHeader::encode`]) and an edited copy keeps every value's type.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredHeader {
    pub version: u32,
    pub metadata: Vec<StoredMetadata>,
    /// Offsets are relative to the data section.
    pub tensors: Vec<TensorDescriptor>,
}

impl StoredHeader {
    /// The header of the GGUF file `source`; tensor payloads need not follow it.
    pub fn read<R: Read + Seek>(source: &mut R) -> Result<Self, Error> {
        let (directory, types) = read_stored_directory(source, false)?;
        if directory.byte_order != ByteOrder::Little {
            return Err(invalid("a stored GGUF header must be little-endian"));
        }
        Ok(Self {
            version: directory.version,
            metadata: directory
                .metadata
                .into_iter()
                .zip(types)
                .map(|(metadata, stored)| StoredMetadata { metadata, stored })
                .collect(),
            tensors: directory.tensors,
        })
    }

    /// A header whose `tensors` (name, outermost-first shape and encoding) are stored in order,
    /// each at the alignment boundary after the one before.
    pub fn new(
        version: u32,
        metadata: Vec<StoredMetadata>,
        tensors: impl IntoIterator<Item = (String, Vec<u64>, Encoding)>,
    ) -> Result<Self, Error> {
        let mut header = Self {
            version,
            metadata,
            tensors: Vec::new(),
        };
        let alignment = header.alignment()?;
        let mut offset = 0u64;
        for (name, shape, encoding) in tensors {
            if shape.contains(&0)
                || shape
                    .last()
                    .is_none_or(|row| row % encoding.block_elements() != 0)
            {
                return Err(invalid(format!(
                    "invalid block geometry on tensor {name:?}"
                )));
            }
            let nbytes = stored_bytes(&name, &shape, encoding)?;
            header.tensors.push(TensorDescriptor {
                name,
                shape,
                encoding,
                offset,
                nbytes,
            });
            offset = aligned(
                offset
                    .checked_add(nbytes)
                    .ok_or_else(|| invalid("GGUF data section overflows"))?,
                alignment,
            )?;
        }
        Ok(header)
    }

    pub fn value(&self, name: &str) -> Option<&Value> {
        self.metadata
            .iter()
            .find(|entry| entry.metadata.name == name)
            .map(|entry| &entry.metadata.value)
    }

    /// `general.alignment`, or GGUF's default of 32.
    pub fn alignment(&self) -> Result<u64, Error> {
        self.value("general.alignment").map_or(Ok(32), alignment)
    }

    /// Every byte of the header through its aligned tensor-data offset.
    pub fn encode(&self) -> Result<Vec<u8>, Error> {
        let count = |count: usize| u64::try_from(count).expect("a count fits u64");
        let mut out = b"GGUF".to_vec();
        out.extend(self.version.to_le_bytes());
        out.extend(count(self.tensors.len()).to_le_bytes());
        out.extend(count(self.metadata.len()).to_le_bytes());
        for entry in &self.metadata {
            encode_string(&entry.metadata.name, &mut out);
            match (entry.stored, &entry.metadata.value) {
                (StoredType::Scalar(stored), Value::Scalar(scalar)) => {
                    out.extend(stored.id().to_le_bytes());
                    stored.encode(scalar_element(scalar), &mut out)?;
                }
                (StoredType::Array(stored), Value::Array(array)) => {
                    out.extend(9u32.to_le_bytes());
                    out.extend(stored.id().to_le_bytes());
                    out.extend(count(array.len()).to_le_bytes());
                    for element in array.iter() {
                        stored.encode(element, &mut out)?;
                    }
                }
                _ => {
                    return Err(invalid(format!(
                        "metadata {:?} is not stored as {:?}",
                        entry.metadata.name, entry.stored
                    )));
                }
            }
        }
        for tensor in &self.tensors {
            encode_string(&tensor.name, &mut out);
            out.extend(
                u32::try_from(tensor.shape.len())
                    .expect("a tensor rank fits u32")
                    .to_le_bytes(),
            );
            for dimension in tensor.shape.iter().rev() {
                out.extend(dimension.to_le_bytes());
            }
            out.extend((tensor.encoding as u32).to_le_bytes());
            out.extend(tensor.offset.to_le_bytes());
        }
        let length = aligned(count(out.len()), self.alignment()?)?;
        out.resize(
            usize::try_from(length).map_err(|_| invalid("GGUF header exceeds host address range"))?,
            0,
        );
        Ok(out)
    }
}

/// The tensor-data alignment a `general.alignment` value declares: a power of two, stored as an
/// integer of any width.
fn alignment(value: &Value) -> Result<u64, Error> {
    value
        .unsigned()
        .filter(|alignment| alignment.is_power_of_two())
        .ok_or_else(|| invalid("alignment must be a positive power of two"))
}

fn scalar_element(scalar: &Scalar) -> Element<'_> {
    match scalar {
        Scalar::String(value) => Element::String(value),
        Scalar::Bool(value) => Element::Bool(*value),
        Scalar::Unsigned(value) => Element::Unsigned(*value),
        Scalar::Signed(value) => Element::Signed(*value),
        Scalar::Float(value) => Element::Float(*value),
    }
}

fn encode_string(value: &str, out: &mut Vec<u8>) {
    out.extend(
        u64::try_from(value.len())
            .expect("a length fits u64")
            .to_le_bytes(),
    );
    out.extend_from_slice(value.as_bytes());
}

fn aligned(offset: u64, alignment: u64) -> Result<u64, Error> {
    offset
        .checked_add(alignment - 1)
        .map(|offset| offset & !(alignment - 1))
        .ok_or_else(|| invalid("GGUF data alignment overflows"))
}

/// The bytes a tensor of `shape` (in either order) stores in `encoding`.
fn stored_bytes(name: &str, shape: &[u64], encoding: Encoding) -> Result<u64, Error> {
    let elements = shape
        .iter()
        .try_fold(1u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| invalid(format!("tensor {name:?} element count overflows")))?;
    (elements / encoding.block_elements())
        .checked_mul(encoding.block_bytes())
        .ok_or_else(|| invalid(format!("tensor {name:?} byte size overflows")))
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

pub fn read_directory<R: Read + Seek>(source: &mut R) -> Result<Directory, Error> {
    read_stored_directory(source, true).map(|(directory, _)| directory)
}

/// Read a GGUF header held in memory (every byte through the aligned data
/// offset); tensor payloads need not follow it.
pub fn read_header(bytes: &[u8]) -> Result<Directory, Error> {
    stored_directory(bytes, bytes.len() as u64, false).map(|(directory, _)| directory)
}

/// Inspect one downloaded GGUF file header before weight payloads exist.
/// Header bounds, tensor geometry, offsets, and non-overlap are validated;
/// only the physical presence of declared tensor bytes is deferred.
pub fn inspect_header(path: impl AsRef<Path>) -> Result<Directory, Error> {
    let source = FileSource::open(path)?;
    read_file_directory(&source, false)
}

/// One GGUF file's validated header directory and logical size.
#[derive(Clone, Debug)]
pub struct FileHeader {
    pub directory: Directory,
    pub size: u64,
}

/// Where header-only inspection reads GGUF headers: the files themselves
/// ([`FileHeaders`]), or a host that already holds them, so assessing one
/// file several ways never rereads it. A header is the file's bytes through
/// its aligned tensor-data offset, parsed by [`read_header`].
pub trait HeaderSource {
    fn header(&self, path: &Path) -> Result<FileHeader, Error>;
}

/// Headers read from the files they describe.
pub struct FileHeaders;

impl HeaderSource for FileHeaders {
    fn header(&self, path: &Path) -> Result<FileHeader, Error> {
        let source = FileSource::open(path)?;
        Ok(FileHeader {
            directory: read_file_directory(&source, false)?,
            size: source.size(),
        })
    }
}

/// Inspect the headers of the GGUF component `path` names: the file itself,
/// or every shard when it is the first shard of a split GGUF (see
/// [`split_paths`]). Tensor payloads need not exist. Returns the component
/// directory and its files (absolute, in shard order) with their sizes.
pub(crate) fn inspect_component_header(
    path: &Path,
    headers: &dyn HeaderSource,
) -> Result<(Directory, Vec<(PathBuf, u64)>), Error> {
    let read = |path: &Path| -> Result<(FileHeader, PathBuf), Error> {
        let header = headers.header(path)?;
        header.directory.require_execution_byte_order()?;
        Ok((header, std::path::absolute(path)?))
    };
    let (first, first_path) = read(path)?;
    match split_paths(path, &first.directory)? {
        None => Ok((first.directory, vec![(first_path, first.size)])),
        Some(paths) => {
            let mut files = vec![(first_path, first.size)];
            let mut shards = vec![first.directory];
            for shard in &paths[1..] {
                let (header, path) = read(shard)?;
                files.push((path, header.size));
                shards.push(header.directory);
            }
            Ok((merge_shards(shards)?.0, files))
        }
    }
}

fn read_file_directory(source: &FileSource, require_payload: bool) -> Result<Directory, Error> {
    let (directory, _) = read_stored_directory(&mut source.reader(), require_payload)?;
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

/// The directory of a GGUF file with the stored type of each of its metadata values.
fn read_stored_directory<R: Read + Seek>(
    source: &mut R,
    require_payload: bool,
) -> Result<(Directory, Vec<StoredType>), Error> {
    let size = source.seek(SeekFrom::End(0))?;
    source.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    read_header_bytes(source, &mut bytes)?;
    stored_directory(&bytes, size, require_payload)
}

/// The directory of the GGUF header `bytes` begin, for a file of `size` bytes, with the stored
/// type of each of its metadata values. With `require_payload`, every tensor's bytes must lie
/// within the file.
fn stored_directory(
    bytes: &[u8],
    size: u64,
    require_payload: bool,
) -> Result<(Directory, Vec<StoredType>), Error> {
    let Walk::Complete(header) = walk(bytes)? else {
        return Err(Error::Truncated {
            offset: bytes.len() as u64,
        });
    };
    let (metadata, types) = header
        .entries
        .iter()
        .map(|entry| (entry.metadata(), entry.head.stored))
        .unzip();
    let alignment = header.alignment;
    let mut tensors = Vec::new();
    let mut unknown = std::collections::BTreeMap::<u32, u32>::new();
    let mut names = HashSet::new();
    for tensor in &header.tensors {
        let name = tensor.name;
        if name.is_empty() || !names.insert(name) {
            return Err(invalid(format!(
                "invalid or duplicate tensor directory entry {name:?}"
            )));
        }
        // A type id this parser doesn't know is a newer quantization, not a damaged header: the
        // rest of the directory is still read, and the ids are reported together.
        let Some(encoding) = Encoding::from_id(tensor.type_id) else {
            *unknown.entry(tensor.type_id).or_default() += 1;
            continue;
        };
        let mut shape = tensor.dimensions().collect::<Vec<_>>();
        if shape.contains(&0) || shape[0] % encoding.block_elements() != 0 {
            return Err(invalid(format!(
                "invalid block geometry on tensor {name:?}"
            )));
        }
        if tensor.offset % alignment != 0 {
            return Err(invalid(format!("misaligned tensor {name:?}")));
        }
        let nbytes = stored_bytes(name, &shape, encoding)?;
        shape.reverse();
        tensors.push(TensorDescriptor {
            name: name.to_owned(),
            shape,
            encoding,
            offset: tensor.offset,
            nbytes,
        });
    }

    let data_offset = header.data_offset as u64;
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

    let directory = Directory {
        version: header.preamble.version,
        byte_order: header.preamble.byte_order,
        alignment,
        data_offset,
        metadata,
        tensors,
    };
    if !unknown.is_empty() {
        return Err(Error::UnknownTensorTypes {
            types: unknown.into_iter().collect(),
            directory: Box::new(directory),
        });
    }
    Ok((directory, types))
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
