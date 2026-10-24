//! The structure of a GGUF header, read from any prefix of its bytes.
//!
//! [`walk`] visits the header in the bytes in hand: its preamble, each metadata entry with its
//! byte range, and each tensor descriptor, checking the container's structure as it goes. When
//! the bytes end inside the header it says which entry they end in and how many more bytes the
//! header needs: certainly (`minimum`) and probably (`estimate`, extrapolated inside a string
//! array), so a reader growing a prefix (from a file, or by range requests) asks for about as many
//! as remain. Values are decoded only on request ([`Entry::value`]). Whether the engine accepts
//! the tensors themselves (their types, geometry and offsets) is decided on the complete header
//! ([`super::read_header`]).

use super::{
    Array, ByteOrder, Error, Metadata, Scalar, StoredType, Strings, Value, ValueType, invalid,
};
use std::{collections::HashSet, io::Read, ops::Range};

/// The largest GGUF header, through its aligned tensor-data offset.
pub const HEADER_LIMIT: usize = 256 * 1024 * 1024;
// A string array's element ends are stored as u32 offsets into its text.
const _: () = assert!(HEADER_LIMIT <= u32::MAX as usize);

/// Magic, version, tensor count and metadata count.
const PREAMBLE_BYTES: usize = 24;
/// The smallest metadata entry or tensor descriptor, bounding how many a header can declare.
const SMALLEST_ITEM_BYTES: u64 = 12;
const DEFAULT_ALIGNMENT: u64 = 32;
const ALIGNMENT_KEY: &str = "general.alignment";
/// ggml's `GGML_MAX_DIMS`; ggml refuses a GGUF tensor of higher rank.
const MAX_TENSOR_RANK: u32 = 4;

/// Bytes assumed for a metadata entry or tensor descriptor not yet seen.
const ENTRY_ESTIMATE: usize = 64;
const TENSOR_ESTIMATE: usize = 96;
/// The bytes assumed per string of a string array (a token or merge), until enough strings are in
/// hand to measure them, and the bounds of that measure: the first strings of a list are no sample
/// of the rest (merges begin with long runs of whitespace).
const STRING_ESTIMATE: usize = 16;
const STRING_ESTIMATE_BOUNDS: (usize, usize) = (8, 24);
const MEASURED_STRINGS: usize = 1024;

/// What a prefix of a GGUF header holds.
#[derive(Clone, Debug, PartialEq)]
pub enum Walk<'a> {
    Complete(Header<'a>),
    Incomplete(Prefix<'a>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Preamble {
    pub version: u32,
    pub byte_order: ByteOrder,
}

/// A whole header.
#[derive(Clone, Debug, PartialEq)]
pub struct Header<'a> {
    pub preamble: Preamble,
    pub entries: Vec<Entry<'a>>,
    pub tensors: Vec<TensorInfo<'a>>,
    pub alignment: u64,
    /// The aligned tensor-data offset, which may lie past the bytes when a file without tensor
    /// data ends before its padding.
    pub data_offset: usize,
}

/// The bytes end inside the header.
#[derive(Clone, Debug, PartialEq)]
pub struct Prefix<'a> {
    /// `None` while the bytes end inside the preamble.
    pub preamble: Option<Preamble>,
    pub entries: Vec<Entry<'a>>,
    /// The alignment the entries in hand declare.
    pub alignment: u64,
    /// The entry the bytes end in, when its fixed part is in hand.
    pub open: Option<EntryHead<'a>>,
    /// More bytes the header certainly needs; at least one, and within [`HEADER_LIMIT`].
    pub minimum: usize,
    /// More bytes the header probably needs; at least `minimum`, and within [`HEADER_LIMIT`].
    pub estimate: usize,
}

/// A metadata entry's fixed part: its key, how its value is stored and how many values it holds
/// (one for a scalar), which ends (`end`) after a string's length or an array's element count. Two
/// entries holding the same kind and amount of data have this part byte for byte in common.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryHead<'a> {
    pub key: &'a str,
    pub stored: StoredType,
    pub count: u64,
    pub start: usize,
    pub end: usize,
}

/// One complete metadata entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry<'a> {
    pub head: EntryHead<'a>,
    pub end: usize,
    /// The value's bytes: everything after the value type (a scalar) or the element count (an
    /// array).
    value: &'a [u8],
    order: ByteOrder,
}

impl Entry<'_> {
    /// The entry's bytes in the header.
    pub fn range(&self) -> Range<usize> {
        self.head.start..self.end
    }

    /// The entry's key and value, decoded.
    pub fn metadata(&self) -> Metadata {
        Metadata {
            name: self.head.key.to_owned(),
            value: self.value(),
        }
    }

    /// The entry's value, decoded.
    pub fn value(&self) -> Value {
        let mut decoder = Decoder {
            bytes: self.value,
            order: self.order,
        };
        match self.head.stored {
            StoredType::Scalar(kind) => Value::Scalar(decoder.scalar(kind)),
            StoredType::Array(kind) => {
                let count = usize::try_from(self.head.count).expect("a walked count fits usize");
                Value::Array(match kind {
                    ValueType::String => {
                        Array::String((0..count).map(|_| decoder.string()).collect::<Strings>())
                    }
                    ValueType::Bool => Array::Bool((0..count).map(|_| decoder.bool()).collect()),
                    ValueType::U8 | ValueType::U16 | ValueType::U32 | ValueType::U64 => {
                        Array::Unsigned((0..count).map(|_| decoder.unsigned(kind)).collect())
                    }
                    ValueType::I8 | ValueType::I16 | ValueType::I32 | ValueType::I64 => {
                        Array::Signed((0..count).map(|_| decoder.signed(kind)).collect())
                    }
                    ValueType::F32 | ValueType::F64 => {
                        Array::Float((0..count).map(|_| decoder.float(kind)).collect())
                    }
                })
            }
        }
    }
}

/// One tensor descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    /// The ggml type id, which may be one this crate doesn't know.
    pub type_id: u32,
    /// Byte offset relative to the data section.
    pub offset: u64,
    dimensions: &'a [u8],
    order: ByteOrder,
}

impl TensorInfo<'_> {
    /// The dimensions in ggml storage order, innermost first.
    pub fn dimensions(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.dimensions
            .chunks_exact(8)
            .map(|bytes| u64_of(self.order, bytes.try_into().expect("eight bytes")))
    }
}

/// The structure of the GGUF header `bytes` begin, or why they can't begin one.
pub fn walk(bytes: &[u8]) -> Result<Walk<'_>, Error> {
    let mut walker = Walker {
        cursor: Cursor {
            bytes,
            offset: 0,
            order: ByteOrder::Little,
        },
        preamble: None,
        entries: Vec::new(),
        alignment: DEFAULT_ALIGNMENT,
    };
    let Some(magic) = walker.cursor.take(4)? else {
        let missing = PREAMBLE_BYTES - bytes.len();
        return walker.stop(None, missing, missing);
    };
    if magic != b"GGUF" {
        return Err(invalid("not a GGUF container"));
    }
    let Some(version) = walker.cursor.take(4)? else {
        let missing = PREAMBLE_BYTES - bytes.len();
        return walker.stop(None, missing, missing);
    };
    if version == [0, 0, 0, 3] {
        walker.cursor.order = ByteOrder::Big;
    }
    let version = u32_of(walker.cursor.order, version.try_into().expect("four bytes"));
    let (Some(tensor_count), Some(entry_count)) = (walker.cursor.u64()?, walker.cursor.u64()?)
    else {
        let missing = PREAMBLE_BYTES - bytes.len();
        return walker.stop(None, missing, missing);
    };
    if !matches!(version, 2 | 3) {
        return Err(invalid(format!("unsupported GGUF version {version}")));
    }
    let preamble = Preamble {
        version,
        byte_order: walker.cursor.order,
    };
    walker.preamble = Some(preamble);
    if tensor_count
        .checked_add(entry_count)
        .is_none_or(|count| count > (HEADER_LIMIT - PREAMBLE_BYTES) as u64 / SMALLEST_ITEM_BYTES)
    {
        return Err(beyond());
    }
    let (tensor_count, entry_count) = (tensor_count as usize, entry_count as usize);
    let tensors_estimate = tensor_count * TENSOR_ESTIMATE;
    let mut keys = HashSet::new();
    for index in 0..entry_count {
        let rest = (entry_count - index - 1) * ENTRY_ESTIMATE + tensors_estimate;
        let start = walker.cursor.offset;
        let Some(key) = walker.cursor.string()? else {
            return walker.stop(None, 12, ENTRY_ESTIMATE + rest);
        };
        if !keys.insert(key) {
            return Err(invalid(format!("duplicate metadata {key:?}")));
        }
        let Some(value_type) = walker.cursor.u32()? else {
            return walker.stop(None, 4, ENTRY_ESTIMATE + rest);
        };
        let (stored, count, head_end, value_start) = if value_type == 9 {
            let (Some(element), Some(count)) = (walker.cursor.u32()?, walker.cursor.u64()?) else {
                return walker.stop(None, 12, ENTRY_ESTIMATE + rest);
            };
            if element == 9 {
                return Err(invalid("nested or oversized GGUF array"));
            }
            let element = ValueType::from_id(element)?;
            let head_end = walker.cursor.offset;
            let open = EntryHead {
                key,
                stored: StoredType::Array(element),
                count,
                start,
                end: head_end,
            };
            // Every element takes at least its type's width, so the count is bounded before
            // anything is read.
            let elements = usize::try_from(count)
                .ok()
                .filter(|elements| {
                    elements
                        .checked_mul(element.width())
                        .is_some_and(|length| length <= HEADER_LIMIT - head_end)
                })
                .ok_or_else(|| invalid("nested or oversized GGUF array"))?;
            if element == ValueType::String {
                for read in 0..elements {
                    let before = walker.cursor.offset;
                    if walker.cursor.string()?.is_none() {
                        let left = elements - read;
                        let average = if read < MEASURED_STRINGS {
                            STRING_ESTIMATE
                        } else {
                            ((before - head_end) / read)
                                .clamp(STRING_ESTIMATE_BOUNDS.0, STRING_ESTIMATE_BOUNDS.1)
                        };
                        let in_string = bytes.len() - before;
                        return walker.stop(
                            Some(open),
                            (left * 8).saturating_sub(in_string),
                            (left * average).saturating_sub(in_string) + rest,
                        );
                    }
                }
            } else {
                let length = elements * element.width();
                let Some(values) = walker.cursor.take(length)? else {
                    let missing = head_end + length - bytes.len();
                    return walker.stop(Some(open), missing, missing + rest);
                };
                if element == ValueType::Bool {
                    values.iter().try_for_each(|byte| valid_bool(*byte))?;
                }
            }
            (StoredType::Array(element), count, head_end, head_end)
        } else {
            let kind = ValueType::from_id(value_type)?;
            let value_start = walker.cursor.offset;
            if kind == ValueType::String {
                let Some(length) = walker.cursor.u64()? else {
                    return walker.stop(None, 8, ENTRY_ESTIMATE + rest);
                };
                let head_end = walker.cursor.offset;
                let length = usize::try_from(length)
                    .ok()
                    .filter(|length| *length <= HEADER_LIMIT - head_end)
                    .ok_or_else(beyond)?;
                let Some(text) = walker.cursor.take(length)? else {
                    let missing = head_end + length - bytes.len();
                    let open = EntryHead {
                        key,
                        stored: StoredType::Scalar(kind),
                        count: 1,
                        start,
                        end: head_end,
                    };
                    return walker.stop(Some(open), missing, missing + rest);
                };
                utf8(text)?;
                (StoredType::Scalar(kind), 1, head_end, value_start)
            } else {
                let width = kind.width();
                let Some(value) = walker.cursor.take(width)? else {
                    return walker.stop(None, width, width + rest);
                };
                if kind == ValueType::Bool {
                    valid_bool(value[0])?;
                }
                (StoredType::Scalar(kind), 1, value_start, value_start)
            }
        };
        let entry = Entry {
            head: EntryHead {
                key,
                stored,
                count,
                start,
                end: head_end,
            },
            end: walker.cursor.offset,
            value: &bytes[value_start..walker.cursor.offset],
            order: walker.cursor.order,
        };
        if key == ALIGNMENT_KEY {
            walker.alignment = super::alignment(&entry.value())?;
        }
        walker.entries.push(entry);
    }
    let alignment = usize::try_from(walker.alignment).unwrap_or(usize::MAX);
    let tensors_start = walker.cursor.offset;
    let mut tensors = Vec::new();
    for index in 0..tensor_count {
        let before = walker.cursor.offset;
        let left = tensor_count - index;
        let average = if index == 0 {
            TENSOR_ESTIMATE
        } else {
            (before - tensors_start) / index
        };
        let estimate = (left * average)
            .saturating_add(alignment)
            .saturating_sub(bytes.len() - before);
        let Some(name) = walker.cursor.string()? else {
            return walker.stop(None, 8, estimate);
        };
        let Some(rank) = walker.cursor.u32()? else {
            return walker.stop(None, 4, estimate);
        };
        if !(1..=MAX_TENSOR_RANK).contains(&rank) {
            return Err(invalid(format!(
                "invalid or duplicate tensor directory entry {name:?}"
            )));
        }
        let fixed = rank as usize * 8 + 4 + 8;
        let Some(descriptor) = walker.cursor.take(fixed)? else {
            return walker.stop(None, fixed, estimate);
        };
        let (dimensions, rest) = descriptor.split_at(rank as usize * 8);
        let (type_id, offset) = rest.split_at(4);
        let order = walker.cursor.order;
        tensors.push(TensorInfo {
            name,
            type_id: u32_of(order, type_id.try_into().expect("four bytes")),
            offset: u64_of(order, offset.try_into().expect("eight bytes")),
            dimensions,
            order,
        });
    }
    let data_offset = walker
        .cursor
        .offset
        .checked_next_multiple_of(alignment)
        .filter(|end| *end <= HEADER_LIMIT)
        .ok_or_else(beyond)?;
    Ok(Walk::Complete(Header {
        preamble,
        entries: walker.entries,
        tensors,
        alignment: walker.alignment,
        data_offset,
    }))
}

/// Reads from `source` onto `bytes` until they hold a whole GGUF header (padding aside), returning
/// its aligned tensor-data offset. Each read asks for the walk's estimate of the rest, so the bytes
/// may run past the header by up to the last estimate. A source that ends first is a truncated
/// header.
pub fn read_header_bytes(source: &mut impl Read, bytes: &mut Vec<u8>) -> Result<usize, Error> {
    loop {
        let wanted = match walk(bytes)? {
            Walk::Complete(header) => return Ok(header.data_offset),
            Walk::Incomplete(prefix) => prefix.estimate,
        };
        if source.by_ref().take(wanted as u64).read_to_end(bytes)? == 0 {
            return Err(Error::Truncated {
                offset: bytes.len() as u64,
            });
        }
    }
}

fn beyond() -> Error {
    invalid(format!("GGUF header extends beyond {HEADER_LIMIT} bytes"))
}

fn utf8(bytes: &[u8]) -> Result<&str, Error> {
    std::str::from_utf8(bytes).map_err(|_| invalid("invalid UTF-8 in GGUF header"))
}

fn valid_bool(byte: u8) -> Result<(), Error> {
    if byte <= 1 {
        Ok(())
    } else {
        Err(invalid("invalid GGUF boolean"))
    }
}

fn u16_of(order: ByteOrder, bytes: [u8; 2]) -> u16 {
    match order {
        ByteOrder::Little => u16::from_le_bytes(bytes),
        ByteOrder::Big => u16::from_be_bytes(bytes),
    }
}

fn u32_of(order: ByteOrder, bytes: [u8; 4]) -> u32 {
    match order {
        ByteOrder::Little => u32::from_le_bytes(bytes),
        ByteOrder::Big => u32::from_be_bytes(bytes),
    }
}

fn u64_of(order: ByteOrder, bytes: [u8; 8]) -> u64 {
    match order {
        ByteOrder::Little => u64::from_le_bytes(bytes),
        ByteOrder::Big => u64::from_be_bytes(bytes),
    }
}

/// The walk's state: what it has read, and what it reports where the bytes end.
struct Walker<'a> {
    cursor: Cursor<'a>,
    preamble: Option<Preamble>,
    entries: Vec<Entry<'a>>,
    alignment: u64,
}

impl<'a> Walker<'a> {
    fn stop(
        self,
        open: Option<EntryHead<'a>>,
        minimum: usize,
        estimate: usize,
    ) -> Result<Walk<'a>, Error> {
        let room = HEADER_LIMIT.saturating_sub(self.cursor.bytes.len());
        let minimum = minimum.max(1);
        if minimum > room {
            return Err(beyond());
        }
        Ok(Walk::Incomplete(Prefix {
            preamble: self.preamble,
            entries: self.entries,
            alignment: self.alignment,
            open,
            minimum,
            estimate: estimate.clamp(minimum, room),
        }))
    }
}

/// Reads the bytes in hand. `None` is bytes that end first (the cursor then stays where it was);
/// an error is a read that would end past [`HEADER_LIMIT`].
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    order: ByteOrder,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, length: usize) -> Result<Option<&'a [u8]>, Error> {
        let end = self
            .offset
            .checked_add(length)
            .filter(|end| *end <= HEADER_LIMIT)
            .ok_or_else(beyond)?;
        let Some(taken) = self.bytes.get(self.offset..end) else {
            return Ok(None);
        };
        self.offset = end;
        Ok(Some(taken))
    }

    fn u32(&mut self) -> Result<Option<u32>, Error> {
        Ok(self
            .take(4)?
            .map(|bytes| u32_of(self.order, bytes.try_into().expect("four bytes"))))
    }

    fn u64(&mut self) -> Result<Option<u64>, Error> {
        Ok(self
            .take(8)?
            .map(|bytes| u64_of(self.order, bytes.try_into().expect("eight bytes"))))
    }

    /// A UTF-8 string.
    fn string(&mut self) -> Result<Option<&'a str>, Error> {
        let start = self.offset;
        let Some(length) = self.u64()? else {
            return Ok(None);
        };
        let length = usize::try_from(length).map_err(|_| beyond())?;
        match self.take(length) {
            Ok(Some(text)) => utf8(text).map(Some),
            Ok(None) => {
                self.offset = start;
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
}

/// Decodes the bytes of a value the walk checked.
struct Decoder<'a> {
    bytes: &'a [u8],
    order: ByteOrder,
}

impl<'a> Decoder<'a> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let (taken, rest) = self.bytes.split_first_chunk().expect("a walked value");
        self.bytes = rest;
        *taken
    }

    fn string(&mut self) -> &'a str {
        let length = u64_of(self.order, self.take()) as usize;
        let (text, rest) = self.bytes.split_at(length);
        self.bytes = rest;
        std::str::from_utf8(text).expect("a walked string is UTF-8")
    }

    fn bool(&mut self) -> bool {
        self.take::<1>()[0] == 1
    }

    fn unsigned(&mut self, kind: ValueType) -> u64 {
        match kind.width() {
            1 => u64::from(self.take::<1>()[0]),
            2 => u64::from(u16_of(self.order, self.take())),
            4 => u64::from(u32_of(self.order, self.take())),
            _ => u64_of(self.order, self.take()),
        }
    }

    fn signed(&mut self, kind: ValueType) -> i64 {
        match kind.width() {
            1 => i64::from(self.take::<1>()[0] as i8),
            2 => i64::from(u16_of(self.order, self.take()) as i16),
            4 => i64::from(u32_of(self.order, self.take()) as i32),
            _ => u64_of(self.order, self.take()) as i64,
        }
    }

    fn float(&mut self, kind: ValueType) -> f64 {
        match kind.width() {
            4 => f64::from(f32::from_bits(u32_of(self.order, self.take()))),
            _ => f64::from_bits(u64_of(self.order, self.take())),
        }
    }

    fn scalar(&mut self, kind: ValueType) -> Scalar {
        match kind {
            ValueType::String => Scalar::String(self.string().to_owned()),
            ValueType::Bool => Scalar::Bool(self.bool()),
            ValueType::U8 | ValueType::U16 | ValueType::U32 | ValueType::U64 => {
                Scalar::Unsigned(self.unsigned(kind))
            }
            ValueType::I8 | ValueType::I16 | ValueType::I32 | ValueType::I64 => {
                Scalar::Signed(self.signed(kind))
            }
            ValueType::F32 | ValueType::F64 => Scalar::Float(self.float(kind)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    /// A header with a name, a token list of `tokens` entries, a trailing quantization key and one
    /// tensor.
    fn header(tokens: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        string(&mut bytes, "general.name");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        string(&mut bytes, "model");
        string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&(tokens as u64).to_le_bytes());
        for token in 0..tokens {
            string(&mut bytes, &format!("token-{token:06}"));
        }
        string(&mut bytes, "general.file_type");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&15_u32.to_le_bytes());
        string(&mut bytes, "token_embd.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&64_u64.to_le_bytes());
        bytes.extend_from_slice(&(tokens as u64).to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes
    }

    #[test]
    fn a_whole_header_lists_its_entries_and_tensors_and_ends_at_its_aligned_data_offset() {
        let bytes = header(10_000);
        let Ok(Walk::Complete(header)) = walk(&bytes) else {
            panic!("a whole header");
        };
        assert_eq!(header.data_offset, bytes.len());
        assert_eq!(
            header
                .entries
                .iter()
                .map(|entry| entry.head.key)
                .collect::<Vec<_>>(),
            ["general.name", "tokenizer.ggml.tokens", "general.file_type"]
        );
        let tokens = &header.entries[1];
        assert_eq!(
            (tokens.head.stored, tokens.head.count),
            (StoredType::Array(ValueType::String), 10_000)
        );
        assert_eq!(
            &bytes[tokens.head.end - 8..tokens.head.end],
            &10_000_u64.to_le_bytes()
        );
        assert_eq!(header.entries[2].head.start, tokens.end);
        let Value::Array(Array::String(pieces)) = tokens.value() else {
            panic!("a text array");
        };
        assert_eq!(pieces.get(9_999), Some("token-009999"));
        assert_eq!(
            header.entries[2].value(),
            Value::Scalar(Scalar::Unsigned(15))
        );
        let tensor = &header.tensors[0];
        assert_eq!(
            (
                tensor.name,
                tensor.dimensions().collect::<Vec<_>>(),
                tensor.type_id,
                tensor.offset
            ),
            ("token_embd.weight", vec![64, 10_000], 0, 0)
        );
    }

    #[test]
    fn a_prefix_names_the_entry_it_ends_in_and_estimates_the_rest() {
        let bytes = header(20_000);
        let Ok(Walk::Complete(header)) = walk(&bytes) else {
            panic!("a whole header");
        };
        let tokens = &header.entries[1];
        let cut = tokens.head.start + tokens.range().len() / 4;
        let Ok(Walk::Incomplete(prefix)) = walk(&bytes[..cut]) else {
            panic!("a prefix");
        };
        assert_eq!(prefix.entries, header.entries[..1]);
        assert_eq!(prefix.open, Some(tokens.head));
        let missing = bytes.len() - cut;
        assert!(prefix.minimum < missing);
        assert!(
            prefix.estimate >= tokens.end - cut && prefix.estimate < missing + 4096,
            "{} for {missing}",
            prefix.estimate
        );
        let Ok(Walk::Incomplete(short)) = walk(&bytes[..3]) else {
            panic!("a prefix");
        };
        assert_eq!(
            (short.preamble, short.open, short.minimum),
            (None, None, 21)
        );
    }

    #[test]
    fn a_count_too_large_for_any_header_is_an_error_not_a_longer_read() {
        let mut bytes = header(10);
        let at = bytes
            .windows(8)
            .position(|window| window == 10_u64.to_le_bytes())
            .unwrap();
        bytes[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(
            matches!(walk(&bytes), Err(Error::Invalid(message)) if message.contains("oversized"))
        );
        assert!(
            matches!(walk(b"GGML"), Err(Error::Invalid(message)) if message.contains("not a GGUF"))
        );
    }
}
