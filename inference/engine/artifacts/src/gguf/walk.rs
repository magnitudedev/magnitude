//! The structure of a GGUF header, read from any prefix of its bytes.
//!
//! [`walk`] visits the header in the bytes in hand: its preamble, each metadata entry with its
//! byte range, and each tensor descriptor, checking the container's structure as it goes. When
//! the bytes end inside the header it says which entry they end in and how many more bytes the
//! header needs: certainly (`minimum`) and probably (`estimate`, extrapolated inside a string
//! array), so a reader growing a prefix (from a file, or by range requests) asks for about as many
//! as remain. A reader grows the prefix in a [`HeaderBytes`], whose walk continues where the walk
//! of the shorter prefix stopped, so each byte is walked once however many reads the header takes.
//! Values are decoded only on request ([`Entry::value`]), without checking their strings again.
//! Whether the engine accepts the tensors themselves (their types, geometry and offsets) is
//! decided on the complete header ([`Header::directory`]).

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
    /// Private, as is `value`: only the walk makes an entry, so its head
    /// states what its value's bytes hold, which they decode by
    /// ([`Self::value`], reading its strings back unchecked).
    head: EntryHead<'a>,
    pub end: usize,
    /// The value's bytes: everything after the value type (a scalar) or the element count (an
    /// array).
    value: &'a [u8],
    order: ByteOrder,
}

impl<'a> Entry<'a> {
    /// The entry's key, stored type, count and where its fixed part lies.
    pub fn head(&self) -> EntryHead<'a> {
        self.head
    }

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
    let mut walked = Progress::default();
    walked.advance(bytes)?;
    Ok(walked.view(bytes))
}

/// The bytes of a GGUF header as a reader grows them, with their walk. The bytes are only ever
/// extended, so each walk continues where the one before stopped: however many reads a header
/// takes, each of its bytes is walked (its strings checked) once, and the reader ends with the
/// whole header's walk in hand.
#[derive(Default)]
pub struct HeaderBytes {
    bytes: Vec<u8>,
    walked: Progress,
}

impl HeaderBytes {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn extend_from_slice(&mut self, more: &[u8]) {
        self.bytes.extend_from_slice(more);
    }

    /// Appends up to `limit` bytes read from `source`, returning how many.
    pub fn read_from(&mut self, source: &mut impl Read, limit: usize) -> std::io::Result<usize> {
        source.by_ref().take(limit as u64).read_to_end(&mut self.bytes)
    }

    /// The walk of the bytes in hand, continuing the walk of the bytes before.
    pub fn walk(&mut self) -> Result<Walk<'_>, Error> {
        self.walked.advance(&self.bytes)?;
        Ok(self.walked.view(&self.bytes))
    }

    /// Reads from `source` onto the bytes until they hold a whole GGUF header (padding aside),
    /// returning its walk. Each read asks for the walk's estimate of the rest, so the bytes may run
    /// past the header by up to the last estimate. A source that ends first is a truncated header.
    pub fn read(&mut self, source: &mut impl Read) -> Result<Header<'_>, Error> {
        loop {
            self.walked.advance(&self.bytes)?;
            let Stop::Incomplete { estimate, .. } = self.walked.stop else {
                break;
            };
            if self.read_from(source, estimate)? == 0 {
                return Err(Error::Truncated {
                    offset: self.bytes.len() as u64,
                });
            }
        }
        match self.walked.view(&self.bytes) {
            Walk::Complete(header) => Ok(header),
            Walk::Incomplete(_) => unreachable!("the reads end at a whole header"),
        }
    }
}

fn beyond() -> Error {
    invalid(format!("GGUF header extends beyond {HEADER_LIMIT} bytes"))
}

fn utf8(bytes: &[u8]) -> Result<&str, Error> {
    std::str::from_utf8(bytes).map_err(|_| invalid("invalid UTF-8 in GGUF header"))
}

/// Text the walk checked ([`utf8`]), read back without checking it again.
fn checked(text: &[u8]) -> &str {
    debug_assert!(std::str::from_utf8(text).is_ok());
    // SAFETY: only text the walk checked comes here: the ranges a `Progress` holds, read back from
    // the bytes it walked (a `HeaderBytes` only ever extends its bytes), and the strings in the
    // values of the entries it reports, which nothing else constructs or changes (an `Entry`'s
    // head and value are private), so each decodes by the head the walk checked it by.
    unsafe { std::str::from_utf8_unchecked(text) }
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

/// Where a walk stopped, with everything it read before as offsets into the bytes it walked, so a
/// walk of those bytes extended continues from here. It moves only past what the walk read whole
/// (an item, or the whole strings of a string array the bytes end in), so a walk that finds the
/// bytes invalid leaves it where it was, and walking them again finds the same error.
struct Progress {
    /// Where the walk continues: the start of the next item, or of the next string of `array`.
    offset: usize,
    preamble: Option<Counted>,
    /// The keys of `entries`.
    keys: HashSet<Box<str>>,
    entries: Vec<Walked>,
    /// The alignment `entries` declare.
    alignment: u64,
    /// The string array the walk stopped in, when it did.
    array: Option<OpenArray>,
    /// Where the tensor descriptors start, once every entry is walked.
    tensors_start: Option<usize>,
    tensors: Vec<WalkedTensor>,
    stop: Stop,
}

impl Default for Progress {
    /// The walk of no bytes.
    fn default() -> Self {
        Self {
            offset: 0,
            preamble: None,
            keys: HashSet::new(),
            entries: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
            array: None,
            tensors_start: None,
            tensors: Vec::new(),
            stop: Stop::Incomplete {
                open: None,
                minimum: PREAMBLE_BYTES,
                estimate: PREAMBLE_BYTES,
            },
        }
    }
}

/// A preamble with the counts it declares.
#[derive(Clone, Copy, Debug)]
struct Counted {
    preamble: Preamble,
    tensors: usize,
    entries: usize,
}

/// Where a walk ends: inside the header ([`Prefix`]) or at its end ([`Header`]).
#[derive(Clone, Copy, Debug)]
enum Stop {
    Incomplete {
        open: Option<Head>,
        minimum: usize,
        estimate: usize,
    },
    Complete {
        preamble: Preamble,
        data_offset: usize,
    },
}

/// Why a walk ends before the header does: the bytes end, or they can't begin a header.
enum Halt {
    Stopped(Stop),
    Invalid(Error),
}

impl From<Error> for Halt {
    fn from(error: Error) -> Self {
        Self::Invalid(error)
    }
}

/// An [`EntryHead`], its key by where it ends.
#[derive(Clone, Copy, Debug)]
struct Head {
    key_end: usize,
    stored: StoredType,
    count: u64,
    start: usize,
    end: usize,
}

/// An [`Entry`].
#[derive(Clone, Copy, Debug)]
struct Walked {
    head: Head,
    value_start: usize,
    end: usize,
}

/// A string array whose first `read` strings are walked.
#[derive(Clone, Copy, Debug)]
struct OpenArray {
    head: Head,
    elements: usize,
    read: usize,
}

/// A [`TensorInfo`].
struct WalkedTensor {
    name: Range<usize>,
    type_id: u32,
    offset: u64,
    dimensions: Range<usize>,
}

impl Head {
    fn view(self, bytes: &[u8]) -> EntryHead<'_> {
        EntryHead {
            // A key's text follows its eight-byte length.
            key: checked(&bytes[self.start + 8..self.key_end]),
            stored: self.stored,
            count: self.count,
            start: self.start,
            end: self.end,
        }
    }
}

impl Walked {
    fn view(self, bytes: &[u8], order: ByteOrder) -> Entry<'_> {
        Entry {
            head: self.head.view(bytes),
            end: self.end,
            value: &bytes[self.value_start..self.end],
            order,
        }
    }
}

impl WalkedTensor {
    fn view<'a>(&self, bytes: &'a [u8], order: ByteOrder) -> TensorInfo<'a> {
        TensorInfo {
            name: checked(&bytes[self.name.clone()]),
            type_id: self.type_id,
            offset: self.offset,
            dimensions: &bytes[self.dimensions.clone()],
            order,
        }
    }
}

/// The walk stopping where `held` bytes end, the header needing at least `minimum` and probably
/// `estimate` more; an error when that would take it past [`HEADER_LIMIT`].
fn stop(held: usize, open: Option<Head>, minimum: usize, estimate: usize) -> Halt {
    let room = HEADER_LIMIT.saturating_sub(held);
    let minimum = minimum.max(1);
    if minimum > room {
        return Halt::Invalid(beyond());
    }
    Halt::Stopped(Stop::Incomplete {
        open,
        minimum,
        estimate: estimate.clamp(minimum, room),
    })
}

/// The preamble `bytes` begin.
fn preamble(bytes: &[u8]) -> Result<Counted, Halt> {
    let short = || {
        let missing = PREAMBLE_BYTES - bytes.len();
        stop(bytes.len(), None, missing, missing)
    };
    let mut cursor = Cursor {
        bytes,
        offset: 0,
        order: ByteOrder::Little,
    };
    let Some(magic) = cursor.take(4)? else {
        return Err(short());
    };
    if magic != b"GGUF" {
        return Err(invalid("not a GGUF container").into());
    }
    let Some(version) = cursor.take(4)? else {
        return Err(short());
    };
    if version == [0, 0, 0, 3] {
        cursor.order = ByteOrder::Big;
    }
    let version = u32_of(cursor.order, version.try_into().expect("four bytes"));
    let (Some(tensor_count), Some(entry_count)) = (cursor.u64()?, cursor.u64()?) else {
        return Err(short());
    };
    if !matches!(version, 2 | 3) {
        return Err(invalid(format!("unsupported GGUF version {version}")).into());
    }
    if tensor_count
        .checked_add(entry_count)
        .is_none_or(|count| count > (HEADER_LIMIT - PREAMBLE_BYTES) as u64 / SMALLEST_ITEM_BYTES)
    {
        return Err(beyond().into());
    }
    Ok(Counted {
        preamble: Preamble {
            version,
            byte_order: cursor.order,
        },
        tensors: tensor_count as usize,
        entries: entry_count as usize,
    })
}

impl Progress {
    /// Walks `bytes`, which extend the bytes walked before, from where that walk stopped.
    fn advance(&mut self, bytes: &[u8]) -> Result<(), Error> {
        if let Stop::Incomplete { .. } = self.stop {
            self.stop = match self.walk(bytes) {
                Ok(stop) | Err(Halt::Stopped(stop)) => stop,
                Err(Halt::Invalid(error)) => return Err(error),
            };
        }
        Ok(())
    }

    /// The walk of `bytes`, which are the bytes walked.
    fn view<'a>(&self, bytes: &'a [u8]) -> Walk<'a> {
        let order = self
            .preamble
            .map_or(ByteOrder::Little, |counted| counted.preamble.byte_order);
        let entries = self
            .entries
            .iter()
            .map(|walked| walked.view(bytes, order))
            .collect();
        match self.stop {
            Stop::Incomplete {
                open,
                minimum,
                estimate,
            } => Walk::Incomplete(Prefix {
                preamble: self.preamble.map(|counted| counted.preamble),
                entries,
                alignment: self.alignment,
                open: open.map(|head| head.view(bytes)),
                minimum,
                estimate,
            }),
            Stop::Complete {
                preamble,
                data_offset,
            } => Walk::Complete(Header {
                preamble,
                entries,
                tensors: self
                    .tensors
                    .iter()
                    .map(|tensor| tensor.view(bytes, order))
                    .collect(),
                alignment: self.alignment,
                data_offset,
            }),
        }
    }

    fn walk(&mut self, bytes: &[u8]) -> Result<Stop, Halt> {
        let held = bytes.len();
        let counted = match self.preamble {
            Some(counted) => counted,
            None => {
                let counted = preamble(bytes)?;
                self.preamble = Some(counted);
                self.offset = PREAMBLE_BYTES;
                counted
            }
        };
        let order = counted.preamble.byte_order;
        let tensors_estimate = counted.tensors * TENSOR_ESTIMATE;
        while self.entries.len() < counted.entries {
            let rest =
                (counted.entries - self.entries.len() - 1) * ENTRY_ESTIMATE + tensors_estimate;
            let mut cursor = Cursor {
                bytes,
                offset: self.offset,
                order,
            };
            let (head, value_start) = match self.array {
                Some(array) => {
                    self.strings(&mut cursor, array, rest)?;
                    (array.head, array.head.end)
                }
                None => self.entry(&mut cursor, rest)?,
            };
            let walked = Walked {
                head,
                value_start,
                end: cursor.offset,
            };
            let entry = walked.view(bytes, order);
            if entry.head.key == ALIGNMENT_KEY {
                self.alignment = super::alignment(&entry.value())?;
            }
            self.keys.insert(entry.head.key.into());
            self.entries.push(walked);
            self.array = None;
            self.offset = cursor.offset;
        }
        let alignment = usize::try_from(self.alignment).unwrap_or(usize::MAX);
        let tensors_start = *self.tensors_start.get_or_insert(self.offset);
        while self.tensors.len() < counted.tensors {
            let index = self.tensors.len();
            let mut cursor = Cursor {
                bytes,
                offset: self.offset,
                order,
            };
            let before = cursor.offset;
            let left = counted.tensors - index;
            let average = if index == 0 {
                TENSOR_ESTIMATE
            } else {
                (before - tensors_start) / index
            };
            let estimate = (left * average)
                .saturating_add(alignment)
                .saturating_sub(held - before);
            let Some(name) = cursor.string()? else {
                return Err(stop(held, None, 8, estimate));
            };
            let name_range = before + 8..cursor.offset;
            let Some(rank) = cursor.u32()? else {
                return Err(stop(held, None, 4, estimate));
            };
            if !(1..=MAX_TENSOR_RANK).contains(&rank) {
                return Err(invalid(format!(
                    "invalid or duplicate tensor directory entry {name:?}"
                ))
                .into());
            }
            let dimensions = rank as usize * 8;
            let fixed = dimensions + 4 + 8;
            let Some(descriptor) = cursor.take(fixed)? else {
                return Err(stop(held, None, fixed, estimate));
            };
            let (type_id, offset) = descriptor[dimensions..].split_at(4);
            let descriptor_start = cursor.offset - fixed;
            self.tensors.push(WalkedTensor {
                name: name_range,
                type_id: u32_of(order, type_id.try_into().expect("four bytes")),
                offset: u64_of(order, offset.try_into().expect("eight bytes")),
                dimensions: descriptor_start..descriptor_start + dimensions,
            });
            self.offset = cursor.offset;
        }
        let data_offset = self
            .offset
            .checked_next_multiple_of(alignment)
            .filter(|end| *end <= HEADER_LIMIT)
            .ok_or_else(beyond)?;
        Ok(Stop::Complete {
            preamble: counted.preamble,
            data_offset,
        })
    }

    /// Walks the metadata entry `cursor` is at, returning its head and where its value starts.
    fn entry(&mut self, cursor: &mut Cursor<'_>, rest: usize) -> Result<(Head, usize), Halt> {
        let held = cursor.bytes.len();
        let start = cursor.offset;
        let Some(key) = cursor.string()? else {
            return Err(stop(held, None, 12, ENTRY_ESTIMATE + rest));
        };
        if self.keys.contains(key) {
            return Err(invalid(format!("duplicate metadata {key:?}")).into());
        }
        let key_end = cursor.offset;
        let Some(value_type) = cursor.u32()? else {
            return Err(stop(held, None, 4, ENTRY_ESTIMATE + rest));
        };
        if value_type == 9 {
            let (Some(element), Some(count)) = (cursor.u32()?, cursor.u64()?) else {
                return Err(stop(held, None, 12, ENTRY_ESTIMATE + rest));
            };
            if element == 9 {
                return Err(invalid("nested or oversized GGUF array").into());
            }
            let element = ValueType::from_id(element)?;
            let head = Head {
                key_end,
                stored: StoredType::Array(element),
                count,
                start,
                end: cursor.offset,
            };
            // Every element takes at least its type's width, so the count is bounded before
            // anything is read.
            let elements = usize::try_from(count)
                .ok()
                .filter(|elements| {
                    elements
                        .checked_mul(element.width())
                        .is_some_and(|length| length <= HEADER_LIMIT - head.end)
                })
                .ok_or_else(|| invalid("nested or oversized GGUF array"))?;
            if element == ValueType::String {
                self.strings(
                    cursor,
                    OpenArray {
                        head,
                        elements,
                        read: 0,
                    },
                    rest,
                )?;
            } else {
                let length = elements * element.width();
                let Some(values) = cursor.take(length)? else {
                    let missing = head.end + length - held;
                    return Err(stop(held, Some(head), missing, missing + rest));
                };
                if element == ValueType::Bool {
                    values.iter().try_for_each(|byte| valid_bool(*byte))?;
                }
            }
            return Ok((head, head.end));
        }
        let kind = ValueType::from_id(value_type)?;
        let value_start = cursor.offset;
        if kind == ValueType::String {
            let Some(length) = cursor.u64()? else {
                return Err(stop(held, None, 8, ENTRY_ESTIMATE + rest));
            };
            let head = Head {
                key_end,
                stored: StoredType::Scalar(kind),
                count: 1,
                start,
                end: cursor.offset,
            };
            let length = usize::try_from(length)
                .ok()
                .filter(|length| *length <= HEADER_LIMIT - head.end)
                .ok_or_else(beyond)?;
            let Some(text) = cursor.take(length)? else {
                let missing = head.end + length - held;
                return Err(stop(held, Some(head), missing, missing + rest));
            };
            utf8(text)?;
            return Ok((head, value_start));
        }
        let width = kind.width();
        let Some(value) = cursor.take(width)? else {
            return Err(stop(held, None, width, width + rest));
        };
        if kind == ValueType::Bool {
            valid_bool(value[0])?;
        }
        let head = Head {
            key_end,
            stored: StoredType::Scalar(kind),
            count: 1,
            start,
            end: value_start,
        };
        Ok((head, value_start))
    }

    /// Walks the strings of `array` from its `read`th, which `cursor` is at. Where the bytes end
    /// first, the walk stops there, and continues from there.
    fn strings(
        &mut self,
        cursor: &mut Cursor<'_>,
        array: OpenArray,
        rest: usize,
    ) -> Result<(), Halt> {
        let held = cursor.bytes.len();
        for read in array.read..array.elements {
            let before = cursor.offset;
            if cursor.string()?.is_none() {
                let left = array.elements - read;
                let average = if read < MEASURED_STRINGS {
                    STRING_ESTIMATE
                } else {
                    ((before - array.head.end) / read)
                        .clamp(STRING_ESTIMATE_BOUNDS.0, STRING_ESTIMATE_BOUNDS.1)
                };
                let in_string = held - before;
                let halt = stop(
                    held,
                    Some(array.head),
                    (left * 8).saturating_sub(in_string),
                    (left * average).saturating_sub(in_string) + rest,
                );
                if let Halt::Stopped(_) = halt {
                    self.array = Some(OpenArray { read, ..array });
                    self.offset = before;
                }
                return Err(halt);
            }
        }
        Ok(())
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
        checked(text)
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
    fn a_growing_prefix_walks_as_the_prefix_walked_whole() {
        let bytes = header(5_000);
        let mut grown = HeaderBytes::new();
        let mut held = 0;
        for cut in (0..=bytes.len()).step_by(997).chain([bytes.len()]) {
            grown.extend_from_slice(&bytes[held..cut]);
            held = cut;
            assert_eq!(grown.walk().unwrap(), walk(&bytes[..cut]).unwrap(), "at {cut}");
        }
        let mut read = HeaderBytes::new();
        let whole = read.read(&mut bytes.as_slice()).unwrap();
        let Ok(Walk::Complete(expected)) = walk(&bytes) else {
            panic!("a whole header");
        };
        assert_eq!(whole, expected);
    }

    #[test]
    fn a_walk_that_fails_fails_the_same_way_when_walked_again() {
        let mut bytes = header(3_000);
        let Ok(Walk::Complete(whole)) = walk(&bytes) else {
            panic!("a whole header");
        };
        let tokens = whole.entries[1];
        let broken = tokens.head.end + tokens.range().len() / 2;
        let at = (broken..)
            .find(|at| bytes[*at..].starts_with(b"token-"))
            .unwrap();
        bytes[at] = 0xff;
        let mut grown = HeaderBytes::new();
        grown.extend_from_slice(&bytes[..broken]);
        assert!(matches!(grown.walk(), Ok(Walk::Incomplete(_))));
        grown.extend_from_slice(&bytes[broken..]);
        for _ in 0..2 {
            assert!(
                matches!(grown.walk(), Err(Error::Invalid(message)) if message.contains("UTF-8"))
            );
        }
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
