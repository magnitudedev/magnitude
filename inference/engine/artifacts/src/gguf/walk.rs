//! The structure of a GGUF header, read from any prefix of its bytes.
//!
//! [`walk`] visits the header in the bytes in hand: its preamble, each metadata entry with its
//! byte range, and each tensor descriptor, checking the container's structure as it goes. When
//! the bytes end inside the header it says which entry they end in and how many more bytes the
//! header needs: certainly (`minimum`) and probably (`estimate`, extrapolated inside a string
//! array), so a reader growing a prefix (from a file, or by range requests) asks for about as many
//! as remain. A reader grows the prefix in a [`HeaderBytes`], whose walk continues where the walk
//! of the shorter prefix stopped, so each byte is walked once however many reads the header takes.
//!
//! A metadata array's payload is hopped by its length alone: a fixed-size element array by its
//! count, a string array by its strings' length prefixes. Its elements are decoded and checked
//! only when they are read ([`super::Array::elements`]). A header skeleton ([`Skeleton`]) lacks
//! some array payloads; the walk passes each by its known length. Whether the engine accepts the
//! tensors themselves (their types, geometry and offsets) is decided on the complete header
//! ([`SharedHeader::directory`]).

use super::{
    invalid, Array, ByteOrder, Element, Elements, Error, Kind, Metadata, Scalar, StoredType,
    Strings, Value, ValueType,
};
use std::{collections::HashSet, io::Read, ops::Range, path::PathBuf, sync::Arc};

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
/// Positions are in the whole header.
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
    /// ([`Self::value`], reading a scalar's text back unchecked).
    head: EntryHead<'a>,
    /// Where the entry ends in the whole header.
    pub end: usize,
    value: Payload<'a>,
    order: ByteOrder,
}

/// The bytes of an entry's value: everything after the value type (a scalar) or the element count
/// (an array), and where they start in the bytes walked; or, for an array read from a skeleton
/// without its payload, where the payload is apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Payload<'a> {
    Held { bytes: &'a [u8], at: usize },
    Apart(&'a Absence),
}

impl<'a> Entry<'a> {
    /// The entry's key, stored type, count and where its fixed part lies.
    pub fn head(&self) -> EntryHead<'a> {
        self.head
    }

    /// The entry's bytes in the whole header.
    pub fn range(&self) -> Range<usize> {
        self.head.start..self.end
    }

    /// The entry's key and value ([`Self::value`]).
    pub(crate) fn metadata(&self, header: &Arc<HeaderBacking>) -> Metadata {
        Metadata {
            name: self.head.key.to_owned(),
            value: self.value(header),
        }
    }

    /// The entry's value. `header` holds the bytes this entry was walked from: an array's payload
    /// is a span of them (or, read from a skeleton without it, the payload apart it names), decoded
    /// when its elements are read.
    ///
    /// # Panics
    /// When `header` is not the bytes the entry was walked from.
    pub(crate) fn value(&self, header: &Arc<HeaderBacking>) -> Value {
        let StoredType::Array(stored) = self.head.stored else {
            return Value::Scalar(self.scalar().expect("a scalar entry holds its value"));
        };
        let count = usize::try_from(self.head.count).expect("a walked count fits usize");
        Value::Array(match self.value {
            Payload::Apart(absence) => {
                Array::apart(self.head.key, stored, self.order, count, absence.clone())
            }
            Payload::Held { bytes, at } => {
                let range = at..at + bytes.len();
                assert!(
                    header
                        .get(range.clone())
                        .is_some_and(|held| std::ptr::eq(held, bytes)),
                    "an entry's value is read from the bytes it was walked from"
                );
                Array::held(
                    self.head.key,
                    stored,
                    self.order,
                    count,
                    Arc::clone(header),
                    range,
                )
            }
        })
    }

    /// The entry's value when it is a scalar.
    fn scalar(&self) -> Option<Scalar> {
        let (StoredType::Scalar(kind), Payload::Held { bytes, .. }) =
            (self.head.stored, self.value)
        else {
            return None;
        };
        Some(
            Decoder {
                bytes,
                order: self.order,
            }
            .scalar(kind)
            .expect("the walk checked a scalar's bytes"),
        )
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

/// A metadata array's payload absent from a [`Skeleton`]: its offset in the whole header, its
/// length, its sha256, and the file that holds it (read only when its elements are).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Absence {
    pub offset: usize,
    pub length: usize,
    pub digest: [u8; 32],
    pub file: PathBuf,
}

/// A GGUF header without some of its metadata arrays' payloads: the header's bytes with each
/// absent payload cut out, and where each was, in order ([`super::read_skeleton`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skeleton {
    pub bytes: Vec<u8>,
    pub absent: Vec<Absence>,
}

/// The structure of the GGUF header `bytes` begin, or why they can't begin one.
pub fn walk(bytes: &[u8]) -> Result<Walk<'_>, Error> {
    let mut walked = Phase::default();
    walked.advance(bytes, &[])?;
    Ok(walked.view(bytes, &[]))
}

/// Backing storage for a walked GGUF header's bytes: either owned in memory or a read-only
/// `MAP_PRIVATE` file mapping.
pub(crate) enum HeaderBacking {
    Owned(Arc<Vec<u8>>),
    #[cfg(unix)]
    Mapped {
        ptr: *const u8,
        map_len: usize,
        len: usize,
    },
}

// SAFETY: `HeaderBacking::Mapped` owns a read-only `MAP_PRIVATE` mapping of `map_len` bytes that
// is never mutated through this pointer and is unmapped only in `Drop`.
unsafe impl Send for HeaderBacking {}
// SAFETY: Only immutable `&[u8]` access is ever exposed from `HeaderBacking`.
unsafe impl Sync for HeaderBacking {}

impl Drop for HeaderBacking {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Self::Mapped { ptr, map_len, .. } = *self {
            if map_len > 0 {
                // SAFETY: `ptr` and `map_len` came from a successful `libc::mmap` call in
                // `SharedHeader::read_mapped`, and `Drop` runs at most once.
                unsafe {
                    let _ = libc::munmap(ptr.cast_mut().cast::<libc::c_void>(), map_len);
                }
            }
        }
    }
}

impl std::ops::Deref for HeaderBacking {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes.as_slice(),
            #[cfg(unix)]
            Self::Mapped { ptr, len, .. } => {
                // SAFETY: `ptr` is non-null and valid for immutable reads of `len <= map_len`
                // bytes for the lifetime of `&self` (until `Drop` unmaps it).
                unsafe { std::slice::from_raw_parts(*ptr, *len) }
            }
        }
    }
}

impl AsRef<[u8]> for HeaderBacking {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

/// A whole GGUF header's bytes, shared with the metadata arrays of its directory
/// ([`Self::directory`]), and their walk.
pub struct SharedHeader {
    bytes: Arc<HeaderBacking>,
    /// The payloads `bytes` lack.
    absent: Vec<Absence>,
    walked: CompletedWalk,
}

impl SharedHeader {
    /// The whole header `bytes` hold, with the payloads `absent` lacks.
    pub(super) fn walk(bytes: Arc<Vec<u8>>, absent: Vec<Absence>) -> Result<Self, Error> {
        let mut walked = Phase::default();
        walked.advance(&bytes, &absent)?;
        let Some(walked) = walked.into_complete() else {
            return Err(Error::Truncated {
                offset: bytes.len() as u64,
            });
        };
        Ok(Self {
            bytes: Arc::new(HeaderBacking::Owned(bytes)),
            absent,
            walked,
        })
    }

    /// Walks a GGUF header directly from a read-only `MAP_PRIVATE` mapping of up to
    /// [`HEADER_LIMIT`] bytes of `file`, avoiding heap zero-fill and `File::read` copies.
    /// Returns `Ok(None)` when the file is empty or `mmap` fails so the caller can fall back to
    /// [`HeaderBytes::read`].
    #[cfg(unix)]
    pub(super) fn read_mapped(
        file: &std::fs::File,
        file_size: u64,
    ) -> Result<Option<Self>, Error> {
        use std::os::fd::AsRawFd;

        if file_size == 0 {
            return Ok(None);
        }
        let map_len = usize::try_from(file_size.min(HEADER_LIMIT as u64)).unwrap_or(HEADER_LIMIT);
        // SAFETY: `file` is an open file descriptor, `map_len > 0`, and we request a read-only
        // private mapping starting at offset 0.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Ok(None);
        }
        // Wrap immediately so `munmap` runs on every error or early return.
        let backing = HeaderBacking::Mapped {
            ptr: ptr.cast::<u8>().cast_const(),
            map_len,
            len: map_len,
        };
        let mut walked = Phase::default();
        walked.advance(&backing, &[])?;
        let Some(walked) = walked.into_complete() else {
            return Err(if file_size <= HEADER_LIMIT as u64 {
                Error::Truncated {
                    offset: map_len as u64,
                }
            } else {
                beyond()
            });
        };
        Ok(Some(Self {
            bytes: Arc::new(backing),
            absent: Vec::new(),
            walked,
        }))
    }

    pub fn bytes(&self) -> &Arc<Vec<u8>> {
        match self.bytes.as_ref() {
            HeaderBacking::Owned(bytes) => bytes,
            #[cfg(unix)]
            HeaderBacking::Mapped { .. } => {
                unreachable!("mapped headers are internal and never expose Arc<Vec<u8>>")
            }
        }
    }

    pub(super) fn backing(&self) -> &Arc<HeaderBacking> {
        &self.bytes
    }

    pub fn header(&self) -> Header<'_> {
        self.walked.view(&self.bytes, &self.absent)
    }
}

/// The bytes of a GGUF header as a reader grows them, with their walk. The bytes are only ever
/// extended, so each walk continues where the one before stopped: however many reads a header
/// takes, each of its bytes is walked once, and the reader ends with the whole header's walk in
/// hand.
#[derive(Default)]
pub struct HeaderBytes {
    bytes: Vec<u8>,
    walked: Phase,
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
        const MAX_RESERVE: usize = 8 * 1024 * 1024;
        const CHUNK: usize = 2 * 1024 * 1024;

        self.bytes.reserve(limit.min(MAX_RESERVE));
        let start = self.bytes.len();
        let mut total = 0;
        while total < limit {
            let step = (limit - total).min(CHUNK);
            let pos = start + total;
            self.bytes.resize(pos + step, 0);
            let mut filled = 0;
            while filled < step {
                match source.read(&mut self.bytes[pos + filled..pos + step]) {
                    Ok(0) => {
                        self.bytes.truncate(pos + filled);
                        return Ok(total + filled);
                    }
                    Ok(n) => filled += n,
                    Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(err) => {
                        self.bytes.truncate(pos + filled);
                        return Err(err);
                    }
                }
            }
            total += filled;
        }
        Ok(total)
    }

    /// The walk of the bytes in hand, continuing the walk of the bytes before.
    pub fn walk(&mut self) -> Result<Walk<'_>, Error> {
        self.walked.advance(&self.bytes, &[])?;
        Ok(self.walked.view(&self.bytes, &[]))
    }

    /// Reads from `source` onto the bytes until they hold a whole GGUF header (padding aside),
    /// returning its walk. Each read asks for the walk's estimate of the rest, so the bytes may run
    /// past the header by up to the last estimate. A source that ends first is a truncated header.
    pub fn read(&mut self, source: &mut impl Read) -> Result<Header<'_>, Error> {
        let completed = loop {
            self.walked.advance(&self.bytes, &[])?;
            let (phase, estimate) = match std::mem::take(&mut self.walked) {
                Phase::Complete(completed) => break completed,
                Phase::Preamble { missing } => (Phase::Preamble { missing }, missing),
                Phase::Entries(state) => {
                    let estimate = state.need.estimate;
                    (Phase::Entries(state), estimate)
                }
                Phase::Tensors(state) => {
                    let estimate = state.estimate;
                    (Phase::Tensors(state), estimate)
                }
            };
            self.walked = phase;
            let to_read = estimate.min(HEADER_LIMIT.saturating_sub(self.bytes.len()));
            if self.read_from(source, to_read)? == 0 {
                return Err(Error::Truncated {
                    offset: self.bytes.len() as u64,
                });
            }
        };
        let header = completed.view(&self.bytes, &[]);
        self.walked = Phase::Complete(completed);
        Ok(header)
    }

    /// The first `length` bytes in hand, which hold the whole header walked (padding aside),
    /// shared with the metadata arrays of its directory, its walk kept.
    pub fn into_shared(mut self, length: usize) -> Result<SharedHeader, Error> {
        let Phase::Complete(walked) = self.walked else {
            return Err(Error::Truncated {
                offset: self.bytes.len() as u64,
            });
        };
        if length > self.bytes.len() {
            return Err(Error::Truncated {
                offset: self.bytes.len() as u64,
            });
        }
        if length < walked.offset {
            return Err(invalid(format!(
                "{length} bytes cut the GGUF header walked through byte {}",
                walked.offset
            )));
        }
        self.bytes.truncate(length);
        Ok(SharedHeader {
            bytes: Arc::new(HeaderBacking::Owned(Arc::new(self.bytes))),
            absent: Vec::new(),
            walked,
        })
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
    // SAFETY: only text the walk checked comes here: the keys and tensor names a `Phase` or
    // `CompletedWalk` holds the ranges of, read back from the bytes it walked (a `HeaderBytes`
    // only ever extends its bytes, and a `SharedHeader` never changes them), and the text of the
    // scalar values of the entries it reports, which nothing else constructs or changes (an
    // `Entry`'s head and value are private), so each decodes by the head the walk checked it by.
    // Array text is checked when it is decoded ([`decode`]).
    unsafe { std::str::from_utf8_unchecked(text) }
}

fn valid_bool(byte: u8) -> Result<bool, Error> {
    match byte {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid("invalid GGUF boolean")),
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

/// Where a walk stopped, with everything it read before as positions in the bytes it walked, so a
/// walk of those bytes extended continues from here. It moves only past what the walk read whole
/// (an item, or the whole strings of a string array the bytes end in), so a walk that finds the
/// bytes invalid leaves it where it was, and walking them again finds the same error.
enum Phase {
    /// Waiting for the 24-byte preamble.
    Preamble { missing: usize },
    /// Walking metadata entries (possibly paused inside a string array).
    Entries(EntriesPhase),
    /// Walking tensor descriptors.
    Tensors(TensorsPhase),
    /// The whole header has been walked.
    Complete(CompletedWalk),
}

impl Default for Phase {
    /// The walk of no bytes.
    fn default() -> Self {
        Self::Preamble {
            missing: PREAMBLE_BYTES,
        }
    }
}

/// Position in the whole header being walked.
#[derive(Clone, Copy, Debug)]
struct Position {
    /// Where the walk continues in the whole header: the start of the next item, or of the next
    /// string of an open string array.
    offset: usize,
    /// The bytes of the absent payloads before `offset`, which the bytes walked lack.
    shift: usize,
    /// How many absent payloads lie before `offset`.
    passed: usize,
}

impl Position {
    /// A cursor at where the walk continues.
    fn cursor<'a>(self, bytes: &'a [u8], absent: &'a [Absence], order: ByteOrder) -> Cursor<'a> {
        Cursor {
            bytes,
            offset: self.offset,
            shift: self.shift,
            absent: &absent[self.passed..],
            order,
        }
    }

    /// Moves the walk on to where `cursor` is.
    fn pass(&mut self, cursor: &Cursor<'_>, absent: &[Absence]) {
        self.offset = cursor.offset;
        self.shift = cursor.shift;
        self.passed = absent.len() - cursor.absent.len();
    }
}

/// Metadata-entry walking state; dropped upon transitioning to [`TensorsPhase`].
struct EntriesPhase {
    pos: Position,
    counted: Counted,
    /// The keys of `entries`.
    keys: HashSet<Box<str>>,
    entries: Vec<Walked>,
    /// The alignment `entries` declare.
    alignment: u64,
    /// The string array the walk stopped in, when it did.
    array: Option<OpenArray>,
    need: Need,
}

/// Tensor-descriptor walking state; dropped upon transitioning to [`CompletedWalk`].
struct TensorsPhase {
    pos: Position,
    preamble: Preamble,
    tensor_count: usize,
    entries: Vec<Walked>,
    alignment: u64,
    /// Where the tensor descriptors start in the whole header.
    tensors_start: usize,
    tensors: Vec<WalkedTensor>,
    minimum: usize,
    estimate: usize,
}

/// A completed GGUF header walk, retaining only the fields needed to construct a [`Header`].
struct CompletedWalk {
    preamble: Preamble,
    entries: Vec<Walked>,
    tensors: Vec<WalkedTensor>,
    alignment: u64,
    data_offset: usize,
    /// Where the walked header ends before alignment padding.
    offset: usize,
}

impl CompletedWalk {
    fn view<'a>(&self, bytes: &'a [u8], absent: &'a [Absence]) -> Header<'a> {
        let order = self.preamble.byte_order;
        Header {
            preamble: self.preamble,
            entries: self
                .entries
                .iter()
                .map(|walked| walked.view(bytes, absent, order))
                .collect(),
            tensors: self
                .tensors
                .iter()
                .map(|tensor| tensor.view(bytes, order))
                .collect(),
            alignment: self.alignment,
            data_offset: self.data_offset,
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

/// Where an incomplete walk stopped: the entry it ended in (when its fixed part is in hand) and
/// how many more bytes the header needs.
#[derive(Clone, Copy, Debug)]
struct Need {
    open: Option<Head>,
    minimum: usize,
    estimate: usize,
}

/// Why a walk ends before the header does: the bytes end, or they can't begin a header.
enum Halt {
    Stopped(Need),
    Invalid(Error),
}

impl From<Error> for Halt {
    fn from(error: Error) -> Self {
        Self::Invalid(error)
    }
}

/// An [`EntryHead`], its key by where it ends. Positions are in the whole header; `shift` is the
/// bytes of absent payloads before the entry, which the bytes walked lack.
#[derive(Clone, Copy, Debug)]
struct Head {
    key_end: usize,
    stored: StoredType,
    count: u64,
    start: usize,
    end: usize,
    shift: usize,
}

/// Where an entry's value lies: from `start` (in the whole header) to the entry's end, or apart,
/// the `index`th absent payload.
#[derive(Clone, Copy, Debug)]
enum WalkedValue {
    Held { start: usize },
    Apart { index: usize },
}

/// An [`Entry`].
#[derive(Clone, Copy, Debug)]
struct Walked {
    head: Head,
    value: WalkedValue,
    end: usize,
}

/// A string array whose first `read` strings are walked.
#[derive(Clone, Copy, Debug)]
struct OpenArray {
    head: Head,
    elements: usize,
    read: usize,
}

/// A [`TensorInfo`], by where its parts lie in the bytes walked.
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
            key: checked(&bytes[self.start - self.shift + 8..self.key_end - self.shift]),
            stored: self.stored,
            count: self.count,
            start: self.start,
            end: self.end,
        }
    }
}

impl Walked {
    fn view<'a>(self, bytes: &'a [u8], absent: &'a [Absence], order: ByteOrder) -> Entry<'a> {
        let shift = self.head.shift;
        Entry {
            head: self.head.view(bytes),
            end: self.end,
            value: match self.value {
                WalkedValue::Held { start } => Payload::Held {
                    bytes: &bytes[start - shift..self.end - shift],
                    at: start - shift,
                },
                WalkedValue::Apart { index } => Payload::Apart(&absent[index]),
            },
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
    Halt::Stopped(Need {
        open,
        minimum,
        estimate: estimate.clamp(minimum, room),
    })
}

/// The preamble `bytes` begin.
fn preamble(bytes: &[u8], absent: &[Absence]) -> Result<Counted, Halt> {
    let short = || {
        let missing = PREAMBLE_BYTES - bytes.len();
        stop(bytes.len(), None, missing, missing)
    };
    let mut cursor = Cursor {
        bytes,
        offset: 0,
        shift: 0,
        absent,
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

impl Phase {
    /// Walks `bytes`, which extend the bytes walked before, from where that walk stopped. The
    /// bytes lack the payloads `absent` names (none, for a walk that may continue).
    fn advance(&mut self, bytes: &[u8], absent: &[Absence]) -> Result<(), Error> {
        let mut phase = std::mem::take(self);
        let result = loop {
            phase = match phase {
                Self::Preamble { missing } => match preamble(bytes, absent) {
                    Ok(counted) => Self::Entries(EntriesPhase::new(counted)),
                    Err(Halt::Stopped(need)) => {
                        phase = Self::Preamble {
                            missing: need.minimum,
                        };
                        break Ok(());
                    }
                    Err(Halt::Invalid(error)) => {
                        phase = Self::Preamble { missing };
                        break Err(error);
                    }
                },
                Self::Entries(mut state) => match state.advance(bytes, absent) {
                    Ok(()) => Self::Tensors(state.into_tensors()),
                    Err(Halt::Stopped(need)) => {
                        state.need = need;
                        phase = Self::Entries(state);
                        break Ok(());
                    }
                    Err(Halt::Invalid(error)) => {
                        phase = Self::Entries(state);
                        break Err(error);
                    }
                },
                Self::Tensors(mut state) => match state.advance(bytes, absent) {
                    Ok(data_offset) => {
                        phase = Self::Complete(state.into_complete(data_offset));
                        break Ok(());
                    }
                    Err(Halt::Stopped(need)) => {
                        state.minimum = need.minimum;
                        state.estimate = need.estimate;
                        phase = Self::Tensors(state);
                        break Ok(());
                    }
                    Err(Halt::Invalid(error)) => {
                        phase = Self::Tensors(state);
                        break Err(error);
                    }
                },
                Self::Complete(completed) => {
                    phase = Self::Complete(completed);
                    break Ok(());
                }
            };
        };
        *self = phase;
        result
    }

    fn into_complete(self) -> Option<CompletedWalk> {
        match self {
            Self::Complete(completed) => Some(completed),
            Self::Preamble { .. } | Self::Entries(_) | Self::Tensors(_) => None,
        }
    }

    /// The walk of `bytes`, which are the bytes walked, lacking the payloads `absent` names.
    fn view<'a>(&self, bytes: &'a [u8], absent: &'a [Absence]) -> Walk<'a> {
        match self {
            Self::Preamble { missing } => Walk::Incomplete(Prefix {
                preamble: None,
                entries: Vec::new(),
                alignment: DEFAULT_ALIGNMENT,
                open: None,
                minimum: *missing,
                estimate: *missing,
            }),
            Self::Entries(state) => {
                let order = state.counted.preamble.byte_order;
                Walk::Incomplete(Prefix {
                    preamble: Some(state.counted.preamble),
                    entries: state
                        .entries
                        .iter()
                        .map(|walked| walked.view(bytes, absent, order))
                        .collect(),
                    alignment: state.alignment,
                    open: state.need.open.map(|head| head.view(bytes)),
                    minimum: state.need.minimum,
                    estimate: state.need.estimate,
                })
            }
            Self::Tensors(state) => {
                let order = state.preamble.byte_order;
                Walk::Incomplete(Prefix {
                    preamble: Some(state.preamble),
                    entries: state
                        .entries
                        .iter()
                        .map(|walked| walked.view(bytes, absent, order))
                        .collect(),
                    alignment: state.alignment,
                    open: None,
                    minimum: state.minimum,
                    estimate: state.estimate,
                })
            }
            Self::Complete(completed) => Walk::Complete(completed.view(bytes, absent)),
        }
    }
}

impl EntriesPhase {
    fn new(counted: Counted) -> Self {
        Self {
            pos: Position {
                offset: PREAMBLE_BYTES,
                shift: 0,
                passed: 0,
            },
            counted,
            keys: HashSet::new(),
            entries: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
            array: None,
            need: Need {
                open: None,
                minimum: 1,
                estimate: 1,
            },
        }
    }

    fn advance(&mut self, bytes: &[u8], absent: &[Absence]) -> Result<(), Halt> {
        let order = self.counted.preamble.byte_order;
        let tensors_estimate = self.counted.tensors * TENSOR_ESTIMATE;
        while self.entries.len() < self.counted.entries {
            let rest = (self.counted.entries - self.entries.len() - 1) * ENTRY_ESTIMATE
                + tensors_estimate;
            let mut cursor = self.pos.cursor(bytes, absent, order);
            let (head, value) = match self.array {
                Some(array) => {
                    self.strings(&mut cursor, array, rest)?;
                    (
                        array.head,
                        WalkedValue::Held {
                            start: array.head.end,
                        },
                    )
                }
                None => self.entry(&mut cursor, rest)?,
            };
            let walked = Walked {
                head,
                value,
                end: cursor.offset,
            };
            let entry = walked.view(bytes, absent, order);
            if entry.head.key == ALIGNMENT_KEY {
                self.alignment = super::alignment(
                    entry
                        .scalar()
                        .map(Value::Scalar)
                        .as_ref()
                        .and_then(Value::unsigned),
                )?;
            }
            self.keys.insert(entry.head.key.into());
            self.entries.push(walked);
            self.array = None;
            self.pos.pass(&cursor, absent);
        }
        Ok(())
    }

    fn into_tensors(self) -> TensorsPhase {
        TensorsPhase {
            tensors_start: self.pos.offset,
            pos: self.pos,
            preamble: self.counted.preamble,
            tensor_count: self.counted.tensors,
            entries: self.entries,
            alignment: self.alignment,
            tensors: Vec::new(),
            minimum: 1,
            estimate: 1,
        }
    }

    /// Walks the metadata entry `cursor` is at, returning its head and where its value lies.
    fn entry(&mut self, cursor: &mut Cursor<'_>, rest: usize) -> Result<(Head, WalkedValue), Halt> {
        let held = cursor.held();
        let start = cursor.offset;
        let shift = cursor.shift;
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
                shift,
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
            if let Some(length) = cursor.absent()? {
                // A string takes at least its length; any other element exactly its width.
                let fits = if element == ValueType::String {
                    length >= elements * element.width()
                } else {
                    length == elements * element.width()
                };
                if !fits {
                    return Err(invalid(format!(
                        "the absent payload of GGUF metadata {key:?} does not fit its count"
                    ))
                    .into());
                }
                return Ok((
                    head,
                    WalkedValue::Apart {
                        index: self.pos.passed,
                    },
                ));
            }
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
                if cursor.take(length)?.is_none() {
                    let missing = head.end + length - held;
                    return Err(stop(held, Some(head), missing, missing + rest));
                }
            }
            return Ok((head, WalkedValue::Held { start: head.end }));
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
                shift,
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
            return Ok((head, WalkedValue::Held { start: value_start }));
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
            shift,
        };
        Ok((head, WalkedValue::Held { start: value_start }))
    }

    /// Hops the strings of `array` from its `read`th, which `cursor` is at, by their lengths.
    /// Where the bytes end first, the walk stops there, and continues from there.
    fn strings(
        &mut self,
        cursor: &mut Cursor<'_>,
        array: OpenArray,
        rest: usize,
    ) -> Result<(), Halt> {
        let held = cursor.held();
        let stop_at = |this: &mut Self, read: usize, before: usize| {
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
                this.array = Some(OpenArray { read, ..array });
                this.pos.offset = before;
            }
            halt
        };
        if cursor.absent.is_empty() && cursor.order == ByteOrder::Little {
            let bytes = cursor.bytes;
            let shift = cursor.shift;
            let max_pos = bytes.len().min(HEADER_LIMIT.saturating_sub(shift));
            let mut pos = cursor.offset - shift;
            let mut read = array.read;
            while read < array.elements {
                if max_pos - pos < 8 {
                    if HEADER_LIMIT - (pos + shift) < 8 {
                        return Err(beyond().into());
                    }
                    break;
                }
                let len = u64::from_le_bytes(
                    bytes[pos..pos + 8]
                        .try_into()
                        .expect("eight bytes"),
                );
                let header_end = pos + 8;
                let rem = (max_pos - header_end) as u64;
                if len > rem {
                    let end = (header_end as u64)
                        .checked_add(shift as u64)
                        .and_then(|base| base.checked_add(len));
                    if end.is_none_or(|end| end > HEADER_LIMIT as u64) {
                        return Err(beyond().into());
                    }
                    break;
                }
                pos = header_end + len as usize;
                read += 1;
            }
            cursor.offset = pos + shift;
            if read < array.elements {
                return Err(stop_at(self, read, cursor.offset));
            }
            return Ok(());
        }
        for read in array.read..array.elements {
            let before = cursor.offset;
            if cursor.text()?.is_none() {
                return Err(stop_at(self, read, before));
            }
        }
        Ok(())
    }
}

impl TensorsPhase {
    fn advance(&mut self, bytes: &[u8], absent: &[Absence]) -> Result<usize, Halt> {
        let order = self.preamble.byte_order;
        let alignment = usize::try_from(self.alignment).unwrap_or(usize::MAX);
        while self.tensors.len() < self.tensor_count {
            let index = self.tensors.len();
            let mut cursor = self.pos.cursor(bytes, absent, order);
            let held = cursor.held();
            let before = cursor.offset;
            let left = self.tensor_count - index;
            let average = if index == 0 {
                TENSOR_ESTIMATE
            } else {
                (before - self.tensors_start) / index
            };
            let estimate = (left * average)
                .saturating_add(alignment)
                .saturating_sub(held - before);
            let Some(name) = cursor.string()? else {
                return Err(stop(held, None, 8, estimate));
            };
            let shift = cursor.shift;
            let name_range = before + 8 - shift..cursor.offset - shift;
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
            let descriptor_start = cursor.offset - fixed - shift;
            self.tensors.push(WalkedTensor {
                name: name_range,
                type_id: u32_of(order, type_id.try_into().expect("four bytes")),
                offset: u64_of(order, offset.try_into().expect("eight bytes")),
                dimensions: descriptor_start..descriptor_start + dimensions,
            });
            self.pos.pass(&cursor, absent);
        }
        if self.pos.passed < absent.len() {
            return Err(invalid("an absent GGUF payload is not a metadata array's").into());
        }
        let data_offset = self
            .pos
            .offset
            .checked_next_multiple_of(alignment)
            .filter(|end| *end <= HEADER_LIMIT)
            .ok_or_else(beyond)?;
        Ok(data_offset)
    }

    fn into_complete(self, data_offset: usize) -> CompletedWalk {
        CompletedWalk {
            preamble: self.preamble,
            entries: self.entries,
            tensors: self.tensors,
            alignment: self.alignment,
            data_offset,
            offset: self.pos.offset,
        }
    }
}

/// Reads the bytes in hand, at a position in the whole header. `None` is bytes that end first (the
/// cursor then stays where it was); an error is a read that would end past [`HEADER_LIMIT`], or
/// into a payload the bytes lack.
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    /// The bytes of the absent payloads before `offset`, which `bytes` lack.
    shift: usize,
    /// The absent payloads from `offset` on.
    absent: &'a [Absence],
    order: ByteOrder,
}

impl<'a> Cursor<'a> {
    /// How far into the whole header the bytes in hand reach.
    fn held(&self) -> usize {
        self.bytes.len() + self.shift
    }

    fn take(&mut self, length: usize) -> Result<Option<&'a [u8]>, Error> {
        let end = self
            .offset
            .checked_add(length)
            .filter(|end| *end <= HEADER_LIMIT)
            .ok_or_else(beyond)?;
        if self
            .absent
            .first()
            .is_some_and(|absence| end > absence.offset)
        {
            return Err(invalid("an absent GGUF payload is not a metadata array's"));
        }
        let Some(taken) = self.bytes.get(self.offset - self.shift..end - self.shift) else {
            return Ok(None);
        };
        self.offset = end;
        Ok(Some(taken))
    }

    /// Passes the absent payload the cursor is at, returning its length, when one is.
    fn absent(&mut self) -> Result<Option<usize>, Error> {
        let Some((absence, rest)) = self.absent.split_first() else {
            return Ok(None);
        };
        if absence.offset != self.offset {
            return Ok(None);
        }
        self.offset = self
            .offset
            .checked_add(absence.length)
            .filter(|end| *end <= HEADER_LIMIT)
            .ok_or_else(beyond)?;
        self.shift += absence.length;
        self.absent = rest;
        Ok(Some(absence.length))
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

    /// A string's bytes, unchecked.
    fn text(&mut self) -> Result<Option<&'a [u8]>, Error> {
        let start = self.offset;
        let Some(length) = self.u64()? else {
            return Ok(None);
        };
        let length = usize::try_from(length).map_err(|_| beyond())?;
        match self.take(length) {
            Ok(Some(text)) => Ok(Some(text)),
            Ok(None) => {
                self.offset = start;
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// A UTF-8 string.
    fn string(&mut self) -> Result<Option<&'a str>, Error> {
        self.text()?.map(utf8).transpose()
    }
}

/// The elements of an array payload of `count` elements stored as `stored`, checked: text must be
/// UTF-8, booleans 0 or 1, and the payload exactly its elements ([`Error::ShortPayload`] when it
/// ends before them, [`Error::TrailingPayload`] when bytes follow them).
pub(super) fn decode(
    stored: ValueType,
    order: ByteOrder,
    count: usize,
    bytes: &[u8],
) -> Result<Elements, Error> {
    let mut decoder = Decoder { bytes, order };
    let elements = match stored.kind() {
        Kind::String => {
            let mut strings =
                Strings::with_capacity(count, bytes.len().saturating_sub(count.saturating_mul(8)));
            for _ in 0..count {
                strings.push(decoder.string()?)?;
            }
            Elements::String(strings)
        }
        Kind::Bool => Elements::Bool(
            (0..count)
                .map(|_| decoder.bool())
                .collect::<Result<_, _>>()?,
        ),
        Kind::Unsigned => Elements::Unsigned(
            (0..count)
                .map(|_| decoder.unsigned(stored))
                .collect::<Result<_, _>>()?,
        ),
        Kind::Signed => Elements::Signed(
            (0..count)
                .map(|_| decoder.signed(stored))
                .collect::<Result<_, _>>()?,
        ),
        Kind::Float => Elements::Float(
            (0..count)
                .map(|_| decoder.float(stored))
                .collect::<Result<_, _>>()?,
        ),
    };
    if !decoder.bytes.is_empty() {
        return Err(Error::TrailingPayload);
    }
    Ok(elements)
}

/// The `index`th element of an array payload stored as `stored`, checked as [`decode`] checks it;
/// the elements before it are passed by their lengths alone.
pub(super) fn element(
    stored: ValueType,
    order: ByteOrder,
    bytes: &[u8],
    index: usize,
) -> Result<Element<'_>, Error> {
    let mut decoder = Decoder { bytes, order };
    match stored.kind() {
        Kind::String => {
            for _ in 0..index {
                decoder.text()?;
            }
        }
        _ => {
            decoder.bytes = index
                .checked_mul(stored.width())
                .and_then(|at| bytes.get(at..))
                .ok_or(Error::ShortPayload)?;
        }
    }
    Ok(match stored.kind() {
        Kind::String => Element::String(decoder.string()?),
        Kind::Bool => Element::Bool(decoder.bool()?),
        Kind::Unsigned => Element::Unsigned(decoder.unsigned(stored)?),
        Kind::Signed => Element::Signed(decoder.signed(stored)?),
        Kind::Float => Element::Float(decoder.float(stored)?),
    })
}

/// The elements at `indices` of an array payload of `count` elements stored as `stored`, returned
/// in the order of `indices` (`None` for any index `>= count`). For string arrays, walks the
/// payload once up to the largest in-bounds index, skipping unrequested elements by length alone.
pub(super) fn elements_at<'a>(
    stored: ValueType,
    order: ByteOrder,
    count: usize,
    bytes: &'a [u8],
    indices: &[usize],
) -> Result<Vec<Option<Element<'a>>>, Error> {
    if stored.kind() != Kind::String {
        return indices
            .iter()
            .map(|&index| {
                if index < count {
                    element(stored, order, bytes, index).map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect();
    }
    let mut out = vec![None; indices.len()];
    let mut targets: Vec<(usize, usize)> = indices
        .iter()
        .enumerate()
        .filter_map(|(pos, &idx)| (idx < count).then_some((idx, pos)))
        .collect();
    if targets.is_empty() {
        return Ok(out);
    }
    targets.sort_unstable_by_key(|&(idx, _)| idx);
    if order == ByteOrder::Little {
        let len = bytes.len();
        let mut pos = 0usize;
        let mut cursor = 0usize;
        let mut i = 0;
        while i < targets.len() {
            let target = targets[i].0;
            while cursor < target {
                if len - pos < 8 {
                    return Err(Error::ShortPayload);
                }
                let n = u64::from_le_bytes(
                    bytes[pos..pos + 8]
                        .try_into()
                        .expect("eight bytes"),
                );
                pos += 8;
                if n > (len - pos) as u64 {
                    return Err(Error::ShortPayload);
                }
                pos += n as usize;
                cursor += 1;
            }
            if len - pos < 8 {
                return Err(Error::ShortPayload);
            }
            let n = u64::from_le_bytes(
                bytes[pos..pos + 8]
                    .try_into()
                    .expect("eight bytes"),
            );
            pos += 8;
            if n > (len - pos) as u64 {
                return Err(Error::ShortPayload);
            }
            let end = pos + n as usize;
            let value = Element::String(utf8(&bytes[pos..end])?);
            pos = end;
            cursor += 1;
            while i < targets.len() && targets[i].0 == target {
                out[targets[i].1] = Some(value);
                i += 1;
            }
        }
        return Ok(out);
    }
    let mut decoder = Decoder { bytes, order };
    let mut cursor = 0;
    let mut i = 0;
    while i < targets.len() {
        let target = targets[i].0;
        while cursor < target {
            decoder.text()?;
            cursor += 1;
        }
        let value = Element::String(decoder.string()?);
        cursor += 1;
        while i < targets.len() && targets[i].0 == target {
            out[targets[i].1] = Some(value);
            i += 1;
        }
    }
    Ok(out)
}

/// Decodes the bytes of a value; one that ends early is [`Error::ShortPayload`].
struct Decoder<'a> {
    bytes: &'a [u8],
    order: ByteOrder,
}

impl<'a> Decoder<'a> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let (taken, rest) = self.bytes.split_first_chunk().ok_or(Error::ShortPayload)?;
        self.bytes = rest;
        Ok(*taken)
    }

    /// A string's bytes, unchecked.
    fn text(&mut self) -> Result<&'a [u8], Error> {
        let length =
            usize::try_from(u64_of(self.order, self.take()?)).map_err(|_| Error::ShortPayload)?;
        let (text, rest) = self
            .bytes
            .split_at_checked(length)
            .ok_or(Error::ShortPayload)?;
        self.bytes = rest;
        Ok(text)
    }

    /// A string, checked to be UTF-8.
    fn string(&mut self) -> Result<&'a str, Error> {
        utf8(self.text()?)
    }

    /// A boolean, checked to be 0 or 1.
    fn bool(&mut self) -> Result<bool, Error> {
        valid_bool(self.take::<1>()?[0])
    }

    fn unsigned(&mut self, kind: ValueType) -> Result<u64, Error> {
        Ok(match kind.width() {
            1 => u64::from(self.take::<1>()?[0]),
            2 => u64::from(u16_of(self.order, self.take()?)),
            4 => u64::from(u32_of(self.order, self.take()?)),
            _ => u64_of(self.order, self.take()?),
        })
    }

    fn signed(&mut self, kind: ValueType) -> Result<i64, Error> {
        Ok(match kind.width() {
            1 => i64::from(self.take::<1>()?[0] as i8),
            2 => i64::from(u16_of(self.order, self.take()?) as i16),
            4 => i64::from(u32_of(self.order, self.take()?) as i32),
            _ => u64_of(self.order, self.take()?) as i64,
        })
    }

    fn float(&mut self, kind: ValueType) -> Result<f64, Error> {
        Ok(match kind.width() {
            4 => f64::from(f32::from_bits(u32_of(self.order, self.take()?))),
            _ => f64::from_bits(u64_of(self.order, self.take()?)),
        })
    }

    /// A scalar, its text and boolean as the walk checked them.
    fn scalar(&mut self, kind: ValueType) -> Result<Scalar, Error> {
        Ok(match kind {
            ValueType::String => Scalar::String(checked(self.text()?).to_owned()),
            ValueType::Bool => Scalar::Bool(self.take::<1>()?[0] == 1),
            ValueType::U8 | ValueType::U16 | ValueType::U32 | ValueType::U64 => {
                Scalar::Unsigned(self.unsigned(kind)?)
            }
            ValueType::I8 | ValueType::I16 | ValueType::I32 | ValueType::I64 => {
                Scalar::Signed(self.signed(kind)?)
            }
            ValueType::F32 | ValueType::F64 => Scalar::Float(self.float(kind)?),
        })
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
        let bytes = Arc::new(HeaderBacking::Owned(Arc::new(header(10_000))));
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
        let Value::Array(pieces) = tokens.value(&bytes) else {
            panic!("an array");
        };
        assert_eq!(pieces.len(), 10_000);
        let Ok(Elements::String(pieces)) = pieces.elements() else {
            panic!("a text array");
        };
        assert_eq!(pieces.get(9_999), Some("token-009999"));
        assert_eq!(
            header.entries[2].value(&bytes),
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
        // The first byte of the text of the key after the token list.
        let at = whole.entries[2].head.start + 8;
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
    fn array_text_is_checked_when_it_is_decoded_not_when_it_is_walked() {
        let mut bytes = header(100);
        let at = bytes
            .windows(b"token-000042".len())
            .position(|window| window == b"token-000042")
            .unwrap();
        bytes[at] = 0xff;
        let bytes = Arc::new(HeaderBacking::Owned(Arc::new(bytes)));
        let Ok(Walk::Complete(whole)) = walk(&bytes) else {
            panic!("a whole header");
        };
        let Value::Array(tokens) = whole.entries[1].value(&bytes) else {
            panic!("an array");
        };
        assert_eq!(tokens.len(), 100);
        for _ in 0..2 {
            assert!(
                matches!(tokens.elements(), Err(Error::Invalid(message)) if message.contains("UTF-8"))
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

    /// A header holding only `pieces` as its token list, every number in `order`.
    fn tokens_header(order: ByteOrder, pieces: &[&str]) -> Vec<u8> {
        let u32 = |value: u32| match order {
            ByteOrder::Little => value.to_le_bytes(),
            ByteOrder::Big => value.to_be_bytes(),
        };
        let u64 = |value: u64| match order {
            ByteOrder::Little => value.to_le_bytes(),
            ByteOrder::Big => value.to_be_bytes(),
        };
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&u32(3));
        bytes.extend_from_slice(&u64(0));
        bytes.extend_from_slice(&u64(1));
        let key = "tokenizer.ggml.tokens";
        bytes.extend_from_slice(&u64(key.len() as u64));
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&u32(9));
        bytes.extend_from_slice(&u32(8));
        bytes.extend_from_slice(&u64(pieces.len() as u64));
        for piece in pieces {
            bytes.extend_from_slice(&u64(piece.len() as u64));
            bytes.extend_from_slice(piece.as_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes
    }

    /// The token list of a directory.
    fn tokens(directory: &crate::gguf::Directory) -> Array {
        match directory.value("tokenizer.ggml.tokens") {
            Some(Value::Array(tokens)) => tokens.clone(),
            other => panic!("a token list, not {other:?}"),
        }
    }

    /// `header(tokens)`'s skeleton, its token list's payload apart in `file`, and that payload.
    fn skeleton(tokens: usize, file: PathBuf) -> (Skeleton, Vec<u8>) {
        let bytes = header(tokens);
        let Ok(Walk::Complete(whole)) = walk(&bytes) else {
            panic!("a whole header");
        };
        let list = whole.entries[1];
        let payload = bytes[list.head.end..list.end].to_vec();
        let mut held = bytes[..list.head.end].to_vec();
        held.extend_from_slice(&bytes[list.end..]);
        let absence = Absence {
            offset: list.head.end,
            length: payload.len(),
            digest: crate::gguf::sha256(&payload),
            file,
        };
        (
            Skeleton {
                bytes: held,
                absent: vec![absence],
            },
            payload,
        )
    }

    /// A file of this test process named `name`, in the system's temporary directory.
    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("magnitude-walk-{}-{name}", std::process::id()))
    }

    #[test]
    fn a_payload_must_hold_exactly_its_elements() {
        let order = ByteOrder::Little;
        assert!(matches!(
            decode(ValueType::U32, order, 3, &[0; 11]),
            Err(Error::ShortPayload)
        ));
        assert!(matches!(
            decode(ValueType::U32, order, 3, &[0; 13]),
            Err(Error::TrailingPayload)
        ));
        assert!(matches!(
            element(ValueType::U32, order, &[0; 11], 2),
            Err(Error::ShortPayload)
        ));
        let mut text = 5_u64.to_le_bytes().to_vec();
        text.extend_from_slice(b"abc");
        assert!(matches!(
            decode(ValueType::String, order, 1, &text),
            Err(Error::ShortPayload)
        ));
        assert!(matches!(
            element(ValueType::String, order, &text, 0),
            Err(Error::ShortPayload)
        ));
        assert!(matches!(
            element(ValueType::String, order, &text, 1),
            Err(Error::ShortPayload)
        ));
        assert!(matches!(
            decode(ValueType::String, order, 1, &[0; 7]),
            Err(Error::ShortPayload)
        ));
    }

    #[test]
    fn an_array_boolean_other_than_zero_or_one_is_refused_when_read() {
        let order = ByteOrder::Little;
        assert!(matches!(
            decode(ValueType::Bool, order, 2, &[1, 0]),
            Ok(Elements::Bool(_))
        ));
        assert!(matches!(
            decode(ValueType::Bool, order, 2, &[1, 2]),
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            element(ValueType::Bool, order, &[1, 2], 1),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn a_payload_apart_that_is_not_its_bytes_is_damaged_and_lost() {
        let file = temporary("damaged");
        let (skeleton, payload) = skeleton(3_001, file.clone());
        let directory = crate::gguf::read_skeleton(skeleton).unwrap();

        let mut longer = payload.clone();
        longer.push(0);
        std::fs::write(&file, &longer).unwrap();
        let list = tokens(&directory);
        assert!(matches!(list.elements(), Err(Error::DamagedPayload { .. })));
        assert!(list.lost_payload(), "a failed read is remembered");

        let mut changed = payload.clone();
        changed[20] ^= 1;
        std::fs::write(&file, &changed).unwrap();
        let list = tokens(&directory);
        assert!(matches!(list.element(0), Err(Error::DamagedPayload { .. })));

        std::fs::remove_file(&file).unwrap();
        assert!(matches!(list.element(0), Err(Error::PayloadRead { .. })));

        std::fs::write(&file, &payload).unwrap();
        assert_eq!(
            list.element(3_000).unwrap(),
            Some(Element::String("token-003000"))
        );
        std::fs::remove_file(&file).unwrap();
    }

    #[test]
    fn an_absence_that_does_not_fit_its_array_is_refused() {
        let file = temporary("unfit");
        let (mut unfit, _) = skeleton(3_002, file.clone());
        unfit.absent[0].length = 3_002 * 8 - 1;
        assert!(matches!(
            crate::gguf::read_skeleton(unfit),
            Err(Error::Invalid(message)) if message.contains("does not fit its count")
        ));

        // An absence inside a scalar entry is no array's payload.
        let bytes = header(10);
        let absence = Absence {
            offset: 30,
            length: 4,
            digest: [0; 32],
            file: file.clone(),
        };
        let inside = Skeleton {
            bytes: bytes.clone(),
            absent: vec![absence.clone()],
        };
        assert!(matches!(
            crate::gguf::read_skeleton(inside),
            Err(Error::Invalid(message)) if message.contains("not a metadata array's")
        ));

        // Absences out of order: the second lies before the first.
        let (mut unordered, _) = skeleton(3_003, file);
        unordered.absent.push(absence);
        assert!(matches!(
            crate::gguf::read_skeleton(unordered),
            Err(Error::Invalid(message)) if message.contains("not a metadata array's")
        ));
    }

    #[test]
    fn header_bytes_share_only_the_whole_walked_header() {
        let bytes = header(100);
        let read = |length: usize| {
            let mut read = HeaderBytes::new();
            read.extend_from_slice(&bytes[..length]);
            read.walk().unwrap();
            read
        };
        assert!(matches!(
            read(100).into_shared(100),
            Err(Error::Truncated { .. })
        ));
        assert!(matches!(
            read(bytes.len()).into_shared(bytes.len() + 1),
            Err(Error::Truncated { .. })
        ));
        assert!(matches!(
            read(bytes.len()).into_shared(10),
            Err(Error::Invalid(message)) if message.contains("cut the GGUF header")
        ));
        assert!(read(bytes.len()).into_shared(bytes.len()).is_ok());
    }

    #[test]
    fn a_text_array_has_one_digest_however_it_is_stored_and_no_other_array_has_one() {
        let pieces = ["<s>", "", "héllo", " world"];
        let little = crate::gguf::read_header(tokens_header(ByteOrder::Little, &pieces)).unwrap();
        let big = crate::gguf::read_header(tokens_header(ByteOrder::Big, &pieces)).unwrap();
        let built = Array::from(pieces.into_iter().collect::<Strings>());
        let digest = tokens(&little).digest().unwrap();
        assert_eq!(tokens(&big).digest().unwrap(), digest);
        assert_eq!(built.digest().unwrap(), digest);
        assert!(matches!(
            Array::from(vec![1_u64]).digest(),
            Err(Error::DigestOfNonText)
        ));
    }

    #[test]
    fn a_detached_directory_keeps_no_header_bytes_alive() {
        let bytes = Arc::new(header(1_000));
        let mut directory = crate::gguf::read_header(Arc::clone(&bytes)).unwrap();
        let text = |directory: &crate::gguf::Directory| {
            let list = tokens(directory);
            let Elements::String(strings) = list.elements().unwrap() else {
                panic!("tokens are text");
            };
            strings.text.as_ptr()
        };
        let decoded = tokens(&directory).elements().unwrap().clone();
        let held = text(&directory);
        assert!(
            Arc::strong_count(&bytes) > 1,
            "the arrays share the header's bytes"
        );
        directory.detach().unwrap();
        assert_eq!(
            Arc::strong_count(&bytes),
            1,
            "a detached directory holds none of them"
        );
        assert_eq!(tokens(&directory).elements().unwrap(), &decoded);
        assert_eq!(
            text(&directory),
            held,
            "decoded elements are moved, not copied"
        );
    }

    #[test]
    fn elements_at_reads_unsorted_duplicate_and_out_of_bounds_indices_in_one_pass() {
        let bytes = Arc::new(header(100));
        let directory = crate::gguf::read_header(Arc::clone(&bytes)).unwrap();
        let list = tokens(&directory);
        let indices = [42, 7, 42, 100, 0, 99];
        let expected = vec![
            Some(Element::String("token-000042")),
            Some(Element::String("token-000007")),
            Some(Element::String("token-000042")),
            None,
            Some(Element::String("token-000000")),
            Some(Element::String("token-000099")),
        ];
        assert_eq!(list.elements_at(&indices).unwrap(), expected);
        assert_eq!(list.elements_at(&[]).unwrap(), vec![]);
        assert_eq!(list.elements_at(&[100, 200]).unwrap(), vec![None, None]);
        // After full decode, elements_at matches the stored single-pass result.
        let _ = list.elements().unwrap();
        assert_eq!(list.elements_at(&indices).unwrap(), expected);
    }
}
