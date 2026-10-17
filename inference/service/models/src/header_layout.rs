//! The byte layout of a GGUF header, read from any prefix of it.
//!
//! Walking a prefix tells a remote reader how many more bytes the header needs (exactly for
//! fixed-width values, by extrapolation inside a string array) and which large metadata entries it
//! is about to read, so it can take them from another header instead. The header cache uses the
//! same walk to store those entries once. Only the container's structure is checked here; whether
//! the engine accepts the header (its tensor types, geometry and offsets) is the engine parser's
//! verdict on the complete bytes.

/// The largest header read or stored.
pub(crate) const MAX_HEADER_BYTES: usize = 128 * 1024 * 1024;
/// A metadata entry at least this large (a tokenizer's token list, scores, types or merges) is
/// shared between headers instead of read or stored again.
pub(crate) const SHARED_ENTRY_BYTES: usize = 64 * 1024;

/// Bytes assumed for a metadata entry or tensor description not yet seen.
const ENTRY_ESTIMATE: usize = 64;
const TENSOR_ESTIMATE: usize = 96;
/// The bytes assumed per string of a string array (a token or merge), until enough strings are in
/// hand to measure them, and the bounds of that measure: the first strings of a list are no sample
/// of the rest (merges begin with long runs of whitespace).
const STRING_ESTIMATE: usize = 16;
const STRING_ESTIMATE_BOUNDS: (usize, usize) = (8, 24);
const MEASURED_STRINGS: usize = 1024;
/// The largest tensor rank a GGUF describes (ggml's `GGML_MAX_DIMS` is 4).
const MAX_TENSOR_RANK: u32 = 8;

/// One complete metadata entry: its key and byte range. `head_end` ends the entry's fixed part
/// (key, value type and, for an array, its element type and count, or a string's length), which
/// two entries holding the same kind and amount of data have byte for byte in common.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) key: String,
    pub(crate) start: usize,
    pub(crate) head_end: usize,
    pub(crate) end: usize,
}

impl Entry {
    pub(crate) fn len(&self) -> usize {
        self.end - self.start
    }

    pub(crate) fn shared(&self) -> bool {
        self.len() >= SHARED_ENTRY_BYTES
    }
}

/// A metadata entry whose fixed part is in hand but whose value is not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OpenEntry {
    pub(crate) key: String,
    pub(crate) start: usize,
    pub(crate) head_end: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    /// The whole header is in hand: its entries and its aligned tensor-data offset, which may lie
    /// past the bytes when a file without tensor data ends before its padding.
    Complete {
        entries: Vec<Entry>,
        data_offset: usize,
    },
    /// The bytes end inside the header. `minimum` more bytes are certainly needed and about
    /// `estimate` are expected; `open` is the entry the bytes end in, when its fixed part is in
    /// hand.
    Incomplete {
        entries: Vec<Entry>,
        open: Option<OpenEntry>,
        minimum: usize,
        estimate: usize,
    },
}

/// The layout of a little-endian GGUF v2 or v3 header from its first bytes, or why they can't
/// begin one.
pub(crate) fn walk(bytes: &[u8]) -> Result<Layout, String> {
    let mut cursor = Cursor { bytes, offset: 0 };
    let mut entries = Vec::new();
    let incomplete = |entries: Vec<Entry>, open, minimum: usize, estimate: usize| {
        Ok(Layout::Incomplete {
            entries,
            open,
            minimum: minimum.max(1),
            estimate: estimate.max(minimum).max(1),
        })
    };
    let Some(magic) = cursor.take(4) else {
        return incomplete(entries, None, 24 - bytes.len(), 24 - bytes.len());
    };
    if magic != b"GGUF" {
        return Err("not a GGUF container".to_owned());
    }
    let (Some(version), Some(tensor_count), Some(entry_count)) =
        (cursor.u32(), cursor.u64(), cursor.u64())
    else {
        return incomplete(entries, None, 24 - bytes.len(), 24 - bytes.len());
    };
    if !matches!(version, 2 | 3) {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let bounded = |count: u64| {
        usize::try_from(count)
            .ok()
            .filter(|count| *count <= MAX_HEADER_BYTES / 12)
    };
    let (Some(tensor_count), Some(entry_count)) = (bounded(tensor_count), bounded(entry_count))
    else {
        return Err(beyond());
    };
    let tensors_estimate = tensor_count * TENSOR_ESTIMATE;
    let mut alignment = 32_usize;
    for index in 0..entry_count {
        let start = cursor.offset;
        let rest = (entry_count - index - 1) * ENTRY_ESTIMATE + tensors_estimate;
        let Some(key) = cursor.string()? else {
            return incomplete(entries, None, 12, ENTRY_ESTIMATE + rest);
        };
        let key = String::from_utf8_lossy(key).into_owned();
        let Some(value_type) = cursor.u32() else {
            return incomplete(entries, None, 4, ENTRY_ESTIMATE + rest);
        };
        let open = |head_end| OpenEntry {
            key: key.clone(),
            start,
            head_end,
        };
        match value_type {
            9 => {
                let (Some(element), Some(count)) = (cursor.u32(), cursor.u64()) else {
                    return incomplete(entries, None, 12, ENTRY_ESTIMATE + rest);
                };
                let count = usize::try_from(count).map_err(|_| beyond())?;
                let head_end = cursor.offset;
                if element == 8 {
                    if count
                        .checked_mul(8)
                        .is_none_or(|length| length > MAX_HEADER_BYTES)
                    {
                        return Err(beyond());
                    }
                    for read in 0..count {
                        let before = cursor.offset;
                        if cursor.string()?.is_none() {
                            let left = count - read;
                            let average = if read < MEASURED_STRINGS {
                                STRING_ESTIMATE
                            } else {
                                ((before - head_end) / read)
                                    .clamp(STRING_ESTIMATE_BOUNDS.0, STRING_ESTIMATE_BOUNDS.1)
                            };
                            let in_string = cursor.bytes.len() - before;
                            return incomplete(
                                entries,
                                Some(open(head_end)),
                                (left * 8).saturating_sub(in_string),
                                (left * average).saturating_sub(in_string) + rest,
                            );
                        }
                    }
                } else {
                    let width = scalar_width(element).ok_or_else(|| {
                        if element == 9 {
                            "nested or oversized GGUF array".to_owned()
                        } else {
                            format!("unsupported GGUF metadata type {element}")
                        }
                    })?;
                    let length = count
                        .checked_mul(width)
                        .filter(|length| *length <= MAX_HEADER_BYTES)
                        .ok_or_else(beyond)?;
                    if cursor.take(length).is_none() {
                        let missing = head_end + length - cursor.bytes.len();
                        return incomplete(entries, Some(open(head_end)), missing, missing + rest);
                    }
                }
            }
            8 => {
                let Some(length) = cursor.u64() else {
                    return incomplete(entries, None, 8, ENTRY_ESTIMATE + rest);
                };
                let length = usize::try_from(length)
                    .ok()
                    .filter(|length| *length <= MAX_HEADER_BYTES)
                    .ok_or_else(beyond)?;
                let head_end = cursor.offset;
                if cursor.take(length).is_none() {
                    let missing = head_end + length - cursor.bytes.len();
                    return incomplete(entries, Some(open(head_end)), missing, missing + rest);
                }
            }
            value_type => {
                let width = scalar_width(value_type)
                    .ok_or_else(|| format!("unsupported GGUF metadata type {value_type}"))?;
                let Some(value) = cursor.take(width) else {
                    return incomplete(entries, None, width, width + rest);
                };
                if key == "general.alignment" && value_type == 4 {
                    alignment = u32::from_le_bytes(value.try_into().unwrap_or_default()) as usize;
                    if !alignment.is_power_of_two() {
                        return Err(format!("invalid GGUF alignment {alignment}"));
                    }
                }
            }
        }
        let head_end = match value_type {
            9 => start + 8 + key.len() + 4 + 12,
            8 => start + 8 + key.len() + 4 + 8,
            _ => start + 8 + key.len() + 4,
        };
        entries.push(Entry {
            key,
            start,
            head_end,
            end: cursor.offset,
        });
    }
    let tensors_start = cursor.offset;
    for index in 0..tensor_count {
        let before = cursor.offset;
        let left = tensor_count - index;
        let average = if index == 0 {
            TENSOR_ESTIMATE
        } else {
            (before - tensors_start) / index
        };
        let estimate = left * average + alignment;
        let missing = |cursor: &Cursor<'_>, minimum: usize| {
            (
                minimum,
                estimate.saturating_sub(cursor.bytes.len() - before),
            )
        };
        let Some(_) = cursor.string()? else {
            let (minimum, estimate) = missing(&cursor, 8);
            return incomplete(entries, None, minimum, estimate);
        };
        let Some(rank) = cursor.u32() else {
            let (minimum, estimate) = missing(&cursor, 4);
            return incomplete(entries, None, minimum, estimate);
        };
        if rank > MAX_TENSOR_RANK {
            return Err(format!("invalid GGUF tensor rank {rank}"));
        }
        let fixed = rank as usize * 8 + 4 + 8;
        if cursor.take(fixed).is_none() {
            let (minimum, estimate) = missing(&cursor, fixed);
            return incomplete(entries, None, minimum, estimate);
        }
    }
    Ok(Layout::Complete {
        entries,
        data_offset: cursor.offset.next_multiple_of(alignment),
    })
}

fn beyond() -> String {
    format!("GGUF header extends beyond {MAX_HEADER_BYTES} bytes")
}

fn scalar_width(value_type: u32) -> Option<usize> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, length: usize) -> Option<&'a [u8]> {
        let end = self.offset.checked_add(length)?;
        let taken = self.bytes.get(self.offset..end)?;
        self.offset = end;
        Some(taken)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    /// A string, `None` when the bytes end inside it; the cursor then stays where it was.
    fn string(&mut self) -> Result<Option<&'a [u8]>, String> {
        let start = self.offset;
        let Some(length) = self.u64() else {
            self.offset = start;
            return Ok(None);
        };
        let length = usize::try_from(length)
            .ok()
            .filter(|length| *length <= MAX_HEADER_BYTES)
            .ok_or_else(beyond)?;
        match self.take(length) {
            Some(bytes) => Ok(Some(bytes)),
            None => {
                self.offset = start;
                Ok(None)
            }
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
    fn a_complete_header_lists_its_entries_and_ends_at_its_aligned_data_offset() {
        let bytes = header(10_000);
        let Ok(Layout::Complete {
            entries,
            data_offset,
        }) = walk(&bytes)
        else {
            panic!("complete");
        };
        assert_eq!(data_offset, bytes.len());
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.key.as_str())
                .collect::<Vec<_>>(),
            ["general.name", "tokenizer.ggml.tokens", "general.file_type"]
        );
        let tokens = &entries[1];
        assert!(tokens.shared());
        assert_eq!(
            &bytes[tokens.head_end - 8..tokens.head_end],
            &10_000_u64.to_le_bytes()
        );
        assert_eq!(entries[2].start, tokens.end);
    }

    #[test]
    fn a_prefix_names_the_entry_it_ends_in_and_estimates_the_rest() {
        let bytes = header(20_000);
        let Ok(Layout::Complete { entries, .. }) = walk(&bytes) else {
            panic!("complete");
        };
        let tokens = &entries[1];
        let cut = tokens.start + (tokens.len() / 4);
        let Ok(Layout::Incomplete {
            open,
            minimum,
            estimate,
            ..
        }) = walk(&bytes[..cut])
        else {
            panic!("incomplete");
        };
        assert_eq!(
            open.map(|open| (open.key, open.start)),
            Some(("tokenizer.ggml.tokens".to_owned(), tokens.start))
        );
        let missing = bytes.len() - cut;
        assert!(minimum < missing);
        assert!(
            estimate >= tokens.end - cut && estimate < missing + 4096,
            "{estimate} for {missing}"
        );
        assert!(matches!(
            walk(&bytes[..3]),
            Ok(Layout::Incomplete { open: None, .. })
        ));
    }

    #[test]
    fn a_count_too_large_for_any_header_is_an_error_not_a_longer_read() {
        let mut bytes = header(10);
        let at = bytes
            .windows(8)
            .position(|window| window == 10_u64.to_le_bytes())
            .unwrap();
        bytes[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(walk(&bytes), Err(beyond()));
        assert_eq!(walk(b"GGML"), Err("not a GGUF container".to_owned()));
    }
}
