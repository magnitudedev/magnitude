//! Reading a GGUF header by reusing a sibling's tokenizer.
//!
//! Every quantization of one model in a repository carries the same tokenizer, which is nearly
//! all of a header's bytes (about 11 MB of an 11 MB header for a 250k-token vocabulary). A header
//! whose sibling is already known is read as its own prefix up to the tokenizer, three verification
//! windows inside the tokenizer (start, middle, end) compared byte for byte with the sibling's,
//! and its own remainder after the tokenizer: well under 1 MB in three requests. Any mismatch, or
//! any header that doesn't parse, falls back to reading the whole header.

use magnitude_service_contracts::InventoryError;

use crate::preview::{RemoteRanges, header_length};

/// Bytes compared at each end and in the middle of the shared tokenizer.
const VERIFY_BYTES: usize = 64 * 1024;
/// Extra bytes read past where the sibling places the tokenizer and past its remainder, for
/// metadata that differs in length between quantizations (names, quantization keys).
const MARGIN_BYTES: usize = 64 * 1024;
/// A shared tokenizer shorter than this is not worth the extra requests.
const MIN_SHARED_BYTES: usize = 1024 * 1024;

/// One metadata entry of a GGUF header: its key and its byte range, entry included.
#[derive(Debug, PartialEq, Eq)]
struct Entry {
    key: String,
    start: usize,
    end: usize,
}

/// The metadata entries wholly inside `bytes`, in order. A little-endian GGUF v2 or v3 header is
/// walked; anything else yields no entries.
fn entries(bytes: &[u8]) -> Vec<Entry> {
    let mut reader = Reader { bytes, offset: 0 };
    let mut entries = Vec::new();
    let Some(count) = reader.preamble() else {
        return entries;
    };
    for _ in 0..count {
        let start = reader.offset;
        let Some(key) = reader.string() else {
            break;
        };
        let Some(value_type) = reader.u32() else {
            break;
        };
        if reader.skip_value(value_type).is_none() {
            break;
        }
        entries.push(Entry {
            key: String::from_utf8_lossy(key).into_owned(),
            start,
            end: reader.offset,
        });
    }
    entries
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
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

    fn string(&mut self) -> Option<&'a [u8]> {
        let length = usize::try_from(self.u64()?).ok()?;
        self.take(length)
    }

    /// The metadata entry count after the magic, version and tensor count.
    fn preamble(&mut self) -> Option<u64> {
        if self.take(4)? != b"GGUF" || !matches!(self.u32()?, 2 | 3) {
            return None;
        }
        self.u64()?;
        self.u64()
    }

    fn skip_value(&mut self, value_type: u32) -> Option<()> {
        match value_type {
            0 | 1 | 7 => self.take(1).map(drop),
            2 | 3 => self.take(2).map(drop),
            4..=6 => self.take(4).map(drop),
            10..=12 => self.take(8).map(drop),
            8 => self.string().map(drop),
            9 => {
                let element_type = self.u32()?;
                let count = self.u64()?;
                match element_type {
                    8 => (0..count).try_for_each(|_| self.string().map(drop)),
                    9 => None,
                    element_type => {
                        let width = match element_type {
                            0 | 1 | 7 => 1,
                            2 | 3 => 2,
                            4..=6 => 4,
                            10..=12 => 8,
                            _ => return None,
                        };
                        self.take(usize::try_from(count).ok()?.checked_mul(width)?)
                            .map(drop)
                    }
                }
            }
            _ => None,
        }
    }
}

/// The sibling's tokenizer: the consecutive `tokenizer.ggml.*` entries, by byte range, with the
/// first entry's key.
fn tokenizer_region(sibling: &[u8]) -> Option<(String, usize, usize)> {
    let entries = entries(sibling);
    let first = entries
        .iter()
        .position(|entry| entry.key.starts_with("tokenizer.ggml."))?;
    let last = entries[first..]
        .iter()
        .take_while(|entry| entry.key.starts_with("tokenizer.ggml."))
        .last()?;
    let region = (entries[first].key.clone(), entries[first].start, last.end);
    (region.2 - region.1 >= MIN_SHARED_BYTES.max(3 * VERIFY_BYTES)).then_some(region)
}

/// The header of `remote` read with `sibling`'s tokenizer, or `None` when the tokenizer is not
/// shared or the header doesn't parse; the caller then reads the whole header.
pub(crate) async fn read_header_sharing_tokenizer(
    remote: &mut RemoteRanges<'_>,
    sibling: &[u8],
) -> Result<Option<Vec<u8>>, InventoryError> {
    let Some((key, shared_start, shared_end)) = tokenizer_region(sibling) else {
        return Ok(None);
    };
    let logical_length = remote.logical_length();
    let prefix_end = (shared_start + MARGIN_BYTES + VERIFY_BYTES).min(logical_length);
    let prefix = remote.read(0, prefix_end).await?;
    let Some(start) = entries(&prefix)
        .into_iter()
        .find(|entry| entry.key == key)
        .map(|entry| entry.start)
        .or_else(|| prefix_entry_start(&prefix, &key))
    else {
        return Ok(None);
    };
    let length = shared_end - shared_start;
    let shared = &sibling[shared_start..shared_end];
    let end = start + length;
    let tail_end = (end + (sibling.len() - shared_end) + MARGIN_BYTES).min(logical_length);
    if prefix.len() < start + VERIFY_BYTES
        || prefix[start..start + VERIFY_BYTES] != shared[..VERIFY_BYTES]
        || tail_end <= end
    {
        return Ok(None);
    }
    let middle = start + length / 2;
    let mut middle_remote = remote.clone();
    let (middle_window, tail) = futures_util::future::join(
        middle_remote.read(middle, middle + VERIFY_BYTES),
        remote.read(end - VERIFY_BYTES, tail_end),
    )
    .await;
    let (middle_window, tail) = (middle_window?, tail?);
    if middle_window[..] != shared[length / 2..length / 2 + VERIFY_BYTES]
        || tail[..VERIFY_BYTES] != shared[length - VERIFY_BYTES..]
    {
        return Ok(None);
    }
    let mut header = Vec::with_capacity(start + length + tail.len() - VERIFY_BYTES);
    header.extend_from_slice(&prefix[..start]);
    header.extend_from_slice(shared);
    header.extend_from_slice(&tail[VERIFY_BYTES..]);
    match parsed_length(header.clone()).await? {
        Some(length) if length <= header.len() => {
            header.truncate(length);
            Ok(Some(header))
        }
        Some(length) if length <= logical_length => {
            let rest = remote.read(header.len(), length).await?;
            header.extend_from_slice(&rest);
            Ok(parsed_length(header.clone()).await?.is_some_and(|parsed| parsed == length).then_some(header))
        }
        _ => Ok(None),
    }
}

/// The aligned tensor-data offset of a reconstructed header, parsed off the async runtime, or
/// `None` when it doesn't parse.
async fn parsed_length(header: Vec<u8>) -> Result<Option<usize>, InventoryError> {
    tokio::task::spawn_blocking(move || header_length(&header).ok())
        .await
        .map_err(|error| InventoryError::Internal(error.to_string()))
}

/// The start of the entry named `key` when it begins inside `prefix` but doesn't end there: the
/// tokenizer's first entry is usually longer than any prefix read.
fn prefix_entry_start(prefix: &[u8], key: &str) -> Option<usize> {
    let complete = entries(prefix);
    let start = complete.last().map_or_else(
        || {
            let mut reader = Reader { bytes: prefix, offset: 0 };
            reader.preamble().map(|_| reader.offset)
        },
        |entry| Some(entry.end),
    )?;
    let mut reader = Reader {
        bytes: prefix,
        offset: start,
    };
    (reader.string()? == key.as_bytes()).then_some(start)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    /// A header with a name, a token list of `tokens` entries and a trailing quantization key.
    fn header(name: &str, tokens: usize, file_type: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        string(&mut bytes, "general.name");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        string(&mut bytes, name);
        string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&(tokens as u64).to_le_bytes());
        for token in 0..tokens {
            string(&mut bytes, &format!("token-{token:06}"));
        }
        string(&mut bytes, "general.file_type");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&file_type.to_le_bytes());
        bytes
    }

    #[test]
    fn entries_cover_each_metadata_key_and_stop_at_a_truncated_one() {
        let bytes = header("model Q4", 3, 15);
        let walked = entries(&bytes);
        assert_eq!(
            walked.iter().map(|entry| entry.key.as_str()).collect::<Vec<_>>(),
            ["general.name", "tokenizer.ggml.tokens", "general.file_type"]
        );
        assert_eq!(walked.last().unwrap().end, bytes.len());
        let truncated = entries(&bytes[..walked[1].start + 40]);
        assert_eq!(truncated.len(), 1);
        assert_eq!(
            prefix_entry_start(&bytes[..walked[1].start + 40], "tokenizer.ggml.tokens"),
            Some(walked[1].start)
        );
    }

    #[test]
    fn the_tokenizer_region_is_found_only_when_it_is_worth_sharing() {
        let large = header("model Q4", 120_000, 15);
        let (key, start, end) = tokenizer_region(&large).expect("a large tokenizer is shared");
        assert_eq!(key, "tokenizer.ggml.tokens");
        assert_eq!(&large[end..end + 8], &17_u64.to_le_bytes(), "the region ends before the next key");
        assert!(end - start >= MIN_SHARED_BYTES);
        assert_eq!(tokenizer_region(&header("model Q4", 10, 15)), None);
    }
}
