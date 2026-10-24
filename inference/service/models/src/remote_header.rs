//! Reading one remote GGUF header by exact range requests.
//!
//! The reader walks the header ([`magnitude_artifacts::gguf::walk`]) in the bytes in hand and asks
//! for about as many more as the walk says the header still needs, so a header costs its own
//! length plus little, in few requests. Every file of a repository after the first is read with
//! the headers already read as references: quantizations of one model carry the same tokenizer,
//! which is nearly all of a header (16 MB of a Gemma 4 header's 16 MB), as a few large metadata
//! entries. When the walk reaches an entry whose fixed part (key, type and element count) matches
//! a reference's, the run of large entries that starts there is checked in one round of concurrent
//! requests: windows at the start, middle and end of each entry, compared byte for byte with the
//! reference's. The leading entries whose windows all match are taken from the reference, and the
//! read continues with the file's own bytes. Small entries around the tokenizer (special token
//! ids, names, quantization keys) may differ freely; an entry that differs (an MTP drafter's token
//! types) is read from the file, and the entries after it are checked again. Equality outside the
//! windows is assumed: same-length differences there go unseen, which is why the windows sit where
//! tokenizers of one model differ (special tokens first, added tokens last).

use std::ops::Range;

use magnitude_service_contracts::InventoryError;

use magnitude_artifacts::gguf::{Entry, EntryHead, HEADER_LIMIT, Walk, walk};

use crate::header_cache::shared;
use crate::preview::RemoteRanges;

/// The first read of a header with no reference.
pub(crate) const INITIAL_READ_BYTES: usize = 64 * 1024;
/// The smallest further read: a header's tail of small entries and tensor descriptions.
const MIN_READ_BYTES: usize = 64 * 1024;
/// The largest further read, so an estimate extrapolated from a list's first strings overshoots
/// little: the next estimate measures every string read so far.
const MAX_READ_BYTES: usize = 4 * 1024 * 1024;
/// Bytes compared at the start and at the end of each shared entry, where tokens that differ
/// between otherwise equal tokenizers (special and added tokens) sit.
const EDGE_BYTES: usize = 16 * 1024;
/// Bytes compared in the middle of each shared entry.
const WINDOW_BYTES: usize = 8 * 1024;

struct Reference<'a> {
    bytes: &'a [u8],
    entries: Vec<Entry<'a>>,
}

/// A run of consecutive large entries of one reference.
struct SharedRun<'a> {
    reference: &'a [u8],
    entries: &'a [Entry<'a>],
}

impl SharedRun<'_> {
    fn start(&self) -> usize {
        self.entries[0].head.start
    }

    fn end(&self) -> usize {
        self.entries[self.entries.len() - 1].end
    }
}

/// Every byte of a remote GGUF header through its aligned tensor-data offset (or the whole file,
/// when it holds no tensor data and ends before that offset's padding), sharing large entries with
/// `references`. An error is a transport failure or bytes that can't be a GGUF header; whether
/// the engine accepts the header is decided on the result.
pub(crate) async fn read_header(
    remote: &mut RemoteRanges<'_>,
    references: &[&[u8]],
) -> Result<Vec<u8>, InventoryError> {
    let references = references
        .iter()
        .filter_map(|bytes| match walk(bytes) {
            Ok(Walk::Complete(header)) if header.entries.iter().any(shared) => Some(Reference {
                bytes,
                entries: header.entries,
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    let logical_length = remote.logical_length();
    let bound = HEADER_LIMIT.min(logical_length);
    // With references, the first read reaches just past where they begin their shared entries.
    let first = references
        .iter()
        .filter_map(|reference| reference.entries.iter().find(|entry| shared(entry)))
        .map(|entry| entry.head.end + EDGE_BYTES)
        .min()
        .unwrap_or(INITIAL_READ_BYTES)
        .min(bound);
    if first == 0 {
        return Err(InventoryError::Integrity(
            "the file ends before its GGUF header does".to_owned(),
        ));
    }
    let mut bytes = remote.read(0, first).await?;
    let mut declined = Vec::new();
    loop {
        match walk(&bytes).map_err(|error| InventoryError::Integrity(error.to_string()))? {
            Walk::Complete(header) => {
                let end = header.data_offset.min(logical_length);
                if bytes.len() < end {
                    let rest = remote.read(bytes.len(), end).await?;
                    bytes.extend_from_slice(&rest);
                }
                bytes.truncate(end);
                return Ok(bytes);
            }
            Walk::Incomplete(prefix) => {
                if bytes.len() + prefix.minimum > logical_length {
                    return Err(InventoryError::Integrity(
                        "the file ends before its GGUF header does".to_owned(),
                    ));
                }
                let estimate = prefix.estimate;
                if let Some(open) = prefix.open.filter(|open| !declined.contains(&open.start))
                    && let Some(run) = shared_run(&references, &bytes, &open)
                {
                    let start = open.start;
                    let taken = share(remote, &mut bytes, start, &run, bound).await?;
                    if taken < run.entries.len() {
                        declined.push(start + run.entries[taken].head.start - run.start());
                    }
                    if taken > 0 {
                        continue;
                    }
                }
                let target = bytes
                    .len()
                    .saturating_add(estimate.clamp(MIN_READ_BYTES, MAX_READ_BYTES))
                    .min(bound);
                let range = remote.read(bytes.len(), target).await?;
                bytes.extend_from_slice(&range);
            }
        }
    }
}

/// The run of large reference entries starting with one whose fixed part is `open`'s.
fn shared_run<'a>(
    references: &'a [Reference<'a>],
    bytes: &[u8],
    open: &EntryHead<'_>,
) -> Option<SharedRun<'a>> {
    let head = &bytes[open.start..open.end];
    references.iter().find_map(|reference| {
        let first = reference.entries.iter().position(|entry| {
            shared(entry)
                && entry.head.key == open.key
                && reference.bytes[entry.head.start..entry.head.end] == *head
        })?;
        let count = reference.entries[first..]
            .iter()
            .take_while(|entry| shared(entry))
            .count();
        Some(SharedRun {
            reference: reference.bytes,
            entries: &reference.entries[first..first + count],
        })
    })
}

/// Takes the verified entries of `run` from its reference into `bytes` (which end inside the
/// run's first entry, which starts at `open`), returning how many: the leading entries whose
/// verification windows (the start and end of each, where differing special tokens sit, and its
/// middle) equal the file's own bytes. When the whole run is taken, the file's bytes after it
/// follow, about as many as the reference has after it.
async fn share(
    remote: &mut RemoteRanges<'_>,
    bytes: &mut Vec<u8>,
    open: usize,
    run: &SharedRun<'_>,
    bound: usize,
) -> Result<usize, InventoryError> {
    let reference = run.reference;
    let held = bytes.len();
    let target = |offset: usize| open + offset - run.start();
    let end = target(run.end());
    if end > bound || end <= held + WINDOW_BYTES {
        return Ok(0);
    }
    if bytes[open..] != reference[run.start()..run.start() + held - open] {
        return Ok(0);
    }
    let checks = run
        .entries
        .iter()
        .map(|entry| {
            let (start, end) = (target(entry.head.start), target(entry.end));
            let middle = target(entry.head.start + entry.range().len() / 2);
            [
                start..start + EDGE_BYTES,
                middle - WINDOW_BYTES / 2..middle + WINDOW_BYTES / 2,
                end - EDGE_BYTES..end,
            ]
            .map(|window| window.start.max(held)..window.end)
        })
        .collect::<Vec<_>>();
    let tail = end..end
        .saturating_add(reference.len() - run.end() + WINDOW_BYTES)
        .min(bound);
    let requests = merged(
        checks
            .iter()
            .flatten()
            .cloned()
            .chain([tail.clone()])
            .collect(),
        held,
    );
    let reads = futures_util::future::try_join_all(requests.iter().map(|request| {
        let mut remote = remote.clone();
        let request = request.clone();
        async move { remote.read(request.start, request.end).await }
    }))
    .await?;
    let file = |window: &Range<usize>| {
        requests
            .iter()
            .zip(&reads)
            .find(|(request, _)| request.start <= window.start && window.end <= request.end)
            .map(|(request, read)| &read[window.start - request.start..window.end - request.start])
    };
    let verified = checks
        .iter()
        .take_while(|windows| {
            windows
                .iter()
                .filter(|window| window.start < window.end)
                .all(|window| {
                    let from = run.start() + window.start - open;
                    file(window) == Some(&reference[from..from + window.len()])
                })
        })
        .count();
    if verified == 0 {
        return Ok(0);
    }
    bytes.extend_from_slice(&reference[run.start() + held - open..run.entries[verified - 1].end]);
    if verified == run.entries.len()
        && !tail.is_empty()
        && let Some(after) = file(&tail)
    {
        bytes.extend_from_slice(after);
    }
    Ok(verified)
}

/// `windows` clipped to start at `from`, sorted, with overlapping ones joined.
fn merged(mut windows: Vec<Range<usize>>, from: usize) -> Vec<Range<usize>> {
    for window in &mut windows {
        window.start = window.start.max(from);
    }
    windows.retain(|window| window.start < window.end);
    windows.sort_by_key(|window| window.start);
    let mut joined: Vec<Range<usize>> = Vec::with_capacity(windows.len());
    for window in windows {
        match joined.last_mut() {
            Some(last) if window.start <= last.end => last.end = last.end.max(window.end),
            _ => joined.push(window),
        }
    }
    joined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_clipped_to_the_bytes_not_in_hand_and_joined_where_they_overlap() {
        assert_eq!(
            merged(vec![50..60, 0..20, 15..30, 70..70], 10),
            vec![10..30, 50..60]
        );
    }
}
