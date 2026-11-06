//! A source's stream reads exactly its range into the file cache.

#![cfg(target_os = "linux")]

use std::os::fd::AsRawFd;

const BLOCK: u64 = 128 * 1024;

/// A source's stream reads exactly its range through the cache: an empty
/// one nothing, a range read whole for its pages and none beside them, in
/// order, chunk by chunk.
#[test]
fn a_source_stream_caches_only_its_range() {
    let path = std::env::temp_dir().join(format!("magnitude-source-range-{}.bin", std::process::id()));
    let bytes = (0..8 * BLOCK).map(|byte| (byte / 4096) as u8).collect::<Vec<_>>();
    std::fs::write(&path, &bytes).unwrap();
    let written = std::fs::File::open(&path).unwrap();
    written.sync_all().unwrap();
    // SAFETY: advice on a descriptor this test owns.
    assert_eq!(unsafe { libc::posix_fadvise(written.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }, 0);
    let cached = |offset: u64, length: u64| seismic_io::cached_pages(&path, offset, length).unwrap();
    let source = magnitude_artifacts::FileSource::open(&path).unwrap();
    let want = seismic_io::Want::undated(0.);
    let mut chunks = Vec::new();
    source.stream::<std::io::Error>(BLOCK, 0, want, |at, chunk| Ok(chunks.push((at, chunk.to_vec())))).unwrap();
    assert!(chunks.is_empty());
    assert_eq!(cached(0, 8 * BLOCK).0, 0);
    let mut read = Vec::new();
    source
        .stream::<std::io::Error>(2 * BLOCK, BLOCK, want, |at, chunk| {
            assert_eq!(at, read.len() as u64, "chunks arrive in order");
            Ok(read.extend_from_slice(chunk))
        })
        .unwrap();
    assert_eq!(read, bytes[2 * BLOCK as usize..3 * BLOCK as usize]);
    let (held, pages) = cached(2 * BLOCK, BLOCK);
    assert_eq!(held, pages);
    assert_eq!(cached(0, 2 * BLOCK).0, 0);
    assert_eq!(cached(3 * BLOCK, 5 * BLOCK).0, 0);
    std::fs::remove_file(path).unwrap();
}
