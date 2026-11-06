//! A source's range asks the file cache for exactly itself.

#![cfg(target_os = "linux")]

use std::io::Read as _;
use std::os::fd::AsRawFd;

const BLOCK: u64 = 128 * 1024;

/// A source's range reads exactly itself into the cache: an empty one
/// nothing (advice of length zero would mean the rest of the file), a
/// range read whole for its pages and none beside them.
#[test]
fn a_source_range_caches_only_itself() {
    let path = std::env::temp_dir().join(format!("magnitude-source-range-{}.bin", std::process::id()));
    std::fs::write(&path, vec![7u8; 8 * BLOCK as usize]).unwrap();
    let written = std::fs::File::open(&path).unwrap();
    written.sync_all().unwrap();
    // SAFETY: advice on a descriptor this test owns.
    assert_eq!(unsafe { libc::posix_fadvise(written.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }, 0);
    let cached = |offset: u64, length: u64| seismic_io::cached_pages(&path, offset, length).unwrap();
    let source = magnitude_artifacts::FileSource::open(&path).unwrap();
    let mut empty = source.range(BLOCK, 0).unwrap();
    assert_eq!(empty.read(&mut [0u8; 1]).unwrap(), 0);
    assert_eq!(cached(0, 8 * BLOCK).0, 0);
    let mut bytes = Vec::new();
    source.range(2 * BLOCK, BLOCK).unwrap().read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, vec![7u8; BLOCK as usize]);
    let (held, pages) = cached(2 * BLOCK, BLOCK);
    assert_eq!(held, pages);
    assert_eq!(cached(0, 2 * BLOCK).0, 0);
    assert_eq!(cached(3 * BLOCK, 5 * BLOCK).0, 0);
    std::fs::remove_file(path).unwrap();
}
