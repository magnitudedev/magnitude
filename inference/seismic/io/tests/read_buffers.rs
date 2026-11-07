//! The read engine keeps no buffer past its last reader. Its own test
//! binary, so no other test's reader is open beside it.

use seismic_io::{CachePolicy, Reader, Want};

/// A cached piece is read through a buffer the engine keeps for the pieces
/// after it while the reader is open; closing the reader, the last one,
/// releases it.
#[test]
fn closing_the_last_reader_releases_the_read_buffers() {
    let path = std::env::temp_dir().join(format!("magnitude-read-buffers-{}.bin", std::process::id()));
    std::fs::write(&path, vec![7u8; 1 << 20]).unwrap();
    let reader = Reader::open(&path, CachePolicy::Cached).unwrap();
    let mut bytes = vec![0; 4096];
    reader.read_at([(1, &mut bytes[..])], Want::now()).unwrap();
    assert_eq!(bytes, vec![7; 4096]);
    assert!(seismic_io::read_buffers() > 0, "the piece's buffer is kept while its reader is open");
    drop(reader);
    assert_eq!(seismic_io::read_buffers(), 0);
    std::fs::remove_file(&path).unwrap();
}
