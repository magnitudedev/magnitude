//! The read engine's pool holds workers only for the reads its probes and
//! readers have in flight. Its own test binary, so no other test's reader
//! or probe holds workers beside it.

use seismic_io::{probe_volume, Buffer, CachePolicy, OpenFile, Range, Reader, Urgent, Want};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Wait up to ten seconds for the pool to hold at most `most` workers, the
/// idle ones above it leaving as they wake.
fn settles_to(most: usize) -> usize {
    let began = Instant::now();
    while seismic_io::read_workers() > most && began.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(10));
    }
    seismic_io::read_workers()
}

/// A probe's workers start and end with it, and a reader holds no more
/// workers than the pieces it had queued at once, none once it closes,
/// whether its requests were read or cancelled: a load whose probe ran
/// never keeps the probe's pool.
#[test]
fn the_pool_ends_with_the_probes_and_readers_it_served() {
    let path = std::env::temp_dir().join(format!("magnitude-read-workers-{}.bin", std::process::id()));
    std::fs::write(&path, vec![7u8; 1 << 20]).unwrap();
    let source = OpenFile::open(&path).unwrap();
    let urgent = Urgent { parts: vec![40_000, 24_000, 64_000], records: 2 };
    probe_volume(&source, Some(&urgent)).unwrap();
    assert_eq!(settles_to(0), 0, "the probe's workers leave once it ends");
    let reader = Reader::open(&path, CachePolicy::Cached).unwrap();
    let mut bytes = vec![0; 4096];
    reader.read_at([(1, &mut bytes[..])], Want::now()).unwrap();
    assert_eq!(bytes, vec![7; 4096]);
    assert!(settles_to(1) <= 1, "a reader that submitted one piece holds one worker at most");
    for _ in 0..4 {
        let range = Range { offset: 0, length: 1 << 20, into: Arc::new(Buffer::new(1 << 20)), at: 0 };
        // SAFETY: the sink is this request's alone, which nothing else
        // touches; a piece taken before the cancel reads on into it.
        let handle = unsafe { reader.request(vec![range], Want::undated(0.)) }.unwrap();
        drop(handle);
    }
    drop(reader);
    assert_eq!(settles_to(0), 0, "the reader's worker leaves once it closes");
    std::fs::remove_file(&path).unwrap();
}
