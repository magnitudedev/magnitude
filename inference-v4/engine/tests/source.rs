use seismic_engine::weights::source::FileSource;
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "seismic-source-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn independent_readers_checked_ranges_and_content_identity() {
    let temp = Temp::new();
    let path = temp.0.join("weights");
    fs::write(&path, b"abc").unwrap();
    let source = FileSource::open(path).unwrap();
    assert_eq!(source.size(), 3);
    assert_eq!(source.read(1, 2).unwrap(), b"bc");
    assert!(source.read(u64::MAX, 2).is_err());
    assert!(source.read(3, 1).is_err());
    assert_eq!(source.read(3, 0).unwrap(), b"");
    let hash = source
        .digest()
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    assert_eq!(
        hash,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let mut a = source.reader();
    let mut b = source.reader();
    a.seek(SeekFrom::Start(1)).unwrap();
    let mut bytes = Vec::new();
    a.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"bc");
    bytes.clear();
    b.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"abc");
    assert!(a.seek(SeekFrom::Start(5)).is_ok());
    assert_eq!(a.read(&mut [0; 1]).unwrap(), 0);
    assert!(b.seek(SeekFrom::Start(0)).is_ok());
    assert!(b.seek(SeekFrom::Current(-1)).is_err());
}
#[test]
fn renamed_path_does_not_change_open_snapshot() {
    let temp = Temp::new();
    let path = temp.0.join("weights");
    fs::write(&path, b"old").unwrap();
    let original = FileSource::open(&path).unwrap();
    fs::rename(&path, temp.0.join("old")).unwrap();
    fs::write(&path, b"new").unwrap();
    assert_eq!(original.read(0, 3).unwrap(), b"old");
    assert_eq!(FileSource::open(&path).unwrap().read(0, 3).unwrap(), b"new");
}
