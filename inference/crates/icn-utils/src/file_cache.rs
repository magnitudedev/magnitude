//! No-fail recovery and best-effort publication for recomputable JSON files.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use fs2::FileExt;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Reads a bounded JSON object. Every filesystem, size, and decoding failure is a cache miss.
pub fn read_object(path: &Path, maximum_bytes: usize) -> Option<Map<String, Value>> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if !metadata.file_type().is_file() || metadata.len() > maximum_bytes as u64 {
        return None;
    }
    let file = fs::File::open(path).ok()?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(maximum_bytes.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > maximum_bytes {
        return None;
    }
    match serde_json::from_slice(&bytes).ok()? {
        Value::Object(object) => Some(object),
        _ => None,
    }
}

/// Decodes independently meaningful map entries and discards only entries that do not decode.
pub fn recover_map<T: DeserializeOwned>(
    value: Option<Value>,
    maximum_entries: usize,
) -> BTreeMap<String, T> {
    let Some(Value::Object(entries)) = value else {
        return BTreeMap::new();
    };
    entries
        .into_iter()
        .take(maximum_entries)
        .filter_map(|(key, value)| serde_json::from_value(value).ok().map(|value| (key, value)))
        .collect()
}

/// Decodes array elements independently up to a caller-provided resource bound.
pub fn recover_array<T: DeserializeOwned>(value: Option<Value>, maximum_entries: usize) -> Vec<T> {
    let Some(Value::Array(entries)) = value else {
        return Vec::new();
    };
    entries
        .into_iter()
        .take(maximum_entries)
        .filter_map(|value| serde_json::from_value(value).ok())
        .collect()
}

/// Publishes a complete replacement atomically when possible. Failure is intentionally invisible.
pub fn write_json_atomic<T: Serialize>(
    path: &Path,
    lock_path: &Path,
    value: &T,
    maximum_bytes: usize,
) {
    let Ok(lock) = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
    else {
        return;
    };
    if lock.try_lock_exclusive().is_err() {
        return;
    }
    let Some(bytes) = serde_json::to_vec_pretty(value)
        .ok()
        .filter(|bytes| bytes.len() <= maximum_bytes)
    else {
        let _ = lock.unlock();
        return;
    };
    let Some(parent) = path.parent() else {
        let _ = lock.unlock();
        return;
    };
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".cache-tmp-{}-{sequence}", std::process::id()));
    let persisted = write_private_file(&temporary, &bytes) && fs::rename(&temporary, path).is_ok();
    if !persisted {
        let _ = fs::remove_file(&temporary);
    }
    let _ = lock.unlock();
}

fn write_private_file(path: &Path, bytes: &[u8]) -> bool {
    let mut options = fs::OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let Ok(mut file) = options.open(path) else {
        return false;
    };
    file.write_all(bytes).is_ok() && file.sync_all().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_files_are_misses_and_valid_map_siblings_survive() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.json");
        assert!(read_object(&path, 1024).is_none());
        fs::write(&path, b"not json").unwrap();
        assert!(read_object(&path, 1024).is_none());
        fs::write(&path, b"[]").unwrap();
        assert!(read_object(&path, 1024).is_none());
        fs::write(&path, vec![b'x'; 1025]).unwrap();
        assert!(read_object(&path, 1024).is_none());

        let recovered = recover_map::<u64>(
            Some(serde_json::json!({
                "one": 1,
                "broken": "no",
                "two": 2
            })),
            3,
        );
        assert_eq!(
            recovered,
            BTreeMap::from([("one".to_owned(), 1), ("two".to_owned(), 2)])
        );
        assert_eq!(
            recover_array::<u64>(Some(serde_json::json!([1, "no", 2, 3])), 3),
            vec![1, 2]
        );
    }

    #[test]
    fn atomic_write_is_recoverable_and_failures_are_invisible() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.json");
        let lock = directory.path().join("cache.lock");
        write_json_atomic(&path, &lock, &serde_json::json!({"value": 1}), 1024);
        assert_eq!(read_object(&path, 1024).unwrap()["value"], 1);

        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        write_json_atomic(&path, &lock, &serde_json::json!({"value": 2}), 1024);
        assert!(path.is_dir());
    }
}
