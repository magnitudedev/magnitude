//! No-fail recovery and best-effort publication for recomputable JSON files.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use fs2::FileExt;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Reads a bounded regular file. Every filesystem and size failure is a cache miss.
pub fn read_bytes(path: &Path, maximum_bytes: usize) -> Option<Vec<u8>> {
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
    Some(bytes)
}

/// Reads and decodes one bounded JSON recovery unit.
pub fn read_json<T: DeserializeOwned>(path: &Path, maximum_bytes: usize) -> Option<T> {
    serde_json::from_slice(&read_bytes(path, maximum_bytes)?).ok()
}

/// Reads a bounded JSON object. Every filesystem, size, and decoding failure is a cache miss.
pub fn read_object(path: &Path, maximum_bytes: usize) -> Option<Map<String, Value>> {
    match read_json::<Value>(path, maximum_bytes)? {
        Value::Object(object) => Some(object),
        _ => None,
    }
}

/// Removes and decodes one independently recoverable object section.
pub fn recover_section<T: DeserializeOwned>(
    object: &mut Map<String, Value>,
    key: &str,
) -> Option<T> {
    serde_json::from_value(object.remove(key)?).ok()
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
    let Some(bytes) = serde_json::to_vec_pretty(value)
        .ok()
        .filter(|bytes| bytes.len() <= maximum_bytes)
    else {
        return;
    };
    write_bytes_atomic(path, lock_path, &bytes, maximum_bytes);
}

/// Publishes bounded bytes as a complete replacement. Failure is intentionally invisible.
pub fn write_bytes_atomic(path: &Path, lock_path: &Path, bytes: &[u8], maximum_bytes: usize) {
    if bytes.len() > maximum_bytes {
        return;
    }
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return;
    };
    let Some(lock_parent) = lock_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return;
    };
    if create_private_directories(parent).is_err()
        || create_private_directories(lock_parent).is_err()
    {
        return;
    }
    match fs::symlink_metadata(lock_path) {
        Ok(metadata) if !metadata.file_type().is_file() => return,
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return,
    }
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
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = temporary_path(parent, path, sequence);
    let persisted = write_private_file(&temporary, bytes) && fs::rename(&temporary, path).is_ok();
    if !persisted {
        let _ = fs::remove_file(&temporary);
    }
    let _ = lock.unlock();
}

fn temporary_path(parent: &Path, path: &Path, sequence: u64) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    parent.join(format!(".{name}.tmp-{}-{sequence}", std::process::id()))
}

fn create_private_directories(path: &Path) -> std::io::Result<()> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
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

        fs::write(&path, br#"{"value": 7}"#).unwrap();
        assert_eq!(
            read_json::<serde_json::Value>(&path, 1024).unwrap()["value"],
            7
        );
        assert_eq!(read_bytes(&path, 1024).unwrap(), br#"{"value": 7}"#);

        let mut object = serde_json::json!({"valid": 2, "invalid": "no"})
            .as_object()
            .unwrap()
            .clone();
        assert_eq!(recover_section::<u64>(&mut object, "valid"), Some(2));
        assert_eq!(recover_section::<u64>(&mut object, "invalid"), None);
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
        let path = directory.path().join("nested/cache.json");
        let lock = directory.path().join("locks/cache.lock");
        write_json_atomic(&path, &lock, &serde_json::json!({"value": 1}), 1024);
        assert_eq!(read_object(&path, 1024).unwrap()["value"], 1);

        let blob = directory.path().join("blobs/value");
        write_bytes_atomic(&blob, &lock, b"cached bytes", 1024);
        assert_eq!(read_bytes(&blob, 1024).unwrap(), b"cached bytes");

        write_bytes_atomic(&blob, &lock, b"too large", 2);
        assert_eq!(read_bytes(&blob, 1024).unwrap(), b"cached bytes");

        fs::remove_file(&lock).unwrap();
        fs::create_dir(&lock).unwrap();
        write_bytes_atomic(&blob, &lock, b"must not publish", 1024);
        assert_eq!(read_bytes(&blob, 1024).unwrap(), b"cached bytes");

        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        write_json_atomic(&path, &lock, &serde_json::json!({"value": 2}), 1024);
        assert!(path.is_dir());
    }
}
