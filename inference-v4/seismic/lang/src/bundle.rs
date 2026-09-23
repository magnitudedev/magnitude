//! The checked-bundle boundary (spec §3.2).
//!
//! A bundle is the serialized form of one [`CheckedModule`], emitted by
//! `seismic-build` and embedded into generated bindings. Decoding validates
//! format/version, compiler semantic-version compatibility, content hash,
//! every arena reference, intrinsic signature ids, identity uniqueness, and
//! every invariant the wire schema does not guarantee. A corrupt or
//! incompatible bundle is a typed error; after decoding, an out-of-range
//! private arena id is a compiler panic (§13.3.2).
//!
//! Arena indices never cross this boundary: the wire form carries stable
//! content-derived identities and rebuilds arenas on decode.
//!
//! W1 owns the wire format.

use crate::checked::{check_source, CheckedModule, SourceFile, SourceSet};
use sha2::{Digest, Sha256};

/// Format version of the bundle wire schema. Bumped on any wire change.
pub const BUNDLE_FORMAT_VERSION: u32 = 3;

/// Semantic version of the checker whose output this crate can decode.
/// Bundles produced under a different semantic version are incompatible.
pub const COMPILER_SEMANTIC_VERSION: &str = "seismic-semantics-v8";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckedBundleError {
    Corrupt,
    IncompatibleVersion,
    HashMismatch,
}

impl std::fmt::Display for CheckedBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corrupt => f.write_str("checked bundle is corrupt"),
            Self::IncompatibleVersion => {
                f.write_str("checked bundle was produced by an incompatible compiler version")
            }
            Self::HashMismatch => f.write_str("checked bundle content hash does not match"),
        }
    }
}

impl std::error::Error for CheckedBundleError {}

/// Serializes a checked module. Deterministic: equal modules yield equal
/// bytes.
pub fn encode_checked_bundle(module: &CheckedModule) -> Vec<u8> {
    let mut bytes = internals::encode(module.internal());
    bytes.extend_from_slice(&(module.assets.len() as u32).to_le_bytes());
    for ((entry, backend), source) in &module.assets {
        internals::string(&mut bytes, &module.entries()[entry.index()].name);
        internals::string(&mut bytes, backend.as_str());
        internals::string(&mut bytes, source);
    }
    let digest = Sha256::digest(&bytes);
    bytes.extend_from_slice(&digest);
    bytes
}

/// The bundle-side constructor of a [`CheckedModule`].
pub fn decode_checked_bundle(bytes: &[u8]) -> Result<CheckedModule, CheckedBundleError> {
    if bytes.len() < 32 { return Err(CheckedBundleError::Corrupt); }
    let (payload, checksum) = bytes.split_at(bytes.len() - 32);
    if &Sha256::digest(payload)[..] != checksum { return Err(CheckedBundleError::HashMismatch); }
    internals::decode(payload)
}

mod internals {
    use super::*;
    use crate::checked::internals::Module;

    const MAGIC: &[u8; 8] = b"SEISBND6";

    pub(super) fn encode(module: &Module) -> Vec<u8> {
        let source_hash = source_hash(&module.sources);
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&BUNDLE_FORMAT_VERSION.to_le_bytes());
        string(&mut out, COMPILER_SEMANTIC_VERSION);
        string(&mut out, crate::registry::REGISTRY_REVISION);
        out.extend_from_slice(&source_hash);
        out.extend_from_slice(module.semantic_hash.digest());
        let count = u32::try_from(module.sources.files().len())
            .expect("checked module has more than u32::MAX source files");
        out.extend_from_slice(&count.to_le_bytes());
        for file in module.sources.files() {
            string(&mut out, &file.path);
            string(&mut out, &file.text);
        }
        out
    }

    pub(super) fn decode(bytes: &[u8]) -> Result<CheckedModule, CheckedBundleError> {
        let mut reader = Reader { bytes, offset: 0 };
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(CheckedBundleError::Corrupt);
        }
        if reader.u32()? != BUNDLE_FORMAT_VERSION {
            return Err(CheckedBundleError::IncompatibleVersion);
        }
        if reader.string()? != COMPILER_SEMANTIC_VERSION
            || reader.string()? != crate::registry::REGISTRY_REVISION
        {
            return Err(CheckedBundleError::IncompatibleVersion);
        }
        let expected_source_hash: [u8; 32] = reader
            .take(32)?
            .try_into()
            .map_err(|_| CheckedBundleError::Corrupt)?;
        let expected_semantic_hash: [u8; 32] = reader
            .take(32)?
            .try_into()
            .map_err(|_| CheckedBundleError::Corrupt)?;
        let count = usize::try_from(reader.u32()?).map_err(|_| CheckedBundleError::Corrupt)?;
        if count > reader.bytes.len().saturating_sub(reader.offset) / 8 {
            return Err(CheckedBundleError::Corrupt);
        }
        let mut files = Vec::with_capacity(count);
        for _ in 0..count {
            files.push(SourceFile {
                path: reader.string()?.to_owned(),
                text: reader.string()?.to_owned(),
            });
        }
        let asset_count = reader.u32()? as usize;
        if asset_count > bytes.len().saturating_sub(reader.offset) / 8 {
            return Err(CheckedBundleError::Corrupt);
        }
        let mut assets = std::collections::BTreeMap::new();
        for _ in 0..asset_count {
            let entry = reader.string()?.to_owned();
            let backend = crate::registry::BackendName::parse(reader.string()?)
                .ok_or(CheckedBundleError::Corrupt)?;
            let value = reader.string()?.to_owned();
            if assets.insert((entry, backend), value).is_some() { return Err(CheckedBundleError::Corrupt); }
        }
        if reader.offset != bytes.len() {
            return Err(CheckedBundleError::Corrupt);
        }
        let source_set = SourceSet::new(files);
        let canonical = source_set
            .clone()
            .canonicalized()
            .map_err(|_| CheckedBundleError::Corrupt)?;
        if canonical != source_set {
            return Err(CheckedBundleError::Corrupt);
        }
        if source_hash(&canonical) != expected_source_hash {
            return Err(CheckedBundleError::HashMismatch);
        }
        let mut rebuilt = check_source(canonical).map_err(|_| CheckedBundleError::Corrupt)?;
        if rebuilt.semantic_hash().digest() != &expected_semantic_hash {
            return Err(CheckedBundleError::HashMismatch);
        }
        for ((name, backend), source) in assets {
            let entry = rebuilt.entry_named(&name).ok_or(CheckedBundleError::Corrupt)?;
            rebuilt.capture_native_asset(entry, backend, source).map_err(|_| CheckedBundleError::Corrupt)?;
        }
        Ok(rebuilt)
    }

    fn source_hash(sources: &SourceSet) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(COMPILER_SEMANTIC_VERSION.as_bytes());
        hash.update(crate::registry::REGISTRY_REVISION.as_bytes());
        for file in sources.files() {
            hash.update((file.path.len() as u64).to_le_bytes());
            hash.update(file.path.as_bytes());
            hash.update((file.text.len() as u64).to_le_bytes());
            hash.update(file.text.as_bytes());
        }
        hash.finalize().into()
    }

    pub(super) fn string(out: &mut Vec<u8>, value: &str) {
        let length =
            u32::try_from(value.len()).expect("checked-bundle string exceeds u32::MAX bytes");
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    struct Reader<'a> {
        bytes: &'a [u8],
        offset: usize,
    }

    impl<'a> Reader<'a> {
        fn take(&mut self, length: usize) -> Result<&'a [u8], CheckedBundleError> {
            let end = self
                .offset
                .checked_add(length)
                .ok_or(CheckedBundleError::Corrupt)?;
            let value = self
                .bytes
                .get(self.offset..end)
                .ok_or(CheckedBundleError::Corrupt)?;
            self.offset = end;
            Ok(value)
        }

        fn u32(&mut self) -> Result<u32, CheckedBundleError> {
            Ok(u32::from_le_bytes(
                self.take(4)?
                    .try_into()
                    .map_err(|_| CheckedBundleError::Corrupt)?,
            ))
        }

        fn string(&mut self) -> Result<&'a str, CheckedBundleError> {
            let length = usize::try_from(self.u32()?).map_err(|_| CheckedBundleError::Corrupt)?;
            std::str::from_utf8(self.take(length)?).map_err(|_| CheckedBundleError::Corrupt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::BackendName;
    fn checked() -> CheckedModule {
        check_source(SourceSet::new(vec![SourceFile {
            path: "snapshot.seismic".into(),
            text: "fn twice(x: f32) -> f32:\n    return x + x\nnative twice for metal from \"twice.metal\":\n    threadgroups (1, 1, 1)\n    threads_per_threadgroup (1, 1, 1)\n".into(),
        }])).unwrap()
    }
    #[test]
    fn snapshot_assets_roundtrip_and_affect_identity() {
        let mut module=checked();
        let twice=module.entry_named("twice").unwrap();
        module.capture_native_asset(twice,BackendName::Metal,"kernel version one".into()).unwrap();
        let first=encode_checked_bundle(&module);
        let restored=decode_checked_bundle(&first).unwrap();
        let restored_twice=restored.entry_named("twice").unwrap();
        assert_eq!(restored.native_asset(restored_twice,BackendName::Metal),Some("kernel version one"));
        assert_eq!(restored.entries()[0].stable,module.entries()[0].stable);
        assert_eq!(restored.entries()[0].parameter_types,module.entries()[0].parameter_types);
        assert_eq!(encode_checked_bundle(&restored),first);
        module.capture_native_asset(twice,BackendName::Metal,"kernel version two".into()).unwrap();
        assert_ne!(first,encode_checked_bundle(&module));
    }
    #[test]
    fn corruption_and_incompatible_wire_fail_closed() {
        let module=checked();
        let mut encoded=encode_checked_bundle(&module);
        let end=encoded.len()-1;encoded[end]^=1;
        assert_eq!(decode_checked_bundle(&encoded).err(),Some(CheckedBundleError::HashMismatch));
        let mut encoded=encode_checked_bundle(&module);
        encoded[8..12].copy_from_slice(&0u32.to_le_bytes());
        let end=encoded.len()-32;
        let hash=Sha256::digest(&encoded[..end]);encoded[end..].copy_from_slice(&hash);
        assert_eq!(decode_checked_bundle(&encoded).err(),Some(CheckedBundleError::IncompatibleVersion));
        assert!(decode_checked_bundle(&[]).is_err());
    }

    #[test]
    fn previous_checker_semantic_version_cannot_reuse_a_bundle() {
        let mut encoded = encode_checked_bundle(&checked());
        let version_start = 8 + 4 + 4;
        let version_end = version_start + COMPILER_SEMANTIC_VERSION.len();
        encoded[version_start..version_end].copy_from_slice(b"seismic-semantics-v7");
        let checksum_start = encoded.len() - 32;
        let digest = Sha256::digest(&encoded[..checksum_start]);
        encoded[checksum_start..].copy_from_slice(&digest);
        assert_eq!(
            decode_checked_bundle(&encoded).err(),
            Some(CheckedBundleError::IncompatibleVersion)
        );
    }
}
