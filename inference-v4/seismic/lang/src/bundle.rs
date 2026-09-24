//! The checked-bundle boundary (spec §3.2).
//!
//! A bundle is the serialized form of one [`CheckedModule`]. `seismic-build`
//! encodes it from the module its checker produced and embeds it into the
//! generated bindings. One wire format carries, in order:
//!
//! - a header: magic, format version, compiler semantic version, registry
//!   revision, source hash and semantic hash;
//! - the canonical source text, inert data behind `CheckedModule::sources`;
//! - the native asset snapshots, keyed by entry name and backend;
//! - the checked section: every field of the checked module the checker
//!   produced (entries, native implementations, definitions with their
//!   bodies, arenas, initialization contracts and loop facts, and families
//!   with their dimension plans), serialized with serde and `postcard`;
//!
//! followed by a SHA-256 checksum of everything before it.
//!
//! Two decoders read it, one per kind of input:
//!
//! - [`decode_checked_bundle`] reads a bundle embedded by the build. It
//!   validates the versions, the checksum and the structure of the checked
//!   section (every handle, arena index and body-local ordinal in range, every
//!   identity unique), and rebuilds the module under owners allocated fresh
//!   for this decode (`crate::wire`). It neither parses nor checks: the
//!   build's checker already did, in the same build as this decoder.
//! - [`check_bundle_sources`] reads a bundle that arrives from outside the
//!   binary (a `.seismicbundle` file). It checks the source section with
//!   [`check_source`] and ignores the checked section.
//!
//! A corrupt or incompatible bundle is a typed error on both paths; after
//! decoding, an out-of-range private arena id is a compiler panic (§13.3.2).
//!
//! W1 owns the wire format.

use crate::check::ir::{Definition, Family};
use crate::checked::internals::Module;
use crate::checked::{check_source, CheckedModule, EntryInfo, NativeImplementation, SourceFile, SourceSet};
use crate::registry::BackendName;
use crate::wire::{self, Scope};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Format version of the bundle wire schema. Bumped on any wire change.
pub const BUNDLE_FORMAT_VERSION: u32 = 4;

/// Semantic version of the checker whose output this crate can decode.
/// Bundles produced under a different semantic version are incompatible.
pub const COMPILER_SEMANTIC_VERSION: &str = "seismic-semantics-v8";

const MAGIC: &[u8; 8] = b"SEISBND6";

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
    let inner = module.internal();
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&BUNDLE_FORMAT_VERSION.to_le_bytes());
    string(&mut out, COMPILER_SEMANTIC_VERSION);
    string(&mut out, crate::registry::REGISTRY_REVISION);
    out.extend_from_slice(&source_hash(&inner.sources));
    out.extend_from_slice(inner.semantic_hash.digest());
    count(&mut out, inner.sources.files().len());
    for file in inner.sources.files() {
        string(&mut out, &file.path);
        string(&mut out, &file.text);
    }
    count(&mut out, module.assets.len());
    for ((entry, backend), source) in &module.assets {
        string(&mut out, &module.entries()[entry.index()].name);
        string(&mut out, backend.as_str());
        string(&mut out, source);
    }
    let checked = wire::encode(inner.id, inner.program, || {
        postcard::to_stdvec(&CheckedFields::of(inner))
    })
    .unwrap_or_else(|error| panic!("a checked module holds a foreign handle (§13.3.2): {error}"));
    out.extend_from_slice(&(checked.len() as u64).to_le_bytes());
    out.extend_from_slice(&checked);
    let digest = Sha256::digest(&out);
    out.extend_from_slice(&digest);
    out
}

/// The bundle-side constructor of a [`CheckedModule`], for a bundle the
/// build embedded: validates and decodes the checked section. No parsing or
/// checking happens.
pub fn decode_checked_bundle(bytes: &[u8]) -> Result<CheckedModule, CheckedBundleError> {
    let frame = Frame::read(verified(bytes)?)?;
    let (parts, decoded) = wire::decode(|| {
        postcard::from_bytes::<CheckedParts>(frame.checked)
            .map_err(|_| wire::WireError("checked section does not decode"))
    })
    .map_err(|_| CheckedBundleError::Corrupt)?;
    let files = frame.files.into_iter().map(|(path, text)| SourceFile { path, text }).collect();
    let module = parts
        .into_module(decoded, crate::ids::ModuleHash::new(frame.semantic_hash), SourceSet::new(files))
        .ok_or(CheckedBundleError::Corrupt)?;
    with_assets(CheckedModule::decoded(module), frame.assets)
}

/// Checks the source section of a bundle that arrives from outside the
/// binary, exactly as [`check_source`] checks source text, and ignores its
/// checked section. A source section that fails to check is `Corrupt`.
pub fn check_bundle_sources(bytes: &[u8]) -> Result<CheckedModule, CheckedBundleError> {
    let frame = Frame::read(verified(bytes)?)?;
    let files = frame.files.into_iter().map(|(path, text)| SourceFile { path, text }).collect();
    let sources = SourceSet::new(files);
    let canonical = sources.clone().canonicalized().map_err(|_| CheckedBundleError::Corrupt)?;
    if canonical != sources {
        return Err(CheckedBundleError::Corrupt);
    }
    if source_hash(&canonical) != frame.source_hash {
        return Err(CheckedBundleError::HashMismatch);
    }
    let checked = check_source(canonical).map_err(|_| CheckedBundleError::Corrupt)?;
    if checked.semantic_hash().digest() != &frame.semantic_hash {
        return Err(CheckedBundleError::HashMismatch);
    }
    with_assets(checked, frame.assets)
}

/// The payload of a bundle whose checksum trailer matches.
fn verified(bytes: &[u8]) -> Result<&[u8], CheckedBundleError> {
    let split = bytes.len().checked_sub(32).ok_or(CheckedBundleError::Corrupt)?;
    let (payload, checksum) = bytes.split_at(split);
    if Sha256::digest(payload)[..] != *checksum {
        return Err(CheckedBundleError::HashMismatch);
    }
    Ok(payload)
}

fn with_assets(
    mut module: CheckedModule,
    assets: Vec<(String, BackendName, String)>,
) -> Result<CheckedModule, CheckedBundleError> {
    for (name, backend, source) in assets {
        let entry = module.entry_named(&name).ok_or(CheckedBundleError::Corrupt)?;
        if module.native_asset(entry, backend).is_some() {
            return Err(CheckedBundleError::Corrupt);
        }
        module
            .capture_native_asset(entry, backend, source)
            .map_err(|_| CheckedBundleError::Corrupt)?;
    }
    Ok(module)
}

/// The checked section: every field of [`Module`] that the header and the
/// source section do not already carry.
#[derive(Serialize)]
struct CheckedFields<'a> {
    entries: &'a [EntryInfo],
    native_implementations: &'a [NativeImplementation],
    entry_families: &'a [usize],
    definitions: &'a [Definition],
    families: &'a [Family],
}

impl<'a> CheckedFields<'a> {
    fn of(module: &'a Module) -> Self {
        // Exhaustive: a new module field fails to compile here until the
        // wire format carries it.
        let Module {
            id: _,
            program: _,
            semantic_hash: _,
            sources: _,
            entries,
            native_implementations,
            entry_families,
            definitions,
            families,
        } = module;
        Self {
            entries,
            native_implementations,
            entry_families,
            definitions,
            families,
        }
    }
}

/// The owned form of [`CheckedFields`]. Each definition and family decodes
/// in its own [`Scope`], which [`CheckedParts::into_module`] checks.
#[derive(Deserialize)]
struct CheckedParts {
    entries: Vec<EntryInfo>,
    native_implementations: Vec<NativeImplementation>,
    entry_families: Vec<usize>,
    #[serde(deserialize_with = "wire::deserialize_scoped")]
    definitions: Vec<(Definition, Scope)>,
    #[serde(deserialize_with = "wire::deserialize_scoped")]
    families: Vec<(Family, Scope)>,
}

impl CheckedParts {
    /// Checks the cross-references the handle types cannot check themselves
    /// and assembles the module. `None` is a corrupt checked section.
    fn into_module(
        self,
        decoded: wire::Decoded,
        semantic_hash: crate::ids::ModuleHash,
        sources: SourceSet,
    ) -> Option<Module> {
        let CheckedParts {
            entries,
            native_implementations,
            entry_families,
            definitions,
            families,
        } = self;
        let within = |reached: u32, length: usize| reached as usize <= length;
        let valid = within(decoded.entries, entries.len())
            && within(decoded.functions, definitions.len())
            && within(decoded.families, families.len())
            && entries.iter().enumerate().all(|(ordinal, entry)| entry.id.index() == ordinal)
            && unique(entries.iter().map(|entry| entry.name.as_str()))
            && unique(entries.iter().map(|entry| entry.stable))
            && entry_families.len() == entries.len()
            && entry_families.iter().all(|family| *family < families.len())
            && unique(native_implementations.iter().map(|native| (native.entry, native.backend)));
        if !valid {
            return None;
        }
        let mut arenas = Vec::with_capacity(definitions.len());
        for (definition, scope) in &definitions {
            let arena = scope.defined?;
            let valid = scope.arena == Some(arena)
                && within(scope.locals, definition.body.locals.len())
                && within(scope.dimensions, definition.dimensions.len())
                && definition.file < sources.files().len();
            if !valid {
                return None;
            }
            arenas.push(arena);
        }
        if !unique(arenas.iter()) {
            return None;
        }
        for (family, scope) in &families {
            let contract = family.contract.index();
            let valid = scope.defined.is_none()
                && scope.locals == 0
                && scope.arena.is_none_or(|arena| arena == arenas[contract])
                && within(scope.dimensions, definitions[contract].0.dimensions.len());
            if !valid {
                return None;
            }
        }
        Some(Module {
            id: decoded.module,
            program: decoded.program,
            semantic_hash,
            sources,
            entries,
            native_implementations,
            entry_families,
            definitions: definitions.into_iter().map(|(definition, _)| definition).collect(),
            families: families.into_iter().map(|(family, _)| family).collect(),
        })
    }
}

fn unique<T: Eq + std::hash::Hash>(items: impl IntoIterator<Item = T>) -> bool {
    let mut seen = std::collections::HashSet::new();
    items.into_iter().all(|item| seen.insert(item))
}

/// The framed sections of a verified payload.
struct Frame<'a> {
    source_hash: [u8; 32],
    semantic_hash: [u8; 32],
    files: Vec<(String, String)>,
    assets: Vec<(String, BackendName, String)>,
    checked: &'a [u8],
}

impl<'a> Frame<'a> {
    fn read(payload: &'a [u8]) -> Result<Self, CheckedBundleError> {
        let mut reader = Reader {
            bytes: payload,
            offset: 0,
        };
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
        let source_hash = reader.digest()?;
        let semantic_hash = reader.digest()?;
        let files = reader.counted(|reader| Ok((reader.string()?.to_owned(), reader.string()?.to_owned())))?;
        let assets = reader.counted(|reader| {
            let entry = reader.string()?.to_owned();
            let backend = BackendName::parse(reader.string()?).ok_or(CheckedBundleError::Corrupt)?;
            Ok((entry, backend, reader.string()?.to_owned()))
        })?;
        let length = usize::try_from(reader.u64()?).map_err(|_| CheckedBundleError::Corrupt)?;
        let checked = reader.take(length)?;
        if reader.offset != payload.len() {
            return Err(CheckedBundleError::Corrupt);
        }
        Ok(Self {
            source_hash,
            semantic_hash,
            files,
            assets,
            checked,
        })
    }
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

fn count(out: &mut Vec<u8>, count: usize) {
    let count = u32::try_from(count).expect("checked-bundle section has more than u32::MAX items");
    out.extend_from_slice(&count.to_le_bytes());
}

fn string(out: &mut Vec<u8>, value: &str) {
    count(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], CheckedBundleError> {
        let end = self.offset.checked_add(length).ok_or(CheckedBundleError::Corrupt)?;
        let value = self.bytes.get(self.offset..end).ok_or(CheckedBundleError::Corrupt)?;
        self.offset = end;
        Ok(value)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], CheckedBundleError> {
        self.take(N)?.try_into().map_err(|_| CheckedBundleError::Corrupt)
    }

    fn u32(&mut self) -> Result<u32, CheckedBundleError> {
        self.array().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Result<u64, CheckedBundleError> {
        self.array().map(u64::from_le_bytes)
    }

    fn digest(&mut self) -> Result<[u8; 32], CheckedBundleError> {
        self.array()
    }

    fn string(&mut self) -> Result<&'a str, CheckedBundleError> {
        let length = usize::try_from(self.u32()?).map_err(|_| CheckedBundleError::Corrupt)?;
        std::str::from_utf8(self.take(length)?).map_err(|_| CheckedBundleError::Corrupt)
    }

    /// A `u32` count followed by that many items. Every item occupies at
    /// least four bytes, which bounds the count before allocating.
    fn counted<T>(
        &mut self,
        mut item: impl FnMut(&mut Self) -> Result<T, CheckedBundleError>,
    ) -> Result<Vec<T>, CheckedBundleError> {
        let count = usize::try_from(self.u32()?).map_err(|_| CheckedBundleError::Corrupt)?;
        if count > (self.bytes.len() - self.offset) / 4 {
            return Err(CheckedBundleError::Corrupt);
        }
        (0..count).map(|_| item(self)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::ElementBindings;

    const SOURCE: &str = "fn twice(x: f32) -> f32:\n    return x + x\n\nfn scale[N](x: &tensor[N] f32, factor: f32, output: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        output[i] = twice(x[i]) * factor\n\nfn prefix[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    let mut total = 0.0\n    for i in 0..N:\n        if x[i] > 0.0:\n            total = total + x[i]\n        output[i] = total\n    return output\n\nnative twice for metal from \"twice.metal\":\n    launch twice:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n";

    fn checked() -> CheckedModule {
        let mut module = check_source(SourceSet::new(vec![SourceFile {
            path: "snapshot.seismic".into(),
            text: SOURCE.into(),
        }]))
        .unwrap();
        let twice = module.entry_named("twice").unwrap();
        module
            .capture_native_asset(twice, BackendName::Metal, "kernel version one".into())
            .unwrap();
        module
    }

    /// Replaces the checksum trailer so only the payload decides the outcome.
    fn resealed(mut bytes: Vec<u8>) -> Vec<u8> {
        let payload = bytes.len() - 32;
        let digest = Sha256::digest(&bytes[..payload]);
        bytes[payload..].copy_from_slice(&digest);
        bytes
    }

    #[test]
    fn decoding_rebuilds_the_checked_module_under_fresh_owners() {
        let module = checked();
        let encoded = encode_checked_bundle(&module);
        let first = decode_checked_bundle(&encoded).unwrap();
        let second = decode_checked_bundle(&encoded).unwrap();
        assert_eq!(encode_checked_bundle(&first), encoded);
        assert_eq!(first.semantic_hash(), module.semantic_hash());
        assert_eq!(first.sources(), module.sources());
        assert_eq!(first.entries().len(), module.entries().len());
        for (decoded, original) in first.entries().iter().zip(module.entries()) {
            assert_eq!(decoded.stable, original.stable);
            assert_eq!(decoded.parameter_types, original.parameter_types);
            assert_eq!(decoded.element_domain, original.element_domain);
        }
        // Two decodes allocate distinct owners, like two checks of one source.
        assert_ne!(first.entries()[0].id, second.entries()[0].id);
        let twice = first.entry_named("twice").unwrap();
        assert_eq!(first.native_asset(twice, BackendName::Metal), Some("kernel version one"));
        for info in first.entries() {
            first
                .entry(info.id, &ElementBindings::default())
                .expect("a decoded entry builds its logical entry");
        }
    }

    #[test]
    fn external_bundles_check_their_sources() {
        let module = checked();
        let encoded = encode_checked_bundle(&module);
        let checked = check_bundle_sources(&encoded).unwrap();
        assert_eq!(encode_checked_bundle(&checked), encoded);
        // A source section that no longer checks is rejected even though the
        // checked section still decodes.
        let broken = SOURCE.replacen("return x + x", "return x + y", 1);
        let mut bytes = encoded.clone();
        let at = bytes
            .windows(SOURCE.len())
            .position(|window| window == SOURCE.as_bytes())
            .unwrap();
        bytes[at..at + broken.len()].copy_from_slice(broken.as_bytes());
        assert_eq!(
            check_bundle_sources(&resealed(bytes.clone())).err(),
            Some(CheckedBundleError::HashMismatch),
            "the header's source hash no longer matches"
        );
        let hash_at = 8 + 4 + 4 + COMPILER_SEMANTIC_VERSION.len() + 4 + crate::registry::REGISTRY_REVISION.len();
        let broken_sources = SourceSet::new(vec![SourceFile {
            path: "snapshot.seismic".into(),
            text: broken,
        }]);
        bytes[hash_at..hash_at + 32].copy_from_slice(&source_hash(&broken_sources));
        let bytes = resealed(bytes);
        assert!(decode_checked_bundle(&bytes).is_ok());
        assert_eq!(check_bundle_sources(&bytes).err(), Some(CheckedBundleError::Corrupt));
    }

    #[test]
    fn assets_affect_the_bytes() {
        let mut module = checked();
        let first = encode_checked_bundle(&module);
        let twice = module.entry_named("twice").unwrap();
        module
            .capture_native_asset(twice, BackendName::Metal, "kernel version two".into())
            .unwrap();
        assert_ne!(first, encode_checked_bundle(&module));
    }

    #[test]
    fn corruption_and_incompatible_wire_fail_closed() {
        let encoded = encode_checked_bundle(&checked());
        assert_eq!(decode_checked_bundle(&[]).err(), Some(CheckedBundleError::Corrupt));
        assert_eq!(decode_checked_bundle(&encoded[..31]).err(), Some(CheckedBundleError::Corrupt));
        let truncated = resealed(encoded[..encoded.len() - 40].to_vec());
        assert_eq!(decode_checked_bundle(&truncated).err(), Some(CheckedBundleError::Corrupt));
        let mut flipped = encoded.clone();
        flipped[encoded.len() / 2] ^= 1;
        assert_eq!(decode_checked_bundle(&flipped).err(), Some(CheckedBundleError::HashMismatch));
        let mut version = encoded.clone();
        version[8..12].copy_from_slice(&3u32.to_le_bytes());
        assert_eq!(
            decode_checked_bundle(&resealed(version)).err(),
            Some(CheckedBundleError::IncompatibleVersion)
        );
    }

    #[test]
    fn previous_checker_semantic_version_cannot_reuse_a_bundle() {
        let mut encoded = encode_checked_bundle(&checked());
        let version_start = 8 + 4 + 4;
        let version_end = version_start + COMPILER_SEMANTIC_VERSION.len();
        encoded[version_start..version_end].copy_from_slice(b"seismic-semantics-v7");
        let encoded = resealed(encoded);
        assert_eq!(
            decode_checked_bundle(&encoded).err(),
            Some(CheckedBundleError::IncompatibleVersion)
        );
    }

    /// Every single-byte change to the checked section, resealed, either
    /// still decodes (a changed span, flag or constant is structurally
    /// valid) or fails as `Corrupt`; none panics.
    #[test]
    fn damaged_checked_sections_fail_closed() {
        let encoded = encode_checked_bundle(&checked());
        let payload = encoded.len() - 32;
        let frame = Frame::read(&encoded[..payload]).unwrap();
        let start = payload - frame.checked.len();
        let mut rejected = 0;
        for position in start..payload {
            for change in [0x01, 0x80, 0xff] {
                let mut bytes = encoded.clone();
                bytes[position] ^= change;
                let bytes = resealed(bytes);
                let outcome = std::panic::catch_unwind(|| decode_checked_bundle(&bytes));
                match outcome.unwrap_or_else(|_| panic!("byte {position} ^ {change:#x} panics")) {
                    Ok(_) => {}
                    Err(error) => {
                        assert_eq!(error, CheckedBundleError::Corrupt);
                        rejected += 1;
                    }
                }
            }
        }
        assert!(rejected > 0);
    }
}
