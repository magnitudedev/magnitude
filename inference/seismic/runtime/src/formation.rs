//! The program former: the one path from rendered source to an executable
//! program on an opened device.
//!
//! Callers render [`ProgramSource`]s; the former keys each by its toolchain
//! and text and serves each requested entry from a compile of that text
//! already live on the device. Entries no live compile holds are compiled
//! together, in one call of the toolchain and never with the entries that
//! are held: a call has a cost of its own and one for each entry it forms,
//! so what is held is not formed again and what is missing shares a call.
//! Variants of a templated kernel share one text, so a text's entries come
//! from as many compiles as the calls that asked for them.
//!
//! A compile lives while a caller holds it. What a caller holds of a formed
//! source is only the compiles its entries are in ([`Formed::entry`]).
//!
//! The former is its toolchain's one caller outside tests for the programs
//! a process serves ([`Toolchain`]). The store keeps each compile's artifact under
//! the text and the entry list compiled, and names the artifact that holds
//! each entry: a later request in any process is loaded from the artifacts
//! that hold its entries, whichever calls formed them. Only what no
//! artifact holds is compiled, once, and stored. Loading compiles nothing.
//! A compile's program is used at once; its artifact is made and stored
//! on the store's thread, off the path that formed it, and any former of
//! the store that wants it meanwhile waits for it and loads it.
//!
//! The store is a cache and may drop any key: an entry no pointer names,
//! or whose named artifact is gone, is classed as compiled and compiled
//! again. Each pointer names its artifact's digest, which a load checks
//! before any toolchain sees the bytes ([`Pointer`]), and bytes were
//! written here under a key that covers their encoding
//! ([`Artifact::ENCODING`]): a pointer that does not parse, an artifact
//! that does not match its digest, or one that does not decode is damaged.
//! Each is reported, missed, and rewritten by the compile it causes. A
//! toolchain may decline an artifact whole and undamaged
//! ([`Declined`](seismic_native_target::Declined)):
//! its platform would have to compile to load it (Vulkan's pipeline cache
//! is a hint a driver may decline, Metal's archive one its OS may), its
//! driver refuses it (a CUBIN after a driver change), or it lacks an entry
//! asked of it. A declined load is a miss too, counted and reported with
//! its reason, and its entries are compiled with the rest and stored again;
//! no load fails a formation.

use crate::artifacts::{ArtifactKey, ArtifactStore, Claim, ProgramStore, Stores};
use crate::native::Linear;
use seismic_native_target::workers::Workers;
use seismic_native_target::{
    Artifact, Compile, Formed, NativeCompilationError, ProgramEntry, ProgramSource, Programs, Toolchain,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// The programs a device's toolchain has formed (compiled, or loaded from
/// its artifact store) and the seconds its calls took, summed over the
/// threads that formed them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FormedPrograms {
    pub programs: usize,
    pub seconds: f64,
    /// Of them, the ones the toolchain compiled: nothing live held them
    /// and no stored artifact served them.
    pub compiles: u64,
    /// The stored artifacts that did not load: damaged bytes, or ones the
    /// toolchain's platform would not serve without compiling. Their
    /// entries were compiled again, and are counted in `compiles`.
    pub load_misses: u64,
}

/// What several toolchains of one device formed, together.
impl std::iter::Sum for FormedPrograms {
    fn sum<I: Iterator<Item = Self>>(formed: I) -> Self {
        formed.fold(Self::default(), |sum, formed| Self {
            programs: sum.programs + formed.programs,
            seconds: sum.seconds + formed.seconds,
            compiles: sum.compiles + formed.compiles,
            load_misses: sum.load_misses + formed.load_misses,
        })
    }
}

/// What forming a program does, known before it is formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FormationClass {
    /// No compiler of the toolchain's runs: compiles live on the device
    /// hold every requested entry, or the store keeps an artifact that
    /// serves those they lack, unless the platform declines to load it
    /// without compiling ([`FormedPrograms::load_misses`]).
    Kept,
    /// The toolchain compiles the entries no live compile holds.
    Compiled,
}

/// `call` on a thread of the former's ([`seismic_threads::run`]) while the
/// caller waits. A call hands its program to the platform's compiler (NVRTC,
/// or a Vulkan or Metal driver's shader compiler), whose recursion depth
/// follows the program and the driver, neither of which the caller chose,
/// so no toolchain call runs on its caller's stack, whoever started that
/// thread. A thread is started for each call, which costs microseconds
/// beside a compile or a platform load, and the callers that wait bound how
/// many run at once.
fn on_compiler_stack<R: Send>(call: impl FnOnce() -> R + Send) -> R {
    seismic_threads::run(COMPILER_THREAD, call)
}

/// The name of every thread a toolchain call runs on.
const COMPILER_THREAD: &str = "seismic-compile";

pub(crate) struct ProgramFormer<T: Toolchain> {
    /// Shared with the stores its compiles leave ([`ProgramFormer::compile_stored`]).
    toolchain: Arc<T>,
    store: Arc<ProgramStore>,
    /// Compiles live on the device, by the key of their text. Weak: a
    /// compile lives as long as some caller holds it.
    live: Mutex<HashMap<ArtifactKey, Vec<Weak<Compile<T::Program>>>>>,
    /// Compiles formed by the toolchain (compiled, or loaded from the
    /// store) rather than found live, and the nanoseconds its calls took.
    formed: std::sync::atomic::AtomicUsize,
    forming_nanos: std::sync::atomic::AtomicU64,
    /// What the store was read to keep, by the key of an entry: the
    /// artifact that holds it ([`ProgramFormer::kept`]). A class is read
    /// whenever a price is asked, and these reads would otherwise read the
    /// store each time. A read, held or missed, holds until a store of
    /// that entry ends ([`ProgramStore::stored`]), for other devices of the
    /// store form beside this one, or until a load of what it names misses
    /// ([`ProgramFormer::forget`]).
    stored: Mutex<Stored>,
    /// The stores its compiles began, which it waits for when it drops.
    stores: Arc<Stores>,
    /// The compiles the toolchain ran: entries no live compile held and no
    /// stored artifact served.
    compiles: std::sync::atomic::AtomicU64,
    /// The stored artifacts that did not load ([`FormedPrograms::load_misses`]).
    load_misses: std::sync::atomic::AtomicU64,
    /// What this toolchain's compiles have taken in this process, each with
    /// the entries it formed: what a compile of a source not yet compiled
    /// here is priced by.
    compiled: Mutex<Linear>,
    /// The threads its owner lent for work off the calling thread
    /// ([`ProgramFormer::staff`]); without them every source is formed in
    /// turn on the thread that asks.
    workers: Mutex<Option<Arc<Workers>>>,
}

impl<T: Toolchain> Drop for ProgramFormer<T> {
    /// A former's stores end with it, and no other's: a later process loads
    /// every program it compiled.
    fn drop(&mut self) {
        self.stores.settle();
    }
}

/// What a former read the store to keep ([`ProgramFormer::stored`]), by the
/// key of an entry ([`ProgramFormer::entry_key`]), with the stores of that
/// entry ended when it was read ([`ProgramStore::stored`]).
type Stored = HashMap<ArtifactKey, (Option<Pointer>, u64)>;

/// What an entry's stored pointer names: the artifact that holds the entry,
/// and the digest of that artifact's bytes as they were stored. A load
/// checks the bytes against it before any toolchain sees them, for every
/// backend: a driver may take damaged bytes as code.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pointer {
    artifact: ArtifactKey,
    digest: ArtifactKey,
}

impl Pointer {
    /// The pointer to the artifact `bytes` of a compile of the text keyed
    /// `key` with the entry list `list`. The artifact's key names its bytes
    /// too, so every compile is stored as a file of its own that no other
    /// compile writes: processes compiling the same entries at once each
    /// write their own artifact and then their pointers, and whichever
    /// pointer is read names a whole artifact its digest matches.
    fn of(key: &ArtifactKey, list: &str, bytes: &[u8]) -> Self {
        let digest = ArtifactKey::of(&[bytes]);
        let artifact = ArtifactKey::of(&[
            b"artifact",
            key.as_str().as_bytes(),
            list.as_bytes(),
            digest.as_str().as_bytes(),
        ]);
        Self { artifact, digest }
    }

    /// The pointer as stored: the artifact's key and its digest.
    fn text(&self) -> String {
        format!("{} {}", self.artifact.as_str(), self.digest.as_str())
    }

    /// The pointer [`Pointer::text`] wrote; none for any other text.
    fn parse(text: &str) -> Option<Self> {
        let (artifact, digest) = text.split_once(' ')?;
        Some(Self { artifact: ArtifactKey::parse(artifact)?, digest: ArtifactKey::parse(digest)? })
    }

    /// Whether `bytes` are the artifact's bytes as they were stored.
    fn holds(&self, bytes: &[u8]) -> bool {
        ArtifactKey::of(&[bytes]) == self.digest
    }
}

/// A source as the former reads it: with the key of its text. Keying
/// hashes the whole text, and a source tuning prices is asked about again
/// at every price, so it is keyed once
/// ([`ProgramFormer::keyed`]); a hash of every rendered source at every
/// price took 0.17 s of each of the allocator's passes (a GB10 load,
/// 2026-10-12).
#[derive(Clone)]
pub(crate) struct Keyed {
    source: ProgramSource,
    key: ArtifactKey,
}

/// One line naming an entry list: its entries, tab-separated, each with
/// its group size and constants.
fn listed(entries: &[ProgramEntry]) -> String {
    entries
        .iter()
        .map(|entry| format!("{}|{:?}|{:?}", entry.symbol, entry.group_size, entry.constants))
        .collect::<Vec<_>>()
        .join("\t")
}

impl<T: Toolchain> ProgramFormer<T> {
    pub(crate) fn new(toolchain: T, store: Arc<ProgramStore>) -> Self {
        Self {
            toolchain: Arc::new(toolchain),
            store,
            live: Mutex::new(HashMap::new()),
            formed: std::sync::atomic::AtomicUsize::new(0),
            forming_nanos: std::sync::atomic::AtomicU64::new(0),
            stored: Mutex::default(),
            stores: Arc::default(),
            compiles: std::sync::atomic::AtomicU64::new(0),
            load_misses: std::sync::atomic::AtomicU64::new(0),
            compiled: Mutex::new(Linear::default()),
            workers: Mutex::new(None),
        }
    }

    /// Lend the former `workers` for what it forms off the calling thread,
    /// or take them back (none). Whoever owns a span of formation owns its
    /// concurrency: the former forms on no thread of its own beyond the one
    /// each toolchain call runs on while its caller waits
    /// ([`on_compiler_stack`]). Only storing a compile's artifact runs
    /// apart, on the store's one thread, so no formation waits for it.
    pub(crate) fn staff(&self, workers: Option<Arc<Workers>>) {
        *self.workers.lock().expect("worker slot lock is never poisoned") = workers;
    }

    /// The threads lent to the former, if any.
    pub(crate) fn workers(&self) -> Option<Arc<Workers>> {
        self.workers
            .lock()
            .expect("worker slot lock is never poisoned")
            .clone()
    }

    /// What this toolchain's compiles have taken, by the entries formed.
    pub(crate) fn compiled(&self) -> Linear {
        *self
            .compiled
            .lock()
            .expect("compile observation lock is never poisoned")
    }

    /// The compiles the toolchain has formed for this device, and the time
    /// its calls took.
    pub(crate) fn programs_formed(&self) -> FormedPrograms {
        FormedPrograms {
            programs: self.formed.load(std::sync::atomic::Ordering::Relaxed),
            seconds: std::time::Duration::from_nanos(self.forming_nanos.load(std::sync::atomic::Ordering::Relaxed))
                .as_secs_f64(),
            compiles: self.compiles.load(std::sync::atomic::Ordering::Relaxed),
            load_misses: self.load_misses.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    pub(crate) fn toolchain(&self) -> &T {
        &self.toolchain
    }

    /// The content address of `source`'s text under this toolchain and the
    /// encoding of its artifacts.
    fn key(&self, source: &ProgramSource) -> ArtifactKey {
        let identity = self.toolchain.identity();
        ArtifactKey::of(&[
            identity.namespace.as_bytes(),
            identity.material.as_bytes(),
            identity.stored.as_bytes(),
            &T::Artifact::ENCODING.to_le_bytes(),
            source.text.as_bytes(),
        ])
    }

    /// Where the store keeps which artifact holds `entry` of the text keyed
    /// `key`: each entry is kept apart, so any set of entries a request
    /// names is served from the artifacts that hold them.
    fn entry_key(key: &ArtifactKey, entry: &ProgramEntry) -> ArtifactKey {
        ArtifactKey::of(&[
            b"entry",
            key.as_str().as_bytes(),
            listed(std::slice::from_ref(entry)).as_bytes(),
        ])
    }

    /// The live compiles of the text keyed `key` that hold entries of
    /// `entries`, and the entries none of them holds.
    fn held(
        &self,
        key: &ArtifactKey,
        entries: &[ProgramEntry],
    ) -> (Vec<Arc<Compile<T::Program>>>, Vec<ProgramEntry>) {
        let live = self
            .live
            .lock()
            .expect("live program lock is never poisoned")
            .get(key)
            .map(|compiles| compiles.iter().filter_map(Weak::upgrade).collect::<Vec<_>>())
            .unwrap_or_default();
        let mut holding = Vec::<Arc<Compile<T::Program>>>::new();
        let mut missing = Vec::new();
        for entry in entries {
            if holding.iter().any(|compile| compile.entries().contains(entry)) {
                continue;
            }
            match live.iter().find(|compile| compile.entries().contains(entry)) {
                Some(compile) => holding.push(compile.clone()),
                None if !missing.contains(entry) => missing.push(entry.clone()),
                None => {}
            }
        }
        (holding, missing)
    }

    /// The pointer to the stored artifact that holds `entry` of the text
    /// keyed `key`, if the store keeps one and the artifact it names; a
    /// miss when the store dropped either, or keeps a pointer that does not
    /// parse (damaged), which is reported and which the compile rewrites:
    /// the store is a cache.
    fn kept(&self, key: &ArtifactKey, entry: &ProgramEntry) -> Option<Pointer> {
        let entry_key = Self::entry_key(key, entry);
        let stores = self.store.stored(&entry_key);
        if let Some((kept, read)) = self.stored().get(&entry_key) {
            if *read == stores {
                return kept.clone();
            }
        }
        let store = &*self.store;
        let namespace = self.toolchain.identity().namespace;
        let kept = (|| {
            let named = store.get(namespace, &entry_key)?;
            let Some(kept) = std::str::from_utf8(&named).ok().and_then(Pointer::parse) else {
                let pointer = entry_key.as_str();
                tracing::warn!("artifact store: the stored pointer {namespace}/{pointer} does not parse; it is compiled again");
                return None;
            };
            store.contains(namespace, &kept.artifact).then_some(kept)
        })();
        self.stored().insert(entry_key, (kept.clone(), stores));
        kept
    }

    /// Forget what the store was read to keep of `entries` of the text
    /// keyed `key`: a load of it missed (the artifact gone, damaged or
    /// declined), so the next read finds what the store keeps now.
    fn forget(&self, key: &ArtifactKey, entries: &[ProgramEntry]) {
        let mut stored = self.stored();
        for entry in entries {
            stored.remove(&Self::entry_key(key, entry));
        }
    }

    fn stored(&self) -> std::sync::MutexGuard<'_, Stored> {
        self.stored
            .lock()
            .expect("store reading lock is never poisoned")
    }

    /// `source` as the former reads it ([`Keyed`]).
    pub(crate) fn keyed(&self, source: ProgramSource) -> Keyed {
        Keyed {
            key: self.key(&source),
            source,
        }
    }

    /// What forming `sources` together does: compiled when the toolchain
    /// compiles any of them, and then the most entries one source's
    /// compile forms (the sources compile at once, so the call takes what
    /// its largest compile does). It reads the device's live compiles and
    /// the store and forms nothing.
    pub(crate) fn class(&self, sources: &[Keyed]) -> (FormationClass, usize) {
        let entries = sources
            .iter()
            .map(|source| self.compiling(source))
            .max()
            .unwrap_or(0);
        match entries {
            0 => (FormationClass::Kept, 0),
            entries => (FormationClass::Compiled, entries),
        }
    }

    /// How many entries of `source` the toolchain compiles: those no live
    /// compile holds, no former of the store is forming or storing (forming
    /// them would wait for it and load), and no stored artifact holds. The
    /// flight is read before the store: a store that ends between the two
    /// has counted its end, so the store's read sees what it stored, where
    /// the other order could miss both.
    fn compiling(&self, keyed: &Keyed) -> usize {
        let (_, missing) = self.held(&keyed.key, &keyed.source.entries);
        missing
            .iter()
            .filter(|entry| {
                !self.store.in_flight(&Self::entry_key(&keyed.key, entry)) && self.kept(&keyed.key, entry).is_none()
            })
            .count()
    }

    /// Form every source, at once on the threads lent to the former
    /// ([`ProgramFormer::staff`]) and in turn without them, each toolchain
    /// call on a compiler thread ([`on_compiler_stack`]). Formed sources
    /// are returned in source order.
    pub(crate) fn form(
        &self,
        sources: &[ProgramSource],
    ) -> Result<Vec<Formed<T::Program>>, NativeCompilationError>
    where
        T: Sync,
        T::Program: Send + Sync,
    {
        match self.workers() {
            Some(workers) => workers
                .each(sources, |source| self.form_one(source))
                .into_iter()
                .collect(),
            None => sources.iter().map(|source| self.form_one(source)).collect(),
        }
    }

    /// The one way a program is had on this device. The entries no live
    /// compile holds are loaded from the stored artifacts that hold them;
    /// those none holds are compiled once, together, their artifact stored
    /// with each of its entries named, off the path that formed it, and its
    /// program used. The missing entries are claimed in the store first
    /// ([`ProgramStore::claim`]): what another device or thread is forming
    /// or storing is waited for and loaded.
    fn form_one(
        &self,
        source: &ProgramSource,
    ) -> Result<Formed<T::Program>, NativeCompilationError> {
        let key = self.key(source);
        let (mut compiles, missing) = self.held(&key, &source.entries);
        if missing.is_empty() {
            return Ok(Formed::new(key, compiles));
        }
        let began = std::time::Instant::now();
        let namespace = self.toolchain.identity().namespace;
        let claimed = missing.iter().map(|entry| Self::entry_key(&key, entry)).collect::<Vec<_>>();
        let claim = self.store.claim(&claimed);
        // What a thread of this device formed while the claim waited.
        let (formed_beside, missing) = self.held(&key, &missing);
        compiles.extend(formed_beside);
        if missing.is_empty() {
            return Ok(Formed::new(key, compiles));
        }
        // What another former formed is read once it is stored.
        claim.settled();
        self.formed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The missing entries by the stored artifact that holds them, in
        // the order first asked, and those no artifact holds.
        let mut loads: Vec<(Pointer, Vec<ProgramEntry>)> = Vec::new();
        let mut compiling = Vec::new();
        for entry in missing {
            match self.kept(&key, &entry) {
                Some(kept) => match loads.iter_mut().find(|(group, _)| *group == kept) {
                    Some((_, entries)) => entries.push(entry),
                    None => loads.push((kept, vec![entry])),
                },
                None => compiling.push(entry),
            }
        }
        let mut formed = Vec::new();
        for (kept, entries) in loads {
            // The store dropped the artifact: its entries are compiled
            // with the rest.
            let Some(artifact) = self.artifact(&kept) else {
                self.forget(&key, &entries);
                compiling.extend(entries);
                continue;
            };
            match on_compiler_stack(|| self.toolchain.load(&artifact, &entries)) {
                Ok(program) => formed.push(Compile::new(entries, program)),
                // The toolchain declined the artifact: its entries are
                // compiled with the rest, and stored again.
                Err(declined) => {
                    self.load_misses
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let kept = kept.artifact.as_str();
                    tracing::warn!("artifact store: the stored artifact {namespace}/{kept} does not load ({declined}); it is compiled again");
                    self.forget(&key, &entries);
                    compiling.extend(entries);
                }
            }
        }
        if !compiling.is_empty() {
            let program = self.compile_stored(&key, source, &compiling, claim)?;
            formed.push(Compile::new(compiling, program));
        }
        let mut live = self
            .live
            .lock()
            .expect("live program lock is never poisoned");
        live.retain(|_, compiles| {
            compiles.retain(|compile| compile.strong_count() > 0);
            !compiles.is_empty()
        });
        let held = live.entry(key.clone()).or_default();
        for compile in formed {
            let compile = Arc::new(compile);
            held.push(Arc::downgrade(&compile));
            compiles.push(compile);
        }
        drop(live);
        let took = u64::try_from(began.elapsed().as_nanos())
            .expect("a formation's nanoseconds fit 64 bits");
        self.forming_nanos
            .fetch_add(took, std::sync::atomic::Ordering::Relaxed);
        Ok(Formed::new(key, compiles))
    }

    /// The artifact `kept` names, its bytes checked against the pointer's
    /// digest and decoded, opened in place where the store keeps them in a
    /// file ([`ArtifactStore::file`]); none when the store has dropped it,
    /// or keeps bytes that do not match the digest or do not decode
    /// (damaged), which are reported, counted as a load miss and compiled
    /// again: the store is a cache, and the compile rewrites them.
    fn artifact(&self, kept: &Pointer) -> Option<T::Artifact> {
        let namespace = self.toolchain.identity().namespace;
        let bytes = self.store.get(namespace, &kept.artifact)?;
        let damage = match kept.holds(&bytes) {
            false => "does not match the digest its pointer names",
            true => match T::Artifact::decode(
                &bytes,
                T::Artifact::IN_PLACE.then(|| self.store.file(namespace, &kept.artifact)).flatten().as_deref(),
            ) {
                Some(artifact) => return Some(artifact),
                None => "does not decode",
            },
        };
        self.load_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let artifact = kept.artifact.as_str();
        tracing::warn!("artifact store: the stored artifact {namespace}/{artifact} {damage}; it is compiled again");
        None
    }

    /// Compile `entries` of `source`'s text and return the program the
    /// compile formed. Its artifact is made and stored, with each entry
    /// named, off this path ([`Claim::store`]): the program is used at once,
    /// and what making the artifact takes (Metal's archive serialization,
    /// ~40 ms a program on an M4 Pro, 2026-11-13) never delays it.
    fn compile_stored(
        &self,
        key: &ArtifactKey,
        source: &ProgramSource,
        entries: &[ProgramEntry],
        claim: Claim,
    ) -> Result<T::Program, NativeCompilationError> {
        let compiling = ProgramSource {
            text: source.text.clone(),
            entries: entries.to_vec(),
        };
        let began = std::time::Instant::now();
        let (pending, program) = on_compiler_stack(|| self.toolchain.compile(&compiling))?;
        let took = began.elapsed();
        self.compiles
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.compiled
            .lock()
            .expect("compile observation lock is never poisoned")
            .observe(entries.len(), took.as_secs_f64());
        let namespace = self.toolchain.identity().namespace;
        let (key, list) = (key.clone(), listed(entries));
        let entry_keys = entries.iter().map(|entry| Self::entry_key(&key, entry)).collect::<Vec<_>>();
        let (toolchain, store, storing) = (self.toolchain.clone(), self.store.clone(), entry_keys.clone());
        claim.store(&storing, &self.stores, move || match toolchain.artifact(pending) {
            Ok(artifact) => {
                let pointer = Pointer::of(&key, &list, &artifact);
                store.put(namespace, &pointer.artifact, &artifact);
                let text = pointer.text();
                for entry_key in &entry_keys {
                    store.put(namespace, entry_key, text.as_bytes());
                }
            }
            Err(error) => tracing::warn!(
                "artifact store: the artifact of {namespace}/{} ({list}) is not made: {error}; a later process compiles it again",
                key.as_str()
            ),
        });
        Ok(program)
    }
}

impl<T: Toolchain + Sync> Programs<T> for ProgramFormer<T>
where
    T::Program: Send + Sync,
{
    fn form(&self, sources: &[ProgramSource]) -> Result<Vec<Formed<T::Program>>, NativeCompilationError> {
        ProgramFormer::form(self, sources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a test's store holds, by namespace and key.
    type Entries = HashMap<(String, ArtifactKey), Vec<u8>>;

    /// A store in memory whose entries a test reads and damages in place.
    #[derive(Default)]
    struct Held(Mutex<Entries>);

    impl Held {
        fn entries(&self) -> std::sync::MutexGuard<'_, Entries> {
            self.0.lock().expect("the test's store")
        }

        /// Where the store holds the artifact that `entry` of the text
        /// keyed `key` was stored in, as its pointer names it.
        fn artifact(&self, key: &ArtifactKey, entry: &ProgramEntry) -> (String, ArtifactKey) {
            let named = self.entries()[&(String::from("echo"), ProgramFormer::<Echo>::entry_key(key, entry))].clone();
            let pointer = std::str::from_utf8(&named).ok().and_then(Pointer::parse).expect("the entry's pointer");
            (String::from("echo"), pointer.artifact)
        }
    }

    impl ArtifactStore for Held {
        fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
            self.entries().get(&(namespace.to_owned(), key.clone())).cloned()
        }

        fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) {
            self.entries().insert((namespace.to_owned(), key.clone()), bytes.to_vec());
        }
    }
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Compiles a source to its entries' symbols, its artifact their names
    /// joined, and loads the requested ones from an artifact that names
    /// them. It counts the calls that compiled and the entries they formed.
    struct Echo {
        identity: seismic_native_target::ToolchainIdentity,
        compiles: AtomicUsize,
        entries: AtomicUsize,
        /// Whether its platform declines every load, as a driver may
        /// decline a pipeline cache's data.
        misses: bool,
        /// The thread each compile and load ran on.
        threads: Mutex<Vec<Option<String>>>,
        /// What each artifact it makes records besides its names and
        /// compile count, which no key covers: toolchains of one material
        /// stamped apart make different bytes of one compile, as two
        /// processes' Metal archives may.
        stamp: usize,
        /// What each compile waits at before it returns, where given: the
        /// formers of a race have each read a miss before any stores.
        gate: Option<Arc<std::sync::Barrier>>,
    }

    /// Recurse, a 4 KiB frame at a time, until the stack holds `bytes`
    /// below `start` (an address on it), as a platform compiler may: what
    /// a caller's stack does not hold. Measured by address, not by frames,
    /// whose size the build's optimization decides.
    #[inline(never)]
    fn deep(start: usize, bytes: usize) -> u8 {
        let pad = std::hint::black_box([1u8; 4096]);
        if start - pad.as_ptr() as usize >= bytes {
            pad[0]
        } else {
            deep(start, bytes).wrapping_add(pad[4095])
        }
    }

    impl Echo {
        fn ran(&self) {
            let thread = std::thread::current().name().map(str::to_owned);
            self.threads.lock().unwrap().push(thread);
        }
    }

    /// An artifact of entry names, joined, then `#` and the count of the
    /// compiles its toolchain had run when it was made and its stamp,
    /// which loading ignores: no two compiles make the same bytes, as
    /// Metal's archives need not.
    struct Names(Vec<String>);

    impl Artifact for Names {
        const ENCODING: u32 = 1;
        fn decode(bytes: &[u8], _: Option<&std::path::Path>) -> Option<Self> {
            let (names, _) = std::str::from_utf8(bytes).ok()?.split_once('#')?;
            Some(Self(names.split(',').map(str::to_owned).collect()))
        }
    }

    impl Toolchain for Echo {
        type Artifact = Names;
        type Program = Vec<String>;
        type Pending = Names;
        fn identity(&self) -> &seismic_native_target::ToolchainIdentity {
            &self.identity
        }
        fn compile(
            &self,
            source: &ProgramSource,
        ) -> Result<(Names, Vec<String>), NativeCompilationError> {
            self.ran();
            if source.text == "deep" {
                // Half the owner's stack: more than twice a libtest thread's.
                let start = std::hint::black_box(0u8);
                std::hint::black_box(deep(&start as *const u8 as usize, seismic_threads::STACK_BYTES / 2));
            }
            self.compiles.fetch_add(1, Ordering::SeqCst);
            self.entries.fetch_add(source.entries.len(), Ordering::SeqCst);
            if let Some(gate) = &self.gate {
                gate.wait();
            }
            let names = source.entries.iter().map(|entry| entry.symbol.clone()).collect::<Vec<_>>();
            Ok((Names(names.clone()), names))
        }
        fn artifact(&self, names: Names) -> Result<Vec<u8>, NativeCompilationError> {
            assert!(!names.0.iter().any(|name| name == "unstorable"), "a store panicked by design");
            Ok(format!("{}#{}#{}", names.0.join(","), self.compiles.load(Ordering::SeqCst), self.stamp).into_bytes())
        }
        fn load(&self, artifact: &Names, entries: &[ProgramEntry]) -> Result<Vec<String>, seismic_native_target::Declined> {
            self.ran();
            if self.misses {
                return Err(seismic_native_target::Declined("the platform declines every load".into()));
            }
            entries
                .iter()
                .map(|entry| {
                    artifact
                        .0
                        .iter()
                        .find(|held| **held == entry.symbol)
                        .cloned()
                        .ok_or_else(|| seismic_native_target::Declined(format!("the artifact holds no `{}`", entry.symbol)))
                })
                .collect()
        }
        fn statistics(_: &Vec<String>, _: usize) -> &[seismic_native_target::ProgramStatistic] {
            &[]
        }
    }

    /// A former of a process of its own over `store` (none: a store of its
    /// own).
    fn former(store: Option<Arc<Held>>) -> ProgramFormer<Echo> {
        former_missing(store, false)
    }

    /// A former of a device opened with `store`, beside any other of it.
    fn former_on(store: &Arc<ProgramStore>) -> ProgramFormer<Echo> {
        former_with(store.clone(), false, "build 1")
    }

    /// A former whose toolchain's loads are all misses when `misses`.
    fn former_missing(store: Option<Arc<Held>>, misses: bool) -> ProgramFormer<Echo> {
        former_built(store, misses, "build 1")
    }

    /// A former whose toolchain's formation code is the build `stored`
    /// names ([`seismic_native_target::ToolchainIdentity::stored`]).
    fn former_built(store: Option<Arc<Held>>, misses: bool, stored: &str) -> ProgramFormer<Echo> {
        let bytes = store.unwrap_or_default() as Arc<dyn ArtifactStore>;
        former_with(Arc::new(ProgramStore::new(bytes)), misses, stored)
    }

    fn former_with(store: Arc<ProgramStore>, misses: bool, stored: &str) -> ProgramFormer<Echo> {
        former_of(store, misses, "echo 1", stored)
    }

    /// A former whose toolchain's material is `material`.
    fn former_of(store: Arc<ProgramStore>, misses: bool, material: &str, stored: &str) -> ProgramFormer<Echo> {
        ProgramFormer::new(echo(misses, material, stored, 0, None), store)
    }

    /// An echo toolchain of `material` and the build `stored`, stamping its
    /// artifacts `stamp`, each compile waiting at `gate` where given.
    fn echo(misses: bool, material: &str, stored: &str, stamp: usize, gate: Option<Arc<std::sync::Barrier>>) -> Echo {
        Echo {
            identity: seismic_native_target::ToolchainIdentity {
                namespace: "echo",
                material: material.into(),
                stored: stored.into(),
            },
            compiles: AtomicUsize::new(0),
            entries: AtomicUsize::new(0),
            misses,
            threads: Mutex::default(),
            stamp,
            gate,
        }
    }

    fn source(text: &str, symbols: &[&str]) -> ProgramSource {
        ProgramSource {
            text: text.into(),
            entries: symbols.iter().copied().map(ProgramEntry::named).collect(),
        }
    }

    /// Wait until every artifact `former` compiled is stored: what a
    /// process that formed them leaves when it ends.
    fn settled(former: &ProgramFormer<Echo>) {
        former.stores.settle();
    }

    fn compiles(former: &ProgramFormer<Echo>) -> usize {
        former.toolchain().compiles.load(Ordering::SeqCst)
    }

    fn entries(former: &ProgramFormer<Echo>) -> usize {
        former.toolchain().entries.load(Ordering::SeqCst)
    }

    fn keyed(former: &ProgramFormer<Echo>, sources: &[ProgramSource]) -> Vec<Keyed> {
        sources
            .iter()
            .map(|source| former.keyed(source.clone()))
            .collect()
    }

    fn class(former: &ProgramFormer<Echo>, sources: &[ProgramSource]) -> (FormationClass, usize) {
        former.class(&keyed(former, sources))
    }

    /// A compile that needs more stack than its caller has is formed: every
    /// compile and load runs on a compiler thread of the former's, whatever
    /// thread asks, and whether or not workers were lent.
    #[test]
    fn every_toolchain_call_runs_on_a_compiler_thread_whatever_its_caller_holds() {
        let store = Arc::new(Held::default());
        let first = former(Some(store.clone()));
        // A caller with 1 MiB of stack forms a compile that takes several.
        std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(move || {
                drop(first.form(&[source("deep", &["k"])]).unwrap());
                first
            })
            .unwrap()
            .join()
            .map(|first| {
                assert_eq!(compiles(&first), 1);
                let threads = first.toolchain().threads.lock().unwrap().clone();
                assert_eq!(threads, [Some(COMPILER_THREAD.to_owned())]);
            })
            .unwrap();
        // A second former loads the stored artifact.
        let second = former(Some(store));
        second.form(&[source("deep", &["k"])]).unwrap();
        assert_eq!(compiles(&second), 0);
        let threads = second.toolchain().threads.lock().unwrap().clone();
        assert_eq!(threads, [Some(COMPILER_THREAD.to_owned())]);
    }

    /// A rebuild of the backend's formation code, with the material the
    /// same, misses every program the store keeps and compiles it again;
    /// a former of the build that stored them loads them. That the tuning
    /// identity names the material and not the build is
    /// `backends::tests::the_tuning_identity_names_the_toolchain_and_not_its_build`.
    #[test]
    fn a_rebuilt_formation_misses_the_programs_stored_before() {
        let store = Arc::new(Held::default());
        let program = [source("launch", &["k"])];
        let formed = |build| {
            let former = former_built(Some(store.clone()), false, build);
            former.form(&program).unwrap();
            compiles(&former)
        };
        assert_eq!([formed("build 1"), formed("build 1"), formed("build 2")], [1, 0, 1]);
    }

    /// A stored program is loaded only under its whole key: one character
    /// of its source, its entry's specialization (constants, group size),
    /// the toolchain's material (compiler, driver, device) or the build of
    /// its formation code changed, it is compiled.
    #[test]
    fn a_program_is_loaded_only_under_its_whole_key() {
        let store = Arc::new(ProgramStore::new(Arc::new(Held::default())));
        let entry = |constants: Vec<u32>, group_size| ProgramEntry { symbol: "k".into(), group_size, constants };
        let program = |text: &str, entry| ProgramSource { text: text.into(), entries: vec![entry] };
        let formed = |material: &str, stored: &str, source: ProgramSource| {
            let former = former_of(store.clone(), false, material, stored);
            former.form(&[source]).unwrap();
            compiles(&former)
        };
        assert_eq!(formed("echo 1", "build 1", program("launch", entry(vec![], None))), 1);
        assert_eq!(formed("echo 1", "build 1", program("launch", entry(vec![], None))), 0);
        assert_eq!(formed("echo 1", "build 1", program("launch ", entry(vec![], None))), 1);
        assert_eq!(formed("echo 1", "build 1", program("launch", entry(vec![1], None))), 1);
        assert_eq!(formed("echo 1", "build 1", program("launch", entry(vec![], Some([64, 1, 1])))), 1);
        assert_eq!(formed("echo 2", "build 1", program("launch", entry(vec![], None))), 1);
        assert_eq!(formed("echo 1", "build 2", program("launch", entry(vec![], None))), 1);
    }

    #[test]
    fn a_live_compile_serves_every_entry_it_holds() {
        let former = former(None);
        let all = former
            .form(&[source("launch", &["k<16>", "k<32>"])])
            .unwrap();
        let one = former.form(&[source("launch", &["k<32>"])]).unwrap();
        let (held, position) = one[0].entry(&ProgramEntry::named("k<32>"));
        assert!(Arc::ptr_eq(held, all[0].entry(&ProgramEntry::named("k<16>")).0));
        assert_eq!(position, 1);
        assert_eq!(compiles(&former), 1);
    }

    /// An entry no live compile holds is compiled alone: what is held is
    /// not formed again, and the source is served from both compiles.
    #[test]
    fn only_the_entries_no_live_compile_holds_are_compiled() {
        let former = former(None);
        let held = former
            .form(&[source("launch", &["k<16>", "k<32>"])])
            .unwrap();
        let wider = former
            .form(&[source("launch", &["k<16>", "k<64>", "k<128>"])])
            .unwrap();
        // Two calls: the first formed two entries, the second the two
        // that were missing, together.
        assert_eq!((compiles(&former), entries(&former)), (2, 4));
        let (first, _) = wider[0].entry(&ProgramEntry::named("k<16>"));
        assert!(Arc::ptr_eq(first, held[0].entry(&ProgramEntry::named("k<32>")).0));
        let (second, position) = wider[0].entry(&ProgramEntry::named("k<128>"));
        assert_eq!(second.entries().len(), 2);
        assert_eq!(second.program()[position], "k<128>");
    }

    /// A compile lives while a caller holds it, and no longer: a caller
    /// that holds one entry's compile does not keep another's. An entry no
    /// live compile holds is formed again, from the store, compiling
    /// nothing.
    #[test]
    fn a_dropped_compile_is_formed_again_and_a_held_one_is_not() {
        let former = former(None);
        let formed = |former: &ProgramFormer<Echo>| former.programs_formed().programs;
        let first = former.form(&[source("launch", &["k<16>"])]).unwrap();
        let both = former
            .form(&[source("launch", &["k<16>", "k<32>"])])
            .unwrap();
        assert_eq!((formed(&former), compiles(&former)), (2, 2));
        // What a caller keeps of a formed source is the compile its entry
        // is in.
        let kept = both[0].entry(&ProgramEntry::named("k<16>")).0.clone();
        drop((first, both));
        former.form(&[source("launch", &["k<16>"])]).unwrap();
        assert_eq!(formed(&former), 2);
        former.form(&[source("launch", &["k<32>"])]).unwrap();
        assert_eq!(formed(&former), 3);
        drop(kept);
        former.form(&[source("launch", &["k<16>"])]).unwrap();
        assert_eq!((formed(&former), compiles(&former)), (4, 2));
    }

    /// A class is what is live and kept when it is asked: what the store
    /// came to keep since the class was last read is kept.
    #[test]
    fn a_class_read_again_follows_what_the_store_keeps() {
        use FormationClass::{Compiled, Kept};
        let former = former(None);
        let other = source("other", &["k"]);
        assert_eq!(class(&former, &[other.clone()]), (Compiled, 1));
        drop(former.form(&[other.clone()]).unwrap());
        assert_eq!(class(&former, &[other]), (Kept, 0));
    }

    /// Another process is loaded from whichever stored artifact of the same
    /// text holds what it asks for, a wider one included, and compiles
    /// nothing the first compiled.
    #[test]
    fn another_process_loads_every_entry_from_the_store_and_compiles_none() {
        let store = Arc::new(Held::default());
        let first = former(Some(store.clone()));
        let _narrow = first.form(&[source("launch", &["k<8>"])]).unwrap();
        // The second call compiles only what the first did not.
        drop(
            first
                .form(&[source("launch", &["k<8>", "k<16>", "k<32>"])])
                .unwrap(),
        );
        assert_eq!((compiles(&first), entries(&first)), (2, 3));
        settled(&first);
        let fresh = former(Some(store));
        fresh.form(&[source("launch", &["k<32>"])]).unwrap();
        fresh.form(&[source("launch", &["k<8>"])]).unwrap();
        assert_eq!(compiles(&fresh), 0);
        fresh.form(&[source("other", &["k<32>"])]).unwrap();
        assert_eq!(compiles(&fresh), 1);
    }

    /// Processes compiling the same entry at once, each through a
    /// directory store of its own over one directory, each store an
    /// artifact of their own, which differ as Metal's archives may, and
    /// then their pointers: a reader meanwhile finds every pointer whole or
    /// none, whichever pointer lands names a whole artifact its digest
    /// matches, and a later process loads the entry and compiles nothing.
    #[test]
    fn processes_compiling_at_once_leave_every_pointer_whole() {
        use crate::artifacts::{DirectoryStore, Eviction, Examining, DEFAULT_CAPACITY};
        fn alone(_: &std::path::Path, _: &Examining<'_>) -> Eviction {
            Eviction::Alone
        }
        let root = std::env::temp_dir().join(format!("seismic-formation-race-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let directory = || Arc::new(DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, alone).expect("the race's directory opens"));
        // A process: its own directory store and program store, its
        // toolchain stamping its artifacts apart from every other's.
        let process = |stamp, gate| ProgramFormer::new(echo(false, "echo 1", "build 1", stamp, gate), Arc::new(ProgramStore::new(directory())));
        let launch = source("launch", &["k<8>"]);
        let gate = Arc::new(std::sync::Barrier::new(2));
        let racing = [process(0, Some(gate.clone())), process(1, Some(gate))];
        let entry = ProgramFormer::<Echo>::entry_key(&racing[0].key(&launch), &launch.entries[0]);
        let reader = directory();
        // The pointer `bytes` and the artifact it names, whole.
        let whole = |bytes: &[u8]| {
            let pointer = std::str::from_utf8(bytes).ok().and_then(Pointer::parse).expect("a pointer read is whole");
            let artifact = reader.get("echo", &pointer.artifact).expect("a pointer read names an artifact stored before it");
            assert!(pointer.holds(&artifact), "a pointer read names the artifact's bytes");
        };
        let done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !done.load(Ordering::SeqCst) {
                    if let Some(bytes) = reader.get("echo", &entry) {
                        whole(&bytes);
                    }
                }
            });
            let formed = racing
                .iter()
                .map(|former| {
                    let launch = &launch;
                    scope.spawn(move || {
                        drop(former.form(&[launch.clone()]).unwrap());
                        settled(former);
                    })
                })
                .collect::<Vec<_>>();
            let ended = formed.into_iter().map(|former| former.join()).collect::<Vec<_>>();
            // The reader stops whatever the formers met, so a failure
            // fails the test rather than hanging it.
            done.store(true, Ordering::SeqCst);
            for ended in ended {
                ended.unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            }
        });
        assert_eq!(racing.iter().map(compiles).collect::<Vec<_>>(), [1, 1], "each process read a miss and compiled");
        let pointer = reader.file("echo", &entry).expect("a pointer landed");
        let held = std::fs::read_dir(pointer.parent().unwrap()).unwrap().flatten().count();
        assert_eq!(held, 3, "the pointer and each process's own artifact, no temporary file");
        whole(&std::fs::read(&pointer).unwrap());
        let later = process(2, None);
        drop(later.form(&[launch]).unwrap());
        assert_eq!((compiles(&later), later.programs_formed().load_misses), (0, 0));
        drop((racing, later));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Only what the store does not return is a miss: an artifact it
    /// dropped is classed as compiled though a pointer names it, and is
    /// compiled again, with the entries no artifact holds, in one call; one
    /// it returns that does not decode is damaged, a miss too, and the
    /// compile rewrites it.
    #[test]
    fn only_a_dropped_artifact_is_a_miss() {
        let store = Arc::new(Held::default());
        let first = former(Some(store.clone()));
        let narrow = source("launch", &["k<8>"]);
        drop(first.form(&[narrow.clone()]).unwrap());
        settled(&first);
        let key = first.key(&narrow);
        let stored = store.artifact(&key, &narrow.entries[0]);
        store.entries().remove(&stored).expect("the artifact was stored");
        let wide = source("launch", &["k<8>", "k<16>"]);
        let fresh = former(Some(store.clone()));
        assert_eq!(class(&fresh, &[narrow.clone()]), (FormationClass::Compiled, 1));
        drop(fresh.form(&[wide.clone()]).unwrap());
        assert_eq!((compiles(&fresh), entries(&fresh)), (1, 2));
        settled(&fresh);
        let wider = store.artifact(&key, &wide.entries[0]);
        store.entries().insert(wider, vec![0xff]);
        let damaged = former(Some(store.clone()));
        drop(damaged.form(&[narrow.clone()]).unwrap());
        assert_eq!(compiles(&damaged), 1, "an undecodable artifact is compiled again");
        assert_eq!(damaged.programs_formed().load_misses, 1, "and counted as a miss");
        settled(&damaged);
        let after = former(Some(store));
        assert_eq!(class(&after, &[narrow]), (FormationClass::Kept, 0), "and its entry's pointer rewritten");
    }

    /// A stored artifact the platform will not load without compiling is a
    /// miss, not a failure: its entries are compiled with those no artifact
    /// holds, in one call, the miss is counted, and the artifact is stored
    /// again, which a platform that serves it then loads.
    #[test]
    fn a_load_that_misses_is_compiled_and_stored_again() {
        let store = Arc::new(Held::default());
        let narrow = source("launch", &["k<8>"]);
        drop(former(Some(store.clone())).form(&[narrow.clone()]).unwrap());
        let declining = former_missing(Some(store.clone()), true);
        assert_eq!(class(&declining, &[narrow.clone()]), (FormationClass::Kept, 0));
        let wide = source("launch", &["k<8>", "k<16>"]);
        drop(declining.form(&[wide.clone()]).unwrap());
        assert_eq!((compiles(&declining), entries(&declining)), (1, 2));
        let counted = declining.programs_formed();
        assert_eq!((counted.compiles, counted.load_misses), (1, 1));
        settled(&declining);
        let serving = former(Some(store));
        drop(serving.form(&[wide]).unwrap());
        assert_eq!(compiles(&serving), 0, "the compile stored the missed entries again");
    }

    /// A damaged entry pointer is a miss: the entry compiles again, and
    /// the compile rewrites the pointer, which a later former then follows.
    #[test]
    fn a_damaged_pointer_is_a_miss_and_is_rewritten() {
        let store = Arc::new(Held::default());
        let first = former(Some(store.clone()));
        let narrow = source("launch", &["k<8>"]);
        drop(first.form(&[narrow.clone()]).unwrap());
        settled(&first);
        let pointer = (
            String::from("echo"),
            ProgramFormer::<Echo>::entry_key(&first.key(&narrow), &narrow.entries[0]),
        );
        store.entries().insert(pointer.clone(), b"not a key".to_vec());
        let damaged = former(Some(store.clone()));
        assert_eq!(class(&damaged, &[narrow.clone()]), (FormationClass::Compiled, 1));
        drop(damaged.form(&[narrow.clone()]).unwrap());
        assert_eq!(compiles(&damaged), 1);
        settled(&damaged);
        let rewritten = store.entries()[&pointer].clone();
        let parsed = std::str::from_utf8(&rewritten).ok().and_then(Pointer::parse);
        assert!(parsed.is_some(), "the compile rewrote the pointer");
        let after = former(Some(store));
        assert_eq!(class(&after, &[narrow]), (FormationClass::Kept, 0));
    }

    /// What forming a source does is known before it is formed, and is
    /// what forming it then does: kept when live compiles or the store
    /// serve every entry, compiled otherwise.
    #[test]
    fn a_formation_is_classed_by_the_live_compiles_and_the_store_before_it_runs() {
        let store = Arc::new(Held::default());
        let first = former(Some(store.clone()));
        let wide = source("launch", &["k<16>", "k<32>"]);
        use FormationClass::{Compiled, Kept};
        assert_eq!(class(&first, &[wide.clone()]), (Compiled, 2));
        let held = first.form(&[wide.clone()]).unwrap();
        assert_eq!(compiles(&first), 1);
        // What the toolchain's compiles took is observed with what they
        // formed.
        assert!(first.compiled().price(2).is_some());
        // Live on the device, and kept by the store once dropped.
        assert_eq!(class(&first, &[wide.clone()]), (Kept, 0));
        // A wider request compiles only what is not live.
        assert_eq!(
            class(&first, &[source("launch", &["k<16>", "k<64>"])]),
            (Compiled, 1)
        );
        drop(held);
        assert_eq!(class(&first, &[wide.clone()]), (Kept, 0));
        settled(&first);
        // Another process with the same store: what the store keeps is
        // loaded, an entry it lacks is compiled, and a call forming both
        // compiles.
        let fresh = former(Some(store));
        let narrow = source("launch", &["k<32>"]);
        let lacking = source("launch", &["k<64>"]);
        assert_eq!(class(&fresh, &[narrow.clone()]), (Kept, 0));
        assert_eq!(class(&fresh, &[lacking.clone()]), (Compiled, 1));
        assert_eq!(
            class(&fresh, &[narrow.clone(), source("other", &["k"])]),
            (Compiled, 1)
        );
        fresh.form(&[narrow]).unwrap();
        assert_eq!(compiles(&fresh), 0);
        fresh.form(&[lacking]).unwrap();
        assert_eq!(compiles(&fresh), 1);
        // A process with a store of its own keeps nothing of another's.
        assert_eq!(class(&former(None), &[wide]), (Compiled, 2));
        assert_eq!(class(&former(None), &[]), (Kept, 0));
    }

    /// Devices of one store form a program once between them: what one
    /// formed another loads, though it read the store before it was formed.
    #[test]
    fn devices_of_one_store_form_a_program_once() {
        let store = Arc::new(ProgramStore::new(Arc::new(Held::default())));
        use FormationClass::{Compiled, Kept};
        let (first, second) = (former_on(&store), former_on(&store));
        let wide = source("launch", &["k<16>", "k<32>"]);
        assert_eq!(class(&second, &[wide.clone()]), (Compiled, 2));
        drop(first.form(&[wide.clone()]).unwrap());
        assert_eq!(class(&second, &[wide.clone()]), (Kept, 0));
        second.form(&[wide]).unwrap();
        assert_eq!((compiles(&first), compiles(&second)), (1, 0));
    }

    /// A read of what the store keeps, held or missed, holds only until a
    /// store of its entry ends or a load of it misses: once a damaged
    /// artifact is stored again, with other bytes, the former follows the
    /// new pointer and loads it, compiling nothing.
    #[test]
    fn a_held_read_follows_its_entry_stored_again() {
        let store = Arc::new(Held::default());
        let narrow = source("launch", &["k<8>"]);
        let forming = former(Some(store.clone()));
        drop(forming.form(&[narrow.clone()]).unwrap());
        settled(&forming);
        let artifact = store.artifact(&forming.key(&narrow), &narrow.entries[0]);
        store.entries().get_mut(&artifact).expect("the artifact was stored").push(b'x');
        drop(forming.form(&[narrow.clone()]).unwrap());
        assert_eq!(compiles(&forming), 2, "the damaged artifact is compiled again");
        assert_eq!(forming.programs_formed().load_misses, 1, "its load counted as a miss");
        settled(&forming);
        drop(forming.form(&[narrow]).unwrap());
        assert_eq!(compiles(&forming), 2, "and the entry stored again is loaded");
    }

    /// A store that panics fails every wait for it, its former's and its
    /// store's, as what it held is not stored; the store's thread goes on
    /// storing what others compile.
    #[test]
    fn a_store_that_panics_fails_every_wait_for_it() {
        let store = Arc::new(ProgramStore::new(Arc::new(Held::default())));
        let failing = former_on(&store);
        drop(failing.form(&[source("launch", &["unstorable"])]).unwrap());
        let waited = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| settled(&failing)));
        let message = waited.expect_err("the former's wait fails");
        assert!(message.downcast_ref::<String>().is_some_and(|message| message.contains("a store panicked by design")));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| store.settle())).is_err());
        let other = former_on(&store);
        drop(other.form(&[source("other", &["k"])]).unwrap());
        settled(&other);
        // Its drop would fail its wait again.
        std::mem::forget(failing);
    }

    /// A memory store that counts its reads.
    #[derive(Default)]
    struct Counted {
        bytes: Held,
        reads: AtomicUsize,
    }

    impl ArtifactStore for Counted {
        fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.bytes.get(namespace, key)
        }

        fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) {
            self.bytes.put(namespace, key, bytes);
        }
    }

    /// A miss is read from the store again only once a store of its own
    /// entry has ended: what other entries store leaves it as read.
    #[test]
    fn a_miss_is_read_again_only_once_its_entry_is_stored() {
        let bytes = Arc::new(Counted::default());
        let store = Arc::new(ProgramStore::new(bytes.clone()));
        use FormationClass::{Compiled, Kept};
        let (first, second) = (former_on(&store), former_on(&store));
        let (wide, other) = (source("launch", &["k<16>", "k<32>"]), source("other", &["k"]));
        assert_eq!(class(&second, &[wide.clone()]), (Compiled, 2));
        assert_eq!(class(&second, &[other.clone()]), (Compiled, 1));
        drop(first.form(&[wide.clone()]).unwrap());
        settled(&first);
        let reads = bytes.reads.load(Ordering::SeqCst);
        assert_eq!(class(&second, &[other]), (Compiled, 1));
        assert_eq!(bytes.reads.load(Ordering::SeqCst), reads, "no store of its entry has ended");
        assert_eq!(class(&second, &[wide]), (Kept, 0));
        assert!(bytes.reads.load(Ordering::SeqCst) > reads, "its entries' stores ended");
    }

    /// A program many devices of a store ask for at once is compiled once:
    /// the rest wait for it and load it.
    #[test]
    fn a_program_formed_at_once_by_many_is_compiled_once() {
        let store = Arc::new(ProgramStore::new(Arc::new(Held::default())));
        let formers = (0..8).map(|_| former_on(&store)).collect::<Vec<_>>();
        let start = std::sync::Barrier::new(formers.len());
        let wide = source("launch", &["k<16>", "k<32>"]);
        std::thread::scope(|scope| {
            for former in &formers {
                let (start, wide) = (&start, wide.clone());
                scope.spawn(move || {
                    start.wait();
                    drop(former.form(&[wide]).unwrap());
                });
            }
        });
        assert_eq!(formers.iter().map(compiles).sum::<usize>(), 1);
    }
}
