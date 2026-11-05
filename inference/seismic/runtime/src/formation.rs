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
//! artifact holds is compiled, once, and stored. Loading compiles nothing
//! on a [`Load::Complete`] toolchain, and is counted and timed on a
//! [`Load::PlatformCached`] one.
//!
//! The store is a cache and may drop any key: an entry no pointer names,
//! or whose named artifact is gone, is classed as compiled and compiled
//! again. Bytes it returns were written here under a key that covers their
//! encoding ([`Artifact::ENCODING`]), so a pointer that does not parse or
//! an artifact that does not decode is damaged: each is reported, missed,
//! and rewritten by the compile it causes.

use crate::artifacts::{ArtifactKey, ArtifactStore};
use crate::native::Linear;
use seismic_native_target::workers::Workers;
use seismic_native_target::{
    Artifact, Load, NativeCompilationError, ProgramEntry, ProgramSource, Toolchain,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// One compile of a source text: the executable program and the entries it
/// holds, in the program's order.
pub struct Compile<P> {
    entries: Vec<ProgramEntry>,
    program: P,
}

impl<P> Compile<P> {
    /// A compile that holds `entries` in `program`, taken out of a compile
    /// the toolchain formed with more: what a holder of one entry keeps
    /// when the others are not its own. It is no compile of the device's
    /// live set: nothing is served from it but its holder.
    pub(crate) fn of(entries: Vec<ProgramEntry>, program: P) -> Self {
        Self { entries, program }
    }

    pub fn program(&self) -> &P {
        &self.program
    }

    pub fn entries(&self) -> &[ProgramEntry] {
        &self.entries
    }
}

/// A formed source: the key of its text and the live compiles that
/// together hold every entry requested of it.
pub struct Formed<P> {
    key: ArtifactKey,
    compiles: Vec<Arc<Compile<P>>>,
}

impl<P> Formed<P> {
    pub fn key(&self) -> &ArtifactKey {
        &self.key
    }

    /// The compile that holds `entry` and the entry's position in it.
    pub fn entry(&self, entry: &ProgramEntry) -> (&Arc<Compile<P>>, usize) {
        self.compiles
            .iter()
            .find_map(|compile| {
                let position = compile.entries.iter().position(|held| held == entry)?;
                Some((compile, position))
            })
            .expect("a formed source holds every entry it was requested for")
    }
}

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
    /// The stored artifacts loaded by forming them again through the
    /// platform's cache ([`Load::PlatformCached`]), and the seconds those
    /// loads took: a few milliseconds each while the platform serves them,
    /// a compile's time where it does not.
    pub platform_loads: u64,
    pub platform_load_seconds: f64,
}

/// What forming a program does, known before it is formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FormationClass {
    /// No compiler of the toolchain's runs: compiles live on the device
    /// hold every requested entry, or the store keeps an artifact that
    /// serves those they lack.
    Kept,
    /// The toolchain compiles the entries no live compile holds.
    Compiled,
}

pub(crate) struct ProgramFormer<T: Toolchain> {
    toolchain: T,
    store: Option<Arc<dyn ArtifactStore>>,
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
    /// store each time. Only this former writes the store while it lives,
    /// and it records here every entry it stores.
    stored: Mutex<Stored>,
    /// The compiles the toolchain ran: entries no live compile held and no
    /// stored artifact served.
    compiles: std::sync::atomic::AtomicU64,
    /// The stored artifacts a [`Load::PlatformCached`] toolchain loaded by
    /// forming them again through the platform's cache, and the
    /// nanoseconds those loads took.
    platform_loads: std::sync::atomic::AtomicU64,
    platform_load_nanos: std::sync::atomic::AtomicU64,
    /// What this toolchain's compiles have taken in this process, each with
    /// the entries it formed: what a compile of a source not yet compiled
    /// here is priced by.
    compiled: Mutex<Linear>,
    /// The threads its owner lent for work off the calling thread
    /// ([`ProgramFormer::staff`]); without them every source is formed in
    /// turn on the thread that asks.
    workers: Mutex<Option<Arc<Workers>>>,
}

/// What a former read the store to keep ([`ProgramFormer::stored`]), by the
/// key of an entry ([`ProgramFormer::entry_key`]).
type Stored = HashMap<ArtifactKey, Option<ArtifactKey>>;

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
    pub(crate) fn new(toolchain: T, store: Option<Arc<dyn ArtifactStore>>) -> Self {
        Self {
            toolchain,
            store,
            live: Mutex::new(HashMap::new()),
            formed: std::sync::atomic::AtomicUsize::new(0),
            forming_nanos: std::sync::atomic::AtomicU64::new(0),
            stored: Mutex::default(),
            compiles: std::sync::atomic::AtomicU64::new(0),
            platform_loads: std::sync::atomic::AtomicU64::new(0),
            platform_load_nanos: std::sync::atomic::AtomicU64::new(0),
            compiled: Mutex::new(Linear::default()),
            workers: Mutex::new(None),
        }
    }

    /// Lend the former `workers` for what it forms off the calling thread,
    /// or take them back (none). Whoever owns a span of formation owns its
    /// concurrency: the former starts no thread of its own.
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
    /// Whether its toolchain loads a stored program through a platform
    /// cache ([`Load::PlatformCached`]).
    pub(crate) fn platform_cached(&self) -> bool {
        T::LOAD == Load::PlatformCached
    }

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
            platform_loads: self.platform_loads.load(std::sync::atomic::Ordering::Relaxed),
            platform_load_seconds: std::time::Duration::from_nanos(
                self.platform_load_nanos.load(std::sync::atomic::Ordering::Relaxed),
            )
            .as_secs_f64(),
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
            &T::Artifact::ENCODING.to_le_bytes(),
            source.text.as_bytes(),
        ])
    }

    /// Where the store keeps the artifact of a compile of the text keyed
    /// `key` with the entry list `list`.
    fn artifact_key(key: &ArtifactKey, list: &str) -> ArtifactKey {
        ArtifactKey::of(&[b"artifact", key.as_str().as_bytes(), list.as_bytes()])
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
            if holding.iter().any(|compile| compile.entries.contains(entry)) {
                continue;
            }
            match live.iter().find(|compile| compile.entries.contains(entry)) {
                Some(compile) => holding.push(compile.clone()),
                None if !missing.contains(entry) => missing.push(entry.clone()),
                None => {}
            }
        }
        (holding, missing)
    }

    /// The stored artifact that holds `entry` of the text keyed `key`, if
    /// the store keeps a pointer to one and the artifact it names; a miss
    /// when the store dropped either, or keeps a pointer that does not
    /// parse (damaged), which is reported and which the compile rewrites:
    /// the store is a cache.
    fn kept(&self, key: &ArtifactKey, entry: &ProgramEntry) -> Option<ArtifactKey> {
        let entry_key = Self::entry_key(key, entry);
        if let Some(kept) = self.stored().get(&entry_key) {
            return kept.clone();
        }
        let kept = self.store.as_deref().and_then(|store| {
            let namespace = self.toolchain.identity().namespace;
            let named = store.get(namespace, &entry_key)?;
            let Some(kept) = std::str::from_utf8(&named).ok().and_then(ArtifactKey::parse) else {
                let pointer = entry_key.as_str();
                eprintln!("seismic: the stored pointer {namespace}/{pointer} does not parse; it is compiled again");
                return None;
            };
            store.contains(namespace, &kept).then_some(kept)
        });
        self.stored().insert(entry_key, kept.clone());
        kept
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
    /// compile holds and no stored artifact holds.
    fn compiling(&self, keyed: &Keyed) -> usize {
        let (_, missing) = self.held(&keyed.key, &keyed.source.entries);
        missing
            .iter()
            .filter(|entry| self.kept(&keyed.key, entry).is_none())
            .count()
    }

    /// Form every source, at once on the threads lent to the former
    /// ([`ProgramFormer::staff`]) and in turn without them. Formed sources
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
    /// with each of its entries named and its program used.
    fn form_one(
        &self,
        source: &ProgramSource,
    ) -> Result<Formed<T::Program>, NativeCompilationError> {
        let key = self.key(source);
        let (mut compiles, missing) = self.held(&key, &source.entries);
        if missing.is_empty() {
            return Ok(Formed { key, compiles });
        }
        self.formed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let began = std::time::Instant::now();
        // The missing entries by the stored artifact that holds them, in
        // the order first asked, and those no artifact holds.
        let mut loads: Vec<(ArtifactKey, Vec<ProgramEntry>)> = Vec::new();
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
            match self.artifact(&kept) {
                Some(artifact) => {
                    let loading = std::time::Instant::now();
                    let program = self.toolchain.load(&artifact, &entries)?;
                    if T::LOAD == Load::PlatformCached {
                        let took = u64::try_from(loading.elapsed().as_nanos())
                            .expect("a load's nanoseconds fit 64 bits");
                        self.platform_loads
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        self.platform_load_nanos
                            .fetch_add(took, std::sync::atomic::Ordering::Relaxed);
                    }
                    formed.push(Compile { entries, program });
                }
                // The store dropped the artifact: its entries are compiled
                // with the rest.
                None => compiling.extend(entries),
            }
        }
        if !compiling.is_empty() {
            let program = self.compile_stored(&key, source, &compiling)?;
            formed.push(Compile {
                entries: compiling,
                program,
            });
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
        Ok(Formed { key, compiles })
    }

    /// The artifact the store keeps under `kept`, which a stored pointer
    /// names; none when the store has dropped it, or keeps bytes that do
    /// not decode (damaged), which are reported and compiled again: the
    /// store is a cache, and the compile rewrites them.
    fn artifact(&self, kept: &ArtifactKey) -> Option<T::Artifact> {
        let store = self
            .store
            .as_deref()
            .expect("an artifact is named only by a store");
        let namespace = self.toolchain.identity().namespace;
        let bytes = store.get(namespace, kept)?;
        let artifact = T::Artifact::decode(&bytes);
        if artifact.is_none() {
            let kept = kept.as_str();
            eprintln!("seismic: the stored artifact {namespace}/{kept} does not decode; it is compiled again");
        }
        artifact
    }

    /// Compile `entries` of `source`'s text, store the artifact with each
    /// entry named, and return the program the compile formed.
    fn compile_stored(
        &self,
        key: &ArtifactKey,
        source: &ProgramSource,
        entries: &[ProgramEntry],
    ) -> Result<T::Program, NativeCompilationError> {
        let compiling = ProgramSource {
            text: source.text.clone(),
            entries: entries.to_vec(),
        };
        let began = std::time::Instant::now();
        let (artifact, program) = self.toolchain.compile(&compiling)?;
        let took = began.elapsed();
        self.compiles
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.compiled
            .lock()
            .expect("compile observation lock is never poisoned")
            .observe(entries.len(), took.as_secs_f64());
        if let Some(store) = self.store.as_deref() {
            let namespace = self.toolchain.identity().namespace;
            let kept = Self::artifact_key(key, &listed(entries));
            store.put(namespace, &kept, &artifact.encode());
            let mut stored = self.stored();
            for entry in entries {
                let entry_key = Self::entry_key(key, entry);
                store.put(namespace, &entry_key, kept.as_str().as_bytes());
                stored.insert(entry_key, Some(kept.clone()));
            }
        }
        Ok(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Compiles a source to its entries' symbols, its artifact their names
    /// joined, and loads the requested ones from an artifact that names
    /// them. It counts the calls that compiled and the entries they formed.
    struct Echo {
        identity: seismic_native_target::ToolchainIdentity,
        compiles: AtomicUsize,
        entries: AtomicUsize,
    }

    /// An artifact of entry names, joined.
    struct Names(Vec<String>);

    impl Artifact for Names {
        const ENCODING: u32 = 1;
        fn encode(&self) -> Vec<u8> {
            self.0.join(",").into_bytes()
        }
        fn decode(bytes: &[u8]) -> Option<Self> {
            Some(Self(
                std::str::from_utf8(bytes).ok()?.split(',').map(str::to_owned).collect(),
            ))
        }
    }

    impl Toolchain for Echo {
        type Artifact = Names;
        type Program = Vec<String>;
        const LOAD: Load = Load::Complete;
        fn identity(&self) -> &seismic_native_target::ToolchainIdentity {
            &self.identity
        }
        fn compile(
            &self,
            source: &ProgramSource,
        ) -> Result<(Names, Vec<String>), NativeCompilationError> {
            self.compiles.fetch_add(1, Ordering::SeqCst);
            self.entries.fetch_add(source.entries.len(), Ordering::SeqCst);
            let names = source.entries.iter().map(|entry| entry.symbol.clone()).collect::<Vec<_>>();
            Ok((Names(names.clone()), names))
        }
        fn load(
            &self,
            artifact: &Names,
            entries: &[ProgramEntry],
        ) -> Result<Vec<String>, NativeCompilationError> {
            entries
                .iter()
                .map(|entry| {
                    artifact
                        .0
                        .iter()
                        .find(|held| **held == entry.symbol)
                        .cloned()
                        .ok_or_else(|| {
                            NativeCompilationError::MalformedToolchainOutput(entry.symbol.clone())
                        })
                })
                .collect()
        }
        fn statistics(_: &Vec<String>, _: usize) -> &[seismic_native_target::ProgramStatistic] {
            &[]
        }
    }

    #[derive(Default)]
    struct Memory(Mutex<HashMap<(String, ArtifactKey), Vec<u8>>>);

    impl ArtifactStore for Memory {
        fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
            self.0
                .lock()
                .unwrap()
                .get(&(namespace.to_owned(), key.clone()))
                .cloned()
        }
        fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) {
            self.0
                .lock()
                .unwrap()
                .insert((namespace.to_owned(), key.clone()), bytes.to_vec());
        }
    }

    fn former(store: Option<Arc<Memory>>) -> ProgramFormer<Echo> {
        ProgramFormer::new(
            Echo {
                identity: seismic_native_target::ToolchainIdentity {
                    namespace: "echo",
                    material: "echo 1".into(),
                },
                compiles: AtomicUsize::new(0),
                entries: AtomicUsize::new(0),
            },
            store.map(|store| store as Arc<dyn ArtifactStore>),
        )
    }

    fn source(text: &str, symbols: &[&str]) -> ProgramSource {
        ProgramSource {
            text: text.into(),
            entries: symbols.iter().copied().map(ProgramEntry::named).collect(),
        }
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
    /// that holds one entry's compile does not keep another's.
    #[test]
    fn a_dropped_compile_is_formed_again_and_a_held_one_is_not() {
        let former = former(None);
        let first = former.form(&[source("launch", &["k<16>"])]).unwrap();
        let both = former
            .form(&[source("launch", &["k<16>", "k<32>"])])
            .unwrap();
        assert_eq!(compiles(&former), 2);
        // What a caller keeps of a formed source is the compile its entry
        // is in.
        let kept = both[0].entry(&ProgramEntry::named("k<16>")).0.clone();
        drop((first, both));
        former.form(&[source("launch", &["k<16>"])]).unwrap();
        assert_eq!(compiles(&former), 2);
        former.form(&[source("launch", &["k<32>"])]).unwrap();
        assert_eq!(compiles(&former), 3);
        drop(kept);
        former.form(&[source("launch", &["k<16>"])]).unwrap();
        assert_eq!(compiles(&former), 4);
    }

    /// A class is what is live and kept when it is asked, whoever changed
    /// them since it was last asked: a compile another caller held and
    /// gave up is compiled again, and what the store came to keep is kept,
    /// though the same class was read before either.
    #[test]
    fn a_class_read_again_follows_what_others_hold_and_the_store_keeps() {
        use FormationClass::{Compiled, Kept};
        let former = former(None);
        let wide = source("launch", &["k<16>", "k<32>"]);
        let held = former.form(&[wide.clone()]).unwrap();
        assert_eq!(class(&former, &[wide.clone()]), (Kept, 0));
        drop(held);
        assert_eq!(class(&former, &[wide.clone()]), (Compiled, 2));
        let former = self::former(Some(Arc::new(Memory::default())));
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
        let store = Arc::new(Memory::default());
        let first = former(Some(store.clone()));
        let _narrow = first.form(&[source("launch", &["k<8>"])]).unwrap();
        // The second call compiles only what the first did not.
        drop(
            first
                .form(&[source("launch", &["k<8>", "k<16>", "k<32>"])])
                .unwrap(),
        );
        assert_eq!((compiles(&first), entries(&first)), (2, 3));
        let fresh = former(Some(store));
        fresh.form(&[source("launch", &["k<32>"])]).unwrap();
        fresh.form(&[source("launch", &["k<8>"])]).unwrap();
        assert_eq!(compiles(&fresh), 0);
        fresh.form(&[source("other", &["k<32>"])]).unwrap();
        assert_eq!(compiles(&fresh), 1);
    }

    /// Only what the store does not return is a miss: an artifact it
    /// dropped is classed as compiled though a pointer names it, and is
    /// compiled again, with the entries no artifact holds, in one call; one
    /// it returns that does not decode is damaged, a miss too, and the
    /// compile rewrites it.
    #[test]
    fn only_a_dropped_artifact_is_a_miss() {
        let store = Arc::new(Memory::default());
        let first = former(Some(store.clone()));
        let narrow = source("launch", &["k<8>"]);
        drop(first.form(&[narrow.clone()]).unwrap());
        let key = first.key(&narrow);
        let stored = |entries: &[ProgramEntry]| {
            (
                String::from("echo"),
                ProgramFormer::<Echo>::artifact_key(&key, &listed(entries)),
            )
        };
        store.0.lock().unwrap().remove(&stored(&narrow.entries)).expect("the artifact was stored");
        let wide = source("launch", &["k<8>", "k<16>"]);
        let fresh = former(Some(store.clone()));
        assert_eq!(class(&fresh, &[narrow.clone()]), (FormationClass::Compiled, 1));
        drop(fresh.form(&[wide.clone()]).unwrap());
        assert_eq!((compiles(&fresh), entries(&fresh)), (1, 2));
        let (namespace, artifact) = stored(&wide.entries);
        store.0.lock().unwrap().insert((namespace, artifact), vec![0xff]);
        let damaged = former(Some(store.clone()));
        drop(damaged.form(&[narrow.clone()]).unwrap());
        assert_eq!(compiles(&damaged), 1, "an undecodable artifact is compiled again");
        let after = former(Some(store));
        assert_eq!(class(&after, &[narrow]), (FormationClass::Kept, 0), "and its entry's pointer rewritten");
    }

    /// A damaged entry pointer is a miss: the entry compiles again, and
    /// the compile rewrites the pointer, which a later former then follows.
    #[test]
    fn a_damaged_pointer_is_a_miss_and_is_rewritten() {
        let store = Arc::new(Memory::default());
        let first = former(Some(store.clone()));
        let narrow = source("launch", &["k<8>"]);
        drop(first.form(&[narrow.clone()]).unwrap());
        let pointer = (
            String::from("echo"),
            ProgramFormer::<Echo>::entry_key(&first.key(&narrow), &narrow.entries[0]),
        );
        store.0.lock().unwrap().insert(pointer.clone(), b"not a key".to_vec());
        let damaged = former(Some(store.clone()));
        assert_eq!(class(&damaged, &[narrow.clone()]), (FormationClass::Compiled, 1));
        drop(damaged.form(&[narrow.clone()]).unwrap());
        assert_eq!(compiles(&damaged), 1);
        let rewritten = store.0.lock().unwrap()[&pointer].clone();
        let parsed = std::str::from_utf8(&rewritten).ok().and_then(ArtifactKey::parse);
        assert!(parsed.is_some(), "the compile rewrote the pointer");
        let after = former(Some(store));
        assert_eq!(class(&after, &[narrow]), (FormationClass::Kept, 0));
    }

    /// What forming a source does is known before it is formed, and is
    /// what forming it then does: kept when live compiles or the store
    /// serve every entry, compiled otherwise.
    #[test]
    fn a_formation_is_classed_by_the_live_compiles_and_the_store_before_it_runs() {
        let store = Arc::new(Memory::default());
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
        // Without a store nothing is kept between processes.
        assert_eq!(class(&former(None), &[wide]), (Compiled, 2));
        assert_eq!(class(&former(None), &[]), (Kept, 0));
    }
}
