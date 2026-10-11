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
//! The store keeps what a toolchain kept of each compile under the text and
//! the entry list compiled, and for each text the entry lists kept, so that
//! a later request is served from a kept compile that holds its entries,
//! whichever call formed it.

use crate::artifacts::{ArtifactKey, ArtifactStore};
use crate::native::tune::Linear;
use seismic_native_target::workers::Workers;
use seismic_native_target::{
    NativeCompilationError, ProgramCache, ProgramEntry, ProgramSource, Toolchain,
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

/// What forming a program does, known before it is formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FormationClass {
    /// No compiler of the toolchain's runs: compiles live on the device
    /// hold every requested entry, or the store keeps what the toolchain
    /// says serves those they lack ([`Toolchain::keeps`]).
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
    /// store) rather than found live.
    formed: std::sync::atomic::AtomicUsize,
    /// Held while a text's kept entry lists are read and rewritten.
    listing: Mutex<()>,
    /// What this toolchain's compiles have taken in this process, each with
    /// the entries it formed: what a compile of a source not yet compiled
    /// here is priced by.
    compiled: Mutex<Linear>,
    /// The threads its owner lent for work off the calling thread
    /// ([`ProgramFormer::staff`]); without them every source is formed in
    /// turn on the thread that asks.
    workers: Mutex<Option<Arc<Workers>>>,
}

/// The store's bytes for one key in one toolchain's namespace.
struct StoreCache<'a> {
    store: &'a dyn ArtifactStore,
    namespace: &'a str,
    key: &'a ArtifactKey,
}

impl ProgramCache for StoreCache<'_> {
    fn get(&self) -> Option<Vec<u8>> {
        self.store.get(self.namespace, self.key)
    }

    fn put(&self, bytes: &[u8]) {
        self.store.put(self.namespace, self.key, bytes);
    }
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
            listing: Mutex::new(()),
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
    pub(crate) fn compiled(&self) -> Linear {
        *self
            .compiled
            .lock()
            .expect("compile observation lock is never poisoned")
    }

    /// How many compiles the toolchain has formed for this device.
    pub(crate) fn formed(&self) -> usize {
        self.formed.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn toolchain(&self) -> &T {
        &self.toolchain
    }

    /// The content address of `source`'s text under this toolchain.
    fn key(&self, source: &ProgramSource) -> ArtifactKey {
        let identity = self.toolchain.identity();
        ArtifactKey::of(&[
            identity.namespace.as_bytes(),
            identity.material.as_bytes(),
            source.text.as_bytes(),
        ])
    }

    /// Where the store keeps a compile of the text keyed `key` with the
    /// entry list `list`.
    fn kept_key(key: &ArtifactKey, list: &str) -> ArtifactKey {
        ArtifactKey::of(&[b"compile", key.as_str().as_bytes(), list.as_bytes()])
    }

    /// Where the store keeps the entry lists kept of the text keyed `key`,
    /// a line each.
    fn lists_key(key: &ArtifactKey) -> ArtifactKey {
        ArtifactKey::of(&[b"entry lists", key.as_str().as_bytes()])
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

    /// Where the store keeps a compile that serves `source`, the entries of
    /// a text keyed `key` that no live compile holds: the compile of
    /// exactly that entry list, or one of a list that holds them all, as
    /// the toolchain says of the bytes kept ([`Toolchain::keeps`]).
    fn kept(&self, key: &ArtifactKey, source: &ProgramSource) -> Option<ArtifactKey> {
        let store = self.store.as_deref()?;
        let namespace = self.toolchain.identity().namespace;
        let keeps = |kept_key: &ArtifactKey| {
            store
                .get(namespace, kept_key)
                .is_some_and(|kept| self.toolchain.keeps(source, &kept))
        };
        let exact = Self::kept_key(key, &listed(&source.entries));
        if keeps(&exact) {
            return Some(exact);
        }
        let lists = store.get(namespace, &Self::lists_key(key))?;
        let wanted = source
            .entries
            .iter()
            .map(|entry| listed(std::slice::from_ref(entry)))
            .collect::<Vec<_>>();
        std::str::from_utf8(&lists)
            .ok()?
            .lines()
            .filter(|list| {
                wanted
                    .iter()
                    .all(|entry| list.split('\t').any(|kept| kept == entry))
            })
            .map(|list| Self::kept_key(key, list))
            .find(keeps)
    }

    /// What forming `sources` together does: compiled when the toolchain
    /// compiles any of them, and then the most entries one source's
    /// compile forms (the sources compile at once, so the call takes what
    /// its largest compile does). It reads the device's live compiles and
    /// the store and forms nothing.
    pub(crate) fn class(&self, sources: &[ProgramSource]) -> (FormationClass, usize) {
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
    /// compile holds, unless the store keeps what serves them.
    fn compiling(&self, source: &ProgramSource) -> usize {
        let key = self.key(source);
        let (_, missing) = self.held(&key, &source.entries);
        if missing.is_empty() {
            return 0;
        }
        let missing = ProgramSource {
            text: source.text.clone(),
            entries: missing,
        };
        match self.kept(&key, &missing) {
            Some(_) => 0,
            None => missing.entries.len(),
        }
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

    fn form_one(
        &self,
        source: &ProgramSource,
    ) -> Result<Formed<T::Program>, NativeCompilationError> {
        let key = self.key(source);
        let (mut compiles, missing) = self.held(&key, &source.entries);
        if missing.is_empty() {
            return Ok(Formed { key, compiles });
        }
        // One call of the toolchain for everything missing, and nothing
        // that is held.
        let missing = ProgramSource {
            text: source.text.clone(),
            entries: missing,
        };
        let kept = self.kept(&key, &missing);
        let list = listed(&missing.entries);
        let new = kept.is_none();
        let kept_key = kept.unwrap_or_else(|| Self::kept_key(&key, &list));
        let cache = self.store.as_deref().map(|store| StoreCache {
            store,
            namespace: self.toolchain.identity().namespace,
            key: &kept_key,
        });
        self.formed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let began = std::time::Instant::now();
        let program = self.toolchain.compile(
            &missing,
            cache.as_ref().map(|cache| cache as &dyn ProgramCache),
        )?;
        if new {
            self.compiled
                .lock()
                .expect("compile observation lock is never poisoned")
                .observe(missing.entries.len(), began.elapsed().as_secs_f64());
            if let Some(store) = self.store.as_deref() {
                self.list(store, &key, &list);
            }
        }
        let compile = Arc::new(Compile {
            entries: missing.entries,
            program,
        });
        let mut live = self
            .live
            .lock()
            .expect("live program lock is never poisoned");
        live.retain(|_, compiles| {
            compiles.retain(|compile| compile.strong_count() > 0);
            !compiles.is_empty()
        });
        live.entry(key.clone())
            .or_default()
            .push(Arc::downgrade(&compile));
        drop(live);
        compiles.push(compile);
        Ok(Formed { key, compiles })
    }

    /// How many entries of `source` publishing it compiles
    /// ([`ProgramFormer::publish`]): all of them, unless the store keeps a
    /// compile of exactly its entry list; none without a store, which
    /// keeps nothing for a later process to find.
    fn publishing(&self, source: &ProgramSource) -> usize {
        let Some(store) = self.store.as_deref() else {
            return 0;
        };
        let exact = Self::kept_key(&self.key(source), &listed(&source.entries));
        let kept = store
            .get(self.toolchain.identity().namespace, &exact)
            .is_some_and(|kept| self.toolchain.keeps(source, &kept));
        if kept { 0 } else { source.entries.len() }
    }

    /// What publishing `sources` together does ([`ProgramFormer::publish`]),
    /// known before it runs: compiled when the store lacks a compile of
    /// exactly one of their entry lists, and then the most entries one
    /// source's compile forms (the sources compile at once). A choice whose
    /// programs the store keeps as a load requests them is published by
    /// reading that, and no compiler runs.
    pub(crate) fn publication(&self, sources: &[ProgramSource]) -> (FormationClass, usize) {
        let entries = sources
            .iter()
            .map(|source| self.publishing(source))
            .max()
            .unwrap_or(0);
        match entries {
            0 => (FormationClass::Kept, 0),
            entries => (FormationClass::Compiled, entries),
        }
    }

    /// How many of `sources` publishing them compiles
    /// ([`ProgramFormer::publish`]): each takes a thread of the former's
    /// while it does, so with what [`ProgramFormer::publication`] says of
    /// their size this is what a publication's time is observed against.
    pub(crate) fn publishing_sources(&self, sources: &[ProgramSource]) -> usize {
        sources
            .iter()
            .filter(|source| self.publishing(source) > 0)
            .count()
    }

    /// Publish each source as a later process will request it: the store
    /// keeps a compile of exactly the source's entry list, whatever
    /// compiles live on the device hold of its entries, and nothing formed
    /// here is held. A choice tuning formed among many variants of its
    /// text is served to this process out of that compile; the process
    /// that loads the choice asks for its entries alone, and what it asks
    /// for is what was published. The sources are compiled at once on the
    /// threads lent to the former and in turn without them.
    pub(crate) fn publish(&self, sources: &[ProgramSource]) -> Result<(), NativeCompilationError>
    where
        T: Sync,
    {
        match self.workers() {
            Some(workers) => workers
                .each(sources, |source| self.publish_one(source))
                .into_iter()
                .collect(),
            None => sources
                .iter()
                .try_for_each(|source| self.publish_one(source)),
        }
    }

    fn publish_one(&self, source: &ProgramSource) -> Result<(), NativeCompilationError> {
        if self.publishing(source) == 0 {
            return Ok(());
        }
        let Some(store) = self.store.as_deref() else {
            return Ok(());
        };
        let key = self.key(source);
        let list = listed(&source.entries);
        let kept_key = Self::kept_key(&key, &list);
        let cache = StoreCache {
            store,
            namespace: self.toolchain.identity().namespace,
            key: &kept_key,
        };
        self.formed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let began = std::time::Instant::now();
        // The program is released as soon as it is formed: what is
        // published is what the toolchain kept of it.
        drop(self.toolchain.compile(source, Some(&cache))?);
        self.compiled
            .lock()
            .expect("compile observation lock is never poisoned")
            .observe(source.entries.len(), began.elapsed().as_secs_f64());
        self.list(store, &key, &list);
        Ok(())
    }

    /// Record that the store keeps a compile of the text keyed `key` with
    /// the entry list `list`.
    fn list(&self, store: &dyn ArtifactStore, key: &ArtifactKey, list: &str) {
        let _listing = self
            .listing
            .lock()
            .expect("entry list lock is never poisoned");
        let namespace = self.toolchain.identity().namespace;
        let lists_key = Self::lists_key(key);
        let mut lists = store
            .get(namespace, &lists_key)
            .and_then(|lists| String::from_utf8(lists).ok())
            .unwrap_or_default();
        if !lists.lines().any(|kept| kept == list) {
            lists.push_str(list);
            lists.push('\n');
            store.put(namespace, &lists_key, lists.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Compiles a source to its entries' symbols, keeping them in the cache
    /// and answering from it when it holds every requested symbol. It
    /// counts the calls that compiled and the entries they formed.
    struct Echo {
        identity: seismic_native_target::ToolchainIdentity,
        compiles: AtomicUsize,
        entries: AtomicUsize,
    }

    impl Toolchain for Echo {
        type Program = Vec<String>;
        fn identity(&self) -> &seismic_native_target::ToolchainIdentity {
            &self.identity
        }
        fn keeps(&self, source: &ProgramSource, kept: &[u8]) -> bool {
            let kept = std::str::from_utf8(kept).unwrap();
            source
                .entries
                .iter()
                .all(|entry| kept.split(',').any(|held| held == entry.symbol))
        }
        fn compile(
            &self,
            source: &ProgramSource,
            cache: Option<&dyn ProgramCache>,
        ) -> Result<Vec<String>, NativeCompilationError> {
            let symbols = source
                .entries
                .iter()
                .map(|entry| entry.symbol.clone())
                .collect::<Vec<_>>();
            if cache
                .and_then(|cache| cache.get())
                .is_some_and(|kept| self.keeps(source, &kept))
            {
                return Ok(symbols);
            }
            self.compiles.fetch_add(1, Ordering::SeqCst);
            self.entries.fetch_add(symbols.len(), Ordering::SeqCst);
            if let Some(cache) = cache {
                cache.put(symbols.join(",").as_bytes());
            }
            Ok(symbols)
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

    /// Another process is served from whichever kept compile of the same
    /// text holds what it asks for.
    #[test]
    fn the_toolchain_is_given_the_stores_bytes_of_a_compile_that_holds_the_entries() {
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

    /// A choice formed among many variants of its text is published as a
    /// later process requests it: a compile of its entry list alone, kept
    /// under exactly that list, formed once and held by no one. The later
    /// process finds it as it finds any compile of exactly its request.
    #[test]
    fn a_published_source_is_kept_as_a_later_process_requests_it() {
        let store = Arc::new(Memory::default());
        let first = former(Some(store.clone()));
        // Tuning's compile of many variants is live and holds the choice.
        let many = first
            .form(&[source("launch", &["k<8>", "k<16>", "k<32>"])])
            .unwrap();
        let chosen = source("launch", &["k<16>"]);
        use FormationClass::Kept;
        assert_eq!(first.class(&[chosen.clone()]), (Kept, 0));
        assert_eq!(first.publishing(&chosen), 1);
        first.publish(&[chosen.clone()]).unwrap();
        // One more call of the toolchain, for the one entry, though a live
        // compile holds it; nothing more of it is live.
        assert_eq!((compiles(&first), entries(&first)), (2, 4));
        assert_eq!(first.publishing(&chosen), 0);
        first.publish(&[chosen.clone()]).unwrap();
        assert_eq!(compiles(&first), 2);
        drop(many);
        // The store keeps the chosen entry's own compile: what the
        // toolchain kept under exactly that list holds it and no other.
        let namespace = "echo";
        let key = first.key(&chosen);
        let exact = ProgramFormer::<Echo>::kept_key(&key, &listed(&chosen.entries));
        assert_eq!(store.get(namespace, &exact).unwrap(), b"k<16>");
        // A later process asking for the choice is given that compile.
        let later = former(Some(store));
        assert_eq!(later.class(&[chosen.clone()]), (Kept, 0));
        later.form(&[chosen]).unwrap();
        assert_eq!(compiles(&later), 0);
        // Without a store there is nothing to publish to.
        let alone = former(None);
        let source = source("launch", &["k<16>"]);
        assert_eq!(alone.publishing(&source), 0);
        alone.publish(&[source]).unwrap();
        assert_eq!(compiles(&alone), 0);
    }

    /// What publishing a choice does is known before it runs, and is what
    /// publishing it then does: a choice the store keeps as a load requests
    /// it (the canonical default the load formed alone, a choice already
    /// published) runs no compiler, and only one the store lacks under
    /// exactly its entry list is a compile, of that list. What a unit owes
    /// for its served choice's artifact is priced by this.
    #[test]
    fn a_publication_is_classed_by_the_store_before_it_runs() {
        use FormationClass::{Compiled, Kept};
        let store = Arc::new(Memory::default());
        let former = former(Some(store));
        // The canonical default, formed as a load requests it.
        let default = source("launch", &["k<8>"]);
        let _canonical = former.form(&[default.clone()]).unwrap();
        assert_eq!(former.publication(&[default.clone()]), (Kept, 0));
        let before = compiles(&former);
        former.publish(&[default.clone()]).unwrap();
        assert_eq!(compiles(&former), before);
        // A choice formed among many variants is live, and the store lacks
        // it as a load requests it: its publication compiles its one entry.
        let _many = former
            .form(&[source("launch", &["k<8>", "k<16>", "k<32>"])])
            .unwrap();
        let chosen = source("launch", &["k<16>"]);
        assert_eq!(former.class(&[chosen.clone()]), (Kept, 0));
        assert_eq!(former.publication(&[chosen.clone()]), (Compiled, 1));
        // Sources are published at once: the call is its largest compile.
        let other = source("other", &["a", "b"]);
        assert_eq!(
            former.publication(&[default, chosen.clone(), other]),
            (Compiled, 2)
        );
        let before = compiles(&former);
        former.publish(&[chosen.clone()]).unwrap();
        assert_eq!(compiles(&former), before + 1);
        assert_eq!(former.publication(&[chosen]), (Kept, 0));
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
        assert_eq!(first.class(&[wide.clone()]), (Compiled, 2));
        let held = first.form(&[wide.clone()]).unwrap();
        assert_eq!(compiles(&first), 1);
        // What the toolchain's compiles took is observed with what they
        // formed.
        assert!(first.compiled().price(2).is_some());
        // Live on the device, and kept by the store once dropped.
        assert_eq!(first.class(&[wide.clone()]), (Kept, 0));
        // A wider request compiles only what is not live.
        assert_eq!(
            first.class(&[source("launch", &["k<16>", "k<64>"])]),
            (Compiled, 1)
        );
        drop(held);
        assert_eq!(first.class(&[wide.clone()]), (Kept, 0));
        // Another process with the same store: what the store keeps is
        // loaded, an entry it lacks is compiled, and a call forming both
        // compiles.
        let fresh = former(Some(store));
        let narrow = source("launch", &["k<32>"]);
        let lacking = source("launch", &["k<64>"]);
        assert_eq!(fresh.class(&[narrow.clone()]), (Kept, 0));
        assert_eq!(fresh.class(&[lacking.clone()]), (Compiled, 1));
        assert_eq!(
            fresh.class(&[narrow.clone(), source("other", &["k"])]),
            (Compiled, 1)
        );
        fresh.form(&[narrow]).unwrap();
        assert_eq!(compiles(&fresh), 0);
        fresh.form(&[lacking]).unwrap();
        assert_eq!(compiles(&fresh), 1);
        // Without a store nothing is kept between processes.
        assert_eq!(former(None).class(&[wide]), (Compiled, 2));
        assert_eq!(former(None).class(&[]), (Kept, 0));
    }
}
