//! The program former: the one path from rendered source to an executable
//! program on an opened device.
//!
//! Callers render [`ProgramSource`]s; the former keys each by its toolchain
//! and text, serves it from a program already live on the device that holds
//! every requested entry, or has the toolchain compile it with the store's
//! cache for that key. Variants of a templated kernel share one text, so a
//! program compiled with every variant serves each of them.

use crate::artifacts::{ArtifactKey, ArtifactStore};
use seismic_native_target::{
    NativeCompilationError, ProgramCache, ProgramEntry, ProgramSource, Toolchain,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// An executable program, the key it was formed under and the entries it
/// holds, in the program's order.
pub struct Formed<P> {
    key: ArtifactKey,
    entries: Vec<ProgramEntry>,
    program: P,
}

impl<P> Formed<P> {
    pub fn key(&self) -> &ArtifactKey {
        &self.key
    }

    pub fn program(&self) -> &P {
        &self.program
    }

    pub fn entries(&self) -> &[ProgramEntry] {
        &self.entries
    }

    /// The position of `entry` in the program.
    pub fn entry(&self, entry: &ProgramEntry) -> usize {
        self.entries
            .iter()
            .position(|held| held == entry)
            .expect("a formed program holds every entry it was requested for")
    }

    fn holds(&self, entries: &[ProgramEntry]) -> bool {
        entries.iter().all(|entry| self.entries.contains(entry))
    }
}

pub(crate) struct ProgramFormer<T: Toolchain> {
    toolchain: T,
    store: Option<Arc<dyn ArtifactStore>>,
    /// Programs live on the device, by key. Weak: a program lives as long
    /// as some caller holds it.
    live: Mutex<HashMap<ArtifactKey, Weak<Formed<T::Program>>>>,
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

impl<T: Toolchain> ProgramFormer<T> {
    pub(crate) fn new(toolchain: T, store: Option<Arc<dyn ArtifactStore>>) -> Self {
        Self {
            toolchain,
            store,
            live: Mutex::new(HashMap::new()),
        }
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

    /// Form every source, compiling concurrently. Programs are returned in
    /// source order.
    pub(crate) fn form(
        &self,
        sources: &[ProgramSource],
    ) -> Result<Vec<Arc<Formed<T::Program>>>, NativeCompilationError> {
        if sources.len() <= 1 {
            return sources.iter().map(|source| self.form_one(source)).collect();
        }
        let workers = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
            .min(sources.len());
        let chunk = sources.len().div_ceil(workers);
        std::thread::scope(|scope| {
            sources
                .chunks(chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|source| self.form_one(source))
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|worker| worker.join().expect("program formation thread panicked"))
                .collect()
        })
    }

    fn form_one(
        &self,
        source: &ProgramSource,
    ) -> Result<Arc<Formed<T::Program>>, NativeCompilationError> {
        let key = self.key(source);
        let live = self
            .live
            .lock()
            .expect("live program lock is never poisoned")
            .get(&key)
            .and_then(Weak::upgrade);
        if let Some(formed) = live.filter(|formed| formed.holds(&source.entries)) {
            return Ok(formed);
        }
        let cache = self.store.as_deref().map(|store| StoreCache {
            store,
            namespace: self.toolchain.identity().namespace,
            key: &key,
        });
        let program = self.toolchain.compile(
            source,
            cache.as_ref().map(|cache| cache as &dyn ProgramCache),
        )?;
        let formed = Arc::new(Formed {
            key: key.clone(),
            entries: source.entries.clone(),
            program,
        });
        let mut live = self
            .live
            .lock()
            .expect("live program lock is never poisoned");
        live.retain(|_, program| program.strong_count() > 0);
        live.insert(key, Arc::downgrade(&formed));
        Ok(formed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Compiles a source to its entries' symbols, keeping them in the cache
    /// and answering from it when it holds every requested symbol.
    struct Echo {
        identity: seismic_native_target::ToolchainIdentity,
        compiles: AtomicUsize,
    }

    impl Toolchain for Echo {
        type Program = Vec<String>;
        fn identity(&self) -> &seismic_native_target::ToolchainIdentity {
            &self.identity
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
            let kept = cache
                .and_then(|cache| cache.get())
                .map(|bytes| String::from_utf8(bytes).unwrap());
            if kept.is_some_and(|kept| {
                symbols
                    .iter()
                    .all(|symbol| kept.split(',').any(|held| held == symbol))
            }) {
                return Ok(symbols);
            }
            self.compiles.fetch_add(1, Ordering::SeqCst);
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

    #[test]
    fn a_live_program_serves_every_entry_it_holds() {
        let former = former(None);
        let all = former
            .form(&[source("launch", &["k<16>", "k<32>"])])
            .unwrap();
        let one = former.form(&[source("launch", &["k<32>"])]).unwrap();
        assert!(Arc::ptr_eq(&all[0], &one[0]));
        assert_eq!(one[0].entry(&ProgramEntry::named("k<32>")), 1);
        assert_eq!(compiles(&former), 1);
    }

    #[test]
    fn an_entry_the_live_program_lacks_is_compiled() {
        let former = former(None);
        let _held = former.form(&[source("launch", &["k<16>"])]).unwrap();
        former.form(&[source("launch", &["k<32>"])]).unwrap();
        assert_eq!(compiles(&former), 2);
    }

    #[test]
    fn a_dropped_program_is_formed_again() {
        let former = former(None);
        drop(former.form(&[source("launch", &["k"])]).unwrap());
        former.form(&[source("launch", &["k"])]).unwrap();
        assert_eq!(compiles(&former), 2);
    }

    #[test]
    fn the_toolchain_is_given_the_stores_bytes_for_the_same_text() {
        let store = Arc::new(Memory::default());
        drop(
            former(Some(store.clone()))
                .form(&[source("launch", &["k<16>", "k<32>"])])
                .unwrap(),
        );
        // Another process: one variant of the same text comes from the store.
        let fresh = former(Some(store));
        fresh.form(&[source("launch", &["k<32>"])]).unwrap();
        fresh.form(&[source("other", &["k<32>"])]).unwrap();
        assert_eq!(compiles(&fresh), 1);
    }
}
