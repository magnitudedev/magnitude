//! Storage for compiled program artifacts.
//!
//! Seismic compiles rendered programs with a backend toolchain at run time.
//! A device keeps what it compiles in its [`ProgramStore`]; the device's
//! program former alone reads and writes its programs: each compile's
//! artifact under its content address, and for each entry a pointer naming
//! the artifact that holds it. Each toolchain keeps its artifacts in its own
//! namespace; the store knows no backends.
//!
//! Every device forms its programs through a [`ProgramStore`]: the store's
//! bytes ([`ArtifactStore`]) and, for every device opened with it, what is
//! being formed now. A program a device of the process is forming is waited
//! for by any other that asks for it, and then loaded, never formed twice.
//! A compile's artifact is made and stored off the path that formed it, on
//! the store's one thread, in turn with every other.
//!
//! Every process has one store, the user's ([`ProgramStore::user`]): a
//! [`DirectoryStore`] in the user's cache directory, which every process of
//! the user shares, so no process forms a program another formed. It holds
//! the programs, and the probe answers devices are opened with, which
//! eviction keeps. A device opened without naming a store forms through it;
//! a test that observes its own compiles names one of its own
//! ([`DeviceOptions::artifacts`]), in memory ([`MemoryStore`]) or in a
//! directory of its own. That store holds the device's programs only: the
//! probe answers a device opens with are the machine's, read from and kept
//! in the user's store whatever store its programs form through, so a
//! device's properties never depend on the store its opener names.

mod directory;

pub use directory::{DirectoryStore, Eviction, Layout, DEFAULT_CAPACITY};

pub use seismic_native_target::{ArtifactKey, ArtifactStore, MemoryStore};
use std::sync::Arc;

/// The store every device forms its programs through: `bytes`, where
/// artifacts and entry pointers are kept, and the formations its devices
/// have under way, so one program is formed once however many devices and
/// threads ask for it at once.
pub struct ProgramStore {
    bytes: Arc<dyn ArtifactStore>,
    /// The entry pointers formers are forming or storing now.
    state: std::sync::Mutex<Flight>,
    changed: std::sync::Condvar,
    /// Its stores under way, whichever former began them
    /// ([`ProgramStore::settle`]).
    stores: Stores,
    /// The one thread its stores run on, in turn, off every path that
    /// formed a program ([`Claim::store`]); started by the first store, and
    /// none where it would not start, each store then running on the
    /// former that compiled it. It ends with the store.
    worker: std::sync::OnceLock<Option<std::sync::mpsc::Sender<Store>>>,
}

/// One store a [`ProgramStore`]'s thread runs.
type Store = Box<dyn FnOnce() + Send>;

/// What a [`ProgramStore`] has in flight, by entry pointer key: a content
/// address that covers its toolchain's namespace, so keys of different
/// toolchains never meet.
#[derive(Default)]
struct Flight {
    /// Entries a former is forming: another former that misses one waits
    /// for it ([`ProgramStore::claim`]).
    forming: std::collections::HashSet<ArtifactKey>,
    /// Entries formed whose artifact is being made and stored, off the
    /// path that formed them ([`Claim::store`]): another former waits for
    /// it and loads it ([`Claim::settled`]).
    storing: std::collections::HashSet<ArtifactKey>,
    /// How many stores of each entry have ended in this process
    /// ([`ProgramStore::stored`]).
    stored: std::collections::HashMap<ArtifactKey, u64>,
}

/// A count of stores under way, which can be waited out: a store's
/// ([`ProgramStore::settle`]) and each former's own.
#[derive(Default)]
pub(crate) struct Stores {
    counted: std::sync::Mutex<Counted>,
    ended: std::sync::Condvar,
}

/// What a [`Stores`] counts.
#[derive(Default)]
struct Counted {
    under_way: usize,
    /// Why the first store counted here that panicked did: what it held is
    /// not stored, so no wait for it succeeds.
    panicked: Option<String>,
}

impl Stores {
    fn counted(&self) -> std::sync::MutexGuard<'_, Counted> {
        self.counted.lock().expect("no store panics counting")
    }

    fn begin(&self) {
        self.counted().under_way += 1;
    }

    /// A store ended, having panicked for `panicked` if it did.
    fn end(&self, panicked: Option<&str>) {
        let mut counted = self.counted();
        counted.under_way -= 1;
        if counted.panicked.is_none() {
            counted.panicked = panicked.map(str::to_owned);
        }
        if counted.under_way == 0 {
            self.ended.notify_all();
        }
    }

    /// Wait until no store counted here is under way. A store that panicked
    /// is that panic again here: what it held is not stored.
    pub(crate) fn settle(&self) {
        let counted = self.ended.wait_while(self.counted(), |counted| counted.under_way > 0).expect("no store panics counting");
        let panicked = counted.panicked.clone();
        drop(counted);
        if let Some(panicked) = panicked {
            panic!("a program store's store panicked, so what it held is not stored: {panicked}");
        }
    }
}

impl ProgramStore {
    /// A store keeping its bytes in `bytes`.
    pub fn new(bytes: Arc<dyn ArtifactStore>) -> Self {
        Self {
            bytes,
            state: Default::default(),
            changed: std::sync::Condvar::new(),
            stores: Stores::default(),
            worker: std::sync::OnceLock::new(),
        }
    }

    /// The user's store, the one store of the process, opened once: a
    /// directory store of [`DEFAULT_CAPACITY`] in the user's cache
    /// directory (`$XDG_CACHE_HOME`, else `~/Library/Caches` on macOS and
    /// `~/.cache` elsewhere, under `seismic/programs`), which every process
    /// of the user shares. A device opened without naming a store forms
    /// through it. Where the directory cannot open, it is the same one
    /// store with persistence unavailable: its bytes are held in memory, so
    /// every claim, wait and count works as on disk and each program forms
    /// once in the process, and every process compiles again until the
    /// directory opens. That is reported once a process, at error level.
    pub fn user() -> Arc<Self> {
        USER.0.clone()
    }

    /// How many stores of the entry pointer `key` have ended in this
    /// process: a miss read of it holds while the count is unchanged.
    pub(crate) fn stored(&self, key: &ArtifactKey) -> u64 {
        self.flight().stored.get(key).copied().unwrap_or(0)
    }

    fn flight(&self) -> std::sync::MutexGuard<'_, Flight> {
        self.state.lock().expect("no former panics holding the store's flight")
    }

    fn wait_while(&self, mut until: impl FnMut(&Flight) -> bool) -> std::sync::MutexGuard<'_, Flight> {
        self.changed
            .wait_while(self.flight(), |flight| until(flight))
            .expect("no former panics holding the store's flight")
    }

    /// Whether a former is forming or storing the entry pointer `key` now:
    /// forming it here would wait for it and load it, compiling nothing.
    pub(crate) fn in_flight(&self, key: &ArtifactKey) -> bool {
        let flight = self.flight();
        flight.forming.contains(key) || flight.storing.contains(key)
    }

    /// Claim the forming of the entry pointers `keys`: once no other former
    /// is forming any of them, all of them at once, so no claim waits while
    /// holding another. What returns is the claim, released when dropped;
    /// whether a waited-for key is now held is the caller's to read again,
    /// once what is still being stored of it is ([`Claim::settled`]).
    pub(crate) fn claim(self: &Arc<Self>, keys: &[ArtifactKey]) -> Claim {
        let mut flight = self.wait_while(|flight| keys.iter().any(|key| flight.forming.contains(key)));
        flight.forming.extend(keys.iter().cloned());
        Claim { store: self.clone(), keys: keys.to_vec() }
    }

    /// Wait until every store under way has ended: every program the
    /// store's devices compiled is then kept, for a later process to load.
    /// A device's former waits for its own stores when it drops; a process
    /// that ends without dropping its devices settles first, or leaves its
    /// last programs for a later process to compile again. A store that
    /// panicked panics here again, as at every wait for it.
    pub fn settle(&self) {
        self.stores.settle();
    }

    /// Run `store` on the store's thread, in turn with every other.
    fn run(&self, store: Store) {
        let worker = self.worker.get_or_init(|| {
            let (sender, stores) = std::sync::mpsc::channel::<Store>();
            match seismic_threads::spawn("seismic-store", move || stores.into_iter().for_each(|store| store()))
            {
                Ok(_) => Some(sender),
                Err(error) => {
                    tracing::warn!("the program store's thread does not start ({error}): each store runs on the former that compiled it");
                    None
                }
            }
        });
        match worker {
            Some(worker) => worker.send(store).expect("the program store's thread lives as long as the store: no store unwinds it"),
            None => store(),
        }
    }
}

/// The user's store ([`ProgramStore::user`]), and the directory store in
/// the user's cache directory that keeps its bytes, none where it did not
/// open and they are held in memory.
static USER: std::sync::LazyLock<(Arc<ProgramStore>, Option<Arc<DirectoryStore>>)> = std::sync::LazyLock::new(|| {
    match user_directory() {
        Ok(directory) => {
            let directory = Arc::new(directory);
            (Arc::new(ProgramStore::new(directory.clone())), Some(directory))
        }
        Err(reason) => {
            tracing::error!(
                "{reason}: the programs this process forms are kept in its memory alone, so every process compiles them again until the store opens"
            );
            (Arc::new(ProgramStore::new(Arc::new(MemoryStore::default()))), None)
        }
    }
});

/// The directory store that keeps the user's store on disk, none where it
/// did not open.
pub(crate) fn user_on_disk() -> Option<&'static DirectoryStore> {
    USER.1.as_deref()
}

/// The directory store in the user's cache directory ([`ProgramStore::user`]).
fn user_directory() -> Result<DirectoryStore, String> {
    let cache = match std::env::var_os("XDG_CACHE_HOME") {
        Some(cache) => std::path::PathBuf::from(cache),
        None => {
            let home = std::env::var_os("HOME")
                .ok_or("the user's program store has no directory to open: neither XDG_CACHE_HOME nor HOME is set")?;
            let home = std::path::PathBuf::from(home);
            if cfg!(target_os = "macos") { home.join("Library/Caches") } else { home.join(".cache") }
        }
    };
    let root = cache.join("seismic").join("programs");
    DirectoryStore::open(root.clone(), DEFAULT_CAPACITY, directory::programs)
        .map_err(|error| format!("the user's program store {} does not open: {error}", root.display()))
}

/// A former's claim on the formations it forms ([`ProgramStore::claim`]).
pub(crate) struct Claim {
    store: Arc<ProgramStore>,
    keys: Vec<ArtifactKey>,
}

impl Claim {
    /// Wait until no other former is storing a claimed key: what it stores
    /// is then read like anything else the store keeps.
    pub(crate) fn settled(&self) {
        let keys = &self.keys;
        drop(self.store.wait_while(|flight| keys.iter().any(|key| flight.storing.contains(key))));
    }

    /// Release the claim, and store what `keys` of it were formed of by
    /// running `store` on the store's thread, off the path that formed
    /// them, counted in `former`'s stores and the store's: until it has
    /// run, another former that misses one of `keys` waits for it and then
    /// loads it.
    pub(crate) fn store(self, keys: &[ArtifactKey], former: &Arc<Stores>, store: impl FnOnce() + Send + 'static) {
        let storing = self.keys.iter().filter(|key| keys.contains(key)).cloned().collect::<Vec<_>>();
        let program_store = self.store.clone();
        program_store.flight().storing.extend(storing.iter().cloned());
        program_store.stores.begin();
        former.begin();
        // The claim's drop releases its forming keys; those stored stay
        // in flight until the store ends.
        drop(self);
        let mut stored = Stored { store: program_store.clone(), keys: storing, former: former.clone(), panicked: None };
        // A store's panic ends as one on the former that compiled it would.
        let former_thread = seismic_threads::Submitter::here();
        program_store.run(Box::new(move || {
            // A store's panic it unwinds from ends the store, not the
            // thread: it is carried to every wait for it ([`Stores::settle`]).
            if let Err(panic) = former_thread.run(store) {
                stored.panicked = Some(panic.message().to_owned());
            }
        }));
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        let mut flight = self.store.flight();
        for key in &self.keys {
            flight.forming.remove(key);
        }
        drop(flight);
        self.store.changed.notify_all();
    }
}

/// A store under way ([`Claim::store`]): when it ends, however it ends, its
/// keys leave the flight, counted as stored, and its counts end, with its
/// panic where it panicked.
struct Stored {
    store: Arc<ProgramStore>,
    keys: Vec<ArtifactKey>,
    former: Arc<Stores>,
    panicked: Option<String>,
}

impl Drop for Stored {
    fn drop(&mut self) {
        let mut flight = self.store.flight();
        for key in self.keys.drain(..) {
            flight.storing.remove(&key);
            *flight.stored.entry(key).or_default() += 1;
        }
        drop(flight);
        self.store.changed.notify_all();
        self.former.end(self.panicked.as_deref());
        self.store.stores.end(self.panicked.as_deref());
    }
}

impl std::fmt::Debug for ProgramStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgramStore").finish_non_exhaustive()
    }
}

/// Its bytes: what the device's program formers keep their artifacts and
/// entry pointers in, and what probe code reads and keeps its answers in.
impl ArtifactStore for ProgramStore {
    fn get(&self, namespace: &str, key: &ArtifactKey) -> Option<Vec<u8>> {
        self.bytes.get(namespace, key)
    }

    fn contains(&self, namespace: &str, key: &ArtifactKey) -> bool {
        self.bytes.contains(namespace, key)
    }

    fn file(&self, namespace: &str, key: &ArtifactKey) -> Option<std::path::PathBuf> {
        self.bytes.file(namespace, key)
    }

    fn put(&self, namespace: &str, key: &ArtifactKey, bytes: &[u8]) {
        self.bytes.put(namespace, key, bytes);
    }
}

/// How a device is opened.
#[derive(Clone, Default)]
pub struct DeviceOptions {
    /// The store the device forms its programs through (its probe answers
    /// are always the user store's); none names the user's, the one store of the process
    /// ([`ProgramStore::user`]), and shares an open device whatever store
    /// it was opened with. A test that observes its own compiles names one
    /// of its own.
    pub artifacts: Option<Arc<ProgramStore>>,
    /// What its recorded graphs hold, as records hold it. Without one its
    /// backend measures it on opening, once a process.
    pub graph_cost: Option<seismic_compiler::graph_cost::GraphCost>,
}
