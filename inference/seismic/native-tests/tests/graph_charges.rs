//! What a graph's run is charged to, on a CUDA device opened in an
//! embedder's ledger, where no entry is ever charged by default: its
//! storage to the charges its slot, outputs and inputs were allocated
//! through, and the recording of its layout's graph on its first run to
//! the device stores' own entries, which no holder's allocation causes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use seismic::memory::{Change, Entry, Refused, Store, Tally, Wiring};
use seismic::{BackendName, Devices, Element, NativeGraphFamily, NativeSpecialization, StoreIdentity, Tensor};
use seismic_native_tests::scale_rows;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

/// An entry counting, besides what it holds, every byte ever charged to
/// it ([`Entry::take`], [`Entry::took`]): which entry a charge lands on,
/// even one carved from a block the same entry already holds (CUDA's).
struct Counted {
    tally: Tally,
    charged: AtomicU64,
}

impl Counted {
    fn new(identity: StoreIdentity) -> Arc<Self> {
        Arc::new(Self { tally: Tally::unbounded(identity), charged: AtomicU64::new(0) })
    }

    fn charged(&self) -> u64 {
        self.charged.load(Ordering::SeqCst)
    }
}

impl Entry for Counted {
    fn store(&self) -> &StoreIdentity {
        Entry::store(&self.tally)
    }

    fn change(&self) -> Change {
        self.tally.change()
    }

    fn on_close(&self, closed: Box<dyn FnOnce() + Send>) {
        self.tally.on_close(closed);
    }

    fn closed(&self) -> bool {
        self.tally.closed()
    }

    fn take(&self, change: &Change, bytes: u64, wiring: Wiring) -> Result<(), Refused> {
        self.tally.take(change, bytes, wiring)?;
        self.charged.fetch_add(bytes, Ordering::SeqCst);
        Ok(())
    }

    fn took(&self, change: &Change, bytes: u64, wiring: Wiring) {
        self.tally.took(change, bytes, wiring);
        self.charged.fetch_add(bytes, Ordering::SeqCst);
    }

    fn freed(&self, change: &Change, bytes: u64, wiring: Wiring) {
        self.tally.freed(change, bytes, wiring);
    }
}

/// An embedder's ledger of one store: the store's own entry and one
/// holder's, each counted apart. Not a tally ([`Store::tally`]), so the
/// device opened in it has no tally charges to fall back on.
struct Ledger {
    identity: StoreIdentity,
    own: Arc<Counted>,
    holder: Arc<Counted>,
}

impl Ledger {
    fn new(identity: StoreIdentity) -> Self {
        Self { own: Counted::new(identity.clone()), holder: Counted::new(identity.clone()), identity }
    }
}

impl Store for Ledger {
    fn identity(&self) -> &StoreIdentity {
        &self.identity
    }

    fn allocated(&self) -> u64 {
        self.own.tally.held() + self.holder.tally.held()
    }

    fn room(&self) -> u64 {
        u64::MAX - self.allocated()
    }

    fn high_water(&self) -> u64 {
        Store::high_water(&self.own.tally).max(Store::high_water(&self.holder.tally))
    }

    fn restart_high_water(&self) {
        Store::restart_high_water(&self.own.tally);
        Store::restart_high_water(&self.holder.tally);
    }

    fn own(&self) -> Arc<dyn Entry> {
        self.own.clone()
    }
}

/// A holder's slot runs a graph whose layout no run has recorded: the run
/// charges the holder's entries nothing (they hold exactly what its slot,
/// outputs and source took), the recording's placeholder is charged to the
/// device store's own entry, and the holder's entries are back at none
/// once what it allocated is dropped.
#[test]
fn a_graphs_first_run_records_on_the_devices_own_entries() {
    let (m, n) = (2u64, 8u64);
    let devices = Devices::discover(&[]).expect("test device discovery");
    let Some(id) = seismic::testing::selected(&devices.topology(), BackendName::Cuda).map(|executor| executor.id()) else {
        eprintln!("no CUDA device: only CUDA records a layout's graph");
        return;
    };
    let ledgers = devices.reached_stores(id).into_iter().map(|identity| Arc::new(Ledger::new(identity))).collect::<Vec<_>>();
    let stores = ledgers.iter().map(|ledger| ledger.clone() as Arc<dyn Store>).collect();
    let device = devices.open_in(id, Default::default(), stores).expect("the CUDA device opens in the embedder's ledger");
    let ledger_of = |identity: &StoreIdentity| {
        ledgers.iter().find(|ledger| ledger.identity == *identity).expect("a ledger of every store the device reaches")
    };
    let holders = || ledgers.iter().map(|ledger| ledger.holder.tally.held()).sum::<u64>();
    let holders_charged = || ledgers.iter().map(|ledger| ledger.holder.charged()).sum::<u64>();
    let device_ledger = ledger_of(&device.topology().store_identity(device.device_store()).unwrap());

    let kernel = scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1)).unwrap();
    let mut graph = device.native_graph();
    let input = graph.port(Element::f32(), &[m, n]).unwrap();
    let scaled = graph
        .enqueue(&kernel, scale_rows::WorkflowArgs { x: input.tensor().into(), factor: seismic::Argument::from(2.0) })
        .unwrap()
        .value;
    graph.export(&scaled).unwrap();
    let plan = graph.seal().unwrap();
    let family = NativeGraphFamily::new(&[plan.clone()], 1).unwrap();

    // The holder's entry of each store, minted by the embedder's ledger.
    let charges = device.charges(|store| ledger_of(store).holder.clone() as Arc<dyn Entry>);
    let mut slot = family.new_slot(&charges, &[family.upload_bytes()]).unwrap();
    let outputs = family.new_output_slot(&charges).unwrap().activate(&plan).unwrap();
    let values = (0..m * n).map(|value| value as f32).collect::<Vec<_>>();
    let source = Tensor::from_host(&charges, Element::f32(), &[m, n], &f32_bytes(&values)).unwrap();
    let held = holders();
    assert!(held > 0, "the slot, outputs and source are the holder's");

    let (holders_before, own_before) = (holders_charged(), device_ledger.own.charged());
    let mut bindings = plan.bindings();
    bindings.set(&charges, &input, &source).unwrap();
    let mut active = slot.activate(&plan).unwrap();
    let (outputs, completion) = active.attach(bindings, outputs).unwrap().submit().unwrap();
    completion.wait().unwrap();
    assert_eq!((holders_charged(), holders()), (holders_before, held), "the first run's recording charges no holder");
    assert!(device_ledger.own.charged() > own_before, "the recording's placeholder is charged to the device store's own entry");
    assert_eq!(
        read_f32(&outputs.exported(&scaled).unwrap()),
        values.iter().map(|value| 2.0 * value).collect::<Vec<_>>(),
        "the recorded run's result"
    );

    drop((outputs, source, slot));
    assert_eq!(holders(), 0, "the holder's storage returns as it drops");
}
