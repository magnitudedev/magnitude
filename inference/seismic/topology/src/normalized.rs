//! Revision-independent material for hardware fingerprints. Stable native
//! selectors and store identities replace every revision-local identifier.

use crate::{AccessPath, DeviceTopology, ExecutorId, StorageKind, StoreId};
use serde_json::{json, Value};

impl DeviceTopology {
    /// Canonical JSON of the reported graph, independent of revision and
    /// discovery order. Status has its own lifetime.
    pub fn normalized(&self) -> String {
        let store = |id: StoreId| {
            self.store_identity(id)
                .expect("a store of this revision has an identity")
        };
        let selector = |id: ExecutorId| {
            self.executor(id)
                .expect("a path's executor is of its revision")
                .selector()
        };
        let access_key = |path: &AccessPath| {
            json!({
                "executor": selector(path.executor()),
                "store": store(path.store()),
                "route": path.route(),
            })
        };

        let mut physical = self
            .physical_devices()
            .iter()
            .map(|device| {
                json!({
                    "identity": device.identity(),
                    "memory": device.memory(),
                    "cache": device.cache(),
                })
            })
            .collect::<Vec<_>>();
        physical.sort_by_key(Value::to_string);

        let mut stores = self
            .stores()
            .iter()
            .map(|each| {
                json!({
                    "identity": store(each.id()),
                    "capacity": each.capacity(),
                })
            })
            .collect::<Vec<_>>();
        stores.sort_by_key(Value::to_string);

        let mut executors = self
            .executors()
            .iter()
            .map(|executor| {
                let physical = self
                    .physical_device(executor.physical())
                    .expect("an executor's device is of its revision");
                let placements = StorageKind::ALL
                    .iter()
                    .map(|&kind| {
                        executor
                            .placement(kind)
                            .iter()
                            .map(|&id| {
                                access_key(
                                    self.access_path(id)
                                        .expect("a placement names paths of its revision"),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                json!({
                    "selector": executor.selector(),
                    "physical": physical.identity(),
                    "name": executor.name(),
                    "kind": executor.kind(),
                    "driver": executor.driver(),
                    "availability": executor.availability(),
                    "max_allocation": executor.limits().max_allocation,
                    "tensor_operations": executor.forms_tensor_operations(),
                    "placements": placements,
                })
            })
            .collect::<Vec<_>>();
        executors.sort_by_key(Value::to_string);

        let mut access = self
            .access_paths()
            .iter()
            .map(|path| {
                json!({
                    "path": access_key(path),
                    "grant": path.grant(),
                })
            })
            .collect::<Vec<_>>();
        access.sort_by_key(Value::to_string);

        let mut transfer = self
            .transfer_paths()
            .iter()
            .map(|path| {
                json!({
                    "executor": selector(path.by()),
                    "from": store(path.from()),
                    "to": store(path.to()),
                })
            })
            .collect::<Vec<_>>();
        transfer.sort_by_key(Value::to_string);

        json!({
            "physical": physical,
            "stores": stores,
            "executors": executors,
            "access": access,
            "transfer": transfer,
        })
        .to_string()
    }
}
