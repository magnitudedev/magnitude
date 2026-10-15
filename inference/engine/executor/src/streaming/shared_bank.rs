//! The streamed experts' bank: layers whose routed weights share one complete
//! resident encoding form a group whose records every member layer binds
//! (group-global slot indices), planned before the device opens and allocated
//! on the first binding, one contiguous tensor per routed component.
use crate::planning::{ExpertBankPlan, BANK_KINDS};
use magnitude_family_contracts::WeightRole;
use seismic::{Device, Element, Tensor};
use std::collections::{HashMap, HashSet};

type Result<T> = std::result::Result<T, String>;
fn checked<T, E: std::fmt::Display>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| e.to_string())
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Encoding {
    elements: [Element; 3],
    matrices: [[u64; 2]; 3],
}
impl Encoding {
    /// Component `kind`'s extents over `records` bank slots.
    fn extents(&self, kind: usize, records: u64) -> [u64; 3] {
        [records, self.matrices[kind][0], self.matrices[kind][1]]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SharedGroupPlan {
    encoding: Encoding,
    pub layers: Vec<usize>,
    pub records: u64,
    pub allocated_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SharedBankPlan {
    pub groups: Vec<SharedGroupPlan>,
    pub group_of_layer: Vec<usize>,
    pub allocated_bytes: u64,
}
impl SharedBankPlan {
    /// Group by the complete resident representation/layout and matrix
    /// geometry. Stored source formats may differ; byte-pitch equality alone
    /// never establishes compatibility. Metadata formation allocates nothing.
    pub fn from_bank(bank: &ExpertBankPlan) -> Result<Self> {
        let mut scopes = HashSet::new();
        let mut encodings = Vec::new();
        for layer in &bank.layers {
            if !scopes.insert(layer.scope) {
                return Err("duplicate routed layer scope".into());
            }
            let mut matrices = [[0; 2]; 3];
            for kind in 0..3 {
                let weight = &layer.resident[kind];
                if weight.role
                    != (WeightRole {
                        scope: layer.scope,
                        kind: BANK_KINDS[kind],
                    })
                {
                    return Err("shared expert role differs from its layer and component".into());
                }
                let [experts, rows, columns] = weight.shape[..] else {
                    return Err("shared expert weight is not a matrix stack".into());
                };
                if experts != bank.experts || rows == 0 || columns == 0 || weight.scale.is_some() {
                    return Err(
                        "shared expert geometry or scale differs from the bank contract".into(),
                    );
                }
                let bytes = checked(Tensor::canonical_layout(
                    weight.resident,
                    &[1, rows, columns],
                ))?
                .0;
                if bytes != layer.tensors[kind].slot_bytes {
                    return Err("canonical expert pitch differs from the bank plan".into());
                }
                matrices[kind] = [rows, columns];
            }
            encodings.push(Encoding {
                elements: layer.resident.each_ref().map(|w| w.resident),
                matrices,
            });
        }
        if bank.slots == 0 || bank.slots >= bank.experts {
            return Err("shared cache needs a capped bank".into());
        }
        Self::form(&encodings, bank.slots)
    }
    fn form(encodings: &[Encoding], slots_per_layer: u64) -> Result<Self> {
        if encodings.is_empty() || slots_per_layer == 0 {
            return Err("empty shared cache".into());
        }
        let mut groups: Vec<SharedGroupPlan> = Vec::new();
        let mut lookup = HashMap::new();
        let mut group_of_layer = Vec::with_capacity(encodings.len());
        for (layer, encoding) in encodings.iter().enumerate() {
            let group = *lookup.entry(encoding.clone()).or_insert_with(|| {
                let index = groups.len();
                groups.push(SharedGroupPlan {
                    encoding: encoding.clone(),
                    layers: Vec::new(),
                    records: 0,
                    allocated_bytes: 0,
                });
                index
            });
            groups[group].layers.push(layer);
            group_of_layer.push(group);
        }
        for group in &mut groups {
            group.records = slots_per_layer
                .checked_mul(group.layers.len() as u64)
                .ok_or("record count overflow")?;
        }
        let mut plan = Self {
            groups,
            group_of_layer,
            allocated_bytes: 0,
        };
        plan.recharge()?;
        Ok(plan)
    }
    /// Each group is one allocation per component, its records contiguous
    /// in the component's canonical layout.
    fn recharge(&mut self) -> Result<()> {
        self.allocated_bytes = 0;
        for group in &mut self.groups {
            group.allocated_bytes = 0;
            for kind in 0..3 {
                let bytes = checked(Tensor::canonical_layout(
                    group.encoding.elements[kind],
                    &group.encoding.extents(kind, group.records),
                ))?
                .0;
                group.allocated_bytes = group
                    .allocated_bytes
                    .checked_add(bytes)
                    .ok_or("shared cache charge overflow")?;
            }
            self.allocated_bytes = self
                .allocated_bytes
                .checked_add(group.allocated_bytes)
                .ok_or("shared cache total overflow")?;
        }
        Ok(())
    }
    /// A serial layer's largest distinct route union determines minimum
    /// capacity once per group, independently of its member layer count.
    pub fn with_records(&self, bank: &ExpertBankPlan, records: &[u64]) -> Result<Self> {
        if bank.selected == 0 || bank.selected > bank.experts || records.len() != self.groups.len()
        {
            return Err("invalid shared group capacities".into());
        }
        let minimum = crate::operators::routed::fused_graph::DECODE_ROWS
            .checked_mul(bank.selected)
            .ok_or("route union overflow")?
            .min(bank.experts);
        let mut plan = self.clone();
        for (group, &count) in plan.groups.iter_mut().zip(records) {
            let maximum = bank
                .experts
                .checked_mul(group.layers.len() as u64)
                .ok_or("group expert count overflow")?;
            if count < minimum || count > maximum || count > i32::MAX as u64 {
                return Err(format!(
                    "group capacity {count} outside {minimum}..={maximum}"
                ));
            }
            group.records = count;
        }
        plan.recharge()?;
        Ok(plan)
    }
    /// Validate the complete admitted plan against its source metadata.
    /// No mutable caller-supplied charge or encoding becomes authoritative.
    pub fn admitted(bank: &ExpertBankPlan) -> Result<Self> {
        let formed = Self::from_bank(bank)?;
        match &bank.shared {
            None => Ok(formed),
            Some(admitted) => {
                let records = admitted
                    .groups
                    .iter()
                    .map(|g| g.records)
                    .collect::<Vec<_>>();
                let expected = formed.with_records(bank, &records)?;
                if admitted != &expected {
                    return Err("shared plan differs from source metadata".into());
                }
                Ok(expected)
            }
        }
    }
}

/// A group's components: each a byte arena, so a lent record range can be
/// viewed in another form (`ConveyorPlan`), and its typed view at offset 0.
struct GroupStorage {
    arenas: [Tensor; 3],
    regions: [Tensor; 3],
}
pub(crate) struct SharedBankStore {
    plan: SharedBankPlan,
    role_layers: HashMap<WeightRole, (usize, usize)>,
    groups: Vec<Option<GroupStorage>>,
}
impl SharedBankStore {
    /// Ownership metadata only. The caller's existing allocation claim must
    /// cover a group's complete charge before its first binding allocates it.
    pub fn new(bank: &ExpertBankPlan) -> Result<Self> {
        let plan = SharedBankPlan::admitted(bank)?;
        let role_layers = bank
            .layers
            .iter()
            .enumerate()
            .flat_map(|(layer, b)| {
                BANK_KINDS.into_iter().enumerate().map(move |(kind, k)| {
                    (
                        WeightRole {
                            scope: b.scope,
                            kind: k,
                        },
                        (layer, kind),
                    )
                })
            })
            .collect();
        let groups = (0..plan.groups.len()).map(|_| None).collect();
        Ok(Self {
            plan,
            role_layers,
            groups,
        })
    }
    #[cfg(test)]
    pub fn plan(&self) -> &SharedBankPlan {
        &self.plan
    }
    pub fn bind(&mut self, device: &Device, role: WeightRole) -> Result<Tensor> {
        let &(layer, kind) = self
            .role_layers
            .get(&role)
            .ok_or("role is outside the shared bank")?;
        self.bind_layer(device, layer, kind)
    }
    fn bind_layer(&mut self, device: &Device, layer: usize, kind: usize) -> Result<Tensor> {
        if kind >= 3 {
            return Err("unknown expert component".into());
        }
        let group = *self
            .plan
            .group_of_layer
            .get(layer)
            .ok_or("unknown routed layer")?;
        Ok(self.storage(device, group)?.regions[kind].clone())
    }
    /// The byte arenas of `group`'s components, which a lent record range is
    /// viewed through (`ConveyorPlan`).
    pub fn arenas(&mut self, device: &Device, group: usize) -> Result<[Tensor; 3]> {
        Ok(self.storage(device, group)?.arenas.clone())
    }
    fn storage(&mut self, device: &Device, group: usize) -> Result<&GroupStorage> {
        let planned = self.plan.groups.get(group).ok_or("unknown bank group")?;
        if let Some(storage) = &self.groups[group] {
            if !storage.arenas[0].belongs_to(device) {
                return Err("shared bank device changed".into());
            }
        } else {
            let mut arenas = Vec::with_capacity(3);
            let mut regions = Vec::with_capacity(3);
            for kind in 0..3 {
                let element = planned.encoding.elements[kind];
                let extents = planned.encoding.extents(kind, planned.records);
                let bytes = checked(Tensor::canonical_layout(element, &extents))?.0;
                let arena = checked(Tensor::zeros(device, Element::bool(), &[bytes]))?;
                regions.push(checked(arena.region(element, &extents, 0))?);
                arenas.push(arena);
            }
            let arenas: [Tensor; 3] = arenas.try_into().expect("three components");
            if arenas.iter().map(Tensor::storage_bytes).sum::<u64>() != planned.allocated_bytes {
                return Err("shared cache allocation differs from plan".into());
            }
            // The store is installed only after the whole group succeeds.
            self.groups[group] = Some(GroupStorage {
                arenas,
                regions: regions.try_into().expect("three components"),
            });
        }
        Ok(self.groups[group].as_ref().expect("allocated"))
    }
    #[cfg(test)]
    pub fn allocated_bytes(&self) -> u64 {
        self.groups
            .iter()
            .flatten()
            .flat_map(|g| g.arenas.iter().map(Tensor::storage_bytes))
            .sum()
    }
    /// Weak observations of every physical allocation exactly once.
    #[cfg(test)]
    pub fn storage_observers(&self) -> Vec<seismic::TensorStorageObserver> {
        self.groups
            .iter()
            .flatten()
            .flat_map(|group| group.arenas.iter().map(Tensor::observe_storage))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planning::{
        ArtifactComponent, ArtifactComponentKind, BankLayer, BankTensor, ScratchPlan,
        StagingPoolPlan, WeightPlan,
    };
    use crate::streaming::layout::BankTensors;
    use magnitude_family_contracts::{SublayerIndex, WeightDescriptor, WeightScope};
    fn bank_metadata(e: &Encoding) -> ExpertBankPlan {
        let layers = (0..2)
            .map(|block| {
                let scope = WeightScope::TargetSublayer(SublayerIndex { block, sublayer: 1 });
                let resident = std::array::from_fn(|kind| {
                    let shape = vec![256, e.matrices[kind][0], e.matrices[kind][1]];
                    let bytes = Tensor::canonical_layout(e.elements[kind], &shape)
                        .unwrap()
                        .0;
                    WeightPlan {
                        role: WeightRole {
                            scope,
                            kind: BANK_KINDS[kind],
                        },
                        component: ArtifactComponent {
                            kind: ArtifactComponentKind::Target,
                            identity: magnitude_artifacts::ArtifactIdentity([0; 32]),
                        },
                        source: e.elements[kind],
                        upload: e.elements[kind],
                        resident: e.elements[kind],
                        descriptor: WeightDescriptor::stored(
                            format!("blk.{block}.kind.{kind}"),
                            shape.clone(),
                        ),
                        shape,
                        source_bytes: bytes,
                        resident_bytes: bytes,
                        scale: None,
                        bound: 0,
                        streamed: None,
                    }
                });
                BankLayer {
                    scope,
                    resident,
                    tensors: std::array::from_fn(|kind| BankTensor {
                        pool: kind,
                        scratch: kind,
                        stored_bytes: 16,
                        slot_bytes: Tensor::canonical_layout(
                            e.elements[kind],
                            &[1, e.matrices[kind][0], e.matrices[kind][1]],
                        )
                        .unwrap()
                        .0,
                    }),
                }
            })
            .collect();
        ExpertBankPlan {
            shared: None,
            slots: 2,
            experts: 256,
            selected: 1,
            staging_records: 10,
            layers,
            pools: (0..3)
                .map(|kind| StagingPoolPlan {
                    kind,
                    source: e.elements[kind],
                    matrix: e.matrices[kind],
                    record_bytes: Tensor::canonical_layout(
                        e.elements[kind],
                        &[1, e.matrices[kind][0], e.matrices[kind][1]],
                    )
                    .unwrap()
                    .0,
                })
                .collect(),
            scratch: (0..3)
                .map(|kind| ScratchPlan {
                    kind,
                    element: e.elements[kind],
                    matrix: e.matrices[kind],
                    slot_bytes: Tensor::canonical_layout(
                        e.elements[kind],
                        &[1, e.matrices[kind][0], e.matrices[kind][1]],
                    )
                    .unwrap()
                    .0,
                })
                .collect(),
            // Each pool is its routed tensor's only one: its conveyor opens
            // the tensor's held arena.
            conveyor: crate::planning::ConveyorPlan {
                lender: None,
                lent: 0,
                offsets: vec![0; 3],
            },
        }
    }
    fn encoding(down: &str) -> Encoding {
        Encoding {
            elements: ["q4k", "q4k", down]
                .map(|r| Element::stored(r, seismic::Layout::Rows32).unwrap()),
            matrices: [[512, 2048], [512, 2048], [2048, 512]],
        }
    }
    #[test]
    fn captured_target_budget_charges_groups_once() {
        let mut layers = vec![encoding("q5k"); 37];
        layers.extend(vec![encoding("q6k"); 2]);
        let mut last = encoding("q6k");
        last.elements[0] = Element::stored("q5k", seismic::Layout::Rows32).unwrap();
        last.elements[1] = last.elements[0];
        layers.push(last);
        let p = SharedBankPlan::form(&layers, 64).unwrap();
        assert_eq!(p.groups.len(), 3);
        assert_eq!(
            p.groups.iter().map(|g| g.layers.len()).collect::<Vec<_>>(),
            [37, 2, 1]
        );
        assert_eq!(p.allocated_bytes, 4_991_221_760);
    }
    #[test]
    fn equal_bytes_do_not_imply_equal_encoding_and_records_are_contiguous() {
        let a = Encoding {
            elements: [Element::f32(); 3],
            matrices: [[2, 2]; 3],
        };
        let b = Encoding {
            elements: [Element::i32(); 3],
            matrices: [[2, 2]; 3],
        };
        let p = SharedBankPlan::form(&[a.clone(), a, b], 17).unwrap();
        assert_eq!(p.groups.len(), 2);
        assert_eq!(p.groups[0].records, 34);
        // Three components of 34 and 17 records of 16 bytes, nothing else.
        assert_eq!(p.groups[0].allocated_bytes, 3 * 34 * 16);
        assert_eq!(p.groups[1].allocated_bytes, 3 * 17 * 16);
    }
    #[test]
    fn failed_allocation_is_unpublished_and_aliases_hold_real_charge() {
        let device = seismic::DeviceCatalog::discover()
            .unwrap()
            .open_backend(seismic::BackendName::Cpu)
            .unwrap();
        let baseline = device.memory_usage().charged;
        let e = Encoding {
            elements: [Element::f32(); 3],
            matrices: [[2, 2]; 3],
        };
        let mut metadata = bank_metadata(&e);
        metadata.slots = 9; // the last component fails after the others succeeded
        let mut bank = SharedBankStore::new(&metadata).unwrap();
        let charge = bank.plan().allocated_bytes;
        let role = |layer: usize| WeightRole {
            scope: metadata.layers[layer].scope,
            kind: BANK_KINDS[0],
        };
        device.set_memory_limit(Some(baseline + charge - 1));
        assert!(bank.bind(&device, role(0)).is_err());
        assert_eq!(bank.allocated_bytes(), 0);
        assert!(bank.storage_observers().is_empty());
        assert_eq!(device.memory_usage().charged, baseline);
        device.set_memory_limit(None);
        let first = bank.bind(&device, role(0)).unwrap();
        let alias = bank.bind(&device, role(1)).unwrap();
        assert!(first.shares_allocation(&alias));
        assert_eq!(first.extents(), &[18, 2, 2]);
        assert_eq!(bank.allocated_bytes(), charge);
        assert_eq!(device.memory_usage().charged - baseline, charge);
        let census = bank.storage_observers();
        assert_eq!(
            census
                .iter()
                .map(|o| o.charged_bytes().unwrap())
                .sum::<u64>(),
            charge
        );
        // Bound components outlive the store and keep exactly their own
        // allocation charged.
        drop(bank);
        assert_eq!(device.memory_usage().charged - baseline, first.storage_bytes());
        assert_eq!(first.storage_bytes(), charge / 3);
        drop(first);
        drop(alias);
        assert_eq!(device.memory_usage().charged, baseline);
        assert!(census.iter().all(|o| o.charged_bytes().is_none()));
    }
    #[test]
    fn actual_bank_metadata_rejects_inconsistent_roles_geometry_and_charge() {
        let e = Encoding {
            elements: [Element::f32(); 3],
            matrices: [[2, 2]; 3],
        };
        let mut b = bank_metadata(&e);
        assert!(SharedBankPlan::from_bank(&b).is_ok());
        b.layers[0].tensors[0].slot_bytes += 4;
        assert!(SharedBankPlan::from_bank(&b).is_err());
        b = bank_metadata(&e);
        b.layers[0].resident[0].shape[0] = 255;
        assert!(SharedBankPlan::from_bank(&b).is_err());
        b = bank_metadata(&e);
        b.layers[0].resident[0].role.scope = b.layers[1].scope;
        assert!(SharedBankPlan::from_bank(&b).is_err());
        b = bank_metadata(&e);
        b.layers[1].scope = b.layers[0].scope;
        assert!(SharedBankPlan::from_bank(&b).is_err());
    }
    #[test]
    fn directory_aliases_group_words_and_charges_exactly_its_allocations() {
        let device = seismic::DeviceCatalog::discover()
            .unwrap()
            .open_backend(seismic::BackendName::Cpu)
            .unwrap();
        let baseline = device.memory_usage().charged;
        let e = Encoding {
            elements: [Element::f32(); 3],
            matrices: [[2, 2]; 3],
        };
        let mut metadata = bank_metadata(&e);
        metadata.slots = 9;
        let shared = SharedBankPlan::from_bank(&metadata).unwrap();
        metadata.shared = Some(shared.clone());
        let tensors =
            BankTensors::allocate(&device, &metadata, &shared, None).unwrap();
        assert!(tensors.layers[0]
            .stamps
            .shares_allocation(&tensors.layers[1].stamps));
        let first = &tensors.layers[0].commits;
        assert_eq!(first.extents(), &[18]);
        assert!(first.shares_allocation(&tensors.layers[1].commits));
        assert_eq!(tensors.copy_lists[0].extents(), &[2 + 2 * metadata.experts]);
        assert_eq!(
            device.memory_usage().charged - baseline,
            metadata.charged_bytes().unwrap()
        );
        let directory =
            super::super::shared_directory::SharedDirectory::new(&shared, metadata.experts)
                .unwrap();
        assert!(directory.validate_ports(&[(18, 18), (18, 18)]).is_ok());
        assert!(directory.validate_ports(&[(9, 18), (18, 18)]).is_err());
        assert!(directory.validate_ports(&[(18, 0), (18, 18)]).is_err());
        drop(tensors);
        assert_eq!(device.memory_usage().charged, baseline);
    }
    #[test]
    fn shared_directory_fits_exact_charge_and_failed_construction_releases_every_byte() {
        let device = seismic::DeviceCatalog::discover()
            .unwrap()
            .open_backend(seismic::BackendName::Cpu)
            .unwrap();
        let baseline = device.memory_usage().charged;
        let e = Encoding {
            elements: [Element::f32(); 3],
            matrices: [[2, 2]; 3],
        };
        let mut metadata = bank_metadata(&e);
        metadata.slots = 9;
        let shared = SharedBankPlan::from_bank(&metadata).unwrap();
        metadata.shared = Some(shared.clone());
        let charge = metadata.charged_bytes().unwrap();
        device.set_memory_limit(Some(baseline + charge - 1));
        assert!(BankTensors::allocate(&device, &metadata, &shared, None).is_err());
        assert_eq!(device.memory_usage().charged, baseline);
        device.set_memory_limit(Some(baseline + charge));
        let tensors =
            BankTensors::allocate(&device, &metadata, &shared, None).unwrap();
        assert_eq!(device.memory_usage().charged, baseline + charge);
        drop(tensors);
        assert_eq!(device.memory_usage().charged, baseline);
        device.set_memory_limit(None);
        let mut foreign = shared.clone();
        foreign.group_of_layer.swap(0, 1); // identical members: still valid
        foreign.groups[0].records += 1;
        assert!(BankTensors::allocate(&device, &metadata, &foreign, None).is_err());
        assert_eq!(device.memory_usage().charged, baseline);
    }
}
