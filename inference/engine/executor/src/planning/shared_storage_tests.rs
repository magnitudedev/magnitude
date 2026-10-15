use super::*;
use crate::programs::native_target_graph::BankPart;
use crate::streaming::{
    layout::{bank_port, BankTensors},
    shared_bank::SharedBankStore,
};
use magnitude_family_contracts::SublayerIndex;
use seismic::{DeviceCatalog, Tensor};

fn resident_plan() -> ModelLoadPlan {
    let target = (0..3)
        .flat_map(|block| {
            BANK_KINDS.into_iter().enumerate().map(move |(kind, name)| {
                let scope = WeightScope::TargetSublayer(SublayerIndex { block, sublayer: 1 });
                let shape = vec![256, 2, if block == 2 { 8 } else { 4 }];
                let element = Element::f32();
                let bytes = Tensor::canonical_layout(element, &shape).unwrap().0;
                WeightPlan {
                    role: WeightRole { scope, kind: name },
                    component: ArtifactComponent {
                        kind: ArtifactComponentKind::Target,
                        identity: ArtifactIdentity([0; 32]),
                    },
                    source: element,
                    upload: element,
                    resident: element,
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
            })
        })
        .collect();
    ModelLoadPlan {
        target,
        head: None,
        vision: None,
        host_tables: vec![],
        readout_candidates: vec![],
        readout_choices: vec![],
        unplaced_output: vec![],
        residual_conditioning: None,
        bank: None,
    }
}

#[test]
fn shared_storage_shapes_ports_and_physical_charge_agree() {
    let resident = resident_plan();
    let comparison = resident.clone().with_expert_bank(9, 1).unwrap();
    let shared = resident.with_shared_expert_bank(9, 1).unwrap();
    let bank = shared.expert_bank().unwrap();
    let groups = bank.shared.as_ref().unwrap();
    assert_eq!(groups.groups.len(), 2);
    assert_eq!(groups.groups[0].records, 18);
    assert_eq!(groups.groups[1].records, 9);
    for weight in shared.target() {
        let records = bank.records(weight.role.scope).unwrap();
        assert_eq!(weight.shape[0], records);
        assert_eq!(bank.routed(weight.role.scope).unwrap().slots, records);
        for port in [BankPart::Stamps, BankPart::Commits] {
            assert_eq!(
                bank_port(&shared, weight.role.scope, port).unwrap().1,
                vec![records]
            );
        }
        assert_eq!(
            bank_port(&shared, weight.role.scope, BankPart::CopyList { kind: 0 })
                .unwrap()
                .1,
            vec![2 + 2 * bank.experts]
        );
        assert_eq!(bank.resident_plan(weight.role).unwrap().shape[0], 256);
        assert_eq!(weight.source_bytes, 0);
    }
    assert!(comparison.expert_bank().unwrap().shared.is_none());
    let scope = bank.layers[0].scope;
    assert!(bank_port(&comparison, scope, BankPart::Commits).is_err());
    assert_eq!(
        weight_bytes_by_component(&shared).unwrap(),
        [groups.allocated_bytes + bank.charged_bytes().unwrap(), 0, 0]
    );
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let baseline = device.memory_usage().charged;
    let mut store = SharedBankStore::new(bank).unwrap();
    let tensors = shared
        .target()
        .iter()
        .map(|weight| store.bind(&device, weight.role).unwrap())
        .collect::<Vec<_>>();
    assert!(tensors[0].shares_allocation(&tensors[3]));
    assert!(!tensors[0].shares_allocation(&tensors[6]));
    let controls = BankTensors::allocate(&device, bank, groups).unwrap();
    assert_eq!(
        device.memory_usage().charged - baseline,
        weight_bytes_by_component(&shared).unwrap()[0]
    );
    drop(controls);
    drop(tensors);
    drop(store);
    assert_eq!(device.memory_usage().charged, baseline);
}

#[test]
fn shared_formation_preserves_source_geometry_and_rejects_double_banking() {
    let original = resident_plan();
    let shared = original.clone().with_shared_expert_bank(9, 1).unwrap();
    let bank = shared.expert_bank().unwrap();
    for weight in original.target() {
        assert_eq!(bank.resident_plan(weight.role).unwrap(), weight);
    }
    assert!(shared.with_shared_expert_bank(9, 1).is_err());
    assert!(original.clone().with_shared_expert_bank(0, 1).is_err());
    assert!(original.with_shared_expert_bank(256, 1).is_err());
}

#[test]
fn group_minimum_is_one_launch_union_and_exact_charge_includes_padding() {
    let load = resident_plan()
        .with_shared_expert_bank_capacity(&[8, 17], 1)
        .unwrap();
    let bank = load.expert_bank().unwrap();
    let shared = bank.shared.as_ref().unwrap();
    assert_eq!(shared.groups[0].layers.len(), 2);
    assert_eq!(shared.groups[0].records, 8); // two layers share ONE minimum
    assert_eq!(shared.groups[1].records, 17);
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let baseline = device.memory_usage().charged;
    let charge = weight_bytes_by_component(&load).unwrap()[0];
    device.set_memory_limit(Some(baseline + charge));
    let mut store = SharedBankStore::new(bank).unwrap();
    let weights = load
        .target()
        .iter()
        .map(|w| store.bind(&device, w.role).unwrap())
        .collect::<Vec<_>>();
    let controls = BankTensors::allocate(&device, bank, shared).unwrap();
    assert_eq!(weights[0].extents(), &[8, 2, 4]);
    assert_eq!(weights[6].extents(), &[17, 2, 8]);
    assert_eq!(device.memory_usage().charged, baseline + charge);
    drop(controls);
    drop(weights);
    drop(store);
    assert_eq!(device.memory_usage().charged, baseline);
}

#[test]
fn group_capacity_rejects_unions_it_cannot_hold_and_foreign_metadata() {
    for records in [&[7, 8][..], &[8, 7], &[513, 8], &[8, 257], &[8], &[8, 8, 8]] {
        assert!(resident_plan()
            .with_shared_expert_bank_capacity(records, 1)
            .is_err());
    }
    assert!(resident_plan()
        .with_shared_expert_bank_capacity(&[8, 8], 0)
        .is_err());
    assert!(resident_plan()
        .with_shared_expert_bank_capacity(&[8, 8], 257)
        .is_err());
    let load = resident_plan()
        .with_shared_expert_bank_capacity(&[64, 64], 8)
        .unwrap();
    let mut bank = load.expert_bank().unwrap().clone();
    bank.shared.as_mut().unwrap().groups[0].allocated_bytes += 1;
    assert!(SharedBankStore::new(&bank).is_err());
    bank = load.expert_bank().unwrap().clone();
    bank.shared.as_mut().unwrap().group_of_layer.swap(0, 2);
    assert!(SharedBankStore::new(&bank).is_err());
}
