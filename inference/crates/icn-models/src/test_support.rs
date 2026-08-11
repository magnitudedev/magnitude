use icn_contracts::{
    HardwareMemoryDomain, HardwareMemoryDomainKind, HardwareSnapshot, HardwareSystemMemory,
    MemoryDomainId, MemoryTopology,
};

pub(crate) fn system_memory_topology(capacity_bytes: u64) -> MemoryTopology {
    MemoryTopology::from_snapshot(&HardwareSnapshot {
        captured_at: 1,
        platform: "test".to_owned(),
        architecture: "test".to_owned(),
        system_product_name: None,
        cpu_model: None,
        logical_cores: 1,
        system_memory: HardwareSystemMemory {
            physical_capacity_bytes: capacity_bytes,
            physical_available_bytes: capacity_bytes,
            allocation_capacity_bytes: capacity_bytes,
            allocation_headroom_bytes: capacity_bytes,
            assess_reserve_bytes: 0,
            abort_reserve_bytes: 0,
        },
        native_build: "test".to_owned(),
        enabled_backends: vec!["cpu".to_owned()],
        topology_fingerprint: "test".to_owned(),
        memory_domains: vec![HardwareMemoryDomain {
            id: MemoryDomainId::system(),
            kind: HardwareMemoryDomainKind::System,
            total_capacity_bytes: capacity_bytes,
            stable_capacity_bytes: capacity_bytes,
            current_free_bytes: Some(capacity_bytes),
            shares_system_memory: true,
            devices: Vec::new(),
        }],
    })
    .expect("valid test hardware snapshot")
}
