//! Necessary service over an interval of retained launch choices. No endpoint
//! execution is sampled: all quantities below are envelopes over every member.
use super::*;
use crate::ptx::{self, AddressBase, Operation, Origin, RegisterName};
use model::{Quantity, ResourceScope};

pub(super) fn derive(
    domain: &IntegerRange<BlockChoice>,
    indices: std::ops::Range<usize>,
    hardware: &CudaHardware,
) -> Result<Option<schedule::Demand>, String> {
    if indices.is_empty() || indices.end > domain.len() {
        return Err("CUDA block relaxation needs a nonempty subdomain".into());
    }
    let first = domain
        .get(indices.start)
        .ok_or("invalid CUDA region index")?;
    let last = domain
        .get(indices.end - 1)
        .ok_or("invalid CUDA region index")?;
    let family = domain.decision.family();
    if domain.decision.phase != family.selected.len() {
        return Err("CUDA region does not identify the retained unresolved phase".into());
    }
    // Block/warp-local pools may be instantiated concurrently. Dropping them is
    // safe; treating them as a single device pool would overstate necessity.
    let mut resources = Vec::new();
    let mut resource_map = Vec::new();
    for resource in &hardware.resources {
        resource_map.push(if resource.scope == ResourceScope::Device {
            let index = resources.len();
            resources.push(schedule::Resource {
                name: resource.name.clone(),
                capacity: resource.capacity,
                unit: resource.unit.clone(),
            });
            Some(index)
        } else {
            None
        });
    }
    let mut demand = schedule::Demand::new(hardware.timebase.clone(), resources)?;
    for (phase, implementation) in family.phases.iter().enumerate() {
        let target = &implementation.target;
        // A relaxation must not hide a missing primitive/helper implementation.
        match model::validate_target(
            target,
            hardware,
            &model::target_requirements(target),
            DerivationLimits {
                instructions: u64::MAX,
                operations: usize::MAX,
            },
        ) {
            Ok(()) => {}
            // An unavailable service supplies no bound. Its explicit reason
            // remains with the terminal analysis when this region is refined.
            Err(DerivationError::Unsupported(_)) => return Ok(None),
            Err(error) => return Err(error.into()),
        }
        let lanes = u64::from(target.domain().lanes_per_item);
        let work_items = target
            .domain()
            .work_items
            .checked_mul(lanes)
            .ok_or("CUDA participant count overflow")?;
        if work_items == 0 {
            continue;
        }
        let interval = family
            .block_interval(phase)
            .ok_or("missing CUDA phase domain")?;
        let (minimum_block, maximum_block) = if phase == domain.decision.phase {
            (first.min(last), first.max(last))
        } else {
            (*interval.start(), *interval.end())
        };
        let minimum_block = minimum_block
            .checked_mul(lanes)
            .ok_or("CUDA block extent overflow")?;
        let maximum_block = maximum_block
            .checked_mul(lanes)
            .ok_or("CUDA block extent overflow")?;
        // A warp contains at most min(warp width, block size) active work items.
        // Blocks, tails, and divergent exits can only increase the issue count.
        let minimum_warps = work_items.div_ceil(u64::from(hardware.warp_width).min(maximum_block));
        let register_bits = model::virtual_register_bits(target)?;
        for instruction in target
            .instructions()
            .take_while(|i| !matches!(i.origin, Origin::Ssa { .. }))
        {
            let minimum_active = match (&instruction.operation, instruction.predicate) {
                (Operation::Return, Some(_)) => 0, // padding exit: active lanes continue
                (Operation::Return | Operation::Branch { .. } | Operation::Call { .. }, _) => {
                    return Ok(None);
                }
                (_, Some(_)) => return Ok(None), // no general conditional-prefix necessity rule
                _ => 1,
            };
            include(
                &mut demand,
                &resource_map,
                hardware,
                &instruction.operation,
                minimum_active,
                minimum_block,
                register_bits,
                minimum_warps,
            )?;
        }
        // Concrete derivation accepts only successful invocations, requiring a
        // status write from every active work item. All return sites share this
        // typed ABI store; count it once across alternative return paths.
        let Some(status) = target
            .registers()
            .iter()
            .position(|r| r.name == RegisterName::Abi(ptx::AbiRegister::StatusAddress))
        else {
            return Ok(None);
        };
        let status_store = target.instructions().find(|instruction| {
            matches!(instruction.operation,
            Operation::Store { space:ptx::Space::Global, data_type:ptx::DataType::U32,
                address:ptx::Address { base:AddressBase::Register(register),offset:0 }, .. }
                if register.0 == status)
        });
        let Some(status_store) = status_store else {
            return Ok(None);
        };
        include(
            &mut demand,
            &resource_map,
            hardware,
            &status_store.operation,
            1,
            minimum_block,
            register_bits,
            minimum_warps,
        )?;
    }
    Ok(Some(demand))
}

fn include(
    demand: &mut schedule::Demand,
    resource_map: &[Option<usize>],
    hardware: &CudaHardware,
    operation: &Operation,
    minimum_active: u64,
    minimum_block: u64,
    register_bits: u64,
    minimum_warps: u64,
) -> Result<(), String> {
    let primitive = operation.primitive();
    let timing = hardware
        .timings
        .iter()
        .find(|t| t.primitive == primitive)
        .ok_or("missing CUDA primitive timing in region relaxation")?;
    let access_bytes = match operation {
        Operation::Load { data_type, .. } | Operation::Store { data_type, .. } => {
            u64::from(data_type.bits() / 8)
        }
        _ => 0,
    }
    .checked_mul(minimum_active)
    .ok_or("CUDA region access size overflow")?;
    let (mut latency, service) = model::primitive_service::<String>(timing, |quantity| {
        Ok(match quantity {
            Quantity::One | Quantity::IssuedLanes => 1,
            Quantity::ActiveLanes => minimum_active,
            Quantity::RequestedBytes => access_bytes,
            Quantity::MemorySectors { bytes } => access_bytes.div_ceil(bytes),
            Quantity::BlockThreads => minimum_block,
            Quantity::BlockWarps => minimum_block.div_ceil(u64::from(hardware.warp_width)),
            Quantity::VirtualRegisterBits => register_bits
                .checked_mul(minimum_block)
                .ok_or("CUDA region register size overflow")?,
        })
    })?;
    let mut reservations = Vec::new();
    for mut reservation in service {
        let Some(resource) = resource_map[reservation.resource] else {
            continue;
        };
        reservation.resource = resource;
        // Every admitted concrete instruction covers all its reservations. The
        // independent lower envelopes retain that necessary completion bound.
        latency = latency.max(
            reservation
                .offset
                .checked_add(reservation.duration)
                .ok_or("CUDA region duration overflow")?,
        );
        reservations.push(reservation);
    }
    demand.include(
        &schedule::Operation {
            name: format!("{primitive:?}"),
            predecessors: Vec::new(),
            start_predecessors: Vec::new(),
            latency,
            reservations,
        },
        minimum_warps,
    )
}
