//! Shared physical realization contracts. This crate owns the physical graph
//! and invocation ABI facts; it does not contain a second execution IR.

use seismic_lang::abi::ScalarParameter;

pub mod dispatch;
pub mod executable;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferSpec {
    pub parameter: String,
    pub plane: String,
    pub role: BufferRole,
    pub bytes: usize,
    pub alignment: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BufferRole {
    Parameter,
    Result { path: Vec<u32> },
    Internal,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InvocationConditions {
    read_only_buffers: Vec<usize>,
    independent_buffers: Vec<usize>,
    alias_pairs: Vec<(usize, usize, bool)>,
}

impl InvocationConditions {
    pub fn from_executable<D: executable::ExecutableDialect>(
        plan: &executable::ResolvedPlan<D>,
        abi_allocations: &[executable::ResolvedStorageId],
    ) -> Result<Self, String> {
        use executable::{
            AccessMode, ResolvedScheduleItem, ResolvedStorageId, ResolvedValueTransport,
        };
        use std::collections::{BTreeMap, BTreeSet};

        fn pair_transports(
            left: &ResolvedValueTransport,
            right: &ResolvedValueTransport,
            aliases: &mut BTreeMap<ResolvedStorageId, BTreeSet<ResolvedStorageId>>,
        ) -> Result<(), String> {
            match (left, right) {
                (ResolvedValueTransport::Void, ResolvedValueTransport::Void) => Ok(()),
                (ResolvedValueTransport::Kernel(_), ResolvedValueTransport::Kernel(_)) => Ok(()),
                (ResolvedValueTransport::Storage(left), ResolvedValueTransport::Storage(right)) => {
                    if left.len() != right.len() {
                        return Err("call binding changes physical plane count".into());
                    }
                    for (left, right) in left.iter().zip(right.iter()) {
                        aliases.entry(*left).or_default().insert(*right);
                        aliases.entry(*right).or_default().insert(*left);
                    }
                    Ok(())
                }
                (ResolvedValueTransport::Tuple(left), ResolvedValueTransport::Tuple(right)) => {
                    if left.len() != right.len() {
                        return Err("call binding changes tuple arity".into());
                    }
                    for (left, right) in left.iter().zip(right.iter()) {
                        pair_transports(left, right, aliases)?;
                    }
                    Ok(())
                }
                _ => Err("call binding changes physical value transport kind".into()),
            }
        }

        fn collect<D: executable::ExecutableDialect>(
            plan: &executable::ResolvedPlan<D>,
            writes: &mut BTreeSet<ResolvedStorageId>,
            aliases: &mut BTreeMap<ResolvedStorageId, BTreeSet<ResolvedStorageId>>,
        ) -> Result<(), String> {
            for item in plan.items().iter() {
                match item {
                    ResolvedScheduleItem::Phase(phase) => {
                        for launch in phase.launches.iter() {
                            for access in &launch.kernel.resources.accesses {
                                if !matches!(access.mode, AccessMode::Read) {
                                    writes.insert(access.storage);
                                }
                            }
                        }
                    }
                    ResolvedScheduleItem::Subplan(subplan) => {
                        for binding in subplan.inputs.iter().chain(&subplan.results) {
                            pair_transports(&binding.caller, &binding.callee, aliases)?;
                        }
                        collect(&subplan.plan, writes, aliases)?;
                    }
                }
            }
            Ok(())
        }

        let mut seen = BTreeSet::new();
        if abi_allocations.iter().any(|id| !seen.insert(*id)) {
            return Err("executable ABI repeats a storage allocation".into());
        }
        let public = plan
            .device_storage()
            .allocations
            .iter()
            .map(|storage| (storage.id, storage))
            .collect::<BTreeMap<_, _>>();
        for id in abi_allocations {
            let storage = public
                .get(id)
                .ok_or_else(|| format!("executable ABI names absent storage#{}", id.0))?;
            if storage.scope != executable::StorageScope::External
                || !matches!(
                    storage.provenance.abi,
                    Some(
                        executable::AbiRole::Parameter { .. } | executable::AbiRole::Result { .. }
                    )
                )
            {
                return Err(format!("storage#{} is not a public ABI allocation", id.0));
            }
        }
        let mut writes = BTreeSet::new();
        let mut aliases = BTreeMap::new();
        collect(plan, &mut writes, &mut aliases)?;
        let class_writes = |root: ResolvedStorageId| {
            let mut pending = vec![root];
            let mut visited = BTreeSet::new();
            while let Some(value) = pending.pop() {
                if !visited.insert(value) {
                    continue;
                }
                if writes.contains(&value) {
                    return true;
                }
                pending.extend(aliases.get(&value).into_iter().flatten().copied());
            }
            false
        };
        let read_only_buffers = abi_allocations
            .iter()
            .enumerate()
            .filter_map(|(slot, id)| (!class_writes(*id)).then_some(slot))
            .collect::<Vec<_>>();
        let independent_buffers = abi_allocations
            .iter()
            .enumerate()
            .filter_map(|(slot, id)| {
                matches!(
                    public[id].provenance.abi,
                    Some(executable::AbiRole::Result { .. })
                )
                .then_some(slot)
            })
            .collect::<Vec<_>>();
        let read_only = read_only_buffers.iter().copied().collect::<BTreeSet<_>>();
        let independent = independent_buffers.iter().copied().collect::<BTreeSet<_>>();
        let mut alias_pairs = Vec::new();
        for left in 0..abi_allocations.len() {
            for right in left + 1..abi_allocations.len() {
                if !read_only.contains(&left)
                    || !read_only.contains(&right)
                    || independent.contains(&left)
                    || independent.contains(&right)
                {
                    alias_pairs.push((left, right, false));
                }
            }
        }
        Ok(Self {
            read_only_buffers,
            independent_buffers,
            alias_pairs,
        })
    }

    pub fn read_only_buffers(&self) -> &[usize] {
        &self.read_only_buffers
    }
    pub fn independent_buffers(&self) -> &[usize] {
        &self.independent_buffers
    }
    pub fn alias_pairs(&self) -> &[(usize, usize, bool)] {
        &self.alias_pairs
    }

    pub fn validate_aliases(
        &self,
        buffers: &[BufferSpec],
        locate: impl Fn(usize) -> (u64, u64),
    ) -> Result<(), String> {
        for &independent in &self.independent_buffers {
            let (identity, _) = locate(independent);
            for other in 0..buffers.len() {
                if other != independent && locate(other).0 == identity {
                    return Err(format!(
                        "owned binding {} must have an independent allocation",
                        buffers
                            .get(independent)
                            .map_or("?", |buffer| buffer.parameter.as_str())
                    ));
                }
            }
        }
        for &(left_slot, right_slot, exact_allowed) in &self.alias_pairs {
            let left = buffers.get(left_slot).ok_or("invalid alias ABI slot")?;
            let right = buffers.get(right_slot).ok_or("invalid alias ABI slot")?;
            let ((left_id, left_offset), (right_id, right_offset)) =
                (locate(left_slot), locate(right_slot));
            let left_end = left_offset
                .checked_add(left.bytes as u64)
                .ok_or("alias range overflow")?;
            let right_end = right_offset
                .checked_add(right.bytes as u64)
                .ok_or("alias range overflow")?;
            if left_id == right_id
                && left.bytes != 0
                && right.bytes != 0
                && left_offset < right_end
                && right_offset < left_end
                && !(exact_allowed && left_offset == right_offset && left.bytes == right.bytes)
            {
                return Err("source parallel binding has unsafe overlapping storage".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathFunction {
    Exp,
    ExpFast,
    Log,
    Sin,
    Cos,
}

impl MathFunction {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Exp | Self::ExpFast => "seismic_exp",
            Self::Log => "seismic_log",
            Self::Sin => "seismic_sin",
            Self::Cos => "seismic_cos",
        }
    }
}

pub fn encode_scalars(schema: &[ScalarParameter], scalars: &[f64]) -> Result<Vec<u64>, String> {
    let bytes = seismic_lang::abi::ScalarLayout::words(schema)?.encode(scalars)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}
