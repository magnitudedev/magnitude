//! Metal grouping choices derived from selected IR, memory plans, and device limits.
//! This is legality, not a throughput preference or native occupancy estimate.
//! The family varies independent launch groupings; other execution choices
//! remain fixed. Constructing and selecting this family never emits target code.
use crate::execution::Execution;
use seismic_accounting::selection::Choices;
use seismic_compiler::tuner::Preparation;
use seismic_realization::dispatch::GroupDispatch;
use std::sync::Arc;

/// Legal groupings derived for one actual launch, including padding constraints.
#[derive(Clone)]
pub struct GroupingChoices {
    pub launch: usize,
    alternatives: Vec<u64>,
    family: Arc<GroupFamily>,
    selected: Vec<u64>,
}
impl std::fmt::Debug for GroupingChoices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupingChoices")
            .field("launch", &self.launch)
            .field("alternatives", &self.alternatives)
            .field("selected", &self.selected)
            .field("function", &self.family.execution.function().name)
            .finish()
    }
}
impl PartialEq for GroupingChoices {
    fn eq(&self, other: &Self) -> bool {
        self.launch == other.launch
            && self.alternatives == other.alternatives
            && self.selected == other.selected
            && (Arc::ptr_eq(&self.family, &other.family)
                || self.family.execution.same_implementation(&other.family.execution))
    }
}
impl GroupingChoices {
    pub fn family(&self) -> &GroupFamily {
        &self.family
    }
    pub fn selected(&self) -> &[u64] {
        &self.selected
    }
    pub fn values(&self) -> &[u64] {
        &self.alternatives
    }
    pub fn index(&self, value: u64) -> Option<usize> {
        self.alternatives.binary_search(&value).ok()
    }
    pub(crate) fn relax(&self, indices: std::ops::Range<usize>, hardware: &crate::model::Hardware, workload: &seismic_accounting::workload::ScalarWorkload, limits: seismic_accounting::workload::DerivationLimits) -> Result<Option<seismic_accounting::schedule::Demand>, String> {
        if indices.is_empty() || indices.end > self.alternatives.len() { return Err("Metal grouping relaxation needs a nonempty subdomain".into()); }
        let mut launches = Vec::new();
        for (launch, dispatch) in self.family.launches.iter().enumerate() {
            let maximum = if let Some(&selected) = self.selected.get(launch) { selected }
            else if launch == self.launch { *self.alternatives[indices.clone()].iter().max().unwrap() }
            else { *self.family.grouping_values(launch)?.last().ok_or("empty Metal grouping domain")? };
            launches.push((dispatch.work_items, maximum));
        }
        let dispatch = crate::model::dispatch_demand(hardware, launches)?;
        let mut demand = match dispatch {
            Some(demand) => demand,
            None => seismic_accounting::schedule::Demand::new(hardware.timebase.clone(), hardware.resources.clone())?,
        };
        let account = self.family.execution.relaxation(workload, limits)?;
        // Regrouping changes launch geometry and shared-array base addresses,
        // but each logical subgroup still performs the same selected body.
        // Ignore integer/address/control work and keep the same conservative
        // mandatory-work predicate used by terminal traversal refinement.
        crate::model::include_preserved_demand(&mut demand, &account, hardware, crate::terminal::traversal::preserves)?;
        Ok(Some(demand))
    }
    /// Apply one original-domain ordinal to this immutable prepared family.
    pub fn refine(&self, index: usize) -> Result<Preparation<Execution>, String> {
        if self.launch != self.selected.len() {
            return Err("Metal grouping choice does not identify the unresolved launch".into());
        }
        let value = self.get(index).ok_or("Metal grouping choice is outside its domain")?;
        let mut selected = self.selected.clone();
        selected.push(value);
        self.family.next(selected)
    }
}
impl Choices for GroupingChoices {
    type Alternative = u64;
    fn len(&self) -> usize {
        self.alternatives.len()
    }
    fn get(&self, index: usize) -> Option<u64> {
        self.alternatives.get(index).copied()
    }
}

#[derive(Clone, Debug)]
pub struct Constraint {
    pub launch: usize,
    pub resource: &'static str,
    pub units_per_item: u64,
    pub capacity: u64,
    pub maximum_items: u64,
}
#[derive(Clone, Debug)]
pub struct Grouping {
    pub items_per_group: u64,
    pub launches: Vec<GroupDispatch>,
    pub shared_bytes_per_group: Vec<u64>,
}
pub struct GroupFamily {
    execution: Execution,
    launches: Vec<GroupDispatch>,
    shared_bytes_per_item: Vec<u64>,
    pub constraints: Vec<Constraint>,
    maximum_items: u64,
    maximum_by_launch: Vec<u64>,
}
impl GroupFamily {
    /// Keep the selected operations/ownership fixed and solve their exact linear
    /// thread/shared-array constraints. Main and split-merge launches both count.
    pub fn derive(execution: Execution) -> Result<Self, String> {
        let launches: Vec<_> = execution
            .phases()
            .iter()
            .flat_map(|phase| {
                std::iter::once(phase.dispatch.clone()).chain(phase.merge_dispatch.clone())
            })
            .collect();
        if launches.len() != execution.memory().launches().len() {
            return Err("group family memory/dispatch launch mismatch".into());
        }
        let mut constraints = Vec::new();
        let mut maximum_items = u64::MAX;
        let mut shared_bytes_per_item = Vec::new();
        let mut maximum_by_launch = Vec::new();
        for (index, (dispatch, memory)) in launches
            .iter()
            .zip(execution.memory().launches())
            .enumerate()
        {
            let mut launch_maximum = u64::MAX;
            let unit = GroupDispatch::new(dispatch.work_items, dispatch.lanes_per_item, 1)?;
            let shared = memory.slots.iter().try_fold(0u64, |sum, declaration| {
                sum.checked_add(declaration.layout(&unit)?.shared_bytes_per_group)
                    .ok_or_else(|| "group family storage sum overflow".to_string())
            })?;
            shared_bytes_per_item.push(shared);
            for (resource, units_per_item, capacity) in [
                (
                    "threads",
                    dispatch.lanes_per_item,
                    execution.config.max_threads_per_threadgroup as u64,
                ),
                (
                    "threadgroup_bytes",
                    shared,
                    execution.config.max_threadgroup_bytes as u64,
                ),
            ] {
                if units_per_item == 0 {
                    continue;
                }
                let maximum = capacity / units_per_item;
                maximum_items = maximum_items.min(maximum);
                launch_maximum = launch_maximum.min(maximum);
                constraints.push(Constraint {
                    launch: index,
                    resource,
                    units_per_item,
                    capacity,
                    maximum_items: maximum,
                });
            }
            maximum_by_launch.push(launch_maximum);
        }
        if maximum_items == u64::MAX || maximum_items == 0 {
            return Err("group family has no bounded nonempty legal domain".into());
        }
        Ok(Self {
            execution,
            launches,
            shared_bytes_per_item,
            constraints,
            maximum_items,
            maximum_by_launch,
        })
    }
    fn grouping(&self, items_per_group: u64) -> Result<Grouping, String> {
        if !(1..=self.maximum_items).contains(&items_per_group) {
            return Err("grouping is outside the derived resource domain".into());
        }
        let launches = self
            .launches
            .iter()
            .map(|d| {
                let dispatch = GroupDispatch::new(d.work_items, d.lanes_per_item, items_per_group)?;
                // Metal's emitted slot is uint, including padding slots. Large
                // domains can admit some groupings and exclude others at this limit.
                if dispatch.dispatched_lanes() / dispatch.lanes_per_item > u64::from(u32::MAX) {
                    return Err("grouping padding exceeds Metal slot index width".into());
                }
                Ok(dispatch)
            })
            .collect::<Result<_, String>>()?;
        let shared_bytes_per_group = self
            .shared_bytes_per_item
            .iter()
            .map(|bytes| {
                bytes
                    .checked_mul(items_per_group)
                    .ok_or_else(|| "grouping storage overflow".to_string())
            })
            .collect::<Result<_, String>>()?;
        Ok(Grouping {
            items_per_group,
            launches,
            shared_bytes_per_group,
        })
    }
    /// Deterministic numerical order only. No performance ranking is implied.
    pub fn groupings(&self) -> impl Iterator<Item = Grouping> + '_ {
        (1..=self.maximum_items).filter_map(|items| self.grouping(items).ok())
    }
    /// Resolve grouping on the selected tree, without repeating lowering,
    /// ownership selection, or reduction selection. Rebuild memory quantities
    /// using the resulting launch geometry before allowing emission.
    pub fn select(&self, items_per_group: u64) -> Result<Execution, String> {
        let grouping = self.grouping(items_per_group)?;
        self.select_grouping(grouping)
    }

    /// Each launch has its own resource domain. Neither an earlier launch's
    /// shared arrays nor a common convenience grouping restricts this choice.
    fn grouping_values(&self, launch: usize) -> Result<Vec<u64>, String> {
        let maximum = *self
            .maximum_by_launch
            .get(launch)
            .ok_or("unknown grouping launch")?;
        let domain = &self.launches[launch];
        let alternatives = (1..=maximum)
            .filter(|&items| {
                // Padding is a typed index-width constraint, not a failed compiler
                // attempt used as a proxy for legality.
                u128::from(domain.work_items).div_ceil(u128::from(items)) * u128::from(items)
                    <= u128::from(u32::MAX)
            })
            .collect();
        Ok(alternatives)
    }

    /// Continue resolving launch geometry while retaining all prior execution
    /// choices. Main and merge launches remain independent legal domains.
    pub fn next(self: &Arc<Self>, selected: Vec<u64>) -> Result<Preparation<Execution>, String> {
        if selected.len() > self.launches.len() {
            return Err("unused Metal grouping decisions".into());
        }
        for (launch, value) in selected.iter().enumerate() {
            if self.grouping_values(launch)?.binary_search(value).is_err() {
                return Err(format!("grouping is outside launch {launch}'s resource domain"));
            }
        }
        if selected.len() == self.launches.len() {
            return Ok(Preparation::Execution(self.select_launches(&selected)?));
        }
        let launch = selected.len();
        let alternatives = GroupingChoices {
            launch,
            alternatives: self.grouping_values(launch)?,
            family: Arc::clone(self),
            selected,
        };
        Ok(Preparation::Choice {
            name: format!("Metal launch {launch} work items per threadgroup"),
            alternatives: seismic_accounting::selection::Domain::new(alternatives)?,
        })
    }

    pub fn select_launches(&self, items: &[u64]) -> Result<Execution, String> {
        if items.len() != self.launches.len() {
            return Err("one grouping is required for every launch".into());
        }
        let mut launches = Vec::new();
        let mut shared_bytes_per_group = Vec::new();
        for (index, (&items, domain)) in items.iter().zip(&self.launches).enumerate() {
            if self.grouping_values(index)?.binary_search(&items).is_err() {
                return Err(format!(
                    "grouping is outside launch {index}'s resource domain"
                ));
            }
            launches.push(GroupDispatch::new(
                domain.work_items,
                domain.lanes_per_item,
                items,
            )?);
            shared_bytes_per_group.push(
                self.shared_bytes_per_item[index]
                    .checked_mul(items)
                    .ok_or("grouping storage overflow")?,
            );
        }
        self.select_grouping(Grouping {
            items_per_group: items.first().copied().ok_or("empty launch domain")?,
            launches,
            shared_bytes_per_group,
        })
    }

    fn select_grouping(&self, grouping: Grouping) -> Result<Execution, String> {
        if self.launches == grouping.launches
            && u64::try_from(self.execution.config.sg_per_tg).ok() == Some(grouping.items_per_group)
        {
            // The validated resource family includes its original geometry.
            // Selecting it changes no allocation or terminal operation.
            return Ok(self.execution.clone());
        }
        let mut execution = self.execution.clone();
        execution.invalidate_terminal();
        execution.config.sg_per_tg =
            i64::try_from(grouping.items_per_group).map_err(|_| "group count overflow")?;
        let mut launches = grouping.launches.into_iter();
        for phase in &mut execution.phases {
            phase.dispatch = launches.next().ok_or("main dispatch missing")?;
            if phase.merge_dispatch.is_some() {
                phase.merge_dispatch = Some(launches.next().ok_or("merge dispatch missing")?);
            }
        }
        let allocation_choices = execution
            .memory
            .launches()
            .iter()
            .flat_map(|l| l.arrays.iter().map(|a| (a.id, a.slot)))
            .collect::<std::collections::HashMap<_, _>>();
        execution.memory = crate::memory::plan_selected(
            &execution.function.vars,
            &execution.function.body,
            &execution.phases,
            &execution.storage,
            &execution.reductions,
            execution.config.max_threadgroup_bytes as u64,
            &mut |choice| {
                allocation_choices
                    .get(&choice.allocation)
                    .copied()
                    .ok_or_else(|| "grouping changed an allocation identity".into())
            },
        )?.with_retained(&execution.retained, &execution.phases)?;
        if execution
            .memory
            .launches()
            .iter()
            .map(|l| l.shared_bytes_per_group)
            .ne(grouping.shared_bytes_per_group)
        {
            return Err("selected grouping disagrees with planned storage".into());
        }
        Ok(execution)
    }
    pub fn execution(&self) -> &Execution {
        &self.execution
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use seismic_accounting::workload::{Allocation, BufferBinding, DerivationLimits, ScalarWorkload};
    use seismic_lang::{Scope, program::{SourceFile, compile}};

    #[test]
    fn identity_grouping_shares_emission_and_account_but_changed_geometry_does_not() {
        let program = compile(&[SourceFile {
            path: "group-cache.seismic.portable".into(), scope: Scope::Portable,
            text: "fn evaluate(out: tensor[2] f32):\n  y = tile[2] f32\n  for i in owned(y): y[i] = 3.0\n  store(y,out)\n".into(),
        }], &[]).unwrap();
        let function = seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap();
        let execution = crate::execution::prepare(&function, crate::execution::Config { sg_per_tg: 1, ..Default::default() }).unwrap();
        let workload = ScalarWorkload {
            identity: "group-cache".into(), integer_domains: Vec::new(), scalars: Vec::new(),
            allocations: vec![Allocation { id: 1, bytes: 8, alignment: 4, known_bytes: Default::default() }],
            buffers: vec![BufferBinding { allocation: 1, offset: 0, bytes: 8 }],
        };
        let limits = DerivationLimits { instructions: 100_000, operations: 100_000 };
        let account = execution.relaxation(&workload, limits).unwrap();
        assert!(account.is_complete());
        let family = GroupFamily::derive(execution.clone()).unwrap();
        let unchanged = family.select(1).unwrap();
        assert!(Arc::ptr_eq(&execution.terminal, &unchanged.terminal));
        assert!(std::ptr::eq(crate::msl::prepare_execution(&execution).unwrap(), crate::msl::prepare_execution(&unchanged).unwrap()));
        assert!(Arc::ptr_eq(&account, &unchanged.relaxation(&workload, limits).unwrap()));
        let regrouped = family.select(2).unwrap();
        assert!(!Arc::ptr_eq(&execution.terminal, &regrouped.terminal));
        assert!(!Arc::ptr_eq(&account, &regrouped.relaxation(&workload, limits).unwrap()));
        let mut other_workload = workload.clone();
        other_workload.identity.push_str("-different");
        assert!(!Arc::ptr_eq(&account, &unchanged.relaxation(&other_workload, limits).unwrap()));
    }
}
