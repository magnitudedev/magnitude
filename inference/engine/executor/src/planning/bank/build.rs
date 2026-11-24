use super::{BankGroup, BankLayer, BankLayers, BankPaths, BankSites, BankSubstrate, BankTable, ExpertBankPlan, NonEmpty, RecordForm, routed_experts};
use super::conveyor;
use super::records::{record_tensor, routed_layers};
use super::table::stored_table_heads;
use crate::error::{PlanError, UnstreamedExperts};
use crate::operators::routed::is_fused;
use crate::planning::ModelLoadPlan;
use crate::planning::resources::FlightDepth;
use crate::planning::weights::Holding;
use magnitude_family_contracts::{ExpertSelection, ModelDefinition, RouterInput, WeightKind, WeightRole};

impl ModelLoadPlan {
    /// The floor of the expert bank [`Self::with_expert_bank`] holds of
    /// these arguments ([`ExpertBankPlan::floor`]): its plan's, whatever its
    /// slots, so found of the bank of every expert.
    pub fn bank_floor(
        &self,
        definition: &ModelDefinition,
        places: u64,
        flights: FlightDepth,
        substrate: BankSubstrate,
    ) -> Result<u64, PlanError> {
        let whole = self.clone().with_whole_expert_bank(definition, None, places, flights, substrate)?;
        Ok(whole.bank.as_ref().expect("a banked plan holds its bank").floor())
    }

    /// The plan with every routed layer of the target's experts held whole in
    /// an expert bank (`slots = ceiling = bank.max_slots()`).
    pub fn with_whole_expert_bank(
        self,
        definition: &ModelDefinition,
        window_slots: Option<u64>,
        places: u64,
        flights: FlightDepth,
        substrate: BankSubstrate,
    ) -> Result<Self, PlanError> {
        let experts = routed_experts(definition)?;
        let load = self.with_expert_bank(definition, experts, window_slots, places, flights, substrate)?;
        let max_slots = load.bank.as_ref().expect("a banked plan holds its bank").max_slots();
        if max_slots == experts {
            return Ok(load);
        }
        load.with_bank_slots(max_slots, max_slots)
    }

    /// The plan with every routed layer of the target's experts held in an
    /// expert bank of `slots` committed records per layer (at least its
    /// floor, [`ExpertBankPlan::floor`], at most the widest layer's experts)
    /// for prompt groups of up to `places` chain places, its scratch
    /// `window_slots` experts of a layer at once (`None`: the widest
    /// layer's), on `substrate`:
    /// its records in its store's records pool, beside the host tier it holds (a bank streams only
    /// beside one). Each expert stack is planned `Banked`, its shape its
    /// group's records (their ceiling), charged its committed records (and
    /// its per-expert second-level scale, which the bank holds), its values
    /// the bank's. Where the host writes the bank's pool in place (unified
    /// memory), each stored table's slots are held in granules of it too
    /// (`BankTable`), so one owner prices the table against the slots.
    /// Layers whose record form or selected experts differ, layers that
    /// select in different orders or whose weights a route reads of the next
    /// layer differ in element (`BankLayer::hinted`), or an expert stack
    /// whose experts are not each one contiguous range as stored and as
    /// resident, are refused.
    pub fn with_expert_bank(
        mut self,
        definition: &ModelDefinition,
        slots: u64,
        window_slots: Option<u64>,
        places: u64,
        flights: FlightDepth,
        substrate: BankSubstrate,
    ) -> Result<Self, PlanError> {
        let BankSubstrate { backend, allocations, host, has_side_queue } = substrate;
        if self.bank.is_some() {
            return Err(PlanError::Topology("the load plan already streams its experts"));
        }
        let mut layers: Vec<BankLayer> = Vec::new();
        let mut groups: Vec<BankGroup> = Vec::new();
        // Every streamed layer's selection order: a layer's hints rank the
        // next layer's choices by this layer's, reading the next layer's
        // weights as its own elements (`routed_select_streamed`,
        // `routed_route_streamed`); and every layer's record form and
        // selected experts, which the bank's entries and words share.
        let mut order = None;
        let mut target_shape = None;
        let mut max_selected = 0u64;
        for (index, scope, routed) in routed_layers(definition) {
            let is_draft = matches!(
                scope,
                magnitude_family_contracts::WeightScope::DraftSublayer(_)
                    | magnitude_family_contracts::WeightScope::HeadSublayer(_)
            );
            if is_draft && self.head.is_none() {
                continue;
            }
            let form = RecordForm::of(&routed.expert_up);
            if is_draft {
                let Some((target_form, _)) = target_shape else {
                    continue;
                };
                if form != target_form {
                    continue;
                }
            } else if *target_shape.get_or_insert((form, routed.selected)) != (form, routed.selected) {
                return Err(PlanError::Unstreamed(UnstreamedExperts::Shape(index)));
            }
            let planned = |kind| {
                let role = WeightRole { scope, kind };
                if is_draft {
                    self.head_weight(role)
                } else {
                    self.target_weight(role)
                }
            };
            if !is_draft {
                let router = &routed.router;
                let selection = (
                    is_fused(routed),
                    router.score,
                    matches!(router.selection, ExpertSelection::TopK { bias: Some(_) }),
                    matches!(router.input, RouterInput::Residual(_)),
                );
                if *order.get_or_insert_with(|| selection.clone()) != selection {
                    return Err(PlanError::Unstreamed(UnstreamedExperts::SelectionOrder(index)));
                }
            }
            let resident = form.each(|tensor| {
                planned(tensor.kind())
                    .cloned()
                    .ok_or(PlanError::Topology("a routed layer lacks an expert stack"))
            })?;
            // A draft or embedded head layer whose expert stacks were
            // transformed on import (e.g. `narrow_head_routed_experts` ->
            // `ImportTransform::DraftQ4_0`, where `upload != source`) cannot be
            // streamed directly as raw stored records; leave it `Holding::Whole`
            // instead of refusing the entire model's expert bank.
            let record = match resident.as_ref().try_map(|_, plan| record_tensor(plan, routed.experts)) {
                Ok(record) => record,
                Err(PlanError::Unstreamed(_)) if is_draft => continue,
                Err(error) => return Err(error),
            };
            let group = match groups.iter().position(|group| {
                group.record == record && layers[group.layers[0]].is_draft() == is_draft
            }) {
                Some(group) => {
                    groups[group].layers.push(layers.len());
                    group
                }
                None => {
                    groups.push(BankGroup {
                        record,
                        layers: NonEmpty::one(layers.len()),
                    });
                    groups.len() - 1
                }
            };
            let shared_down = is_fused(routed)
                .then(|| planned(WeightKind::SharedDown).map(|plan| (plan.resident, plan.resident_bytes, plan.shape.clone())))
                .flatten();
            max_selected = max_selected.max(routed.selected);
            layers.push(BankLayer {
                scope,
                group,
                experts: routed.experts,
                selected: routed.selected,
                resident,
                shared_down,
            });
        }
        let _ = target_shape.ok_or(PlanError::Unstreamed(UnstreamedExperts::Unrouted))?;
        let selected = max_selected;
        // A routed layer streamed has experts to stream.
        let layers = BankLayers::new(layers).ok_or(PlanError::Topology("a streamed routed layer has no experts"))?;
        // A conveyor or a scratch without a layout, or a pool without a
        // granule, is refused here, so every sizing past this may rely on
        // them (`ExpertBankPlan::prompt_records`, `ExpertBankPlan::granule`).
        let structured = conveyor::Structured::new(selected, host, groups, layers, window_slots, self.decode_rows)
            .map_err(PlanError::InvalidDefinition)?;
        // Every stored table whose slots the host fills in place, held in
        // granules of the bank's largest records pool: only where the host
        // writes it.
        let mut tables = Vec::new();
        let table_granule = structured.table_granule();
        // Each distinct granule's records pool blocks, which every charge of
        // its granules is whole blocks of: the rule a load opens them by
        // (`memory::ledger::DeviceLedgers::open_records_pools_at`).
        let blocks = super::size_pool_blocks(structured.granules(), allocations)
            .map_err(PlanError::ResourcePlanning)?;
        for weight in self.target.iter_mut().filter(|weight| weight.stored_in_slots() && matches!(host, super::HostTier::Unified(_))) {
            let heads = stored_table_heads(definition, weight.role)?;
            let step_rows = heads
                .checked_mul(magnitude_batching::MAX_CLASS_ROWS as u64)
                .ok_or(PlanError::Arithmetic("a step's table rows overflow"))?;
            let Some(table) = BankTable::of(weight, step_rows, table_granule).map_err(PlanError::InvalidDefinition)? else {
                continue;
            };
            *weight = table.plan(table_granule).map_err(PlanError::ResourcePlanning)?;
            if let Some(existing) = tables.iter_mut().find(|existing: &&mut BankTable| existing.shares_storage_with(&table)) {
                existing.merge_shared(&table);
            } else {
                tables.push(table);
            }
        }
        let bank = ExpertBankPlan {
            structured,
            backend,
            has_side_queue,
            allocations,
            blocks,
            slots,
            ceiling: slots,
            resident: false,
            host_slots: 0,
            host_cap: None,
            places,
            pass: crate::PassLaunches::new(
                usize::try_from(places)
                    .ok()
                    .and_then(std::num::NonZeroUsize::new)
                    .ok_or(PlanError::Arithmetic("a bank's places are one to usize::MAX"))?,
            ),
            flights,
            tables,
            sites: BankSites::Unadmitted,
            paths: BankPaths::Unpriced,
            recording: self.recorded_routes,
            decode_hits_min_bytes: 0,
        };
        for layer in bank.layers() {
            let records = bank.records(layer.group);
            for (kind, _) in layer.resident.iter() {
                let role = WeightRole { scope: layer.scope, kind: kind.kind() };
                if let Some(weight) = self.bank_weight_mut(role) {
                    weight.shape[0] = records;
                    // The bank holds the values: empty, or every expert imported
                    // into its records (`ExpertBankPlan::whole`).
                    weight.source_bytes = 0;
                    weight.holding = Holding::Banked { group: layer.group };
                }
            }
        }
        self.bank = Some(bank);
        self.rebuild_bank_table_index();
        self.with_front_stagings().with_bank_slots(slots, slots)
    }

    /// Mutable access to a banked weight's plan: in `head` for draft and head
    /// routed layers, in `target` for target routed layers.
    fn bank_weight_mut(&mut self, role: WeightRole) -> Option<&mut crate::planning::weights::WeightPlan> {
        match role.scope {
            magnitude_family_contracts::WeightScope::DraftSublayer(_)
            | magnitude_family_contracts::WeightScope::HeadSublayer(_) => self.head_weight_mut(role),
            _ => self.target_weight_mut(role),
        }
    }

    /// The plan with its expert bank at `slots` committed records per
    /// layer, growing to `ceiling` however few it holds now (from its floor
    /// to every expert of each layer): what a bank's slots move of its plan,
    /// each banked stack's charge of its committed records
    /// (`ExpertBankPlan::banked_bytes`). Everything else of the plan, its
    /// program plan included, is of the bank's structure alone.
    pub fn with_bank_slots(mut self, slots: u64, ceiling: u64) -> Result<Self, PlanError> {
        let mut bank = self.bank.take().ok_or(PlanError::Topology("a plan without a bank has no slots"))?;
        if slots < bank.floor() || slots > ceiling || ceiling > bank.max_slots() {
            return Err(PlanError::Topology(
                "an expert bank holds from its floor to its ceiling, at most every expert of each layer",
            ));
        }
        bank.slots = slots;
        bank.ceiling = ceiling;
        bank.resident = bank.whole_at(slots);
        for (index, layer) in bank.layers().iter().enumerate() {
            let bytes = bank
                .banked_bytes(index)
                .map_err(PlanError::InvalidDefinition)?;
            for (kind, _) in layer.resident.iter() {
                let role = WeightRole { scope: layer.scope, kind: kind.kind() };
                if let Some(weight) = self.bank_weight_mut(role) {
                    weight.resident_bytes = *bytes
                        .get(kind)
                        .ok_or(PlanError::Topology("a banked layer's stack outside its record"))?;
                }
            }
        }
        self.bank = Some(bank);
        Ok(self)
    }

    /// The plan committing `slots` of its bank and `host_slots` of its host
    /// tier at the load, its ceiling and its forms kept: the bank grows back
    /// to it while it serves, and the programs planned are the load's.
    pub fn with_load_slots(self, slots: u64, host_slots: u64) -> Result<Self, PlanError> {
        let bank = self.bank.as_ref().ok_or(PlanError::Topology("a plan without a bank has no slots"))?;
        let (ceiling, resident, planned_host) = (bank.ceiling, bank.resident, bank.host_slots);
        if host_slots > planned_host {
            return Err(PlanError::Topology("load host slots exceed the planned host tier"));
        }
        let mut load = self.with_bank_slots(slots, ceiling)?;
        let built = load.bank.as_mut().expect("the load was just resized");
        built.resident = resident;
        built.host_slots = host_slots;
        Ok(load)
    }

    /// The plan with its expert bank's path `paths` (`ExpertBankPlan::paths`):
    /// its staging, its tier's reserve, and each banked stack's shape and
    /// committed records (clamped to at least the bank's floor) with them.
    pub(crate) fn with_bank_paths(mut self, paths: BankPaths) -> Result<Self, PlanError> {
        let mut bank = self.bank.take().ok_or(PlanError::Topology("a plan without a bank has no path"))?;
        bank.paths = paths;
        for layer in bank.layers() {
            let records = bank.records(layer.group);
            for (kind, _) in layer.resident.iter() {
                let role = WeightRole { scope: layer.scope, kind: kind.kind() };
                if let Some(weight) = self.bank_weight_mut(role) {
                    weight.shape[0] = records;
                }
            }
        }
        let slots = bank.slots.max(bank.floor()).min(bank.ceiling);
        let ceiling = bank.ceiling;
        self.bank = Some(bank);
        self.with_front_stagings().with_bank_slots(slots, ceiling)
    }

    pub(crate) fn with_decode_hits_min_bytes(mut self, min_bytes: u64) -> Self {
        if let Some(bank) = self.bank.as_mut() {
            bank.decode_hits_min_bytes = min_bytes;
        }
        self
    }

    /// The plan with its experts banked as `bank` holds them, at its path,
    /// slots, ceiling, host slots, launches and residency, its scratch
    /// `window_slots` experts of a layer at once: the bank
    /// [`Self::with_expert_bank`] builds of this plan's weights unbanked,
    /// whose floor is of that window and path from the first.
    pub(crate) fn banked_as(self, definition: &ModelDefinition, bank: &ExpertBankPlan, window_slots: u64) -> Result<Self, PlanError> {
        let mut load = self
            .unbanked()
            .with_expert_bank(definition, bank.slots, Some(window_slots), bank.places, bank.flights, bank.substrate())?
            .with_bank_paths(bank.paths)?
            .with_bank_slots(bank.slots, bank.ceiling)?;
        let built = load.bank.as_mut().expect("the load was just banked");
        (built.host_slots, built.pass, built.resident, built.decode_hits_min_bytes) =
            (bank.host_slots, bank.pass, bank.resident, bank.decode_hits_min_bytes);
        Ok(load)
    }

    /// The plan without its expert bank: each banked expert stack planned
    /// resident again, as the bank keeps it (`BankLayer::resident`).
    pub fn unbanked(mut self) -> Self {
        let Some(bank) = self.bank.take() else {
            return self;
        };
        self.rebuild_bank_table_index();
        for layer in bank.layers() {
            for (_, resident) in layer.resident.iter() {
                if let Some(weight) = self.bank_weight_mut(resident.role) {
                    *weight = resident.clone();
                }
            }
        }
        for table in &bank.tables {
            for role in table.roles() {
                if let Some(weight) = self.target_weight_mut(role) {
                    let mut fixed = table.fixed.clone();
                    fixed.role = role;
                    *weight = fixed;
                }
            }
        }
        self.with_front_stagings()
    }
}
