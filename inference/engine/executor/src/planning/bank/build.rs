use super::{BankGroup, BankLayer, BankLayers, BankPaths, BankSites, BankSubstrate, BankTable, ExpertBankPlan, HintWeight, NonEmpty, RecordForm, routed_experts};
use super::conveyor;
use super::records::{record_tensor, routed_layers};
use super::table::stored_table_heads;
use crate::error::{PlanError, UnstreamedExperts};
use crate::operators::routed::{hint_kinds, is_fused};
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
        let whole = self.clone().with_expert_bank(definition, routed_experts(definition)?, None, places, flights, substrate)?;
        Ok(whole.bank.as_ref().expect("a banked plan holds its bank").floor())
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
        let BankSubstrate { backend, allocations, host } = substrate;
        if self.bank.is_some() {
            return Err(PlanError::Topology("the load plan already streams its experts"));
        }
        let mut layers = Vec::new();
        let mut groups: Vec<BankGroup> = Vec::new();
        // Every streamed layer's selection order: a layer's hints rank the
        // next layer's choices by this layer's, reading the next layer's
        // weights as its own elements (`routed_select_streamed`,
        // `routed_route_streamed`); and every layer's record form and
        // selected experts, which the bank's entries and words share.
        let mut order = None;
        let mut shape = None;
        for (index, scope, routed) in routed_layers(definition) {
            let form = RecordForm::of(&routed.expert_up);
            if *shape.get_or_insert((form, routed.selected)) != (form, routed.selected) {
                return Err(PlanError::Unstreamed(UnstreamedExperts::Shape(index)));
            }
            let planned = |kind| self.target_weight(WeightRole { scope, kind });
            let hinted = hint_kinds(routed)
                .into_iter()
                .map(|kind| HintWeight {
                    kind,
                    element: planned(kind).map(|plan| plan.resident),
                })
                .collect::<Vec<_>>();
            let router = &routed.router;
            let selection = (
                is_fused(routed),
                router.score,
                matches!(router.selection, ExpertSelection::TopK { bias: Some(_) }),
                matches!(router.input, RouterInput::Residual(_)),
                hinted.clone(),
            );
            if *order.get_or_insert_with(|| selection.clone()) != selection {
                return Err(PlanError::Unstreamed(UnstreamedExperts::SelectionOrder(index)));
            }
            let resident = form.each(|tensor| {
                planned(tensor.kind())
                    .cloned()
                    .ok_or(PlanError::Topology("a routed layer lacks an expert stack"))
            })?;
            let record = resident.as_ref().try_map(|_, plan| record_tensor(plan, routed.experts))?;
            let group = match groups.iter().position(|group| group.record == record) {
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
            layers.push(BankLayer {
                scope,
                group,
                experts: routed.experts,
                resident,
                hinted,
                shared_down,
            });
        }
        let (_, selected) = shape.ok_or(PlanError::Unstreamed(UnstreamedExperts::Unrouted))?;
        // A routed layer streamed has experts to stream.
        let layers = BankLayers::new(layers).ok_or(PlanError::Topology("a streamed routed layer has no experts"))?;
        // A conveyor or a scratch without a layout, or a pool without a
        // granule, is refused here, so every sizing past this may rely on
        // them (`ExpertBankPlan::prompt_records`, `ExpertBankPlan::granule`).
        let structured = conveyor::Structured::new(selected, host, groups, layers, window_slots)
            .map_err(PlanError::InvalidDefinition)?;
        // Every stored table whose slots the host fills in place, held in
        // granules of the bank's pool: only where the host writes it.
        let mut tables = Vec::new();
        let granule = structured.granule();
        // The records pools' blocks, which every charge of their granules is
        // whole blocks of: the rule a load opens them by
        // (`memory::ledger::Ledgers::open_pool`).
        let block = seismic::block_bytes(seismic::BlockSizing { granule, allocations })
            .map_err(|error| PlanError::ResourcePlanning(format!("the bank's records pool blocks: {error:?}")))?;
        for weight in self.target.iter_mut().filter(|weight| weight.stored_in_slots() && matches!(host, super::HostTier::Unified(_))) {
            let heads = stored_table_heads(definition, weight.role)?;
            let step_rows = heads
                .checked_mul(magnitude_batching::MAX_CLASS_ROWS as u64)
                .ok_or(PlanError::Arithmetic("a step's table rows overflow"))?;
            let Some(table) = BankTable::of(weight, step_rows, granule).map_err(PlanError::InvalidDefinition)? else {
                continue;
            };
            *weight = table.plan(granule).map_err(PlanError::ResourcePlanning)?;
            tables.push(table);
        }
        let bank = ExpertBankPlan {
            structured,
            backend,
            allocations,
            block,
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
        };
        for layer in bank.layers() {
            let records = bank.records(layer.group);
            for (kind, _) in layer.resident.iter() {
                if let Some(weight) = self.target_weight_mut(WeightRole { scope: layer.scope, kind: kind.kind() }) {
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

    /// The plan with its expert bank at `slots` committed records per
    /// layer, growing to `ceiling` however few it holds now (from its floor
    /// to every expert of each layer): what a bank's slots move of its plan,
    /// each banked stack's charge of its committed records
    /// (`ExpertBankPlan::banked_bytes`). Everything else of the plan, its
    /// program plan included, is of the bank's structure alone.
    pub fn with_bank_slots(mut self, slots: u64, ceiling: u64) -> Result<Self, PlanError> {
        let mut bank = self.bank.take().ok_or(PlanError::Topology("a plan without a bank has no slots"))?;
        if slots < bank.floor() || slots > ceiling || ceiling > bank.widest().get() {
            return Err(PlanError::Topology(
                "an expert bank holds from its floor to its ceiling, at most every expert of each layer",
            ));
        }
        bank.slots = slots;
        bank.ceiling = ceiling;
        bank.resident = slots == bank.widest().get();
        for (index, layer) in bank.layers().iter().enumerate() {
            let bytes = bank
                .banked_bytes(index)
                .map_err(PlanError::InvalidDefinition)?;
            for (kind, _) in layer.resident.iter() {
                if let Some(weight) = self.target_weight_mut(WeightRole { scope: layer.scope, kind: kind.kind() }) {
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
                if let Some(weight) = self.target_weight_mut(WeightRole { scope: layer.scope, kind: kind.kind() }) {
                    weight.shape[0] = records;
                }
            }
        }
        let slots = bank.slots.max(bank.floor()).min(bank.ceiling);
        let ceiling = bank.ceiling;
        self.bank = Some(bank);
        self.with_front_stagings().with_bank_slots(slots, ceiling)
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
        (built.host_slots, built.pass, built.resident) = (bank.host_slots, bank.pass, bank.resident);
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
                if let Some(weight) = self.target_weight_mut(resident.role) {
                    *weight = resident.clone();
                }
            }
        }
        for table in &bank.tables {
            if let Some(weight) = self.target_weight_mut(table.role) {
                *weight = table.fixed.clone();
            }
        }
        self.with_front_stagings()
    }
}
