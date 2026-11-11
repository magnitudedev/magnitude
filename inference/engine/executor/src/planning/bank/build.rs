use super::{BankGroup, BankLayer, BankLayers, BankSites, BankTable, ExpertBankPlan, HintWeight, HostTier, NonEmpty, RecordForm, RoutedTensor, routed_experts};
use super::conveyor;
use super::records::{record_tensor, routed_layers};
use super::table::stored_table_heads;
use crate::error::{PlanError, UnstreamedExperts};
use crate::operators::routed::{hint_kinds, is_fused};
use crate::planning::ModelLoadPlan;
use crate::planning::resources::FlightDepth;
use crate::planning::weights::Holding;
use magnitude_family_contracts::{ExpertSelection, ModelDefinition, RouterInput, WeightRole};

impl ModelLoadPlan {
    /// The floor of the expert bank [`Self::with_expert_bank`] holds of
    /// these arguments ([`ExpertBankPlan::floor`]): its plan's, whatever its
    /// slots, so found of the bank of every expert.
    pub fn bank_floor(
        &self,
        definition: &ModelDefinition,
        places: u64,
        flights: FlightDepth,
        reservation: seismic::Reservation,
        host: HostTier,
    ) -> Result<u64, PlanError> {
        let whole = self.clone().with_expert_bank(definition, routed_experts(definition)?, places, flights, reservation, host)?;
        Ok(whole.bank.as_ref().expect("a banked plan holds its bank").floor())
    }

    /// The plan with every routed layer of the target's experts held in an
    /// expert bank of `slots` committed records per layer (at least its
    /// floor, [`ExpertBankPlan::floor`], at most the widest layer's experts)
    /// for prompt groups of up to `places` chain
    /// places, reserved as `reservation` says, beside the host tier `host`
    /// holds (a bank streams only beside one): each expert stack is planned
    /// `Banked`, its shape its group's records (their ceiling), charged its
    /// committed records (and its per-expert second-level scale, which the
    /// bank holds), its values the bank's. Each stored table's slots the
    /// device can reserve are held in the bank's records too
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
        places: u64,
        flights: FlightDepth,
        reservation: seismic::Reservation,
        host: HostTier,
    ) -> Result<Self, PlanError> {
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
            let planned = |kind| {
                self.target
                    .iter()
                    .find(|plan| plan.role == WeightRole { scope, kind })
            };
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
            layers.push(BankLayer {
                scope,
                group,
                experts: routed.experts,
                resident,
                hinted,
            });
        }
        let (_, selected) = shape.ok_or(PlanError::Unstreamed(UnstreamedExperts::Unrouted))?;
        // A routed layer streamed has experts to stream.
        let layers = BankLayers::new(layers).ok_or(PlanError::Topology("a streamed routed layer has no experts"))?;
        // Every stored table, its slots held in the bank's reservation.
        let mut tables = Vec::new();
        for weight in self.target.iter_mut().filter(|weight| weight.stored_in_slots()) {
            let heads = stored_table_heads(definition, weight.role)?;
            let step_rows = heads
                .checked_mul(magnitude_batching::MAX_CLASS_ROWS as u64)
                .ok_or(PlanError::Arithmetic("a step's table rows overflow"))?;
            let Some(table) = BankTable::of(weight, step_rows, reservation).map_err(PlanError::InvalidDefinition)? else {
                continue;
            };
            *weight = table.plan(reservation).map_err(PlanError::ResourcePlanning)?;
            tables.push(table);
        }
        // A conveyor or a scratch no record of its lender holds whole is
        // refused here, so every sizing past this may rely on them
        // (`ExpertBankPlan::prompt_records`).
        let structured = conveyor::Structured::new(selected, reservation, host, groups, layers)
            .map_err(PlanError::InvalidDefinition)?;
        let bank = ExpertBankPlan {
            structured,
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
        };
        for weight in &mut self.target {
            let Some((_, layer)) = bank.layer(weight.role.scope) else {
                continue;
            };
            if RoutedTensor::of(weight.role.kind).is_none() {
                continue;
            }
            weight.shape[0] = bank.records(layer.group);
            // The bank holds the values: empty, or every expert imported
            // into its records (`ExpertBankPlan::whole`).
            weight.source_bytes = 0;
            weight.holding = Holding::Banked { group: layer.group };
        }
        self.bank = Some(bank);
        self.with_bank_slots(slots, slots)
    }

    /// The plan with its expert bank at `slots` committed records per
    /// layer, growing to `ceiling` however few it holds now (from its floor
    /// to every expert of each layer): what a bank's slots move of its plan,
    /// each banked stack's charge of its committed records
    /// (`ExpertBankPlan::banked_bytes`). Everything else of the plan, its
    /// program plan included, is of the bank's structure alone.
    pub fn with_bank_slots(mut self, slots: u64, ceiling: u64) -> Result<Self, PlanError> {
        let bank = self.bank.as_mut().ok_or(PlanError::Topology("a plan without a bank has no slots"))?;
        if slots < bank.floor() || slots > ceiling || ceiling > bank.widest().get() {
            return Err(PlanError::Topology(
                "an expert bank holds from its floor to its ceiling, at most every expert of each layer",
            ));
        }
        bank.slots = slots;
        bank.ceiling = ceiling;
        bank.resident = slots == bank.widest().get();
        let bank = self.bank.as_ref().expect("the bank was just resized");
        for weight in &mut self.target {
            let Some((index, _)) = bank.layer(weight.role.scope) else {
                continue;
            };
            let Some(kind) = RoutedTensor::of(weight.role.kind) else {
                continue;
            };
            weight.resident_bytes = *bank
                .banked_bytes(index)
                .map_err(PlanError::InvalidDefinition)?
                .get(kind)
                .ok_or(PlanError::Topology("a banked layer's stack outside its record"))?;
        }
        Ok(self)
    }

    /// The plan committing `slots` of its bank at the load, its ceiling and
    /// its forms kept: the bank grows back to it while it serves, and the
    /// programs planned are the load's.
    pub fn with_load_slots(self, slots: u64) -> Result<Self, PlanError> {
        let bank = self.bank.as_ref().ok_or(PlanError::Topology("a plan without a bank has no slots"))?;
        let (ceiling, resident) = (bank.ceiling, bank.resident);
        let mut load = self.with_bank_slots(slots, ceiling)?;
        load.bank.as_mut().expect("the load was just resized").resident = resident;
        Ok(load)
    }

    /// The plan without its expert bank: each banked expert stack planned
    /// resident again, as the bank keeps it (`BankLayer::resident`).
    pub fn unbanked(mut self) -> Self {
        let Some(bank) = self.bank.take() else {
            return self;
        };
        for weight in &mut self.target {
            if let Some(resident) = bank.resident_plan(weight.role) {
                *weight = resident.clone();
            }
            if let Some(table) = bank.tables.iter().find(|table| table.role == weight.role) {
                *weight = table.fixed.clone();
            }
        }
        self
    }
}
