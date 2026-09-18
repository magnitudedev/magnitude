//! Aggregate demand obtained by relaxing this model's reservations and ordering.
//! Multiplicity stays symbolic: one retained implementation operation can account
//! for billions of mandatory instances without creating an event per instance.
use super::{Model, Operation, Resource, Timebase};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Demand {
    timebase: Timebase,
    resources: Vec<Resource>,
    occupancy: Vec<u128>,
    latency: u64,
}
impl Demand {
    pub fn new(timebase: Timebase, resources: Vec<Resource>) -> Result<Self, String> {
        // The same model boundary checks the capacities and units used by exact
        // schedules; this projection has no independently interpreted resources.
        Model {
            relationship: crate::authority::ModelRelationship::OptimisticRelaxation,
            identity: "resource demand relaxation".into(),
            timebase: timebase.clone(),
            resources: resources.clone(),
            operations: Vec::new(),
            lifetimes: Vec::new(),
            static_orders: Vec::new(),
            unmapped: Vec::new(),
        }
        .validate()?;
        let occupancy = vec![0; resources.len()];
        Ok(Self {
            timebase,
            resources,
            occupancy,
            latency: 0,
        })
    }
    pub fn timebase(&self) -> &Timebase {
        &self.timebase
    }

    /// The caller derives a necessary instance count from the retained execution
    /// domain. Dropping ordering and unlisted operations weakens the bound. This
    /// cannot be used as a feasible upper or as a claim about native mappings.
    pub fn include(&mut self, operation: &Operation, minimum_instances: u64) -> Result<(), String> {
        super::validate_reservations(operation, self.resources.len())?;
        if minimum_instances == 0 {
            return Ok(());
        }
        self.latency = self.latency.max(operation.latency);
        for r in &operation.reservations {
            self.add_occupancy(r.resource, r.units, r.duration, minimum_instances)?;
        }
        Ok(())
    }
    pub(super) fn add_occupancy(
        &mut self,
        resource: usize,
        units: u64,
        duration: u64,
        instances: u64,
    ) -> Result<(), String> {
        let term = u128::from(units)
            .checked_mul(u128::from(duration))
            .and_then(|n| n.checked_mul(u128::from(instances)))
            .ok_or("resource demand overflow")?;
        self.occupancy[resource] = self.occupancy[resource]
            .checked_add(term)
            .ok_or("resource demand overflow")?;
        Ok(())
    }
    pub fn lower_bound(&self) -> Result<u64, String> {
        self.occupancy.iter().zip(&self.resources).try_fold(
            self.latency,
            |floor, (&work, resource)| {
                let duration = u64::try_from(work.div_ceil(u128::from(resource.capacity)))
                    .map_err(|_| "resource lower bound overflow")?;
                Ok(floor.max(duration))
            },
        )
    }
}
