//! A union of typed execution regions with operation-local presence. All
//! activities, storage and static-order variables share the caller's resources.
use super::{Model, Schedule, static_order, symbolic::Encoding};
use crate::{algebra::{Algebra, ResourceUse, Symbolic}, objective::Objective};
use magnitude_solver::{model::{Constraint, Domain, LinearTerm, Literal, ModelBuilder, ObligationKind, VarId}, scheduling::{Activity, Event, Demand, Lifetime, SchedulingConstraint}};
use std::sync::Arc;

pub struct Binding {
    original: Arc<Model>,
    presence: Vec<Option<VarId>>,
    starts: Vec<VarId>,
}
impl Binding {
    pub fn append(builder: &mut ModelBuilder, encoding: &mut Encoding, original: Arc<Model>, presence: Vec<Option<VarId>>) -> Result<Self, String> {
        if original.operations.len() != presence.len() { return Err("guarded operation arity mismatch".into()); }
        original.validate()?;
        for reason in &original.unmapped {
            builder.obligation(Vec::new(), ObligationKind::Analysis, reason.clone());
        }
        if original.resources != encoding.resources() { return Err("guarded operations have a different resource scope".into()); }
        encoding.bind_timebase(&original.timebase).map_err(|e| e.to_string())?;
        let horizon = encoding.horizon();
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for (index, operation) in original.operations.iter().enumerate() {
            let active = presence[index];
            let append = |builder: &mut ModelBuilder| -> Result<_, String> {
                let start = builder.local_variable(format!("guarded.{index}.start"), Domain::interval(0, horizon).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                let end = builder.local_variable(format!("guarded.{index}.end"), Domain::interval(0, horizon).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                let duration = Symbolic::new(builder, "guarded.duration").constant(operation.latency).map_err(|e|e.to_string())?;
                let activity = Activity { start, end, duration: duration.id(), presence: active };
                encoding.activity(builder, activity.clone());
                let mut uses = Vec::new();
                for reservation in &operation.reservations {
                    let mut algebra = Symbolic::new(builder, "guarded.service");
                    uses.push(ResourceUse { resource: reservation.resource, offset: reservation.offset,
                        duration: algebra.constant(reservation.duration).map_err(|e| e.to_string())?,
                        units: algebra.constant(reservation.units).map_err(|e| e.to_string())? });
                }
                encoding.service(builder, &activity, &uses).map_err(|e|e.to_string())?;
                Ok((start,end))
            };
            let (start, end) = match active { Some(active) => builder.when(Literal::new(active,1), append), None => { let mut append = append; append(builder) } }?;
            starts.push(start); ends.push(end);
        }
        for (index, operation) in original.operations.iter().enumerate() {
            for &before in &operation.predecessors {
                edge(builder, Event { time: ends[before], presence: presence[before] }, Event { time: starts[index], presence: presence[index] });
            }
            for &before in &operation.start_predecessors {
                edge(builder, Event { time: starts[before], presence: presence[before] }, Event { time: starts[index], presence: presence[index] });
            }
        }
        let event = |event: super::Event| Event { time: match event.point { super::Point::Start => starts[event.operation], super::Point::Completion => ends[event.operation] }, presence: presence[event.operation] };
        for lifetime in &original.lifetimes {
            if presence[lifetime.begin.operation] != presence[lifetime.end.operation] {
                return Err("a conditional lifetime must have matching entry and exit activation".into());
            }
            encoding.reservation(lifetime.resource, Lifetime { begin: event(lifetime.begin), end: event(lifetime.end), demand: Demand::Constant(lifetime.units) }.reservation()).map_err(|e|e.to_string())?;
        }
        orders(builder, &original.static_orders, &starts, &presence)?;
        Ok(Self { original, presence, starts })
    }
    pub fn reconstruct(&self, values: &[i64], lower_bound: u64) -> Result<Objective, String> {
        if !self.original.unmapped.is_empty() {
            return Err("guarded execution still has unmapped primitive timing".into());
        }
        let mut original = (*self.original).clone();
        let active = self.presence.iter().map(|presence| match presence {
            None => Ok(true), Some(id) => match values.get(id.0) { Some(0) => Ok(false), Some(1) => Ok(true), _ => Err("missing guarded operation presence".to_owned()) }
        }).collect::<Result<Vec<_>,_>>()?;
        let mut remap = vec![None; active.len()];
        let mut operations = Vec::new();
        let mut starts = Vec::new();
        for (index, operation) in original.operations.iter().enumerate() {
            if active[index] {
                remap[index] = Some(operations.len());
                operations.push(operation.clone());
                starts.push(values.get(self.starts[index].0).and_then(|&n|u64::try_from(n).ok()).ok_or("missing guarded operation start")?);
            }
        }
        for operation in &mut operations {
            operation.predecessors = operation.predecessors.iter().filter_map(|&p|remap[p]).collect();
            operation.start_predecessors = operation.start_predecessors.iter().filter_map(|&p|remap[p]).collect();
        }
        for order in &mut original.static_orders {
            order.visits.retain(|visit| visit.roots.iter().flatten().any(|&id|active[id]));
            for visit in &mut order.visits {
                for roots in &mut visit.roots { *roots = roots.iter().filter_map(|&id|remap[id]).collect(); }
                if visit.roots.iter().any(Vec::is_empty) { return Err("conditional block visit has incomplete instruction participation".into()); }
            }
        }
        original.lifetimes.retain(|lifetime|active[lifetime.begin.operation] && active[lifetime.end.operation]);
        for lifetime in &mut original.lifetimes {
            lifetime.begin.operation = remap[lifetime.begin.operation].ok_or("inactive lifetime begin")?;
            lifetime.end.operation = remap[lifetime.end.operation].ok_or("inactive lifetime end")?;
        }
        original.operations = operations;
        let completion = original.operations.iter().zip(&starts).try_fold(0u64, |end,(op,start)|start.checked_add(op.latency).map(|value|end.max(value)).ok_or("guarded completion overflow"))?;
        Objective::from_flat(Arc::new(original), Schedule { starts, completion }, lower_bound)
    }
}
fn edge(builder: &mut ModelBuilder, before: Event, after: Event) {
    builder.constraint(Constraint::Schedule(SchedulingConstraint::Precedence { before, after, lag: 0 }));
}
fn orders(builder: &mut ModelBuilder, orders: &[static_order::Constraint], starts: &[VarId], presence: &[Option<VarId>]) -> Result<(),String> {
    for (block,order) in orders.iter().enumerate() {
        if order.instructions.is_empty() || order.visits.is_empty() {continue;}
        let active=block_presence(builder,order,presence)?;
        let append=|builder:&mut ModelBuilder|->Result<(),String> {
            let maximum=i64::try_from(order.instructions.len()-1).map_err(|_|"instruction order exceeds integer range")?;
            let ranks=(0..order.instructions.len()).map(|index|builder.local_variable(format!("guarded.block.{block}.rank.{index}"),Domain::interval(0,maximum).expect("nonnegative rank domain")).map_err(|error|error.to_string())).collect::<Result<Vec<_>,_>>()?;
            for &(before,after) in &order.predecessors {
                builder.constraint(Constraint::LinearLe {terms:vec![LinearTerm::new(ranks[before],1),LinearTerm::new(ranks[after],-1)],rhs:-1});
            }
            for left in 0..ranks.len() {for right in left+1..ranks.len() {
                let first=builder.local_variable(format!("guarded.block.{block}.before.{left}.{right}"),Domain::boolean()).map_err(|error|error.to_string())?;
                for (value,a,b) in [(1,left,right),(0,right,left)] {
                    builder.guarded_constraint(vec![Literal::new(first,value)],Constraint::LinearLe {terms:vec![LinearTerm::new(ranks[a],1),LinearTerm::new(ranks[b],-1)],rhs:-1});
                    builder.when(Literal::new(first,value),|builder| {
                        for visit in &order.visits {for &before in &visit.roots[a] {for &after in &visit.roots[b] {
                            edge(builder,Event {time:starts[before],presence:presence[before]},Event {time:starts[after],presence:presence[after]});
                        }}}
                    });
                }
            }}
            Ok(())
        };
        match active {Some(active)=>builder.when(Literal::new(active,1),append),None=>append(builder)}?;
    }
    Ok(())
}
fn block_presence(builder:&mut ModelBuilder,order:&static_order::Constraint,presence:&[Option<VarId>])->Result<Option<VarId>,String> {
    let mut guards=std::collections::BTreeSet::new();
    for root in order.visits.iter().flat_map(|visit|visit.roots.iter().flatten()) {
        match presence.get(*root).ok_or("static block names an absent operation")? {
            None=>return Ok(None),Some(active)=>{guards.insert(*active);},
        }
    }
    let mut inactive=Vec::new();
    for guard in guards {
        let inverse=builder.variable("guarded.block.inactive",Domain::boolean());
        builder.constraint(Constraint::NotEqual {left:guard,right:inverse});inactive.push(inverse);
    }
    let absent=builder.variable("guarded.block.absent",Domain::boolean());
    builder.constraint(Constraint::BoolAnd {output:absent,inputs:inactive});
    let active=builder.variable("guarded.block.active",Domain::boolean());
    builder.constraint(Constraint::NotEqual {left:active,right:absent});
    Ok(Some(active))
}
