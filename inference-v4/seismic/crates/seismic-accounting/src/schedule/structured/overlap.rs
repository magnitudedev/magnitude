//! Original-model validation of shifted reservation profiles.
//! Translation and witness checks own no scheduling search.
use super::*;

fn offset_model(model: &Structured, children: &[Arc<Plan>], profile_limit: u64) -> Result<Option<Model>, String> {
    let mut result = model.empty_model();
    for (index, child) in children.iter().enumerate() {
        let Plan::Selected(witness) = child.as_ref() else { return Err("offset child witness missing".into()); };
        let Some(profile) = witness.reservation_profile(profile_limit)? else { return Ok(None); };
        result.operations.push(Operation { name: format!("parallel child @{index}"), predecessors: vec![], start_predecessors: vec![],
            latency: witness.completion, reservations: profile.intervals.into_iter().map(|(resource, begin, end, units)| Reservation {
                resource, offset: begin, duration: end - begin, units,
            }).collect() });
    }
    result.validate()?;
    Ok(Some(result))
}
pub(super) fn checked_peak(model: &Structured, children: &[Arc<Plan>], starts: &[u64], duration: u64, profile_limit: u64) -> Result<Vec<u64>, String> {
    let model = offset_model(model, children, profile_limit)?.ok_or("offset profile no longer fits its derivation limit")?;
    let schedule = Schedule { starts: starts.to_vec(), completion: duration };
    model.check_execution_upper(&schedule)?;
    let mut events = Vec::new();
    for (operation, start) in model.operations.iter().zip(starts) {
        for r in &operation.reservations {
            events.push((r.resource, start + r.offset, true, r.units));
            events.push((r.resource, start + r.offset + r.duration, false, r.units));
        }
    }
    events.sort_unstable();
    let mut used = vec![0u64; model.resources.len()];
    let mut peak = used.clone();
    for (resource, _, begin, units) in events {
        used[resource] = if begin { used[resource].checked_add(units).ok_or("offset occupancy overflow")? }
            else { used[resource].checked_sub(units).ok_or("offset occupancy mismatch")? };
        peak[resource] = peak[resource].max(used[resource]);
    }
    Ok(peak)
}
