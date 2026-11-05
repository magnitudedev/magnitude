use crate::experts::landing::TierTensor;
use crate::planning::{BankLayout, ExpertBankPlan, Placer};
use seismic::Tensor;

/// The site layers the executor serves: each streamed layer whose CPU site
/// the precision system admitted at its qualification
/// (`BankSite::admitted`), over its group's tier records (the CPU site reads
/// a host tier, which only a bank the device places into has); the
/// executor's threads, at the workers the job sized or `machine`'s widest
/// pool; and `lines` with each sited layer's CPU lines as the job stored
/// them, none until it has: its policy is then pending (`SiteLines::Pending`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn site_lines(
    plan: &ExpertBankPlan,
    layout: &BankLayout<Tensor>,
    tiers: Option<&[Vec<TierTensor>]>,
    sites: &[Option<crate::native::BankSite>],
    lines: std::sync::Arc<[crate::assessment::BankLines]>,
    machine: &super::Machine,
    cpu: &crate::assessment::StoredCpuSite,
    pulse: &super::exchange::Pulse,
) -> Result<(Vec<super::SiteLayer>, std::sync::Arc<[crate::assessment::BankLines]>, super::ComputeThreads), String> {
    let mut sited = Vec::new();
    for (index, site) in sites.iter().enumerate() {
        let Some(site) = site else { continue };
        let (Placer::Tier, Some(tiers)) = (plan.placer(), tiers) else {
            return Err(format!("streamed layer {index}'s CPU site has no host tier to read"));
        };
        sited.push((index, site_layer(plan, layout, index, tiers, site.form, pulse)?));
    }
    // At the workers the job sized, where it has, within what the load's
    // reads leave: its lines are read wherever they are stored.
    let threads = match cpu.workers {
        Some(workers) => machine.at_most(workers)?,
        None => machine.widest()?,
    };
    let mut measured = lines.to_vec();
    for (index, _) in &sited {
        measured[*index].cpu = cpu.lines.get(index).copied();
    }
    Ok((sited.into_iter().map(|(_, layer)| layer).collect(), measured.into(), threads))
}

/// Streamed layer `index`'s site layer over its group's tier records
/// (`tiers`, the views the landings convert into, re-addressed with them on
/// a resize) and its site words and buffers, in its site's form.
fn site_layer(
    plan: &ExpertBankPlan,
    layout: &BankLayout<Tensor>,
    index: usize,
    tiers: &[Vec<TierTensor>],
    form: crate::native::SiteForm,
    pulse: &super::exchange::Pulse,
) -> Result<super::SiteLayer, String> {
    use super::{SiteLayer, SiteRecords, SiteScales, SiteStorage, SiteTensors};
    let Some(sited) = &layout.layers[index].sited else {
        return Err(format!("streamed layer {index}'s entries serve a CPU site its bank allocates none for"));
    };
    let group = plan.layers[index].group;
    let records = plan.groups[group].record;
    let mut views = tiers[group].iter().map(|tensor| tensor.records.clone());
    let mut tensor = |record: crate::planning::RecordTensor| -> Result<SiteRecords, String> {
        let [rows, columns] = record.matrix.map(|extent| extent as usize);
        SiteRecords::new(views.next().ok_or("a tier holds its record's tensors")?, rows, columns, record.resident)
    };
    let tensors = SiteTensors {
        gate: records.gate.map(&mut tensor).transpose()?,
        up: tensor(records.up)?,
        down: tensor(records.down)?,
    };
    // Each expert's second-level scales of each stack that has them, which
    // the site layer admits (one a layer's expert).
    let scales = |scale: &Option<Tensor>| -> Result<Option<std::sync::Arc<[f32]>>, String> {
        scale
            .as_ref()
            .map(|tensor| {
                let bytes = tensor.read_to_host().map_err(|error| format!("an expert scale: {error}"))?;
                if bytes.len() % 4 != 0 {
                    return Err(format!("an expert scale of {} bytes, no whole f32s", bytes.len()));
                }
                Ok(bytes.chunks_exact(4).map(|word| f32::from_le_bytes(word.try_into().expect("four bytes"))).collect())
            })
            .transpose()
    };
    let layer = &layout.layers[index];
    let scales = SiteScales { gate: scales(&layer.expert_scales.gate.clone().flatten())?, up: scales(&layer.expert_scales.up)? };
    let shape = super::SiteShape::of(&crate::assessment::CpuForm::of(&plan.binding(index)?, form.function))?;
    let storage = SiteStorage {
        table: sited.table.clone(),
        rows: sited.rows.clone(),
        output: sited.output.clone(),
    };
    SiteLayer::over(storage, index, group, tensors, scales, shape, pulse)
}
