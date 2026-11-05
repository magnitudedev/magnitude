use crate::experts::source::Records;
use seismic::{Device, Tensor};

/// The samples of a request of the most choices a site's measurement takes,
/// each over records no sample of it read before (`measure_site`), where the
/// form's experts hold them.
const SITE_SAMPLES: u64 = 16;

/// A CPU site's pool at `form` on `device`, on records of no model, what the
/// measurement job times (`assessment::CpuCosts`): the workers among the
/// machine's candidates ([`super::Machine::sized`], each timed on a decode
/// request, a row's choices, one row each: the site's stage) and its lines
/// there ([`crate::assessment::CpuLines::measured`]: a request of no entries,
/// the handshake; requests of one expert of one row, of a decode row's
/// choices and of one expert of every decode row), each on a pool of its
/// own, over a layer of its own.
pub(crate) fn measure_cpu_site(device: &Device, form: &crate::assessment::CpuForm) -> Result<(usize, crate::assessment::CpuLines), String> {
    use super::{SiteRecords, SiteScales, SiteTensors};
    let count = (form.choices * SITE_SAMPLES).min(form.experts);
    let records = |record: crate::planning::RecordTensor| -> Result<SiteRecords, String> {
        let shape = [count, record.matrix[0], record.matrix[1]];
        let tensor = Tensor::shared(device, record.resident, &shape).map_err(|error| format!("a site's records: {error}"))?;
        // Bytes of no value a format reads as a fast case: real work is a
        // model's words, none of them all zero.
        // SAFETY: nothing reads the tensor yet.
        unsafe { tensor.write_shared(0, &vec![0x55; tensor.byte_len() as usize]) }.map_err(|error| format!("a site's records: {error}"))?;
        let views = Records::new(tensor, record.record_bytes()?, count)?;
        let [rows, columns] = record.matrix.map(|extent| extent as usize);
        SiteRecords::new(views, rows, columns, record.resident)
    };
    let tensors = SiteTensors { gate: form.record.gate.map(records).transpose()?, up: records(form.record.up)?, down: records(form.record.down)? };
    let scales = SiteScales { gate: None, up: None };
    let shape = super::SiteShape::of(form)?;
    let owned = || super::SiteLayer::owned(device, tensors.clone(), scales.clone(), shape);
    let machine = super::Machine::discover(device, 0.)?;
    let request = (shape.choices.min(tensors.up.count()), 1);
    let threads = machine
        .sized(|threads| Ok(super::measure_site(owned()?, threads, &[request])?.remove(0)))
        .map_err(|reason| format!("the expert compute pool could not be sized: {reason}"))?;
    let experts = shape.choices.min(tensors.up.count());
    let measure = |requests: &[(usize, usize)]| super::measure_site(owned()?, &threads, requests);
    let lines = crate::assessment::CpuLines::measured(experts, shape.rows, &measure).map_err(|error| error.to_string())?;
    Ok((threads.count(), lines))
}
