//! A streamed layer's CPU site admission: the measurement job,
//! once per sited binding ([`crate::assessment::SiteRequest`]), has the host
//! compute executor (sized as a load's) answer a request of every decode
//! row's choices over the binding's representative records, and compares
//! its output with the reference, f32 math over the stored weights, at the
//! load's numerical requirement, as every device form is (`seismic::admit`).
//! The site is never compared with a device's output. A load runs none of
//! it: it reads the verdict.

use super::super::*;
use crate::experts::{SiteLayer, SiteRecords, SiteScales, SiteShape, SiteTensors};
use crate::assessment::{Priced, SiteRequest};
use crate::planning::{BankBinding, RecordTensor};
use magnitude_kernels::cpu_library::cpu::lib::core::functions;


/// The verdict the measurement job stored of `binding`'s CPU site of form
/// `form` under `requirement`; pending where it has stored none, as without
/// a `cache` to keep one.
pub(super) fn site_verdict(
    device: &Device,
    binding: &BankBinding,
    form: SiteForm,
    requirement: &seismic::NumericalRequirement,
    cache: Option<&crate::kernel_cache::KernelCache>,
) -> Priced<()> {
    let Some(cache) = cache else {
        return Priced::Pending("no kernel cache keeps a CPU site's verdict".into());
    };
    let records = crate::kernel_cache::KernelCache::measurement_records(cache.root());
    SiteRequest::of(binding, form, requirement.clone()).verdict(device.topology(), device.info(), &records)
}

/// Run `request`'s site in the measurement job, on its contained `device`:
/// the precision system's verdict, or why the pool failed to answer, which
/// says nothing of the arithmetic.
///
/// Every decode row names `K` choices, choice k of row m the representative
/// tier record `(m K + k) mod records` (and the expert of its index), with
/// representative weights and rows, and representative per-expert scales of
/// the gate (where gated) and the up, so the scaled arithmetic a layer with
/// second-level scales runs is what is admitted (an unscaled stack's is the
/// same arithmetic at a scale of one, which is exact); the site's output
/// `[DECODE_ROWS, H]` is compared with the reference's.
pub(crate) fn admit_site(device: &Device, request: &SiteRequest) -> Result<Result<(), String>, String> {
    let failed = |error: &dyn fmt::Display| error.to_string();
    let to = |value: u64| usize::try_from(value).map_err(|error| failed(&error));
    let shape = SiteShape::of(&request.form)?;
    let (rows, choices, hidden, features) = (shape.rows, shape.choices, shape.hidden, shape.features);
    let records = (rows * choices).min(to(request.tier)?);
    if records == 0 {
        return Err(failed(&"a sited layer's bank has no host tier record"));
    }
    // The records the site reads, in the bank's resident form, each the
    // conversion of a representative stored record, with its values.
    let seed = std::cell::Cell::new(0);
    let held = request.form.record.try_map(|_, tensor| {
        seed.set(seed.get() + 1);
        held_records(device, tensor, records as u64, seed.get())
    })?;
    let tensors = SiteTensors {
        gate: held.gate.as_ref().map(|held| held.records.clone()),
        up: held.up.records.clone(),
        down: held.down.records.clone(),
    };
    let shape = SiteShape { experts: records, ..shape };
    let scale = |seed| pattern(records, seed, 0.5).iter().map(|scale| scale.abs() + 0.5).collect::<std::sync::Arc<[f32]>>();
    let scales = SiteScales { gate: held.gate.is_some().then(|| scale(13)), up: Some(scale(17)) };
    let layer = SiteLayer::owned(device, tensors, scales.clone(), shape).map_err(|error| failed(&error))?;
    // No file is read beside a qualification: the read engine needs no core.
    let threads = crate::experts::Machine::discover(device, 0.).and_then(|machine| machine.widest()).map_err(|error| failed(&error))?;
    let compute = crate::experts::HostCompute::start(vec![layer], &threads).map_err(|error| failed(&error))?;
    let handle = compute.handle();
    let _flight = handle.flight().expect("a site of one layer");
    let layer = &handle.layers()[0];

    let values = pattern(rows * hidden, 7, 1.0);
    let weights = pattern(rows * choices, 11, 0.5).iter().map(|weight| weight.abs() + 0.25).collect::<Vec<_>>();
    let entries =
        (0..rows * choices).map(|entry| (entry / choices, entry % records, weights[entry], entry % records)).collect::<Vec<_>>();
    let step = 1;
    let slot = shape.slot(step);
    layer.request(step, &entries, &values);
    // A request is answered: every taken one ends in its answer.
    let ready = loop {
        match layer.ready(slot) {
            0 => match handle.panicked() {
                Some(reason) => return Err(failed(&reason)),
                None => std::hint::spin_loop(),
            },
            ready => break ready,
        }
    };
    if ready != step {
        let reasons = handle.take_site_failure(step).map_or_else(Vec::new, |failure| failure.reasons);
        return Err(format!("the CPU pool failed its request: {}", reasons.join("; ")));
    }
    let output = layer.output(slot);

    // The reference: f32 math over the stored values, each dot summed in
    // f64 and rounded once.
    let dot = |weights: &[f32], x: &[f32]| weights.iter().zip(x).map(|(w, x)| f64::from(*w) * f64::from(*x)).sum::<f64>() as f32;
    let mut reference = vec![0.0f32; rows * hidden];
    let glu = crate::experts::gated(request.form.function, request.form.limit);
    for &(row, record, weight, expert) in &entries {
        let x = &values[row * hidden..(row + 1) * hidden];
        fn matrix(held: &Held, record: usize, row: usize, width: usize) -> &[f32] {
            &held.values[(record * held.records.rows + row) * width..][..width]
        }
        let scaled = |scales: &Option<std::sync::Arc<[f32]>>, sum: f32| scales.as_ref().map_or(sum, |scales| scales[expert] * sum);
        let product = (0..features)
            .map(|feature| {
                let up = scaled(&scales.up, dot(matrix(&held.up, record, feature, hidden), x));
                match &held.gate {
                    Some(gate) => {
                        let gate = scaled(&scales.gate, dot(matrix(gate, record, feature, hidden), x));
                        glu(gate, up)
                    }
                    None => functions::activate(request.form.function, up),
                }
            })
            .collect::<Vec<_>>();
        for channel in 0..hidden {
            reference[row * hidden + channel] += weight * dot(matrix(&held.down, record, channel, features), &product);
        }
    }
    let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let activation = request.activation.dtype().filter(|dtype| dtype.is_float());
    let comparison = seismic::SubjectComparison::of_values(seismic::DType::F32, &bits(&reference), &bits(&output), activation);
    let observation = seismic::NumericalObservation {
        point: "site".into(),
        key: seismic::LaunchKey { launches: Vec::new(), values: Default::default() },
        rotations: vec![vec![seismic::SubjectMeasurement { subject: "value".into(), comparison }]],
    };
    Ok(seismic::admit(&observation, &request.requirement)
        .map(|_| ())
        .map_err(|refusal| format!("the precision system refused its arithmetic against the reference: {refusal}")))
}

/// `count` deterministic values in `[-scale, scale)` of seed `seed`.
fn pattern(count: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32).wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(40_503)) >> 8) as f32 / (1u32 << 24) as f32)
        .map(|unit| (2.0 * unit - 1.0) * scale)
        .collect()
}

/// A routed tensor's representative records as the site reads them, and
/// their values `[records, N, K]`, as stored.
struct Held {
    records: SiteRecords,
    values: Vec<f32>,
}

/// `count` records of `tensor` in host-shared storage, in its resident
/// form, each the conversion of a representative stored record (seed
/// `seed`): a dense source's values in `[-1/2, 1/2)`, a packed source's
/// pseudo-random bytes in its domain with every value finite
/// (`Element::finite_source`), so the comparison is over numbers.
fn held_records(device: &Device, tensor: RecordTensor, count: u64, seed: u64) -> Result<Held, String> {
    let failed = |error: &dyn fmt::Display| error.to_string();
    let [n, k] = tensor.matrix;
    let record = [1, n, k];
    let stored_bytes = usize::try_from(tensor.stored_bytes().map_err(|error| failed(&error))?).map_err(|error| failed(&error))?;
    let held = Tensor::shared(device, tensor.resident, &[count, n, k]).map_err(|error| failed(&error))?;
    let placement = held.record_placement().ok_or_else(|| failed(&"tier records the host does not place"))?;
    for index in 0..count {
        let seed = (seed * 131 + index) as u32;
        let stored = match tensor.source.dtype() {
            Some(dtype) if dtype.is_float() => {
                let values = pattern((n * k) as usize, seed, 0.5).iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
                tensor.source.repack_host(Element::f32(), &record, &values)
            }
            _ => {
                let bytes = pattern(stored_bytes, seed, 1.0).iter().map(|value| (value * 127.0) as i8 as u8).collect();
                tensor.resident.finite_source(tensor.source, bytes)
            }
        }
        .ok_or_else(|| failed(&format!("no representative {} record", tensor.source.name())))?;
        let resident = tensor
            .resident
            .repack_host(tensor.source, &record, &stored)
            .ok_or_else(|| failed(&"the stored record has no conversion into its resident form"))?;
        // SAFETY: no work uses the records yet.
        unsafe { held.write_shared(placement.offset(index), &resident) }.map_err(|error| failed(&error))?;
    }
    let bytes = tensor.record_bytes().map_err(|error| failed(&error))?;
    let records = crate::experts::Records::new(held, bytes, count).map_err(|error| failed(&error))?;
    let [rows, columns] = tensor.matrix.map(|extent| extent as usize);
    let records = SiteRecords::new(records, rows, columns, tensor.resident).map_err(|error| failed(&error))?;
    // The stored values: the resident form is their exact conversion, which
    // the CPU's components decode as the registry does, at the speed the
    // records' count and size ask of a load (the registry's interpreter
    // takes microseconds a value).
    let mut values = Vec::with_capacity(count as usize * rows * columns);
    for index in 0..records.count() {
        values.extend(records.values(index));
    }
    Ok(Held { records, values })
}
