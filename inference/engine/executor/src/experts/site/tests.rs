use super::*;
use crate::experts::compute::Compute;
use crate::experts::{ComputeThreads, HostCompute};
use seismic::Tensor;
use seismic::cpu::{Bf16, Dense};
use std::time::{Duration, Instant};

fn cpu() -> seismic::Device {
    seismic::Devices::discover(&[]).unwrap().open_backend(seismic::BackendName::Cpu).unwrap()
}

const RECORDS: usize = 3;

/// Each of the fixture's experts' gate and up scales.
fn scales() -> SiteScales {
    SiteScales { gate: Some([0.5, 1.25, 2.0].into()), up: Some([1.5, 0.75, 1.0].into()) }
}

/// A layer of gated bf16 records of F = 128 features and H = 64 channels,
/// and per-expert scales ([`scales`], expert e's record e), two rows of two
/// choices, two slots, and the stored values `[gate, up, down]` of every
/// record.
fn fixture(device: &seismic::Device) -> (SiteLayer, [Vec<f32>; 3]) {
    let shape = SiteShape {
        table: magnitude_kernels::SlotTable::site(2, 4),
        rows: 2,
        choices: 2,
        experts: RECORDS,
        hidden: 64,
        features: 128,
        function: functions::SILU,
    };
    let pattern = |count: usize, seed: u32| {
        (0..count)
            .map(|index| ((index as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 20) as f32 / 4096.0 - 0.5)
            .map(Bf16::round)
            .collect::<Vec<_>>()
    };
    let (f, h) = (shape.features, shape.hidden);
    let values = [pattern(RECORDS * f * h, 1), pattern(RECORDS * f * h, 2), pattern(RECORDS * h * f, 3)];
    let element = seismic::Element::bf16();
    let records = |data: &[f32], rows: usize, columns: usize| {
        let bytes = data.iter().flat_map(|value| Bf16::narrow(*value).to_le_bytes()).collect::<Vec<u8>>();
        let tensor = Tensor::shared(device, element, &[RECORDS as u64, rows as u64, columns as u64]).unwrap();
        let placement = tensor.record_placement().unwrap();
        for record in 0..RECORDS {
            let at = record * rows * columns * 2;
            // SAFETY: no work uses the tensor.
            unsafe { tensor.write_shared(placement.offset(record as u64), &bytes[at..at + rows * columns * 2]).unwrap() };
        }
        SiteRecords::new(Records::new(tensor, (rows * columns * 2) as u64, RECORDS as u64).unwrap(), rows, columns, element).unwrap()
    };
    let tensors = SiteTensors { gate: Some(records(&values[0], f, h)), up: records(&values[1], f, h), down: records(&values[2], h, f) };
    (SiteLayer::owned(device, tensors, scales(), shape).unwrap(), values)
}

/// The rows of a request: two rows of H.
fn rows() -> Vec<f32> {
    (0..2 * 64).map(|index| ((index * 37 % 101) as f32 / 101.0 - 0.5) * 2.0).collect()
}

/// The reference output of `entries` `(row, record, weight, expert)` and
/// the site's departure bound ([`super::reference`]), row after row.
fn reference(values: &[Vec<f32>; 3], entries: &[(usize, usize, f32, usize)], x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (f, h) = (128, 64);
    let (gate, up) = (scales().gate.unwrap(), scales().up.unwrap());
    let (mut value, mut bound) = (Vec::new(), Vec::new());
    for row in 0..2 {
        let experts = entries
            .iter()
            .filter(|entry| entry.0 == row)
            .map(|&(_, record, weight, expert)| {
                let at = record * f * h..(record + 1) * f * h;
                let values = ExpertValues {
                    gate: Some(&values[0][at.clone()]),
                    up: &values[1][at.clone()],
                    down: &values[2][at],
                    scales: (gate[expert], up[expert]),
                };
                (weight, values)
            })
            .collect::<Vec<_>>();
        let (row_value, row_bound) = super::reference(functions::SILU, &x[row * h..(row + 1) * h], &experts);
        value.extend(row_value);
        bound.extend(row_bound);
    }
    (value, bound)
}

/// Waits (at most ten seconds) for slot `slot`'s answer to reach `step`.
fn answer(layer: &SiteLayer, slot: usize, step: u64) -> u64 {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let ready = layer.ready(slot);
        if ready & !FAILED >= step {
            return ready;
        }
        assert!(Instant::now() < deadline, "the site never answered slot {slot} step {step}");
        std::hint::spin_loop();
    }
}

fn threads(workers: usize) -> ComputeThreads {
    ComputeThreads::placeable(workers)
}

/// While a flight is issued, a request a worker sees by polling is
/// answered with each row's weighted sum of its entries' whole experts, in
/// q8 arithmetic within the bound it admits of the reference
/// ([`super::reference`]), every row written (a
/// row no entry names is zero), for any number of workers sharing it. With
/// no flight the workers park: a request published then waits for the next
/// flight.
#[test]
fn the_pool_answers_each_rows_weighted_sum_while_a_flight_is_issued() {
    let device = cpu();
    let x = rows();
    for workers in [1usize, 2, 5] {
        let (layer, values) = fixture(&device);
        let compute = HostCompute::start(vec![layer], &threads(workers)).unwrap();
        let handle: Compute = compute.handle();
        let layer = &handle.layers()[0];
        let parked = [(0, 1, 1.0, 1)];
        layer.request(3, &parked, &x);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(layer.ready(1), 0, "{workers} workers: parked without a flight");
        let flight = handle.flight().expect("a site layer issues flights");
        assert_eq!(answer(layer, 1, 3), 3, "{workers} workers: the flight wakes them");
        for (step, entries) in [(4u64, vec![(0, 2, 0.75, 2), (0, 0, 0.25, 0), (1, 2, 0.5, 2)]), (5, vec![(1, 1, 1.0, 1)])] {
            let slot = step as usize % 2;
            layer.request(step, &entries, &x);
            assert_eq!(answer(layer, slot, step), step, "{workers} workers, step {step}");
            let (output, (expected, bound)) = (layer.output(slot), reference(&values, &entries, &x));
            for (index, ((got, want), bound)) in output.iter().zip(&expected).zip(&bound).enumerate() {
                assert!((got - want).abs() <= *bound, "{workers} workers, step {step}, value {index}: {got} against {want} ±{bound}");
            }
            if step == 5 {
                assert!(output[..64].iter().all(|value| *value == 0.0), "a row no entry names is zero");
            }
        }
        // A request of no entries is answered, every row zero.
        layer.request(6, &[], &x);
        assert_eq!(answer(layer, 0, 6), 6, "{workers} workers: a request of no entries");
        assert!(layer.output(0).iter().all(|value| *value == 0.0), "{workers} workers: a request of no entries");
        drop(flight);
    }
}

/// An answer only moves its slot's word forward: a late publish for an
/// earlier step, failed or not, leaves a later step's answer as it is.
#[test]
fn an_answer_never_moves_a_slots_word_backwards() {
    let device = cpu();
    let (layer, _) = fixture(&device);
    // Odd steps, all slot 1's.
    layer.publish(7, false);
    layer.publish(5, false);
    assert_eq!(layer.ready(1), 7);
    layer.publish(5, true);
    assert_eq!(layer.ready(1), 7);
    layer.publish(9, true);
    assert_eq!(layer.ready(1), 9 | FAILED);
    layer.publish(9, false);
    assert_eq!(layer.ready(1), 9 | FAILED, "a step is answered once");
    layer.publish(11, false);
    assert_eq!(layer.ready(1), 11);
}

/// A request the device wrote naming a row, a tier record or an expert
/// beyond the layer's is an invariant violation: reading it panics, never
/// a failure the step's rows carry.
#[test]
fn a_malformed_request_panics_when_read() {
    let device = cpu();
    let (layer, _) = fixture(&device);
    let malformed = [(4, (0, RECORDS + 4, 1.0, 0), "tier record"), (6, (1, 0, 1.0, RECORDS), "expert"), (8, (2, 0, 1.0, 0), "row")];
    for (step, entry, named) in malformed {
        layer.request(step, &[entry], &rows());
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| layer.work(layer.shape.slot(step), step, 1)));
        let why = read.err().unwrap_or_else(|| panic!("step {step}'s request is read"));
        let why = why.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(why.contains(named), "{why}");
    }
}

/// A request waits only on its own work: with every worker but one inside
/// a job's piece that does not end until the request is answered, the one
/// claims every unit and answers it.
#[test]
fn a_request_is_answered_while_workers_are_held_by_a_piece() {
    /// Its pieces entered, and whether they may end.
    struct Held(Arc<AtomicUsize>, Arc<AtomicBool>);
    impl crate::experts::compute::Job for Held {
        fn pieces(&self) -> usize {
            2
        }
        fn piece(&self, _: usize) {
            self.0.fetch_add(1, Ordering::AcqRel);
            while !self.1.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
        }
        fn end(&self) {}
        fn want(&self) -> seismic_io::Want {
            seismic_io::Want::now()
        }
    }
    let device = cpu();
    let (layer, values) = fixture(&device);
    let compute = HostCompute::start(vec![layer], &threads(3)).unwrap();
    let handle = compute.handle();
    let layer = &handle.layers()[0];
    let (entered, released) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicBool::new(false)));
    handle.submit(Arc::new(Held(entered.clone(), released.clone())));
    let deadline = Instant::now() + Duration::from_secs(10);
    while entered.load(Ordering::Acquire) < 2 {
        assert!(Instant::now() < deadline, "two workers never took the pieces");
        std::thread::yield_now();
    }
    let _flight = handle.flight().unwrap();
    let (x, entries) = (rows(), [(0, 1, 1.0, 1), (1, 2, 0.5, 2)]);
    layer.request(1, &entries, &x);
    assert_eq!(answer(layer, 1, 1), 1, "answered by the one worker free");
    released.store(true, Ordering::Release);
    let (output, (expected, bound)) = (layer.output(1), reference(&values, &entries, &x));
    assert!(output.iter().zip(&expected).zip(&bound).all(|((got, want), bound)| (got - want).abs() <= *bound));
}

/// Teardown with a request issued is correct: the executor closes issuing,
/// then its workers serve every request issued before they exit, so the
/// request is answered and the drop returns.
#[test]
fn teardown_answers_every_request_issued() {
    let device = cpu();
    let (layer, _) = fixture(&device);
    let compute = HostCompute::start(vec![layer], &threads(4)).unwrap();
    let handle = compute.handle();
    let flight = handle.flight().unwrap();
    let layer = &handle.layers()[0];
    for step in 1..=20u64 {
        let slot = step as usize % 2;
        layer.request(step, &[(0, 0, 1.0, 0), (1, 1, 1.0, 1)], &rows());
        answer(layer, slot, step);
    }
    layer.request(21, &[(0, 2, 1.0, 2)], &rows());
    let deadline = Instant::now() + Duration::from_secs(10);
    while !handle.site_busy() && layer.ready(1) & !FAILED < 21 {
        assert!(Instant::now() < deadline, "the request was never taken");
        std::hint::spin_loop();
    }
    drop(flight);
    drop(compute);
    assert_eq!(layer.ready(1) & !FAILED, 21, "answered before the workers exited");
}

/// Every layer's requests in flight at once are all answered: every worker
/// serves them in one order, so none waits at a barrier another skipped.
#[test]
fn requests_of_every_layer_and_slot_at_once_are_all_answered() {
    let device = cpu();
    let layers = (0..3).map(|_| fixture(&device).0).collect::<Vec<_>>();
    let compute = HostCompute::start(layers, &threads(4)).unwrap();
    let handle = compute.handle();
    let _flight = handle.flight().unwrap();
    let x = rows();
    for step in 1..=40u64 {
        let slot = step as usize % 2;
        for layer in handle.layers() {
            layer.request(step, &[(0, step as usize % RECORDS, 1.0, 0), (1, 1, 0.5, 1)], &x);
        }
        for layer in handle.layers() {
            assert_eq!(answer(layer, slot, step), step);
        }
    }
}

/// Every unit of a record is expanded by exactly one worker, whatever the
/// pool, every worker's count within one of every other's, and a record's
/// deal is its own: two records start at different workers.
#[test]
fn each_unit_has_one_worker_and_the_counts_are_even() {
    for workers in [1usize, 3, 10, 15, 30] {
        for record in 0..8 {
            let ranges = (0..workers).map(|worker| expands(worker, workers, record, 256)).collect::<Vec<_>>();
            let mut covered = ranges.iter().flat_map(|range| range.clone()).collect::<Vec<_>>();
            covered.sort();
            assert_eq!(covered, (0..256).collect::<Vec<_>>());
            let (least, most) = (ranges.iter().map(|range| range.len()).min().unwrap(), ranges.iter().map(|range| range.len()).max().unwrap());
            assert!(most - least <= 1, "{workers} workers: {least}..{most}");
        }
    }
    assert_ne!(expands(0, 15, 0, 256), expands(0, 15, 1, 256));
    let covered = (0..30).flat_map(|worker| share(worker, 30, 128)).collect::<Vec<_>>();
    assert_eq!(covered, (0..128).collect::<Vec<_>>(), "phase two's channel tiles, each once, in order");
}

/// A measurement serves each request through an executor of its own and
/// times its answer, its warm-up request discarded; a request of no entries
/// is the handshake alone. Each is sampled with its resolution.
#[test]
fn the_executor_measures_its_answers() {
    let device = cpu();
    let (layer, _) = fixture(&device);
    let samples = measure(layer, &threads(3), &[(0, 0), (1, 1), (1, 2)]).unwrap();
    assert_eq!(samples.len(), 3);
    assert!(samples.iter().all(|sample| sample.seconds > 0.0 && sample.resolution >= 0.0), "{samples:?}");
}

/// `count` tier records of a layer of `h` channels and `f` features in the
/// device's packed `mma16` form (gate and up `q4k`, down `q5k`), each the
/// conversion of finite pseudo-random stored bytes.
fn packed(device: &seismic::Device, h: usize, f: usize, count: usize) -> SiteTensors {
    let tier = seismic::cpu::Tier::detected().unwrap();
    let element = |name: &str| seismic::Element::named(name).unwrap_or_else(|| panic!("{name}"));
    let tensor = |name: &str, source: &str, rows: usize, columns: usize, seed: u32| {
        let (resident, source) = (element(name), element(source));
        assert!(resident.cpu_operand(tier).is_some(), "the CPU reads {name}");
        let record = [1, rows as u64, columns as u64];
        let stored_bytes = source.canonical_byte_len(&record).unwrap() as usize;
        let tensor = Tensor::shared(device, resident, &[count as u64, rows as u64, columns as u64]).unwrap();
        let placement = tensor.record_placement().unwrap();
        for index in 0..count as u32 {
            let bytes = (0..stored_bytes).map(|at| ((at as u32).wrapping_mul(2_654_435_761).wrapping_add(seed + index) >> 24) as u8).collect();
            let stored = resident.finite_source(source, bytes).unwrap();
            let converted = resident.repack_host(source, &record, &stored).unwrap();
            // SAFETY: no work uses the tensor.
            unsafe { tensor.write_shared(placement.offset(u64::from(index)), &converted).unwrap() };
        }
        let bytes = placement.record();
        SiteRecords::new(Records::new(tensor, bytes, count as u64).unwrap(), rows, columns, resident).unwrap()
    };
    SiteTensors {
        gate: Some(tensor("q4k@mma16", "gguf_q4_k", f, h, 1)),
        up: tensor("q4k@mma16", "gguf_q4_k", f, h, 2),
        down: tensor("q5k@mma16", "gguf_q5_k", h, f, 3),
    }
}

/// A site shape of `rows` decode rows of `choices` each over `experts`
/// experts of `h` channels and `f` features, two slots.
fn packed_shape(rows: usize, choices: usize, experts: usize, h: usize, f: usize) -> SiteShape {
    SiteShape {
        table: magnitude_kernels::SlotTable::site(2, (rows * choices) as u64),
        rows,
        choices,
        experts,
        hidden: h,
        features: f,
        function: functions::SILU,
    }
}

/// Records in the device's packed `mma16` form (gate and up `q4k`, down
/// `q5k`) are read where they lie: the answer is the stored values' weighted
/// sum within the q8 activations' error.
#[test]
fn packed_records_are_served_where_they_lie() {
    let device = cpu();
    let (h, f) = (512usize, 256usize);
    let tensors = packed(&device, h, f, 2);
    // The stored values as the CPU's components decode them, the registry's
    // decode (`SiteRecords::values`).
    let values = [tensors.gate.as_ref().unwrap(), &tensors.up, &tensors.down]
        .map(|records| (0..2).flat_map(|record| records.values(record)).collect::<Vec<_>>());
    let shape = packed_shape(2, 2, 2, h, f);
    let layer = SiteLayer::owned(&device, tensors, SiteScales { gate: None, up: None }, shape).unwrap();
    let compute = HostCompute::start(vec![layer], &threads(3)).unwrap();
    let handle = compute.handle();
    let _flight = handle.flight().unwrap();
    let layer = &handle.layers()[0];
    let x = (0..2 * h).map(|index| ((index * 53 % 211) as f32 / 211.0 - 0.5) * 2.0).collect::<Vec<_>>();
    let entries = [(0, 1, 0.5, 1), (0, 0, 0.5, 0), (1, 1, 1.0, 1)];
    layer.request(1, &entries, &x);
    assert_eq!(answer(layer, 1, 1), 1);
    let output = layer.output(1);
    let dot = |weights: &[f32], x: &[f32]| weights.iter().zip(x).map(|(w, x)| f64::from(*w) * f64::from(*x)).sum::<f64>() as f32;
    let mut expected = vec![0.0f32; 2 * h];
    for &(row, record, weight, _) in &entries {
        let x = &x[row * h..(row + 1) * h];
        let product = (0..f)
            .map(|feature| {
                let at = (record * f + feature) * h;
                functions::glu::<F32>(functions::SILU, dot(&values[0][at..at + h], x), dot(&values[1][at..at + h], x))
            })
            .collect::<Vec<_>>();
        for channel in 0..h {
            let at = (record * h + channel) * f;
            expected[row * h + channel] += weight * dot(&values[2][at..at + f], &product);
        }
    }
    let scale = expected.iter().fold(0.0f32, |most, value| most.max(value.abs()));
    for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
        assert!((got - want).abs() <= 0.03 * scale, "value {index}: {got} against {want} (scale {scale})");
    }
}

/// An expert named by several rows is computed once over all of them, and
/// each row's answer is the one it has alone, bit for bit: every share of
/// the expert is fixed by its record and the pool alone, and the quantized
/// dot of several activation rows gives each row the sums it has alone
/// (`Weights::dot_q8`), so nothing is reassociated, for any pool.
#[test]
fn an_expert_of_several_rows_answers_each_as_alone() {
    let device = cpu();
    let (h, f, rows) = (512usize, 256usize, 4usize);
    let x = (0..rows * h).map(|index| ((index * 53 % 211) as f32 / 211.0 - 0.5) * 2.0).collect::<Vec<_>>();
    for workers in [1usize, 3, 5] {
        let layer = SiteLayer::owned(&device, packed(&device, h, f, 2), SiteScales { gate: None, up: None }, packed_shape(rows, 2, 2, h, f)).unwrap();
        let compute = HostCompute::start(vec![layer], &threads(workers)).unwrap();
        let handle = compute.handle();
        let _flight = handle.flight().unwrap();
        let layer = &handle.layers()[0];
        let weights = [0.5f32, 1.0, 0.75, 0.25];
        let together = (0..rows).map(|row| (row, 1, weights[row], 1)).collect::<Vec<_>>();
        layer.request(1, &together, &x);
        assert_eq!(answer(layer, 1, 1), 1);
        let output = layer.output(1);
        for row in 0..rows {
            let step = 2 + row as u64;
            let slot = step as usize % 2;
            layer.request(step, &[(row, 1, weights[row], 1)], &x);
            assert_eq!(answer(layer, slot, step), step);
            let alone = layer.output(slot);
            let bits = |values: &[f32]| values[row * h..(row + 1) * h].iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&output), bits(&alone), "{workers} workers, row {row}");
        }
    }
}

/// The site counts each answered request of any entries, its experts and
/// each term of its path, as its answer is written: what a load's
/// telemetry divides into per-miss terms. A request of no entries has no
/// path and counts nothing.
#[test]
fn the_site_counts_each_answered_requests_experts_and_path() {
    let device = cpu();
    let x = rows();
    let (layer, _) = fixture(&device);
    let compute = HostCompute::start(vec![layer], &threads(2)).unwrap();
    let handle: Compute = compute.handle();
    let flight = handle.flight().expect("a site layer issues flights");
    let layer = &handle.layers()[0];
    // Each step's experts are read once it is answered, before its slot is
    // requested again (as a step's completion does, within the flight).
    layer.request(1, &[(0, 1, 1.0, 1), (1, 2, 1.0, 2)], &x);
    answer(layer, 1, 1);
    assert_eq!(handle.sited(0, 1).0, 2);
    layer.request(2, &[(0, 0, 1.0, 0)], &x);
    answer(layer, 0, 2);
    assert_eq!(handle.sited(0, 2).0, 1);
    layer.request(3, &[], &x);
    answer(layer, 1, 3);
    assert_eq!(handle.sited(0, 3).0, 0, "a request of no entries serves none");
    let deadline = Instant::now() + Duration::from_secs(10);
    while handle.site_served().requests < 2 {
        assert!(Instant::now() < deadline, "the answered requests were never counted");
        std::hint::spin_loop();
    }
    let served = handle.site_served();
    assert_eq!(served.requests, 2);
    assert!(served.terms.iter().sum::<Duration>() > Duration::ZERO, "{served:?}");
    assert_eq!(served.since(&served), super::SiteServed::default());
    drop(flight);
}
