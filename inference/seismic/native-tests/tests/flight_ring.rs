//! A flight ring's completion guard against each backend's own completion
//! signal: host-shared staging written in place, without waiting, is read by
//! device batches kept in flight up to the ring's depth.

use seismic::{
    BackendName, Devices, Element, FlightRing, NativeSpecialization, NativeTensorBatch,
    NativeTensorBatchCompletion, Tensor,
};
use seismic_native_tests::scale_rows;
use std::collections::VecDeque;

const ROWS: u64 = 1024;
const COLUMNS: u64 = 64;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// At depths 1, 2 and 3, `depth` batches stay in flight, each scaling the
/// staging entry the ring gave it. The submitter observes the oldest
/// batch's result before it takes the next entry, as a lane takes a launch
/// only once its oldest completed, so the ring's guard, the batch's own
/// completion, finds each entry complete when its turn comes; and every
/// result is the scaled values written into its entry, none overwritten
/// while the device still read it.
fn an_entry_is_rewritten_only_after_its_batch_completed(backend: BackendName) {
    let devices = Devices::discover(&[]).unwrap();
    let Ok(device) = devices.open_backend(backend) else {
        return;
    };
    let kernel =
        scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
            .unwrap();
    for depth in 1..=3 {
        let mut ring = FlightRing::<Tensor, NativeTensorBatchCompletion>::default();
        ring.resize(depth, || Tensor::shared(&device, Element::f32(), &[ROWS, COLUMNS]))
            .unwrap();
        let mut flying: VecDeque<(Tensor, Vec<f32>)> = VecDeque::new();
        let observe = |(result, expected): (Tensor, Vec<f32>)| {
            assert_eq!(result.read_to_host().unwrap(), bytes(&expected), "{backend:?} depth {depth}");
        };
        for step in 0..4 * depth {
            if flying.len() == depth {
                observe(flying.pop_front().unwrap());
            }
            let mut entry = ring.next().unwrap();
            let values = (0..ROWS * COLUMNS)
                .map(|index| (step * 7 + (index % 97) as usize) as f32)
                .collect::<Vec<_>>();
            // SAFETY: the ring gave the entry only once the batch that last
            // read it completed, and no other work reaches it.
            unsafe { entry.value().write_shared(0, &bytes(&values)) }.unwrap();
            // SAFETY: the entry writes every f32 result element.
            let mut result =
                unsafe { Tensor::uninitialized(&device, Element::f32(), &[ROWS, COLUMNS]) }.unwrap();
            let mut batch = NativeTensorBatch::new(&device);
            batch
                .push(
                    &kernel,
                    scale_rows::Args { x: entry.value(), factor: 2.0 },
                    scale_rows::OutputArgs { value: &mut result },
                )
                .unwrap();
            entry.submitted(batch.submit().unwrap());
            flying.push_back((result, values.iter().map(|value| value * 2.0).collect()));
        }
        flying.into_iter().for_each(observe);
    }
}

#[test]
fn cuda_flight_ring_reuses_an_entry_only_after_its_batch_completed() {
    an_entry_is_rewritten_only_after_its_batch_completed(BackendName::Cuda);
}

#[test]
fn vulkan_flight_ring_reuses_an_entry_only_after_its_batch_completed() {
    an_entry_is_rewritten_only_after_its_batch_completed(BackendName::Vulkan);
}

#[test]
fn metal_flight_ring_reuses_an_entry_only_after_its_batch_completed() {
    an_entry_is_rewritten_only_after_its_batch_completed(BackendName::Metal);
}
