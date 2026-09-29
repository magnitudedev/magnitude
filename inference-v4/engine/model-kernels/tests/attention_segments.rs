// Decode attention time against the number of history segments a sequence's
// visible rows occupy, at Qwen3.5-4B geometry on the local GPU (Metal, or
// Vulkan elsewhere) and the CPU. The same number
// of visible rows is split into `segments` equal runs spread across the history
// arena, as elastic placement and prefix sharing can leave them. This prices
// the repacking cost model: attention time saved per segment removed.
//
// `cargo test --release -p magnitude-model-kernels --test attention_segments
// -- --ignored --nocapture` (indicative on a shared development GPU; evidence
// runs hold the GPU lock on a measurement host).

use magnitude_model_kernels::gated_attention_decode;
use seismic::{BackendName, Device, DeviceCatalog, Element, NativeSpecialization, Tensor};

include!("attention_common/fixtures.rs");

#[test]
#[ignore]
fn decode_time_by_segment_count() {
    for device in devices() {
        decode_time_by_segment_count_on(&device);
    }
}

fn decode_time_by_segment_count_on(device: &Device) {
    // The tuned 4B decode configuration: on the GPU (SPAN, PARTS, SIMDS)
    // with the statics; on CPU its default PARTS, no statics.
    let specialization = if is_cpu(device) {
        NativeSpecialization::new().with_param("PARTS", 8)
    } else {
        statics(QWEN)
            .with_param("SPAN", 32)
            .with_param("PARTS", 16)
            .with_param("SIMDS", 4)
    };
    let kernel = gated_attention_decode::native_for_device_with(
        device,
        gated_attention_decode::Elements { A: Element::bf16() },
        &specialization,
    )
    .unwrap();
    for context in [4096usize, 16384] {
        for batch in [1usize, 4] {
            let mut baseline = None;
            for segments in [1usize, 2, 4, 8, 16] {
                let visible = context - 1;
                let piece = visible.div_ceil(segments);
                // Runs are spread with a gap as large as a run between them.
                let arena = batch * 2 * (piece * segments) + 64 * batch;
                let rows = (0..batch)
                    .map(|sequence| {
                        let base = (sequence * 2 * piece * segments) as i32;
                        let spans = (0..segments)
                            .map(|index| {
                                let start = base + (2 * index * piece) as i32;
                                let count = piece.min(visible - index * piece) as i32;
                                (start, start + count)
                            })
                            .collect::<Vec<_>>();
                        Row {
                            spans,
                            fresh: (sequence as i32, sequence as i32 + 1),
                            destination: (arena - 64 * batch + sequence) as i32,
                            position: visible as i32,
                        }
                    })
                    .collect::<Vec<_>>();
                let case = Case::new(QWEN, arena, segments, &rows, 3);
                let Geometry { kv, g, p, .. } = case.geometry;
                let (m, w, t) = (case.rows, case.geometry.w(), case.history_rows);
                let query_gate = bf16_tensor(device, &[m, kv * g * 2 * w], &case.query_gate);
                let key = bf16_tensor(device, &[m, kv * w], &case.key);
                let value = bf16_tensor(device, &[m, kv * w], &case.value);
                let query_norm = f32_tensor(device, &[w], &case.query_norm);
                let key_norm = f32_tensor(device, &[w], &case.key_norm);
                let components = i32_tensor(device, &[p], &case.components);
                let frequencies = f32_tensor(device, &[p], &case.frequencies);
                let coordinates = i32_tensor(device, &[m, 4], &case.coordinates);
                let visible_spans = i32_tensor(device, &[m, case.spans, 2], &case.visible);
                let fresh = i32_tensor(device, &[m, 2], &case.fresh);
                let destinations = i32_tensor(device, &[m], &case.destinations);
                let mut history_key = bf16_tensor(device, &[t, kv, w], &case.history_key);
                let mut history_value = bf16_tensor(device, &[t, kv, w], &case.history_value);
                let measurement = kernel
                    .measure(
                        vec![gated_attention_decode::Args {
                            query_gate: &query_gate,
                            key: &key,
                            value: &value,
                            query_norm: &query_norm,
                            key_norm: &key_norm,
                            rotary_components: &components,
                            rotary_frequencies: &frequencies,
                            coordinates: &coordinates,
                            visible: &visible_spans,
                            fresh: &fresh,
                            destinations: &destinations,
                            history_key: &mut history_key,
                            history_value: &mut history_value,
                            epsilon: case.epsilon,
                            scale: case.scale,
                        }],
                        &TIMING,
                    )
                    .unwrap();
                let micros = measurement.median * 1e6;
                let one = *baseline.get_or_insert(micros);
                eprintln!(
                    "attention-segments {:?} context={context} batch={batch} segments={segments}: {micros:.1} us ({:+.1}% vs 1 segment)",
                    device.backend(),
                    (micros / one - 1.0) * 100.0
                );
            }
        }
    }
}
