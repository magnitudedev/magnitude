// `indexer_keys` against an F32 host model of DeepSeek V4.1's index keys
// (RMS(wk latent) k_norm, rotated in adjacent trailing pairs at the unit's
// first coordinates, rounded to E2M1 per 32 with UE8M0 scales), the portable
// body and every backend this host opens: each completing row's key lands at
// its key-plane row as the model's codes (a value whose unrounded F32 lies
// within arithmetic order of a code boundary may take the neighbouring
// code), and no other row is written. Also a half-split leading layout
// without rounding (Qwen4's form).

mod common;
mod indexer_common;
mod rounding_model;
mod versions_common;

use indexer_common::*;
use magnitude_kernels::indexer_keys;
use rounding_model::{INDEX, NONE, Rounding};
use seismic::{Element, NativeSpecialization, Tensor};
use versions_common::*;

struct Case {
    rows: usize,
    layout: Layout,
    rounding: Rounding,
    key: Vec<f32>,
    key_norm: Vec<f32>,
    /// Each row's key-plane row, or -1.
    latent_rows: Vec<i32>,
    /// Each row's unit's first position.
    positions: Vec<usize>,
    plane_rows: usize,
}

impl Case {
    fn new(rows: usize, layout: Layout, rounding: Rounding, seed: u64) -> Self {
        let mut random = Random(seed);
        let key = (0..rows * W).map(|i| bf16_round(random.next() * [0.02, 1.0, 40.0][i / W % 3])).collect();
        let key_norm = (0..W).map(|_| 1.0 + 0.5 * random.next()).collect();
        // Every other row completes a unit; the units' rows cross a slab
        // edge.
        let latent_rows = (0..rows)
            .map(|row| if row % 2 == 1 { (SLAB_ROWS - rows / 2 + row * 3) as i32 } else { -1 })
            .collect::<Vec<_>>();
        let positions = (0..rows).map(|row| (random.next().abs() * 65_000.0) as usize + row).collect();
        let plane_rows = 2 * SLAB_ROWS;
        Self { rows, layout, rounding, key, key_norm, latent_rows, positions, plane_rows }
    }

    fn coordinates(&self) -> Vec<i32> {
        self.positions.iter().flat_map(|position| coordinates(*position)).collect()
    }

    /// The model's plane: SENTINEL, with each completing row's key.
    fn expected(&self) -> Vec<f32> {
        let mut plane = vec![SENTINEL; self.plane_rows * W];
        for row in 0..self.rows {
            if let Ok(at) = usize::try_from(self.latent_rows[row]) {
                let key = prepare(&self.key[row * W..][..W], Some(&self.key_norm), self.positions[row], self.layout, self.rounding);
                plane[at * W..][..W].copy_from_slice(&key);
            }
        }
        plane
    }
}

fn portable(case: &Case) -> Vec<f32> {
    let m = case.rows;
    let [block, element, scale, floor] = rounding_inputs(case.rounding);
    let (_, inputs) = interpret(
        "indexer_keys",
        true,
        vec![
            Input::Bf16(vec![m, W], case.key.clone()),
            Input::F32(vec![1, W], case.key_norm.clone()),
            Input::I32(vec![m], case.latent_rows.clone()),
            Input::I32(vec![P], vec![0; P]),
            Input::F32(vec![P], frequencies()),
            Input::F32(vec![P], vec![1.0; P]),
            Input::I32(vec![m, 4], case.coordinates()),
            Input::F32(vec![case.plane_rows, W], vec![SENTINEL; case.plane_rows * W]),
            Input::I32(vec![0, case.layout.ra, case.layout.rt], Vec::new()),
            Input::ScalarF32(EPSILON),
            block,
            element,
            scale,
            floor,
            Input::ScalarU32(SLAB_ROWS as u32),
        ],
    );
    after(&inputs, 7).iter().map(|v| *v as f32).collect()
}

fn native(device: &seismic::Device, case: &Case) -> Vec<f32> {
    let m = case.rows as u64;
    let statics = [("P", P as u64), ("S", S as u64), ("RA", case.layout.ra as u64), ("RT", case.layout.rt as u64)];
    let specialization = statics.iter().fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value));
    let entry = indexer_keys::native_entry_with(indexer_keys::Elements { A: Element::bf16() });
    let specialization = entry.offered(device).unwrap().unwrap().default_specialization(&specialization).unwrap();
    let kernel = entry.prepare(device, &specialization).unwrap();
    let (_slabs, mut keys) = plane(device, Element::f32(), case.plane_rows, &vec![SENTINEL; case.plane_rows * W]);
    kernel
        .call(indexer_keys::Args {
            key: &tensor(device, Element::bf16(), &[m, W as u64], &case.key),
            key_norm: &tensor(device, Element::f32(), &[1, W as u64], &case.key_norm),
            latent_rows: &ints(device, &[m], &case.latent_rows),
            rotary_components: &ints(device, &[P as u64], &[0; P]),
            rotary_frequencies: &tensor(device, Element::f32(), &[P as u64], &frequencies()),
            rotary_amplitudes: &tensor(device, Element::f32(), &[P as u64], &[1.0; P]),
            coordinates: &ints(device, &[m, 4], &case.coordinates()),
            keys: &mut keys,
            form: &Tensor::from_host(device, Element::i32(), &[0, case.layout.ra as u64, case.layout.rt as u64], &[]).unwrap(),
            epsilon: EPSILON,
            block: case.rounding.block,
            element: case.rounding.element,
            scale: case.rounding.scale,
            amax_floor: case.rounding.amax_floor,
            slab_rows: SLAB_ROWS as u32,
        })
        .unwrap();
    read(&keys)
}

fn check(case: &Case, label: &str) {
    let expected = case.expected();
    let compare = |actual: &[f32], who: &str| {
        for at in 0..case.plane_rows {
            let (a, e) = (&actual[at * W..][..W], &expected[at * W..][..W]);
            if case.latent_rows.contains(&(at as i32)) {
                assert!(near_codes(a, e, case.rounding, 2), "{who} {label}: key row {at} {a:?} vs model {e:?}");
            } else {
                assert!(same_bits(a, e), "{who} {label}: plane row {at} written");
            }
        }
    };
    if case.rows <= 8 {
        compare(&portable(case), "portable");
    }
    for device in offering(&indexer_keys::native_entry_with(indexer_keys::Elements { A: Element::bf16() })) {
        compare(&native(&device, case), device.backend().as_str());
    }
}

#[test]
fn indexer_keys_equal_the_reference_keys() {
    check(&Case::new(8, DEEPSEEK, INDEX, 1), "deepseek small");
    check(&Case::new(256, DEEPSEEK, INDEX, 2), "deepseek prefill");
    check(&Case::new(8, HALF_SPLIT, NONE, 3), "half-split unrounded");
}
