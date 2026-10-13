// Reads of a streamed gathered table through its directory (`gathered_row`,
// target.seismic), dense and hashed, on every backend this host opens:
//
// - `stored_rows` decodes the stored IQ4_NL rows its directory names exactly
//   as its portable body does and as an independent host decode does;
// - `embedding_rows` through a hashed directory reads what a direct read of
//   the slots it names reads.

mod versions_common;

use magnitude_kernels::{embedding_rows, stored_rows};
use seismic::{Device, Element, NativeSpecialization, Tensor};
use versions_common::*;

const IQ4: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];
const EMPTY: i32 = -1;
const TOMBSTONE: i32 = -2;

/// murmur3's 32-bit finalizer masked to `buckets`: `gathered_bucket`.
fn bucket(row: i32, buckets: usize) -> usize {
    let mut h = row as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h as usize & (buckets - 1)
}

/// A table's directory over `rows` rows whose slot `s` holds row
/// `held[s]`, in both forms: dense `[rows, 1]`, and hashed `[buckets, 2]`
/// with a tombstone in every other bucket a probe passes before its row.
fn directories(rows: usize, held: &[usize], buckets: usize) -> (Vec<i32>, Vec<i32>) {
    let mut dense = vec![EMPTY; rows];
    let mut hashed = vec![[EMPTY, EMPTY]; buckets];
    for (slot, row) in held.iter().enumerate() {
        dense[*row] = slot as i32;
        let mut probe = bucket(*row as i32, buckets);
        while hashed[probe][0] != EMPTY {
            probe = (probe + 1) % buckets;
        }
        hashed[probe] = [*row as i32, slot as i32];
    }
    // Tombstones: rows that left, which every probe passes.
    let mut tombstones = 0;
    for probe in 0..buckets {
        if hashed[probe][0] == EMPTY && probe % 3 == 0 && tombstones < buckets / 8 {
            hashed[probe] = [TOMBSTONE, EMPTY];
            tombstones += 1;
        }
    }
    (dense, hashed.into_iter().flatten().collect())
}

/// The row a read of selection `selected` finds through the dense
/// directory: its slot, or slot 0.
fn found(dense: &[i32], selected: i32) -> usize {
    usize::try_from(selected).map_or(0, |row| dense[row].max(0) as usize)
}

/// Stored IQ4_NL rows of `width` values: per 32-value block an f16 scale
/// (finite, of either sign, some subnormal) and sixteen bytes of nibbles.
fn stored(random: &mut Random, rows: usize, width: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for _ in 0..rows * width / 32 {
        let scale = random.next() * 0.01;
        let bits = match bytes.len() % 7 {
            0 => 0x0003, // subnormal
            _ => seismic_lang::registry::f16_bits(scale),
        };
        bytes.extend_from_slice(&bits.to_le_bytes());
        for _ in 0..16 {
            bytes.push((random.next().to_bits() >> 8) as u8);
        }
    }
    bytes
}

/// The host decode of stored row `row`, rounded to bf16.
fn decode(stored: &[u8], row: usize, width: usize) -> Vec<f32> {
    let row_bytes = width / 32 * 18;
    let row = &stored[row * row_bytes..(row + 1) * row_bytes];
    (0..width)
        .map(|column| {
            let block = &row[column / 32 * 18..][..18];
            let scale = seismic_lang::registry::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let within = column % 32;
            let code = (block[2 + within % 16] >> (4 * (within / 16))) & 15;
            bf16_round(IQ4[code as usize] * scale)
        })
        .collect()
}

fn stored_rows_native(device: &Device, width: u64) -> seismic::NativeKernel<stored_rows::Entry> {
    let spec = NativeSpecialization::new().with_static("W", width);
    let spec = seismic::generated::native_implementation::<stored_rows::Entry>(device)
        .unwrap()
        .unwrap()
        .default_specialization(&spec)
        .unwrap();
    stored_rows::native_for_device_with(
        device,
        stored_rows::Elements {
            E: Element::named("gguf_iq4_nl_block").unwrap(),
            A: Element::bf16(),
        },
        &spec,
    )
    .unwrap()
}

/// The portable body's rows, from its interpreter.
fn stored_rows_portable(
    slots: &[u8],
    shape: [usize; 2],
    tokens: &[i32],
    directory: &[i32],
    directory_shape: [usize; 2],
) -> Vec<f32> {
    use seismic_lang::{
        checked::{SourceFile, check_source},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        registry,
        types::DType,
    };
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "target.seismic".into(),
        text: include_str!("../kernels/target.seismic").into(),
    });
    let module = check_source(sources).unwrap();
    let elements = ElementBindings::new()
        .bind("E", registry::representation("gguf_iq4_nl_block").unwrap())
        .bind("A", registry::dense(DType::BF16));
    let logical = module
        .entry(module.entry_named("stored_rows").unwrap(), &elements)
        .unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let ints = |shape: Vec<usize>, values: &[i32]| {
        TensorData::dense(
            DType::I32,
            shape,
            values.iter().map(|value| f64::from(*value)).collect(),
        )
    };
    let arguments = [
        TensorData::encoded(
            registry::representation("gguf_iq4_nl_block").unwrap(),
            shape.to_vec(),
            slots.to_vec(),
        )
        .unwrap(),
        ints(vec![tokens.len() / 2, 2], tokens),
        ints(directory_shape.to_vec(), directory),
    ]
    .into_iter()
    .map(|tensor| Arg::Tensor(interpreter.add_tensor(tensor)))
    .collect::<Vec<_>>();
    let outcome = interpreter.run_bounded(&arguments, u64::MAX).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("portable stored_rows failed: {failure}");
    }
    let values = match outcome.results().next().unwrap().value() {
        OutcomeValue::Tensor(reader) => (0..reader.element_count())
            .map(|index| reader.read(index).unwrap() as f32)
            .collect(),
        _ => panic!("the result is a tensor"),
    };
    values
}

#[test]
fn stored_rows_decode_the_rows_their_directory_names() {
    let (rows, slots, width, buckets) = (300usize, 24usize, 160usize, 64usize);
    let mut random = Random(41);
    let stored_slots = stored(&mut random, slots, width);
    let held = (0..slots).map(|slot| (slot * 37 + 5) % rows).collect::<Vec<_>>();
    let (dense, hashed) = directories(rows, &held, buckets);
    // Held rows, rows no slot holds, a failed selection.
    let selections = (0..20)
        .flat_map(|index| {
            let row = match index % 5 {
                4 => -1,
                3 => ((index * 11 + 1) % rows) as i32,
                _ => held[(index * 7) % slots] as i32,
            };
            [row, (index % 2) as i32]
        })
        .collect::<Vec<_>>();
    let expected = selections
        .chunks_exact(2)
        .flat_map(|pair| decode(&stored_slots, found(&dense, pair[0]), width))
        .collect::<Vec<_>>();
    let forms = [
        (dense.clone(), [rows, 1]),
        (hashed.clone(), [buckets, 2]),
        // The empty directory: the selections are slots themselves.
        (Vec::new(), [0, 1]),
    ];
    for (directory, shape) in &forms {
        let expected = match shape[0] {
            0 => selections
                .chunks_exact(2)
                .flat_map(|pair| {
                    let slot = usize::try_from(pair[0]).map_or(0, |row| row.min(slots - 1));
                    decode(&stored_slots, slot, width)
                })
                .collect::<Vec<_>>(),
            _ => expected.clone(),
        };
        let selections = match shape[0] {
            0 => selections
                .chunks_exact(2)
                .flat_map(|pair| [pair[0].min(slots as i32 - 1), pair[1]])
                .collect::<Vec<_>>(),
            _ => selections.clone(),
        };
        let portable =
            stored_rows_portable(&stored_slots, [slots, width], &selections, directory, *shape);
        assert!(same_bits(&portable, &expected), "portable, directory {shape:?}");
        for device in devices() {
            let slots_tensor = Tensor::from_host(
                &device,
                Element::named("gguf_iq4_nl_block").unwrap(),
                &[slots as u64, width as u64],
                &stored_slots,
            )
            .unwrap();
            let result = stored_rows_native(&device, width as u64)
                .call(stored_rows::Args {
                    slots: &slots_tensor,
                    tokens: &ints(&device, &[selections.len() as u64 / 2, 2], &selections),
                    directory: &ints(&device, &[shape[0] as u64, shape[1] as u64], directory),
                })
                .unwrap()
                .value;
            let actual = read(&result);
            assert!(
                same_bits(&actual, &expected),
                "{:?} directory {shape:?}: {:?}",
                device.backend(),
                first_difference(&actual, &expected)
            );
        }
    }
}

#[test]
fn embedding_rows_through_a_hashed_directory_read_the_slots_it_names() {
    let (rows, slots, width, buckets) = (500usize, 40usize, 64usize, 128usize);
    let mut random = Random(7);
    let table = (0..slots * width)
        .map(|_| random.next())
        .collect::<Vec<_>>();
    let held = (0..slots).map(|slot| (slot * 97 + 3) % rows).collect::<Vec<_>>();
    let (dense, hashed) = directories(rows, &held, buckets);
    let selections = (0..33)
        .flat_map(|index| {
            let row = match index % 4 {
                3 => ((index * 13) % rows) as i32,
                2 if index % 8 == 2 => -1,
                _ => held[(index * 5) % slots] as i32,
            };
            [row, 0]
        })
        .collect::<Vec<_>>();
    let direct = selections
        .chunks_exact(2)
        .flat_map(|pair| [found(&dense, pair[0]) as i32, pair[1]])
        .collect::<Vec<_>>();
    for device in devices() {
        let kernel = embedding_rows::native_for_device_with(
            &device,
            embedding_rows::Elements {
                EW: Element::bf16(),
                A: Element::bf16(),
            },
            &seismic::generated::native_implementation::<embedding_rows::Entry>(&device)
                .unwrap()
                .unwrap()
                .default_specialization(&match is_cpu(&device) {
                    // The CPU declares no static width.
                    true => NativeSpecialization::new(),
                    false => NativeSpecialization::new().with_static("D", width as u64),
                })
                .unwrap(),
        )
        .unwrap();
        let table = tensor(&device, Element::bf16(), &[slots as u64, width as u64], &table);
        let read_through = |selections: &[i32], directory: &[i32], shape: [u64; 2]| {
            let result = kernel
                .call(embedding_rows::Args {
                    table: &table,
                    tokens: &ints(&device, &[selections.len() as u64 / 2, 2], selections),
                    directory: &ints(&device, &shape, directory),
                    scale: 1.0,
                    normalize: 0,
                    epsilon: 0.0,
                })
                .unwrap();
            read(&result.r0)
        };
        let expected = read_through(&direct, &[], [0, 1]);
        for (directory, shape) in [
            (&dense, [rows as u64, 1]),
            (&hashed, [buckets as u64, 2]),
        ] {
            assert!(
                same_bits(&read_through(&selections, directory, shape), &expected),
                "{:?} directory {shape:?}",
                device.backend()
            );
        }
    }
}
