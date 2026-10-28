// Reads of a streamed gathered table through its directory (`gathered_row`,
// target.seismic), dense and hashed, on every backend this host opens:
//
// - `stored_rows` decodes the resident rows its directory names exactly as
//   its portable body does, slots holding stored rows converted by the
//   registry's host repack; for IQ4_NL and Q8_0 also exactly as the decode
//   of the rows as stored does (what the slots held before they were
//   resident);
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
fn stored(random: &mut Random, rows: usize, width: usize, q8: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    for _ in 0..rows * width / 32 {
        let scale = random.next() * 0.01;
        let bits = match bytes.len() % 7 {
            0 => 0x0003, // subnormal
            _ => seismic_lang::registry::f16_bits(scale),
        };
        bytes.extend_from_slice(&bits.to_le_bytes());
        for _ in 0..if q8 { 32 } else { 16 } {
            bytes.push((random.next().to_bits() >> 8) as u8);
        }
    }
    bytes
}

/// The host decode of stored row `row`, rounded to bf16.
fn decode(stored: &[u8], row: usize, width: usize, q8: bool) -> Vec<f32> {
    let block_bytes = if q8 { 34 } else { 18 };
    let row_bytes = width / 32 * block_bytes;
    let row = &stored[row * row_bytes..(row + 1) * row_bytes];
    (0..width)
        .map(|column| {
            let block = &row[column / 32 * block_bytes..][..block_bytes];
            let scale =
                seismic_lang::registry::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let within = column % 32;
            let code = if q8 {
                block[2 + within] as i8 as f32
            } else {
                IQ4[((block[2 + within % 16] >> (4 * (within / 16))) & 15) as usize]
            };
            bf16_round(code * scale)
        })
        .collect()
}

/// Whether a native decode `actual` is the portable decode `expected`: the
/// same bits, or both zero (the shared decoders form a biased code's value
/// as one fused `d * code - offset * d`, whose exact cancellation is +0
/// where the registry's reference keeps the sign of `d * 0`), or both NaN
/// (random scale fields include NaNs, whose payloads no conversion or
/// decode keeps).
fn same_value(actual: f32, expected: f32) -> bool {
    actual.to_bits() == expected.to_bits()
        || (actual == 0.0 && expected == 0.0)
        || (actual.is_nan() && expected.is_nan())
}

fn stored_rows_native(
    device: &Device,
    width: u64,
    resident: Element,
) -> seismic::NativeKernel<stored_rows::Entry> {
    let spec = NativeSpecialization::new().with_static("W", width);
    let spec = seismic::generated::native_implementation::<stored_rows::Entry>(device)
        .unwrap()
        .unwrap()
        .default_specialization(&spec)
        .unwrap();
    stored_rows::native_for_device_with(
        device,
        stored_rows::Elements {
            E: resident,
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
    resident: &str,
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
        .bind("E", registry::representation(resident).unwrap())
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
            registry::representation(resident).unwrap(),
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
    // The 32-value block encodings an n-gram table of width 160 is stored
    // in, each with its rows16 resident form; IQ4_NL and Q8_0 (`Some(q8)`)
    // were read as stored before.
    for (source, resident, as_stored) in [
        ("gguf_iq4_nl", "iq4g32s@rows16", Some(false)),
        ("gguf_q8_0", "q8g32s@rows16", Some(true)),
        ("gguf_q4_0", "q4g32s@rows16", None),
        ("gguf_q5_0", "q5g32s@rows16", None),
        ("gguf_q5_1", "q5g32@rows16", None),
        ("gguf_q4_1", "q4g32@rows16", None),
    ] {
        stored_rows_match(source, resident, as_stored, 160);
    }
}

/// Rows 256 wide: of 256-value block encodings (Q2_K, Q3_K, IQ4_XS), and of
/// IQ4_NL blocks at a width that is a multiple of 256.
#[test]
fn stored_rows_decode_k_quant_rows() {
    for (source, resident) in [
        ("gguf_q2_k", "q2k@rows16"),
        ("gguf_q3_k", "q3k@rows16"),
        ("gguf_iq4_xs", "iq4xs@rows16"),
        ("gguf_iq4_nl", "iq4g32s@rows16"),
    ] {
        stored_rows_match(source, resident, None, 256);
    }
}

fn stored_rows_match(source: &str, resident: &str, as_stored: Option<bool>, width: usize) {
    let (rows, slots, buckets) = (300usize, 24usize, 64usize);
    let mut random = Random(41);
    let (source_element, resident_element) =
        (Element::named(source).unwrap(), Element::named(resident).unwrap());
    let stored_slots = match as_stored {
        Some(q8) => stored(&mut random, slots, width, q8),
        None => (0..source_element
            .canonical_byte_len(&[slots as u64, width as u64])
            .unwrap())
            .map(|_| (random.next().to_bits() >> 8) as u8)
            .collect(),
    };
    let stored_slots = resident_element
        .repack_source(source_element, stored_slots)
        .unwrap();
    let resident_slots = resident_element
        .repack_host(source_element, &[slots as u64, width as u64], &stored_slots)
        .unwrap();
    // The portable body is the reference of an encoding not read as stored
    // before.
    let decode = |slot: usize| match as_stored {
        Some(q8) => decode(&stored_slots, slot, width, q8),
        None => {
            let shape = [1, width];
            let row = resident_slots.len() / slots;
            stored_rows_portable(
                &resident_slots[slot * row..(slot + 1) * row],
                shape,
                &[0, 0],
                &[],
                [0, 1],
                resident,
            )
        }
    };
    let held = (0..slots)
        .map(|slot| (slot * 37 + 5) % rows)
        .collect::<Vec<_>>();
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
        .flat_map(|pair| decode(found(&dense, pair[0])))
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
                    decode(slot)
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
        let portable = stored_rows_portable(
            &resident_slots,
            [slots, width],
            &selections,
            directory,
            *shape,
            resident,
        );
        assert!(
            same_bits(&portable, &expected),
            "{source} portable, directory {shape:?}"
        );
        for device in devices() {
            let slots_tensor = Tensor::from_host(
                &device,
                resident_element,
                &[slots as u64, width as u64],
                &resident_slots,
            )
            .unwrap();
            let result = stored_rows_native(&device, width as u64, resident_element)
                .call(stored_rows::Args {
                    slots: &slots_tensor,
                    tokens: &ints(&device, &[selections.len() as u64 / 2, 2], &selections),
                    directory: &ints(&device, &[shape[0] as u64, shape[1] as u64], directory),
                })
                .unwrap()
                .value;
            let actual = read(&result);
            // Rows read as stored before decode to their former bits. The
            // others decode to the portable values (`same_value`).
            let decoded = match as_stored {
                Some(_) => same_bits(&actual, &expected),
                None => {
                    actual.len() == expected.len()
                        && actual.iter().zip(&expected).all(|(a, e)| same_value(*a, *e))
                }
            };
            assert!(
                decoded,
                "{source} {:?} directory {shape:?}: {:?}",
                device.backend(),
                actual
                    .iter()
                    .zip(&expected)
                    .enumerate()
                    .find(|(_, (a, e))| match as_stored {
                        Some(_) => a.to_bits() != e.to_bits(),
                        None => !same_value(**a, **e),
                    })
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
    let held = (0..slots)
        .map(|slot| (slot * 97 + 3) % rows)
        .collect::<Vec<_>>();
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
        let table = tensor(
            &device,
            Element::bf16(),
            &[slots as u64, width as u64],
            &table,
        );
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
        for (directory, shape) in [(&dense, [rows as u64, 1]), (&hashed, [buckets as u64, 2])] {
            assert!(
                same_bits(&read_through(&selections, directory, shape), &expected),
                "{:?} directory {shape:?}",
                device.backend()
            );
        }
    }
}
