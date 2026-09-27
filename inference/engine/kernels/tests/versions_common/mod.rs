// Shared fixtures of the bank-version mixer entry tests (`short_conv.rs`,
// `state_space.rs`): slots over bank versions, deterministic values, device
// discovery over every backend, host and slab-stored device tensors, and the
// portable-body interpreter.

#![allow(dead_code)]

use seismic::{BackendName, Device, DeviceCatalog, Element, SlabRegion, SlabTensor, Tensor};

pub const SENTINEL: f32 = 7.25;

/// One slot: its rows, published prefix, the version it reads (accepted bank
/// and its tape rows) and its successor bank.
#[derive(Clone, Copy, Debug)]
pub struct SlotCase {
    pub rows: usize,
    pub stop: usize,
    pub previous: usize,
    pub following: usize,
    pub taped: usize,
}

/// A slot reading bank `previous` with no tape rows.
pub fn slot(rows: usize, stop: usize, previous: usize, following: usize) -> SlotCase {
    SlotCase {
        rows,
        stop,
        previous,
        following,
        taped: 0,
    }
}

/// A slot reading version (`previous`, `taped`).
pub fn version(rows: usize, stop: usize, previous: usize, following: usize, taped: usize) -> SlotCase {
    SlotCase {
        rows,
        stop,
        previous,
        following,
        taped,
    }
}

/// The `segments` rows of `slots` packed from row 0, then the end row pair.
pub fn segments(slots: &[SlotCase], rows: usize) -> Vec<i32> {
    let mut segments = Vec::new();
    let mut row = 0;
    for slot in slots {
        segments.extend([row as i32, (row + slot.rows) as i32]);
        row += slot.rows;
    }
    segments.extend([rows as i32, rows as i32]);
    segments
}

/// The first row of every slot.
pub fn firsts(slots: &[SlotCase]) -> Vec<usize> {
    let mut first = 0;
    slots
        .iter()
        .map(|slot| {
            let at = first;
            first += slot.rows;
            at
        })
        .collect()
}

pub struct Random(pub u64);

impl Random {
    /// Uniform in [-1, 1).
    pub fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

pub fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

/// Every backend this host can open, the CPU always.
pub fn devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().unwrap();
    let devices = [
        BackendName::Metal,
        BackendName::Cuda,
        BackendName::Vulkan,
        BackendName::Cpu,
    ]
    .into_iter()
    .filter_map(|backend| catalog.open_backend(backend).ok())
    .collect::<Vec<_>>();
    println!(
        "backends: {}",
        devices.iter().map(|device| device.backend().as_str()).collect::<Vec<_>>().join(", ")
    );
    devices
}

pub fn is_cpu(device: &Device) -> bool {
    device.backend() == BackendName::Cpu
}

pub fn bytes(element: Element, values: &[f32]) -> Vec<u8> {
    if element == Element::bf16() {
        values
            .iter()
            .flat_map(|value| ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    } else {
        assert_eq!(element, Element::f32());
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }
}

pub fn tensor(device: &Device, element: Element, shape: &[u64], values: &[f32]) -> Tensor {
    Tensor::from_host(device, element, shape, &bytes(element, values)).unwrap()
}

pub fn ints(device: &Device, shape: &[u64], values: &[i32]) -> Tensor {
    Tensor::from_host(
        device,
        Element::i32(),
        shape,
        &values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unwrap()
}

pub fn read(tensor: &Tensor) -> Vec<f32> {
    let bytes = tensor.read_to_host().unwrap();
    if tensor.element() == Element::bf16() {
        bytes
            .chunks_exact(2)
            .map(|word| f32::from_bits(u32::from(u16::from_le_bytes(word.try_into().unwrap())) << 16))
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }
}

/// The slot control tensors of an entry.
pub struct SlotTensors {
    pub segments: Tensor,
    pub stop: Tensor,
    pub previous: Tensor,
    pub previous_tape: Tensor,
    pub following: Tensor,
}

impl SlotTensors {
    pub fn new(device: &Device, slots: &[SlotCase], rows: usize) -> Self {
        let count = slots.len() as u64;
        let each = |f: fn(&SlotCase) -> usize| slots.iter().map(|slot| f(slot) as i32).collect::<Vec<_>>();
        Self {
            segments: ints(device, &[count + 1, 2], &segments(slots, rows)),
            stop: ints(device, &[count], &each(|slot| slot.stop)),
            previous: ints(device, &[count], &each(|slot| slot.previous)),
            previous_tape: ints(device, &[count], &each(|slot| slot.taped)),
            following: ints(device, &[count], &each(|slot| slot.following)),
        }
    }
}

/// One bank arena: its element, the shape of one bank, and every bank's
/// values (banks × bank elements).
pub struct Arena<'a> {
    pub element: Element,
    pub bank_shape: Vec<u64>,
    pub values: &'a [f32],
}

/// Bank arenas stored in slabs of `slab_banks` banks, in one slab tensor
/// (one region per arena). With `reuse`, slab 1 is freed and its index
/// reused, so bank placement skips a released slot as it does after
/// compaction. Returns the slab tensor (kept alive by the caller) and each
/// arena's logical tensor.
pub fn slab_arenas(
    device: &Device,
    banks: usize,
    slab_banks: u64,
    arenas: &[Arena],
    reuse: bool,
) -> (SlabTensor, Vec<Tensor>) {
    let regions = arenas
        .iter()
        .map(|arena| SlabRegion {
            element: arena.element,
            row_shape: arena.bank_shape.clone(),
        })
        .collect();
    let mut slabs = SlabTensor::new(device, slab_banks, banks as u64, regions).unwrap();
    let count = (banks as u64).div_ceil(slab_banks);
    for slab in 0..count {
        assert_eq!(slabs.add_slab().unwrap(), slab as usize);
    }
    if reuse {
        assert!(count >= 3);
        slabs.free_slab(1).unwrap();
        assert_eq!(slabs.add_slab().unwrap(), 1);
    }
    for (region, arena) in arenas.iter().enumerate() {
        let contents = bytes(arena.element, arena.values);
        let row_bytes = contents.len() / banks;
        for slab in 0..count {
            let start = slab * slab_banks;
            let rows = slab_banks.min(banks as u64 - start);
            slabs
                .region_rows(region, start, rows)
                .unwrap()
                .write_from_host(&contents[start as usize * row_bytes..(start + rows) as usize * row_bytes])
                .unwrap();
        }
    }
    let logical = (0..arenas.len())
        .map(|region| slabs.logical_region(region).unwrap())
        .collect();
    (slabs, logical)
}

/// The largest absolute difference relative to the reference's RMS.
pub fn relative_max(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let scale = (expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / expected.len().max(1) as f64)
        .sqrt()
        .max(1e-12);
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| {
            assert!(a.is_finite(), "non-finite output");
            (*a as f64 - *e as f64).abs() / scale
        })
        .fold(0.0, f64::max)
}

pub fn same_bits(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len() && actual.iter().zip(expected).all(|(a, e)| a.to_bits() == e.to_bits())
}

/// The index of the first differing element, for messages.
pub fn first_difference(actual: &[f32], expected: &[f32]) -> Option<(usize, f32, f32)> {
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (a, e))| a.to_bits() != e.to_bits())
        .map(|(index, (a, e))| (index, *a, *e))
}
