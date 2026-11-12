//! Expert stacks under a companion grid of block scales stream as records
//! of their stored parts: each expert's codes and its slice of the grid,
//! staged planar as the file stores them, converted in place into the
//! bank's resident form byte for byte as the resident import converts the
//! whole stack. At GLM-5.3-Flash's routed shape (gate and up
//! `[2048, 4096]`, down `[4096, 2048]`) under each admitted tile grid: F32
//! scales over 128×128 (GLM's own, fp8g128 resident) and UE8M0 ones over
//! 32×32 and 1×32 (mxfp8g32).

use super::bank::{HostTier, RecordTensor, RoutedTensor, routed_experts};
use super::fixtures::{LIMITS, RoutedShape, fixture_manifest, routed_fixture_definition};
use super::resources::FlightDepth;
use super::weights::representation_bytes;
use super::ModelLoadPlan;
use magnitude_artifacts::gguf::{Encoding, GgufArtifact, StoredHeader, TensorDescriptor};
use magnitude_family_contracts::{
    FeedForwardUp, ImportTransform, ModelDefinition, Operator, ScaleCodes, WeightDescriptor,
};
use seismic::{BackendName, Element};

/// The routed shape: GLM-5.3-Flash's widths, two experts a layer.
const SHAPE: RoutedShape = RoutedShape { layers: 1, experts: 2, selected: 1, hidden: 4096, intermediate: 2048 };

/// Host reservations backed byte by byte.
const BYTES: seismic::Reservation = seismic::Reservation { granule: 1, host_addressed: true, chunks: None };

/// A grid's codes and tile (one scale per tile of values), the
/// representation its stacks reside in, and an expert's read bytes at GLM's
/// shape (3 × 8,388,608 codes, then the grid slices): the three admitted
/// grids, GLM's own F32 grid over 128×128 first.
const GRIDS: [(ScaleCodes, [u64; 2], &str, u64); 3] = [
    (ScaleCodes::F32, [128, 128], "fp8g128", 25_171_968),
    (ScaleCodes::Ue8m0, [32, 32], "mxfp8g32", 25_190_400),
    (ScaleCodes::Ue8m0, [1, 32], "mxfp8g32", 25_952_256),
];

/// The fixture's one routed layer's stacks, in fill order (gate, up,
/// down).
fn stacks(definition: &mut ModelDefinition) -> [&mut WeightDescriptor; 3] {
    let routed = definition
        .decoder
        .blocks
        .iter_mut()
        .flat_map(|block| block.sublayers.iter_mut())
        .find_map(|sublayer| match &mut sublayer.op {
            Operator::RoutedFfn(routed) => Some(routed),
            _ => None,
        })
        .expect("the fixture routes");
    let FeedForwardUp::Gated { gate, up, .. } = &mut routed.expert_up else {
        unreachable!("the fixture's experts are gated");
    };
    [gate, up, &mut routed.expert_down]
}

/// The grid of a `[experts, rows, columns]` stack.
fn grid_shape(shape: &[u64], tile: [u64; 2]) -> Vec<u64> {
    vec![shape[0], shape[1] / tile[0], shape[2] / tile[1]]
}

/// `count` pseudo-random bytes of `seed`.
fn bytes(count: u64, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

/// `count` scales of `codes`, distinct and every one the importer reads:
/// F32 ones finite and at least 2^-117; UE8M0 ones any byte, NaN (0xff)
/// included, which the conversion moves bit for bit.
fn scales(codes: ScaleCodes, count: u64, seed: u32) -> Vec<u8> {
    match codes {
        ScaleCodes::F32 => (0..count as u32)
            .flat_map(|at| {
                let at = at.wrapping_add(seed);
                ((1.0 + (at % 7) as f32 / 8.0) * 2f32.powi(-((at % 13) as i32))).to_le_bytes()
            })
            .collect(),
        ScaleCodes::Ue8m0 => bytes(count, seed),
    }
}

/// The fixture under `codes`: its definition, whose stacks are stored
/// as `<stem>.weight` F8_E4M3 codes under their grid `<stem>.scale`, and
/// the GGUF holding those stacks and grids (the files the bank reads).
fn fixture(codes: ScaleCodes, tile: [u64; 2]) -> (ModelDefinition, std::path::PathBuf) {
    let mut definition = routed_fixture_definition(SHAPE);
    let mut stored = Vec::new();
    for stack in stacks(&mut definition) {
        stack.name = format!("{}.weight", stack.name);
        let scale = ImportTransform::companion_scale(&stack.name).expect("a weight's companion scale");
        stack.transforms = vec![ImportTransform::ScaleBlocks { tensor: scale.clone(), tile, codes }];
        stored.push((stack.name.clone(), stack.shape.clone(), Encoding::F8E4M3));
        stored.push((scale, grid_shape(&stack.shape, tile), codes.stored()));
    }
    let header = StoredHeader::new(3, Vec::new(), stored.clone()).unwrap();
    let mut file = header.encode().unwrap();
    let data = file.len();
    for (seed, (tensor, (_, shape, encoding))) in (0u32..).zip(header.tensors.iter().zip(&stored)) {
        let count = shape.iter().product::<u64>();
        file.resize(data + tensor.offset as usize, 0);
        file.extend_from_slice(&match encoding {
            Encoding::F8E4M3 => bytes(count, seed),
            _ => scales(codes, count, seed),
        });
    }
    let path = std::env::temp_dir().join(format!("magnitude-grid-records-{codes:?}-{}.gguf", std::process::id()));
    std::fs::write(&path, &file).unwrap();
    (definition, path)
}

/// The fixture's manifest: its other weights F16, its stacks and grids as
/// `artifact` stores them.
fn manifest(definition: &ModelDefinition, artifact: &GgufArtifact) -> magnitude_artifacts::PackageManifest {
    let mut manifest = fixture_manifest(definition);
    let stored = &artifact.directory().tensors;
    let tensors = &mut manifest.target.tensors;
    tensors.retain(|tensor| !stored.iter().any(|each| each.name == tensor.name));
    tensors.extend(stored.iter().cloned());
    let mut offset = 0;
    for tensor in tensors.iter_mut() {
        tensor.offset = offset;
        offset += tensor.nbytes;
    }
    manifest.target.files[0].size = offset;
    manifest
}

/// The stored tensor `name` of `artifact`.
fn tensor<'a>(artifact: &'a GgufArtifact, name: &str) -> &'a TensorDescriptor {
    artifact.directory().tensors.iter().find(|tensor| tensor.name == name).unwrap()
}

/// `bytes` bytes of `artifact`'s data at `offset` past tensor `name`'s.
fn read(artifact: &GgufArtifact, name: &str, offset: u64, bytes: u64) -> Vec<u8> {
    let tensor = tensor(artifact, name);
    let stored = crate::residency::Stored::from_gguf(
        artifact,
        &WeightDescriptor::stored(tensor.name.clone(), tensor.shape.clone()),
    )
    .unwrap();
    let (file, at, _) = stored.file_range();
    file.read(at + offset, bytes as usize).unwrap()
}

/// On every backend's resident layout, unified or beside a linked host
/// tier, a stack of FP8 experts under a grid of tile scales, F32
/// or UE8M0, streams:
/// - its records are admitted, each tensor's `[N, K]` in the grid's
///   resident representation;
/// - a record is read as its stored parts, each tensor's codes (`N·K`
///   bytes) then its grid slice (`N·K / 128²` scales), which the read
///   names, never the assembled blocks (3.1% larger under F32 scales);
/// - the bank's stored stacks name each expert's codes where the file
///   holds them;
/// - each expert's record, its parts as read staged planar (the codes,
///   then the grid slice) and converted in place by its record form's
///   conversion, is byte for byte that expert's slice of the resident
///   import of the whole stack: the record an all-hit step reads is the
///   resident one.
#[test]
fn fp8_experts_under_tile_grids_stream_records_of_their_stored_parts() {
    for (codes_of_grid, tile, representation, read_bytes) in GRIDS {
        let (definition, path) = fixture(codes_of_grid, tile);
        let artifact = GgufArtifact::open(&path).unwrap();
        let manifest = manifest(&definition, &artifact);
        let selection = crate::ComponentSelection { head: false, vision: false };
        let (n, k) = (SHAPE.intermediate, SHAPE.hidden);
        let matrix = |kind| match kind {
            RoutedTensor::Down => [k, n],
            _ => [n, k],
        };
        let codes = |kind| matrix(kind)[0] * matrix(kind)[1];
        let scale_bytes = codes_of_grid.stored().block_bytes();
        let grid = |kind| codes(kind) / (tile[0] * tile[1]) * scale_bytes;
        for (backend, host) in [(BackendName::Metal, HostTier::Unified(super::Parking::of(BackendName::Metal))), (BackendName::Cuda, HostTier::Linked(BYTES))] {
            let case = format!("{codes_of_grid:?} {tile:?} on {backend:?}");
            let layout = crate::resident_layout(crate::ExecutionPath::Native, backend);
            let resident = ModelLoadPlan::derive(&manifest, &definition, selection, layout, crate::PlannedMethod::Plain, LIMITS)
                .unwrap();
            let slots = routed_experts(&definition).unwrap();
            let banked = resident
                .clone()
                .with_expert_bank(&definition, slots, 1, FlightDepth::PIPELINED, super::bank::BankSubstrate { backend, reservation: BYTES, host })
                .unwrap_or_else(|error| panic!("{case}: {error:?}"));
            let bank = banked.bank.as_ref().expect("a banked plan holds its bank");
            let [group] = bank.groups() else {
                panic!("{case}: one routed layer is one group");
            };
            let record = group.record;
            for (kind, tensor) in record.iter() {
                let RecordTensor { resident: element, matrix: [rows, columns], .. } = *tensor;
                assert_eq!(element, Element::stored(representation, layout).unwrap(), "{case} {kind:?}");
                assert_eq!([rows, columns], matrix(kind), "{case} {kind:?}");
                assert_eq!(tensor.record_bytes(), representation_bytes(element, &[1, rows, columns]), "{case} {kind:?}");
            }
            let urgent = bank.urgent().unwrap();
            let parts = RoutedTensor::ALL.iter().flat_map(|&kind| [codes(kind), grid(kind)]).collect::<Vec<_>>();
            assert_eq!(urgent.parts, parts, "{case}: a record is read as its stored parts");
            assert_eq!(urgent.parts.iter().sum::<u64>(), read_bytes, "{case}");
            // One read per part of an expert's record: its three tensors' codes and grids.
            assert_eq!(urgent.parts.len(), 2 * RoutedTensor::ALL.len(), "{case}: a record is six reads");
            // On a 24 GB device a record of these bytes keeps every expert a slot, the floor's one
            // included: the stored parts are priced at the bytes the record holds.
            assert_eq!(bank.most_slots(24 << 30).unwrap(), SHAPE.experts, "{case}: every expert has a slot");

            let [stacks] = &bank.stored_stacks(&artifact).unwrap()[..] else {
                panic!("{case}: one routed layer");
            };
            let layer = &bank.layers()[0];
            for (kind, stack) in stacks.iter() {
                let stored = tensor(&artifact, &layer.resident.get(kind).unwrap().descriptor.name);
                let scale = tensor(&artifact, &ImportTransform::companion_scale(&stored.name).unwrap());
                assert_eq!(stack.experts, SHAPE.experts, "{case} {kind:?}");
                let [codes_part, grid_part] = &stack.parts[..] else {
                    panic!("{case} {kind:?}: a record is its codes and its grid");
                };
                for (part, bytes, file) in [(codes_part, codes(kind), stored), (grid_part, grid(kind), scale)] {
                    assert_eq!((part.pitch, part.first, part.bytes), (bytes, 0, bytes), "{case} {kind:?}");
                    assert_eq!(part.pitch * SHAPE.experts, file.nbytes, "{case} {kind:?}: the part as stored");
                }
            }

            for (kind, tensor) in record.iter() {
                let name = &layer.resident.get(kind).unwrap().descriptor.name;
                let plan = resident.target.iter().find(|plan| &plan.descriptor.name == name).unwrap();
                let scale = ImportTransform::companion_scale(name).unwrap();
                let stored = crate::residency::Stored::from_gguf(&artifact, &plan.descriptor).unwrap();
                let (file, offset, nbytes) = stored.file_range();
                let blocks = crate::import_transforms::upload_bytes(plan, &stored, file.read(offset, nbytes as usize).unwrap()).unwrap();
                let imported = plan.resident.repack_host(plan.upload, &plan.shape, &blocks).expect("the resident import converts");
                let record_bytes = tensor.record_bytes().unwrap();
                assert_eq!(imported.len() as u64, SHAPE.experts * record_bytes, "{case} {kind:?}");
                for expert in 0..SHAPE.experts {
                    let mut staged = read(&artifact, name, expert * codes(kind), codes(kind));
                    staged.extend(read(&artifact, &scale, expert * grid(kind), grid(kind)));
                    let [rows, columns] = tensor.matrix;
                    let converted = tensor
                        .resident
                        .repack_host(tensor.source, &[1, rows, columns], &staged)
                        .unwrap_or_else(|| panic!("{case} {kind:?}: the record form converts its planar parts"));
                    let at = (expert * record_bytes) as usize;
                    assert!(
                        converted == imported[at..at + record_bytes as usize],
                        "{case} {kind:?}: expert {expert}'s record is its resident slice"
                    );
                }
            }
        }
        std::fs::remove_file(&path).unwrap();
    }
}
