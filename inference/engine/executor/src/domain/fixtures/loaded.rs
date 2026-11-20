//! Loads of a written GGUF as the engine's load makes them: weights
//! imported, graphs prepared, sealed and bound, a drafter bound on its
//! first work. Test-only: what tests of real steps, and an owner over a
//! real domain, run on.
// Most of it is the executor's own tests'; the `fixtures` feature serves
// only [`drafted`], [`plain`] and [`chaining`] above.
#![cfg_attr(not(test), allow(dead_code))]

use crate::planning::fixtures::{qwen4exp_fixture_definition, stored_table_manifest};
use crate::{
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, ExecutorDomain, PlannedMethod, ResidencyStore,
    ResourceAllocator, ResourceCapacity, ResourceDomainId, ResourceLimits, ResourcePlanner, domain::StatusReading, platform,
};
use magnitude_artifacts::{Package, gguf::Encoding};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::KvCodec;
use std::rc::Rc;
use std::sync::Arc;

/// A drafting load (an MTP head over the planning fixture's decoder) on
/// the CPU, its memory read as the fixture's alone
/// (`Machine::Still`), its drafter unbound until its first work: the
/// directory its GGUF was written to, for the test to remove once the
/// domain is dropped, the definition, the domain and its bindings.
pub fn drafted(
    test: &str,
) -> (
    std::path::PathBuf,
    Rc<ModelDefinition>,
    Box<ExecutorDomain>,
    crate::StateBindings,
) {
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let mtp = crate::planning::fixtures::with_mtp_head(crate::planning::fixtures::fixture_definition());
    let (directory, definition, package) = fixture_with(test, mtp, crate::planning::fixtures::fixture_manifest);
    let method = PlannedMethod::Mtp { greedy_proposals: 2, sampled_proposals: 2, widths: crate::DraftWidths::Adaptive };
    let (domain, bindings, _, _) = load_read(&LoadRequest {
        catalog: &catalog,
        definition: &definition,
        package: &package,
        backend: seismic::BackendName::Cpu,
        limits: limits(4, false, 1, 0),
        banked: Banked::Resident,
        method,
        codec: KvCodec::Dense,
        machine: Machine::Still,
    });
    (directory, definition, domain, bindings)
}

/// A plain load of the planning fixture's decoder at `context` rows on the
/// CPU, its memory read as the fixture's alone (`Machine::Still`): what an
/// owner's tests of a request writing toward a long context decode on. The
/// directory its GGUF was written to, the definition, the domain and its
/// bindings, as [`drafted`] gives them.
pub fn plain(
    test: &str,
    context: u64,
) -> (
    std::path::PathBuf,
    Rc<ModelDefinition>,
    Box<ExecutorDomain>,
    crate::StateBindings,
) {
    chaining(test, context, 1)
}

/// [`plain`], its prompt groups chaining up to `chain_capacity` chunks.
pub fn chaining(
    test: &str,
    context: u64,
    chain_capacity: usize,
) -> (
    std::path::PathBuf,
    Rc<ModelDefinition>,
    Box<ExecutorDomain>,
    crate::StateBindings,
) {
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let mut definition = crate::planning::fixtures::fixture_definition();
    definition.decoder.context_limit = context;
    let (directory, definition, package) = fixture_with(test, definition, crate::planning::fixtures::fixture_manifest);
    let (domain, bindings, _, _) = load_read(&LoadRequest {
        catalog: &catalog,
        definition: &definition,
        package: &package,
        backend: seismic::BackendName::Cpu,
        limits: limits(4, false, chain_capacity, 0),
        banked: Banked::Resident,
        method: PlannedMethod::Plain,
        codec: KvCodec::Dense,
        machine: Machine::Still,
    });
    (directory, definition, domain, bindings)
}

/// A GGUF of every tensor `manifest` names (its n-gram tables IQ4_NL
/// blocks, a streamed load's expert stacks Q8_0 or Q4_0 blocks, everything else
/// F16, BF16 or F32 as it lays them out), with small deterministic values:
/// each float tensor `constant` names holds its value throughout.
///
/// The payload is generated in fixed-size chunks and streamed straight to the
/// file, with no full-payload in-memory copy.
pub(crate) fn write_gguf(
    manifest: &magnitude_artifacts::PackageManifest,
    path: &std::path::Path,
    constant: &dyn Fn(&str) -> Option<f32>,
) {
    use std::io::Write;

    const CHUNK_BYTES: usize = 1 << 20;

    fn tensor_bytes(encoding: Encoding, values: usize) -> usize {
        match encoding {
            Encoding::Iq4Nl | Encoding::Q4_0 => {
                assert_eq!(values % 32, 0, "{encoding:?} tensor must be a multiple of 32 values");
                (values / 32) * 18
            }
            Encoding::Q8_0 => {
                assert_eq!(values % 32, 0, "Q8_0 tensor must be a multiple of 32 values");
                (values / 32) * 34
            }
            Encoding::F16 | Encoding::BF16 => values * 2,
            Encoding::F32 => values * 4,
            Encoding::F8E4M3 => values,
            other => panic!("the fixture stores no {other:?} tensor"),
        }
    }

    fn encode_float(encoding: Encoding, value: f32) -> [u8; 4] {
        match encoding {
            Encoding::F16 => {
                let bytes = half_bits(value).to_le_bytes();
                [bytes[0], bytes[1], 0, 0]
            }
            Encoding::BF16 => {
                let bytes = ((value.to_bits() >> 16) as u16).to_le_bytes();
                [bytes[0], bytes[1], 0, 0]
            }
            Encoding::F32 => value.to_le_bytes(),
            other => unreachable!("not a float encoding: {other:?}"),
        }
    }

    let mut header = Vec::new();
    let string = |bytes: &mut Vec<u8>, value: &str| {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    };
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(manifest.target.tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());

    let mut offset = 0usize;
    for tensor in &manifest.target.tensors {
        offset = offset.div_ceil(32) * 32;
        string(&mut header, &tensor.name);
        header.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for extent in tensor.shape.iter().rev() {
            header.extend_from_slice(&extent.to_le_bytes());
        }
        header.extend_from_slice(&(tensor.encoding as u32).to_le_bytes());
        header.extend_from_slice(&(offset as u64).to_le_bytes());
        let values = tensor.shape.iter().product::<u64>() as usize;
        offset += tensor_bytes(tensor.encoding, values);
    }
    header.resize(header.len().div_ceil(32) * 32, 0);

    let file = std::fs::File::create(path).unwrap();
    let mut writer = std::io::BufWriter::with_capacity(CHUNK_BYTES, file);
    writer.write_all(&header).unwrap();

    let mut state = 0x5eed_u64;
    let mut next_u64 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut chunk = vec![0u8; CHUNK_BYTES];
    let mut written = 0usize;
    for tensor in &manifest.target.tensors {
        let pad = written.div_ceil(32) * 32 - written;
        if pad > 0 {
            writer.write_all(&[0u8; 32][..pad]).unwrap();
            written += pad;
        }
        let values = tensor.shape.iter().product::<u64>() as usize;
        let bytes = tensor_bytes(tensor.encoding, values);
        match tensor.encoding {
            // Both 32 values a block: d = 1/64, then 16 bytes of 4-bit codes.
            Encoding::Iq4Nl | Encoding::Q4_0 => {
                let d = 0x2400u16.to_le_bytes();
                let mut remaining = values / 32;
                let per_chunk = CHUNK_BYTES / 18;
                while remaining > 0 {
                    let count = remaining.min(per_chunk);
                    let out = &mut chunk[..count * 18];
                    for block in out.chunks_exact_mut(18) {
                        block[..2].copy_from_slice(&d);
                        block[2..10].copy_from_slice(&next_u64().to_le_bytes());
                        block[10..18].copy_from_slice(&next_u64().to_le_bytes());
                    }
                    writer.write_all(out).unwrap();
                    remaining -= count;
                }
            }
            // 32 values a block: d = 1/1024, then 32 8-bit codes.
            Encoding::Q8_0 => {
                let d = 0x1400u16.to_le_bytes();
                let mut remaining = values / 32;
                let per_chunk = CHUNK_BYTES / 34;
                while remaining > 0 {
                    let count = remaining.min(per_chunk);
                    let out = &mut chunk[..count * 34];
                    for block in out.chunks_exact_mut(34) {
                        block[..2].copy_from_slice(&d);
                        block[2..10].copy_from_slice(&next_u64().to_le_bytes());
                        block[10..18].copy_from_slice(&next_u64().to_le_bytes());
                        block[18..26].copy_from_slice(&next_u64().to_le_bytes());
                        block[26..34].copy_from_slice(&next_u64().to_le_bytes());
                    }
                    writer.write_all(out).unwrap();
                    remaining -= count;
                }
            }
            Encoding::F16 | Encoding::BF16 | Encoding::F32 => {
                let elem = match tensor.encoding {
                    Encoding::F32 => 4,
                    _ => 2,
                };
                let per_chunk = CHUNK_BYTES / elem;
                if let Some(value) = constant(&tensor.name) {
                    let encoded = encode_float(tensor.encoding, value);
                    let fill = values.min(per_chunk);
                    for slot in chunk[..fill * elem].chunks_exact_mut(elem) {
                        slot.copy_from_slice(&encoded[..elem]);
                    }
                    let mut remaining = values;
                    while remaining > 0 {
                        let count = remaining.min(fill);
                        writer.write_all(&chunk[..count * elem]).unwrap();
                        remaining -= count;
                    }
                } else {
                    // A gated delta decay rate (`ssm_a`, -exp(A_log)) is negative,
                    // as every model's is, so the recurrent state decays: a
                    // positive one grows it exponentially along the prompt.
                    let decay_rate = tensor.name.ends_with(".ssm_a");
                    let mut remaining = values;
                    while remaining > 0 {
                        let count = remaining.min(per_chunk);
                        let out = &mut chunk[..count * elem];
                        for slot in out.chunks_exact_mut(elem) {
                            // Within ±1/16: a small value of the tensor's element.
                            let value = ((next_u64() % 2001) as f32 - 1000.0) / 16_000.0;
                            let value = if decay_rate { -value.abs() } else { value };
                            slot.copy_from_slice(&encode_float(tensor.encoding, value)[..elem]);
                        }
                        writer.write_all(out).unwrap();
                        remaining -= count;
                    }
                }
            }
            // E4M3 codes of a sign, an exponent of one to eight and a mantissa: finite, none of the NaN codes.
            Encoding::F8E4M3 => {
                let e4m3 = |code: u8| (code & 0x80) | ((1 + ((code >> 3) & 7)) << 3) | (code & 7);
                let mut remaining = values;
                while remaining > 0 {
                    let count = remaining.min(CHUNK_BYTES);
                    let out = &mut chunk[..count];
                    let mut chunks = out.chunks_exact_mut(8);
                    for group in &mut chunks {
                        let raw = next_u64().to_le_bytes();
                        for (dst, src) in group.iter_mut().zip(raw) {
                            *dst = e4m3(src);
                        }
                    }
                    let rem = chunks.into_remainder();
                    if !rem.is_empty() {
                        let raw = next_u64().to_le_bytes();
                        for (dst, src) in rem.iter_mut().zip(raw) {
                            *dst = e4m3(src);
                        }
                    }
                    writer.write_all(out).unwrap();
                    remaining -= count;
                }
            }
            other => panic!("the fixture stores no {other:?} tensor"),
        }
        written += bytes;
    }
    assert_eq!(written, offset, "GGUF payload length matches header offsets");
    writer.flush().unwrap();
}

/// The F16 bits nearest a small normal `value`.
pub(crate) fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return sign;
    }
    let mantissa = (bits >> 13) & 0x3ff;
    sign | ((exponent as u16) << 10) | mantissa as u16
}

/// What a test load is priced on, as a load's worker prices it once its
/// warm-up ends (`LoadCosts::of`): the executor's costs, which every load
/// waits for before it serves (the synthetic costs here,
/// `assessment::synthetic::costs`), and the load's plans.
pub(crate) struct Pricing {
    pub(crate) definition: Rc<ModelDefinition>,
    /// The plan before any bank.
    pub(crate) resident: crate::ExecutionPlanDraft,
    /// The plan the load runs.
    pub(crate) planned: crate::ExecutionPlanDraft,
    pub(crate) costs: crate::assessment::ExecutorCosts,
}

impl Pricing {
    /// The load's measures: its costs; a banked load's experts by `lines`
    /// and `convert`, the routed pass the costs price, and every other
    /// input pending, as the job has not stored it.
    pub(crate) fn measures(
        &self,
        lines: crate::assessment::Priced<std::sync::Arc<[magnitude_cost_model::ExpertLines]>>,
        convert: crate::assessment::Priced<Option<std::sync::Arc<[magnitude_cost_model::RecordLine]>>>,
    ) -> crate::LoadMeasures {
        use crate::assessment::Priced;
        fn pending<T>() -> Priced<T> {
            Priced::Pending("not measured by the test".to_owned())
        }
        let experts = self.planned.load().expert_bank().map(|bank| crate::ExpertMeasures {
            lines,
            read: pending(),
            pass: Priced::Measured(crate::assessment::expert_pass(bank, &self.costs).expect("the synthetic costs price the bank's pass")),
            tier_copy: pending(),
            convert,
        });
        let costs = crate::LoadCosts::of(&self.definition, &self.resident, &self.planned, &self.costs).map_err(|error| error.to_string());
        crate::LoadMeasures { costs, experts }
    }
}

/// The fixture decoder written as a GGUF in a fresh directory named for
/// `test`, and its definition naming the written artifact.
pub(crate) fn fixture(test: &str) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    fixture_of(test, stored_table_manifest)
}

/// [`fixture`] of the tensors `manifest` lays out.
pub(crate) fn fixture_of(
    test: &str,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    fixture_with(test, qwen4exp_fixture_definition(), manifest)
}

/// [`fixture_of`] the decoder `definition`.
pub(crate) fn fixture_with(
    test: &str,
    definition: ModelDefinition,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    fixture_valued(test, definition, manifest, &|_| None)
}

/// [`fixture_with`], each tensor `constant` names holding its value.
pub(crate) fn fixture_valued(
    test: &str,
    definition: ModelDefinition,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
    constant: &dyn Fn(&str) -> Option<f32>,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    let manifest = manifest(&definition);
    fixture_stored(test, definition, &manifest, constant)
}

/// [`fixture_valued`] of the tensors `manifest` names.
pub(crate) fn fixture_stored(
    test: &str,
    mut definition: ModelDefinition,
    manifest: &magnitude_artifacts::PackageManifest,
    constant: &dyn Fn(&str) -> Option<f32>,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    let directory = std::env::temp_dir().join(format!(
        "magnitude-{test}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("qwen4exp-fixture.gguf");
    write_gguf(manifest, &path, constant);
    let package = Package::open_without_projector(&path).unwrap();
    definition.artifact_identity = package.manifest().identity;
    (directory, Rc::new(definition), Arc::new(package))
}

/// Launches of `rows` rows, `lookahead` or not, in chains of up to
/// `chain_capacity`, exporting up to `logits` rows of logits.
pub(crate) fn limits(rows: usize, lookahead: bool, chain_capacity: usize, logits: usize) -> ResourceLimits {
    ResourceLimits {
        max_launch_rows: rows,
        max_launch_slots: 2,
        max_selected_rows: 2,
        max_drafting_slots: 2,
        exported_logits_rows: logits,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        max_image_cells: 0,
        lookahead,
        chain_capacity: std::num::NonZeroUsize::new(chain_capacity).unwrap(),
        conversations: std::num::NonZeroUsize::MIN,
    }
}

/// How a load holds its routed experts: resident, streamed through an
/// expert bank whose weights stay within a share of the resident load's,
/// in a bank as an uncapped load holds them, or in a bank of about half
/// the slots.
#[derive(Clone, Copy)]
pub(crate) enum Banked {
    Resident,
    /// The bank's cap as a fraction of the resident load's allocation.
    Streamed { numerator: u64, denominator: u64 },
    /// No cap.
    Whole,
    /// Capped at the whole bank's weights less half its slots'.
    Half,
}

/// Load `definition` from `package` on `backend`, an available one, as the
/// engine's load does, within `limits`, with the most rows a group's pass
/// covers (`group_pass_rows`).
pub(crate) fn load(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    limits: ResourceLimits,
    banked: Banked,
) -> (Box<ExecutorDomain>, crate::StateBindings, u64, Pricing) {
    load_with(catalog, definition, package, backend, limits, banked, PlannedMethod::Plain)
}

/// [`load`] serving `method`: one that drafts binds the definition's drafter
/// as the engine's load does, its graphs sealed beside the target's and its
/// state allocated.
pub(crate) fn load_with(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    limits: ResourceLimits,
    banked: Banked,
    method: PlannedMethod,
) -> (Box<ExecutorDomain>, crate::StateBindings, u64, Pricing) {
    load_read(&LoadRequest {
        catalog,
        definition,
        package,
        backend,
        limits,
        banked,
        method,
        codec: KvCodec::Dense,
        machine: Machine::Live,
    })
}

/// What a loaded fixture's memory readings read.
#[derive(Clone, Copy)]
pub(crate) enum Machine {
    /// The live machine.
    Live,
    /// The machine as first read, moved only by the fixture's own
    /// allocations (`StillHost`): what a test decides by memory is the
    /// fixture's alone.
    Still,
}

/// A test load: `definition` from `package` on `backend`, an available one,
/// within `limits`, its experts held as `banked`, serving `method`, its
/// history kept in `codec` and its memory read as `machine`.
pub(crate) struct LoadRequest<'a> {
    pub(crate) catalog: &'a seismic::Devices,
    pub(crate) definition: &'a Rc<ModelDefinition>,
    pub(crate) package: &'a Arc<Package>,
    pub(crate) backend: seismic::BackendName,
    pub(crate) limits: ResourceLimits,
    pub(crate) banked: Banked,
    pub(crate) method: PlannedMethod,
    pub(crate) codec: KvCodec,
    pub(crate) machine: Machine,
}

/// `request` loaded as the engine's load does: its device opened, its plan
/// compiled, its domain bound and priced. The domain, its bindings, the
/// most rows a group's pass covers (`group_pass_rows`) and its pricing.
pub(crate) fn load_read(request: &LoadRequest<'_>) -> (Box<ExecutorDomain>, crate::StateBindings, u64, Pricing) {
    // A hung step reports every owner and every thread's stack on SIGUSR1,
    // as the rig asks a job past its estimate.
    crate::diagnostics::install();
    let (selected, device, release) = open_device(request.catalog, request.backend);
    let (resident, draft) = plan(request, &selected);
    let Compiled { planned, execution, programs, charges, pass_rows } = compile(request, &selected, &device, draft);
    let (mut domain, mut bindings) = bind_domain(request, device, release, execution, programs, &charges);
    let pricing = price(&mut domain, &mut bindings, request, resident, planned);
    domain.observe_memory();
    (domain, bindings, pass_rows, pricing)
}

/// `backend`'s device, selected and opened as the engine's load selects it,
/// and the release rate it was selected by.
fn open_device(
    catalog: &seismic::Devices,
    backend: seismic::BackendName,
) -> (platform::SelectedDevice, Rc<seismic::Device>, platform::ReleaseRate) {
    let release = platform::ReleaseRate::of_machine(None);
    let selected = platform::select_device(
        catalog,
        ExecutionPath::Native,
        platform::DeviceRequest::Backend(backend),
        &release,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: an available device is selected: {error}"));
    let device = Rc::new(
        crate::test_support::open(catalog, catalog.resolve(selected.info.selector()).unwrap())
            .unwrap_or_else(|error| panic!("{backend:?}: the selected device opens: {error}")),
    );
    (selected, device, release)
}

/// `request`'s plan on `selected`: the plan before any bank, and the plan
/// holding its experts as `request.banked` does.
fn plan(
    request: &LoadRequest<'_>,
    selected: &platform::SelectedDevice,
) -> (crate::ExecutionPlanDraft, crate::ExecutionPlanDraft) {
    let LoadRequest { package, definition, backend, limits, banked, method, codec, .. } = *request;
    let draft = ExecutionPlanner::prepare(
        selected,
        &package.manifest(),
        definition,
        ComponentSelection {
            head: method.drafts(),
            vision: false,
        },
        ExecutionPath::Native,
        method,
        codec,
        limits,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let resident = draft.clone();
    (resident, banked.apply(draft, definition, backend))
}

impl Banked {
    /// `draft` holding its routed experts as this says.
    fn apply(
        self,
        draft: crate::ExecutionPlanDraft,
        definition: &ModelDefinition,
        backend: seismic::BackendName,
    ) -> crate::ExecutionPlanDraft {
        match self {
            Banked::Resident => draft,
            Banked::Streamed { numerator, denominator } => {
                let resident = draft.allocation_bytes(definition).unwrap();
                draft
                    .with_bank::<crate::AssessmentMemoryCharge>(definition, Some(crate::WeightBound::Cap(resident * numerator / denominator)), &crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices()))
                    .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
                    .into_draft()
            }
            Banked::Whole => draft.with_bank::<crate::AssessmentMemoryCharge>(definition, None, &crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices())).unwrap_or_else(|error| panic!("{backend:?}: {error}")).into_draft(),
            Banked::Half => {
                let whole = draft.clone().with_bank::<crate::AssessmentMemoryCharge>(definition, None, &crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices())).unwrap_or_else(|error| panic!("{backend:?}: {error}")).into_draft();
                let bank = whole.load().expert_bank().unwrap_or_else(|| panic!("{backend:?}: the uncapped load banks its experts"));
                let widest = bank.widest().get();
                let saved = bank.resident_charge(widest).unwrap() - bank.resident_charge(widest / 2).unwrap();
                let half = whole.load().weight_bytes().unwrap() - saved;
                let cap = half.min(draft.load().weight_bytes().unwrap() - 1);
                draft.with_bank::<crate::AssessmentMemoryCharge>(definition, Some(crate::WeightBound::Cap(cap)), &crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices())).unwrap_or_else(|error| panic!("{backend:?}: {error}")).into_draft()
            }
        }
    }
}

/// A plan compiled on its device ([`compile`]).
struct Compiled {
    /// The plan the load runs.
    planned: crate::ExecutionPlanDraft,
    execution: Rc<crate::ExecutionPlan>,
    programs: Rc<AttestedPrograms>,
    /// What the load allocates outside its claims: each store's own entry.
    charges: seismic::Charges,
    /// The most rows a group's pass covers.
    pass_rows: u64,
}

/// `draft` compiled on `device` as a load compiles it: its state planned,
/// its programs prepared, the plan finalized with their readout and sites,
/// its graphs prepared and installed, and the plan admitted with their
/// resources.
fn compile(
    request: &LoadRequest<'_>,
    selected: &platform::SelectedDevice,
    device: &Rc<seismic::Device>,
    draft: crate::ExecutionPlanDraft,
) -> Compiled {
    let LoadRequest { definition, backend, limits, codec, .. } = *request;
    let capacity = ResourceCapacity {
        domain_bytes: crate::StableRoom::of_selected(selected).capacity().unwrap().min(1024 * 1024 * 1024),
        tensor_operations: crate::TensorOperations::of(selected.info.forms_tensor_operations()),
        allocations: seismic::NativeAllocations::EXACT,
    };
    let state = ResourcePlanner::state_plan(
        definition,
        draft.load(),
        draft.policy().method(),
        codec,
        limits,
        capacity,
    )
    .unwrap();
    let charges = seismic::Charges::tally(device);
    let mut programs = AttestedPrograms::prepare_draft(
        &draft,
        &charges,
        crate::TuningContext {
            definition,
            weights: &crate::ZeroTuningWeights,
            observer: &crate::UnreportedTuning,
            cache: None,
            numerical_profile: crate::ResolvedNumericalProfile::standard(),
            settings: crate::TuningSettings::STANDARD,
        },
        crate::UnpreparedUnits::ServeDefaults,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    // As a load finalizes it (`execution`): its readout, and the CPU sites
    // its qualification admitted, which its graphs encode.
    let draft = draft
        .finalize_readout(definition, programs.readout_choices(), &crate::StableRoom::of_selected(selected))
        .and_then(|draft| draft.finalize_sites(|bank| programs.sited_layers(bank)))
        .unwrap();
    let (resources_plan, pass_rows) = prepare_graphs(request, device, &draft, &mut programs, state);
    let planned = draft.clone();
    let execution = Rc::new(draft.admit(resources_plan).unwrap());
    Compiled { planned, execution, programs: Rc::new(programs), charges, pass_rows }
}

/// `draft`'s target, readout and auxiliary graphs prepared on `device` over
/// `state` and installed in `programs`: the resources they plan and the most
/// rows a group's pass covers.
fn prepare_graphs(
    request: &LoadRequest<'_>,
    device: &Rc<seismic::Device>,
    draft: &crate::ExecutionPlanDraft,
    programs: &mut AttestedPrograms,
    state: crate::StateResourcePlan,
) -> (crate::ResourcePlan, u64) {
    let LoadRequest { definition, backend, limits, method, .. } = *request;
    let target_graphs = programs
        .prepare_target_graphs(
            device,
            draft.load(),
            &definition.decoder,
            &state,
            draft.programs().target(),
            limits,
            draft.load().pass(limits),
            &seismic::Workers::new(1),
        )
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let target_readout_graphs = programs
        .prepare_target_readout_graphs(device, draft.load(), &definition.decoder, limits)
        .unwrap();
    programs
        .prepare_auxiliary_graphs(
            device,
            draft.load(),
            definition,
            draft.programs(),
            &state,
            limits,
            method,
        )
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let resources_plan = ResourcePlanner::plan_with_state(
        state,
        &target_graphs,
        &target_readout_graphs,
        programs.drafter_graphs(),
        None,
        programs.state_graphs().unwrap(),
    )
    .unwrap();
    let pass_rows = target_graphs.pass_rows();
    programs.install_target_graphs(target_graphs);
    programs.install_target_readout_graphs(target_readout_graphs);
    (resources_plan, pass_rows)
}

/// The memory readings of a load on `device` as `machine` reads them, and
/// the ledgers of the stores it reaches, first read by them.
fn open_memory(
    machine: Machine,
    device: &Rc<seismic::Device>,
    execution: &Rc<crate::ExecutionPlan>,
) -> (Rc<dyn StatusReading>, crate::memory::ledger::DeviceLedgers) {
    match machine {
        Machine::Live => {
            let live = Rc::new(seismic::Devices::discover(&[]).unwrap());
            let ledgers = super::fixture_ledgers(device, execution, &*live);
            (live, ledgers)
        }
        Machine::Still => {
            let clock = super::TestClock::new();
            let still = Rc::new(super::StillHost::of(seismic::Devices::discover(&[]).unwrap(), device, super::Pressed::default(), clock.clone()));
            let ledgers = super::fixture_ledgers_at(device, execution, &*still, &clock);
            assert!(still.ledgers.set(ledgers.clone()).is_ok(), "a fixture opens its ledgers once");
            (still, ledgers)
        }
    }
}

/// The domain of `execution` and `programs` on `device`, bound as the
/// engine's load binds it: its arena allocated, the target imported under
/// its claims, the drafter's loader, its states and constants, and its
/// request slot. Its memory is read as `request.machine` reads it.
fn bind_domain(
    request: &LoadRequest<'_>,
    device: Rc<seismic::Device>,
    release: platform::ReleaseRate,
    execution: Rc<crate::ExecutionPlan>,
    programs: Rc<AttestedPrograms>,
    charges: &seismic::Charges,
) -> (Box<ExecutorDomain>, crate::StateBindings) {
    let LoadRequest { definition, package, backend, method, machine, .. } = *request;
    let identity = ResourceDomainId::new("ngram-steps").unwrap();
    let (reading, ledgers) = open_memory(machine, &device, &execution);
    // As the engine's load records them: the arena it allocated, the
    // target's weights, each charged to its claim as it was made.
    let mut recorded = crate::memory::ledger::Recorded::default();
    let scratch = crate::memory::ClaimDemand { device: execution.resources().bytes().scratch, ..Default::default() };
    let resources = crate::domain::fixtures::fixture_charged(
        &device,
        &ledgers,
        &mut recorded,
        crate::memory::ledger::CommittedKind::Arena,
        scratch,
        |charges| {
            ResourceAllocator::allocate(
                &execution,
                charges,
                identity.clone(),
                programs.target_graphs().unwrap(),
                programs.target_readout_graphs().unwrap(),
                programs.drafter_graphs(),
                None,
                programs.state_graphs().unwrap(),
                &ledgers,
            )
        },
    )
    .unwrap();
    // As the engine's load claims the target import, whose claim the host
    // tier of a capped bank backs its load-time slots in.
    let load = execution.load();
    let demand = match load.expert_bank() {
        Some(bank) => load.clone().with_load_slots(bank.slots).unwrap().target_weight_demand(),
        None => load.target_weight_demand(),
    }
    .unwrap();
    let mut weights = ledgers.changes().change(|keeping| ledgers.claim(&keeping.claims(), &device, crate::memory::ledger::CommittedKind::Load, demand)).unwrap();
    // As the engine's load claims them: its host tables' row caches.
    let host_tables = crate::memory::ClaimDemand { host: load.host_table_bytes().unwrap(), ..Default::default() };
    let host_tables = ledgers.changes().change(|keeping| ledgers.claim(&keeping.claims(), &device, crate::memory::ledger::CommittedKind::Load, host_tables)).unwrap();
    // As the engine's load opens them: the state's pool, and the bank's
    // records pool where it has a bank.
    let pools = crate::LoadPools::open(&device, &ledgers, &execution).unwrap();
    let mut residency = ResidencyStore::for_target(
        device.clone(),
        programs.clone(),
        (*execution).clone(),
        identity,
        &ledgers,
        &mut weights,
        pools.clone(),
    )
    .unwrap();
    let target = weights
        .allocate(&device, |charges| residency.load_target(charges, definition, package, &host_tables, Box::new(|_, _| ())))
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    recorded.record(weights);
    recorded.record(host_tables);
    // As the engine's load does: the drafter is imported from the store that
    // imported the target, when its first work binds it.
    let head_loader = match method.drafts() {
        true => Some(
            crate::ComponentLoader::head(residency, definition.clone(), package.clone())
                .unwrap_or_else(|error| panic!("{backend:?}: {error}")),
        ),
        false => None,
    };
    let plan = execution.resources().clone();
    let target_state = plan.allocate_target_state(charges, pools.state.clone()).unwrap();
    let head_state = plan.head_state().map(|state| state.allocate(charges, pools.state.clone()).unwrap());
    // The constants its domain binds, as the engine's load claims them.
    let constants = crate::memory::ClaimDemand {
        device: programs.target_graphs().unwrap().binding_constant_bytes().unwrap(),
        ..Default::default()
    };
    let constants = resources
        .ledgers()
        .changes()
        .change(|keeping| resources.ledgers().claim(&keeping.claims(), &device, crate::memory::ledger::CommittedKind::Load, constants))
        .unwrap();
    let request_slot = crate::domain::fixtures::request_slot(resources.ledgers(), &device, &execution, 0);
    let reader = crate::DeviceReader::open(reading.clone(), release, device).unwrap();
    let (domain, bindings) = ExecutorDomain::new(
        execution,
        definition.clone(),
        reader,
        programs,
        resources,
        head_loader,
        None,
        target,
        target_state,
        head_state,
        recorded,
        constants,
        request_slot,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    if let Machine::Still = machine {
        super::settle_ledgers(domain.domain.device(), domain.resources.ledgers(), &*reading);
    }
    (domain, bindings)
}

/// `domain` priced as a load ends, on the costs it waited for: the pricing
/// of its plan before any bank (`resident`) and the plan it runs
/// (`planned`).
fn price(
    domain: &mut ExecutorDomain,
    bindings: &mut crate::StateBindings,
    request: &LoadRequest<'_>,
    resident: crate::ExecutionPlanDraft,
    planned: crate::ExecutionPlanDraft,
) -> Pricing {
    let backend = request.backend;
    let pricing = Pricing { definition: request.definition.clone(), resident, planned, costs: crate::assessment::synthetic::costs() };
    let pending = "not measured by the test";
    domain
        .measure_load(pricing.measures(crate::assessment::Priced::Pending(pending.into()), crate::assessment::Priced::Pending(pending.into())), bindings)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    pricing
}
