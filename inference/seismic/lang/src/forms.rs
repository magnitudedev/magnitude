//! Consumer-form admission. Which of a backend's consumer forms a packed
//! row-layout weight takes is decided here, once, and emitted as
//! `<prefix>_FORM_<NAME>`: 0 or 1, and the values an admitted form reads its
//! codes with. Native libraries select on these and assert they implement
//! what is admitted, so a weight never falls back to another form by its
//! decoder's shape; host planning reads the same admission
//! (`metal_packing_of`).
//!
//! A form admits a weight when two independent facts hold:
//! - what the form's arithmetic requires of the weight's profile
//!   (`registry::profile::Profile`: its integer codes and levels), and
//! - that its library's producer delivers the form's operand from how the
//!   weight is stored (its decode axes: code planes, codebooks, grids and
//!   where its local codes live).
//!
//! A form's arithmetic requirements change only with the form, and its
//! producer's reach only with the library's producers.
//!
//! Native declarations read the same admission (`form(W, NAME)`, a
//! [`Form`]), so a configuration names only forms its weights take.
//!
//! The GPUs' forms are of packed rows, a dense weight having one path; the
//! CPU's are of every weight, whose dense rows take its quantized
//! activations too, and so are Vulkan's: `DECODED`, which admits a weight
//! to its natives at all, and the bytes of the table its decoder stages for
//! the weight (`TABLE`).
//!
//! A form whose producer reads every weight admits every weight and has no
//! row here: Metal's tall GEMM (`TALL`), whose staging simdgroups decode
//! any decoder's packets (`packets::Block`, dense rows included) into the
//! same half tiles the default GEMM stages, so its admission is its shape
//! guard alone (`dense.seismic`).

use crate::ids::RepresentationId;
use crate::registry::{self, BackendName, CodeInterpretation, OperandRecipe, RepresentationInfo, ValueCodes};
use crate::producers::{self, Weight};
use crate::registry::profile::{Bias, Domain, LocalScale, Profile, ValueMap};
use crate::types::DType;

/// CUDA's forms, in emission order: the folded GEMM recipe (`GEMM_FOLD`),
/// the int8 GEMM (`INT8`), the records GEMV (`GEMV_RECORDS`) and the int8
/// GEMM's local scale codes multiplied into the codes (`INT8_LOCAL`), negated
/// (`INT8_LOCAL_NEGATED`), whether its int8 codes are in entry order
/// (`S8_ENTRIES`), and the shared bytes that bound the int8 GEMM's 128-row
/// layout of the weight (`INT8_SHARED`, [`cuda_int8_shared`]). [`cuda_forms`]
/// values them in this order.
const CUDA_FORMS: [&str; 7] =
    ["GEMM_FOLD", "INT8", "GEMV_RECORDS", "INT8_LOCAL", "INT8_LOCAL_NEGATED", "S8_ENTRIES", "INT8_SHARED"];

/// Vulkan's forms: whether its packet decoder reads the weight (`DECODED`),
/// and the bytes of the table its decoder stages in shared memory for it
/// (`TABLE`). [`vulkan_decoded`] and [`vulkan_table`] value them.
const VULKAN_FORMS: [&str; 2] = ["DECODED", "TABLE"];


/// Metal's forms, in emission order (`packets::Forms` reads those its
/// library selects on). [`metal_forms`] values them in this order.
const METAL_FORMS: [&str; 13] = [
    "PACK",
    "PACK_CENTRE",
    "PACK_OFFSET",
    "PACK_RESIDUE",
    "PACK_FOLDS",
    "PACK_SUMS",
    "PACK_LANES",
    "INT8",
    "INT8_INTERLEAVED",
    "MATRIX_CODES",
    "MATRIX_RUNS",
    "MATRIX_SYMMETRIC",
    "MATRIX",
];

/// The CPU's forms, in emission order: the quantized dot (`INT8`) and its
/// tile products on AMX tiles (`INT8_AMX`). [`cpu_forms`] values them in this
/// order.
const CPU_FORMS: [&str; 2] = ["INT8", "INT8_AMX"];

/// Values of one CPU INT8 activation block (`quant::Q8Block`, the k-quant
/// super block): its `i8` codes, then the `i16` sums of each 16, then its
/// `f32` scale.
pub const CPU_Q8_VALUES: u64 = 256;
/// The alignment of a CPU INT8 activation block: the widest vector of any
/// CPU tier (`isa::Tier::vector_bytes`, the X86V4 tiers' 512-bit), so no
/// vector load of its codes straddles a cache line, wherever a block lies.
pub const CPU_Q8_ALIGNMENT: u64 = 64;
/// Bytes of one CPU INT8 activation block, its fields padded to its
/// alignment.
pub const CPU_Q8_BYTES: u64 =
    (CPU_Q8_VALUES + CPU_Q8_VALUES / 16 * 2 + 4).next_multiple_of(CPU_Q8_ALIGNMENT);

/// Activation rows of one CPU INT8 tile (`quant::Q8Tile`), the tile
/// kernel's strip: several rows quantized for one projection lie as tiles
/// of this many, each tile at most this many blocks' bytes, so `ceil_div(M,
/// Q8_ROWS) * Q8_ROWS` blocks a row hold either form.
pub const CPU_Q8_ROWS: u64 = 8;

/// Most participants a CPU worker pool holds (`seismic_cpu::workers::Workers`),
/// which bounds every `device.units` partition of a scratch buffer.
pub const CPU_MAX_UNITS: u64 = 256;

/// The named constants `backend`'s native declarations may use in their
/// expressions (`scratch quantized bytes (M * ceil_div(H, Q8_VALUES) *
/// Q8_BYTES)`), the facts of its forms they size buffers by.
pub fn constants(backend: BackendName) -> &'static [(&'static str, u64)] {
    match backend {
        BackendName::Cpu => &[
            ("Q8_VALUES", CPU_Q8_VALUES),
            ("Q8_BYTES", CPU_Q8_BYTES),
            ("Q8_ROWS", CPU_Q8_ROWS),
            ("MAX_UNITS", CPU_MAX_UNITS),
        ],
        BackendName::Vulkan => &[("FLOOR_SHARED_BYTES", crate::device::VULKAN_FLOOR_SHARED_BYTES)],
        BackendName::Cuda | BackendName::Metal => &[],
    }
}

/// The names of `backend`'s forms, in emission order.
pub const fn names(backend: BackendName) -> &'static [&'static str] {
    match backend {
        BackendName::Cuda => &CUDA_FORMS,
        BackendName::Metal => &METAL_FORMS,
        BackendName::Cpu => &CPU_FORMS,
        BackendName::Vulkan => &VULKAN_FORMS,
    }
}

/// The values of `backend`'s forms for representation `id` ([`admitted`]),
/// decided once per backend over the registry's representations: every
/// native node a graph binds reads them, and so does every host query of a
/// form (`metal_packing_of`, `cuda_gemm_fold_of`, `cpu_int8_of`), which
/// planning makes per weight of every model.
fn values(backend: BackendName, id: RepresentationId) -> Option<&'static [u32]> {
    type Table = std::sync::OnceLock<Vec<Option<Vec<u32>>>>;
    static CUDA: Table = Table::new();
    static METAL: Table = Table::new();
    static CPU: Table = Table::new();
    static VULKAN: Table = Table::new();
    let table = match backend {
        BackendName::Cuda => &CUDA,
        BackendName::Metal => &METAL,
        BackendName::Cpu => &CPU,
        BackendName::Vulkan => &VULKAN,
    };
    table.get_or_init(|| {
        let representations = registry::representations();
        let mut values = vec![None; representations.len()];
        for info in representations {
            values[info.id.position()] = admitted(backend, info.id);
        }
        values
    })[id.position()]
    .as_deref()
}

/// The values of `backend`'s forms for representation `id`, in the order of
/// [`names`]; `None` where the backend chooses no form of it: on Metal and
/// CUDA, storage other than packed rows.
fn admitted(backend: BackendName, id: RepresentationId) -> Option<Vec<u32>> {
    let info = registry::representation_info(id);
    if backend == BackendName::Vulkan {
        return Some(vec![u32::from(vulkan_decoded(info)), vulkan_table(info)]);
    }
    if backend == BackendName::Cpu {
        return Some(cpu_forms(info).to_vec());
    }
    let (registry::RepresentationKind::PackedRows(rows), Some(weight)) = (&info.kind, Weight::of(info)) else {
        return None;
    };
    Some(match backend {
        BackendName::Cuda => cuda_forms(rows, &weight, info.operands).to_vec(),
        BackendName::Metal => metal_forms(&weight).to_vec(),
        BackendName::Cpu | BackendName::Vulkan => Vec::new(),
    })
}

/// The forms admitted for representation `id` on `backend`, in emission
/// order, with their values; none where the backend chooses no form of it
/// (`values`).
pub fn forms(backend: BackendName, id: RepresentationId) -> Vec<(&'static str, u32)> {
    values(backend, id).map_or_else(Vec::new, |values| names(backend).iter().copied().zip(values.iter().copied()).collect())
}

/// One of a backend's forms, as a native declaration of that backend names
/// it (`form(W, NAME)`): only a name of [`names`] is represented, so its
/// value is defined for every representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "(BackendName, String)", into = "(BackendName, String)")]
pub struct Form {
    backend: BackendName,
    /// Its position in `names(backend)`.
    index: usize,
}

impl Form {
    /// Form `name` of `backend`, where the name is written here: the value
    /// of a constant (`const METAL_PACK: Form = ..`), so a backend without a
    /// form of that name fails the build.
    const fn constant(backend: BackendName, name: &str) -> Self {
        const fn same(a: &str, b: &str) -> bool {
            let (a, b) = (a.as_bytes(), b.as_bytes());
            if a.len() != b.len() {
                return false;
            }
            let mut at = 0;
            while at < a.len() {
                if a[at] != b[at] {
                    return false;
                }
                at += 1;
            }
            true
        }
        let names = names(backend);
        let mut index = 0;
        while index < names.len() {
            if same(names[index], name) {
                return Self { backend, index };
            }
            index += 1;
        }
        panic!("a backend has no form of this name")
    }

    /// Form `name` of `backend`, or `None` when the backend has no form of
    /// that name.
    pub fn named(backend: BackendName, name: &str) -> Option<Self> {
        let index = names(backend).iter().position(|candidate| *candidate == name)?;
        Some(Self { backend, index })
    }

    pub fn name(self) -> &'static str {
        names(self.backend)[self.index]
    }

    pub fn backend(self) -> BackendName {
        self.backend
    }

    /// The form's admitted value for representation `id` (`forms`): 0 where
    /// the backend chooses no form of it.
    pub fn of(self, id: RepresentationId) -> u32 {
        values(self.backend, id).map_or(0, |values| values[self.index])
    }

    /// Every value the form takes over the registry's representations, in
    /// increasing order.
    pub fn values(self) -> Vec<u32> {
        let values = registry::representations()
            .iter()
            .map(|info| self.of(info.id))
            .collect::<std::collections::BTreeSet<_>>();
        values.into_iter().collect()
    }
}

/// The forms whose reach their producers bound (`producers.decl`), as
/// admission asks of them.
const VULKAN_DECODED: Form = Form::constant(BackendName::Vulkan, "DECODED");
const CUDA_INT8_LOCAL: Form = Form::constant(BackendName::Cuda, "INT8_LOCAL");
pub(crate) const METAL_PACK: Form = Form::constant(BackendName::Metal, "PACK");
const METAL_INT8: Form = Form::constant(BackendName::Metal, "INT8");
const METAL_MATRIX_CODES: Form = Form::constant(BackendName::Metal, "MATRIX_CODES");
const METAL_MATRIX_RUNS: Form = Form::constant(BackendName::Metal, "MATRIX_RUNS");
const METAL_MATRIX_SYMMETRIC: Form = Form::constant(BackendName::Metal, "MATRIX_SYMMETRIC");

impl TryFrom<(BackendName, String)> for Form {
    type Error = String;

    fn try_from((backend, name): (BackendName, String)) -> Result<Self, Self::Error> {
        Self::named(backend, &name).ok_or_else(|| format!("`{}` has no form `{name}`", backend.as_str()))
    }
}

impl From<Form> for (BackendName, String) {
    fn from(form: Form) -> Self {
        (form.backend, form.name().to_owned())
    }
}

/// Whether Vulkan's packet decoder reads weight `info` (`DECODED`): dense
/// rows, and packed rows whose decode axes its producers decode, within
/// their requirements (`vulkan/producers.decl`). It reads neither packet
/// nor external storage, which a Vulkan native never binds.
fn vulkan_decoded(info: &RepresentationInfo) -> bool {
    match &info.kind {
        registry::RepresentationKind::Dense(_) => true,
        _ => Weight::of(info).is_some_and(|weight| producers::covers(VULKAN_DECODED, &weight)),
    }
}

/// The bytes of the table Vulkan's decoder of weight `info` stages in shared
/// memory once a workgroup (`packets_stage_tables`, `vulkan/packets.glsl`):
/// its grid's points as `u32` words (`GRIDS_WORDS_<ID>`), its codebook's
/// values as `f32`s (`CODEBOOKS_WORDS_<ID>`), or 0 for a weight that reads
/// neither. A declaration's launches that decode hold their weights' sum at
/// the end of their shared regions where every device holds the region with
/// it (`dense.seismic`); the library stages each distinct table there once.
fn vulkan_table(info: &RepresentationInfo) -> u32 {
    let Some(weight) = Weight::of(info) else {
        return 0;
    };
    let words = match weight.axes.coding.codes() {
        ValueCodes::Grid(grid) => grid.grid.points.words(),
        ValueCodes::Fields(CodeInterpretation::Table(codebook)) => codebook.values.len(),
        ValueCodes::Fields(_) | ValueCodes::E4m3 => 0,
    };
    u32::try_from(4 * words).expect("a table of u32 bytes")
}

/// CUDA's forms of a packed-rows weight, in the order of [`CUDA_FORMS`].
fn cuda_forms(rows: &registry::PackedRowLayout, weight: &Weight, operands: OperandRecipe) -> [u32; CUDA_FORMS.len()] {
    let local = cuda_int8_local(weight, operands);
    [
        u32::from(cuda_gemm_fold(&weight.axes)),
        u32::from(cuda_int8(&weight.axes, operands)),
        // The GEMV reads a records layout as records (`gemv_records_group`,
        // `packets::Records`). Its launch keeps the blocks per SM that hold
        // its grid in one wave (`projection::GemvResidency`), so it takes no
        // tuned residency.
        u32::from(rows.records()),
        u32::from(local.is_some()),
        u32::from(local == Some(true)),
        // The int8 producer reads grid codes in entry order (a lane's eight
        // codes of a row are one entry: `GridPlanes::s8_half`), every other
        // weight's in slot order; a launch's int8 weights that share rows
        // share one order (`dense.seismic`, `routed.seismic`).
        u32::from(matches!(weight.axes.coding.codes(), ValueCodes::Grid(_))),
        cuda_int8_shared(rows, &weight.axes, operands),
    ]
}

/// The shared bytes that bound the int8 GEMM's layout of one weight
/// (`projection::GemmShape::LAYOUT_BYTES` of the 128-row band:
/// `projection::LargeGemm`, 2 stages, 8 warps of one weight tile each), 0
/// where the int8 GEMM does not take the weight (`cuda_int8`). A
/// declaration admits two blocks an SM to the int8 GEMM of weights whose
/// bound fits half an SM's shared memory (`dense.seismic`, `BLOCKS`), and
/// declares it as the launch's shared bytes; the GEMM traps where its layout
/// exceeds what its launch declares. Per stage: the activation rows (128
/// rows of 64 codes at a pitch of 80 bytes, then each row's two 16-byte
/// groups) and 8 tiles' codes of one k-block (16 rows of every code plane's
/// bytes of 64 values); per record buffer (two): 128 rows' coefficient
/// records of a 256-code superblock (each coefficient plane's bytes of it in
/// whole words, the record in whole 16 bytes); per warp: its 16 rows'
/// coefficients of a k-block (`GemmShape::COEF_ROW`: a scale, with a bias
/// or local code where the weight has either, for each group of the
/// k-block, in whole 16 bytes). Each term is the GEMM's own or more.
fn cuda_int8_shared(rows: &registry::PackedRowLayout, axes: &registry::DecodeAxes, operands: OperandRecipe) -> u32 {
    if !cuda_int8(axes, operands) {
        return 0;
    }
    const STAGES: u64 = 2;
    const ROWS: u64 = 128;
    const WARPS: u64 = 8;
    let group = u64::from(rows.packet.group);
    let per = |plane: &registry::RowPlaneInfo, values: u64| (u64::from(plane.bytes_per_group) * values).div_ceil(group);
    let mut codes = 0;
    let mut words = 0;
    for plane in &rows.planes {
        match plane.content {
            registry::RowPlaneContent::Codes { .. }
            | registry::RowPlaneContent::Grids { .. }
            | registry::RowPlaneContent::GridHigh { .. } => codes += per(plane, 64),
            registry::RowPlaneContent::Groups { .. } => words += per(plane, 256).div_ceil(4),
            // A records plane holds a group's entries and its factors
            // together: bounded as codes and as coefficients both.
            registry::RowPlaneContent::Records { .. } => {
                codes += per(plane, 64);
                words += per(plane, 256).div_ceil(4);
            }
        }
    }
    let profile = axes.profile();
    let groups = 64 / u64::from(profile.local.group.min(64));
    let single = profile.local.bias == Bias::None && matches!(profile.local.scale, LocalScale::Stored(_));
    let coefficient_row = (groups * if single { 1 } else { 2 } * 4).next_multiple_of(16);
    let stage = ROWS * 96 + 8 * 16 * codes;
    let records = ROWS * (4 * words).next_multiple_of(16);
    let bytes = STAGES * stage + 2 * records + WARPS * 16 * coefficient_row;
    u32::try_from(bytes).expect("a layout of under 4 GiB")
}

/// Whether CUDA's GEMM multiplies the weight's exact codes and applies each
/// group's coefficients to its F32 partial (`fold`), rather than rounding
/// each value to the operand type first.
///
/// The fold recipe requires groups that split a 32-code half (16 or 32
/// codes) and codes its producer reads as exact operand integers (`q` of at
/// most 8 bits, which BF16 and F16 hold). Of the weights it admits, the fold
/// is chosen for grid codes and for linear codes below 8 bits under direct
/// coefficients.
fn cuda_gemm_fold(axes: &registry::DecodeAxes) -> bool {
    let profile = axes.profile();
    let admitted = matches!(profile.local.group, 16 | 32) && profile.codes.fits_i8();
    let chosen = match profile.map {
        ValueMap::Grid => true,
        ValueMap::Linear | ValueMap::Table => {
            matches!(profile.local.scale, LocalScale::Stored(_)) && axes.low != 8
        }
        // Floating codes are no operand integers of 8 bits.
        ValueMap::Float => false,
    };
    admitted && chosen
}

/// Whether CUDA's GEMM folds representation `id`'s codes
/// (`cuda_gemm_fold`, its `GEMM_FOLD` form, [`values`]); false for storage
/// other than packed rows.
pub fn cuda_gemm_fold_of(id: RepresentationId) -> bool {
    values(BackendName::Cuda, id).is_some_and(|values| values[0] == 1)
}

/// Whether CUDA's int8 GEMM (`gemm_tile_s8`, a declaration's `INT8 = 1`)
/// takes the weight: it multiplies the weight's codes as s8 against the
/// activations it quantizes per 32 codes, one m16n8k32 MMA per 32 codes
/// into an exact int32 product, and folds each product under its group's
/// coefficients.
///
/// Its arithmetic requires integer codes of `i8`; groups of 16 codes (half
/// an MMA each) or of 32 and more (whole MMAs); and activations it
/// quantizes itself, so not a weight whose products take NVFP4 operand
/// rows, which its recipe defines (`OperandRecipe`).
///
/// Its producer delivers every packed-rows weight's codes as s8
/// (`packets::Packed::s8`, `packets::Records::s8`).
fn cuda_int8(axes: &registry::DecodeAxes, operands: OperandRecipe) -> bool {
    let profile = axes.profile();
    let group = profile.local.group;
    profile.codes.fits_i8() && (group == 16 || group >= 32) && operands != OperandRecipe::Nvfp4Rows
}

/// Whether CUDA's int8 GEMM multiplies the weight's local scale codes into
/// its codes, and whether negated (`INT8_LOCAL`, `INT8_LOCAL_NEGATED`): each
/// 32-code half's exact product then folds under the upper factor alone, one
/// MMA per 32 codes whatever the local groups.
///
/// Its arithmetic requires the int8 GEMM's admission (`cuda_int8`); local
/// scales that are integer codes without a bias under the upper factor; local
/// groups of whole k16 steps, so each step's codes share one local code; and
/// every product of a code and a local code of `i8`, or every negated product.
///
/// Its producer multiplies the s8 codes of any code planes by a local code
/// (`packets::s8_times`) it reads from the scales plane
/// (`packets::Hier::staged_local`): the weights its declaration covers.
fn cuda_int8_local(weight: &Weight, operands: OperandRecipe) -> Option<bool> {
    let axes = &weight.axes;
    let profile = axes.profile();
    let LocalScale::Code(locals) = profile.local.scale else {
        return None;
    };
    let admitted = cuda_int8(axes, operands)
        && profile.integer_through_upper()
        && profile.local.bias == Bias::None
        && profile.local.group % 16 == 0
        && producers::covers(CUDA_INT8_LOCAL, weight);
    if !admitted {
        return None;
    }
    let codes = profile.codes;
    let products = [
        codes.least * locals.least,
        codes.least * locals.greatest,
        codes.greatest * locals.least,
        codes.greatest * locals.greatest,
    ];
    let least = products.into_iter().min().expect("four products");
    let greatest = products.into_iter().max().expect("four products");
    let fits = |least, greatest| Domain { least, greatest }.fits_i8();
    if fits(least, greatest) {
        Some(false)
    } else {
        fits(-greatest, -least).then_some(true)
    }
}

/// Metal's forms, in the order of [`METAL_FORMS`]: the token-packing GEMM
/// with its codes' entry and its operand's layout (`PACK`, `PACK_CENTRE`,
/// `PACK_OFFSET`, `PACK_RESIDUE`, `PACK_FOLDS`, `PACK_SUMS`, and the most
/// rows the packed-rows GEMV packs an element, `PACK_LANES`; 0 where not
/// admitted), the int8 GEMM with its operand's order (`INT8`,
/// `INT8_INTERLEAVED`) and the matrix GEMV, by one of its three products or
/// none (`MATRIX_CODES`, `MATRIX_RUNS` with its operands' type,
/// `MATRIX_SYMMETRIC`; `MATRIX` whether one serves it, which a declaration
/// offers its batched rows as the alternative to). Each admits
/// the weight on any device; whether a device runs the form (tensor
/// operations) is the library's condition. One operand serves several weights of a declaration only
/// where their operand layout (`PACK_FOLDS`, `PACK_RESIDUE`, `PACK_SUMS`) or
/// order (`INT8_INTERLEAVED`) agree.
fn metal_forms(weight: &Weight) -> [u32; METAL_FORMS.len()] {
    let packing = metal_packing(weight);
    let int8 = metal_int8(weight);
    let matrix = metal_matrix(weight);
    [
        u32::from(packing.is_some()),
        packing.map_or(0, |packing| packing.centre),
        packing.map_or(0, |packing| packing.offset),
        packing.map_or(0, |packing| u32::from(packing.residue)),
        packing.map_or(0, |packing| packing.folds),
        packing.map_or(0, |packing| u32::from(packing.sums)),
        packing.map_or(0, |packing| packing.lanes),
        u32::from(int8.is_some()),
        u32::from(int8 == Some(MetalInt8::Nibbles)),
        u32::from(matrix == Some(MetalMatrix::Codes)),
        match matrix {
            Some(MetalMatrix::Runs { halves }) => 1 + u32::from(halves),
            _ => 0,
        },
        u32::from(matrix == Some(MetalMatrix::Symmetric)),
        u32::from(matrix.is_some()),
    ]
}

/// Columns of a Metal packet, whose coefficient groups a decoder holds
/// (`packets::Direct::groups`, `packets::Hier::groups`): a `rows32` code
/// chunk.
const METAL_PACKET: u32 = registry::ROWS32_CHUNK as u32;

/// The coefficient groups of a Metal packet: two of 16 columns, else one.
fn metal_packet_groups(profile: &Profile) -> u32 {
    let group = profile.local.group;
    if group < METAL_PACKET { METAL_PACKET / group } else { 1 }
}

/// A weight's admission to Metal's token-packing GEMM (`PACK`,
/// `kernels/metal/lib/projection/packing.h`), which multiplies its codes
/// minus their centre against two tokens' packed operand and folds each
/// run's sums under the run's scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalPacking {
    /// Runs a 32-column block: the coefficient groups of a packet.
    pub folds: u32,
    /// A code enters as its value + `offset`, an unsigned integer of the
    /// entry's width, centred at `centre`, half the entry's range.
    pub centre: u32,
    pub offset: u32,
    /// Whether the packed sums take residue bits beside their 16-bit fields:
    /// the entries are values of 8 bits, whose sums 16 bits a token do not
    /// hold at a useful gain.
    pub residue: bool,
    /// Whether each of a block's two runs folds a min term of its own: the
    /// operand then carries each run's code sums beside its scales, and the
    /// weight's factors are each run's scale and min term.
    pub sums: bool,
    /// The most rows one operand element of the packed-rows GEMV carries
    /// (`packing_lanes`), which a declaration offers as `PACKED_ROWS` up to
    /// this: a lane's field pays for its run's largest code magnitude about
    /// the centre, so codes of magnitude at most 2 (2-bit, ternary and +-1
    /// entries) take 3, without residue bits (the 8-bit entries' layout).
    /// Every other weight keeps 2. Which count serves a unit is the
    /// tuner's, under admission. The token-packing GEMM packs two
    /// (`packing_gemm_lanes`).
    pub lanes: u32,
}

impl MetalPacking {
    /// The columns of one run, which share a scale: the packing bound holds
    /// one value per run (`packing_bound`).
    pub fn run(self) -> u32 {
        METAL_PACKET / self.folds
    }
}

/// Metal's token-packing admission of representation `id`
/// (`MetalPacking`), as its `PACK` forms value it ([`values`]); `None` for
/// storage other than packed rows, or which the form does not serve.
pub fn metal_packing_of(id: RepresentationId) -> Option<MetalPacking> {
    let values = values(BackendName::Metal, id)?;
    let [pack, centre, offset, residue, folds, sums, lanes, ..] = *<&[u32; METAL_FORMS.len()]>::try_from(values)
        .expect("Metal's forms are valued in the order of its names");
    (pack == 1).then_some(MetalPacking {
        folds,
        centre,
        offset,
        residue: residue == 1,
        sums: sums == 1,
        lanes,
    })
}

/// The token-packing GEMM (`packing_layout`, `packing_factors`).
///
/// Its arithmetic requires integer values whose coefficients fold per run
/// of a 32-column block: a factor per 256 columns over local codes, or
/// direct groups of whole runs in 256 columns. A block with a bias term,
/// (centre - offset) scale + bias, folds it per run, so a block of two runs
/// with one carries each run's code sums (`sums`). A power of two scaling
/// the values (E2M1's halves) is in the coefficients, which the decoder
/// folds it into.
///
/// Its producer packs a weight's lane codes (`producers::lanes`) as entries,
/// the values + `offset`, of the fewest bits that hold them centred as the
/// packing bound centres them (`packing_bound`): unsigned codes as
/// themselves, centred at half their range, and signed values at the
/// centre, a power of two holding their magnitudes. Entries below 8 bits pack
/// in runs of 32 or 16 columns (`packing_layout`), entries of 8 bits only as
/// bytes in runs of 32 (the residue bits' layout); local codes come from the
/// scales plane, which its declaration covers.
fn metal_packing(weight: &Weight) -> Option<MetalPacking> {
    let profile = weight.axes.profile();
    let lanes = producers::lanes(METAL_PACK, weight)?;
    let (bits, offset) = if profile.codes.least >= 0 {
        (
            u32::BITS - u32::try_from(profile.codes.greatest).ok()?.leading_zeros(),
            0,
        )
    } else {
        let reach = u32::try_from((-profile.codes.least).max(profile.codes.greatest + 1)).ok()?;
        let centre = reach.next_power_of_two();
        (centre.trailing_zeros() + 1, centre)
    };
    let folds = metal_packet_groups(&profile);
    let entries = (bits < 8 || folds == 1) && lanes.enter_at(offset, bits == 8);
    let centre = 1u32 << (bits - 1);
    let biased = profile.local.bias != Bias::None || centre != offset;
    let factors = match profile.upper {
        Some(upper) => upper.group == 256,
        None => 256 % profile.local.group == 0,
    };
    // The centred codes' largest magnitude: entries are values + offset,
    // centred at `centre`.
    let centred = |code: i64| (code + i64::from(offset) - i64::from(centre)).unsigned_abs();
    let magnitude = centred(profile.codes.least.into()).max(centred(profile.codes.greatest.into()));
    let residue = bits == 8;
    let sums = folds == 2 && biased;
    let lanes = if !residue && magnitude <= 2 { 3 } else { 2 };
    (factors && entries).then_some(MetalPacking {
        folds,
        centre,
        offset,
        residue,
        sums,
        lanes,
    })
}

/// The codes of Metal's int8 GEMM (`int8_codes`), by the order a lane's
/// fragment takes them in, which the quantized operand matches
/// (`int8_codes::interleaved`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalInt8 {
    /// Unsigned 4-bit codes, a byte two of them: a lane's fragment takes
    /// its even columns, then its odd ones (`INT8_INTERLEAVED`).
    Nibbles,
    /// Two's complement 8-bit codes, in column order.
    Bytes,
}

/// Metal's int8 GEMM (`int8_codes`, on tensor operations), whose operands
/// are the weight's codes as int8.
///
/// Its arithmetic requires integer codes of `i8` under groups of 32
/// columns, not scaled by a power of two of their own: under a factor per
/// 256 columns of at most 16 bits over local codes of `u8`, so a factor
/// times a local code is exact in F32, or under a stored scale without a
/// bias.
///
/// Its producer reads unsigned 4-bit codes of one plane with plain unsigned
/// local codes of the scales plane (`Nibbles`), or 8-bit two's complement
/// codes (`Bytes`).
fn metal_int8(weight: &Weight) -> Option<MetalInt8> {
    let profile = weight.axes.profile();
    let codes = profile.codes.fits_i8() && profile.code_exponent == 0 && profile.local.group == 32;
    let coefficients = match (profile.upper, profile.local.scale, profile.local.bias) {
        (Some(upper), LocalScale::Code(locals), _) => {
            let narrow = upper.factor != registry::ScaleEncoding::F32;
            upper.group == 256 && narrow && 0 <= locals.least && locals.greatest <= i32::from(u8::MAX)
        }
        (None, LocalScale::Stored(_), Bias::None) => true,
        _ => false,
    };
    let produced = if !producers::covers(METAL_INT8, weight) {
        None
    } else if profile.upper.is_some() {
        Some(MetalInt8::Nibbles)
    } else {
        Some(MetalInt8::Bytes)
    };
    produced.filter(|_| codes && coefficients)
}

/// Whether the CPU's quantized dot (`weights::Format::dot_q8`, a
/// declaration's `INT8 = 1`) takes the weight: it multiplies activations
/// quantized to Q8 blocks (`quant::Q8Block`: i8 codes under a scale per
/// block, with their sums per 16 values) by the weight's values, packed
/// rows in exact integer products a packet at a time (AVX2 pair sums, VNNI
/// or SDOT) folded under each coefficient group's scale and bias, dense
/// rows in F32.
///
/// Its arithmetic requires, of packed rows, integer codes of `i8`, so that
/// every tier's products are exact (two products of an AVX2 pair sum are at
/// most 2 · 128 · 127, inside i16; non-negative codes enter the unsigned
/// byte products as themselves, others as magnitudes against sign-adjusted
/// activations); coefficient groups of 16 values (half a
/// packet) or of whole packets; and activations it quantizes itself, so not
/// a weight whose products take NVFP4 operand rows, which its recipe
/// defines (`OperandRecipe`).
///
/// Its producer reads the codes of every packed-rows decode
/// (`weights::Codes`) and widens the floating dense elements
/// (`weights::DenseRows`).
pub fn cpu_int8_of(id: RepresentationId) -> bool {
    values(BackendName::Cpu, id).is_some_and(|values| values[0] == 1)
}

/// The CPU's form values of a representation, in [`CPU_FORMS`] order.
fn cpu_forms(info: &RepresentationInfo) -> [u32; CPU_FORMS.len()] {
    let int8 = cpu_int8(info);
    [u32::from(int8), u32::from(int8 && cpu_int8_amx(info))]
}

/// Whether the CPU's tile products over the weight (`Weights::tile_q8`, the
/// quantized dot of several activation rows) take AMX tiles where the host
/// has them (`INT8_AMX`): sixteen weight rows a tile pair, each a byte tile
/// product of a strip's signed activation codes by unsigned weight bytes,
/// each code times its unit's scale code split as `8 high + low` so that
/// every product stays a byte, giving the same integers as the VNNI tiers.
/// A row range below sixteen rows, or the tail of one, runs the VNNI tier
/// inside the same call, so a declaration offers its sixteen-row block
/// (`ROWS = 16`) only where a weight it multiplies takes this form.
///
/// Its arithmetic requires, of the quantized dot's packed rows, linear
/// integer codes of at most five bits, none negative (`0 ..= 31`, unscaled
/// by a power of two of their own), under integer local scale codes of at
/// most six bits, none negative (`0 ..= 63`, three bits a part), and a
/// minimum code, where the group has one, of the same range; and units of
/// 16 or 32 values (a coefficient group of 16, else a 32-value packet).
/// Q4_K and Q5_K take it; Q3_K's signed scales and Q8_0's byte codes do
/// not.
pub fn cpu_int8_amx_of(id: RepresentationId) -> bool {
    values(BackendName::Cpu, id).is_some_and(|values| values[1] == 1)
}

/// [`cpu_int8_amx_of`]'s rule over the representation's registry record.
fn cpu_int8_amx(info: &RepresentationInfo) -> bool {
    let (registry::RepresentationKind::PackedRows(_), Some(axes)) = (&info.kind, registry::decode_axes(info.id)) else {
        return false;
    };
    let profile = axes.profile();
    let within = |domain: Domain, greatest: i32| 0 <= domain.least && domain.greatest <= greatest;
    let scales = matches!(profile.local.scale, LocalScale::Code(scales) if within(scales, 63));
    let minimums = match profile.local.bias {
        Bias::None => true,
        Bias::Minimum { codes, .. } => within(codes, 63),
        Bias::Stored | Bias::Signed { .. } => false,
    };
    let group = profile.local.group;
    profile.map == ValueMap::Linear
        && profile.code_exponent == 0
        && within(profile.codes, 31)
        && scales
        && minimums
        && (group == 16 || group % 32 == 0)
}

/// [`cpu_int8_of`]'s rule over the representation's registry record.
fn cpu_int8(info: &RepresentationInfo) -> bool {
    if info.operands == OperandRecipe::Nvfp4Rows {
        return false;
    }
    match (&info.kind, registry::decode_axes(info.id)) {
        (registry::RepresentationKind::Dense(dtype), _) => matches!(dtype, DType::F32 | DType::BF16 | DType::F16),
        (registry::RepresentationKind::PackedRows(_), Some(axes)) => {
            let profile = axes.profile();
            let group = profile.local.group;
            profile.codes.fits_i8() && (group == 16 || group % 32 == 0)
        }
        _ => false,
    }
}

/// The products of Metal's matrix GEMV (`matrix_form`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalMatrix {
    /// Unsigned codes under hierarchical coefficients with a minimum, an
    /// unsigned local scale code per 32 columns and factors per 256, where
    /// a code times a scale code stays below 2048, an exact F16 integer of
    /// the operand's 2^-24 (`matrix_codes`).
    Codes,
    /// Codes under hierarchical coefficients without a minimum, a local
    /// scale code per 16 or 32 columns and factors per 256, a code times a
    /// scale code an exact F32 integer, or, `halves`, below 2048, an exact
    /// F16 integer (`matrix_runs`, `MATRIX_RUNS` 2).
    Runs { halves: bool },
    /// Codes of `i8` under a direct scale per 32 columns or a multiple of
    /// them, with or without a stored bias, exact F16 integers
    /// (`matrix_symmetric`): a bias folds as itself times the activation
    /// rows' block sums.
    Symmetric,
}

/// The matrix GEMV product serving a weight.
///
/// Each product's arithmetic requires integer codes under its levels
/// (`MetalMatrix`), a power of two scaling them folded into their scales
/// (E2M1's halves, `packets::Direct`). Its producer takes
/// the weight's lane codes (`producers::lanes`): unsigned codes with plain
/// unsigned local codes of the scales plane (`Codes`), any lane codes with
/// local codes of the scales plane (`Runs`), or any lane codes
/// (`Symmetric`), as its declaration states.
fn metal_matrix(weight: &Weight) -> Option<MetalMatrix> {
    let profile = weight.axes.profile();
    let factors = profile.upper.is_some_and(|upper| upper.group == 256);
    match (profile.local.group, profile.local.scale, profile.local.bias) {
        (32, LocalScale::Code(locals), Bias::Minimum { .. })
            if factors && producers::lanes(METAL_MATRIX_CODES, weight).is_some() =>
        {
            (profile.codes.greatest * locals.greatest < 2048).then_some(MetalMatrix::Codes)
        }
        (16 | 32, LocalScale::Code(locals), Bias::None)
            if factors && producers::lanes(METAL_MATRIX_RUNS, weight).is_some() =>
        {
            let magnitude = |least: i32, greatest: i32| least.abs().max(greatest.abs());
            let product = magnitude(profile.codes.least, profile.codes.greatest)
                * magnitude(locals.least, locals.greatest);
            Some(MetalMatrix::Runs {
                halves: product < 2048,
            })
        }
        (group, LocalScale::Stored(_), Bias::None | Bias::Stored)
            if group % 32 == 0
                && profile.codes.fits_i8()
                && producers::lanes(METAL_MATRIX_SYMMETRIC, weight).is_some() =>
        {
            Some(MetalMatrix::Symmetric)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checked::{
        ElementQuery, ElementSummary, NativeComparison, NativeCondition, NativeNatExpr, ParameterSummaryKind,
    };
    use crate::native_name::native_macro;

    /// CUDA's GEMM recipe of every representation it imports, as main's
    /// library derived it (`ScaledGemm`), and for q4g32, q2k, q2g256,
    /// q2g64, q1g128, iq2s, iq1s, iq1m, q3k, iq4xs and iq3s as the same
    /// choice takes them.
    #[test]
    fn cuda_folds_are_mains() {
        let folded = [
            "q4g32s",
            "iq4g32s",
            "q5g32s",
            "q5g32",
            "mxfp4g32",
            "nvfp4g16",
            "nvfp4g16a4",
            "iq2xs",
            "iq3xxs",
            "iq2xxs",
            "q4g32",
            "iq2s",
            "iq1s",
            "iq1m",
            "iq3s",
        ];
        let rounded = [
            "q4k", "q5k", "q6k", "q8g32s", "q2k", "q2g256", "q2g64", "q1g128", "q3k", "iq4xs",
        ];
        let expected = folded
            .map(|storage| (storage, true))
            .into_iter()
            .chain(rounded.map(|storage| (storage, false)));
        for (storage, fold) in expected {
            let id = registry::storage(storage, registry::Layout::Mma16).unwrap();
            assert_eq!(cuda_gemm_fold_of(id), fold, "{storage}");
        }
    }

    /// Whether packed-rows storage `id`'s value codes are all integers of
    /// `i8`, which an int8 form multiplies (mxfp8g32's E4M3 codes are not).
    fn codes_fit_i8(id: crate::ids::RepresentationId) -> bool {
        registry::decode_axes(id).is_some_and(|axes| axes.profile().codes.fits_i8())
    }

    /// CUDA's int8 GEMM admission of every representation it imports: main's
    /// library took the INT8 path for every packed weight whose products do
    /// not take NVFP4 operand rows (`projection::quantizable`), and so does
    /// the admission, of every packed-rows representation of the registry
    /// whose codes fit `i8` (every one main had).
    #[test]
    fn cuda_int8_is_mains() {
        let admitted = [
            "q4k", "q5k", "q6k", "q8g32s", "q4g32s", "iq4g32s", "q5g32s", "q5g32", "mxfp4g32",
            "nvfp4g16", "iq2xs", "iq3xxs", "iq2xxs", "q4g32", "q2k", "q2g256", "q2g64", "q1g128", "iq2s",
            "iq1s", "iq1m", "q3k", "iq4xs", "iq3s",
        ];
        let excluded = ["nvfp4g16a4", "mxfp8g32"];
        let int8 = Form::named(BackendName::Cuda, "INT8").unwrap();
        let expected = admitted.map(|storage| (storage, 1)).into_iter().chain(excluded.map(|storage| (storage, 0)));
        for (storage, value) in expected {
            let id = registry::storage(storage, registry::Layout::Mma16).unwrap();
            assert_eq!(int8.of(id), value, "{storage}");
        }
        for info in registry::representations() {
            let main = match info.kind {
                registry::RepresentationKind::PackedRows(_) => {
                    u32::from(info.operands != OperandRecipe::Nvfp4Rows && codes_fit_i8(info.id))
                }
                _ => 0,
            };
            assert_eq!(int8.of(info.id), main, "{}", info.name);
        }
    }

    /// CUDA's int8 GEMM multiplies local scale codes into the codes of q3k's
    /// packed rows alone, negated: its codes and local codes ([-4, 3] and
    /// [-32, 31]) have products in [-124, 128], negated within `i8`; no other
    /// representation's products, or its bias, allow it.
    #[test]
    fn cuda_int8_local_is_q3k_negated() {
        let local = Form::named(BackendName::Cuda, "INT8_LOCAL").unwrap();
        let negated = Form::named(BackendName::Cuda, "INT8_LOCAL_NEGATED").unwrap();
        for info in registry::representations() {
            let packed = matches!(info.kind, registry::RepresentationKind::PackedRows(_));
            let expected = u32::from(packed && info.representation == "q3k");
            assert_eq!((local.of(info.id), negated.of(info.id)), (expected, expected), "{}", info.name);
        }
    }

    /// The CPU's quantized dot admission of every representation: main's
    /// projections quantized their activations for every weight with a CPU
    /// component, the floating dense elements and every packed-rows decode,
    /// except one whose products take NVFP4 operand rows, whose quantized
    /// rows they dropped for the F32 dot (`projection::project_arithmetic`),
    /// and one whose codes do not fit `i8` (none main had), which takes the
    /// F32 dot too.
    #[test]
    fn cpu_int8_is_mains() {
        let admitted = [
            "f32",
            "bf16",
            "f16",
            "q4k@rows16",
            "q5k@rows16",
            "q6k@rows16",
            "q8g32s@rows16",
            "q4g32s@rows16",
            "iq4g32s@rows16",
            "q5g32s@rows16",
            "q5g32@rows16",
            "mxfp4g32@rows16",
            "nvfp4g16@rows16",
            "iq2xs@rows16",
            "iq3xxs@rows16",
            "iq2xxs@rows16",
            "q4g32@rows16",
            "q2k@rows16",
            "q2g256@rows16",
            "q2g64@rows16",
            "q1g128@rows16",
            "iq2s@rows16",
            "iq1s@rows16",
            "iq1m@rows16",
            "q3k@rows16",
            "iq4xs@rows16",
            "iq3s@rows16",
            "q4k@rows8",
        ];
        let excluded = ["nvfp4g16a4@rows16", "mxfp8g32@rows16"];
        let int8 = Form::named(BackendName::Cpu, "INT8").unwrap();
        let expected = admitted.map(|name| (name, 1)).into_iter().chain(excluded.map(|name| (name, 0)));
        for (name, value) in expected {
            let id = registry::representation(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(int8.of(id), value, "{name}");
        }
        for info in registry::representations() {
            let main = match info.kind {
                registry::RepresentationKind::Dense(dtype) => matches!(dtype, DType::F32 | DType::BF16 | DType::F16),
                registry::RepresentationKind::PackedRows(_) => {
                    info.operands != OperandRecipe::Nvfp4Rows && codes_fit_i8(info.id)
                }
                _ => false,
            };
            assert_eq!(int8.of(info.id), u32::from(main), "{}", info.name);
        }
    }

    /// The AMX tile products take the K-quants of unsigned codes under
    /// unsigned six-bit scales, and nothing the quantized dot does not.
    #[test]
    fn cpu_int8_amx_takes_unsigned_k_quants() {
        let amx = Form::named(BackendName::Cpu, "INT8_AMX").unwrap();
        for (name, value) in [
            ("q4k@rows16", 1),
            ("q5k@rows16", 1),
            ("q4k@rows8", 1),
            ("q3k@rows16", 0),
            ("q6k@rows16", 0),
            ("q8g32s@rows16", 0),
            ("q4g32s@rows16", 0),
            ("f16", 0),
        ] {
            let id = registry::representation(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(amx.of(id), value, "{name}");
        }
        let int8 = Form::named(BackendName::Cpu, "INT8").unwrap();
        for info in registry::representations() {
            assert!(amx.of(info.id) <= int8.of(info.id), "{}", info.name);
        }
    }

    /// The host's queries of a form read the table the forms are decided in
    /// once (`values`), and take what its rule decides of every
    /// representation of the registry.
    #[test]
    fn form_queries_are_their_rules() {
        for info in registry::representations() {
            let packed = matches!(info.kind, registry::RepresentationKind::PackedRows(_));
            let axes = registry::decode_axes(info.id);
            assert_eq!(metal_packing_of(info.id), Weight::of(info).as_ref().and_then(metal_packing), "{}", info.name);
            assert_eq!(
                cuda_gemm_fold_of(info.id),
                packed && axes.as_ref().is_some_and(cuda_gemm_fold),
                "{}",
                info.name
            );
            assert_eq!(cpu_int8_of(info.id), cpu_int8(info), "{}", info.name);
            assert_eq!(cpu_int8_amx_of(info.id), cpu_int8(info) && cpu_int8_amx(info), "{}", info.name);
        }
    }

    /// Every backend's forms are its names, valued on Metal and CUDA for
    /// packed rows only and on the CPU and Vulkan for every representation.
    #[test]
    fn forms_are_named_by_their_backend() {
        for backend in BackendName::ALL {
            for info in registry::representations() {
                let forms = forms(backend, info.id);
                let valued = matches!(backend, BackendName::Cpu | BackendName::Vulkan)
                    || matches!(info.kind, registry::RepresentationKind::PackedRows(_));
                let named = forms.iter().map(|(name, _)| *name).collect::<Vec<_>>();
                assert_eq!(named, if valued { names(backend).to_vec() } else { Vec::new() }, "{}", info.name);
                for (name, value) in forms {
                    assert_eq!(Form::named(backend, name).unwrap().of(info.id), value, "{} {name}", info.name);
                }
            }
            assert_eq!(Form::named(backend, "INT9"), None);
        }
    }

    /// The operand layout and order Metal's library takes for every
    /// representation of the registry, as main's library read them from an
    /// admitted weight's decoder: the token-packing operand's runs a block,
    /// the coefficient groups of a packet (`packing_codes::folds`,
    /// `Packet::groups`), and whether the int8 operand is interleaved, its
    /// codes unsigned nibbles (`int8_codes::interleaved`); 0 for weights not
    /// admitted.
    #[test]
    fn metal_operand_layouts_are_mains() {
        let name = |form| Form::named(BackendName::Metal, form).unwrap();
        let (pack, folds, int8, interleaved) = (name("PACK"), name("PACK_FOLDS"), name("INT8"), name("INT8_INTERLEAVED"));
        for info in registry::representations() {
            let groups = registry::decode_axes(info.id).map_or(0, |axes| metal_packet_groups(&axes.profile()));
            let nibbles = registry::decode_axes(info.id)
                .is_some_and(|axes| {
                    axes.coding.codes() == registry::ValueCodes::Fields(registry::CodeInterpretation::Unsigned)
                        && (axes.low, axes.high) == (4, 0)
                });
            assert_eq!(folds.of(info.id), pack.of(info.id) * groups, "{}", info.name);
            assert_eq!(interleaved.of(info.id), int8.of(info.id) * u32::from(nibbles), "{}", info.name);
        }
        let expected = [
            ("q4k", 1, 1),
            ("q6k", 2, 0),
            ("q8g32s", 1, 0),
            ("q4g32s", 1, 0),
            ("mxfp4g32", 1, 0),
            ("nvfp4g16", 2, 0),
        ];
        for (storage, runs, nibbles) in expected {
            let id = registry::storage(storage, registry::Layout::Rows32).unwrap();
            assert_eq!((folds.of(id), interleaved.of(id)), (runs, nibbles), "{storage}");
        }
    }

    /// Metal's admissions of every representation it imports: PACK (folds,
    /// centre, offset, residue, sums, lanes), INT8 and the matrix GEMV product, as
    /// `metal_packing`, `metal_int8` and `metal_matrix` admit each over the
    /// lane codes `producers::lanes` delivers (every code-plane split up to 8
    /// bits, one-bit odd codes, codebook and grid of `i8` values).
    #[test]
    fn metal_admissions_are_pinned() {
        let pack = |folds, centre, offset, residue, lanes| {
            Some(MetalPacking {
                folds,
                centre,
                offset,
                residue,
                sums: false,
                lanes,
            })
        };
        let expected = [
            ("q4k", pack(1, 8, 0, false, 2), Some(MetalInt8::Nibbles), Some(MetalMatrix::Codes)),
            ("q5k", pack(1, 16, 0, false, 2), None, Some(MetalMatrix::Codes)),
            (
                "q6k",
                pack(2, 32, 32, false, 2),
                None,
                Some(MetalMatrix::Runs { halves: false }),
            ),
            (
                "q8g32s",
                pack(1, 128, 128, true, 2),
                Some(MetalInt8::Bytes),
                Some(MetalMatrix::Symmetric),
            ),
            (
                "q4g32s",
                pack(1, 8, 8, false, 2),
                None,
                Some(MetalMatrix::Symmetric),
            ),
            (
                "iq4g32s",
                pack(1, 128, 128, true, 2),
                None,
                Some(MetalMatrix::Symmetric),
            ),
            (
                "q5g32s",
                pack(1, 16, 16, false, 2),
                None,
                Some(MetalMatrix::Symmetric),
            ),
            ("q5g32", pack(1, 16, 0, false, 2), None, Some(MetalMatrix::Symmetric)),
            ("mxfp4g32", pack(1, 16, 16, false, 2), None, Some(MetalMatrix::Symmetric)),
            ("nvfp4g16", pack(2, 16, 16, false, 2), None, None),
            ("nvfp4g16a4", pack(2, 16, 16, false, 2), None, None),
            (
                "iq2xs",
                pack(2, 64, 64, false, 2),
                None,
                Some(MetalMatrix::Runs { halves: true }),
            ),
            (
                "iq3xxs",
                pack(1, 64, 64, false, 2),
                None,
                Some(MetalMatrix::Runs { halves: true }),
            ),
            ("iq2xxs", pack(1, 64, 64, false, 2), None, None),
            ("q4g32", pack(1, 8, 0, false, 2), None, Some(MetalMatrix::Symmetric)),
            ("q2k", Some(MetalPacking { folds: 2, centre: 2, offset: 0, residue: false, sums: true, lanes: 3 }), None, None),
            (
                "q2g256",
                pack(1, 4, 4, false, 3),
                None,
                Some(MetalMatrix::Symmetric),
            ),
            (
                "q2g64",
                pack(1, 4, 4, false, 3),
                None,
                Some(MetalMatrix::Symmetric),
            ),
            ("q1g128", pack(1, 2, 2, false, 3), None, Some(MetalMatrix::Symmetric)),
            (
                "iq2s",
                pack(2, 64, 64, false, 2),
                None,
                Some(MetalMatrix::Runs { halves: true }),
            ),
            ("iq1s", pack(1, 2, 2, false, 3), None, None),
            (
                "iq1m",
                pack(2, 16, 16, false, 2),
                None,
                Some(MetalMatrix::Runs { halves: true }),
            ),
            (
                "q3k",
                pack(2, 4, 4, false, 2),
                None,
                Some(MetalMatrix::Runs { halves: true }),
            ),
            (
                "iq4xs",
                pack(1, 128, 128, true, 2),
                None,
                Some(MetalMatrix::Runs { halves: false }),
            ),
            (
                "iq3s",
                pack(1, 16, 16, false, 2),
                None,
                Some(MetalMatrix::Runs { halves: true }),
            ),
        ];
        for (storage, packing, int8, matrix) in expected {
            let id = registry::storage(storage, registry::Layout::Rows32).unwrap();
            let weight = Weight::of(registry::representation_info(id)).unwrap();
            assert_eq!(
                (metal_packing(&weight), metal_int8(&weight), metal_matrix(&weight)),
                (packing, int8, matrix),
                "{storage}"
            );
        }
    }

    /// A grid weight's code planes are its entries' whole bytes (the low
    /// plane) and the rest (the high plane), in the bits the native ABI
    /// renders (`CODE_LOW_BITS`, `CODE_HIGH_BITS`, the decode axes' own) and
    /// the producers read (`AxisKey`): the units of Vulkan's grid patterns.
    #[test]
    fn grid_planes_are_an_entrys_whole_bytes_and_the_rest() {
        let mut grids = 0;
        for info in registry::representations() {
            let Some(axes) = registry::decode_axes(info.id) else { continue };
            let registry::ValueCodes::Grid(grid) = axes.coding.codes() else { continue };
            grids += 1;
            let bits = grid.entry_bits();
            assert_eq!((axes.low, axes.high), (bits / 8 * 8, bits % 8), "{}", info.name);
            let key = axes.axis_key();
            assert_eq!((key.low, key.high), (axes.low, axes.high), "{}", info.name);
        }
        assert!(grids > 0, "the registry has grid weights");
    }

    /// Vulkan's decoder reads every packed-rows weight of its resident layout
    /// (`rows16`), within its producers' requirements (`DECODED`, which every
    /// Vulkan native that binds a packed weight declares, `dense.seismic`).
    #[test]
    fn vulkan_decodes_every_rows16_weight() {
        let undecoded: Vec<&str> = registry::representations()
            .iter()
            .filter(|info| {
                matches!(&info.kind, registry::RepresentationKind::PackedRows(rows) if rows.layout == registry::Layout::Rows16)
            })
            .filter(|info| !vulkan_decoded(info))
            .map(|info| info.name)
            .collect();
        assert!(undecoded.is_empty(), "rows16 weights Vulkan's producers do not decode (`vulkan/producers.decl`): {undecoded:?}");
    }

    /// The parameter prefix a native source line binds a packet decoder
    /// slot to (`#define KERNEL_W<n> SEISMIC_<PARAMETER>`, `packets.glsl`).
    fn slot_binding(line: &str) -> Option<&str> {
        let slot = line.trim_start().strip_prefix("#define KERNEL_W")?;
        let bound = slot.strip_prefix(|digit: char| digit.is_ascii_digit())?;
        bound.strip_prefix(char::is_whitespace)?.split_whitespace().next()
    }

    /// Whether `conjuncts` (a native's `where`, its projections' included)
    /// hold only where `form(element, DECODED) == 1`: one of them is that
    /// comparison, or some are it under guards (`GUARD or form(..) == 1`, a
    /// gated projection's) that never all hold. The guards are decided over
    /// the one dimension they read, at every value up to past their greatest
    /// constant, which is exact for comparisons of it with constants; guards
    /// of any other shape prove nothing.
    fn admits_only_decoded(conjuncts: &[&NativeCondition], element: &str) -> bool {
        let admission = |condition: &NativeCondition| {
            let NativeCondition::Compare { comparison: NativeComparison::Eq, left, right } = condition else {
                return false;
            };
            matches!(
                (left, right),
                (NativeNatExpr::Element(read, ElementQuery::Form(form)), NativeNatExpr::Constant(1))
                    | (NativeNatExpr::Constant(1), NativeNatExpr::Element(read, ElementQuery::Form(form)))
                    if read == element && *form == VULKAN_DECODED
            )
        };
        let mut guards: Vec<&NativeCondition> = Vec::new();
        for &conjunct in conjuncts {
            if admission(conjunct) {
                return true;
            }
            if let NativeCondition::Or(left, right) = conjunct {
                if admission(&**right) {
                    guards.push(&**left);
                } else if admission(&**left) {
                    guards.push(&**right);
                }
            }
        }
        let mut names = Vec::new();
        let mut constants = std::collections::BTreeSet::new();
        for guard in &guards {
            guard.dimensions(&mut names);
            guard.constants(&mut constants);
        }
        names.sort();
        names.dedup();
        let [name] = names.as_slice() else {
            return false;
        };
        let past = constants.last().map_or(0, |greatest| greatest + 1);
        (0..=past).all(|value| {
            guards.iter().any(|guard| {
                guard.decided(&|dimension: &str| (dimension == name.as_str()).then_some(value), &|_: &str| None)
                    == Some(false)
            })
        })
    }

    /// Every weight a Vulkan native binds to a packet decoder slot is one
    /// its declaration admits only where `DECODED` holds, through its own
    /// `where` or its projection's, so `packets.glsl`'s `#error`s (a slot
    /// its producers do not decode, or of neither dense nor packed rows) are
    /// invariant violations no admitted configuration reaches.
    #[test]
    fn vulkan_natives_bind_only_decoded_weights() {
        let module = crate::analysis::tests::kernels();
        let mut bound = 0;
        let mut unadmitted = Vec::new();
        for entry in module.entries() {
            let Some(native) = module.native_implementation(entry.id, BackendName::Vulkan) else { continue };
            let asset = module.native_asset(entry.id, BackendName::Vulkan).expect("a Vulkan native has a source");
            let mut conjuncts = Vec::new();
            if let Some(constraint) = &native.constraint {
                constraint.conjuncts(&mut conjuncts);
            }
            let slots: std::collections::BTreeSet<&str> = asset.text.lines().filter_map(slot_binding).collect();
            for slot in slots {
                let parameter = entry
                    .parameters
                    .iter()
                    .find(|parameter| format!("SEISMIC_{}", native_macro(&parameter.name)) == slot)
                    .unwrap_or_else(|| panic!("`{}` binds a weight slot to `{slot}`, no parameter of its entry", entry.name));
                let ParameterSummaryKind::Tensor { element, .. } = &parameter.kind else {
                    panic!("`{}` binds a weight slot to `{}`, which is no tensor", entry.name, parameter.name);
                };
                bound += 1;
                let admitted = match element {
                    ElementSummary::Fixed(name) => {
                        registry::representation(name).is_some_and(|id| VULKAN_DECODED.of(id) == 1)
                    }
                    ElementSummary::Parameter(element) => admits_only_decoded(&conjuncts, element),
                };
                if !admitted {
                    unadmitted.push(format!("{} ({}): {}", entry.name, native.declared_in, parameter.name));
                }
            }
        }
        assert!(bound > 0, "some Vulkan native binds a weight");
        assert!(
            unadmitted.is_empty(),
            "Vulkan natives bind weights they do not admit by `form(W, DECODED) == 1`: {unadmitted:#?}"
        );
    }

    /// A decoding launch's region (`dense.seismic`) holds its weights' tables
    /// where every Vulkan device holds the region with them, marked by an odd
    /// u32 length, and only there: whatever it holds, its group memory (the
    /// 16-byte views' largest, as the runtime reserves it) stays within the
    /// 32 KiB floor wherever its own region does. A pair of iq1_s weights
    /// (`dense_expand`, 32 KiB of tables) forms with every region within the
    /// floor and its tables read in place; one iq1_s weight (`dense_output`,
    /// 16 KiB) is held by the GEMM's region and by the GEMV's, which stages
    /// fewer packets at once to hold it; iq2_xxs's 2 KiB grid by both.
    #[test]
    fn vulkan_tables_are_held_where_every_device_holds_them() {
        const DECODING: [&str; 11] = [
            "gemv1", "gemv2", "gemv4", "gemv8", "gemm", "ordered_gemv", "ordered_gemv1", "ordered_gemv2",
            "ordered_gemv4", "ordered_gemv8", "ordered_gemm",
        ];
        let floor = crate::device::VULKAN_FLOOR_SHARED_BYTES;
        let module = crate::analysis::tests::kernels();
        let id = |name: &str| registry::representation(name).expect("a registered representation");
        // Each decoding launch's region at one and at 512 rows: its bytes.
        let regions = |entry: &str, bindings: crate::entry::ElementBindings, statics: &[(&str, u64)]| {
            let entry = module.entries().iter().find(|candidate| candidate.name == entry).expect("the entry");
            let native = module.native_implementation(entry.id, BackendName::Vulkan).expect("a Vulkan native");
            let bound = native.at_elements(&bindings).expect("the weights form");
            let specialization = statics
                .iter()
                .fold(crate::checked::NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value));
            let configuration = bound
                .default_specialization_without_device(&specialization)
                .expect("the declaration's defaults read no device")
                .expect("the statics are admissible");
            let mut regions = Vec::new();
            for rows in [1u64, 512] {
                let dimension = |name: &str| match name {
                    "M" | "O" => Some(rows),
                    _ => configuration.static_value(name),
                };
                for (ordinal, launch) in bound.launches.iter().enumerate() {
                    let decoding = launch
                        .kernel
                        .strip_prefix(entry.name.as_str())
                        .and_then(|rest| rest.strip_prefix('_'))
                        .is_some_and(|suffix| DECODING.contains(&suffix));
                    if !decoding {
                        continue;
                    }
                    let parameter = |name: &str| configuration.launch_param(ordinal, name).or(configuration.param(name));
                    let bytes = launch.shared_bytes.evaluate(&dimension, &parameter).expect("the region evaluates");
                    regions.push((format!("{} M {rows}", launch.kernel), bytes));
                }
            }
            regions
        };
        let iq1s = id("iq1s@rows16");
        let output = |weight| {
            regions(
                "dense_output",
                crate::entry::ElementBindings::new().bind("A", id("bf16")).bind("DW", weight),
                &[("H", 4096), ("F", 12288), ("DS", 0)],
            )
        };
        let expand = regions(
            "dense_expand",
            crate::entry::ElementBindings::new()
                .bind("A", id("bf16"))
                .bind("NW", id("f32"))
                .bind("GW", iq1s)
                .bind("UW", iq1s),
            &[("H", 4096), ("F", 12288), ("GS", 0), ("US", 0), ("LB", 0)],
        );
        let held = |regions: &[(String, u64)], launch: &str| {
            regions.iter().any(|(kernel, bytes)| kernel.contains(launch) && (bytes / 4) % 2 == 1)
        };
        for (regions, gemv, gemm) in [
            (expand, false, false),
            (output(iq1s), true, true),
            (output(id("iq2xxs@rows16")), true, true),
        ] {
            assert!(!regions.is_empty(), "the projection decodes");
            for (launch, bytes) in &regions {
                assert!(16 * bytes.div_ceil(16) <= floor, "{launch}: {bytes} bytes past the {floor}-byte floor");
                assert_eq!(bytes % 4, 0, "{launch}: a whole u32 length");
            }
            assert_eq!([held(&regions, "gemv1 "), held(&regions, "gemm ")], [gemv, gemm], "{regions:?}");
        }
    }

    /// A native whose launches stage their weights' tables at the end of
    /// their shared regions (its source defines `PACKETS_STAGED_TABLES` 1,
    /// `vulkan/packets.glsl`) has every decoding launch of its projection
    /// (its GEMVs and GEMMs) read `form(W, TABLE)` of every element a slot
    /// of its binds in its `shared_bytes`: the library stages each distinct
    /// table once at the region's end.
    #[test]
    fn vulkan_staged_tables_are_in_their_launches_shared_regions() {
        const DECODING: [&str; 11] = [
            "gemv1", "gemv2", "gemv4", "gemv8", "gemm", "ordered_gemv", "ordered_gemv1", "ordered_gemv2",
            "ordered_gemv4", "ordered_gemv8", "ordered_gemm",
        ];
        fn reads(expression: &NativeNatExpr, element: &str, form: Form) -> bool {
            match expression {
                NativeNatExpr::Element(read, ElementQuery::Form(read_form)) => read == element && *read_form == form,
                NativeNatExpr::Add(a, b)
                | NativeNatExpr::Sub(a, b)
                | NativeNatExpr::Mul(a, b)
                | NativeNatExpr::Div(a, b)
                | NativeNatExpr::Rem(a, b)
                | NativeNatExpr::CeilDiv(a, b)
                | NativeNatExpr::Min(a, b)
                | NativeNatExpr::Max(a, b) => reads(a, element, form) || reads(b, element, form),
                _ => false,
            }
        }
        let forms = [Form::named(BackendName::Vulkan, "TABLE").unwrap()];
        let module = crate::analysis::tests::kernels();
        let mut staging = 0;
        let mut unaccounted = Vec::new();
        for entry in module.entries() {
            let Some(native) = module.native_implementation(entry.id, BackendName::Vulkan) else { continue };
            let asset = module.native_asset(entry.id, BackendName::Vulkan).expect("a Vulkan native has a source");
            if !asset.text.lines().any(|line| line.trim() == "#define PACKETS_STAGED_TABLES 1") {
                continue;
            }
            staging += 1;
            let elements: std::collections::BTreeSet<&str> = asset
                .text
                .lines()
                .filter_map(slot_binding)
                .filter_map(|slot| {
                    let parameter = entry
                        .parameters
                        .iter()
                        .find(|parameter| format!("SEISMIC_{}", native_macro(&parameter.name)) == slot)?;
                    match &parameter.kind {
                        ParameterSummaryKind::Tensor { element: ElementSummary::Parameter(element), .. } => {
                            Some(element.as_str())
                        }
                        _ => None,
                    }
                })
                .collect();
            for launch in &native.launches {
                let Some(suffix) = launch.kernel.strip_prefix(entry.name.as_str()).and_then(|rest| rest.strip_prefix('_'))
                else {
                    continue;
                };
                if !DECODING.contains(&suffix) {
                    continue;
                }
                for element in &elements {
                    for form in &forms {
                        if !reads(&launch.shared_bytes, element, *form) {
                            unaccounted.push(format!("{} {}: {element} {}", entry.name, launch.kernel, form.name()));
                        }
                    }
                }
            }
        }
        assert!(staging > 0, "some Vulkan native stages its tables");
        assert!(unaccounted.is_empty(), "decoding launches whose shared regions omit a slot's table: {unaccounted:#?}");
    }
}
