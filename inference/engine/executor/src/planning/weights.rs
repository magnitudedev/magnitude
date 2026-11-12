use super::{
    ArtifactComponent, ArtifactComponentKind, ComponentSelection, FeedForwardProgramSlot,
    ProgramPlan,
};
use crate::error::PlanError;
use crate::memory::ClaimDemand;
use crate::streamed_table::STREAMED_TABLE_ROWS;
use crate::{ExecutionPath, operators};
use magnitude_artifacts::{
    ArtifactIdentity, PackageHeaders, PackageIdentity, PackageManifest,
    gguf::{Encoding, TensorDescriptor},
};
use magnitude_family_contracts::{
    ActivationDType, BlockElement, BlockScale, HeadComponent, ImportTransform, ModelDefinition,
    ProgressiveCodeDepth, ProgressivePlane, RowRounding, ScaleCodes, ScaledActivations, VisionDescription,
    WeightDescriptor, WeightKind, WeightRole, WeightScope,
};
use magnitude_state::KvCodec;
use crate::operators::{code, scalar};
use crate::programs::graph::GraphError;
use crate::programs::graph::facts::{Numbers, constant};
use seismic::{Argument, BackendName, DType, Dim, Element, GgufScales, Layout, MetalPacking, OperandRecipe};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightPlan {
    pub role: WeightRole,
    pub component: ArtifactComponent,
    /// The stored representation of the artifact tensor.
    pub source: Element,
    /// The representation the importer uploads: `source`, or F32 for a
    /// packed weight the host dequantizes exactly (`import_transforms::
    /// dequantize`) because its kernels read dense weights only.
    pub upload: Element,
    pub resident: Element,
    pub shape: Vec<u64>,
    pub descriptor: WeightDescriptor,
    /// Startup upload charge: the logical shape in `upload`.
    pub source_bytes: u64,
    /// The resident representation's bytes, without `scale`.
    pub resident_bytes: u64,
    /// The weight's resident second-level scale (`ImportTransform::
    /// ScaleByTensor`), when it has one.
    pub scale: Option<WeightScalePlan>,
    /// The extent of the weight's resident packing bound (`packing_bound`):
    /// one F32 value per run of its columns, or 0 without one.
    pub bound: u64,
    /// How the load holds the weight's values.
    pub holding: Holding,
}

/// How a load holds a planned weight's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holding {
    /// Every value, imported at load.
    Whole,
    /// A gathered table streamed by rows (`device_table`, `stored_table`):
    /// `shape` is its slots', which hold only some of the table's rows.
    Streamed(StreamedRows),
    /// A routed expert stack streamed through the load's expert bank
    /// (`ExpertBankPlan`), bank group `group`: `shape` is the group's
    /// `[records, N, K]`, which every member layer's entries address, and
    /// `resident_bytes` charges the layer's own records. Its values are the
    /// bank's (empty, or imported whole into its records when it holds
    /// every expert), and it holds its `scale` per expert.
    Banked { group: usize },
}

/// A load's weights as their files store them, split by where it holds them
/// as it runs, so the four sum to the model's stored weights: `held` in the
/// serving device's memory, `host` the routed experts only its host tier
/// holds in host RAM, `streamed` the routed experts neither holds, read from
/// the model file as steps miss them, and `tables` the rows of its tables read
/// by row ([`ModelLoadPlan::unheld_table_bytes`]) that no slot holds, which
/// steps look up a few rows at a time wherever the OS keeps the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightPlacement {
    pub held: u64,
    pub host: u64,
    pub streamed: u64,
    pub tables: u64,
}

/// The rows of a streamed gathered table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamedRows {
    /// The rows of the stored table.
    pub table_rows: u64,
    /// The rows one step may bring: the rows of the device staging its
    /// tiles pass through, as stored. 0 of a table whose rows the host
    /// writes into their slots (`TableFill::Stored`).
    pub staging_rows: u64,
    /// The landing slots, the last rows of `shape`'s slots: where a read of
    /// rows the device selects finds each row it names that the other slots
    /// do not hold, landed by the host as stored (`landing_bytes`) and
    /// converted ahead of the read. The most rows one such read of the load
    /// names (`selected_read_rows`); 0 of a load without
    /// one, and of a stored table, whose filler writes every row a read
    /// names into its slots.
    pub landing_rows: u64,
    /// How readers find a row's slot.
    pub directory: DirectoryForm,
    pub fill: TableFill,
}

/// The launches a streamed table's stagings serve: the lanes whose launches
/// may name its rows, at the load's depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableFlights {
    pub flights: super::FlightDepth,
    pub lanes: usize,
}

impl TableFlights {
    /// A launch brings a table's absent tiles at most twice: as the host
    /// builds it (`ResidentTable::indices`), and as the host observes its
    /// completion, landing its selected read's tiles in slots of their own
    /// (`SubmittedRead::complete`).
    pub const BRINGS: usize = 2;

    /// The stagings a held class of `places` places has in flight
    /// ([`super::FlightDepth::capacity`]).
    pub fn stagings(self, places: usize) -> usize {
        self.flights.capacity(places, self.lanes, Self::BRINGS)
    }
}

/// How the readers of a streamed table find a row's slot (`gathered_row`):
/// a device directory of `[entries, fields]` I32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectoryForm {
    /// Every table row's slot, or -1: 4 bytes per table row.
    Dense,
    /// (row, slot) buckets, linearly probed, a power of two at least twice
    /// the slots: 8 bytes per bucket, whatever the table's rows.
    Hashed,
}

impl DirectoryForm {
    /// The form of a table of `table_rows` rows in `slots` slots: dense
    /// while the table has no more than 16 times its slots of rows (where a
    /// dense directory costs at most what a hashed one of its slots does,
    /// and is read in one access), else hashed.
    pub fn of(table_rows: u64, slots: u64) -> Self {
        match table_rows <= slots.saturating_mul(16) {
            true => Self::Dense,
            false => Self::Hashed,
        }
    }

    /// Entries (`R`) of the directory of a table of `table_rows` rows in
    /// `slots` slots.
    pub fn entries(self, table_rows: u64, slots: u64) -> u64 {
        match self {
            Self::Dense => table_rows,
            Self::Hashed => slots.saturating_mul(2).max(1).next_power_of_two(),
        }
    }

    /// I32 fields (`F`) of an entry: a dense one's slot, a hashed one's row
    /// and slot.
    pub fn fields(self) -> u64 {
        match self {
            Self::Dense => 1,
            Self::Hashed => 2,
        }
    }
}

/// How a streamed table's rows reach their slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableFill {
    /// In the table's resident representation, converted by the weight's
    /// import entry from staging, queued ahead of the step that names them;
    /// the stagings serve the load's flights (`TableFlights`).
    Imported(TableFlights),
    /// In the table's resident representation, read from the artifact as
    /// stored and converted row by row by the host (the CPU backend's
    /// registered row conversion), which writes them into slots the device
    /// reads in place (`stored_rows`). A step reading them waits at a gate
    /// until the host has written them.
    Stored,
}

/// The bytes of the first `leading` of `rows` leading rows of a tensor of
/// `bytes`, which lie first: in proportion. None where there are fewer
/// rows, or the bytes do not divide evenly into them. The one rule of every
/// prefix of an expert stack, planned ([`WeightPlan::leading`]) and stored.
pub(crate) fn leading_bytes(bytes: u64, rows: u64, leading: u64) -> Option<u64> {
    (leading <= rows && bytes % rows == 0).then(|| bytes / rows * leading)
}

impl WeightPlan {
    /// The plan of this expert stack's first `experts` experts: its shape,
    /// its source and resident bytes, and its second-level scale, of that
    /// many leading rows (one scale per expert, the first `experts` of its
    /// stored values; a single stored one serves any number). None where it
    /// has fewer experts, or its bytes or scales do not divide into its
    /// rows.
    pub fn leading(&self, experts: u64) -> Option<Self> {
        let rows = *self.shape.first()?;
        (experts <= rows).then_some(())?;
        let part = |bytes: u64| leading_bytes(bytes, rows, experts);
        let scale = match &self.scale {
            None => None,
            Some(scale) if scale.stored == 1 => Some(WeightScalePlan {
                extent: experts,
                resident_bytes: scale.resident_bytes / scale.extent * experts,
                ..scale.clone()
            }),
            Some(scale) if scale.extent == rows => Some(WeightScalePlan {
                extent: experts,
                resident_bytes: part(scale.resident_bytes)?,
                ..scale.clone()
            }),
            Some(_) => return None,
        };
        Some(Self {
            shape: [&[experts][..], &self.shape[1..]].concat(),
            source_bytes: part(self.source_bytes)?,
            resident_bytes: part(self.resident_bytes)?,
            scale,
            ..self.clone()
        })
    }

    /// The rows of a gathered table that is streamed; `None` for a weight
    /// held otherwise.
    pub fn streamed(&self) -> Option<StreamedRows> {
        match self.holding {
            Holding::Streamed(streamed) => Some(streamed),
            Holding::Whole | Holding::Banked { .. } => None,
        }
    }

    /// The bank group of a routed expert stack the load streams.
    pub fn banked(&self) -> Option<usize> {
        match self.holding {
            Holding::Banked { group } => Some(group),
            Holding::Whole | Holding::Streamed(_) => None,
        }
    }

    /// Whether the importer uploads host-prepared bytes (transformed or
    /// dequantized) rather than the stored range as is. A second-level
    /// scale leaves the weight's bytes as stored.
    pub fn host_prepared(&self) -> bool {
        self.descriptor
            .transforms
            .iter()
            .any(|transform| !matches!(transform, ImportTransform::ScaleByTensor { .. }))
            || self.upload != self.source
    }

    /// Whether the host places the weight's resident bytes itself (a
    /// progressive plane): the device holds them as uploaded, with no import
    /// entry.
    pub fn placed_on_host(&self) -> bool {
        self.progressive_depth().is_some()
    }

    /// Whether the weight is a table whose slots the host fills
    /// (`TableFill::Stored`): it takes no import entry.
    pub fn stored_in_slots(&self) -> bool {
        matches!(
            self.streamed(),
            Some(StreamedRows {
                fill: TableFill::Stored,
                ..
            })
        )
    }

    /// Whether the weight is an imported table whose landing slots hold the
    /// rows a read of rows the device selects names that its other slots do
    /// not (`StreamedRows::landing_rows`).
    pub fn lands_selected_rows(&self) -> bool {
        matches!(
            self.streamed(),
            Some(StreamedRows {
                fill: TableFill::Imported(_),
                landing_rows: 1..,
                ..
            })
        )
    }

    /// The source code depth retained by a progressive placement.
    pub(crate) fn progressive_depth(&self) -> Option<ProgressiveCodeDepth> {
        self.descriptor.progressive().map(|(_, depth)| depth)
    }

    /// The shape of the stored tensor this weight is made of: `shape`, with
    /// a streamed table's rows in place of its slots, and a banked stack's
    /// experts in place of its group's records.
    pub fn stored_shape(&self) -> Vec<u64> {
        match self.holding {
            Holding::Whole => self.shape.clone(),
            Holding::Streamed(streamed) => self.table_rows_shape(streamed.table_rows),
            Holding::Banked { .. } => self.descriptor.shape.clone(),
        }
    }

    /// The shape `[R, F]` of a streamed table's device directory
    /// (`DirectoryForm`); `[0, 1]`, the empty directory, of a weight held
    /// whole.
    pub fn directory_shape(&self) -> [u64; 2] {
        self.streamed().map_or([0, 1], |streamed| {
            [
                streamed
                    .directory
                    .entries(streamed.table_rows, self.table_slots()),
                streamed.directory.fields(),
            ]
        })
    }

    /// A streamed table's slots: the rows of `shape` (a stored table's
    /// records of rows, `[records, rows, width]`).
    pub fn table_slots(&self) -> u64 {
        self.shape[..self.shape.len() - 1].iter().product()
    }

    /// The stored bytes of a streamed table's rows its slots do not hold
    /// (its landing slots hold a read's rows only until it converts them),
    /// each as the model file stores it: what its steps read from the file.
    /// None of a weight held otherwise.
    pub fn unheld_bytes(&self) -> Result<u64, String> {
        let Some(streamed) = self.streamed() else {
            return Ok(0);
        };
        let held = self.table_slots().saturating_sub(streamed.landing_rows).min(streamed.table_rows);
        representation_bytes(self.source, &[1, self.table_width()])?
            .checked_mul(streamed.table_rows - held)
            .ok_or_else(|| "unheld table bytes overflow".to_owned())
    }

    /// The width of a streamed table's rows.
    pub fn table_width(&self) -> u64 {
        self.shape[self.shape.len() - 1]
    }

    /// `rows` rows of a streamed table, `[rows, width]`.
    fn table_rows_shape(&self, rows: u64) -> Vec<u64> {
        vec![rows, self.table_width()]
    }

    /// The allocations a streamed table holds beside its slots: the
    /// directory, the staging its tiles pass through and the landing staging
    /// its landing slots are converted from, both as stored, and the notice a
    /// selected read's step copies its `[rows, 2]` selection into. Zeros for
    /// a weight held whole.
    pub fn streamed_bytes(&self) -> Result<[u64; 4], String> {
        let Some(streamed) = self.streamed() else {
            return Ok([0; 4]);
        };
        let rows = |rows| self.table_rows_shape(rows);
        Ok([
            representation_bytes(Element::i32(), &self.directory_shape())?,
            self.staging_bytes(1)?,
            representation_bytes(self.upload, &rows(streamed.landing_rows))?,
            representation_bytes(Element::i32(), &[streamed.landing_rows, 2])?,
        ])
    }

    /// The stagings a streamed table's tiles pass through, as stored, that a
    /// held class of `places` places has in flight (`TableFlights::stagings`,
    /// `DeviceTable::hold_places`); none of a weight held whole or a stored
    /// table.
    pub fn staging_bytes(&self, places: usize) -> Result<u64, String> {
        let Some(StreamedRows { fill: TableFill::Imported(flights), staging_rows, .. }) = self.streamed() else {
            return Ok(0);
        };
        representation_bytes(self.upload, &self.table_rows_shape(staging_rows))?
            .checked_mul(flights.stagings(places) as u64)
            .ok_or_else(|| "a streamed table's stagings overflow".to_owned())
    }

    /// Every resident byte of the weight: its representation, its scale, its
    /// packing bound and what a streamed table holds beside its slots.
    pub fn storage_bytes(&self) -> Result<u64, String> {
        let overflow = || "resident weight byte count overflows".to_owned();
        let scale = self.scale.as_ref().map_or(0, |scale| scale.resident_bytes);
        let bound = self.bound.checked_mul(4).ok_or_else(overflow)?;
        [scale, bound]
            .into_iter()
            .chain(self.streamed_bytes()?)
            .try_fold(self.resident_bytes, u64::checked_add)
            .ok_or_else(overflow)
    }

    /// The entries of the record table the weight's storage holds where its
    /// import entry writes it through a record port
    /// (`ModelLoadPlan::with_record_ports`): a weight imported whole binds
    /// its import view's matrices (`native_import::import_view`), and a
    /// streamed imported table's slots hold the table of their tiles, into
    /// which every run and landed read binds (`DeviceTable::new`); a stored
    /// table's slots, which `stored_rows` reads through a record port, hold
    /// the table of their records (`StoredTable::new`). None of a weight no
    /// entry imports: a banked stack (its records are the bank's) and a
    /// plane the host places.
    pub fn record_table_entries(&self) -> Result<u64, String> {
        match self.holding {
            Holding::Banked { .. } => Ok(0),
            Holding::Streamed(StreamedRows {
                fill: TableFill::Stored,
                ..
            }) => Ok(self.shape[0]),
            Holding::Streamed(_) => Ok(self.shape[0] / self.resident.tile_rows()),
            Holding::Whole if self.placed_on_host() => Ok(0),
            Holding::Whole => crate::programs::native_import::import_view(&self.shape)
                .map(|[matrices, ..]| matrices)
                .map_err(|error| error.to_string()),
        }
    }

    /// The extent of the weight's accumulator-scale port: 0 without a
    /// second-level scale, else its resident scale's (1 for a matrix, the
    /// stack's matrix count for a stacked weight).
    pub fn scale_extent(&self) -> u64 {
        self.scale.as_ref().map_or(0, |scale| scale.extent)
    }
}

/// A resident weight's token packing: a weight in the layout the
/// token-packing projection forms read (Metal's `rows32`) that the form
/// admits (`Element::metal_packing`); `None` for any other.
fn packing(resident: Element) -> Option<MetalPacking> {
    (resident.layout() == Layout::Rows32)
        .then(|| resident.metal_packing())
        .flatten()
}

/// The columns of one run of a resident weight's packing bound, the columns
/// that share a scale, for a weight with a token packing; `None` for any
/// other.
pub fn packing_run(resident: Element) -> Option<u64> {
    packing(resident).map(|packing| u64::from(packing.run()))
}

/// The extent of a weight's packing bound (the `packing_bound` entry), the
/// runs of its K columns for a matrix with a packing run, else 0. Per run it
/// holds the largest squared l2 norm of a row's codes minus their centre
/// over the matrix's rows, made with the weight on the device when it is
/// imported: the packed projection forms
/// (`kernels/metal/lib/projection/packing.h`) certify their activation gains
/// against it rather than against every row of the format.
fn packing_bound(resident: Element, logical: &[u64]) -> u64 {
    match (logical, packing_run(resident)) {
        ([_, columns], Some(run)) if columns % run == 0 => columns / run,
        _ => 0,
    }
}

/// A packed weight's second-level scale (NVFP4's per-tensor or per-matrix
/// F32 `.scale`): a stored tensor of its own, `[1]` or `[stack]`, resident
/// beside the weight's representation as one F32 value per matrix. Entries
/// multiply their F32 accumulator by it through their accumulator-scale
/// port; an entry without one refuses the weight at plan time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightScalePlan {
    /// The stored scale tensor.
    pub tensor: String,
    /// Its stored extent: 1, or the weight's matrix count (of which a plan
    /// of a stack's first experts takes the leading `extent`).
    pub stored: u64,
    /// Its resident extent: 1 for a matrix, the matrix count for a stacked
    /// weight (a stored `[1]` is repeated per matrix).
    pub extent: u64,
    /// `extent` F32 values.
    pub resident_bytes: u64,
}

/// A projection weight's planned form at an entry with an accumulator-scale
/// port: its resident element and the port's extent (0 without a scale).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScalableWeight {
    pub element: Element,
    pub scale: u64,
}

/// Physical resident storage identity. Multiple semantic roles may name one
/// tensor, while distinct resident representations, and distinct import
/// transforms of one stored tensor, require distinct storage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightStorageIdentity {
    pub component: ArtifactComponent,
    pub tensor_name: String,
    pub transforms: Vec<ImportTransform>,
    pub resident: Element,
}

impl WeightPlan {
    pub fn storage_identity(&self) -> WeightStorageIdentity {
        WeightStorageIdentity {
            component: self.component,
            tensor_name: self.descriptor.name.clone(),
            transforms: self.descriptor.transforms.clone(),
            resident: self.resident,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EmbeddingBinding {
    pub table: Element,
    pub activation: Element,
}

/// Where an attention layer's values come from (`ValueSource`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttentionValues {
    /// Their own projection (`V` rows). A Shared layer, which projects
    /// none, reads its source's own values (`admit_shared`).
    Projected,
    /// The raw key projection, before key norm and rotary.
    RawKey,
    /// The stored key row (`VK = 1`): the history holds no value plane.
    StoredKey,
}

/// A rounding as the entries take it (`rounding.seismic`'s four scalars):
/// the one conversion of a [`RowRounding`] to them, for every entry that
/// rounds a row. A graph holds its scalars as numbers
/// ([`RoundingArgs::words`]) and its floor, held as its bits, as a real.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RoundingArgs {
    pub block: i32,
    pub element: i32,
    pub scale: i32,
    amax_floor_bits: u32,
}

impl RoundingArgs {
    /// No rounding: `block` 0.
    pub const NONE: Self = Self { block: 0, element: 0, scale: 0, amax_floor_bits: 0 };

    /// Whether it rounds anything (`block` > 0).
    pub fn rounds(self) -> bool {
        self.block > 0
    }

    pub fn amax_floor(self) -> f32 {
        f32::from_bits(self.amax_floor_bits)
    }

    /// A graph's part of it: its integer scalars as numbers and its floor
    /// as a real, which reach only its entries' arguments; none where it
    /// rounds nothing.
    pub(crate) fn part(self) -> Option<(RoundingWords<u64>, f64)> {
        self.rounds().then(|| {
            (
                RoundingWords { block: code(self.block), element: code(self.element), scale: code(self.scale) },
                f64::from(self.amax_floor()),
            )
        })
    }

    /// Whether the Metal and CUDA entries can round a `width`-value row
    /// with it (`round_lanes`): they spread the row over 32 lanes of
    /// `width / 32` values, and a run lies within one lane or spans a power
    /// of two of aligned lanes, so `width / block` is a multiple of 32 or a
    /// power of two. Contract validation holds `block` to dividing `width`.
    pub fn rounds_in_lanes(self, width: u64) -> bool {
        if !self.rounds() {
            return true;
        }
        let block = u64::try_from(self.block).expect("a rounding's positive run");
        assert!(
            width.is_multiple_of(block),
            "a validated rounding's run of {block} divides its row's width {width}"
        );
        // A row spreads over 32 lanes of width / 32 values each; a run
        // stays within a lane or spans a power of two of whole lanes.
        let runs = width / block;
        width.is_multiple_of(32) && (runs.is_multiple_of(32) || runs.is_power_of_two())
    }
}

/// A rounding's integer scalars (`block`, `element`, `scale`) as numbers of
/// a graph; its floor is a real ([`RoundingArgs::part`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RoundingWords<D = Dim> {
    pub block: D,
    pub element: D,
    pub scale: D,
}

impl<D> Numbers<D> for RoundingWords<D> {
    type Of<E> = RoundingWords<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> RoundingWords<E> {
        RoundingWords { block: f(self.block), element: f(self.element), scale: f(self.scale) }
    }
}

/// A rounding as an entry's four scalar arguments.
#[derive(Clone, Copy)]
pub(crate) struct RoundingArguments {
    pub block: Argument<i32>,
    pub element: Argument<i32>,
    pub scale: Argument<i32>,
    pub amax_floor: Argument<f32>,
}

impl RoundingArguments {
    /// The arguments of a graph's rounding part ([`RoundingArgs::part`]):
    /// every scalar 0 where it has none.
    pub(crate) fn of(part: Option<(&RoundingWords, &Argument<f64>)>) -> Result<Self, GraphError> {
        Ok(match part {
            Some((words, amax_floor)) => Self {
                block: scalar(&words.block, "a rounding's run")?,
                element: scalar(&words.element, "a rounding's element code")?,
                scale: scalar(&words.scale, "a rounding's scale code")?,
                amax_floor: amax_floor.f32(),
            },
            None => Self { block: 0.into(), element: 0.into(), scale: 0.into(), amax_floor: 0.0.into() },
        })
    }
}

/// `rounding` as the entries' scalars: `block` 0 for none. Contract
/// validation bounds the run and the floor; a run past `i32` is an invariant
/// violation.
pub fn rounding_args(rounding: RowRounding) -> RoundingArgs {
    match rounding {
        RowRounding::None => RoundingArgs::NONE,
        RowRounding::Block(block) => RoundingArgs {
            block: i32::try_from(block.block).expect("a validated rounding run fits i32"),
            amax_floor_bits: {
                assert!((block.amax_floor as f32).is_normal(), "a validated rounding's floor is a positive normal f32");
                (block.amax_floor as f32).to_bits()
            },
            element: match block.element {
                BlockElement::E4M3 => 0,
                BlockElement::E2M1 => 1,
            },
            scale: match block.scale {
                BlockScale::Ue8m0 => 0,
                BlockScale::E4M3 => 1,
            },
        },
    }
}

/// The static axes the attention entries are specialized to: its
/// structure and its numbers. Optional parts of the operator are axes of
/// extent 0 or 1 (`attention.seismic`), which its [`AttentionForm`]
/// decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionShape {
    pub form: AttentionForm,
    pub widths: AttentionWidths<u64>,
    /// The layer's trained key rounding (`ProjectedRows::rounding`), applied
    /// to each prepared key before it is attended and stored: the dense
    /// entries' `KR` ([`Self::rounds_keys`]) and their four `key_*`
    /// scalars.
    pub key_rounding: RoundingArgs,
}

impl AttentionShape {
    /// Its numbers as constant dimensions, with its form: what every
    /// arithmetic of it below is computed from ([`AttentionWidths`]).
    fn dims(&self) -> (AttentionForm, AttentionWidths<Dim>) {
        (self.form, constant(self.widths))
    }

    pub fn heads(&self) -> u64 {
        self.dims().1.heads().fixed()
    }

    /// `KR` ∈ {0, 1}: whether the dense entries round each prepared key.
    pub fn rounds_keys(&self) -> u64 {
        u64::from(self.key_rounding.rounds())
    }

    /// `I` ∈ {0, W}: gate columns after each query head's W columns.
    pub fn interleaved_gate(&self) -> u64 {
        let (form, widths) = self.dims();
        widths.interleaved_gate(&form).fixed()
    }

    /// `U` ∈ {0, 1, W}: separate gate values per query head.
    pub fn separate_gate(&self) -> u64 {
        let (form, widths) = self.dims();
        widths.separate_gate(&form).fixed()
    }

    /// Whether the shape takes a form beyond the base dense mix every
    /// backend's decode entry is held to: a query norm apart from the key's,
    /// stored-key values, adjacent or trailing pairs, the output rotary,
    /// rounded keys or picks. Only such a shape's decode classes may attend
    /// with the prefill entry (`operators::attention::graph::dense_decodes`).
    pub fn extends_base_form(&self) -> bool {
        let form = &self.form;
        form.query_norm != form.key_norm
            || form.stored_key()
            || form.rotary_adjacent
            || form.rotary_trailing
            || form.output_rotary
            || self.key_rounding.rounds()
            || self.widths.picks != 0
    }

    /// `Q`: query (and interleaved gate) projection rows.
    pub fn query_rows(&self) -> u64 {
        let (form, widths) = self.dims();
        widths.query_rows(&form).fixed()
    }

    /// `GR`: separate gate projection rows.
    pub fn gate_rows(&self) -> u64 {
        let (form, widths) = self.dims();
        widths.gate_rows(&form).fixed()
    }

    /// `K`: key projection rows.
    pub fn key_rows(&self) -> u64 {
        let (form, widths) = self.dims();
        widths.key_rows(&form).fixed()
    }

    /// `V`: value projection rows.
    pub fn value_rows(&self) -> u64 {
        let (form, widths) = self.dims();
        widths.value_rows(&form).fixed()
    }

    /// `VK` ∈ {0, 1}: whether the values are the stored keys.
    pub fn stored_key(&self) -> u64 {
        u64::from(self.form.stored_key())
    }

    /// Unrotated columns of a head.
    pub fn static_width(&self) -> u64 {
        self.dims().1.static_width().fixed()
    }

    /// The projection entry's dimensions at `rows` rows: the query
    /// segment projects the query rows, or a low-rank query's `query_rank`
    /// rows of `qr`.
    pub fn project_dimensions(
        &self,
        rows: u64,
        query_rank: Option<u64>,
    ) -> [(&'static str, u64); 6] {
        let (form, widths) = self.dims();
        let query_rank = query_rank.map(Dim::from);
        fixed(widths.project_dimensions(&form, Dim::from(rows), query_rank.as_ref()))
    }

    /// Projection of history rows has no query or gate outputs.
    pub fn key_value_dimensions(&self, rows: u64) -> [(&'static str, u64); 6] {
        let (form, widths) = self.dims();
        fixed(widths.key_value_dimensions(&form, Dim::from(rows)))
    }

    /// The static axes of the fused attention entries.
    pub fn mix_statics(&self) -> [(&'static str, u64); 15] {
        let (form, widths) = self.dims();
        fixed(widths.mix_statics(&form))
    }

    /// The static axes of the dense fused entries: the shared ones, then
    /// the key rounding and the second source's picks, which no K8/V4 entry
    /// holds.
    pub fn dense_mix_statics(&self) -> Vec<(&'static str, u64)> {
        let widths = self.dims().1;
        self.mix_statics()
            .into_iter()
            .chain(fixed(widths.dense_statics(self.key_rounding.rounds())))
            .collect()
    }

    /// Whether the cache-publishing entries lack the fused entries' axis
    /// `name`.
    pub fn unread_by_appends(name: &str) -> bool {
        matches!(name, "G" | "I" | "U" | "SK" | "NQ" | "RO" | "R" | "KS" | "TS")
    }

    /// The dense fused entries' dimensions: [`Self::mix_dimensions`], the
    /// key rounding, the picks and the second source's plane rows `TS`.
    pub fn dense_mix_dimensions(
        &self,
        rows: u64,
        history_rows: u64,
        segments: u64,
        selected_rows: u64,
    ) -> Vec<(&'static str, u64)> {
        let widths = self.dims().1;
        self.mix_dimensions(rows, history_rows, segments)
            .into_iter()
            .chain(fixed(widths.dense_statics(self.key_rounding.rounds())))
            .chain([("TS", selected_rows)])
            .collect()
    }

    /// The fused entries' dimensions for `rows` rows over `history_rows`
    /// history rows read in `segments` visible spans.
    pub fn mix_dimensions(
        &self,
        rows: u64,
        history_rows: u64,
        segments: u64,
    ) -> [(&'static str, u64); 18] {
        let (form, widths) = self.dims();
        fixed(widths.mix_dimensions(
            &form,
            Dim::from(rows),
            Dim::from(history_rows),
            Dim::from(segments),
        ))
    }
}

/// Named dimensions of constant [`Dim`]s as their values.
fn fixed<const N: usize>(dimensions: [(&'static str, Dim); N]) -> [(&'static str, u64); N] {
    dimensions.map(|(name, dimension)| (name, dimension.fixed()))
}

/// Where an attention's gate values come from: none, `W` columns after each
/// query head's in its projection (`I = W`), or a separate projection
/// segment ([`SeparateGate`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GateForm {
    None,
    Interleaved,
    Separate(SeparateGate),
}

/// `U`: a separate gate's values per query head.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SeparateGate {
    /// `U = 1`: one gate value per head.
    Head,
    /// `U = W`: one gate value per column.
    Element,
}

/// An [`AttentionShape`]'s structure: its flags and kinds, which decide its
/// graphs' topology. It holds no number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionForm {
    /// `I`, `U`: the gate.
    pub gate: GateForm,
    /// `F`: whether the layer projects and appends its own keys and values.
    pub fresh: bool,
    /// `NQ`, `N`, `NV`: the per-head RMS norms (a Shared layer's `N` is its
    /// `NQ`).
    pub query_norm: bool,
    pub key_norm: bool,
    pub value_norm: bool,
    pub values: AttentionValues,
    /// `SK`: a sink logit per query head.
    pub sinks: bool,
    /// `RA`, `RT`, `RO`: the rotary layout and output rotary.
    pub rotary_adjacent: bool,
    pub rotary_trailing: bool,
    pub output_rotary: bool,
}

impl AttentionForm {
    /// Whether the projection has a separate gate segment (`GR > 0`).
    pub(crate) fn gated_separately(&self) -> bool {
        matches!(self.gate, GateForm::Separate(_))
    }

    /// Whether the projection has a value segment (`V > 0`).
    pub(crate) fn projects_values(&self) -> bool {
        self.fresh && self.values == AttentionValues::Projected
    }

    /// `VK`: whether the values are the stored keys.
    pub(crate) fn stored_key(&self) -> bool {
        self.values == AttentionValues::StoredKey
    }
}

/// An [`AttentionShape`]'s numbers: `D`, `KV`, `G`, `P`, `W` and `KS`. As
/// [`Dim`]s, with its [`AttentionForm`], they are the one owner of the
/// shape's arithmetic, which a graph's builder records and the shape's
/// own accessors evaluate at constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionWidths<D> {
    /// `D`: the hidden width.
    pub hidden: D,
    /// `KV`: key/value heads.
    pub kv_heads: D,
    /// `G`: query heads per key/value head.
    pub group: D,
    /// `P`: rotated pairs of a head.
    pub rotary_pairs: D,
    /// `W = 2P + S`: the head width.
    pub width: D,
    /// `KS`: the picks each row reads from the layer's second, index-list
    /// source (its selection's `top`); 0 for a layer reading one source.
    /// The dense attending entries alone hold it.
    pub picks: D,
}

impl<D> Numbers<D> for AttentionWidths<D> {
    type Of<E> = AttentionWidths<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> AttentionWidths<E> {
        AttentionWidths {
            hidden: f(self.hidden),
            kv_heads: f(self.kv_heads),
            group: f(self.group),
            rotary_pairs: f(self.rotary_pairs),
            width: f(self.width),
            picks: f(self.picks),
        }
    }
}

/// `1` where `flag` holds, else `0`: a 0/1 axis.
pub(crate) fn axis(flag: bool) -> Dim {
    Dim::from(u64::from(flag))
}

impl AttentionWidths<Dim> {
    /// `KV · G`: the query heads.
    pub(crate) fn heads(&self) -> Dim {
        &self.kv_heads * &self.group
    }

    /// `I`: gate columns after each query head's.
    pub(crate) fn interleaved_gate(&self, form: &AttentionForm) -> Dim {
        match form.gate {
            GateForm::Interleaved => self.width.clone(),
            GateForm::None | GateForm::Separate(_) => Dim::from(0),
        }
    }

    /// `U`: separate gate values per query head.
    pub(crate) fn separate_gate(&self, form: &AttentionForm) -> Dim {
        match form.gate {
            GateForm::None | GateForm::Interleaved => Dim::from(0),
            GateForm::Separate(SeparateGate::Head) => Dim::from(1),
            GateForm::Separate(SeparateGate::Element) => self.width.clone(),
        }
    }

    /// `Q`: query (and interleaved gate) projection rows.
    pub(crate) fn query_rows(&self, form: &AttentionForm) -> Dim {
        self.heads() * (&self.width + self.interleaved_gate(form))
    }

    /// `GR`: separate gate projection rows.
    pub(crate) fn gate_rows(&self, form: &AttentionForm) -> Dim {
        self.heads() * self.separate_gate(form)
    }

    /// `KV · W`: a row's keys or values.
    pub(crate) fn key_width(&self) -> Dim {
        &self.kv_heads * &self.width
    }

    /// `K`: key projection rows.
    pub(crate) fn key_rows(&self, form: &AttentionForm) -> Dim {
        match form.fresh {
            true => self.key_width(),
            false => Dim::from(0),
        }
    }

    /// `V`: value projection rows.
    pub(crate) fn value_rows(&self, form: &AttentionForm) -> Dim {
        match form.projects_values() {
            true => self.key_width(),
            false => Dim::from(0),
        }
    }

    /// `S`: a head's unrotated columns.
    pub(crate) fn static_width(&self) -> Dim {
        &self.width - &self.rotary_pairs * 2
    }

    /// The projection entry's dimensions at `rows` rows: its query segment
    /// projects the query rows, or a low-rank query's `query_rank` rows of
    /// `qr`.
    pub(crate) fn project_dimensions(
        &self,
        form: &AttentionForm,
        rows: Dim,
        query_rank: Option<&Dim>,
    ) -> [(&'static str, Dim); 6] {
        [
            ("M", rows),
            ("D", self.hidden.clone()),
            ("Q", query_rank.cloned().unwrap_or_else(|| self.query_rows(form))),
            ("GR", self.gate_rows(form)),
            ("K", self.key_rows(form)),
            ("V", self.value_rows(form)),
        ]
    }

    /// A projection of history rows alone: no query or gate outputs.
    pub(crate) fn key_value_dimensions(
        &self,
        form: &AttentionForm,
        rows: Dim,
    ) -> [(&'static str, Dim); 6] {
        let mut dimensions = self.project_dimensions(form, rows, None);
        dimensions[2].1 = Dim::from(0);
        dimensions[3].1 = Dim::from(0);
        dimensions
    }

    /// The static axes of the fused attention entries.
    pub(crate) fn mix_statics(&self, form: &AttentionForm) -> [(&'static str, Dim); 15] {
        [
            ("KV", self.kv_heads.clone()),
            ("G", self.group.clone()),
            ("P", self.rotary_pairs.clone()),
            ("S", self.static_width()),
            ("I", self.interleaved_gate(form)),
            ("U", self.separate_gate(form)),
            ("F", axis(form.fresh)),
            ("NQ", axis(form.query_norm)),
            ("N", axis(form.key_norm)),
            ("NV", axis(form.value_norm)),
            ("SK", axis(form.sinks)),
            ("RA", axis(form.rotary_adjacent)),
            ("RT", axis(form.rotary_trailing)),
            ("RO", axis(form.output_rotary)),
            ("VK", axis(form.stored_key())),
        ]
    }

    /// The dense fused entries' own static axes: whether they round keys
    /// (`rounds`) and the second source's picks, which no K8/V4 entry holds.
    pub(crate) fn dense_statics(&self, rounds: bool) -> [(&'static str, Dim); 2] {
        [("KR", axis(rounds)), ("KS", self.picks.clone())]
    }

    /// The fused entries' dimensions for `rows` rows over `history_rows`
    /// history rows read in `segments` visible spans.
    pub(crate) fn mix_dimensions(
        &self,
        form: &AttentionForm,
        rows: Dim,
        history_rows: Dim,
        segments: Dim,
    ) -> [(&'static str, Dim); 18] {
        let [kv, g, p, s, i, u, f, nq, n, nv, sk, ra, rt, ro, vk] = self.mix_statics(form);
        [
            ("M", rows),
            ("T", history_rows),
            kv,
            g,
            p,
            s,
            i,
            u,
            f,
            nq,
            n,
            nv,
            sk,
            ra,
            rt,
            ro,
            vk,
            ("R", segments),
        ]
    }
}

/// An attention operator's kernel binding. Weights a form lacks (a separate
/// gate, own keys or values) are empty segments of the projection; they
/// bind a zero-row view of the query weight and carry its element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionBinding {
    pub shape: AttentionShape,
    /// State-only projection omits query and gate segments statically.
    pub key_value_only: bool,
    pub norm: Element,
    /// The query heads' projection: a full query's, or a low-rank query's
    /// `up`.
    pub query: Element,
    /// A low-rank query's `qr` row (`QueryProjection::LowRank`); none for a
    /// full query.
    pub low_rank_query: Option<LowRankQueryBinding>,
    pub gate: Element,
    pub key: Element,
    pub value: Element,
    /// The output projection's (the second, `b`, of a grouped output).
    pub output: Element,
    pub activation: Element,
    /// How the block's history planes encode keys and values.
    pub history: magnitude_state::CodecSpec,
    pub tail: SublayerTail,
    /// A selected read's indexer (`HistoryReads::Selected`); none for a
    /// read of every visible row.
    pub selection: Option<SelectionBinding>,
    /// A grouped output's first product (`OutputProjection::GroupedLowRank`'s
    /// `a`); none for a full output projection.
    pub grouped: Option<GroupedOutputBinding>,
    /// The second source a row reads by its picks (`KS = shape.widths.picks`);
    /// none for a layer reading one source.
    pub picked: Option<PickedBinding>,
}

/// A layer's second source as its entries bind it: its own pooled rows
/// and indexer, its own indexer over another layer's rows and their index
/// keys (Reindex), or another layer's rows through that layer's picks. Its
/// rows are a one-head `[W]` plane in the activation element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PickedBinding {
    Own(PooledBinding),
    /// Its indexer over part `part` of target block `source`'s pooled rows
    /// (`LayerRef::Pooled`) and their index keys.
    Reindex {
        source: u32,
        part: u32,
        indexer: IndexerBinding,
    },
    Reused,
}

impl PickedBinding {
    /// The layer's own history part its picked rows are
    /// (`LayerRef::Pooled { part, .. }`): its second source
    /// (`operators::attention::picked_source`).
    pub fn part(&self) -> u32 {
        match self {
            Self::Own(pooled) => pooled.part,
            Self::Reindex { .. } | Self::Reused => 1,
        }
    }

    /// The indexer the layer runs itself; none for reused picks.
    pub fn indexer(&self) -> Option<IndexerBinding> {
        match self {
            Self::Own(pooled) => Some(pooled.indexer),
            Self::Reindex { indexer, .. } => Some(*indexer),
            Self::Reused => None,
        }
    }
}

/// A layer's own pooled source (`pool_rows`; at rate 1
/// `attention_append_dense`), the projection of its rows' index keys (`indexer_keys`) and the indexer that picks its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PooledBinding {
    /// The source's part of the layer's history (`LayerRef::Pooled`).
    pub part: u32,
    /// `RATE`: tokens per pooled row.
    pub rate: u64,
    /// The per-token projection's element and the softmax gate's (none for
    /// a mean pool).
    pub projection: Element,
    pub gate: Option<Element>,
    /// The pooled rows' norm (at rate 1, the index keys' projection's input
    /// norm, which forms the latent) and rounding.
    pub norm: Element,
    pub rounding: RoundingArgs,
    /// The index keys' projection of the latent and their norm.
    pub key: Element,
    pub key_norm: Element,
    pub indexer: IndexerBinding,
}

impl PooledBinding {
    /// `GT` ∈ {0, 1}: whether the pool is softmax-gated (it has a gate).
    pub fn gt(&self) -> u64 {
        u64::from(self.gate.is_some())
    }
}

/// An indexer over a pooled source's index keys (`selection.rs`): its
/// heads, their width and its picks, its query's and head weights'
/// elements, and the candidates it scores. The most rows a row's list
/// holds (`N`) is its class's
/// ([`crate::operators::attention::list_scores`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IndexerBinding {
    pub heads: u64,
    pub width: u64,
    pub top: u64,
    /// `NQ` ∈ {0, 1}: whether the query heads have their own norm.
    pub query_norm: u64,
    pub query: Element,
    /// The head weights' projection; none for uniform heads.
    pub head_weights: Option<Element>,
    pub candidates: CandidatesBinding,
    /// The trained roundings of its query rows (`indexer_score`) and of the
    /// keys it writes (`indexer_keys`); the keys' is none for keys it
    /// shares, which their owner writes.
    pub query_rounding: RoundingArgs,
    pub key_rounding: RoundingArgs,
}

/// The units an indexer scores (`Candidates`): every complete unit, those
/// of every complete unit while it also produces the candidate pool, or
/// those of the pool an earlier indexer produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CandidatesBinding {
    All,
    Produce(CandidatePool),
    Restrict(CandidatePool),
}

/// A candidate pool: `blocks` blocks of `block_units` units, listed for
/// the restricted indexers' `top` picks (`candidate_rows`' `K`, which fills
/// a pool of fewer units).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CandidatePool<D = u64> {
    pub blocks: D,
    pub block_units: D,
    pub top: D,
}

impl CandidatePool {
    /// [`CandidatePool::units`] of the model's pool.
    pub fn units(&self) -> u64 {
        constant(*self).units().fixed()
    }
}

impl CandidatePool<Dim> {
    /// The most units a row's pool list holds: its blocks' units, which
    /// the fill never exceeds (`top ≤ blocks · block_units`).
    pub(crate) fn units(&self) -> Dim {
        &self.blocks * &self.block_units
    }
}

impl<D> Numbers<D> for CandidatePool<D> {
    type Of<E> = CandidatePool<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> CandidatePool<E> {
        CandidatePool { blocks: f(self.blocks), block_units: f(self.block_units), top: f(self.top) }
    }
}

impl AttentionBinding {
    /// The output projection's input rows as `attention_output` reads them,
    /// `(Q, W)`: the attended heads, or a grouped output's `a` products.
    pub fn output_input(&self) -> (u64, u64) {
        match self.grouped {
            None => (self.shape.heads(), self.shape.widths.width),
            Some(grouped) => (grouped.shape.groups, grouped.shape.rank),
        }
    }
}

/// A grouped attention output's block-diagonal first product
/// (`project_rows` with G groups) and the element of its weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupedOutputBinding {
    pub shape: GroupedOutputShape,
    pub element: Element,
}

/// `project_rows`' axes of a grouped attention output: `groups` (G), each
/// projecting a run of `inputs` attended columns (K) to `rank` (N).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupedOutputShape {
    pub groups: u64,
    pub inputs: u64,
    pub rank: u64,
}

impl AttentionBinding {
    /// The weight the projection's query segment reads: the query's, or a
    /// low-rank query's `down`.
    pub fn query_segment(&self) -> Element {
        self.low_rank_query
            .map_or(self.query, |low_rank| low_rank.down)
    }
}

/// A low-rank query's `qr` row: the attention projection's query segment
/// projects its `rank` rows through `down`; widened to F32
/// (`widen_rows`), a second `attention_project` normalizes them by `norm`
/// in its prologue and projects them through the query weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LowRankQueryBinding {
    pub rank: u64,
    pub down: Element,
    pub norm: Element,
}

/// The indexer of a selected attention read (`indexer_select`): its shape
/// and the elements of its query and key projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SelectionBinding {
    pub shape: SelectionShape,
    pub query: Element,
    pub key: Element,
}

/// A selected read's static axes: `heads` indexer query heads of `width`
/// columns score blocks of `block` keys, of which a row past `blocks`
/// complete ones reads the `blocks` best and its tail. Prefill rows read
/// their blocks through a bitmap of `words` 32-block words (`B`), enough for
/// every block of the served context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SelectionShape {
    pub heads: u64,
    pub width: u64,
    pub block: u64,
    pub blocks: u64,
    pub words: u64,
}

/// How a sublayer's output projection joins the residual stream (the
/// contract's `OutputForm`): added by the operator's own output entry, or
/// projected to F32 and added through a row op that normalizes it first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SublayerTail {
    Residual,
    /// `(residual + RMS(projection)·norm)·scale`, the norm weight's element;
    /// `scaled` when the scale is the layer's output scale (else 1).
    PostNorm {
        norm: Element,
        scaled: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecurrentBinding {
    pub key_heads: u64,
    pub value_heads: u64,
    pub width: u64,
    pub convolution_width: u64,
    pub norm: Element,
    pub qkv: Element,
    pub gate: Element,
    pub alpha: Element,
    pub beta: Element,
    pub recurrent_norm: Element,
    pub output: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DenseBinding {
    /// The expansion width (`F`), a static of the specializations: layers
    /// of different widths (Gemma's double-wide layers) need their own.
    pub features: u64,
    pub norm: Element,
    pub gate: Element,
    pub up: Element,
    pub down: Element,
    pub activation: Element,
    /// The semantic activation code of the expansion (functions.seismic).
    pub function: i32,
    /// The expansion's `glu` clamp.
    pub limit: Option<magnitude_family_contracts::GluLimit>,
    pub tail: SublayerTail,
    pub scales: DenseScales,
}

/// The extents of a dense feed-forward's per-tensor accumulator-scale ports
/// (`dense_expand`/`dense_up` gate and up, `dense_output` or the post-norm
/// tail's `project_rows` down): 1 where the weight has a resident
/// second-level scale, else 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DenseScales {
    pub gate: u64,
    pub up: u64,
    pub down: u64,
}

/// A dense branch beside a general routed branch
/// (`operators::DenseBesideRouted`): the dense branch's expansion and its
/// down projection into F32, the routed branch summed onto zeros, and
/// `moe_tail`, which normalizes each branch, their sum, and adds it through
/// the sublayer's post-norm tail (scaled or not). `norm` is the element of
/// all three tail norms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParallelBinding {
    pub hidden: u64,
    pub dense: DenseBranchBinding,
    pub routed: crate::operators::routed::GeneralRoutedBinding,
    pub norm: Element,
    pub scaled: bool,
}

/// The dense branch of a [`ParallelBinding`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DenseBranchBinding {
    /// The expansion width (`F`).
    pub features: u64,
    pub norm: Element,
    pub gate: Element,
    pub up: Element,
    pub down: Element,
    pub activation: Element,
    pub function: i32,
    /// The expansion's `glu` clamp.
    pub limit: Option<magnitude_family_contracts::GluLimit>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RoutedBinding {
    pub hidden: u64,
    pub experts: u64,
    pub selected: u64,
    pub features: u64,
    pub shared: u64,
    pub normalize_selected: bool,
    pub norm: Element,
    pub router: Element,
    pub expert_gate: Element,
    pub expert_up: Element,
    pub expert_down: Element,
    pub shared_gate: Element,
    pub shared_up: Element,
    pub shared_down: Element,
    pub activation: Element,
    /// The experts' per-expert scale ports (the shared expert binds none).
    pub scales: crate::ExpertScales,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadoutBinding {
    pub norm: Element,
    pub head: ReadoutHead,
    pub activation: Element,
}

/// How the plan places the vocabulary projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReadoutHead {
    /// The output projection's resident element and the extent of its
    /// accumulator-scale port.
    Packed { weight: Element, weight_scale: u64 },
    /// Stored as its planes: certified classes run the bounded levels.
    Qualified { weight: Element, weight_scale: u64 },
}

impl ReadoutHead {
    pub fn packed(self) -> ScalableWeight {
        match self {
            Self::Packed {
                weight,
                weight_scale,
            }
            | Self::Qualified {
                weight,
                weight_scale,
            } => ScalableWeight {
                element: weight,
                scale: weight_scale,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FeaturesBinding {
    pub norm: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadBinding {
    pub streams: Option<crate::operators::streams::StreamsBinding>,
    pub conditioning: Element,
    pub embedding_table: Element,
    pub embedding_norm: Element,
    pub hidden_norm: Element,
    pub combine: Element,
    pub attention: AttentionBinding,
    pub feed_forward: FeedForwardProgramSlot,
    pub output_norm: Element,
    pub projection: ReadoutHead,
    pub activation: Element,
}

/// One reader of a projection scope stored as its views: the reader's listed
/// selected-row classes run the certified levels, and its other reads the
/// views' full pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadoutClassChoice {
    pub scope: WeightScope,
    pub norm_scope: WeightScope,
    pub prepared: bool,
    pub rows: Vec<u64>,
}

/// What a load is planned to hold in host RAM at its most, by what holds
/// it. Each term is derived from what is planned; what a load holds on the
/// host that no term here names (its prepared kernels and their records,
/// its domain's bookkeeping) is outside the plan, so the sum is at most the
/// load's host peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostPeak {
    /// The host-resident gathered tables.
    pub tables: u64,
    /// The largest weight the host prepares or places, held whole while it
    /// is imported ([`ModelLoadPlan::host_staging_peak_bytes`]).
    pub staging: u64,
    /// The sealed graphs' plans, held from sealing on.
    pub sealed_graphs: u64,
    /// The sealed graphs' bound instances, made when the domain is
    /// allocated, after the import.
    pub bound_graphs: u64,
}

impl HostPeak {
    /// What a load holds on the host that no term names, by holder. Each
    /// is real growth of a load that whatever is held to the planned peak
    /// is not given: a search refused for bytes names them, so that what
    /// its ceiling lacks can be read against what would supply it.
    pub const UNPLANNED: &'static [&'static str] = &[
        "prepared kernels: loaded modules, their records and receipts",
        "sealing span beyond its planned terms: readout graphs, head, draft, shared-head and \
         vision graphs, and what certification leaves live",
        "bindings of the readout, head, draft, shared-head and vision graphs",
        "graph pools and residency bookkeeping",
        "warm-up",
        "allocator rounding of every planned block",
    ];

    /// The planned host peak. Graphs are sealed before the import begins
    /// and bound after it ends, so the tables and the sealed plans are held
    /// with the staged weight during the import and with the bound
    /// instances after it: the larger of the two.
    pub fn bytes(&self) -> Result<u64, String> {
        self.tables
            .checked_add(self.process_bytes()?)
            .ok_or_else(|| "planned host peak overflows".to_owned())
    }

    /// The part of the planned host peak that is the load process's own
    /// memory: everything but the tables. A host table is read by row from
    /// its artifact (`HostTable` holds the file and an offset), so its
    /// bytes are the system's cache of that file, a claim on system RAM
    /// that no process's footprint counts. What a process may hold and
    /// stay under the load is this.
    pub fn process_bytes(&self) -> Result<u64, String> {
        self.sealed_graphs
            .checked_add(self.staging.max(self.bound_graphs))
            .ok_or_else(|| "planned host peak overflows".to_owned())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelLoadPlan {
    /// Width of raw F32 residual conditioning requested by the selected head.
    pub(super) residual_conditioning: Option<u64>,
    pub(super) target: Vec<WeightPlan>,
    pub(super) head: Option<Vec<WeightPlan>>,
    pub(super) vision: Option<Vec<WeightPlan>>,
    /// The target's host-resident gathered tables.
    pub(super) host_tables: Vec<HostTablePlan>,
    /// The lossless readout views a scope may be stored as. They are not
    /// resident requirements until a reader's choice stores its scope as them.
    pub(super) readout_candidates: Vec<WeightPlan>,
    pub(super) readout_choices: Vec<ReadoutClassChoice>,
    /// The output projections of the scopes stored as planes
    /// ([`ModelLoadPlan::finalize_readout`]): planned, so bindings that name
    /// them stand and their demand is derived, and not placed. A weight has
    /// one stored form.
    pub(super) unplaced_output: Vec<WeightPlan>,
    /// The expert bank of a load that streams its routed experts
    /// (`with_expert_bank`): the target's expert stacks are `Banked`.
    pub(super) bank: Option<super::ExpertBankPlan>,
    /// The backend whose native import entries write the weights through
    /// record ports (`with_record_ports`), which on a backend that binds
    /// record tables leaves each imported weight's storage holding one
    /// (`WeightPlan::record_table_entries`); none of a load no native
    /// entry imports.
    pub(super) record_ports: Option<seismic::BackendName>,
}

/// Whether a readout scope's weights are planned with the head component's.
fn head_component_scope(scope: WeightScope) -> Result<bool, String> {
    match scope {
        WeightScope::Target => Ok(false),
        WeightScope::HeadBlock(_) | WeightScope::Draft => Ok(true),
        _ => Err("qualified readout representation has an unsupported scope".into()),
    }
}

impl ModelLoadPlan {
    pub(crate) fn residual_conditioning(&self) -> Option<u64> {
        self.residual_conditioning
    }

    pub fn host_tables(&self) -> &[HostTablePlan] {
        &self.host_tables
    }

    /// The host tables' claim in the system-RAM domain: each is streamed,
    /// and claims its row cache.
    pub fn host_table_bytes(&self) -> Result<u64, String> {
        self.host_tables.iter().try_fold(0u64, |total, table| {
            let bytes =
                table.bytes / table.shape[0] * crate::streamed_table::cached_rows(table.shape[0]);
            total
                .checked_add(bytes)
                .ok_or_else(|| "host table bytes overflow".to_owned())
        })
    }

    /// The stored bytes of the rows of the load's tables read by row that
    /// no slot holds: each streamed table's rows beyond its slots
    /// ([`WeightPlan::unheld_bytes`]) and each host table's beyond its row
    /// cache. None of a load that holds every table whole.
    pub fn unheld_table_bytes(&self) -> Result<u64, String> {
        let overflow = || "unheld table bytes overflow".to_owned();
        let streamed = weight_storages(self)?
            .into_iter()
            .try_fold(0u64, |total, (_, weight)| total.checked_add(weight.unheld_bytes()?).ok_or_else(overflow))?;
        self.host_tables.iter().try_fold(streamed, |total, table| {
            let rows = table.shape[0];
            representation_bytes(table.source, &[1, table.columns()])?
                .checked_mul(rows - crate::streamed_table::cached_rows(rows))
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(overflow)
        })
    }

    /// Physical weight demand by component, with tied storage owned by its
    /// first component, exactly as the resident byte total.
    pub fn weight_demands(&self) -> Result<[ClaimDemand; 3], String> {
        weight_demands_by_component(self)
    }

    /// The expert bank of a load that streams its routed experts.
    pub fn expert_bank(&self) -> Option<&super::ExpertBankPlan> {
        self.bank.as_ref()
    }

    /// The launches one merged-expert prompt pass of the load covers: its
    /// bank's settled pass, or a whole chain's where its experts are held
    /// whole.
    pub fn pass(&self, limits: super::ResourceLimits) -> crate::PassLaunches {
        self.bank.as_ref().map_or(crate::PassLaunches::whole_chain(limits), |bank| bank.pass)
    }

    /// Every weight byte the load holds, wherever it lies (what a weight
    /// cap bounds): each component's, the expert bank's, and its host
    /// tier's slots. The tier's landing reserve (the tier at no slots) is
    /// staging, as the import's is, and counts none.
    pub fn weight_bytes(&self) -> Result<u64, String> {
        self.weight_owners()?
            .into_iter()
            .try_fold(0u64, |total, (_, bytes)| total.checked_add(bytes))
            .ok_or_else(|| "resident weight byte count overflow".to_owned())
    }

    /// The model's weights as its files store them, by where the load holds
    /// them as it runs ([`WeightPlacement`]).
    pub fn weight_placement(&self) -> Result<WeightPlacement, String> {
        let overflow = || "weight placement bytes overflow".to_owned();
        // Each stored tensor once, however many roles or representations
        // the load gives it (a tied embedding is a table and a head).
        let mut counted = HashSet::new();
        let mut stored = 0u64;
        let weights = [self.target.as_slice(), self.head.as_deref().unwrap_or_default(), self.vision.as_deref().unwrap_or_default()];
        for weight in weights.into_iter().flatten() {
            if counted.insert((weight.component, weight.descriptor.name.clone())) {
                stored = stored.checked_add(representation_bytes(weight.source, &weight.stored_shape())?).ok_or_else(overflow)?;
            }
        }
        let host = self.bank.as_ref().map_or(Ok(0), super::ExpertBankPlan::host_only_bytes)?;
        for table in &self.host_tables {
            if counted.insert((table.component, table.descriptor.name.clone())) {
                let rows = table.shape[0];
                stored = representation_bytes(table.source, &[rows, table.columns()])?.checked_add(stored).ok_or_else(overflow)?;
            }
        }
        let streamed = self.bank.as_ref().map_or(Ok(0), super::ExpertBankPlan::unheld_bytes)?;
        let tables = self.unheld_table_bytes()?;
        let held = [host, streamed, tables]
            .into_iter()
            .try_fold(stored, u64::checked_sub)
            .ok_or("a plan's tiers hold more than its weights")?;
        Ok(WeightPlacement { held, host, streamed, tables })
    }

    /// The host tier's slots beyond its landing reserve; none without a
    /// tier.
    pub fn host_tier_bytes(&self) -> Result<u64, String> {

        let Some(bank) = self.bank.as_ref().filter(|bank| bank.host_ceiling().is_some()) else {
            return Ok(0);
        };
        bank.host_charge(bank.host_slots)?
            .checked_sub(bank.host_charge(0)?)
            .ok_or_else(|| "a host tier charges less at its slots than at none".to_owned())
    }

    /// What a weight cap bounds ([`Self::weight_bytes`]), by the owner of
    /// each byte: each component's weights beside the expert bank, the
    /// bank's parts ([`super::ExpertBankPlan::charges`]), then its host
    /// tier's slots; their sum is the weight bytes.
    pub fn weight_owners(&self) -> Result<Vec<(&'static str, u64)>, String> {
        // A component's weight byte: the host tier's are counted apart.
        let total = |demand: ClaimDemand| {
            demand.device.checked_add(demand.reserved)?.checked_add(demand.upload)?.checked_add(demand.shared)
        };
        let overflow = || "resident weight byte count overflow".to_owned();
        let [target, head, vision] = self.weight_demands()?;
        let bank = match &self.bank {
            Some(bank) => bank.charges()?.into_iter().map(|(owner, demand)| Some((owner, total(demand)?))).collect::<Option<Vec<_>>>().ok_or_else(overflow)?,
            None => Vec::new(),
        };
        let banked = bank.iter().map(|&(_, bytes)| bytes).sum::<u64>();
        let target = total(target).ok_or_else(overflow)?.checked_sub(banked).ok_or("the bank charges more than the target's weights")?;
        let mut owners = vec![
            ("target weights beside the bank", target),
            ("head weights", total(head).ok_or_else(overflow)?),
            ("vision weights", total(vision).ok_or_else(overflow)?),
        ];
        owners.extend(bank);
        owners.push(("host tier slots", self.host_tier_bytes()?));
        Ok(owners)
    }

    /// The most of one target weight's import the host holds at once. A
    /// weight uploaded as stored never lies on the host whole: its upload
    /// is the device's (or a view of the mapped artifact), written through
    /// a block. A weight the host prepares (transformed or dequantized) or
    /// places itself (a progressive plane) is held on the host in its
    /// upload representation before it is written. Imports are serialized,
    /// so the peak is the largest such weight.
    pub fn host_staging_peak_bytes(&self) -> u64 {
        self.target
            .iter()
            .filter(|weight| weight.host_prepared() || weight.placed_on_host())
            .map(|weight| weight.source_bytes)
            .max()
            .unwrap_or(0)
    }

    /// What the load holds in host RAM at its most, by what holds it, given
    /// the host bytes of its sealed graphs' plans and of their bound
    /// instances.
    pub fn host_peak(&self, sealed_graphs: u64, bound_graphs: u64) -> Result<HostPeak, String> {
        Ok(HostPeak {
            tables: self.host_table_bytes()?,
            staging: self.host_staging_peak_bytes(),
            sealed_graphs,
            bound_graphs,
        })
    }

    /// Admission for the same bounded source batches that residency imports.
    pub(crate) fn imports(&self) -> super::ImportPlan {
        super::ImportPlan::new(self.weights())
    }

    /// Peak upload backing while the target component is imported.
    pub fn target_upload_peak_bytes(&self) -> u64 {
        super::ImportPlan::new(self.target.iter()).peak_bytes()
    }

    /// The target's weights as its import claims them, held for the load:
    /// each target weight allocation in the storage it lies in, and a
    /// linked host tier backed to its load-time slots past the floor its
    /// layout commits (`ExpertBank::back_host`), which backs after the
    /// import ([`Self::host_tier_bytes`]).
    pub fn target_weight_demand(&self) -> Result<ClaimDemand, String> {
        let weights = self.weight_demands()?[0];
        Ok(ClaimDemand { host: weights.host.checked_add(self.host_tier_bytes()?).ok_or("target import demand overflows")?, ..weights })
    }

    /// The target import's uploads: device storage staged through host
    /// RAM, held while the target imports.
    pub fn target_upload_demand(&self) -> Result<ClaimDemand, String> {
        let upload = self.target_upload_peak_bytes();
        Ok(ClaimDemand { device: upload, staged: upload, ..ClaimDemand::default() })
    }

    /// Peak upload allocation while importing a lazy optional component.
    pub fn head_upload_peak_bytes(&self) -> u64 {
        super::ImportPlan::new(self.head.iter().flatten()).peak_bytes()
    }

    pub fn vision_upload_peak_bytes(&self) -> u64 {
        super::ImportPlan::new(self.vision.iter().flatten()).peak_bytes()
    }
}

/// The elements of a tensor of `extents`; `None` beyond `u64`.
pub(crate) fn elements(extents: &[u64]) -> Option<u64> {
    extents
        .iter()
        .try_fold(1u64, |count, extent| count.checked_mul(*extent))
}

/// Each component's weight bytes in the device's allocation domain: all its
/// storage but a linked host tier's records, which lie in host storage.
pub(super) fn weight_bytes_by_component(load: &ModelLoadPlan) -> Result<[u64; 3], String> {
    let [target, head, vision] = load.weight_demands()?;
    let bytes = |demand: ClaimDemand| {
        demand
            .checked_bytes()
            .and_then(|bytes| bytes.checked_sub(demand.host))
            .ok_or_else(|| "resident weight byte count overflow".to_owned())
    };
    Ok([bytes(target)?, bytes(head)?, bytes(vision)?])
}

/// Each weight of `load` whose storage it holds, once (tied roles share one
/// storage, `WeightPlan::storage_identity`), with its component's index
/// (target, head, vision), in the order the load imports them.
fn weight_storages(load: &ModelLoadPlan) -> Result<Vec<(usize, &WeightPlan)>, String> {
    let mut seen = HashMap::new();
    let mut stored = Vec::new();
    for (index, weights) in [
        load.target.as_slice(),
        load.head.as_deref().unwrap_or_default(),
        load.vision.as_deref().unwrap_or_default(),
    ]
    .into_iter()
    .enumerate()
    {
        for weight in weights {
            let storage = (
                weight.upload,
                weight.shape.clone(),
                weight.source_bytes,
                weight.resident_bytes,
                weight.scale.clone(),
            );
            match seen.entry(weight.storage_identity()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(storage);
                    stored.push((index, weight));
                }
                std::collections::hash_map::Entry::Occupied(entry) if entry.get() != &storage => {
                    return Err("tied weight roles disagree on their physical storage".into());
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
            }
        }
    }
    Ok(stored)
}

fn weight_demands_by_component(load: &ModelLoadPlan) -> Result<[ClaimDemand; 3], String> {
    let mut demands = [ClaimDemand::default(); 3];
    for (index, weight) in weight_storages(load)? {
        demands[index] = demands[index]
            .checked_plus(load.weight_storage_demand(weight)?)
            .ok_or("resident weight byte count overflow")?;
    }
    // The bank's committed records are charged by its banked weights; what
    // it is charged beyond them is the target's weights' too (a cap bounds
    // every weight byte the plan holds).
    if let Some(bank) = &load.bank {
        demands[0] = demands[0]
            .checked_plus(bank.storage_beyond_records()?)
            .ok_or("resident weight byte count overflow")?;
    }
    Ok(demands)
}

impl ModelLoadPlan {
    pub fn target(&self) -> &[WeightPlan] {
        &self.target
    }

    pub fn head(&self) -> Option<&[WeightPlan]> {
        self.head.as_deref()
    }

    pub fn vision(&self) -> Option<&[WeightPlan]> {
        self.vision.as_deref()
    }

    /// The separate draft's fusion, where the target's features are its
    /// fused taps (the fusion planned with the target, and no residual
    /// conditioning): publishing them costs a step the fusion and the
    /// tapped blocks' writes, where an MTP head's feature is the step's own
    /// output row.
    pub(crate) fn draft_fusion(&self) -> Option<&WeightPlan> {
        let role = WeightRole { scope: WeightScope::Draft, kind: WeightKind::DraftFusion };
        self.residual_conditioning.is_none().then(|| self.target.iter().find(|weight| weight.role == role)).flatten()
    }

    /// Resolve the ordered checked-entry topology from the same weight facts
    /// used for import and residency.
    pub(crate) fn program_plan(
        &self,
        definition: &ModelDefinition,
        history: KvCodec,
    ) -> Result<ProgramPlan, PlanError> {
        // Bindings name an output projection whether or not it is placed.
        let planned = |placed: &[WeightPlan], head: bool| {
            placed
                .iter()
                .chain(
                    self.unplaced_output
                        .iter()
                        .filter(|weight| head_component_scope(weight.role.scope) == Ok(head)),
                )
                .cloned()
                .collect::<Vec<_>>()
        };
        let target = planned(&self.target, false);
        let head = self.head.as_deref().map(|head| planned(head, true));
        super::programs::derive_program_plan(
            definition,
            &target,
            head.as_deref(),
            self.vision.as_deref(),
            &self.host_tables,
            &self.readout_choices,
            self.bank.as_ref(),
            history,
        )
    }

    /// Plan every selected weight of a load of `limits` drafting by
    /// `method`. Packed weights become resident in `layout`
    /// (`resident_layout` of the engine's execution path and device
    /// backend); streamed tables stage for the load's flights and land the
    /// rows its reads of selected rows name (`selected_read_rows`).
    pub fn derive(
        manifest: &PackageManifest,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        layout: Layout,
        method: super::PlannedMethod,
        limits: super::ResourceLimits,
    ) -> Result<Self, String> {
        Self::derive_components(
            manifest.identity,
            &manifest.target.tensors,
            manifest
                .projector
                .as_ref()
                .map(|projector| (projector.identity, projector.tensors.as_slice())),
            manifest
                .draft
                .as_ref()
                .map(|draft| (draft.identity, draft.tensors.as_slice())),
            definition,
            selection,
            layout,
            method,
            limits,
        )
    }

    /// Plan resident representations from a pre-download header bundle.
    /// The resulting plan describes bytes and formats but cannot import
    /// weights until a payload-backed `Package` is opened separately.
    pub fn derive_headers(
        headers: &PackageHeaders,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        layout: Layout,
        method: super::PlannedMethod,
        limits: super::ResourceLimits,
    ) -> Result<Self, String> {
        let identity = headers.identity();
        let projector = headers
            .projector()
            .map(|directory| {
                identity
                    .projector
                    .map(|component| (component, directory.tensors.as_slice()))
                    .ok_or_else(|| "projector header has no component identity".to_owned())
            })
            .transpose()?;
        let manifest = headers.manifest();
        Self::derive_components(
            identity,
            &headers.target().tensors,
            projector,
            manifest
                .draft
                .as_ref()
                .map(|draft| (draft.identity, draft.tensors.as_slice())),
            definition,
            selection,
            layout,
            method,
            limits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn derive_components(
        identity: PackageIdentity,
        target_tensors: &[TensorDescriptor],
        projector: Option<(ArtifactIdentity, &[TensorDescriptor])>,
        draft: Option<(ArtifactIdentity, &[TensorDescriptor])>,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        layout: Layout,
        method: super::PlannedMethod,
        limits: super::ResourceLimits,
    ) -> Result<Self, String> {
        definition.validate().map_err(|error| error.to_string())?;
        if identity != definition.artifact_identity {
            return Err("load plan package identity mismatch".into());
        }
        // Every lane of the load may name a streamed table's rows: the
        // target's, and a drafter head's, whose embedding may be the
        // target's.
        let reads = TableReads {
            flights: TableFlights { flights: limits.flights(), lanes: 1 + usize::from(selection.head) },
            selected_rows: selected_read_rows(definition, method, limits)?,
        };
        let target_component = ArtifactComponent {
            kind: ArtifactComponentKind::Target,
            identity: identity.target,
        };
        let decoder = &definition.decoder;
        let target_form = ResidentForm {
            activation: activation_dtype(decoder.activation_dtype),
            layout,
        };
        let target_role = |kind| WeightRole {
            scope: WeightScope::Target,
            kind,
        };
        let mut target = Vec::new();
        // A per-layer entry projects the embedding on the device; its table
        // is a host table.
        let per_layer_weights = decoder.entry.per_layer.iter().flat_map(|entry| {
            [
                (
                    target_role(WeightKind::PerLayerModelProjection),
                    &entry.projection,
                ),
                (
                    target_role(WeightKind::PerLayerProjectionNorm),
                    &entry.projection_norm.weight,
                ),
            ]
        });
        // The per-layer entry's table is a host table.
        let host_tables = decoder
            .entry
            .per_layer
            .iter()
            .map(|entry| (target_role(WeightKind::PerLayerTable), &entry.table))
            .map(|(role, descriptor)| {
                host_table(
                    target_tensors,
                    target_component,
                    role,
                    descriptor,
                    target_form,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        push_weights(
            &mut target,
            target_tensors,
            target_component,
            std::iter::once((target_role(WeightKind::Embedding), &decoder.entry.embedding))
                .chain(per_layer_weights)
                .chain(operators::decoder_weights(decoder))
                .chain(operators::exit_weights(decoder))
                .chain([(target_role(WeightKind::Output), &decoder.exit.output)])
                .collect(),
            target_form,
            reads,
        )?;

        // The executed definition carries at most one drafter.
        if selection.head && definition.head.is_some() == definition.draft.is_some() {
            return Err(
                "head execution was selected without exactly one drafter definition".into(),
            );
        }
        if definition.vision.is_some() && projector.is_none() {
            return Err("vision definition requires a projector component".into());
        }
        // An optional draft or projector can remain in the opened package
        // after semantic admission rejected it. Only an admitted definition
        // requires its component; an unbound artifact contributes no weights.
        // A draft head may be stored in the draft component too.
        let separate_head = definition
            .head
            .as_ref()
            .is_some_and(|head| head.component == HeadComponent::Draft);
        if (definition.draft.is_some() || separate_head) && draft.is_none() {
            return Err("draft component and draft definition disagree".into());
        }
        if selection.vision && definition.vision.is_none() {
            return Err("vision execution was selected without a vision definition".into());
        }
        let push_all = |plans: &mut Vec<WeightPlan>,
                        tensors: &[TensorDescriptor],
                        component: ArtifactComponent,
                        weights: Vec<(WeightRole, &WeightDescriptor)>| {
            push_weights(plans, tensors, component, weights, target_form, reads)
        };
        let head = match (selection.head, &definition.head, &definition.draft, draft) {
            (false, ..) => None,
            (true, Some(head), None, draft) => {
                let (tensors, component) = match (head.component, draft) {
                    (HeadComponent::Target, _) => (target_tensors, target_component),
                    (HeadComponent::Draft, Some((identity, tensors))) => (
                        tensors,
                        ArtifactComponent {
                            kind: ArtifactComponentKind::Draft,
                            identity,
                        },
                    ),
                    (HeadComponent::Draft, None) => {
                        return Err("the selected drafter has no component".into());
                    }
                };
                let mut plans = Vec::new();
                push_all(
                    &mut plans,
                    tensors,
                    component,
                    operators::head_weights(head).map_err(|error| error.to_string())?,
                )?;
                Some(plans)
            }
            (true, None, Some(separate), Some((identity, tensors))) => {
                let component = ArtifactComponent {
                    kind: ArtifactComponentKind::Draft,
                    identity,
                };
                // The fusion is read by every target step, so it is resident
                // with the target; the drafter's own weights load with it.
                push_all(
                    &mut target,
                    tensors,
                    component,
                    operators::draft::fusion_weights(separate).to_vec(),
                )?;
                let mut plans = Vec::new();
                push_all(
                    &mut plans,
                    tensors,
                    component,
                    operators::draft::draft_weights(separate).map_err(|error| error.to_string())?,
                )?;
                Some(plans)
            }
            _ => return Err("the selected drafter has no component".into()),
        };

        let vision = selection
            .vision
            .then_some(definition.vision.as_ref())
            .flatten()
            .map(|vision| {
                let (projector_identity, projector_tensors) =
                    projector.ok_or("vision definition requires a projector component")?;
                let component = ArtifactComponent {
                    kind: ArtifactComponentKind::Projector,
                    identity: projector_identity,
                };
                plan_vision(projector_tensors, component, vision, layout)
            })
            .transpose()?;
        validate_unique_roles(
            target
                .iter()
                .chain(head.as_deref().unwrap_or_default())
                .chain(vision.as_deref().unwrap_or_default()),
        )?;

        Ok(Self {
            residual_conditioning: definition
                .head
                .as_ref()
                .filter(|head| selection.head && head.residual_conditioning())
                .map(|head| head.conditioning_width(&definition.decoder)),
            target,
            head,
            vision,
            host_tables,
            readout_candidates: Vec::new(),
            readout_choices: Vec::new(),
            unplaced_output: Vec::new(),
            bank: None,
            record_ports: None,
        })
    }

    /// The load as `backend`'s native import entries import it: through
    /// record ports, whose tables its weights' storage holds where the
    /// backend binds them.
    pub fn with_record_ports(self, backend: seismic::BackendName) -> Self {
        Self {
            record_ports: Some(backend),
            ..self
        }
    }

    /// Every byte `weight`, one of this load's, holds: its allocations
    /// (`WeightPlan::storage_bytes`) and the record table its storage holds
    /// (Seismic's rule, `seismic::record_table_bytes`).
    pub fn weight_storage_bytes(&self, weight: &WeightPlan) -> Result<u64, String> {
        self.weight_storage_demand(weight)?.checked_bytes()
            .ok_or_else(|| "resident weight byte count overflows".to_owned())
    }

    /// A weight's physical demand ([`Self::weight_storage_bytes`]) by where
    /// each of its allocations lies: its device storage is
    /// [`Self::weight_device_allocations`].
    fn weight_storage_demand(&self, weight: &WeightPlan) -> Result<ClaimDemand, String> {
        let bytes = weight.storage_bytes()?;
        let device = self
            .weight_device_allocations(weight)?
            .into_iter()
            .try_fold(0u64, u64::checked_add)
            .ok_or("device weight storage overflows")?;
        if weight.banked().is_some() {
            return Ok(ClaimDemand { reserved: weight.resident_bytes, device, ..ClaimDemand::default() });
        }
        if weight.stored_in_slots() {
            if let Some((bank, table)) = self.bank_table(weight) {
                // StoredTable::new commits the complete reservation at load,
                // not its floor. Its own table replaces the ordinary table.
                let committed = table.record_bytes.get().checked_mul(table.records.get())
                    .ok_or("stored table backing overflows")?;
                let charges = bank.reservation().charges(bank.host().contract(crate::planning::BankTier::Bank), seismic::ReservedRecords::of(table.record_bytes), table.records.get(), committed)
                    .ok_or("stored table demand overflows")?;
                let mut demand = ClaimDemand::reservation(crate::planning::BankTier::Bank, charges);
                demand.shared = demand.shared.checked_add(bytes.checked_sub(weight.resident_bytes)
                    .ok_or("stored table directory charge underflows")?).ok_or("stored table demand overflows")?;
                return Ok(demand);
            }
            return Ok(ClaimDemand { shared: bytes, device, ..ClaimDemand::default() });
        }
        // DeviceTable owns ordinary slots and regular staging, but its
        // directory, landed-read staging and notice are host-shared.
        let [directory, _, landed, notice] = weight.streamed_bytes()?;
        let shared = directory.checked_add(landed).and_then(|bytes| bytes.checked_add(notice))
            .ok_or("streamed table shared storage overflows")?;
        Ok(ClaimDemand { device, shared, ..ClaimDemand::default() })
    }

    /// The bank's own table of `weight`, a table whose slots the host fills
    /// (`WeightPlan::stored_in_slots`), where the bank holds it.
    fn bank_table(&self, weight: &WeightPlan) -> Option<(&super::ExpertBankPlan, &super::bank::BankTable)> {
        self.bank.as_ref().and_then(|bank| {
            bank.tables.iter().find(|table| table.role == weight.role).map(|table| (bank, table))
        })
    }

    /// The device storage `weight`, one of this load's, allocates from the
    /// device's own memory (a dedicated device's blocks,
    /// `seismic::device_block_packing`), each allocation in the order its
    /// import makes it: its resident tensor, packing bound and scale, the
    /// record table its import binds (Seismic's rule,
    /// `seismic::record_table_bytes`; ordinary record tables use
    /// `DeviceInner::allocate` on every backend, independently of the
    /// tensor's own placement) and a streamed table's staging. A banked
    /// stack's records lie in the bank's reserved pages, and a stored
    /// table's slots and every table's directory, landing staging and
    /// notice in host-shared storage, outside these. Their sum is the
    /// weight's device demand.
    pub fn weight_device_allocations(&self, weight: &WeightPlan) -> Result<Vec<u64>, String> {
        let tables = match self.record_ports {
            Some(backend) if backend.binds_record_tables() => seismic::record_table_bytes(weight.record_table_entries()?)
                .ok_or_else(|| "record table byte count overflows".to_owned())?,
            Some(_) | None => 0,
        };
        let allocations = if weight.stored_in_slots() {
            // The bank's own table replaces the ordinary record table.
            vec![if self.bank_table(weight).is_some() { 0 } else { tables }]
        } else {
            vec![
                if weight.banked().is_some() { 0 } else { weight.resident_bytes },
                weight.bound.checked_mul(4).ok_or("packing bound byte count overflows")?,
                weight.scale.as_ref().map_or(0, |scale| scale.resident_bytes),
                tables,
                weight.streamed_bytes()?[1],
            ]
        };
        Ok(allocations.into_iter().filter(|bytes| *bytes > 0).collect())
    }

    /// Every device allocation of the load's weights
    /// ([`Self::weight_device_allocations`]) in the order the load imports
    /// them, each storage once: what a replay of the device's block
    /// allocator places.
    pub fn device_allocations(&self) -> Result<Vec<u64>, String> {
        let mut allocations = Vec::new();
        for (_, weight) in weight_storages(self)? {
            allocations.extend(self.weight_device_allocations(weight)?);
        }
        Ok(allocations)
    }

    /// Offer lossless bounded-selection views of the target's output
    /// projection where every reader of it can read them at no loss: a Q8_0
    /// projection without a soft cap. Indexed policies require the
    /// canonical table, so there the projection keeps its packed form, as
    /// does a head's own projection. A separate drafter that projects
    /// through the target's projection reads its full logits through the
    /// views' full pass, as every reader of a class the levels do not serve
    /// does. A view becomes resident, in place of the packed projection, only
    /// after whole-recipe class-cost qualification.
    pub fn with_readout_candidates(mut self, definition: &ModelDefinition) -> Result<Self, String> {
        let mut indexed = HashSet::new();
        if let Some(head) = &definition.head {
            for (index, block) in head.blocks.iter().enumerate() {
                if matches!(
                    &block.readout.policy,
                    magnitude_family_contracts::ProposalReadoutPolicy::Clustered { .. }
                ) {
                    let scope = match &block.readout.projection {
                        magnitude_family_contracts::VocabularyProjection::Target => {
                            WeightScope::Target
                        }
                        magnitude_family_contracts::VocabularyProjection::Own { .. } => {
                            WeightScope::HeadBlock(
                                u32::try_from(index).map_err(|_| "head block index exceeds u32")?,
                            )
                        }
                    };
                    indexed.insert(scope);
                }
            }
        }
        let select = |weights: &[WeightPlan]| -> Result<Vec<WeightPlan>, String> {
            let mut views = Vec::with_capacity(weights.len());
            for weight in weights {
                // The views' full pass serves every row the bounded levels
                // do not. It is measured against the packed projection's
                // only for 8-bit codes (Q8_0 projections) without a soft
                // cap, so only those are offered.
                let depth = ProgressiveCodeDepth::Eight;
                let admitted = weight.role.kind == WeightKind::Output
                    && weight.role.scope == WeightScope::Target
                    && definition.decoder.exit.softcap.is_none_or(|cap| cap == 0.0)
                    && !indexed.contains(&weight.role.scope)
                    && weight.scale.is_none()
                    && weight.descriptor.transforms.is_empty()
                    && weight.upload == weight.source;
                let placed = crate::import_transforms::progressive_matrix(
                    depth,
                    &weight.shape,
                    weight.source,
                )
                .is_some();
                if !(admitted && placed) {
                    continue;
                }
                for plane in ProgressivePlane::ALL {
                    let mut descriptor = WeightDescriptor {
                        transforms: vec![ImportTransform::Progressive { plane, depth }],
                        ..weight.descriptor.clone()
                    };
                    // The weight has no transforms: its shape is as stored.
                    let shape = crate::import_transforms::admit_transforms(
                        &descriptor,
                        &weight.shape,
                        weight.source,
                    )?;
                    descriptor.shape = shape.clone();
                    let element = crate::progressive::element(plane);
                    views.push(WeightPlan {
                        role: WeightRole {
                            scope: weight.role.scope,
                            kind: WeightKind::OutputPlane(plane),
                        },
                        component: weight.component,
                        source: weight.source,
                        upload: element,
                        resident: element,
                        source_bytes: representation_bytes(element, &shape)?,
                        resident_bytes: representation_bytes(element, &shape)?,
                        descriptor,
                        shape,
                        scale: None,
                        bound: 0,
                        holding: Holding::Whole,
                    });
                }
            }
            Ok(views)
        };
        self.readout_candidates = select(&self.target)?;
        Ok(self)
    }

    /// Commit the stored forms before resource admission. A scope with a
    /// reader's choice is stored as its views: they are
    /// placed, and its packed projection is planned and not placed
    /// (`unplaced_output`), so every reader of the scope reads the views.
    /// Unselected transient views are discarded.
    pub fn finalize_readout(mut self, choices: &[ReadoutClassChoice]) -> Result<Self, String> {
        if !self.readout_choices.is_empty() && self.readout_choices == choices {
            return Ok(self);
        }
        let mut scopes = HashSet::new();
        let mut recipes = HashSet::new();
        for choice in choices {
            if choice.rows.contains(&0) {
                return Err("a certified readout class needs nonzero selected rows".into());
            }
            if !recipes.insert((choice.scope, choice.norm_scope, choice.prepared)) {
                return Err("readout choices duplicate a semantic recipe".into());
            }
            scopes.insert(choice.scope);
            if !ProgressivePlane::ALL.into_iter().all(|plane| {
                self.readout_candidates.iter().any(|weight| {
                    weight.role
                        == WeightRole {
                            scope: choice.scope,
                            kind: WeightKind::OutputPlane(plane),
                        }
                })
            }) {
                return Err(
                    "readout choice has no complete qualified candidate representation".into(),
                );
            }
        }
        for candidate in std::mem::take(&mut self.readout_candidates) {
            if !scopes.contains(&candidate.role.scope) {
                continue;
            }
            if head_component_scope(candidate.role.scope)? {
                self.head
                    .as_mut()
                    .ok_or("qualified head representation has no head component")?
                    .push(candidate);
            } else {
                self.target.push(candidate);
            }
        }
        for scope in scopes {
            let output = WeightRole {
                scope,
                kind: WeightKind::Output,
            };
            let placed = if head_component_scope(scope)? {
                self.head
                    .as_mut()
                    .ok_or("qualified head representation has no head component")?
            } else {
                &mut self.target
            };
            let index = placed
                .iter()
                .position(|weight| weight.role == output)
                .ok_or("a selected readout scope has no output projection")?;
            self.unplaced_output.push(placed.remove(index));
        }
        self.readout_choices = choices.to_vec();
        Ok(self)
    }

    /// The output projection of `scope` as planned, placed or not.
    pub(crate) fn planned_output(&self, scope: WeightScope) -> Option<&WeightPlan> {
        let role = WeightRole {
            scope,
            kind: WeightKind::Output,
        };
        self.planned_weights().find(|weight| weight.role == role)
    }

    pub fn readout_choices(&self) -> &[ReadoutClassChoice] {
        &self.readout_choices
    }

    pub(crate) fn readout_candidates(&self) -> &[WeightPlan] {
        &self.readout_candidates
    }

    /// Every weight the plan names, for the kernels prepared over it: the
    /// placed weights, the output projections stored as planes (planned and
    /// not placed, which a finalized plan's packed readers still bind) and
    /// the optional views preparation may import under its transient
    /// budget. Final residency, assessment and ordinary graphs use only
    /// `weights()`.
    pub(crate) fn planned_weights(&self) -> impl Iterator<Item = &WeightPlan> {
        self.weights()
            .chain(self.unplaced_output.iter())
            .chain(self.readout_candidates.iter())
    }

    pub fn weights(&self) -> impl Iterator<Item = &WeightPlan> {
        self.target
            .iter()
            .chain(self.head.iter().flatten())
            .chain(self.vision.iter().flatten())
    }
}

pub(super) fn validate_unique_roles<'a>(
    plans: impl Iterator<Item = &'a WeightPlan>,
) -> Result<(), String> {
    let mut roles = HashSet::new();
    for plan in plans {
        if !roles.insert((plan.component, plan.role)) {
            return Err(format!(
                "load plan contains duplicate semantic role {:?}/{:?}",
                plan.component, plan.role
            ));
        }
    }
    Ok(())
}

/// The dense element dtype of a decoder's activations.
pub(crate) fn activation_dtype(dtype: ActivationDType) -> DType {
    match dtype {
        ActivationDType::F16 => DType::F16,
        ActivationDType::BF16 => DType::BF16,
    }
}

/// The per-layer entry (Gemma PLE, `EntryForm.per_layer`): the batch rows'
/// host-table rows (uploaded as `table_source`, converted to `table`) and
/// the embedding's projection to `layers · width` channels, combined by
/// `per_layer_inputs` into every block's per-layer input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PerLayerEntryBinding {
    pub hidden: u64,
    pub layers: u64,
    pub width: u64,
    pub table_source: Element,
    pub table: Element,
    pub projection: Element,
    pub norm: Element,
    pub activation: Element,
}

/// An n-gram input sublayer (`operators::ngram`): its rows per position
/// (`ngram_rows`), read from its stored table's slots, resident as `table`
/// (`stored_rows`), projected to the key and value rows, gated into the
/// streams (`ngram_gate`), then added (`ngram_add`) or convolved over its
/// window (`ngram_inject`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NGramBinding {
    pub hidden: u64,
    pub streams: u64,
    /// The row rule's form, which `ngram_rows` specializes on.
    pub rule: NGramRowRule,
    /// Each gathered row's width.
    pub width: u64,
    pub gate: NGramGateForm,
    /// The convolution's, when the input mixes its gated value.
    pub convolution: Option<NGramTaps>,
    pub table: Element,
    pub key: Element,
    pub value: Element,
    /// The gate's weights' and the convolution norm's.
    pub norm: Element,
    pub activation: Element,
}

/// The form of an n-gram input's row rule (`NGramHash`): the one owner of
/// `ngram_rows`'s statics, which every backend specializes on a subset of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NGramRowRule {
    /// Predecessors the rule reads (C): its orders are 2..=C + 1.
    pub context: u64,
    /// Rows gathered per order (P).
    pub heads_per_order: u64,
    /// The token map's entries (V), 0 where tokens hash as themselves.
    pub mapped: u64,
    /// Boundary tokens (NB): 1 where one cuts the lookback, else 0.
    pub boundaries: u64,
}

impl NGramRowRule {
    pub fn of(hash: &magnitude_family_contracts::NGramHash) -> Self {
        Self {
            context: hash.context,
            heads_per_order: hash.heads_per_order,
            mapped: hash.token_map.as_ref().map_or(0, |map| map.len() as u64),
            boundaries: u64::from(hash.boundary.is_some()),
        }
    }

    /// `ngram_rows`'s statics.
    pub fn statics(self) -> [(&'static str, u64); 4] {
        [
            ("C", self.context),
            ("P", self.heads_per_order),
            ("V", self.mapped),
            ("NB", self.boundaries),
        ]
    }
}

/// The weights `ngram_gate` weighs its product by: the key and query norms,
/// or their stored product alone (its `PR` form).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NGramGateForm {
    Norms,
    Product,
}

impl NGramGateForm {
    /// `ngram_gate`'s `PR`.
    pub fn product(self) -> u64 {
        match self {
            Self::Norms => 0,
            Self::Product => 1,
        }
    }
}

/// An n-gram convolution's taps (K) and their dilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NGramTaps {
    pub kernel: u64,
    pub dilation: u64,
}

impl NGramTaps {
    /// The window rows before a state: `dilation (kernel - 1)`.
    pub fn reach(self) -> u64 {
        self.dilation * (self.kernel - 1)
    }
}

impl NGramBinding {
    /// Rows gathered per position.
    pub fn heads(&self) -> u64 {
        self.rule.context * self.rule.heads_per_order
    }

    /// The width of a position's gathered rows.
    pub fn embedding(&self) -> u64 {
        self.heads() * self.width
    }

    /// The streams' channels, `S * H`.
    pub fn channels(&self) -> u64 {
        self.streams * self.hidden
    }

    /// Its bank components: under a convolution its window, then its token
    /// window.
    pub fn bank_components(&self) -> usize {
        1 + usize::from(self.convolution.is_some())
    }
}

/// A block's per-layer input sublayer: `per_layer_gate` over its layer's
/// slice, `project_rows` back to the hidden width, and its post-norm tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PerLayerBinding {
    pub hidden: u64,
    pub layers: u64,
    pub width: u64,
    pub gate: Element,
    pub projection: Element,
    pub activation: Element,
    /// Always a post-norm tail.
    pub tail: SublayerTail,
}

/// A host-resident gathered table (model-family plan §3.7): read by row
/// gather from its artifact and never a device weight. Each step uploads the
/// batch rows' `source` rows, which the graph converts to `resident` rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostTablePlan {
    pub role: WeightRole,
    pub component: ArtifactComponent,
    pub descriptor: WeightDescriptor,
    /// `[rows, columns]`.
    pub shape: Vec<u64>,
    pub source: Element,
    pub resident: Element,
    /// The table's bytes: its claim in the system-RAM domain.
    pub bytes: u64,
}

impl HostTablePlan {
    pub fn columns(&self) -> u64 {
        self.shape[1]
    }
}

/// Plan the host table `descriptor` names: a stored `[rows, columns]`
/// matrix without import transforms, in a source encoding whose rows the
/// backend layout can convert on the device.
fn host_table(
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    role: WeightRole,
    descriptor: &WeightDescriptor,
    form: ResidentForm,
) -> Result<HostTablePlan, String> {
    let stored = inventory
        .iter()
        .find(|tensor| tensor.name == descriptor.name)
        .ok_or_else(|| {
            format!(
                "host table {:?} is absent from its component",
                descriptor.name
            )
        })?;
    if !descriptor.transforms.is_empty() || stored.shape != descriptor.shape {
        return Err(format!(
            "host table {:?} must be the stored matrix itself",
            descriptor.name
        ));
    }
    let [_, width] = stored.shape[..] else {
        return Err(format!("host table {:?} is not a matrix", descriptor.name));
    };
    let source = source_element(stored.encoding, width)
        .ok_or_else(|| format!("unsupported source encoding {:?}", stored.encoding))?;
    let resident = resident_element(
        stored.encoding,
        crate::import_transforms::scaled_activations(descriptor),
        width,
        form.activation,
        form.layout,
    )
    .ok_or_else(|| {
        format!(
            "{:?} has no resident form in the `{}` layout",
            stored.encoding,
            form.layout.as_str()
        )
    })?;
    let bytes = representation_bytes(source, &stored.shape)?;
    if bytes != stored.nbytes {
        return Err(format!(
            "host table {:?} stores {} bytes, but its {:?} shape {:?} is {bytes}",
            descriptor.name, stored.nbytes, stored.encoding, stored.shape
        ));
    }
    Ok(HostTablePlan {
        role,
        component,
        descriptor: descriptor.clone(),
        shape: stored.shape.clone(),
        source,
        resident,
        bytes,
    })
}

/// Whether every reader of a weight of `kind` gathers it by row index: a
/// gathered table, which is streamed where it can be.
fn gathered_table(kind: WeightKind) -> bool {
    matches!(kind, WeightKind::Embedding | WeightKind::NGramTable)
}

/// Plan the weights of one component's roles. A gathered table whose stored
/// tensor no other of these roles reads is planned streamed
/// (`push_gathered_table`); a tied output projection reads the whole table.
fn push_weights(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    weights: Vec<(WeightRole, &WeightDescriptor)>,
    form: ResidentForm,
    reads: TableReads,
) -> Result<(), String> {
    for (role, descriptor) in &weights {
        if gathered_table(role.kind) {
            let read_whole = weights
                .iter()
                .any(|(other, stored)| other != role && stored.name == descriptor.name);
            push_gathered_table(
                out, inventory, component, *role, descriptor, form, read_whole, reads,
            )?;
        } else {
            push_weight(
                out, inventory, component, role.scope, role.kind, descriptor, form,
            )?;
        }
    }
    Ok(())
}

/// Plan a gathered table: a weight its readers gather by row index
/// (`device_table`). It is streamed, the device holding slots for
/// `STREAMED_TABLE_ROWS` of its rows, unless:
/// - a reader reads it whole (`read_whole`);
/// - it has no more rows than that;
/// - its rows are not whole tiles of the streamed layout;
/// - its rows do not reach the device as stored (the host prepares them, or
///   a second-level scale accompanies them).
///
/// Then it is planned as any weight is. An n-gram table is a stored table
/// instead (`push_stored_table`). A streamed table stages for the load's
/// flights and holds landing slots after its slots for the most rows one
/// of its reads of rows the device selects names (`StreamedRows::
/// landing_rows`), whole tiles of them (`reads`).
#[allow(clippy::too_many_arguments)]
fn push_gathered_table(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    role: WeightRole,
    descriptor: &WeightDescriptor,
    form: ResidentForm,
    read_whole: bool,
    reads: TableReads,
) -> Result<(), String> {
    if role.kind == WeightKind::NGramTable {
        return push_stored_table(out, inventory, component, role, descriptor, form);
    }
    let plan = |form| {
        let mut planned = Vec::with_capacity(1);
        push_weight(
            &mut planned,
            inventory,
            component,
            role.scope,
            role.kind,
            descriptor,
            form,
        )?;
        Ok::<_, String>(planned.remove(0))
    };
    let mut streamed = plan(ResidentForm {
        layout: streamed_layout(form.layout),
        ..form
    })?;
    let tile = streamed.resident.tile_rows();
    let slots = STREAMED_TABLE_ROWS - STREAMED_TABLE_ROWS % tile;
    let table_rows = streamed.shape[0];
    if read_whole
        || slots == 0
        || slots >= table_rows
        || table_rows % tile != 0
        || streamed.host_prepared()
        || streamed.placed_on_host()
        || streamed.scale.is_some()
    {
        out.push(plan(form)?);
        return Ok(());
    }
    streamed.shape[0] = slots;
    // A streamed table is read by row, never through a packed projection.
    streamed.bound = 0;
    // The load uploads the rows that fill the slots; nothing more of the
    // table is ever uploaded at once.
    streamed.source_bytes = representation_bytes(streamed.upload, &streamed.shape)?;
    let landing_rows = reads.selected_rows.div_ceil(tile) * tile;
    streamed.shape[0] = slots + landing_rows;
    streamed.resident_bytes = representation_bytes(streamed.resident, &streamed.shape)?;
    streamed.holding = Holding::Streamed(StreamedRows {
        table_rows,
        // The largest step any backend admits, each row of which may bring
        // its tile.
        staging_rows: slots.min(magnitude_batching::MAX_CLASS_ROWS as u64 * tile),
        landing_rows,
        directory: DirectoryForm::of(table_rows, slots),
        fill: TableFill::Imported(reads.flights),
    });
    out.push(streamed);
    Ok(())
}

/// What a load's streamed tables stage and land: the flights their stagings
/// serve, and the most rows one read of rows the device selects names
/// (`selected_read_rows`).
#[derive(Clone, Copy)]
struct TableReads {
    flights: TableFlights,
    selected_rows: u64,
}

/// The most table rows one read of rows the device selects names in a load
/// of `limits` drafting by `method`: a target step of selected tokens, in
/// the row class of the selection bound; an MTP head's chained pass, a
/// token per slot of a drafting class; a LiLiCorr corrector, its `top_k`
/// candidates of every mask row of every slot of a drafting class.
fn selected_read_rows(
    definition: &ModelDefinition,
    method: super::PlannedMethod,
    limits: super::ResourceLimits,
) -> Result<u64, String> {
    let class = |rows: usize| {
        magnitude_batching::row_class(rows)
            .map(|class| class as u64)
            .ok_or_else(|| format!("{rows} selected rows have no row class"))
    };
    let target = class(limits.max_selected_rows)?;
    let drafter = match (method, &definition.draft) {
        (super::PlannedMethod::Mtp { .. }, _) => class(limits.max_drafting_slots)?,
        (super::PlannedMethod::DFlash { .. }, Some(draft)) => match &draft.method {
            magnitude_family_contracts::DraftMethod::LiLiCorr { corrector } => {
                class(limits.max_drafting_slots)? * draft.max_proposals() * corrector.top_k
            }
            _ => 0,
        },
        _ => 0,
    };
    Ok(target.max(drafter))
}

/// The byte budget of an n-gram table's slots, in its rows as stored: the
/// rows its hit rate plateaus at (2.9 M; 253 MiB of IQ4_NL rows as stored).
/// The slots hold the rows resident, so they are the budget over a resident
/// row's bytes (`push_stored_table`).
pub const NGRAM_TABLE_STORED_ROWS: u64 = 180 * 16_384;

/// Plan an n-gram table (`TableFill::Stored`): resident as any weight is, in
/// the streamed layout (`streamed_layout`), in slots holding the bytes of
/// `NGRAM_TABLE_STORED_ROWS` of its rows as stored (all of them, of a table
/// of no more), which the host converts from the artifact's rows
/// (`stored_table::row_conversion`) and writes, and a directory of the form
/// the plan picks (`DirectoryForm::of`). A row as stored is a range of each
/// of its row tensors (`import_transforms::row_tensors`), in any of the
/// component's files, which the importer's own transforms assemble
/// (`stored_row_ranges`). A table with rows past I32, without a host row
/// conversion into its resident representation, or with a transform that
/// acts across rows is refused. The slots, `[records, rows, width]` (one
/// record of them here), are the weight's shape and resident bytes; the load
/// uploads nothing.
fn push_stored_table(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    role: WeightRole,
    descriptor: &WeightDescriptor,
    form: ResidentForm,
) -> Result<(), String> {
    let refused = |reason: &str| {
        format!(
            "n-gram table {:?} cannot be a stored table: {reason}",
            descriptor.name
        )
    };
    let stored = inventory
        .iter()
        .find(|tensor| tensor.name == descriptor.name)
        .ok_or_else(|| {
            format!(
                "n-gram table {:?} is absent from its component",
                descriptor.name
            )
        })?;
    let [table_rows, width] = stored.shape[..] else {
        return Err(refused("it is not a matrix"));
    };
    // Its own rows, each with its own row of block scales at most: each row
    // as stored is then its tensors' ranges of that row alone.
    let within_rows = match &descriptor.transforms[..] {
        [] => true,
        [ImportTransform::ScaleBlocks { tile, .. }] => tile[0] == 1,
        _ => false,
    };
    if !within_rows || stored.shape != descriptor.shape {
        return Err(refused("it is not the stored matrix itself, under a scale grid of its rows at most"));
    }
    if table_rows > i32::MAX as u64 {
        return Err(refused("its rows exceed I32"));
    }
    let mut planned = Vec::with_capacity(1);
    push_weight(
        &mut planned,
        inventory,
        component,
        role.scope,
        role.kind,
        descriptor,
        ResidentForm {
            layout: streamed_layout(form.layout),
            ..form
        },
    )?;
    let mut table = planned.remove(0);
    if crate::stored_table::row_conversion(table.upload, table.resident).is_none() {
        return Err(refused(&format!(
            "the host has no row conversion of {} into {}",
            table.upload.name(),
            table.resident.name()
        )));
    }
    let tensors = crate::import_transforms::row_tensors(descriptor)
        .into_iter()
        .map(|name| {
            inventory
                .iter()
                .find(|tensor| tensor.name == name)
                .map(|tensor| (tensor.name.as_str(), tensor.nbytes))
                .ok_or_else(|| refused(&format!("its row tensor {name:?} is absent from its component")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    stored_row_ranges(&tensors, table_rows, representation_bytes(table.upload, &[1, width])?).map_err(|error| refused(&error))?;
    let budget = representation_bytes(table.upload, &[NGRAM_TABLE_STORED_ROWS, width])?;
    let slots = (budget / representation_bytes(table.resident, &[1, width])?).min(table_rows);
    // One record of every slot; a bank holding the table records them in
    // the units it commits (`BankTable::of`).
    table.shape = vec![1, slots, width];
    table.resident_bytes = representation_bytes(table.resident, &table.shape)?;
    table.source_bytes = 0;
    // A stored table is read by row, never through a packed projection.
    table.bound = 0;
    table.holding = Holding::Streamed(StreamedRows {
        table_rows,
        staging_rows: 0,
        landing_rows: 0,
        directory: DirectoryForm::of(table_rows, slots),
        fill: TableFill::Stored,
    });
    out.push(table);
    Ok(())
}

/// The bytes of each range of a stored table's row: `tensors`, its row
/// tensors (`import_transforms::row_tensors`) with their stored bytes, each
/// whole ranges of the table's `rows`, which together hold one row as
/// stored, `row_bytes` (the importer's assembly moves every byte, adding
/// none). The planner and the importer each take a table's ranges from this.
pub(crate) fn stored_row_ranges(tensors: &[(&str, u64)], rows: u64, row_bytes: u64) -> Result<Vec<u64>, String> {
    let ranges = tensors
        .iter()
        .map(|&(name, bytes)| match bytes % rows {
            0 => Ok(bytes / rows),
            _ => Err(format!("{name:?}'s {bytes} bytes are not {rows} rows")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ranges.iter().sum::<u64>() != row_bytes {
        return Err(format!("ranges of {ranges:?} bytes are not a row of {row_bytes} as stored"));
    }
    Ok(ranges)
}

fn push_weight(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    scope: WeightScope,
    kind: WeightKind,
    descriptor: &WeightDescriptor,
    form: ResidentForm,
) -> Result<(), String> {
    let stored = inventory
        .iter()
        .find(|tensor| tensor.name == descriptor.name)
        .ok_or_else(|| {
            format!(
                "planned weight {:?} is absent from its component",
                descriptor.name
            )
        })?;
    let width = stored.shape.last().copied().unwrap_or(1);
    let scales = crate::import_transforms::stored_scales(descriptor);
    let source = scaled_source_element(stored.encoding, scales, width).ok_or_else(|| {
        format!(
            "weight {:?}: unsupported source encoding {:?} with {scales:?} scales",
            descriptor.name, stored.encoding
        )
    })?;
    // The logical shape the importer's exact transforms make of the stored
    // tensor (the stored shape when there are none).
    let logical = crate::import_transforms::admit(descriptor, &stored.shape, source)?;
    let role = WeightRole { scope, kind };
    // Projector weights keep their stored dense element: the vision kernels
    // read f32, f16 and bf16 matrices and vectors directly, and converting
    // f16 matrices or f32 biases to the activation dtype only loses bits.
    let stored_dense = source.dtype();
    let dense_resident = match (role.kind, stored_dense) {
        (WeightKind::Vision(_), Some(dtype)) => dtype,
        _ => resident_dtype(role, form.activation),
    };
    if operators::is_fixed_dense_role(kind) && stored_dense.is_none() {
        return Err(format!(
            "fixed-f32 weight {scope:?}/{kind:?} cannot use packed source encoding {:?}",
            stored.encoding
        ));
    }
    // Packed projector weights become dense in the activation dtype: the GPU
    // vision kernels read dense weights only. The host dequantizes them
    // exactly to F32 and the device rounds once to the activation dtype.
    let (upload, resident) = match (role.kind, stored_dense) {
        (WeightKind::Vision(_), None) => (Element::f32(), Element::dense(form.activation)),
        _ => (
            source,
            held_storage(
                stored.encoding,
                scales,
                width,
                crate::import_transforms::scaled_activations(descriptor),
                dense_resident,
                form.layout,
            )
            .ok_or_else(|| {
                format!(
                    "weight {:?}: {} has no resident form in the `{}` layout",
                    descriptor.name,
                    source.name(),
                    form.layout.as_str()
                )
            })?,
        ),
    };
    // The bytes the tensor stores: its source's, or under a grid of block
    // scales its codes', one block of its encoding a value (the grid is the
    // binder's, `Tensors::companion_scale`).
    let source_bytes = match scales {
        GgufScales::Own => representation_bytes(source, &stored.shape)?,
        GgufScales::Grid(_) => {
            elements(&stored.shape)
                .and_then(|values| values.checked_mul(stored.encoding.block_bytes()))
                .ok_or("a weight's byte count overflows")?
        }
    };
    if source_bytes != stored.nbytes {
        return Err(format!(
            "weight {:?} ({role:?}) stores {} bytes, but its {:?} shape {:?} is {source_bytes}",
            descriptor.name, stored.nbytes, stored.encoding, stored.shape
        ));
    }
    let resident_bytes = representation_bytes(resident, &logical)?;
    let scale = weight_scale(inventory, descriptor, &logical)?;
    if scale.is_some() && upload != source {
        return Err(format!(
            "weight {:?} ({role:?}) is dequantized for dense kernels, which have no \
             accumulator-scale port for its second-level scale",
            descriptor.name
        ));
    }
    let mut plan = WeightPlan {
        role,
        component,
        source,
        upload,
        resident,
        shape: logical.clone(),
        descriptor: descriptor.clone(),
        // The bytes the importer uploads: the transformed source, in the
        // upload representation.
        source_bytes: representation_bytes(upload, &logical)?,
        resident_bytes,
        scale,
        bound: 0,
        holding: Holding::Whole,
    };
    // The import makes the bound: a weight the host places has none.
    if !plan.placed_on_host() {
        plan.bound = packing_bound(resident, &logical);
    }
    out.push(plan);
    Ok(())
}

/// The resident second-level scale of a weight whose transforms name one
/// (`import_transforms::admit` allows at most one): a stored F32 tensor of
/// one value, or one per matrix of a stacked `logical` weight.
fn weight_scale(
    inventory: &[TensorDescriptor],
    descriptor: &WeightDescriptor,
    logical: &[u64],
) -> Result<Option<WeightScalePlan>, String> {
    let Some(tensor) = crate::import_transforms::scale_tensor(descriptor) else {
        return Ok(None);
    };
    let stored = inventory
        .iter()
        .find(|stored| stored.name == tensor)
        .ok_or_else(|| {
            format!(
                "second-level scale {tensor:?} of {:?} is absent from its component",
                descriptor.name
            )
        })?;
    let matrices = match logical {
        [stack, _, _] => *stack,
        [_, _] => 1,
        _ => {
            return Err(format!(
                "second-level scale {tensor:?} scales {:?}, which is not a matrix or a stack of \
                 matrices",
                descriptor.name
            ));
        }
    };
    let [extent] = stored.shape[..] else {
        return Err(format!(
            "second-level scale {tensor:?} of {:?} has shape {:?}, not [1] or [{matrices}]",
            descriptor.name, stored.shape
        ));
    };
    if stored.encoding != Encoding::F32 || !(extent == 1 || extent == matrices) {
        return Err(format!(
            "second-level scale {tensor:?} of {:?} is {:?} {:?}, not F32 [1] or [{matrices}]",
            descriptor.name, stored.encoding, stored.shape
        ));
    }
    if representation_bytes(Element::f32(), &stored.shape)? != stored.nbytes {
        return Err(format!(
            "second-level scale {tensor:?} stores {} bytes for its shape {:?}",
            stored.nbytes, stored.shape
        ));
    }
    Ok(Some(WeightScalePlan {
        tensor: tensor.to_owned(),
        stored: extent,
        extent: matrices,
        resident_bytes: representation_bytes(Element::f32(), &[matrices])?,
    }))
}

/// The resident dense element of a role (`operators::resident_dtype`).
pub(super) fn resident_dtype(role: WeightRole, activation: DType) -> DType {
    operators::resident_dtype(role.kind, activation)
}

fn plan_vision(
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    vision: &VisionDescription,
    layout: Layout,
) -> Result<Vec<WeightPlan>, String> {
    let form = ResidentForm {
        activation: activation_dtype(vision.activation_dtype),
        layout,
    };
    let mut out = Vec::new();
    for (role, descriptor) in vision.weights() {
        push_weight(
            &mut out, inventory, component, role.scope, role.kind, descriptor, form,
        )?;
    }
    Ok(out)
}

/// The planned element of a role an entry binds without an accumulator-scale
/// port: a weight with a second-level scale is refused.
pub(super) fn planned_element(
    plans: &[WeightPlan],
    scope: WeightScope,
    kind: WeightKind,
) -> Result<Element, PlanError> {
    let weight = planned_scalable(plans, scope, kind)?;
    if weight.scale != 0 {
        return Err(PlanError::UnportedScale(WeightRole { scope, kind }));
    }
    Ok(weight.element)
}

/// The planned form of a role an entry binds with an accumulator-scale port.
pub(super) fn planned_readout(
    weights: &[WeightPlan],
    scope: WeightScope,
) -> Result<ReadoutHead, crate::PlanError> {
    let weight = planned_scalable(weights, scope, WeightKind::Output)?;
    Ok(ReadoutHead::Packed {
        weight: weight.element,
        weight_scale: weight.scale,
    })
}

pub(super) fn planned_readout_recipe(
    weights: &[WeightPlan],
    scope: WeightScope,
    choices: &[ReadoutClassChoice],
) -> Result<ReadoutHead, crate::PlanError> {
    let packed = planned_readout(weights, scope)?;
    // Every reader of a scope stored as views reads them.
    if !choices.iter().any(|choice| choice.scope == scope) {
        return Ok(packed);
    }
    if !ProgressivePlane::ALL.into_iter().all(|plane| {
        weights.iter().any(|weight| {
            weight.role
                == WeightRole {
                    scope,
                    kind: WeightKind::OutputPlane(plane),
                }
        })
    }) {
        return Err(PlanError::Topology(
            "qualified readout has no complete resident view",
        ));
    }
    let weight = packed.packed();
    Ok(ReadoutHead::Qualified {
        weight: weight.element,
        weight_scale: weight.scale,
    })
}

pub(super) fn planned_scalable(
    plans: &[WeightPlan],
    scope: WeightScope,
    kind: WeightKind,
) -> Result<ScalableWeight, PlanError> {
    plans
        .iter()
        .find(|plan| plan.role == WeightRole { scope, kind })
        .map(|plan| ScalableWeight {
            element: plan.resident,
            scale: plan.scale_extent(),
        })
        .ok_or_else(|| {
            PlanError::InvalidDefinition(format!("load plan is missing {scope:?}/{kind:?}"))
        })
}

/// How the weights of one component become resident: the activation dtype
/// dense weights follow, and the layout packed weights are stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidentForm {
    activation: DType,
    layout: Layout,
}

/// The layout of a streamed table's slots where whole resident weights use
/// `layout`: `rows16`, one row to a tile, which every native backend's
/// row-gathering kernels read besides its resident layout, so that a slot
/// holds the one row a step names and no other.
pub fn streamed_layout(layout: Layout) -> Layout {
    match layout {
        Layout::Packet => Layout::Packet,
        Layout::Rows16 | Layout::Rows8 | Layout::Mma16 | Layout::Rows32 => Layout::Rows16,
    }
}

/// The layout resident packed weights use for an execution path on a device
/// backend (spec S1/E7): native kernels read the backend's own
/// (`BackendName::resident_layout`), every planned (compiled) kernel the
/// `packet` layout.
pub fn resident_layout(path: ExecutionPath, backend: BackendName) -> Layout {
    match path {
        ExecutionPath::Native => backend.resident_layout(),
        ExecutionPath::Planned => Layout::Packet,
    }
}

/// The external storage of an artifact encoding whose rows hold `width`
/// values (the stored last axis), with its own scales
/// ([`scaled_source_element`]).
pub fn source_element(encoding: Encoding, width: u64) -> Option<Element> {
    scaled_source_element(encoding, GgufScales::Own, width)
}

/// The external storage of an artifact encoding whose rows hold `width`
/// values, its scales where `scales` says: its dense element, or the
/// registry's source of such rows (`None` when the registry has none, and
/// of a dense encoding under a scale grid).
pub fn scaled_source_element(encoding: Encoding, scales: GgufScales, width: u64) -> Option<Element> {
    let dense = |dtype| (scales == GgufScales::Own).then(|| Element::dense(dtype));
    match encoding {
        Encoding::F32 => dense(DType::F32),
        Encoding::F16 => dense(DType::F16),
        Encoding::BF16 => dense(DType::BF16),
        _ => Element::gguf_source(encoding, scales, width),
    }
}

/// The resident storage of an artifact encoding with its own scales: that
/// of its source (`source_element`, of rows of `width` values) by
/// [`resident_storage`].
pub fn resident_element(
    encoding: Encoding,
    activations: ScaledActivations,
    width: u64,
    dense: DType,
    layout: Layout,
) -> Option<Element> {
    held_storage(
        encoding,
        GgufScales::Own,
        width,
        activations,
        dense,
        layout,
    )
}

/// The resident storage weight planning holds a stored weight of
/// `encoding` in, its scales where `scales` says, in rows of `width`
/// values: its source ([`scaled_source_element`]), then that source's
/// resident storage ([`resident_storage`]) for products taking
/// `activations` in `layout`. `None` when it has either none. The one
/// classification weight planning (`push_weight`) and model admission (the
/// engine's assessment) apply.
pub fn held_storage(
    encoding: Encoding,
    scales: GgufScales,
    width: u64,
    activations: ScaledActivations,
    dense: DType,
    layout: Layout,
) -> Option<Element> {
    resident_storage(scaled_source_element(encoding, scales, width)?, activations, dense, layout)
}

/// The tensors of a component (`tensors`) weight planning reads as weights,
/// each with where its scales are: under its grid of block scales where it
/// is read under one ([`ImportTransform::scale_grid`], the rule the binder
/// binds it by, `Tensors::companion_scale`, which also checks the grid),
/// of the codes its stored type names (`ScaleCodes::of_stored`, as the
/// binder reads them), else its own; `None` for a weight whose grid is
/// absent or of a type that names no codes, whose scales are read nowhere.
/// A grid is read only with its weight, so it is not among them.
pub fn stored_weights(
    tensors: &[TensorDescriptor],
) -> impl Iterator<Item = (&TensorDescriptor, Option<GgufScales>)> {
    let grid = |tensor: &TensorDescriptor| ImportTransform::scale_grid(&tensor.name, tensor.encoding);
    let grids = tensors.iter().filter_map(grid).collect::<HashSet<_>>();
    tensors
        .iter()
        .filter(move |tensor| !grids.contains(&tensor.name))
        .map(move |tensor| {
            let scales = match grid(tensor) {
                Some(name) => tensors
                    .iter()
                    .find(|stored| stored.name == name)
                    .and_then(|stored| ScaleCodes::of_stored(stored.encoding))
                    .map(|codes| crate::import_transforms::scales_of(Some(codes))),
                None => Some(GgufScales::Own),
            };
            (tensor, scales)
        })
}

/// The packet storage holding the values alone of a matrix stored as
/// `source` (the activations a product takes do not change them): its
/// resident storage for rounded activations, decoding to F32, in the packet
/// layout, where a row's packets are consecutive bytes.
pub(crate) fn value_packets(source: Element) -> Option<Element> {
    resident_storage(
        source,
        ScaledActivations::Rounded,
        DType::F32,
        Layout::Packet,
    )
}

/// The one map from a stored source element to resident storage: dense
/// sources become `dense`; packed sources become what the registry's
/// conversion of the source writes in `layout` (from the device backend)
/// for the activations the weight's producer calibrated it for
/// (`ScaledActivations`). NVFP4's per-tensor F32 `.scale` is a separate
/// tensor; its W4A4 storage alone takes NVFP4 operands.
pub(crate) fn resident_storage(
    source: Element,
    activations: ScaledActivations,
    dense: DType,
    layout: Layout,
) -> Option<Element> {
    if source.dtype().is_some() {
        return (activations == ScaledActivations::Rounded).then_some(Element::dense(dense));
    }
    let operands = match activations {
        ScaledActivations::Rounded => OperandRecipe::Rounded,
        ScaledActivations::Nvfp4 => OperandRecipe::Nvfp4Rows,
    };
    source.converted(layout, operands)
}

/// Every element a weight can be resident as in `layout`: the image of
/// [`resident_storage`] over every encoding with its scales its own or a
/// grid's, activations and dense dtype, and both IQ4_NL sources (rows whose
/// width is a multiple of 256, and rows whose width is not), each once, in
/// a fixed order (the encodings with their own scales first).
pub fn resident_elements(layout: Layout) -> Vec<Element> {
    let mut elements = Vec::new();
    for scales in GgufScales::ALL {
        for dense in [DType::BF16, DType::F16, DType::F32] {
            for encoding in Encoding::ALL {
                for activations in [ScaledActivations::Rounded, ScaledActivations::Nvfp4] {
                    for width in [256, 32] {
                        let resident = held_storage(
                            encoding,
                            scales,
                            width,
                            activations,
                            dense,
                            layout,
                        );
                        match resident {
                            Some(element) if !elements.contains(&element) => elements.push(element),
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    elements
}

pub(super) fn representation_bytes(element: Element, shape: &[u64]) -> Result<u64, String> {
    element
        .canonical_byte_len(shape)
        .map_err(|error| format!("{} shape {shape:?}: {error}", element.name()))
}

#[cfg(test)]
mod host_peak_tests {
    use super::*;

    /// The staged weight and the bound graph instances are never held
    /// together (graphs are bound after the import), so the peak takes the
    /// larger with the tables and the sealed plans, not their sum.
    #[test]
    fn the_planned_host_peak_is_its_terms_as_the_load_holds_them() {
        let peak = |staging, bound_graphs| HostPeak {
            tables: 100,
            staging,
            sealed_graphs: 500,
            bound_graphs,
        };
        assert_eq!(peak(300, 200).bytes().unwrap(), 100 + 500 + 300);
        assert_eq!(peak(0, 200).bytes().unwrap(), 100 + 500 + 200);
        // The tables are the system's cache of the artifact, not the
        // process's memory.
        assert_eq!(peak(300, 200).process_bytes().unwrap(), 500 + 300);
        assert!(
            HostPeak {
                tables: u64::MAX,
                ..peak(1, 1)
            }
            .bytes()
            .is_err()
        );
    }
}

#[cfg(test)]
mod representation_byte_tests {
    use super::*;

    #[test]
    fn packed_residency_charges_each_row_and_packet_alignment() {
        let q4 = Element::named("q4k").unwrap();
        // Two half-packet rows occupy two packets, even though their total
        // logical element count is only one full packet.
        assert_eq!(representation_bytes(q4, &[2, 128]).unwrap(), 288);

        let q8 = Element::named("q8g32s").unwrap();
        // The resident packet has aligned planes, unlike the 34-byte GGUF
        // source packet.
        assert_eq!(representation_bytes(q8, &[2, 32]).unwrap(), 72);

        let q6 = Element::named("q6k").unwrap();
        assert_eq!(representation_bytes(q6, &[1, 256]).unwrap(), 212);
        assert_eq!(representation_bytes(Element::f16(), &[2, 3]).unwrap(), 12);
    }

    #[test]
    fn row_layouts_pad_rows_to_sixteen_bytes_and_mma16_rows_to_tiles() {
        // Qwen3.5 K = 2560, q4k: codes 1280 | scales 120 -> 128 | supers 40 -> 48.
        let rows16 = Element::stored("q4k", Layout::Rows16).unwrap();
        assert_eq!(representation_bytes(rows16, &[3, 2560]).unwrap(), 3 * 1456);
        let mma16 = Element::stored("q4k", Layout::Mma16).unwrap();
        assert_eq!(representation_bytes(mma16, &[17, 2560]).unwrap(), 32 * 1456);
        // A q6k row of one group: codes 128 | 64 | scales 16 | supers 2 -> 16.
        let q6 = Element::stored("q6k", Layout::Rows16).unwrap();
        assert_eq!(representation_bytes(q6, &[1, 256]).unwrap(), 224);
    }

    /// The registry owns the encoding map: every GGUF source it converts
    /// from is an artifact encoding's, its scales its own or a grid's, and
    /// every encoding with a source is resident in every layout, a dense one
    /// in the activation dtype and a packed one as its conversion writes it,
    /// in W4A4's NVFP4-operand element only where the conversion has one
    /// (sharing its storage).
    #[test]
    fn the_registry_owns_the_encoding_map() {
        let sources = || {
            Encoding::ALL.into_iter().flat_map(|encoding| {
                GgufScales::ALL.into_iter().flat_map(move |scales| {
                    [256, 32].map(|width| scaled_source_element(encoding, scales, width))
                })
            })
        };
        let reached: HashSet<Element> = sources().flatten().collect();
        for source in Element::external_sources() {
            assert!(reached.contains(&source), "{} is no encoding's source", source.name());
        }
        for source in sources().flatten() {
            for layout in Layout::ALL {
                let resident =
                    |activations| resident_storage(source, activations, DType::BF16, layout);
                let rounded = resident(ScaledActivations::Rounded).unwrap();
                let w4a4 = resident(ScaledActivations::Nvfp4);
                if source.dtype().is_some() {
                    assert_eq!((rounded, w4a4), (Element::bf16(), None));
                    continue;
                }
                assert_eq!(rounded.layout(), layout);
                assert_eq!(rounded.operand_recipe(), OperandRecipe::Rounded);
                if let Some(w4a4) = w4a4 {
                    assert_eq!(w4a4.operand_recipe(), OperandRecipe::Nvfp4Rows);
                    assert_eq!(
                        (w4a4.representation(), w4a4.layout()),
                        (rounded.representation(), layout)
                    );
                }
            }
        }
    }

    /// IQ4_NL rows of every width (a 2560-wide projection, a 640-wide
    /// expert down, a 160-wide table) read single blocks into iq4g32s, whose
    /// bytes are the stored blocks'.
    #[test]
    fn iq4_nl_reads_iq4g32s_at_every_row_width() {
        for (width, source, representation) in [
            (2560, "gguf_iq4_nl", "iq4g32s"),
            (640, "gguf_iq4_nl", "iq4g32s"),
            (160, "gguf_iq4_nl", "iq4g32s"),
        ] {
            let stored = source_element(Encoding::Iq4Nl, width).unwrap();
            assert_eq!(stored.name(), source);
            // 18 bytes per 32 stored values, whatever the packet.
            assert_eq!(
                stored.canonical_byte_len(&[2, width]).unwrap(),
                2 * width / 32 * 18
            );
            for layout in Layout::ALL {
                let element = resident_element(
                    Encoding::Iq4Nl,
                    ScaledActivations::Rounded,
                    width,
                    DType::BF16,
                    layout,
                )
                .unwrap();
                assert_eq!(
                    (element.representation(), element.layout()),
                    (representation, layout)
                );
            }
        }
    }
}

#[cfg(test)]
mod stored_table_tests {
    use super::*;
    use magnitude_family_contracts::SublayerIndex;

    #[test]
    fn imported_table_shared_state_uses_upload_but_slots_and_record_table_use_device() {
        let definition = crate::planning::fixtures::fixture_definition();
        let manifest = crate::planning::fixtures::fixture_manifest(&definition);
        let limits = crate::ResourceLimits { max_launch_rows: 2, max_launch_slots: 2, max_selected_rows: 2,
            max_drafting_slots: 2, exported_logits_rows: 0, max_images_per_request: 1, max_image_cells: 0,
            lookahead: false, chain_capacity: std::num::NonZeroUsize::MIN, conversations: std::num::NonZeroUsize::MIN };
        let mut load = ModelLoadPlan::derive(&manifest, &definition,
            ComponentSelection { head: false, vision: false }, Layout::Rows16, crate::PlannedMethod::Plain, limits).unwrap();
        let rows = STREAMED_TABLE_ROWS * 2;
        let descriptor = WeightDescriptor::stored("gathered", [rows, 128]);
        let inventory = [TensorDescriptor { name: descriptor.name.clone(), shape: vec![rows, 128],
            encoding: Encoding::F16, offset: 0, nbytes: rows * 128 * 2 }];
        let component = load.target[0].component;
        load.target.clear();
        push_gathered_table(&mut load.target, &inventory, component,
            WeightRole { scope: WeightScope::Target, kind: WeightKind::Embedding }, &descriptor,
            ResidentForm { activation: DType::F16, layout: Layout::Rows16 }, false, TableReads {
                flights: TableFlights { flights: limits.flights(), lanes: 1 },
                selected_rows: selected_read_rows(&definition, super::super::PlannedMethod::Plain, limits).unwrap(),
            }).unwrap();
        let load = load.with_record_ports(BackendName::Vulkan);
        let table = &load.target[0];
        assert!(table.lands_selected_rows());
        let [directory, staged, landed, notice] = table.streamed_bytes().unwrap();
        let demand = load.weight_demands().unwrap()[0];
        assert_eq!(demand.shared, directory + landed + notice);
        assert_eq!(demand.device, table.resident_bytes + staged
            + seismic::record_table_bytes(table.record_table_entries().unwrap()).unwrap());
        assert_eq!(demand.checked_bytes().unwrap(), table.storage_bytes().unwrap()
            + seismic::record_table_bytes(table.record_table_entries().unwrap()).unwrap());
    }

    fn ngram_role() -> WeightRole {
        WeightRole {
            scope: WeightScope::TargetSublayer(SublayerIndex {
                block: 0,
                sublayer: 2,
            }),
            kind: WeightKind::NGramTable,
        }
    }

    /// Flash Next's n-gram table (320,001,536 rows 160 wide) in each
    /// 32-value block encoding a preset stores it in (160 is not a multiple
    /// of 256): resident in its rows16 form on every native backend, in as
    /// many slots as the bytes of `NGRAM_TABLE_STORED_ROWS` rows as stored
    /// hold (gguf-quant-unification.md §2, Memory). Q5_0 is Q4_K_M's.
    #[test]
    fn an_ngram_table_is_resident_in_the_bytes_of_its_stored_rows() {
        const ROWS: u64 = 320_001_536;
        let component = ArtifactComponent {
            kind: ArtifactComponentKind::Target,
            identity: ArtifactIdentity([0; 32]),
        };
        // (encoding, stored block bytes, resident, resident row bytes, slots)
        for (encoding, block, resident, row, slots) in [
            (Encoding::Iq4Nl, 18, "iq4g32s", 96, 2_764_800),
            (Encoding::Q8_0, 34, "q8g32s", 176, 2_848_581),
            (Encoding::Q4_0, 18, "q4g32s", 96, 2_764_800),
            (Encoding::Q5_0, 22, "q5g32s", 128, 2_534_400),
            (Encoding::Q5_1, 24, "q5g32", 144, 2_457_600),
            (Encoding::Mxfp4, 17, "mxfp4g32", 96, 2_611_200),
        ] {
            let stored_row = 5 * block;
            let inventory = [TensorDescriptor {
                name: "ngram".into(),
                shape: vec![ROWS, 160],
                encoding,
                offset: 0,
                nbytes: ROWS * stored_row,
            }];
            let descriptor = WeightDescriptor::stored("ngram", [ROWS, 160]);
            for backend in BackendName::ALL {
                let mut out = Vec::new();
                push_stored_table(
                    &mut out,
                    &inventory,
                    component,
                    ngram_role(),
                    &descriptor,
                    ResidentForm {
                        activation: DType::BF16,
                        layout: resident_layout(ExecutionPath::Native, backend),
                    },
                )
                .unwrap();
                let [table] = &out[..] else {
                    panic!("one plan per table")
                };
                assert_eq!(
                    table.resident,
                    Element::stored(resident, Layout::Rows16).unwrap()
                );
                assert_eq!(table.shape, vec![1, slots, 160], "{encoding:?}");
                assert_eq!(table.resident_bytes, slots * row, "{encoding:?}");
                let budget = NGRAM_TABLE_STORED_ROWS * stored_row;
                assert!(table.resident_bytes <= budget && budget < table.resident_bytes + row);
                assert_eq!(table.streamed().unwrap().fill, TableFill::Stored);
                assert_eq!(
                    representation_bytes(table.upload, &table.stored_shape()).unwrap(),
                    ROWS * stored_row,
                );
                assert_eq!(
                    super::super::ImportPlan::new(out.iter()).peak_bytes(),
                    0,
                );
            }
        }
    }

    /// A table without a host row conversion into its resident form (a
    /// dense one) is refused.
    #[test]
    fn an_ngram_table_without_a_host_row_conversion_is_refused() {
        let inventory = [TensorDescriptor {
            name: "ngram".into(),
            shape: vec![64, 160],
            encoding: Encoding::F16,
            offset: 0,
            nbytes: 64 * 160 * 2,
        }];
        let error = push_stored_table(
            &mut Vec::new(),
            &inventory,
            ArtifactComponent {
                kind: ArtifactComponentKind::Target,
                identity: ArtifactIdentity([0; 32]),
            },
            ngram_role(),
            &WeightDescriptor::stored("ngram", [64, 160]),
            ResidentForm {
                activation: DType::BF16,
                layout: Layout::Rows16,
            },
        )
        .unwrap_err();
        assert!(error.contains("no row conversion"), "{error}");
    }

    /// V4.1's n-gram table (384,006,168 E4M3 rows 256 wide under a `[1, 32]`
    /// UE8M0 grid, slice B's) plans as a stored table of the assembled
    /// 264-byte rows of mxfp8g32, each read as 256 bytes of codes and 8 of
    /// scales, in the bytes of `NGRAM_TABLE_STORED_ROWS` assembled rows. A
    /// grid that one row's scales do not tile alone, and a grid absent from
    /// the component, are refused, naming it.
    #[test]
    fn a_gridded_ngram_table_plans_its_rows_as_their_tensors_ranges() {
        use magnitude_family_contracts::ScaleCodes;
        const ROWS: u64 = 384_006_168;
        const WEIGHT: &str = "per_layer_token_embd.weight";
        const SCALE: &str = "per_layer_token_embd.scale";
        let tensor = |name: &str, columns, encoding| TensorDescriptor {
            name: name.into(),
            shape: vec![ROWS, columns],
            encoding,
            offset: 0,
            nbytes: ROWS * columns,
        };
        let inventory = [tensor(WEIGHT, 256, Encoding::F8E4M3), tensor(SCALE, 8, Encoding::I8)];
        let descriptor = |tile| WeightDescriptor {
            name: WEIGHT.into(),
            shape: vec![ROWS, 256],
            transforms: vec![ImportTransform::ScaleBlocks { tensor: SCALE.into(), tile, codes: ScaleCodes::Ue8m0 }],
        };
        let component = ArtifactComponent { kind: ArtifactComponentKind::Target, identity: ArtifactIdentity([0; 32]) };
        let plan = |inventory: &[TensorDescriptor], tile| {
            let mut out = Vec::new();
            let form = ResidentForm { activation: DType::BF16, layout: Layout::Rows16 };
            push_stored_table(&mut out, inventory, component, ngram_role(), &descriptor(tile), form).map(|()| out.remove(0))
        };
        let table = plan(&inventory, [1, 32]).unwrap();
        assert_eq!(table.upload, Element::gguf_source(Encoding::F8E4M3, GgufScales::Grid(seismic::GridScale::E8m0), 32).unwrap());
        assert_eq!(table.resident, Element::stored("mxfp8g32", Layout::Rows16).unwrap());
        assert_eq!(table.streamed().unwrap().fill, TableFill::Stored);
        let row = representation_bytes(table.upload, &[1, 256]).unwrap();
        assert_eq!(row, 264);
        assert_eq!(stored_row_ranges(&[(WEIGHT, ROWS * 256), (SCALE, ROWS * 8)], ROWS, row), Ok(vec![256, 8]));
        let slot = representation_bytes(table.resident, &[1, 256]).unwrap();
        let budget = NGRAM_TABLE_STORED_ROWS * row;
        assert_eq!(table.shape, vec![1, budget / slot, 256]);
        assert!(table.resident_bytes <= budget && budget < table.resident_bytes + slot);
        let error = plan(&inventory, [32, 32]).unwrap_err();
        assert!(error.contains(WEIGHT) && error.contains("scale grid of its rows"), "{error}");
        let error = plan(&inventory[..1], [1, 32]).unwrap_err();
        assert!(error.contains(SCALE), "{error}");
    }

    /// The qwen4exp fixture with its n-gram table stored as Q5_0 blocks (as
    /// a Q4_K_M preset stores Flash Next's) plans on every native backend,
    /// its n-gram input reading the table's resident rows16 slots.
    #[test]
    fn a_q5_0_ngram_table_plans() {
        let definition = crate::planning::fixtures::qwen4exp_fixture_definition();
        let mut manifest = crate::planning::fixtures::stored_table_manifest(&definition);
        let mut offset = 0;
        for tensor in &mut manifest.target.tensors {
            if tensor.encoding == Encoding::Iq4Nl {
                tensor.encoding = Encoding::Q5_0;
                tensor.nbytes = tensor.shape.iter().product::<u64>() / 32 * 22;
            }
            tensor.offset = offset;
            offset += tensor.nbytes;
        }
        manifest.target.files[0].size = offset;
        for backend in BackendName::ALL {
            let load = ModelLoadPlan::derive(
                &manifest,
                &definition,
                ComponentSelection {
                    head: false,
                    vision: false,
                },
                resident_layout(ExecutionPath::Native, backend),
                crate::PlannedMethod::Plain,
                crate::planning::fixtures::LIMITS,
            )
            .unwrap();
            let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            let bindings = plan
                .target()
                .blocks()
                .iter()
                .filter_map(|block| block.ngram())
                .collect::<Vec<_>>();
            assert!(!bindings.is_empty());
            for binding in bindings {
                assert_eq!(
                    binding.table,
                    Element::stored("q5g32s", Layout::Rows16).unwrap()
                );
            }
        }
    }
}

#[cfg(test)]
mod scale_grid_tests {
    use super::*;
    use crate::residency::Stored;
    use magnitude_family_contracts::ScaleCodes;
    use seismic::GridScale;
    use magnitude_artifacts::gguf::{GgufArtifact, StoredHeader};

    const WEIGHT: &str = "blk.0.attn_q_b.weight";
    const SCALE: &str = "blk.0.attn_q_b.scale";
    /// Two 32×32 tiles of rows, eight of columns: every row holds every
    /// E4M3 code once.
    const SHAPE: [u64; 2] = [64, 256];
    /// The tiles' UE8M0 scales: the smallest, 2^0, 2^127, NaN, and between.
    const SCALES: [u8; 16] = [0, 1, 2, 64, 100, 120, 126, 127, 128, 134, 140, 200, 250, 253, 254, 255];

    /// One F8_E4M3 weight of a test file: `<stem>.weight`, a `shape` tensor
    /// of code `at` (as a byte) at element `at`, under its `<stem>.scale`
    /// grid shaped `grid` of `codes`, stored as `scales`.
    #[derive(Clone, Copy)]
    struct Gridded<'a> {
        stem: &'a str,
        shape: &'a [u64],
        grid: &'a [u64],
        codes: ScaleCodes,
        scales: &'a [u8],
    }

    /// The UE8M0-gridded weight [`WEIGHT`]: an I8 32×32 grid of [`SCALES`]
    /// over [`SHAPE`].
    const UE8M0_WEIGHT: Gridded<'static> = Gridded {
        stem: "blk.0.attn_q_b",
        shape: &SHAPE,
        grid: &[SHAPE[0] / 32, SHAPE[1] / 32],
        codes: ScaleCodes::Ue8m0,
        scales: &SCALES,
    };

    /// A GGUF of `weights`, each with its grid, and no metadata (a grid's
    /// stored type names its codes); `name` tells concurrent tests' files
    /// apart.
    fn gguf(weights: &[Gridded], name: &str) -> std::path::PathBuf {
        let header = StoredHeader::new(
            3,
            Vec::new(),
            weights.iter().flat_map(|weight| {
                [
                    (format!("{}.weight", weight.stem), weight.shape.to_vec(), Encoding::F8E4M3),
                    (format!("{}.scale", weight.stem), weight.grid.to_vec(), weight.codes.stored()),
                ]
            }),
        )
        .unwrap();
        let mut bytes = header.encode().unwrap();
        let data = bytes.len();
        let payloads = weights.iter().flat_map(|weight| {
            let codes = (0..weight.shape.iter().product::<u64>()).map(|at| at as u8).collect::<Vec<_>>();
            [codes, weight.scales.to_vec()]
        });
        for (tensor, payload) in header.tensors.iter().zip(payloads) {
            bytes.resize(data + tensor.offset as usize, 0);
            bytes.extend_from_slice(&payload);
        }
        let path = std::env::temp_dir().join(format!(
            "magnitude-scale-grid-{name}-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    /// The E4M3 value of `code` (S.EEEE.MMM, bias 7, NaN at S.1111.111,
    /// no infinities), decoded independently of the registry.
    fn e4m3(code: u8) -> f64 {
        let (exponent, mantissa) = ((code >> 3) & 15, code & 7);
        let magnitude = match (exponent, mantissa) {
            (15, 7) => f64::NAN,
            (0, _) => f64::from(mantissa) * 2f64.powi(-9),
            _ => (1.0 + f64::from(mantissa) / 8.0) * 2f64.powi(i32::from(exponent) - 7),
        };
        if code & 0x80 != 0 { -magnitude } else { magnitude }
    }

    /// The F32 values [`UE8M0_WEIGHT`] imports to: each code's value times
    /// 2^(scale − 127) of its tile, exact in F64 and F32 alike, NaN under a
    /// NaN code or the NaN scale (255).
    fn ue8m0_reference() -> Vec<f32> {
        let columns = SHAPE[1] as usize;
        let tiles = columns / 32;
        (0..(SHAPE[0] * SHAPE[1]) as usize)
            .map(|at| match SCALES[at / columns / 32 * tiles + at % columns / 32] {
                255 => f32::NAN,
                scale => (e4m3(at as u8) * 2f64.powi(i32::from(scale) - 127)) as f32,
            })
            .collect()
    }

    /// Whether `values` are `expected` bit for bit, NaN for NaN.
    fn assert_bits(name: &str, values: &[f32], expected: &[f32]) {
        assert_eq!(values.len(), expected.len(), "{name}");
        for (at, (value, expected)) in values.iter().zip(expected).enumerate() {
            assert!(
                value.to_bits() == expected.to_bits() || value.is_nan() && expected.is_nan(),
                "{name} value {at}: code {:#04x}: {value:e}, not {expected:e}",
                at as u8
            );
        }
    }

    fn plan(
        inventory: &[TensorDescriptor],
        descriptor: &WeightDescriptor,
        layout: Layout,
    ) -> Result<WeightPlan, String> {
        let mut out = Vec::new();
        push_weight(
            &mut out,
            inventory,
            ArtifactComponent {
                kind: ArtifactComponentKind::Target,
                identity: ArtifactIdentity([0; 32]),
            },
            WeightScope::Target,
            WeightKind::Query,
            descriptor,
            ResidentForm {
                activation: DType::F16,
                layout,
            },
        )?;
        Ok(out.remove(0))
    }

    /// Binds `weight` (`<stem>.weight`) of the file at `path` and imports it
    /// in `layout` as its plan uploads it, with its values as the registry
    /// decodes the upload.
    fn import(path: &std::path::Path, layout: Layout, weight: &Gridded) -> Result<(WeightPlan, Vec<f32>), String> {
        let artifact = GgufArtifact::open(path).unwrap();
        let descriptor = magnitude_family_common::Tensors::new(artifact.directory())
            .bind_scaled(&format!("{}.weight", weight.stem), weight.shape)
            .unwrap();
        let plan = plan(&artifact.directory().tensors, &descriptor, layout).unwrap();
        let stored = Stored::from_gguf(&artifact, &descriptor).unwrap();
        let (source, offset, nbytes) = stored.file_range();
        let codes = source.read(offset, nbytes as usize).unwrap();
        let blocks = crate::import_transforms::upload_bytes(&plan, &stored, codes)?;
        let values = crate::import_transforms::dequantize(plan.source, weight.shape, &blocks).unwrap();
        let values = values.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect();
        Ok((plan, values))
    }

    /// The pair imports as mxfp8g32, and its values are exactly code ×
    /// 2^(scale − 127): E4M3 subnormals and −0 kept, NaN codes and the NaN
    /// scale (255) NaN, a product past F32's range infinite.
    #[test]
    fn an_f8_e4m3_weight_under_its_ue8m0_grid_imports_its_exact_values() {
        let path = gguf(&[UE8M0_WEIGHT], "ue8m0");
        let (plan, values) = import(&path, Layout::Rows8, &UE8M0_WEIGHT).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(plan.source, Element::gguf_source(Encoding::F8E4M3, GgufScales::Grid(GridScale::E8m0), 32).unwrap());
        assert_eq!(plan.upload, plan.source);
        assert_eq!(plan.resident, Element::stored("mxfp8g32", Layout::Rows8).unwrap());
        // The upload: one 33-byte block (scale, 32 codes) a row's 32 values.
        assert_eq!(plan.source_bytes, SHAPE[0] * SHAPE[1] / 32 * 33);
        assert_bits("ue8m0", &values, &ue8m0_reference());
    }

    fn gridded(tile: [u64; 2]) -> (WeightDescriptor, TensorDescriptor) {
        let descriptor = WeightDescriptor {
            name: WEIGHT.into(),
            shape: SHAPE.to_vec(),
            transforms: vec![ImportTransform::ScaleBlocks {
                tensor: SCALE.into(),
                tile,
                codes: ScaleCodes::Ue8m0,
            }],
        };
        let weight = TensorDescriptor {
            name: WEIGHT.into(),
            shape: SHAPE.to_vec(),
            encoding: Encoding::F8E4M3,
            offset: 0,
            nbytes: SHAPE[0] * SHAPE[1],
        };
        (descriptor, weight)
    }

    /// A component's weights are read with their scales: an F8_E4M3 matrix
    /// under its `.scale` grid, which is read only with it, of E8M0 scales
    /// stored as I8 and F32 ones as F32, and under none when its grid is
    /// absent or stored otherwise; an NVFP4
    /// weight's F32 `.scale` is a weight of its own, and so its weight's
    /// scales are its blocks'.
    #[test]
    fn stored_weights_read_a_grid_with_its_weight() {
        let tensor = |name: &str, encoding| TensorDescriptor {
            name: name.into(),
            shape: vec![64, 256],
            encoding,
            offset: 0,
            nbytes: 0,
        };
        let tensors = [
            tensor(WEIGHT, Encoding::F8E4M3),
            tensor(SCALE, Encoding::I8),
            tensor("blk.0.ffn_up.weight", Encoding::Nvfp4),
            tensor("blk.0.ffn_up.scale", Encoding::F32),
            tensor("blk.1.attn_q_b.weight", Encoding::F8E4M3),
            tensor("blk.1.attn_q_b.scale", Encoding::F32),
            tensor("blk.2.attn_q_b.weight", Encoding::F8E4M3),
            tensor("blk.2.attn_q_b.scale", Encoding::F16),
            tensor("blk.3.attn_q_b.weight", Encoding::F8E4M3),
        ];
        let read = stored_weights(&tensors)
            .map(|(tensor, scales)| (tensor.name.as_str(), scales))
            .collect::<Vec<_>>();
        assert_eq!(
            read,
            [
                (WEIGHT, Some(GgufScales::Grid(GridScale::E8m0))),
                ("blk.0.ffn_up.weight", Some(GgufScales::Own)),
                ("blk.0.ffn_up.scale", Some(GgufScales::Own)),
                ("blk.1.attn_q_b.weight", Some(GgufScales::Grid(GridScale::F32))),
                ("blk.2.attn_q_b.weight", None),
                ("blk.3.attn_q_b.weight", None),
            ]
        );
    }

    /// A tile whose columns are no whole packets of the source (32 values,
    /// one assembled block each) is refused, naming the weight and grid.
    #[test]
    fn a_tile_of_partial_source_packets_is_refused() {
        let (descriptor, weight) = gridded([32, 16]);
        let error = plan(&[weight], &descriptor, Layout::Rows8).unwrap_err();
        assert!(error.contains(WEIGHT) && error.contains(SCALE), "{error}");
    }

    /// The weight is planned resident, and measured, in every backend's
    /// layout.
    #[test]
    fn mxfp8_is_planned_and_measured_on_every_backend() {
        let (descriptor, weight) = gridded([32, 32]);
        for backend in BackendName::ALL {
            let layout = backend.resident_layout();
            let resident = Element::stored("mxfp8g32", layout).unwrap();
            let planned = plan(std::slice::from_ref(&weight), &descriptor, layout);
            assert_eq!(planned.unwrap().resident, resident, "{backend:?}");
            assert!(resident_elements(layout).contains(&resident), "{backend:?}");
        }
    }

    /// F32 tile scales of GLM's `weight_scale_inv` kind: amax / 448 values,
    /// 1, the least admitted (2^-117) and its negation, zero and −0, a
    /// value whose products overflow F32, and values whose products with
    /// most codes round.
    const F32_SCALES: [f32; 16] = [
        1.0,
        0.012_345_678,
        2.232_142_9e-3,
        6.103_515_6e-5,
        1.0 / (1u128 << 117) as f32,
        -1.0 / (1u128 << 117) as f32,
        0.0,
        -0.0,
        3.0e38,
        -3.5,
        0.1,
        1.000_000_1,
        7.0e-30,
        -0.333_333_34,
        1234.567,
        2.0e-4,
    ];

    /// The F32 values a `shape` tensor of codes `at` (as a byte) under the
    /// `[128, 128]`-tiled grid of `scales` imports to, as an F32 reference
    /// computes them: each code's value times its tile's scale, one
    /// correctly rounded F32 product (an E4M3 value has 4 significant bits
    /// and an F32 scale 24, so their product is exact in F64 and rounds once
    /// to F32), NaN under a NaN code.
    fn f32_reference(shape: &[u64], scales: &[f32]) -> Vec<f32> {
        let (rows, columns) = (shape[shape.len() - 2] as usize, shape[shape.len() - 1] as usize);
        let tiles = columns / 128;
        (0..shape.iter().product::<u64>() as usize)
            .map(|at| {
                let (matrix, row, column) = (at / (rows * columns), at / columns % rows, at % columns);
                let scale = scales[(matrix * rows / 128 + row / 128) * tiles + column / 128];
                (e4m3(at as u8) * f64::from(scale)) as f32
            })
            .collect()
    }

    fn f32_bytes(scales: &[f32]) -> Vec<u8> {
        scales.iter().flat_map(|scale| scale.to_le_bytes()).collect()
    }

    /// One file holds an F8_E4M3 matrix under an I8 grid of UE8M0 scales
    /// beside a matrix, and a stack of them, under F32 scales per 128×128
    /// tile (an FP8 checkpoint's `weight_scale_inv`). In every backend's
    /// layout each imports in its own grid's codes: the UE8M0 one as
    /// mxfp8g32, every value its reference's (`ue8m0_reference`), and the
    /// F32 ones as fp8g128, the codes bit for bit (a block of 128 of a
    /// row's codes after its tile's scale, 132 bytes) and every value the
    /// F32 reference's (`f32_reference`), E4M3 subnormals, −0 and NaN
    /// codes, the least scale and products past F32's range included.
    #[test]
    fn one_file_imports_ue8m0_and_f32_grids_to_their_exact_values() {
        let scales = f32_bytes(&F32_SCALES);
        let f32_weights = [
            Gridded { stem: "blk.1.ffn_down", shape: &[512, 512], grid: &[4, 4], codes: ScaleCodes::F32, scales: &scales },
            Gridded {
                stem: "blk.1.ffn_down_exps",
                shape: &[4, 256, 256],
                grid: &[4, 2, 2],
                codes: ScaleCodes::F32,
                scales: &scales,
            },
        ];
        let path = gguf(&[UE8M0_WEIGHT, f32_weights[0], f32_weights[1]], "mixed");
        for backend in BackendName::ALL {
            let layout = backend.resident_layout();
            let (plan, values) = import(&path, layout, &UE8M0_WEIGHT).unwrap();
            assert_eq!(plan.resident, Element::stored("mxfp8g32", layout).unwrap(), "{backend:?}");
            assert_bits(&format!("ue8m0 on {backend:?}"), &values, &ue8m0_reference());
            for weight in &f32_weights {
                let (plan, values) = import(&path, layout, weight).unwrap();
                let source = Element::gguf_source(Encoding::F8E4M3, GgufScales::Grid(GridScale::F32), 128).unwrap();
                assert_eq!((plan.source, plan.upload), (source, source));
                assert_eq!(plan.resident, Element::stored("fp8g128", layout).unwrap(), "{backend:?}");
                assert_eq!(plan.shape, weight.shape);
                assert_eq!(plan.source_bytes, weight.shape.iter().product::<u64>() / 128 * 132);
                let name = format!("{} on {backend:?}", weight.stem);
                assert_bits(&name, &values, &f32_reference(weight.shape, &F32_SCALES));
            }
        }
        std::fs::remove_file(&path).unwrap();
    }

    /// A non-finite F32 scale, and a nonzero one below 2^-117 (where a
    /// backend's folded `(s · 2^-9) · q` stops being the one product),
    /// fail the import naming the scale; 2^-117 itself and zero import
    /// (`one_file_imports_ue8m0_and_f32_grids_to_their_exact_values`).
    #[test]
    fn an_f32_grid_of_unreadable_scales_is_refused() {
        let least = 1.0 / (1u128 << 117) as f32;
        for (bad, reason) in [
            (f32::NAN, "not finite"),
            (f32::INFINITY, "not finite"),
            (f32::NEG_INFINITY, "not finite"),
            (least / 2.0, "below 2^-117"),
            (-f32::from_bits(least.to_bits() - 1), "below 2^-117"),
            (f32::from_bits(1), "below 2^-117"),
        ] {
            let mut scales = F32_SCALES;
            scales[5] = bad;
            let scales = f32_bytes(&scales);
            let weight =
                Gridded { stem: "blk.0.attn_q_b", shape: &[512, 512], grid: &[4, 4], codes: ScaleCodes::F32, scales: &scales };
            let path = gguf(&[weight], "f32-refused");
            let error = import(&path, Layout::Rows8, &weight).unwrap_err();
            std::fs::remove_file(&path).unwrap();
            assert!(error.contains("scale 5 of the grid") && error.contains(reason), "{bad:e}: {error}");
        }
    }

    /// Under F32 scales a source block is 128 values, so a tile narrower
    /// than 128 columns is refused, naming the weight and grid; a 128-wide
    /// tile of any power-of-two rows plans, in every backend's layout.
    #[test]
    fn an_f32_grid_tiles_by_whole_128_value_blocks() {
        let (mut descriptor, weight) = gridded([32, 32]);
        let ImportTransform::ScaleBlocks { codes, .. } = &mut descriptor.transforms[0] else {
            unreachable!("a gridded weight");
        };
        *codes = ScaleCodes::F32;
        let tiled = |tile| {
            let mut descriptor = descriptor.clone();
            let ImportTransform::ScaleBlocks { tile: at, .. } = &mut descriptor.transforms[0] else {
                unreachable!("a gridded weight");
            };
            *at = tile;
            descriptor
        };
        let error = plan(std::slice::from_ref(&weight), &tiled([32, 64]), Layout::Rows8).unwrap_err();
        assert!(error.contains(WEIGHT) && error.contains(SCALE), "{error}");
        for backend in BackendName::ALL {
            let layout = backend.resident_layout();
            let planned = plan(std::slice::from_ref(&weight), &tiled([1, 128]), layout).unwrap();
            assert_eq!(planned.resident, Element::stored("fp8g128", layout).unwrap(), "{backend:?}");
            assert!(resident_elements(layout).contains(&planned.resident), "{backend:?}");
        }
    }
}
