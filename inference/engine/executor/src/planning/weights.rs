use super::{
    ArtifactComponent, ArtifactComponentKind, ComponentSelection, FeedForwardProgramSlot,
    ProgramPlan,
};
use crate::error::PlanError;
use crate::streamed_table::STREAMED_TABLE_ROWS;
use crate::{ExecutionPath, operators};
use magnitude_artifacts::{
    ArtifactIdentity, PackageHeaders, PackageIdentity, PackageManifest,
    gguf::{Encoding, TensorDescriptor},
};
use magnitude_family_contracts::{
    ActivationDType, HeadComponent, ImportTransform, ModelDefinition, ProgressiveCodeDepth,
    ProgressivePlane, ScaledActivations, VisionDescription, WeightDescriptor, WeightKind,
    WeightRole, WeightScope,
};
use magnitude_state::KvCodec;
use seismic::{BackendName, DType, Element, Layout, OperandRecipe, Packing};
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
    /// A gathered table that is streamed (`device_table`): `shape` is then
    /// its slots', which hold only some of the table's rows. `None` for a
    /// weight held whole.
    pub streamed: Option<StreamedRows>,
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
    /// names (`ModelLoadPlan::with_selected_reads`); 0 of a load without
    /// one, and of a stored table, whose filler writes every row a read
    /// names into its slots.
    pub landing_rows: u64,
    /// How readers find a row's slot.
    pub directory: DirectoryForm,
    pub fill: TableFill,
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
    /// import entry from staging, queued ahead of the step that names them.
    Imported,
    /// In the table's resident representation, read from the artifact as
    /// stored and converted row by row by the host (the CPU backend's
    /// registered row conversion), which writes them into slots the device
    /// reads in place (`stored_rows`). A step reading them waits at a gate
    /// until the host has written them.
    Stored,
}

impl WeightPlan {
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
            self.streamed,
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
            self.streamed,
            Some(StreamedRows {
                fill: TableFill::Imported,
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
    /// a streamed table's rows in place of its slots.
    pub fn stored_shape(&self) -> Vec<u64> {
        let mut shape = self.shape.clone();
        if let Some(streamed) = self.streamed {
            shape[0] = streamed.table_rows;
        }
        shape
    }

    /// The shape `[R, F]` of a streamed table's device directory
    /// (`DirectoryForm`); `[0, 1]`, the empty directory, of a weight held
    /// whole.
    pub fn directory_shape(&self) -> [u64; 2] {
        self.streamed.map_or([0, 1], |streamed| {
            [
                streamed
                    .directory
                    .entries(streamed.table_rows, self.shape[0]),
                streamed.directory.fields(),
            ]
        })
    }

    /// The allocations a streamed table holds beside its slots: the
    /// directory, the staging its tiles pass through and the landing staging
    /// its landing slots are converted from, both as stored, and the notice a
    /// selected read's step copies its `[rows, 2]` selection into. Zeros for
    /// a weight held whole.
    pub fn streamed_bytes(&self) -> Result<[u64; 4], String> {
        let Some(streamed) = self.streamed else {
            return Ok([0; 4]);
        };
        let rows = |rows| {
            let mut shape = self.shape.clone();
            shape[0] = rows;
            shape
        };
        Ok([
            representation_bytes(Element::i32(), &self.directory_shape())?,
            representation_bytes(self.upload, &rows(streamed.staging_rows))?,
            representation_bytes(self.upload, &rows(streamed.landing_rows))?,
            representation_bytes(Element::i32(), &[streamed.landing_rows, 2])?,
        ])
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

    /// The extent of the weight's accumulator-scale port: 0 without a
    /// second-level scale, else its resident scale's (1 for a matrix, the
    /// stack's matrix count for a stacked weight).
    pub fn scale_extent(&self) -> u64 {
        self.scale.as_ref().map_or(0, |scale| scale.extent)
    }
}

/// How the token-packing projection forms read a resident weight
/// (`Element::packing`), in the layout they read (Metal's `rows32`); `None`
/// for any other weight.
fn packing(resident: Element) -> Option<Packing> {
    (resident.layout() == Layout::Rows32)
        .then(|| resident.packing())
        .flatten()
}

/// The columns of one run of a resident weight's packing bound, the columns
/// of a 32-column block that share a scale (`packing`); `None` for a weight
/// the packed forms do not read.
pub fn packing_run(resident: Element) -> Option<u64> {
    packing(resident).map(|packing| u64::from(packing.run))
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

/// Whether a weight's packed sums take residue bits beside their 16-bit
/// fields (`Packing::residue`: codes of 8-bit values, whose sums 16 bits a
/// token do not hold at a useful gain). Only such weights make the packed
/// forms' second residue width (`RESIDUE` 2) apply.
pub fn packing_residue(resident: Element) -> bool {
    packing(resident).is_some_and(|packing| packing.residue())
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
    /// Its stored extent: 1, or the weight's matrix count.
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

/// The static axes the attention entries are specialized to. Optional
/// parts of the operator are axes of extent 0 or 1 (`attention.seismic`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionShape {
    /// `D`: the hidden width.
    pub hidden: u64,
    /// `KV`: key/value heads.
    pub kv_heads: u64,
    /// `G`: query heads per key/value head.
    pub group: u64,
    /// `P`: rotated pairs of a head.
    pub rotary_pairs: u64,
    /// `W = 2P + S`: the head width.
    pub width: u64,
    /// `I` ∈ {0, W}: gate columns after each query head's W columns.
    pub interleaved_gate: u64,
    /// `U` ∈ {0, 1, W}: separate gate values per query head.
    pub separate_gate: u64,
    /// `F` ∈ {0, 1}: whether the layer projects and appends its own keys and
    /// values.
    pub fresh: u64,
    /// `N` ∈ {0, 1}: whether queries and keys are RMS-normalized per head.
    pub head_norm: u64,
    /// `NV` ∈ {0, 1}: whether values are RMS-normalized per head.
    pub value_norm: u64,
    /// Whether values have their own projection (else they are the raw key).
    pub projected_value: bool,
    /// `SK` ∈ {0, 1}: whether each query head has a sink logit
    /// (`Attention::sinks`).
    pub sinks: u64,
}

impl AttentionShape {
    pub fn heads(&self) -> u64 {
        self.kv_heads * self.group
    }

    /// `Q`: query (and interleaved gate) projection rows.
    pub fn query_rows(&self) -> u64 {
        self.heads() * (self.width + self.interleaved_gate)
    }

    /// `GR`: separate gate projection rows.
    pub fn gate_rows(&self) -> u64 {
        self.heads() * self.separate_gate
    }

    /// `K`: key projection rows.
    pub fn key_rows(&self) -> u64 {
        self.fresh * self.kv_heads * self.width
    }

    /// `V`: value projection rows.
    pub fn value_rows(&self) -> u64 {
        if self.projected_value {
            self.key_rows()
        } else {
            0
        }
    }

    /// Unrotated columns of a head.
    pub fn static_width(&self) -> u64 {
        self.width - 2 * self.rotary_pairs
    }

    /// The projection entry's dimensions at `rows` rows.
    pub fn project_dimensions(&self, rows: u64) -> [(&'static str, u64); 6] {
        [
            ("M", rows),
            ("D", self.hidden),
            ("Q", self.query_rows()),
            ("GR", self.gate_rows()),
            ("K", self.key_rows()),
            ("V", self.value_rows()),
        ]
    }

    /// Projection of history rows has no query or gate outputs.
    pub fn key_value_dimensions(&self, rows: u64) -> [(&'static str, u64); 6] {
        let mut dimensions = self.project_dimensions(rows);
        dimensions[2].1 = 0;
        dimensions[3].1 = 0;
        dimensions
    }

    /// The static axes of the fused attention entries.
    pub fn mix_statics(&self) -> [(&'static str, u64); 10] {
        [
            ("KV", self.kv_heads),
            ("G", self.group),
            ("P", self.rotary_pairs),
            ("S", self.static_width()),
            ("I", self.interleaved_gate),
            ("U", self.separate_gate),
            ("F", self.fresh),
            ("N", self.head_norm),
            ("NV", self.value_norm),
            ("SK", self.sinks),
        ]
    }

    /// Cache publication has no query-head, gate or sink geometry.
    pub fn append_statics(&self) -> Vec<(&'static str, u64)> {
        self.mix_statics()
            .into_iter()
            .filter(|(name, _)| !matches!(*name, "G" | "I" | "U" | "SK"))
            .collect()
    }

    /// The fused entries' dimensions for `rows` rows over `history_rows`
    /// history rows read in `segments` visible spans.
    pub fn mix_dimensions(
        &self,
        rows: u64,
        history_rows: u64,
        segments: u64,
    ) -> [(&'static str, u64); 13] {
        let [kv, g, p, s, i, u, f, n, nv, sk] = self.mix_statics();
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
            n,
            nv,
            sk,
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
    pub query: Element,
    pub gate: Element,
    pub key: Element,
    pub value: Element,
    pub output: Element,
    pub activation: Element,
    /// How the block's history planes encode keys and values.
    pub history: magnitude_state::CodecSpec,
    pub tail: SublayerTail,
    /// A selected read's indexer (`HistoryReads::Selected`); none for a
    /// read of every visible row.
    pub selection: Option<SelectionBinding>,
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
    /// Canonical packed projection plus cost-qualified bounded classes.
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

/// Completed first-load physical choice of one reader of a projection scope.
/// The scope is stored as its views; the reader's listed selected-row classes
/// use bounded readout, and its other classes, and the scope's other
/// readers, the views' full pass.
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
    /// Optional lossless readout views used only for first-load qualification.
    /// They are not resident requirements until a served class selects them.
    pub(super) readout_candidates: Vec<WeightPlan>,
    pub(super) readout_choices: Vec<ReadoutClassChoice>,
    /// The output projections of the scopes stored as planes
    /// ([`ModelLoadPlan::finalize_readout`]): planned, so bindings that name
    /// them stand and their demand is derived, and not placed. A weight has
    /// one stored form.
    pub(super) unplaced_output: Vec<WeightPlan>,
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

    /// The host-shared bytes of the target's stored tables
    /// (`TableFill::Stored`): their slots and directories, part of the
    /// target's weights, and upload storage, held where the executor's upload
    /// placement lands.
    pub fn shared_table_bytes(&self) -> Result<u64, String> {
        self.target
            .iter()
            .filter(|weight| weight.stored_in_slots())
            .try_fold(0u64, |total, weight| {
                total
                    .checked_add(weight.storage_bytes()?)
                    .ok_or_else(|| "stored table bytes overflow".to_owned())
            })
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
    pub fn target_upload_peak_bytes(&self) -> Result<u64, String> {
        super::ImportPlan::new(self.target.iter()).peak_bytes()
    }

    /// Peak upload allocation while importing a lazy optional component.
    pub fn head_upload_peak_bytes(&self) -> Result<u64, String> {
        super::ImportPlan::new(self.head.iter().flatten()).peak_bytes()
    }

    pub fn vision_upload_peak_bytes(&self) -> Result<u64, String> {
        super::ImportPlan::new(self.vision.iter().flatten()).peak_bytes()
    }
}

/// The elements of a tensor of `extents`; `None` beyond `u64`.
pub(crate) fn elements(extents: &[u64]) -> Option<u64> {
    extents
        .iter()
        .try_fold(1u64, |count, extent| count.checked_mul(*extent))
}

pub(super) fn weight_bytes_by_component(load: &ModelLoadPlan) -> Result<[u64; 3], String> {
    let mut seen = HashMap::new();
    let mut bytes = [0u64; 3];
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
                    bytes[index] = bytes[index]
                        .checked_add(weight.storage_bytes()?)
                        .ok_or("resident weight byte count overflow")?;
                }
                std::collections::hash_map::Entry::Occupied(entry) if entry.get() != &storage => {
                    return Err("tied weight roles disagree on their physical storage".into());
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
            }
        }
    }
    Ok(bytes)
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
            history,
        )
    }

    /// Plan every selected weight. Packed weights become resident in
    /// `layout` (`resident_layout` of the engine's execution path and
    /// device backend).
    pub fn derive(
        manifest: &PackageManifest,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        layout: Layout,
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
        )
    }

    fn derive_components(
        identity: PackageIdentity,
        target_tensors: &[TensorDescriptor],
        projector: Option<(ArtifactIdentity, &[TensorDescriptor])>,
        draft: Option<(ArtifactIdentity, &[TensorDescriptor])>,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        layout: Layout,
    ) -> Result<Self, String> {
        definition.validate().map_err(|error| error.to_string())?;
        if identity != definition.artifact_identity {
            return Err("load plan package identity mismatch".into());
        }
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
            push_weights(plans, tensors, component, weights, target_form)
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
        })
    }

    /// Offer lossless bounded-selection views of the target's output
    /// projection where every reader of it can read them at no loss: a Q8_0
    /// projection without a soft cap. Indexed policies require the
    /// canonical table, and a separate drafter that projects through the
    /// target's projection has no kernel for the views, so there the
    /// projection keeps its packed form, as does a head's own projection. A
    /// view becomes resident, in place of the packed projection, only after
    /// whole-recipe class-cost qualification.
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
                        streamed: None,
                    });
                }
            }
            Ok(views)
        };
        let drafter_reads_target = definition
            .draft
            .as_ref()
            .is_some_and(|draft| draft.output.is_none())
            && self
                .target
                .iter()
                .any(|weight| weight.role.scope == WeightScope::Draft);
        self.readout_candidates = if drafter_reads_target {
            Vec::new()
        } else {
            select(&self.target)?
        };
        Ok(self)
    }

    /// The load of `limits` drafting by `method`: every imported streamed
    /// table holds landing slots after its slots for the most rows one of
    /// its reads of rows the device selects names (`StreamedRows::
    /// landing_rows`), whole tiles of them.
    pub fn with_selected_reads(
        mut self,
        definition: &ModelDefinition,
        method: super::PlannedMethod,
        limits: super::ResourceLimits,
    ) -> Result<Self, String> {
        let rows = selected_read_rows(definition, method, limits)?;
        let tables = self
            .target
            .iter_mut()
            .chain(self.head.iter_mut().flatten())
            .filter(|weight| {
                matches!(
                    weight.streamed,
                    Some(StreamedRows {
                        fill: TableFill::Imported,
                        ..
                    })
                )
            });
        for weight in tables {
            let streamed = weight.streamed.as_mut().expect("an imported streamed table");
            let tile = weight.resident.tile_rows();
            let landing = rows.div_ceil(tile) * tile;
            weight.shape[0] = weight.shape[0] - streamed.landing_rows + landing;
            streamed.landing_rows = landing;
            weight.resident_bytes = representation_bytes(weight.resident, &weight.shape)?;
        }
        Ok(self)
    }

    /// Commit completed physical choices before resource admission. A scope
    /// in which any reader selected a class is stored as its views: they are
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
            if choice.rows.is_empty() || choice.rows.iter().any(|&rows| rows == 0) {
                return Err("a retained readout view needs nonzero selected-row classes".into());
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
/// streams (`ngram_gate`) and convolved over its window (`ngram_inject`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NGramBinding {
    pub hidden: u64,
    pub streams: u64,
    /// Predecessors the row rule reads (C): its orders are 2..=C + 1.
    pub context: u64,
    /// Rows gathered per order (P), each `width` wide.
    pub heads_per_order: u64,
    pub width: u64,
    pub kernel: u64,
    pub dilation: u64,
    pub table: Element,
    pub key: Element,
    pub value: Element,
    /// The key, query and convolution norms'.
    pub norm: Element,
    pub activation: Element,
}

impl NGramBinding {
    /// Rows gathered per position.
    pub fn heads(&self) -> u64 {
        self.context * self.heads_per_order
    }

    /// The width of a position's gathered rows.
    pub fn embedding(&self) -> u64 {
        self.heads() * self.width
    }

    /// The streams' channels, `S * H`.
    pub fn channels(&self) -> u64 {
        self.streams * self.hidden
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
    if stored.shape.len() != 2 {
        return Err(format!("host table {:?} is not a matrix", descriptor.name));
    }
    let source = source_element(stored.encoding)
        .ok_or_else(|| format!("unsupported source encoding {:?}", stored.encoding))?;
    let resident = resident_element(
        stored.encoding,
        crate::import_transforms::scaled_activations(descriptor),
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
) -> Result<(), String> {
    for (role, descriptor) in &weights {
        if gathered_table(role.kind) {
            let read_whole = weights
                .iter()
                .any(|(other, stored)| other != role && stored.name == descriptor.name);
            push_gathered_table(
                out, inventory, component, *role, descriptor, form, read_whole,
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
/// instead (`push_stored_table`).
fn push_gathered_table(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    role: WeightRole,
    descriptor: &WeightDescriptor,
    form: ResidentForm,
    read_whole: bool,
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
    streamed.resident_bytes = representation_bytes(streamed.resident, &streamed.shape)?;
    // The load uploads the rows that fill the slots; nothing more of the
    // table is ever uploaded at once.
    streamed.source_bytes = representation_bytes(streamed.upload, &streamed.shape)?;
    streamed.streamed = Some(StreamedRows {
        table_rows,
        // The largest step any backend admits, each row of which may bring
        // its tile.
        staging_rows: slots.min(magnitude_batching::MAX_CLASS_ROWS as u64 * tile),
        landing_rows: 0,
        directory: DirectoryForm::of(table_rows, slots),
        fill: TableFill::Imported,
    });
    out.push(streamed);
    Ok(())
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
/// the plan picks (`DirectoryForm::of`). A table with import transforms,
/// rows past I32 or without a host row conversion into its resident
/// representation is refused. The slots are the weight's shape and resident
/// bytes; the load uploads nothing.
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
    if !descriptor.transforms.is_empty() || stored.shape != descriptor.shape {
        return Err(refused("it is not the stored matrix itself"));
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
    let budget = representation_bytes(table.upload, &[NGRAM_TABLE_STORED_ROWS, width])?;
    let slots = (budget / representation_bytes(table.resident, &[1, width])?).min(table_rows);
    table.shape = vec![slots, width];
    table.resident_bytes = representation_bytes(table.resident, &table.shape)?;
    table.source_bytes = 0;
    // A stored table is read by row, never through a packed projection.
    table.bound = 0;
    table.streamed = Some(StreamedRows {
        table_rows,
        staging_rows: 0,
        landing_rows: 0,
        directory: DirectoryForm::of(table_rows, slots),
        fill: TableFill::Stored,
    });
    out.push(table);
    Ok(())
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
    let source = source_element(stored.encoding)
        .ok_or_else(|| format!("unsupported source encoding {:?}", stored.encoding))?;
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
            resident_element(
                stored.encoding,
                crate::import_transforms::scaled_activations(descriptor),
                dense_resident,
                form.layout,
            )
            .ok_or_else(|| {
                format!(
                    "{:?} has no resident form in the `{}` layout",
                    stored.encoding,
                    form.layout.as_str()
                )
            })?,
        ),
    };
    let source_bytes = representation_bytes(source, &stored.shape)?;
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
        streamed: None,
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
    // Every reader of a scope stored as views reads them, whichever of the
    // scope's readers selected a class.
    if !choices
        .iter()
        .any(|choice| choice.scope == scope && !choice.rows.is_empty())
    {
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

/// The external storage of an artifact encoding: its dense element, or the
/// registry's source (`None` when the registry has none).
pub fn source_element(encoding: Encoding) -> Option<Element> {
    match encoding {
        Encoding::F32 => Some(Element::dense(DType::F32)),
        Encoding::F16 => Some(Element::dense(DType::F16)),
        Encoding::BF16 => Some(Element::dense(DType::BF16)),
        _ => Element::gguf_source(encoding),
    }
}

#[cfg(test)]
#[test]
fn every_written_weight_type_has_a_source() {
    // GGUF's support set: every type but those no producer writes and the
    // integer tensors, which are model work.
    for encoding in Encoding::ALL {
        let integer = matches!(
            encoding,
            Encoding::I8 | Encoding::I16 | Encoding::I32 | Encoding::I64
        );
        assert_eq!(
            source_element(encoding).is_some(),
            encoding.produced() && !integer,
            "{encoding:?}"
        );
    }
}

/// The resident storage of an artifact encoding: that of its source
/// (`source_element`) by [`resident_storage`].
pub fn resident_element(
    encoding: Encoding,
    activations: ScaledActivations,
    dense: DType,
    layout: Layout,
) -> Option<Element> {
    resident_storage(source_element(encoding)?, activations, dense, layout)
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
fn resident_storage(
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
/// [`resident_element`] over every encoding, activations and dense dtype,
/// each once, in a fixed order.
pub fn resident_elements(layout: Layout) -> Vec<Element> {
    let mut elements = Vec::new();
    for dense in [DType::BF16, DType::F16, DType::F32] {
        for encoding in Encoding::ALL {
            for activations in [ScaledActivations::Rounded, ScaledActivations::Nvfp4] {
                match resident_element(encoding, activations, dense, layout) {
                    Some(element) if !elements.contains(&element) => elements.push(element),
                    _ => {}
                }
            }
        }
    }
    elements
}

fn representation_bytes(element: Element, shape: &[u64]) -> Result<u64, String> {
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
    /// from is an artifact encoding's, and every encoding with a source is
    /// resident in every layout, a dense one in the activation dtype and a
    /// packed one as its conversion writes it, in W4A4's NVFP4-operand
    /// element only where the conversion has one (sharing its storage).
    #[test]
    fn the_registry_owns_the_encoding_map() {
        let reached: HashSet<Element> = Encoding::ALL
            .into_iter()
            .filter_map(source_element)
            .collect();
        for source in Element::external_sources() {
            assert!(reached.contains(&source), "{} is no encoding's source", source.name());
        }
        for encoding in Encoding::ALL {
            let Some(source) = source_element(encoding) else {
                continue;
            };
            for layout in Layout::ALL {
                let resident =
                    |activations| resident_element(encoding, activations, DType::BF16, layout);
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
}

#[cfg(test)]
mod stored_table_tests {
    use super::*;
    use magnitude_family_contracts::SublayerIndex;

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
            (Encoding::Q4_1, 20, "q4g32", 112, 2_633_142),
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
                assert_eq!(table.shape, vec![slots, 160], "{encoding:?}");
                assert_eq!(table.resident_bytes, slots * row, "{encoding:?}");
                let budget = NGRAM_TABLE_STORED_ROWS * stored_row;
                assert!(table.resident_bytes <= budget && budget < table.resident_bytes + row);
                assert_eq!(table.streamed.unwrap().fill, TableFill::Stored);
                assert_eq!(
                    representation_bytes(table.upload, &table.stored_shape()).unwrap(),
                    ROWS * stored_row,
                );
                assert_eq!(
                    super::super::ImportPlan::new(out.iter()).peak_bytes().unwrap(),
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

    /// The qwen4exp fixture with its n-gram table stored as Q5_0 blocks (as
    /// a Q4_K_M preset stores Flash Next's) plans on every native backend,
    /// its n-gram input reading the table's resident rows16 slots.
    #[test]
    fn a_q5_0_ngram_table_plans() {
        let definition = crate::planning::tests::qwen4exp_fixture_definition();
        let mut manifest = crate::planning::tests::stored_table_manifest(&definition);
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
