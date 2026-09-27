//! The measurement basis: the fixed, model-free set of operation classes the
//! execution implementation runs on one device, each timed with its shipped
//! default configuration on synthetic resident tensors. Every model is
//! assessed analytically against this basis; nothing here is per model.
//!
//! A class's key names the operation and the elements it binds, never a
//! model's geometry. Geometry enters as the variable of a class's cost
//! model: output rows for a weight-streaming launch, head geometry for decode
//! attention, bytes for everything else.
//!
//! The basis is also the compatibility set. Every entry binding a weight
//! representation is formed on the device at every representation the load
//! planner produces; one that cannot be formed is recorded unsupported, and a
//! model that needs it is incompatible with this device.

use crate::StreamingCost;
use seismic::Element;

/// Changes whenever the measured classes, key rules, sizes or timing rules
/// change, so a cached basis from an older protocol is never reused.
pub const MEASUREMENT_PROTOCOL_VERSION: u32 = 11;

/// One native entry a plain target decode step launches. A plain step is one
/// row through the embedding entry graph, every decoder block graph (decode
/// row class) and the selection readout graph (`readout_features_rows`,
/// `readout_head_rows`, then `sample_rows`; an unshaped selection, as greedy
/// decoding and plain temperature-1 sampling take). Three classes are not
/// entries: the relative cost of streaming one weight representation, the
/// cost every entry call adds when it depends on the previous one, and the
/// cost of submitting and waiting for one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OperationClass {
    EmbeddingRows,
    AttentionProject,
    AttentionDecode,
    AttentionDecodeK8V4,
    AttentionOutput,
    DeltaProject,
    DeltaStep,
    DeltaOutput,
    /// The Mamba-2 state advance of one row (`state_space_step`).
    StateSpaceStep,
    /// The Mamba-2 gated group norm of one row (`state_space_gate`).
    StateSpaceGate,
    /// The short convolution's normed `u | C` projection
    /// (`short_conv_project`).
    ShortConvProject,
    /// The short convolution's gated taps over one row (`short_conv_rows`).
    ShortConvRows,
    DenseExpand,
    /// An up-only dense expansion (`dense_up`).
    DenseUp,
    DenseOutput,
    /// General routing: scores, biased top-k, weights (`routed_select`).
    RoutedSelect,
    /// General decode expert expansions (`routed_gate_up`, `routed_up`) and
    /// down projection (`routed_down`).
    RoutedGateUp,
    RoutedUp,
    RoutedDown,
    RoutedRoute,
    RoutedExpand,
    RoutedOutput,
    /// A post-norm sublayer's output projection into F32 rows.
    ProjectRows,
    /// A post-norm sublayer's row op: normalize, then add to the residual.
    PostNormResidual,
    /// The tail of a dense branch beside a routed branch: both branch rows
    /// normalized, their sum post-normalized into the residual (`moe_tail`).
    MoeTail,
    /// A per-layer input sublayer's gate projection of one row times its
    /// layer's per-layer input (`per_layer_gate`).
    PerLayerGate,
    /// The per-layer entry's combination of one row's table rows and
    /// projected rows for every layer (`per_layer_inputs`).
    PerLayerInputs,
    /// One row converted between dense representations (`import_dense`).
    ImportRows,
    /// One row converted into a packed resident representation
    /// (`repack_weight`).
    RepackRows,
    /// One F32 row copied into a program's rows (`conditioning_overlay`).
    CopyRows,
    /// Host-observed time of gathering one row of a host-resident table from
    /// memory and writing it into a graph's upload input, per byte.
    TableUpload,
    ReadoutFeatures,
    ReadoutHead,
    SampleRows,
    /// The device time per byte of streaming one weight representation
    /// through a bandwidth-bound projection (`project_rows` at its largest
    /// launch). A weight-streaming class is timed at one reference
    /// representation; another representation scales its streaming time by
    /// the ratio of their per-byte times.
    WeightFormat,
    /// Extra device time of an entry call that depends on the previous call
    /// (as every launch of a decode step does), beyond its independent
    /// back-to-back time that the entry classes measure.
    LaunchDependency,
    /// Host-observed time of submitting one step and waiting for its
    /// completion, beyond the device time of its launches.
    StepSubmission,
}

impl OperationClass {
    pub const ALL: [Self; 37] = [
        Self::EmbeddingRows,
        Self::AttentionProject,
        Self::AttentionDecode,
        Self::AttentionDecodeK8V4,
        Self::AttentionOutput,
        Self::DeltaProject,
        Self::DeltaStep,
        Self::DeltaOutput,
        Self::StateSpaceStep,
        Self::StateSpaceGate,
        Self::ShortConvProject,
        Self::ShortConvRows,
        Self::DenseExpand,
        Self::DenseUp,
        Self::DenseOutput,
        Self::RoutedSelect,
        Self::RoutedGateUp,
        Self::RoutedUp,
        Self::RoutedDown,
        Self::RoutedRoute,
        Self::RoutedExpand,
        Self::RoutedOutput,
        Self::ProjectRows,
        Self::PostNormResidual,
        Self::MoeTail,
        Self::PerLayerGate,
        Self::PerLayerInputs,
        Self::ImportRows,
        Self::RepackRows,
        Self::CopyRows,
        Self::TableUpload,
        Self::ReadoutFeatures,
        Self::ReadoutHead,
        Self::SampleRows,
        Self::WeightFormat,
        Self::LaunchDependency,
        Self::StepSubmission,
    ];

    /// The native entry name (the measured quantity's name for the three
    /// classes that are not entries).
    pub const fn name(self) -> &'static str {
        match self {
            Self::EmbeddingRows => "embedding_rows",
            Self::AttentionProject => "attention_project",
            Self::AttentionDecode => "attention_decode",
            Self::AttentionDecodeK8V4 => "attention_decode_k8v4",
            Self::AttentionOutput => "attention_output",
            Self::DeltaProject => "gated_delta_project",
            Self::DeltaStep => "gated_delta_step",
            Self::DeltaOutput => "gated_delta_output",
            Self::StateSpaceStep => "state_space_step",
            Self::StateSpaceGate => "state_space_gate",
            Self::ShortConvProject => "short_conv_project",
            Self::ShortConvRows => "short_conv_rows",
            Self::DenseExpand => "dense_expand",
            Self::DenseUp => "dense_up",
            Self::DenseOutput => "dense_output",
            Self::RoutedSelect => "routed_select",
            Self::RoutedGateUp => "routed_gate_up",
            Self::RoutedUp => "routed_up",
            Self::RoutedDown => "routed_down",
            Self::RoutedRoute => "routed_route",
            Self::RoutedExpand => "routed_expand",
            Self::RoutedOutput => "routed_output",
            Self::ProjectRows => "project_rows",
            Self::PostNormResidual => "post_norm_residual",
            Self::MoeTail => "moe_tail",
            Self::PerLayerGate => "per_layer_gate",
            Self::PerLayerInputs => "per_layer_inputs",
            Self::ImportRows => "import_dense",
            Self::RepackRows => "repack_weight",
            Self::CopyRows => "conditioning_overlay",
            Self::TableUpload => "table_upload",
            Self::ReadoutFeatures => "readout_features_rows",
            Self::ReadoutHead => "readout_head_rows",
            Self::SampleRows => "sample_rows",
            Self::WeightFormat => "weight_format",
            Self::LaunchDependency => "launch_dependency",
            Self::StepSubmission => "step_submission",
        }
    }

    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.name() == name)
    }

    /// How the class's launch time depends on what it runs over:
    ///
    /// - Weight-streaming GEMV entries are `Projection` costs: a launch floor
    ///   plus the bytes streamed at a per-byte time that depends on the
    ///   launch's output rows, the parallelism the entry has. At one byte
    ///   count a launch's time differs with how the bytes split into rows
    ///   and reduction: a 16 MB q4k expansion of 4,096 rows read 2.6× slower
    ///   than one of 7,168 rows on an M4 Pro (2026-09-27), and every entry
    ///   sits on a latency floor below a few MB on GB10 (2026-09-25). The
    ///   representation enters through `WeightFormat`.
    /// - Decode attention is a `History` cost: a launch floor plus its
    ///   history bytes at a per-byte time that depends on the head geometry
    ///   (it differs 5–6× between 2 and 4 KV heads) and is linear from 1k to
    ///   262k tokens on Metal and GB10.
    /// - Row ops, state advances, routing, conversions, the host upload and
    ///   sampling are `Linear` in the bytes they touch.
    /// - One-row launches that touch a hidden-width row (the row embedding,
    ///   norms, the branch tail) cost the launch itself (`PerLaunch`), as do
    ///   the dependency and submission costs, fixed per call and per step.
    /// - `WeightFormat` is a time per byte (`PerByte`).
    pub const fn cost_shape(self) -> CostShape {
        match self {
            Self::EmbeddingRows
            | Self::ReadoutFeatures
            | Self::PostNormResidual
            | Self::MoeTail
            | Self::LaunchDependency
            | Self::StepSubmission => CostShape::PerLaunch,
            Self::DeltaStep
            | Self::StateSpaceStep
            | Self::StateSpaceGate
            | Self::ShortConvRows
            | Self::RoutedSelect
            | Self::RoutedRoute
            | Self::PerLayerInputs
            | Self::ImportRows
            | Self::RepackRows
            | Self::CopyRows
            | Self::TableUpload
            | Self::SampleRows => CostShape::Linear,
            Self::AttentionProject
            | Self::AttentionOutput
            | Self::DeltaProject
            | Self::DeltaOutput
            | Self::ShortConvProject
            | Self::DenseExpand
            | Self::DenseUp
            | Self::DenseOutput
            | Self::RoutedGateUp
            | Self::RoutedUp
            | Self::RoutedDown
            | Self::RoutedExpand
            | Self::RoutedOutput
            | Self::ProjectRows
            | Self::PerLayerGate
            | Self::ReadoutHead => CostShape::Projection,
            Self::AttentionDecode | Self::AttentionDecodeK8V4 => CostShape::History,
            Self::WeightFormat => CostShape::PerByte,
        }
    }

    /// Whether the class binds a stored representation (a weight, table or
    /// router) or converts into one. Its cost is timed at one reference
    /// representation and keyed without it ([`MeasurementKey::cost`]); its
    /// exact bindings are formed for compatibility.
    pub const fn binds_representation(self) -> bool {
        matches!(self.cost_shape(), CostShape::Projection)
            || matches!(
                self,
                Self::EmbeddingRows
                    | Self::RoutedSelect
                    | Self::RoutedRoute
                    | Self::PerLayerInputs
                    | Self::ImportRows
                    | Self::RepackRows
            )
    }
}

/// How a class's measured cost is modelled (see [`OperationClass::cost_shape`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostShape {
    PerLaunch,
    Linear,
    Projection,
    History,
    PerByte,
}

/// The identity of one measured or formed class: the operation and the
/// elements it binds. Demand derivation and the measurement plan build keys
/// through the same constructors, so equal keys mean the same entry.
///
/// A weight-streaming entry whose segments bind different representations
/// (for example a Q4_K query and a Q6_K value in one attention projection)
/// contributes one demand term per segment representation; its key names the
/// segment's representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeasurementKey {
    pub class: OperationClass,
    pub bindings: Vec<Element>,
}

impl MeasurementKey {
    pub fn new(class: OperationClass, bindings: &[Element]) -> Self {
        Self {
            class,
            bindings: bindings.to_vec(),
        }
    }

    /// The key the class's cost is measured under. A class binding a
    /// representation keeps only its activation (the element its rows are
    /// published in); conversions keep nothing. Every other key is its own
    /// cost key.
    pub fn cost(&self) -> Self {
        if !self.class.binds_representation() {
            return self.clone();
        }
        let bindings = match (self.class, self.bindings.as_slice()) {
            (OperationClass::ImportRows | OperationClass::RepackRows, _) => Vec::new(),
            (_, [.., last]) => vec![*last],
            (_, []) => Vec::new(),
        };
        Self {
            class: self.class,
            bindings,
        }
    }

    /// One table row decoded into activations: `[table, activation]`.
    pub fn embedding_rows(table: Element, activation: Element) -> Self {
        Self::new(OperationClass::EmbeddingRows, &[table, activation])
    }

    /// One Q/K/V projection segment: `[norm, weight, activation]`.
    pub fn attention_project(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::AttentionProject, &[norm, weight, activation])
    }

    /// Fused attention over dense history (`affine` false) or affine K8/V4
    /// history: `[activation]`.
    pub fn attention_decode(affine: bool, activation: Element) -> Self {
        let class = if affine {
            OperationClass::AttentionDecodeK8V4
        } else {
            OperationClass::AttentionDecode
        };
        Self::new(class, &[activation])
    }

    /// The attention output projection: `[weight, activation]`.
    pub fn attention_output(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::AttentionOutput, &[weight, activation])
    }

    /// One recurrent projection segment: `[norm, weight, activation]`.
    pub fn delta_project(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::DeltaProject, &[norm, weight, activation])
    }

    /// The recurrent state advance of one row: `[activation]`.
    pub fn delta_step(activation: Element) -> Self {
        Self::new(OperationClass::DeltaStep, &[activation])
    }

    /// The gated recurrent output projection: `[norm, weight, activation]`.
    pub fn delta_output(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::DeltaOutput, &[norm, weight, activation])
    }

    /// The state-space advance of one row: `[activation]`.
    pub fn state_space_step(activation: Element) -> Self {
        Self::new(OperationClass::StateSpaceStep, &[activation])
    }

    /// The state-space gated group norm of one row: `[activation]`.
    pub fn state_space_gate(activation: Element) -> Self {
        Self::new(OperationClass::StateSpaceGate, &[activation])
    }

    /// An up-only dense expansion: `[norm, weight, activation]`.
    pub fn dense_up(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::DenseUp, &[norm, weight, activation])
    }

    /// One segment (`B`, `C` or `X`) of the short convolution's input
    /// projection: `[norm, weight, activation]`.
    pub fn short_conv_project(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::ShortConvProject, &[norm, weight, activation])
    }

    /// The short convolution's gated taps of one row: `[activation]`.
    pub fn short_conv_rows(activation: Element) -> Self {
        Self::new(OperationClass::ShortConvRows, &[activation])
    }

    /// General routing: `[norm, router, activation]`.
    pub fn routed_select(norm: Element, router: Element, activation: Element) -> Self {
        Self::new(OperationClass::RoutedSelect, &[norm, router, activation])
    }

    /// One general decode expansion segment of the selected experts (gated
    /// or up-only): `[weight, activation]`.
    pub fn routed_expansion(gated: bool, weight: Element, activation: Element) -> Self {
        let class = if gated {
            OperationClass::RoutedGateUp
        } else {
            OperationClass::RoutedUp
        };
        Self::new(class, &[weight, activation])
    }

    /// The general decode down projection of the selected experts:
    /// `[weight, activation]`.
    pub fn routed_down(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::RoutedDown, &[weight, activation])
    }

    /// One paired gate/up segment: `[norm, weight, activation]`.
    pub fn dense_expand(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::DenseExpand, &[norm, weight, activation])
    }

    /// The down projection plus residual: `[weight, activation]`.
    pub fn dense_output(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::DenseOutput, &[weight, activation])
    }

    /// Router logits and top-k selection: `[norm, router, activation]`.
    pub fn routed_route(norm: Element, router: Element, activation: Element) -> Self {
        Self::new(OperationClass::RoutedRoute, &[norm, router, activation])
    }

    /// One routed gate/up segment (selected experts or shared expert):
    /// `[weight, activation]`.
    pub fn routed_expand(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::RoutedExpand, &[weight, activation])
    }

    /// One routed down segment: `[weight, activation]`.
    pub fn routed_output(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::RoutedOutput, &[weight, activation])
    }

    /// A post-norm sublayer's output projection into F32 rows:
    /// `[weight, activation]`.
    pub fn project_rows(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::ProjectRows, &[weight, activation])
    }

    /// A post-norm sublayer's row op: `[norm]`.
    pub fn post_norm_residual(norm: Element) -> Self {
        Self::new(OperationClass::PostNormResidual, &[norm])
    }

    /// The tail of parallel branches: `[norm]` (the branch and tail norms
    /// share it).
    pub fn moe_tail(norm: Element) -> Self {
        Self::new(OperationClass::MoeTail, &[norm])
    }

    /// A per-layer input gate: `[gate weight, activation]`.
    pub fn per_layer_gate(gate: Element, activation: Element) -> Self {
        Self::new(OperationClass::PerLayerGate, &[gate, activation])
    }

    /// The per-layer inputs of one row: `[table, norm]`.
    pub fn per_layer_inputs(table: Element, norm: Element) -> Self {
        Self::new(OperationClass::PerLayerInputs, &[table, norm])
    }

    /// Rows converted between dense representations: `[source, destination]`.
    pub fn import_rows(source: Element, destination: Element) -> Self {
        Self::new(OperationClass::ImportRows, &[source, destination])
    }

    /// Rows converted into a packed resident representation:
    /// `[source, destination]`.
    pub fn repack_rows(source: Element, destination: Element) -> Self {
        Self::new(OperationClass::RepackRows, &[source, destination])
    }

    /// F32 rows copied.
    pub fn copy_rows() -> Self {
        Self::new(OperationClass::CopyRows, &[])
    }

    /// A host table's rows gathered and uploaded.
    pub fn table_upload() -> Self {
        Self::new(OperationClass::TableUpload, &[])
    }

    /// The final norm of the output rows: `[norm, activation]`.
    pub fn readout_features(norm: Element, activation: Element) -> Self {
        Self::new(OperationClass::ReadoutFeatures, &[norm, activation])
    }

    /// The vocabulary projection: `[norm, weight, activation]`.
    pub fn readout_head(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::ReadoutHead, &[norm, weight, activation])
    }

    /// Selection of one token from an F32 logits row.
    pub fn sample_rows() -> Self {
        Self::new(OperationClass::SampleRows, &[])
    }

    /// The per-byte streaming time of `weight` into `activation` rows.
    pub fn weight_format(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::WeightFormat, &[weight, activation])
    }

    /// The dependency cost of one entry call.
    pub fn launch_dependency() -> Self {
        Self::new(OperationClass::LaunchDependency, &[])
    }

    /// The submission and completion cost of one step.
    pub fn step_submission() -> Self {
        Self::new(OperationClass::StepSubmission, &[])
    }
}

/// `class[bindings]`, e.g. `dense_expand[bf16,q4k@rows16,bf16]`.
impl std::fmt::Display for MeasurementKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bindings = self
            .bindings
            .iter()
            .map(|element| element.name())
            .collect::<Vec<_>>()
            .join(",");
        write!(formatter, "{}[{bindings}]", self.class.name())
    }
}

/// The head geometry of a decode attention launch: `kv_heads` key/value
/// heads, `group` query heads per key/value head, heads `width` wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadGeometry {
    pub kv_heads: u64,
    pub group: u64,
    pub width: u64,
}

/// What one measured point ran over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointShape {
    /// A size of a class whose cost follows bytes (or a fixed launch).
    Size,
    /// A weight-streaming launch of `rows` output rows over a `reduction`
    /// wide input, streaming `weight`.
    Launch {
        rows: u64,
        reduction: u64,
        weight: Element,
    },
    /// A decode attention launch at its head geometry; the point's bytes
    /// are the history it reads.
    Heads(HeadGeometry),
}

/// One timed point of a class. `bytes` is what one launch streams at this
/// point; `samples` are per-launch device seconds of every repeated sample.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredPoint {
    pub shape: PointShape,
    pub bytes: u64,
    pub samples: Vec<f64>,
}

/// A weight-streaming class's cost at its reference representation: a
/// launch floor, and seconds per streamed byte by the launch's output rows
/// (strictly ascending rows). Rows between two measured counts interpolate
/// in log rows; beyond the ends they take the nearest end's value.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionCost {
    pub launch_seconds: f64,
    /// The representation the class was timed at.
    pub weight: Element,
    pub seconds_per_byte: Vec<(u64, f64)>,
}

/// Decode attention's cost: a launch floor, and seconds per history byte at
/// the reference head geometry, scaled by one factor per geometry axis. Each
/// axis curve is strictly ascending, holds the reference value at factor 1,
/// and interpolates in log value, clamped to its ends.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryCost {
    pub launch_seconds: f64,
    pub seconds_per_byte: f64,
    pub reference: HeadGeometry,
    pub kv_heads: Vec<(u64, f64)>,
    pub group: Vec<(u64, f64)>,
    pub width: Vec<(u64, f64)>,
}

impl HistoryCost {
    /// Seconds per history byte at `heads`.
    pub fn seconds_per_byte(&self, heads: HeadGeometry) -> f64 {
        self.seconds_per_byte
            * log_interpolate(&self.kv_heads, heads.kv_heads)
            * log_interpolate(&self.group, heads.group)
            * log_interpolate(&self.width, heads.width)
    }
}

impl ProjectionCost {
    /// Seconds per streamed byte of a launch of `rows` output rows.
    pub fn seconds_per_byte(&self, rows: u64) -> f64 {
        log_interpolate(&self.seconds_per_byte, rows)
    }
}

/// `curve` at `at`: linear in log `at` between the two nearest values,
/// clamped to the ends. `curve` is nonempty and strictly ascending.
fn log_interpolate(curve: &[(u64, f64)], at: u64) -> f64 {
    let at = at.max(1);
    let [(first, first_value), ..] = curve else {
        unreachable!("a cost curve has at least one value");
    };
    if at <= *first {
        return *first_value;
    }
    for pair in curve.windows(2) {
        let [(low, low_value), (high, high_value)] = pair else {
            unreachable!("windows of two");
        };
        if at <= *high {
            let position =
                ((at as f64).ln() - (*low as f64).ln()) / ((*high as f64).ln() - (*low as f64).ln());
            return low_value + (high_value - low_value) * position;
        }
    }
    curve[curve.len() - 1].1
}

/// How a class's launch time depends on what it runs over.
#[derive(Clone, Debug, PartialEq)]
pub enum CostModel {
    /// One measured time per launch.
    PerLaunch { seconds: f64 },
    /// Launch cost plus a cost per byte, from two sizes.
    Linear(StreamingCost),
    Projection(ProjectionCost),
    History(HistoryCost),
    /// Seconds per byte.
    PerByte { seconds_per_byte: f64 },
}

/// A class's cost at the median samples, plus the measured variation: the
/// slowest and fastest samples of its points, relative to their medians.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassCost {
    pub model: CostModel,
    /// `max(slowest / median)` over the class's points; at least 1.
    pub slow_factor: f64,
    /// `min(fastest / median)` over the class's points; at most 1.
    pub fast_factor: f64,
}

/// Seconds for a set of launches at the fastest, median and slowest measured
/// behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SecondsBand {
    pub fast: f64,
    pub median: f64,
    pub slow: f64,
}

impl SecondsBand {
    pub const ZERO: Self = Self {
        fast: 0.0,
        median: 0.0,
        slow: 0.0,
    };

    pub fn plus(self, other: Self) -> Self {
        Self {
            fast: self.fast + other.fast,
            median: self.median + other.median,
            slow: self.slow + other.slow,
        }
    }
}

/// The median of `samples`; the mean of the middle two for an even count.
pub(crate) fn median(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

/// The line through two measured sizes, taken as measured: its slope and
/// intercept each held at zero or above. Timing noise can make a larger size
/// measure no slower (no rate) or a small size slower than the line's
/// extension (no floor); the estimate then carries that measured value.
fn two_point_line(small: (u64, f64), large: (u64, f64)) -> StreamingCost {
    let ((small_bytes, small_seconds), (large_bytes, large_seconds)) =
        if small.0 <= large.0 { (small, large) } else { (large, small) };
    let span = large_bytes.saturating_sub(small_bytes) as f64;
    let seconds_per_byte = if span > 0.0 {
        ((large_seconds - small_seconds) / span).max(0.0)
    } else {
        0.0
    };
    StreamingCost {
        launch_seconds: (small_seconds - seconds_per_byte * small_bytes as f64).max(0.0),
        seconds_per_byte,
    }
}

/// Seconds per byte of a launch measured at `seconds` over `bytes` above a
/// `floor`; held at zero or above.
fn above_floor(seconds: f64, floor: f64, bytes: u64) -> f64 {
    ((seconds - floor) / bytes.max(1) as f64).max(0.0)
}

impl ClassCost {
    /// The cost of `class` from its measured points at their medians, and
    /// the extreme samples relative to each point's median. Measured values
    /// are taken as they are; only points the measurement itself did not
    /// produce correctly (no finite samples, or a point set that is not the
    /// class's measurement plan) are errors.
    pub fn from_points(class: OperationClass, points: &[MeasuredPoint]) -> Result<Self, String> {
        let medians = points
            .iter()
            .map(|point| {
                median(&point.samples)
                    .map(|median| median.max(0.0))
                    .ok_or_else(|| {
                        format!(
                            "{} point at {} bytes has no finite samples",
                            class.name(),
                            point.bytes
                        )
                    })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let wrong = || format!("{} has {} measured points", class.name(), points.len());
        let model = match (class.cost_shape(), points, medians.as_slice()) {
            (CostShape::PerLaunch, [_], [seconds]) => CostModel::PerLaunch { seconds: *seconds },
            (CostShape::Linear, [small, large], [small_median, large_median]) => CostModel::Linear(
                two_point_line((small.bytes, *small_median), (large.bytes, *large_median)),
            ),
            (CostShape::PerByte, [point], [seconds]) => CostModel::PerByte {
                seconds_per_byte: *seconds / point.bytes.max(1) as f64,
            },
            (CostShape::Projection, points, medians) => {
                CostModel::Projection(projection_cost(class, points, medians)?)
            }
            (CostShape::History, points, medians) => {
                CostModel::History(history_cost(class, points, medians)?)
            }
            _ => return Err(wrong()),
        };
        let mut slow_factor = 1.0f64;
        let mut fast_factor = 1.0f64;
        // A zero cost has no relative spread: every band bound is zero.
        for (point, median) in points
            .iter()
            .zip(&medians)
            .filter(|(_, median)| **median > 0.0)
        {
            for sample in &point.samples {
                slow_factor = slow_factor.max(sample / median);
                fast_factor = fast_factor.min(sample / median);
            }
        }
        Ok(Self {
            model,
            slow_factor,
            fast_factor: fast_factor.max(0.0),
        })
    }

    /// Seconds of `launches` launches of a class whose cost follows bytes
    /// (`PerLaunch`, `Linear`), streaming `bytes` in total.
    pub fn plain_seconds(&self, launches: u64, bytes: u64) -> Option<SecondsBand> {
        let median = match &self.model {
            CostModel::PerLaunch { seconds } => seconds * launches as f64,
            CostModel::Linear(cost) => {
                cost.launch_seconds * launches as f64 + cost.seconds_per_byte * bytes as f64
            }
            CostModel::Projection(_) | CostModel::History(_) | CostModel::PerByte { .. } => {
                return None;
            }
        };
        Some(self.band(median))
    }

    /// `median` scaled by the class's measured variation.
    pub fn band(&self, median: f64) -> SecondsBand {
        SecondsBand {
            fast: median * self.fast_factor,
            median,
            slow: median * self.slow_factor,
        }
    }
}

/// The projection points: output-row counts at the largest reduction, and
/// one row count also timed at a smaller reduction, whose two sizes give the
/// launch floor.
fn projection_cost(
    class: OperationClass,
    points: &[MeasuredPoint],
    medians: &[f64],
) -> Result<ProjectionCost, String> {
    let launches = points
        .iter()
        .zip(medians)
        .map(|(point, median)| match point.shape {
            PointShape::Launch {
                rows,
                reduction,
                weight,
            } => Ok((rows, reduction, weight, point.bytes, *median)),
            _ => Err(format!("{} point is not a launch", class.name())),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let reduction = launches
        .iter()
        .map(|(_, reduction, ..)| *reduction)
        .max()
        .ok_or_else(|| format!("{} has no measured launch", class.name()))?;
    let weight = launches[0].2;
    let floor = launches
        .iter()
        .find(|(_, point_reduction, ..)| *point_reduction < reduction)
        .ok_or_else(|| format!("{} has no floor launch", class.name()))?;
    let wide = launches
        .iter()
        .find(|(rows, point_reduction, ..)| *rows == floor.0 && *point_reduction == reduction)
        .ok_or_else(|| format!("{} floor has no full-width launch", class.name()))?;
    let launch_seconds = two_point_line((floor.3, floor.4), (wide.3, wide.4)).launch_seconds;
    let mut seconds_per_byte = launches
        .iter()
        .filter(|(_, point_reduction, ..)| *point_reduction == reduction)
        .map(|(rows, _, _, bytes, seconds)| (*rows, above_floor(*seconds, launch_seconds, *bytes)))
        .collect::<Vec<_>>();
    seconds_per_byte.sort_by_key(|(rows, _)| *rows);
    seconds_per_byte.dedup_by_key(|(rows, _)| *rows);
    Ok(ProjectionCost {
        launch_seconds,
        weight,
        seconds_per_byte,
    })
}

/// The history points: the reference geometry at two depths (the floor and
/// its rate), and every other geometry at one depth, each differing from the
/// reference in one axis.
fn history_cost(
    class: OperationClass,
    points: &[MeasuredPoint],
    medians: &[f64],
) -> Result<HistoryCost, String> {
    let heads = points
        .iter()
        .zip(medians)
        .map(|(point, median)| match point.shape {
            PointShape::Heads(heads) => Ok((heads, point.bytes, *median)),
            _ => Err(format!("{} point is not a head geometry", class.name())),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let reference = heads
        .iter()
        .find(|(candidate, ..)| {
            heads
                .iter()
                .filter(|(other, ..)| other == candidate)
                .count()
                == 2
        })
        .map(|(heads, ..)| *heads)
        .ok_or_else(|| format!("{} has no reference geometry at two depths", class.name()))?;
    let [shallow, deep] = heads
        .iter()
        .filter(|(candidate, ..)| *candidate == reference)
        .map(|(_, bytes, seconds)| (*bytes, *seconds))
        .collect::<Vec<_>>()[..]
    else {
        unreachable!("the reference has two points");
    };
    let line = two_point_line(shallow, deep);
    let rate = |bytes, seconds| above_floor(seconds, line.launch_seconds, bytes);
    // A reference rate lost to timing noise is the deeper point's own rate.
    let deeper = if shallow.0 >= deep.0 { shallow } else { deep };
    let reference_rate = if line.seconds_per_byte > 0.0 {
        line.seconds_per_byte
    } else {
        rate(deeper.0, deeper.1)
    }
    .max(f64::MIN_POSITIVE);
    let mut kv_heads = vec![(reference.kv_heads, 1.0)];
    let mut group = vec![(reference.group, 1.0)];
    let mut width = vec![(reference.width, 1.0)];
    for (candidate, bytes, seconds) in &heads {
        if *candidate == reference {
            continue;
        }
        let factor = rate(*bytes, *seconds) / reference_rate;
        match (
            candidate.kv_heads != reference.kv_heads,
            candidate.group != reference.group,
            candidate.width != reference.width,
        ) {
            (true, false, false) => kv_heads.push((candidate.kv_heads, factor)),
            (false, true, false) => group.push((candidate.group, factor)),
            (false, false, true) => width.push((candidate.width, factor)),
            _ => {
                return Err(format!(
                    "{} point {candidate:?} differs from the reference in more than one axis",
                    class.name()
                ))
            }
        }
    }
    for curve in [&mut kv_heads, &mut group, &mut width] {
        curve.sort_by_key(|(value, _)| *value);
        curve.dedup_by_key(|(value, _)| *value);
    }
    Ok(HistoryCost {
        launch_seconds: line.launch_seconds,
        seconds_per_byte: reference_rate,
        reference,
        kv_heads,
        group,
        width,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum ClassMeasurement {
    Measured {
        points: Vec<MeasuredPoint>,
        cost: ClassCost,
    },
    /// Formed on the device and not timed: an exact binding of a class
    /// whose cost is measured under its cost key.
    Formed,
    /// The backend cannot form this binding. This is compatibility
    /// evidence, not a measurement failure.
    Unsupported { reason: String },
}

/// What a basis was measured on. Every field participates in cache identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasisIdentity {
    pub engine_build: String,
    pub backend: String,
    /// Seismic's device and toolchain identity (`Device::tuning_identity`).
    pub device: String,
    pub protocol_version: u32,
}

impl BasisIdentity {
    /// The identity of a basis measured on `device` by this build. The
    /// native kernel bundle's identity is part of the execution
    /// implementation, so it is folded into the build.
    pub fn for_device(device: &seismic::Device, engine_build: &str) -> Self {
        Self {
            engine_build: format!("{engine_build}+kernels.{}", magnitude_kernels::IDENTITY),
            backend: device.backend().as_str().to_owned(),
            device: device.tuning_identity(),
            protocol_version: MEASUREMENT_PROTOCOL_VERSION,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementBasis {
    pub identity: BasisIdentity,
    pub classes: Vec<(MeasurementKey, ClassMeasurement)>,
}

impl MeasurementBasis {
    pub fn get(&self, key: &MeasurementKey) -> Option<&ClassMeasurement> {
        self.classes
            .iter()
            .find_map(|(candidate, measurement)| (candidate == key).then_some(measurement))
    }

    /// The measured cost under `key`.
    pub fn cost(&self, key: &MeasurementKey) -> Option<&ClassCost> {
        match self.get(key) {
            Some(ClassMeasurement::Measured { cost, .. }) => Some(cost),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(bytes: u64, samples: &[f64]) -> MeasuredPoint {
        MeasuredPoint {
            shape: PointShape::Size,
            bytes,
            samples: samples.to_vec(),
        }
    }

    fn launch(rows: u64, reduction: u64, seconds: f64) -> MeasuredPoint {
        MeasuredPoint {
            shape: PointShape::Launch {
                rows,
                reduction,
                weight: Element::bf16(),
            },
            bytes: rows * reduction * 2,
            samples: vec![seconds],
        }
    }

    fn heads(kv_heads: u64, group: u64, width: u64, bytes: u64, seconds: f64) -> MeasuredPoint {
        MeasuredPoint {
            shape: PointShape::Heads(HeadGeometry {
                kv_heads,
                group,
                width,
            }),
            bytes,
            samples: vec![seconds],
        }
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-30), "{a} != {b}");
    }

    #[test]
    fn linear_cost_uses_point_medians_and_extreme_sample_ratios() {
        let cost = ClassCost::from_points(
            OperationClass::SampleRows,
            &[
                point(1_000_000, &[1.1e-4, 1.0e-4, 0.9e-4]),
                point(9_000_000, &[5.0e-4, 5.5e-4, 4.0e-4, 5.0e-4]),
            ],
        )
        .unwrap();
        let CostModel::Linear(linear) = cost.model else {
            panic!("sampling is linear");
        };
        close(linear.seconds_per_byte, 5.0e-11);
        close(linear.launch_seconds, 5.0e-5);
        close(cost.slow_factor, 1.1);
        close(cost.fast_factor, 0.8);
        let band = cost.plain_seconds(2, 4_000_000).unwrap();
        close(band.median, 1.0e-4 + 2.0e-4);
        close(band.slow, band.median * 1.1);
        close(band.fast, band.median * 0.8);
    }

    #[test]
    fn noisy_two_point_fits_keep_the_measured_values() {
        // A larger size measured faster: no rate, the small size's time.
        let flat = ClassCost::from_points(
            OperationClass::SampleRows,
            &[point(1_000_000, &[2.0e-4]), point(9_000_000, &[1.0e-4])],
        )
        .unwrap();
        assert_eq!(
            flat.model,
            CostModel::Linear(StreamingCost {
                launch_seconds: 2.0e-4,
                seconds_per_byte: 0.0,
            })
        );
        // A rate whose extension passes below zero: no floor.
        let steep = ClassCost::from_points(
            OperationClass::SampleRows,
            &[point(1_000_000, &[1.0e-3]), point(10_000_000, &[2.0e-2])],
        )
        .unwrap();
        let CostModel::Linear(line) = steep.model else {
            panic!("sampling is linear");
        };
        assert_eq!(line.launch_seconds, 0.0);
        close(line.seconds_per_byte, 1.9e-2 / 9.0e6);
    }

    #[test]
    fn a_point_set_that_is_not_the_plan_is_an_error() {
        assert!(ClassCost::from_points(OperationClass::SampleRows, &[point(1, &[1.0])]).is_err());
        assert!(ClassCost::from_points(
            OperationClass::EmbeddingRows,
            &[point(1, &[1.0]), point(2, &[2.0])]
        )
        .is_err());
        assert!(
            ClassCost::from_points(OperationClass::EmbeddingRows, &[point(1, &[])]).is_err()
        );
        assert!(ClassCost::from_points(
            OperationClass::EmbeddingRows,
            &[point(1, &[f64::NAN])]
        )
        .is_err());
    }

    #[test]
    fn projection_floor_and_row_rates_interpolate_in_log_rows() {
        // Floor 10 µs; 1 ps/B at 1,000 rows, 0.5 ps/B from 100,000 rows.
        let floor = 1e-5;
        let at = |rows: u64, reduction: u64, rate: f64| {
            launch(rows, reduction, floor + rate * (rows * reduction * 2) as f64)
        };
        let cost = ClassCost::from_points(
            OperationClass::DenseExpand,
            &[
                at(1_000, 4096, 1e-12),
                at(10_000, 4096, 0.75e-12),
                at(100_000, 4096, 0.5e-12),
                at(10_000, 1024, 0.75e-12),
            ],
        )
        .unwrap();
        let CostModel::Projection(projection) = cost.model else {
            panic!("dense expand is a projection");
        };
        close(projection.launch_seconds, floor);
        assert_eq!(projection.weight, Element::bf16());
        close(projection.seconds_per_byte(1_000), 1e-12);
        close(projection.seconds_per_byte(100), 1e-12);
        close(projection.seconds_per_byte(1_000_000), 0.5e-12);
        // Between 1,000 and 10,000 rows, linear in log rows.
        let position = (3_162f64.ln() - 1_000f64.ln()) / (10_000f64.ln() - 1_000f64.ln());
        close(
            projection.seconds_per_byte(3_162),
            1e-12 + (0.75e-12 - 1e-12) * position,
        );
        assert!(ClassCost::from_points(
            OperationClass::DenseExpand,
            &[at(1_000, 4096, 1e-12), at(10_000, 4096, 1e-12)]
        )
        .is_err());
    }

    #[test]
    fn history_factors_scale_the_reference_rate_per_axis() {
        let reference = (2, 8, 256);
        let rate = 1e-12;
        let floor = 5e-6;
        let at = |(kv, group, width): (u64, u64, u64), bytes: u64, scale: f64| {
            heads(kv, group, width, bytes, floor + rate * scale * bytes as f64)
        };
        let cost = ClassCost::from_points(
            OperationClass::AttentionDecode,
            &[
                at(reference, 1_000_000, 1.0),
                at(reference, 8_000_000, 1.0),
                at((1, 8, 256), 4_000_000, 2.0),
                at((8, 8, 256), 16_000_000, 0.5),
                at((2, 32, 256), 8_000_000, 1.5),
                at((2, 8, 128), 4_000_000, 1.2),
            ],
        )
        .unwrap();
        let CostModel::History(history) = cost.model else {
            panic!("attention decode is a history cost");
        };
        close(history.launch_seconds, floor);
        close(history.seconds_per_byte, rate);
        let rate_at = |kv_heads, group, width| {
            history.seconds_per_byte(HeadGeometry {
                kv_heads,
                group,
                width,
            })
        };
        close(rate_at(2, 8, 256), rate);
        close(rate_at(1, 32, 128), rate * 2.0 * 1.5 * 1.2);
        // Halfway in log kv heads between 2 and 8.
        close(rate_at(4, 8, 256), rate * 0.75);
        // Clamped beyond the measured ends.
        close(rate_at(16, 1, 512), rate * 0.5);
        assert!(ClassCost::from_points(
            OperationClass::AttentionDecode,
            &[at(reference, 1_000_000, 1.0), at((1, 4, 256), 1_000_000, 1.0)]
        )
        .is_err());
    }

    #[test]
    fn weight_format_is_seconds_per_byte() {
        let cost =
            ClassCost::from_points(OperationClass::WeightFormat, &[point(4_000_000, &[1e-4])])
                .unwrap();
        assert_eq!(
            cost.model,
            CostModel::PerByte {
                seconds_per_byte: 2.5e-11
            }
        );
    }

    #[test]
    fn cost_keys_drop_representations() {
        let q4 = Element::named("q4k").unwrap();
        let bf16 = Element::bf16();
        assert_eq!(
            MeasurementKey::dense_expand(bf16, q4, bf16).cost(),
            MeasurementKey::new(OperationClass::DenseExpand, &[bf16])
        );
        assert_eq!(
            MeasurementKey::repack_rows(Element::f32(), q4).cost(),
            MeasurementKey::new(OperationClass::RepackRows, &[])
        );
        let step = MeasurementKey::delta_step(bf16);
        assert_eq!(step.cost(), step);
    }

    #[test]
    fn class_names_round_trip() {
        for class in OperationClass::ALL {
            assert_eq!(OperationClass::named(class.name()), Some(class));
        }
    }
}
