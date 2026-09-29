//! The measurement basis: the fixed set of generic operation classes the
//! execution implementation can run on one device, each timed once with its
//! shipped default configuration on synthetic resident tensors. Every model
//! is assessed analytically against this basis; nothing here is per model.
//!
//! The basis is also the qualified support set. A class that could not be
//! formed on the backend is recorded as unsupported, and a model that needs
//! it is incompatible with this device.

use crate::{AttentionShape, StreamingCost};
use seismic::Element;

/// Changes whenever the measured classes, key rules, sizes or timing rules
/// change, so a cached basis from an older protocol is never reused. The
/// declared plan itself is part of the cache key (`persist`).
pub const MEASUREMENT_PROTOCOL_VERSION: u32 = 5;

/// One native entry a plain target decode step launches. A plain step is one
/// row through the embedding entry graph, every decoder block graph (decode
/// row class) and the selection readout graph (`readout_features_rows`,
/// `readout_head_rows`, then `sample_rows`; an unshaped selection, as greedy
/// decoding and plain temperature-1 sampling take). Two classes are not
/// entries: the cost every entry call adds when it depends on the previous
/// one, and the cost of submitting and waiting for one step.
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
    DenseExpand,
    DenseOutput,
    RoutedRoute,
    RoutedExpand,
    RoutedOutput,
    ReadoutFeatures,
    ReadoutHead,
    SampleRows,
    /// Extra device time of an entry call that depends on the previous call
    /// (as every launch of a decode step does), beyond its independent
    /// back-to-back time that the entry classes measure.
    LaunchDependency,
    /// Host-observed time of submitting one step and waiting for its
    /// completion, beyond the device time of its launches.
    StepSubmission,
}

impl OperationClass {
    pub const ALL: [Self; 18] = [
        Self::EmbeddingRows,
        Self::AttentionProject,
        Self::AttentionDecode,
        Self::AttentionDecodeK8V4,
        Self::AttentionOutput,
        Self::DeltaProject,
        Self::DeltaStep,
        Self::DeltaOutput,
        Self::DenseExpand,
        Self::DenseOutput,
        Self::RoutedRoute,
        Self::RoutedExpand,
        Self::RoutedOutput,
        Self::ReadoutFeatures,
        Self::ReadoutHead,
        Self::SampleRows,
        Self::LaunchDependency,
        Self::StepSubmission,
    ];

    /// The native entry name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::EmbeddingRows => "embedding_rows",
            Self::AttentionProject => "gated_attention_project",
            Self::AttentionDecode => "gated_attention_decode",
            Self::AttentionDecodeK8V4 => "gated_attention_decode_k8v4",
            Self::AttentionOutput => "attention_output",
            Self::DeltaProject => "gated_delta_project",
            Self::DeltaStep => "gated_delta_step",
            Self::DeltaOutput => "gated_delta_output",
            Self::DenseExpand => "dense_expand",
            Self::DenseOutput => "dense_output",
            Self::RoutedRoute => "routed_route",
            Self::RoutedExpand => "routed_expand",
            Self::RoutedOutput => "routed_output",
            Self::ReadoutFeatures => "readout_features_rows",
            Self::ReadoutHead => "readout_head_rows",
            Self::SampleRows => "sample_rows",
            Self::LaunchDependency => "launch_dependency",
            Self::StepSubmission => "step_submission",
        }
    }

    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.name() == name)
    }

    /// How the class's launch time depends on what it streams:
    ///
    /// - Weight-streaming GEMV classes are shape-generic `Curve` costs: the
    ///   time of one launch as a function of the bytes it streams, measured
    ///   at a ladder of sizes and interpolated. Evidence: on Metal and CUDA
    ///   the time of these entries depends on the bytes of one launch, not
    ///   on which synthetic dimension carries them, but not linearly: on a
    ///   GB10 every entry sits on a latency floor below about 4 MB and then
    ///   streams at 200–250 GB/s, so a two-point line through a floor-bound
    ///   and a large size overstates launches of 5–20 MB by 15–30%
    ///   (2026-09-25 size ladders; production kernel times from nsys).
    /// - Attention decode is `Linear` in history bytes, but geometry-keyed:
    ///   its cost per KV byte differs 5–6× between 2 and 4 KV heads, and it
    ///   is linear from 1k to 262k tokens on Metal and GB10.
    /// - Sampling is `Linear` in the logits row it scans.
    /// - The recurrent step and routing are fixed work per geometry
    ///   (`PerLaunch`); the row embedding and feature norm are one-row
    ///   launches whose cost is the launch itself (`PerLaunch`).
    /// - The dependency and submission costs are fixed per call and per step
    ///   (`PerLaunch`).
    pub const fn cost_shape(self) -> CostShape {
        match self {
            Self::EmbeddingRows
            | Self::DeltaStep
            | Self::RoutedRoute
            | Self::ReadoutFeatures
            | Self::LaunchDependency
            | Self::StepSubmission => CostShape::PerLaunch,
            Self::AttentionDecode | Self::AttentionDecodeK8V4 | Self::SampleRows => {
                CostShape::Linear
            }
            Self::AttentionProject
            | Self::AttentionOutput
            | Self::DeltaProject
            | Self::DeltaOutput
            | Self::DenseExpand
            | Self::DenseOutput
            | Self::RoutedExpand
            | Self::RoutedOutput
            | Self::ReadoutHead => CostShape::Curve,
        }
    }

    /// Whether the class is a measured difference that can be zero: a device
    /// whose dependent calls cost no more than independent ones has no
    /// dependency cost, and one whose step submission is free has no
    /// submission cost. Every other class times real work and must be
    /// positive.
    pub const fn may_be_zero(self) -> bool {
        matches!(self, Self::LaunchDependency | Self::StepSubmission)
    }
}

/// How a class's measured cost is modelled (see [`OperationClass::cost_shape`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostShape {
    PerLaunch,
    Linear,
    Curve,
}

/// The identity of one measured class: the operation, its element bindings,
/// and only the geometry values that change its cost per streamed byte.
/// Demand derivation and the measurement plan build keys through the same
/// constructors, so equal keys mean the same measured behavior.
///
/// A weight-streaming entry whose segments bind different representations
/// (for example a Q4_K query and a Q6_K value in one attention projection)
/// contributes one demand term per segment representation; the key names the
/// segment's streamed representation, and the class is measured with every
/// segment in that representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeasurementKey {
    pub class: OperationClass,
    pub bindings: Vec<Element>,
    /// Sorted by name.
    pub geometry: Vec<(String, u64)>,
}

impl MeasurementKey {
    fn new(class: OperationClass, bindings: &[Element], geometry: &[(&str, u64)]) -> Self {
        let mut geometry = geometry
            .iter()
            .map(|(name, value)| ((*name).to_owned(), *value))
            .collect::<Vec<_>>();
        geometry.sort();
        Self {
            class,
            bindings: bindings.to_vec(),
            geometry,
        }
    }

    /// One table row decoded into activations: `[table, activation]`.
    pub fn embedding_rows(table: Element, activation: Element) -> Self {
        Self::new(OperationClass::EmbeddingRows, &[table, activation], &[])
    }

    /// One Q/K/V projection segment: `[norm, weight, activation]`.
    pub fn attention_project(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(
            OperationClass::AttentionProject,
            &[norm, weight, activation],
            &[],
        )
    }

    /// Fused attention over dense history (`affine` false) or affine K8/V4
    /// history: `[activation]` at the block's head geometry.
    pub fn attention_decode(affine: bool, shape: AttentionShape, activation: Element) -> Self {
        let class = if affine {
            OperationClass::AttentionDecodeK8V4
        } else {
            OperationClass::AttentionDecode
        };
        Self::new(
            class,
            &[activation],
            &[
                ("group", shape.group),
                ("kv_heads", shape.kv_heads),
                ("rotary_pairs", shape.rotary_pairs),
                ("width", shape.width),
            ],
        )
    }

    /// The attention output projection: `[weight, activation]`.
    pub fn attention_output(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::AttentionOutput, &[weight, activation], &[])
    }

    /// One recurrent projection segment: `[norm, weight, activation]`.
    pub fn delta_project(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(
            OperationClass::DeltaProject,
            &[norm, weight, activation],
            &[],
        )
    }

    /// The recurrent state advance of one row at its head geometry.
    pub fn delta_step(
        key_heads: u64,
        value_heads: u64,
        width: u64,
        convolution_width: u64,
        activation: Element,
    ) -> Self {
        Self::new(
            OperationClass::DeltaStep,
            &[activation],
            &[
                ("convolution_width", convolution_width),
                ("key_heads", key_heads),
                ("value_heads", value_heads),
                ("width", width),
            ],
        )
    }

    /// The gated recurrent output projection: `[norm, weight, activation]`.
    pub fn delta_output(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(
            OperationClass::DeltaOutput,
            &[norm, weight, activation],
            &[],
        )
    }

    /// One paired gate/up segment: `[norm, weight, activation]`.
    pub fn dense_expand(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(
            OperationClass::DenseExpand,
            &[norm, weight, activation],
            &[],
        )
    }

    /// The down projection plus residual: `[weight, activation]`.
    pub fn dense_output(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::DenseOutput, &[weight, activation], &[])
    }

    /// Router logits and top-k selection: `[norm, router, activation]` at the
    /// routing geometry.
    pub fn routed_route(
        norm: Element,
        router: Element,
        activation: Element,
        hidden: u64,
        experts: u64,
        selected: u64,
    ) -> Self {
        Self::new(
            OperationClass::RoutedRoute,
            &[norm, router, activation],
            &[
                ("experts", experts),
                ("hidden", hidden),
                ("selected", selected),
            ],
        )
    }

    /// One routed gate/up segment (selected experts or shared expert):
    /// `[weight, activation]`.
    pub fn routed_expand(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::RoutedExpand, &[weight, activation], &[])
    }

    /// One routed down segment: `[weight, activation]`.
    pub fn routed_output(weight: Element, activation: Element) -> Self {
        Self::new(OperationClass::RoutedOutput, &[weight, activation], &[])
    }

    /// The final norm of the output rows: `[norm, activation]`.
    pub fn readout_features(norm: Element, activation: Element) -> Self {
        Self::new(OperationClass::ReadoutFeatures, &[norm, activation], &[])
    }

    /// The vocabulary projection: `[norm, weight, activation]`.
    pub fn readout_head(norm: Element, weight: Element, activation: Element) -> Self {
        Self::new(
            OperationClass::ReadoutHead,
            &[norm, weight, activation],
            &[],
        )
    }

    /// Selection of one token from an F32 logits row.
    pub fn sample_rows() -> Self {
        Self::new(OperationClass::SampleRows, &[], &[])
    }

    /// The dependency cost of one entry call.
    pub fn launch_dependency() -> Self {
        Self::new(OperationClass::LaunchDependency, &[], &[])
    }

    /// The submission and completion cost of one step.
    pub fn step_submission() -> Self {
        Self::new(OperationClass::StepSubmission, &[], &[])
    }

    pub fn geometry_value(&self, name: &str) -> Option<u64> {
        self.geometry
            .iter()
            .find_map(|(candidate, value)| (candidate == name).then_some(*value))
    }
}

/// One timed size of a class. `bytes` is what one launch streams at this
/// size; `samples` are per-launch device seconds of every repeated sample.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredPoint {
    pub bytes: u64,
    pub samples: Vec<f64>,
}

/// How a class's launch time depends on the bytes it streams.
#[derive(Clone, Debug, PartialEq)]
pub enum CostModel {
    /// Fixed-geometry class: one measured time per launch.
    PerLaunch { seconds: f64 },
    /// Launch cost plus a cost per streamed byte, from two sizes.
    Linear(StreamingCost),
    /// Median seconds of one launch at each measured launch size, strictly
    /// ascending in bytes. A launch between two sizes interpolates linearly;
    /// below the smallest it costs the smallest's time (the latency floor);
    /// above the largest it extends the last segment.
    Curve(Vec<(u64, f64)>),
}

impl CostModel {
    /// Seconds of one launch streaming `bytes` under a `Curve`.
    fn curve_seconds(points: &[(u64, f64)], bytes: u64) -> f64 {
        let at = |index: usize| (points[index].0 as f64, points[index].1);
        let bytes = bytes as f64;
        let (first_bytes, first_seconds) = at(0);
        if bytes <= first_bytes {
            return first_seconds;
        }
        let upper = (1..points.len())
            .find(|&index| bytes <= points[index].0 as f64)
            .unwrap_or(points.len() - 1);
        let (low_bytes, low_seconds) = at(upper - 1);
        let (high_bytes, high_seconds) = at(upper);
        low_seconds + (high_seconds - low_seconds) * (bytes - low_bytes) / (high_bytes - low_bytes)
    }
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

/// The least-squares line through two measured sizes, constrained to a
/// nonnegative launch cost. When the unconstrained line's intercept is
/// negative (the cost per byte rises with size), the constrained optimum
/// passes through the origin. A cost that does not grow with size
/// establishes no streaming rate and is an error.
fn nonnegative_linear_fit(
    small_bytes: u64,
    small_seconds: f64,
    large_bytes: u64,
    large_seconds: f64,
) -> Result<StreamingCost, String> {
    if small_bytes == 0 || large_bytes <= small_bytes || large_seconds <= small_seconds {
        return Err("measurements do not establish a positive size slope".into());
    }
    let (small, large) = (small_bytes as f64, large_bytes as f64);
    let slope = (large_seconds - small_seconds) / (large - small);
    let intercept = small_seconds - slope * small;
    let cost = if intercept >= 0.0 {
        StreamingCost {
            launch_seconds: intercept,
            seconds_per_byte: slope,
        }
    } else {
        StreamingCost {
            launch_seconds: 0.0,
            seconds_per_byte: (small * small_seconds + large * large_seconds)
                / (small * small + large * large),
        }
    };
    if cost.seconds_per_byte.is_finite() && cost.seconds_per_byte > 0.0 {
        Ok(cost)
    } else {
        Err("measurements do not establish a finite positive rate".into())
    }
}

impl ClassCost {
    /// The cost of `class` from its measured points: the median fit (two
    /// points for a linear class, one for a per-launch class) and the
    /// extreme samples relative to each point's median. A fit that is not
    /// physical is an error, never a coefficient.
    pub fn from_points(class: OperationClass, points: &[MeasuredPoint]) -> Result<Self, String> {
        let medians = points
            .iter()
            .map(|point| {
                median(&point.samples)
                    .filter(|median| *median > 0.0 || (class.may_be_zero() && *median == 0.0))
                    .ok_or_else(|| {
                        format!(
                            "{} point at {} bytes has no positive finite median",
                            class.name(),
                            point.bytes
                        )
                    })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let model = match (class.cost_shape(), points, medians.as_slice()) {
            (CostShape::Linear, [small, large], [small_median, large_median]) => CostModel::Linear(
                nonnegative_linear_fit(small.bytes, *small_median, large.bytes, *large_median)
                    .map_err(|error| format!("{}: {error}", class.name()))?,
            ),
            (CostShape::PerLaunch, [_], [seconds]) => CostModel::PerLaunch { seconds: *seconds },
            (CostShape::Curve, points, medians) if points.len() >= 2 => {
                let curve = points
                    .iter()
                    .map(|point| point.bytes)
                    .zip(medians.iter().copied())
                    .collect::<Vec<_>>();
                if curve.windows(2).any(|pair| pair[1].0 <= pair[0].0) {
                    return Err(format!("{} sizes are not strictly ascending", class.name()));
                }
                // The extension above the largest size must be a positive
                // rate: a cost that does not grow with size streams nothing.
                let [.., (low_bytes, low_seconds), (high_bytes, high_seconds)] = curve[..] else {
                    unreachable!("a curve has at least two sizes");
                };
                if !(high_seconds > low_seconds && high_bytes > low_bytes) {
                    return Err(format!(
                        "{}: the largest sizes do not establish a positive rate",
                        class.name()
                    ));
                }
                CostModel::Curve(curve)
            }
            _ => {
                return Err(format!(
                    "{} has {} measured points",
                    class.name(),
                    points.len()
                ))
            }
        };
        let mut slow_factor = 1.0f64;
        let mut fast_factor = 1.0f64;
        // A zero cost has no relative spread: every band bound is zero.
        for (point, median) in points.iter().zip(&medians).filter(|(_, median)| **median > 0.0) {
            for sample in &point.samples {
                slow_factor = slow_factor.max(sample / median);
                fast_factor = fast_factor.min(sample / median);
            }
        }
        if !(slow_factor.is_finite() && fast_factor.is_finite() && fast_factor >= 0.0) {
            return Err(format!("{} sample spread is not finite", class.name()));
        }
        Ok(Self {
            model,
            slow_factor,
            fast_factor,
        })
    }

    /// Seconds of `launches` launches streaming `bytes` in total. Under a
    /// `Curve`, `bytes` belong to launches that each stream `launch_bytes`
    /// (every segment of one entry call), and cost their share of those
    /// launches' time: `bytes × time(launch_bytes) / launch_bytes`.
    pub fn seconds(&self, launches: u64, bytes: u64, launch_bytes: u64) -> SecondsBand {
        let median = match &self.model {
            CostModel::PerLaunch { seconds } => seconds * launches as f64,
            CostModel::Linear(cost) => {
                cost.launch_seconds * launches as f64 + cost.seconds_per_byte * bytes as f64
            }
            CostModel::Curve(points) => {
                if bytes == 0 {
                    0.0
                } else {
                    CostModel::curve_seconds(points, launch_bytes) * bytes as f64
                        / launch_bytes as f64
                }
            }
        };
        SecondsBand {
            fast: median * self.fast_factor,
            median,
            slow: median * self.slow_factor,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ClassMeasurement {
    Measured {
        points: Vec<MeasuredPoint>,
        cost: ClassCost,
    },
    /// The backend cannot form this class. This is compatibility evidence,
    /// not a measurement failure.
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
            engine_build: format!(
                "{engine_build}+kernels.{}",
                magnitude_model_kernels::IDENTITY
            ),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(bytes: u64, samples: &[f64]) -> MeasuredPoint {
        MeasuredPoint {
            bytes,
            samples: samples.to_vec(),
        }
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
        assert!((linear.seconds_per_byte - 5.0e-11).abs() < 1e-20);
        assert!((linear.launch_seconds - 5.0e-5).abs() < 1e-12);
        assert!((cost.slow_factor - 1.1).abs() < 1e-12);
        assert!((cost.fast_factor - 0.8).abs() < 1e-12);
        let band = cost.seconds(2, 4_000_000, 0);
        assert!((band.median - (1.0e-4 + 2.0e-4)).abs() < 1e-12);
        assert!((band.slow - band.median * 1.1).abs() < 1e-15);
        assert!((band.fast - band.median * 0.8).abs() < 1e-15);
    }

    #[test]
    fn per_launch_cost_is_the_median_and_factors_bracket_one() {
        let cost = ClassCost::from_points(
            OperationClass::DeltaStep,
            &[point(0, &[2.0e-5, 2.0e-5, 2.0e-5])],
        )
        .unwrap();
        assert_eq!(cost.model, CostModel::PerLaunch { seconds: 2.0e-5 });
        assert_eq!((cost.slow_factor, cost.fast_factor), (1.0, 1.0));
        assert!((cost.seconds(3, 123, 0).median - 6.0e-5).abs() < 1e-18);
    }

    #[test]
    fn nonphysical_fits_and_wrong_point_counts_are_errors() {
        assert!(ClassCost::from_points(
            OperationClass::SampleRows,
            &[point(1_000_000, &[2.0e-4]), point(9_000_000, &[1.0e-4])],
        )
        .is_err());
        assert!(ClassCost::from_points(OperationClass::SampleRows, &[point(1, &[1.0])]).is_err());
        assert!(ClassCost::from_points(
            OperationClass::DeltaStep,
            &[point(1, &[1.0]), point(2, &[2.0])]
        )
        .is_err());
        assert!(ClassCost::from_points(OperationClass::DeltaStep, &[point(1, &[])]).is_err());
    }

    #[test]
    fn only_a_difference_class_may_measure_zero() {
        let zero = ClassCost::from_points(
            OperationClass::LaunchDependency,
            &[point(0, &[0.0, 0.0, 1.0e-7])],
        )
        .unwrap();
        assert_eq!(zero.model, CostModel::PerLaunch { seconds: 0.0 });
        assert_eq!(zero.seconds(100, 0, 0).slow, 0.0);
        assert!(ClassCost::from_points(OperationClass::DeltaStep, &[point(0, &[0.0])]).is_err());
    }

    #[test]
    fn rising_cost_per_byte_fits_through_the_origin() {
        // The unconstrained line through (1 MB, 1 ms) and (10 MB, 20 ms) has
        // a negative intercept; the constrained least-squares line has none.
        let cost = ClassCost::from_points(
            OperationClass::SampleRows,
            &[point(1_000_000, &[1.0e-3]), point(10_000_000, &[2.0e-2])],
        )
        .unwrap();
        let CostModel::Linear(streaming) = cost.model else {
            panic!("sampling is linear");
        };
        assert_eq!(streaming.launch_seconds, 0.0);
        let expected = (1.0e6 * 1.0e-3 + 1.0e7 * 2.0e-2) / (1.0e12 + 1.0e14);
        assert!((streaming.seconds_per_byte - expected).abs() < 1e-20);
    }

    #[test]
    fn curve_interpolates_launch_sizes_and_apportions_segments() {
        let cost = ClassCost::from_points(
            OperationClass::DenseExpand,
            &[
                point(1_000_000, &[2.0e-5, 2.2e-5, 2.0e-5]),
                point(2_000_000, &[2.0e-5]),
                point(4_000_000, &[3.0e-5]),
                point(8_000_000, &[5.0e-5]),
            ],
        )
        .unwrap();
        let median = |launches, bytes, launch_bytes| cost.seconds(launches, bytes, launch_bytes).median;
        let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-15, "{a} != {b}");
        // The latency floor below the smallest size.
        close(median(1, 500_000, 500_000), 2.0e-5);
        // Interpolation between sizes; the launch count plays no part.
        close(median(0, 3_000_000, 3_000_000), 2.5e-5);
        // Extension of the last segment: 5 µs per MB.
        close(median(1, 10_000_000, 10_000_000), 6.0e-5);
        // A segment streaming a quarter of a 4 MB launch costs a quarter
        // of that launch's time, whatever launch count it carries.
        close(median(0, 1_000_000, 4_000_000), 0.75e-5);
        close(median(3, 0, 4_000_000), 0.0);
        assert!((cost.slow_factor - 1.1).abs() < 1e-12);
        // Sizes out of order, or a flat last segment, are no curve.
        assert!(ClassCost::from_points(
            OperationClass::DenseExpand,
            &[point(2_000_000, &[1.0e-5]), point(1_000_000, &[2.0e-5])]
        )
        .is_err());
        assert!(ClassCost::from_points(
            OperationClass::DenseExpand,
            &[point(1_000_000, &[1.0e-5]), point(2_000_000, &[1.0e-5])]
        )
        .is_err());
        assert!(
            ClassCost::from_points(OperationClass::DenseExpand, &[point(1, &[1.0])]).is_err()
        );
    }

    #[test]
    fn class_names_round_trip() {
        for class in OperationClass::ALL {
            assert_eq!(OperationClass::named(class.name()), Some(class));
        }
    }
}
