//! Device measurements used by metadata-only model assessment.
//!
//! A runner measures shipped native entries against synthetic resident
//! tensors. It never inspects a model artifact, tunes a parameter, or loads a
//! model. Each measurement class owns its geometry and argument construction;
//! Seismic owns formation, device timing and the native implementation.

use crate::{
    AttentionShape, FeedForwardProgramSlot, MixerProgramSlot, ModelLoadPlan, ProgramPlan,
    StreamingCost,
};
use magnitude_model_contracts::{ModelDefinition, WeightKind, WeightScope};
use magnitude_model_kernels::{dense_expand, dense_output};
use magnitude_model_state::KvCodec;
use seismic::{
    generated, Device, Element, MeasureOptions, Measurement, NativeSpecialization, Tensor,
};

const DENSE_OUTPUT_HIDDEN: u64 = 4096;
const SMALL_WEIGHT_BYTES: u64 = 2 * 1024 * 1024;
const LARGE_WEIGHT_BYTES: u64 = 64 * 1024 * 1024;
const ROTATION_BYTES: u64 = 128 * 1024 * 1024;

/// One observed point. `weight_bytes` is the actual resident allocation read
/// by one launch; the timing excludes native formation and tensor creation.
#[derive(Clone, Debug)]
pub struct StreamingPoint {
    pub hidden: u64,
    pub features: u64,
    pub weight_bytes: u64,
    pub native_artifact: String,
    pub timing: Measurement,
}

/// The first streaming class: one default-configured dense output launch at
/// two weight sizes. The measured slope and intercept are kept with their
/// points so later prediction can reject an unavailable class honestly.
#[derive(Clone, Debug)]
pub struct DenseOutputStreaming {
    pub device_identity: String,
    pub timing_protocol: MeasureOptions,
    pub weight: Element,
    pub activation: Element,
    pub small: StreamingPoint,
    pub large: StreamingPoint,
    /// A noisy or nonphysical two-point fit retains its raw measurements but
    /// cannot supply a speed coefficient.
    pub cost: Result<StreamingCost, String>,
}

/// A paired gate/up projection measured in its shipped default configuration.
/// Both matrices use the same resident representation; mixed bindings remain
/// unmeasured until the catalog requires them.
#[derive(Clone, Debug)]
pub struct DenseExpandStreaming {
    pub device_identity: String,
    pub timing_protocol: MeasureOptions,
    pub weight: Element,
    pub activation: Element,
    pub small: StreamingPoint,
    pub large: StreamingPoint,
    pub cost: Result<StreamingCost, String>,
}

/// The portion of one decode step served by the dense-output streaming
/// measurement. Its bytes come from the model's resident weight plan and its
/// launch count from the same program plan used for native preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseOutputDemand {
    pub weight: Element,
    pub activation: Element,
    pub launches: u64,
    pub resident_bytes: u64,
}

/// Header-derived paired projection work for one plain decode step. The
/// gate and up matrices may have different resident encodings; the current
/// paired runner supplies a cost only when their measured binding matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseExpandDemand {
    pub norm: Element,
    pub gate: Element,
    pub up: Element,
    pub activation: Element,
    pub launches: u64,
    pub resident_bytes: u64,
}

/// Header-derived Q/K/V projection work for a plain decode step. This is a
/// separate measured class: its one launch streams three matrices, and its
/// geometry and bindings must match a device measurement before costing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionProjectDemand {
    pub shape: AttentionShape,
    pub norm: Element,
    pub query_gate: Element,
    pub key: Element,
    pub value: Element,
    pub activation: Element,
    pub launches: u64,
    pub resident_bytes: u64,
}

/// One target readout launch per plain decode step. The resident bytes are
/// the matrix's physical representation, even when its storage is tied to
/// the embedding table and charged only once for memory fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadoutDemand {
    pub norm: Element,
    pub weight: Element,
    pub activation: Element,
    pub resident_bytes: u64,
}

/// The measured-class inputs for one plain target decode step. Derive the
/// production program once per model; checking its metadata is more expensive
/// than grouping these demands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssessmentDecodeDemand {
    pub dense_output: Vec<DenseOutputDemand>,
    pub dense_expand: Vec<DenseExpandDemand>,
    pub attention_project: Vec<AttentionProjectDemand>,
    pub readout: ReadoutDemand,
}

impl AssessmentDecodeDemand {
    pub fn from_model(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
    ) -> Result<Self, String> {
        let program = load
            .program_plan(definition, codec)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            dense_output: DenseOutputDemand::from_plan(&program, load)?,
            dense_expand: DenseExpandDemand::from_plan(&program, load)?,
            attention_project: AttentionProjectDemand::from_plan(&program, load)?,
            readout: ReadoutDemand::from_plan(&program, load)?,
        })
    }
}

impl ReadoutDemand {
    fn from_plan(program: &ProgramPlan, load: &ModelLoadPlan) -> Result<Self, String> {
        let binding = program.target().readout();
        let resident = |kind| {
            load.target()
                .iter()
                .find(|weight| weight.role.scope == WeightScope::Target && weight.role.kind == kind)
                .ok_or_else(|| format!("{kind:?} target readout weight is absent"))
        };
        let norm = resident(WeightKind::OutputNorm)?;
        let output = resident(WeightKind::Output)?;
        if (norm.resident, output.resident) != (binding.norm, binding.weight) {
            return Err("readout binding disagrees with the resident load plan".into());
        }
        Ok(Self {
            norm: binding.norm,
            weight: binding.weight,
            activation: binding.activation,
            resident_bytes: output.resident_bytes,
        })
    }
}

impl AttentionProjectDemand {
    fn from_plan(program: &ProgramPlan, load: &ModelLoadPlan) -> Result<Vec<Self>, String> {
        let mut groups: Vec<Self> = Vec::new();
        for (index, block) in program.target().blocks().iter().enumerate() {
            let MixerProgramSlot::Attention(binding) = block.mixer() else {
                continue;
            };
            let scope = WeightScope::TargetBlock(
                u32::try_from(index).map_err(|_| "target block index exceeds u32")?,
            );
            let resident = |kind| {
                load.target()
                    .iter()
                    .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
                    .ok_or_else(|| format!("{kind:?} weight is absent for {scope:?}"))
            };
            let norm = resident(WeightKind::InputNorm)?;
            let query_gate = resident(WeightKind::QueryGate)?;
            let key = resident(WeightKind::Key)?;
            let value = resident(WeightKind::Value)?;
            if (
                norm.resident,
                query_gate.resident,
                key.resident,
                value.resident,
            ) != (binding.norm, binding.query_gate, binding.key, binding.value)
            {
                return Err(format!("attention project binding disagrees for {scope:?}"));
            }
            let bytes = query_gate
                .resident_bytes
                .checked_add(key.resident_bytes)
                .and_then(|bytes| bytes.checked_add(value.resident_bytes))
                .ok_or("attention project resident bytes overflow")?;
            if let Some(group) = groups.iter_mut().find(|group| {
                group.shape == binding.shape
                    && group.norm == binding.norm
                    && group.query_gate == binding.query_gate
                    && group.key == binding.key
                    && group.value == binding.value
                    && group.activation == binding.activation
            }) {
                group.launches = group
                    .launches
                    .checked_add(1)
                    .ok_or("attention project launch count overflow")?;
                group.resident_bytes = group
                    .resident_bytes
                    .checked_add(bytes)
                    .ok_or("attention project resident byte count overflow")?;
            } else {
                groups.push(Self {
                    shape: binding.shape,
                    norm: binding.norm,
                    query_gate: binding.query_gate,
                    key: binding.key,
                    value: binding.value,
                    activation: binding.activation,
                    launches: 1,
                    resident_bytes: bytes,
                });
            }
        }
        Ok(groups)
    }
}

impl DenseExpandDemand {
    fn from_plan(program: &ProgramPlan, load: &ModelLoadPlan) -> Result<Vec<Self>, String> {
        let mut groups: Vec<Self> = Vec::new();
        for (index, block) in program.target().blocks().iter().enumerate() {
            let FeedForwardProgramSlot::Dense(binding) = block.feed_forward() else {
                continue;
            };
            let scope = WeightScope::TargetBlock(
                u32::try_from(index).map_err(|_| "target block index exceeds u32")?,
            );
            let resident = |kind| {
                load.target()
                    .iter()
                    .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
                    .ok_or_else(|| format!("{kind:?} weight is absent for {scope:?}"))
            };
            let norm = resident(WeightKind::FeedForwardNorm)?;
            let gate = resident(WeightKind::DenseGate)?;
            let up = resident(WeightKind::DenseUp)?;
            if (norm.resident, gate.resident, up.resident)
                != (binding.norm, binding.gate, binding.up)
            {
                return Err(format!("dense expand binding disagrees for {scope:?}"));
            }
            let bytes = gate
                .resident_bytes
                .checked_add(up.resident_bytes)
                .ok_or("paired projection resident bytes overflow")?;
            if let Some(group) = groups.iter_mut().find(|group| {
                group.norm == binding.norm
                    && group.gate == binding.gate
                    && group.up == binding.up
                    && group.activation == binding.activation
            }) {
                group.launches = group
                    .launches
                    .checked_add(1)
                    .ok_or("dense expand launch count overflow")?;
                group.resident_bytes = group
                    .resident_bytes
                    .checked_add(bytes)
                    .ok_or("dense expand resident byte count overflow")?;
            } else {
                groups.push(Self {
                    norm: binding.norm,
                    gate: binding.gate,
                    up: binding.up,
                    activation: binding.activation,
                    launches: 1,
                    resident_bytes: bytes,
                });
            }
        }
        Ok(groups)
    }

    pub fn seconds_from(self, calibration: &DenseExpandStreaming) -> Option<f64> {
        if self.norm != Element::f32()
            || self.gate != calibration.weight
            || self.up != calibration.weight
            || self.activation != calibration.activation
        {
            return None;
        }
        let cost = calibration.cost.as_ref().ok()?;
        let seconds = cost.launch_seconds * self.launches as f64
            + cost.seconds_per_byte * self.resident_bytes as f64;
        (seconds.is_finite() && seconds > 0.0).then_some(seconds)
    }
}

impl DenseOutputDemand {
    fn from_plan(program: &ProgramPlan, load: &ModelLoadPlan) -> Result<Vec<Self>, String> {
        let mut groups: Vec<Self> = Vec::new();
        for (index, block) in program.target().blocks().iter().enumerate() {
            let FeedForwardProgramSlot::Dense(binding) = block.feed_forward() else {
                continue;
            };
            let scope = WeightScope::TargetBlock(
                u32::try_from(index).map_err(|_| "target block index exceeds u32")?,
            );
            let down = load
                .target()
                .iter()
                .find(|weight| {
                    weight.role.scope == scope && weight.role.kind == WeightKind::DenseDown
                })
                .ok_or_else(|| format!("dense output weight is absent for {scope:?}"))?;
            if down.resident != binding.down {
                return Err(format!(
                    "dense output weight representation disagrees for {scope:?}"
                ));
            }
            if let Some(group) = groups.iter_mut().find(|group| {
                group.weight == binding.down && group.activation == binding.activation
            }) {
                group.launches = group
                    .launches
                    .checked_add(1)
                    .ok_or("dense output launch count overflow")?;
                group.resident_bytes = group
                    .resident_bytes
                    .checked_add(down.resident_bytes)
                    .ok_or("dense output resident byte count overflow")?;
            } else {
                groups.push(Self {
                    weight: binding.down,
                    activation: binding.activation,
                    launches: 1,
                    resident_bytes: down.resident_bytes,
                });
            }
        }
        Ok(groups)
    }

    /// Only this kernel class's contribution, never a whole-model speed.
    pub fn seconds_from(self, calibration: &DenseOutputStreaming) -> Option<f64> {
        if self.weight != calibration.weight || self.activation != calibration.activation {
            return None;
        }
        let cost = calibration.cost.as_ref().ok()?;
        let seconds = cost.launch_seconds * self.launches as f64
            + cost.seconds_per_byte * self.resident_bytes as f64;
        (seconds.is_finite() && seconds > 0.0).then_some(seconds)
    }
}

pub struct DeviceMeasurementRunner<'a> {
    device: &'a Device,
    options: MeasureOptions,
}

impl<'a> DeviceMeasurementRunner<'a> {
    pub fn new(device: &'a Device) -> Self {
        Self {
            device,
            // Seven measured submissions at at least five milliseconds each
            // land within the proposal's 25–75 ms timing window per point.
            options: MeasureOptions {
                samples: 7,
                min_sample_seconds: 0.005,
            },
        }
    }

    pub fn measure_dense_output(
        &self,
        weight: Element,
        activation: Element,
    ) -> Result<DenseOutputStreaming, String> {
        let small = self.dense_output_point(weight, activation, SMALL_WEIGHT_BYTES)?;
        let large = self.dense_output_point(weight, activation, LARGE_WEIGHT_BYTES)?;
        let cost = StreamingCost::from_samples(
            small.weight_bytes,
            small.timing.median,
            large.weight_bytes,
            large.timing.median,
        );
        Ok(DenseOutputStreaming {
            device_identity: self.device.tuning_identity(),
            timing_protocol: self.options.clone(),
            weight,
            activation,
            small,
            large,
            cost,
        })
    }

    pub fn measure_dense_expand(
        &self,
        weight: Element,
        activation: Element,
    ) -> Result<DenseExpandStreaming, String> {
        let small = self.dense_expand_point(weight, activation, SMALL_WEIGHT_BYTES)?;
        let large = self.dense_expand_point(weight, activation, LARGE_WEIGHT_BYTES)?;
        let cost = StreamingCost::from_samples(
            small.weight_bytes,
            small.timing.median,
            large.weight_bytes,
            large.timing.median,
        );
        Ok(DenseExpandStreaming {
            device_identity: self.device.tuning_identity(),
            timing_protocol: self.options.clone(),
            weight,
            activation,
            small,
            large,
            cost,
        })
    }

    fn dense_expand_point(
        &self,
        weight: Element,
        activation: Element,
        target_bytes: u64,
    ) -> Result<StreamingPoint, String> {
        let hidden = DENSE_OUTPUT_HIDDEN;
        let features = features_for_bytes(weight, hidden, target_bytes.div_ceil(2))?;
        let implementation = generated::native_implementation_for_backend::<dense_expand::Entry>(
            self.device.backend(),
        )
        .map_err(|error| error.to_string())?
        .ok_or_else(|| {
            format!(
                "dense_expand has no native implementation for {}",
                self.device.backend().as_str()
            )
        })?;
        let statics = assessment_statics(self.device.backend(), hidden, features);
        let defaults = implementation
            .default_specialization(&statics)
            .map_err(|error| error.to_string())?;
        let kernel = dense_expand::native_for_device_with(
            self.device,
            dense_expand::Elements {
                NW: Element::f32(),
                GW: weight,
                UW: weight,
                A: activation,
            },
            &defaults,
        )
        .map_err(|error| error.to_string())?;

        let residual = Tensor::zeros(self.device, Element::f32(), &[1, hidden])
            .map_err(|error| error.to_string())?;
        let norm = Tensor::zeros(self.device, Element::f32(), &[hidden])
            .map_err(|error| error.to_string())?;
        let out_rows =
            Tensor::zeros(self.device, Element::i32(), &[1]).map_err(|error| error.to_string())?;
        let matrix_bytes = weight
            .canonical_byte_len(&[features, hidden])
            .map_err(|error| error.to_string())?;
        let weight_bytes = matrix_bytes
            .checked_mul(2)
            .ok_or("paired streaming weight byte count overflow")?;
        let copies = ROTATION_BYTES.div_ceil(weight_bytes).clamp(1, 64) as usize;
        let weights = (0..copies)
            .map(|_| {
                Ok((
                    Tensor::zeros(self.device, weight, &[features, hidden])
                        .map_err(|error| error.to_string())?,
                    Tensor::zeros(self.device, weight, &[features, hidden])
                        .map_err(|error| error.to_string())?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let rotation = weights
            .iter()
            .map(|(gate_weight, up_weight)| dense_expand::Args {
                residual: &residual,
                norm: &norm,
                gate_weight,
                up_weight,
                out_rows: &out_rows,
                eps: 1.0e-5,
            })
            .collect();
        let timing = kernel
            .measure(rotation, &self.options)
            .map_err(|error| error.to_string())?;
        Ok(StreamingPoint {
            hidden,
            features,
            weight_bytes,
            native_artifact: kernel.artifact().0.clone(),
            timing,
        })
    }

    fn dense_output_point(
        &self,
        weight: Element,
        activation: Element,
        target_bytes: u64,
    ) -> Result<StreamingPoint, String> {
        let hidden = DENSE_OUTPUT_HIDDEN;
        let features = features_for_bytes(weight, hidden, target_bytes)?;
        let implementation = generated::native_implementation_for_backend::<dense_output::Entry>(
            self.device.backend(),
        )
        .map_err(|error| error.to_string())?
        .ok_or_else(|| {
            format!(
                "dense_output has no native implementation for {}",
                self.device.backend().as_str()
            )
        })?;
        let statics = assessment_statics(self.device.backend(), hidden, features);
        let defaults = implementation
            .default_specialization(&statics)
            .map_err(|error| error.to_string())?;
        let kernel = dense_output::native_for_device_with(
            self.device,
            dense_output::Elements {
                DW: weight,
                A: activation,
            },
            &defaults,
        )
        .map_err(|error| error.to_string())?;

        // Zero is a valid synthetic payload for the registered resident
        // formats. Allocation and initialization happen on the device; no
        // model or host-side weight conversion is involved.
        let residual = Tensor::zeros(self.device, Element::f32(), &[1, hidden])
            .map_err(|error| error.to_string())?;
        let product = Tensor::zeros(self.device, activation, &[1, features])
            .map_err(|error| error.to_string())?;
        let out_rows =
            Tensor::zeros(self.device, Element::i32(), &[1]).map_err(|error| error.to_string())?;
        let weight_bytes = weight
            .canonical_byte_len(&[hidden, features])
            .map_err(|error| error.to_string())?;
        let copies = ROTATION_BYTES.div_ceil(weight_bytes).clamp(1, 64) as usize;
        let weights = (0..copies)
            .map(|_| Tensor::zeros(self.device, weight, &[hidden, features]))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        let rotation = weights
            .iter()
            .map(|down_weight| dense_output::Args {
                residual: &residual,
                product: &product,
                down_weight,
                out_rows: &out_rows,
            })
            .collect();
        let timing = kernel
            .measure(rotation, &self.options)
            .map_err(|error| error.to_string())?;
        Ok(StreamingPoint {
            hidden,
            features,
            weight_bytes,
            native_artifact: kernel.artifact().0.clone(),
            timing,
        })
    }
}

/// CPU forms derive the feature width from the tensor view rather than a
/// static invocation dimension. GPU forms use both dimensions as compile-time
/// specialization values for their streaming shapes.
fn assessment_statics(
    backend: seismic::BackendName,
    hidden: u64,
    features: u64,
) -> NativeSpecialization {
    if backend == seismic::BackendName::Cpu {
        NativeSpecialization::new()
    } else {
        NativeSpecialization::new()
            .with_static("H", hidden)
            .with_static("F", features)
    }
}

/// Smallest aligned feature width whose canonical resident weight reaches
/// the requested size. The registry remains the byte-layout authority.
fn features_for_bytes(weight: Element, hidden: u64, target: u64) -> Result<u64, String> {
    if hidden == 0 || target == 0 {
        return Err("streaming measurement needs positive geometry and bytes".into());
    }
    let group = weight.logical_group().unwrap_or(1);
    let bytes = |groups: u64| -> Result<u64, String> {
        let features = groups
            .checked_mul(group)
            .ok_or("streaming feature width overflow")?;
        weight
            .canonical_byte_len(&[hidden, features])
            .map_err(|error| error.to_string())
    };
    let mut upper = 1u64;
    while bytes(upper)? < target {
        upper = upper
            .checked_mul(2)
            .ok_or("streaming feature search overflow")?;
    }
    let mut lower = 0u64;
    while upper - lower > 1 {
        let middle = lower + (upper - lower) / 2;
        if bytes(middle)? < target {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    upper
        .checked_mul(group)
        .ok_or_else(|| "streaming feature width overflow".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::Layout;

    #[test]
    fn measurement_sizes_follow_registry_layout() {
        let weight = Element::stored("q4k", Layout::Rows16).unwrap();
        for target in [SMALL_WEIGHT_BYTES, LARGE_WEIGHT_BYTES] {
            let features = features_for_bytes(weight, DENSE_OUTPUT_HIDDEN, target).unwrap();
            let actual = weight
                .canonical_byte_len(&[DENSE_OUTPUT_HIDDEN, features])
                .unwrap();
            assert!(actual >= target);
            assert_eq!(features % weight.logical_group().unwrap(), 0);
            let previous = features - weight.logical_group().unwrap();
            if previous > 0 {
                assert!(
                    weight
                        .canonical_byte_len(&[DENSE_OUTPUT_HIDDEN, previous])
                        .unwrap()
                        < target
                );
            }
        }
    }

    #[test]
    fn assessment_statics_follow_backend_dimension_contract() {
        let cpu = assessment_statics(seismic::BackendName::Cpu, 4096, 19968);
        assert!(cpu.statics().is_empty());
        let metal = assessment_statics(seismic::BackendName::Metal, 4096, 19968);
        assert_eq!(metal.statics().get("H"), Some(&4096));
        assert_eq!(metal.statics().get("F"), Some(&19968));
    }

    #[test]
    fn dense_output_demand_uses_program_slots_and_resident_weight_bytes() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let demands =
            AssessmentDecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        let [demand] = demands.dense_output.as_slice() else {
            panic!("fixture has one dense output");
        };
        let down = load
            .target()
            .iter()
            .find(|weight| weight.role.kind == WeightKind::DenseDown)
            .unwrap();
        assert_eq!(demand.launches, 1);
        assert_eq!(demand.weight, down.resident);
        assert_eq!(demand.resident_bytes, down.resident_bytes);
    }

    #[test]
    fn dense_expand_demand_counts_both_matrices_from_the_load_plan() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let demands =
            AssessmentDecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        let [demand] = demands.dense_expand.as_slice() else {
            panic!("fixture has one paired projection");
        };
        let gate = load
            .target()
            .iter()
            .find(|weight| weight.role.kind == WeightKind::DenseGate)
            .unwrap();
        let up = load
            .target()
            .iter()
            .find(|weight| weight.role.kind == WeightKind::DenseUp)
            .unwrap();
        assert_eq!(demand.launches, 1);
        assert_eq!(demand.gate, gate.resident);
        assert_eq!(demand.up, up.resident);
        assert_eq!(
            demand.resident_bytes,
            gate.resident_bytes + up.resident_bytes
        );
    }

    #[test]
    fn attention_project_demand_counts_three_planned_matrices() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let demands =
            AssessmentDecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
        let [demand] = demands.attention_project.as_slice() else {
            panic!("fixture has one attention project");
        };
        let planned = load
            .target()
            .iter()
            .filter(|weight| {
                matches!(
                    weight.role.kind,
                    WeightKind::QueryGate | WeightKind::Key | WeightKind::Value
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(demand.launches, 1);
        assert_eq!(planned.len(), 3);
        assert_eq!(
            demand.resident_bytes,
            planned
                .iter()
                .map(|weight| weight.resident_bytes)
                .sum::<u64>()
        );
    }

    #[test]
    fn readout_demand_uses_the_planned_output_weight() {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows16,
        )
        .unwrap();
        let demand = AssessmentDecodeDemand::from_model(&definition, &load, KvCodec::Dense)
            .unwrap()
            .readout;
        let output = load
            .target()
            .iter()
            .find(|weight| weight.role.kind == WeightKind::Output)
            .unwrap();
        assert_eq!(demand.weight, output.resident);
        assert_eq!(demand.resident_bytes, output.resident_bytes);
    }
}
