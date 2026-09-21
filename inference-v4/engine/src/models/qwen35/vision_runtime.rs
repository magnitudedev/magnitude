//! Owner-local image execution. Preparation owns every pending operand; only
//! successful numerical completion publishes a shareable conditioning lease.
//! `Encoder::prepare` is the explicit input-preparation phase: it compiles the
//! stage compositions for the image's exact geometry, in full, before the
//! pending image's single submission is built; completion executes prepared
//! invocations only.
use super::{
    preparation::{spatial_controls, ImagePatches},
    vision::{Description, Geometry},
};
use crate::{
    execution,
    inputs::media::DType as HostDType,
    preparation::{
        CompositionSpec, EnvelopeShape, PreparationSession, Program, Settings, WorkloadEnvelope,
    },
    weights::{descriptor::WeightDescriptor, residency::ResidentWeight},
};
use seismic_lang::types::{DType, Elem};
use seismic_runtime::{
    invocation::PreparedInvocation,
    plan::InvocationResults,
    submission::Submission,
    Buffer, Device,
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
};

use crate::Error;

#[derive(Clone)]
pub struct Features {
    identity: String,
    rows: usize,
    width: usize,
    buffer: Buffer,
}
impl Features {
    #[cfg(test)]
    pub(crate) fn test_fixture(
        identity: String,
        rows: usize,
        width: usize,
        buffer: Buffer,
    ) -> Self {
        assert_eq!(buffer.len(), rows * width * 4);
        Self {
            identity,
            rows,
            width,
            buffer,
        }
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn storage_bytes(&self) -> usize {
        self.buffer().len()
    }
    /// Completed FP32 feature rows. Clones retain the same charged allocation.
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}
pub struct PendingImage {
    submission: Submission,
    features: Features,
}
impl PendingImage {
    /// Consuming completion prevents publication after failure or cancellation.
    pub fn complete(self) -> Result<Features, String> {
        let PendingImage {
            submission,
            features,
        } = self;
        submission.execute().map_err(|e| e.to_string())?;
        Ok(features)
    }
}
/// One stage of the vision tower: its entry, static dimensions, imported
/// weights, external tensors, and fixed scalars. The row dimension is bound
/// exactly, per image, by the input-preparation phase.
struct StageSpec {
    entry: String,
    dimensions: Vec<(String, u64)>,
    weights: HashMap<String, ResidentWeight>,
    external: HashSet<String>,
    scalars: HashMap<String, f64>,
    controls: Vec<String>,
}
pub struct Encoder {
    device: Rc<Device>,
    geometry: Geometry,
    program: Program,
    settings: Settings,
    stem: StageSpec,
    blocks: Vec<StageSpec>,
    merger: StageSpec,
    output: usize,
}
fn result_buffer(results: &InvocationResults, path: &[u32]) -> Result<Buffer, Error> {
    execution::result_buffer(results, path).map_err(Error::from)
}
fn spec(
    entry: &str,
    dimensions: &[(&str, u64)],
    weights: HashMap<String, ResidentWeight>,
    external: &[&str],
    controls: &[&str],
    norm: bool,
) -> Result<StageSpec, String> {
    Ok(StageSpec {
        entry: entry.into(),
        dimensions: dimensions
            .iter()
            .map(|(name, value)| ((*name).into(), *value))
            .collect(),
        weights,
        external: external.iter().map(|s| (*s).to_string()).collect(),
        scalars: if norm {
            HashMap::from([("epsilon".into(), 1e-6)])
        } else {
            HashMap::new()
        },
        controls: controls.iter().map(|s| (*s).to_string()).collect(),
    })
}
impl Encoder {
    pub fn load(
        device: Rc<Device>,
        description: &Description,
        settings: Settings,
        mut import: impl FnMut(&WeightDescriptor, DType) -> Result<ResidentWeight, String>,
    ) -> Result<Self, String> {
        let g = &description.geometry;
        // Descriptions may also be constructed by callers, outside artifact parsing.
        let group = g
            .image
            .merge
            .checked_mul(g.image.merge)
            .ok_or("vision merge overflow")?;
        let patch_width = g.image.patch_width()?;
        if [
            g.hidden,
            g.intermediate,
            g.output,
            g.heads,
            g.depth,
            g.table_side,
        ]
        .iter()
        .any(|&n| n == 0 || n > i32::MAX as usize)
            || description.blocks.len() != g.depth
            || patch_width > i32::MAX as usize
            || g.hidden
                .checked_mul(3)
                .is_none_or(|n| n > i32::MAX as usize)
            || g.heads.checked_mul(4).is_none_or(|n| g.hidden % n != 0)
            || g.hidden
                .checked_mul(group)
                .is_none_or(|n| n > i32::MAX as usize)
            || g.table_side
                .checked_mul(g.table_side)
                .is_none_or(|n| n > i32::MAX as usize)
        {
            return Err("invalid vision encoder geometry".into());
        }
        let mut bind = |roles: Vec<(&str, &WeightDescriptor)>| -> Result<HashMap<String, ResidentWeight>, String> {
            roles
                .into_iter()
                .map(|(name, role)| {
                    let weight = import(role, DType::BF16)?;
                    if !weight.belongs_to(&device) || weight.descriptor() != role {
                        return Err(format!("vision weight {name} differs from its role or execution owner"));
                    }
                    Ok((name.into(), weight))
                })
                .collect()
        };
        let stem = spec(
            "qwen_vision_stem",
            &[
                ("C", g.image.channels as u64),
                ("T", g.image.temporal_patch as u64),
                ("P", g.image.patch as u64),
                ("H", g.hidden as u64),
                ("L", (g.table_side * g.table_side) as u64),
            ],
            bind(vec![
                ("weight", &description.patch.weight),
                ("bias", &description.patch.bias),
                ("table", &description.positions),
            ])?,
            &["pixels", "indices", "coefficients"],
            &["indices"],
            false,
        )?;
        let mut blocks = Vec::new();
        for b in &description.blocks {
            blocks.push(spec(
                "qwen_vision_block",
                &[
                    ("H", g.heads as u64),
                    ("P", (g.hidden / g.heads / 4) as u64),
                    ("F", g.intermediate as u64),
                ],
                bind(vec![
                    ("norm1_weight", &b.norm1.weight),
                    ("norm1_bias", &b.norm1.bias),
                    ("qkv_weight", &b.qkv.weight),
                    ("qkv_bias", &b.qkv.bias),
                    ("projection_weight", &b.projection.weight),
                    ("projection_bias", &b.projection.bias),
                    ("norm2_weight", &b.norm2.weight),
                    ("norm2_bias", &b.norm2.bias),
                    ("up_weight", &b.up.weight),
                    ("up_bias", &b.up.bias),
                    ("down_weight", &b.down.weight),
                    ("down_bias", &b.down.bias),
                ])?,
                &["hidden", "coordinates"],
                &["coordinates"],
                true,
            )?);
        }
        let merger = spec(
            "qwen_vision_merger",
            &[
                ("G", group as u64),
                ("H", g.hidden as u64),
                ("D", g.output as u64),
            ],
            bind(vec![
                ("norm_weight", &description.merger_norm.weight),
                ("norm_bias", &description.merger_norm.bias),
                ("up_weight", &description.merger_up.weight),
                ("up_bias", &description.merger_up.bias),
                ("down_weight", &description.merger_down.weight),
                ("down_bias", &description.merger_down.bias),
            ])?,
            &["hidden"],
            &[],
            true,
        )?;
        Ok(Self {
            device,
            geometry: g.clone(),
            program: super::program::program()?,
            settings,
            stem,
            blocks,
            merger,
            output: g.output,
        })
    }
    pub fn prepare(&self, image: &ImagePatches) -> Result<PendingImage, Error> {
        let g = &self.geometry;
        let controls = spatial_controls(image.grid, g.image.merge, g.table_side)?;
        let rows = controls.coordinates.len();
        if image.pixels.dtype() != HostDType::F32
            || image.pixels.shape() != [rows, g.image.patch_width()?]
            || image
                .pixels
                .data()
                .chunks_exact(4)
                .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        {
            return Err("image pixels differ from encoder geometry or are nonfinite".into());
        }
        let output_rows = rows / (g.image.merge * g.image.merge);
        let pixels = self.device.buffer_from(image.pixels.data())?;
        let coordinates = self.device.buffer_from(
            &controls
                .coordinates
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let indices = self.device.buffer_from(
            &(0..rows)
                .flat_map(|row| {
                    (0..4)
                        .flat_map(|i| controls.indices[i][row].to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?;
        let coefficients = self.device.buffer_from(
            &(0..rows)
                .flat_map(|row| {
                    (0..4)
                        .flat_map(|i| controls.coefficients[i][row].to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?;
        let mut session = PreparationSession::new(&self.device, &self.program, self.settings.clone());
        let conversion = spec(
            "qwen_vision_feature_output",
            &[("D", self.output as u64)],
            HashMap::new(),
            &["source"],
            &[],
            false,
        )
        .map_err(Error::from)?;
        let mut prepare = |stage: &StageSpec,
                           count: usize,
                           bindings: HashMap<String, Buffer>|
         -> Result<(PreparedInvocation, InvocationResults), Error> {
            let mut dimensions: BTreeMap<String, u64> =
                stage.dimensions.iter().cloned().collect();
            dimensions.insert("M".into(), count as u64);
            let shapes = dimensions
                .iter()
                .map(|(name, &value)| (name.clone(), EnvelopeShape::Exact(value)))
                .collect::<BTreeMap<String, EnvelopeShape>>();
            let envelope = WorkloadEnvelope::new(
                shapes,
                BTreeMap::from([("A".into(), Elem::Dtype(DType::BF16))]),
                Vec::new(),
            )?;
            let mut composition = session.prepare(CompositionSpec {
                entry: stage.entry.clone(),
                envelope,
                weights: stage.weights.clone(),
                external: stage.external.clone(),
                intermediates: HashSet::new(),
                scalars: stage.scalars.clone(),
            })?;
            if !stage.controls.is_empty() {
                let controls = stage
                    .controls
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                composition = composition.control_inputs(&controls)?;
            }
            let invocation =
                execution::invoke(&composition, &dimensions, &bindings, &HashMap::new())?;
            let results = execution::result_planes(&invocation);
            Ok((invocation, results))
        };
        let (invocation, stem_results) = prepare(
            &self.stem,
            rows,
            HashMap::from([
                ("pixels".into(), pixels),
                ("indices".into(), indices),
                ("coefficients".into(), coefficients),
            ]),
        )?;
        let mut submission = Submission::single(invocation);
        let mut hidden = result_buffer(&stem_results, &[])?;
        for block in &self.blocks {
            let (invocation, results) = prepare(
                block,
                rows,
                HashMap::from([
                    ("hidden".into(), hidden),
                    ("coordinates".into(), coordinates.clone()),
                ]),
            )?;
            submission.append(Submission::single(invocation));
            hidden = result_buffer(&results, &[])?;
        }
        let (invocation, results) = prepare(
            &self.merger,
            output_rows,
            HashMap::from([("hidden".into(), hidden)]),
        )?;
        submission.append(Submission::single(invocation));
        let compact = result_buffer(&results, &[])?;
        let (invocation, results) = prepare(
            &conversion,
            output_rows,
            HashMap::from([("source".into(), compact)]),
        )?;
        submission.append(Submission::single(invocation));
        let buffer = result_buffer(&results, &[])?;
        Ok(PendingImage {
            submission,
            features: Features {
                identity: image.identity.clone(),
                rows: output_rows,
                width: self.output,
                buffer,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a Metal device"]
    fn feature_leases_share_one_charge_until_last_owner_drops() {
        let device = Device::metal().unwrap();
        let features = Features {
            identity: "lifetime-only-fixture".into(),
            rows: 2,
            width: 3,
            buffer: device.buffer(24).unwrap(),
        };
        let retained = features.clone();
        assert_eq!(features.buffer().len(), 24);
        assert_eq!(device.memory_usage().charged, 24);
        drop(features);
        assert_eq!(retained.rows(), 2);
        assert_eq!(retained.width(), 3);
        assert_eq!(device.memory_usage().charged, 24);
        drop(retained);
        assert_eq!(device.memory_usage().charged, 0);
    }
}
