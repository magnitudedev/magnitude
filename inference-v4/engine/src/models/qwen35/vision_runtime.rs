//! Owner-local image execution. Preparation owns every pending operand; only
//! successful numerical completion publishes a shareable conditioning lease.
use super::{
    preparation::{spatial_controls, ImagePatches},
    vision::{Description, Geometry},
};
use crate::{
    execution::{Composition, CompositionSpec},
    inputs::media::DType as HostDType,
    weights::{descriptor::WeightDescriptor, residency::ResidentWeight},
};
use seismic_lang::{
    sir::Program,
    types::{DType, Elem},
};
use seismic_runtime::{
    plan::{InvocationResults, PlanCompiler, Settings, Submission},
    Buffer, Device, Error,
};
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

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
    pub fn complete(mut self) -> Result<Features, String> {
        self.submission.execute_sequential()?;
        Ok(self.features)
    }
}
pub struct Encoder {
    device: Rc<Device>,
    geometry: Geometry,
    program: Program,
    settings: Settings,
    stem: CompositionSpec,
    blocks: Vec<CompositionSpec>,
    merger: CompositionSpec,
}
fn names(values: &[&str]) -> HashSet<String> {
    values.iter().map(|s| s.to_string()).collect()
}
fn result_buffer(results: &InvocationResults, path: &[u32]) -> Result<Buffer, String> {
    results
        .iter()
        .find(|result| result.path == path && result.plane.is_empty())
        .map(|result| result.buffer.clone())
        .ok_or_else(|| format!("owned result path {path:?} has no dense buffer"))
}
fn spec(
    entry: &str,
    shapes: &[(&str, usize)],
    weights: HashMap<String, ResidentWeight>,
    external: &[&str],
    intermediates: &[&str],
    norm: bool,
) -> Result<CompositionSpec, String> {
    Ok(CompositionSpec {
        entry: entry.into(),
        shapes: shapes
            .iter()
            .map(|(name, n)| {
                Ok((
                    name.to_string(),
                    i64::try_from(*n).map_err(|_| "vision dimension overflow")?,
                ))
            })
            .collect::<Result<_, String>>()?,
        elements: HashMap::from([("A".into(), Elem::Dtype(DType::BF16))]),
        weights,
        external: names(external),
        intermediates: names(intermediates),
        scalars: if norm {
            HashMap::from([("epsilon".into(), 1e-6)])
        } else {
            HashMap::new()
        },
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
                ("M", 1),
                ("C", g.image.channels),
                ("T", g.image.temporal_patch),
                ("P", g.image.patch),
                ("H", g.hidden),
                ("L", g.table_side * g.table_side),
            ],
            bind(vec![
                ("weight", &description.patch.weight),
                ("bias", &description.patch.bias),
                ("table", &description.positions),
            ])?,
            &["pixels", "indices", "coefficients"],
            &[],
            false,
        )?;
        let mut blocks = Vec::new();
        for b in &description.blocks {
            blocks.push(spec(
                "qwen_vision_block",
                &[
                    ("M", 1),
                    ("H", g.heads),
                    ("P", g.hidden / g.heads / 4),
                    ("F", g.intermediate),
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
                &[],
                true,
            )?);
        }
        let merger = spec(
            "qwen_vision_merger",
            &[("M", 1), ("G", group), ("H", g.hidden), ("D", g.output)],
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
        let mut compiler = PlanCompiler::new(&self.device, &self.program, self.settings.clone());
        let mut prepare =
            |template: &CompositionSpec,
             count: usize,
             bindings: HashMap<String, Buffer>,
             controls: &[&str]|
             -> Result<(Submission, seismic_runtime::plan::InvocationResults), Error> {
                let mut spec = template.clone();
                spec.shapes.insert(
                    "M".into(),
                    i64::try_from(count).map_err(|_| "vision rows overflow")?,
                );
                let composition =
                    Composition::compile(&mut compiler, spec)?.control_inputs(controls)?;
                let submission = composition.prepare(&bindings, &HashMap::new())?;
                let results = submission.results_for(0)?.clone();
                Ok((submission, results))
            };
        let (mut submission, stem_results) = prepare(
            &self.stem,
            rows,
            HashMap::from([
                ("pixels".into(), pixels),
                ("indices".into(), indices),
                ("coefficients".into(), coefficients),
            ]),
            &["indices"],
        )?;
        let mut hidden = result_buffer(&stem_results, &[])?;
        for block in &self.blocks {
            let (prepared, results) = prepare(
                block,
                rows,
                HashMap::from([
                    ("hidden".into(), hidden),
                    ("coordinates".into(), coordinates.clone()),
                ]),
                &["coordinates"],
            )?;
            submission.append(prepared);
            hidden = result_buffer(&results, &[])?;
        }
        let (prepared, results) = prepare(
            &self.merger,
            output_rows,
            HashMap::from([("hidden".into(), hidden)]),
            &[],
        )?;
        submission.append(prepared);
        let compact = result_buffer(&results, &[])?;
        let conversion = spec(
            "qwen_vision_feature_output",
            &[("M", output_rows), ("D", g.output)],
            HashMap::new(),
            &["source"],
            &[],
            false,
        )?;
        let (prepared, results) = prepare(
            &conversion,
            output_rows,
            HashMap::from([("source".into(), compact)]),
            &[],
        )?;
        submission.append(prepared);
        let buffer = result_buffer(&results, &[])?;
        Ok(PendingImage {
            submission,
            features: Features {
                identity: image.identity.clone(),
                rows: output_rows,
                width: g.output,
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
