//! Owner-local image execution through the generated Seismic API. Model
//! orchestration stays here; kernel preparation, shape inference, allocation,
//! and execution stay behind typed generated bindings.

use super::{
    preparation::{spatial_controls, ImagePatches},
    vision::{Description, Geometry},
};
use crate::{
    inputs::media::DType as HostDType,
    kernels,
    weights::{descriptor::WeightDescriptor, residency::ResidentWeight},
    Error,
};
use seismic::{DType, Device, Element, Kernel, PreparationOptions, Tensor};
use std::rc::Rc;

#[derive(Clone)]
pub struct Features {
    identity: String,
    rows: usize,
    width: usize,
    buffer: Tensor,
}

impl Features {
    #[cfg(test)]
    pub(crate) fn test_fixture(
        identity: String,
        rows: usize,
        width: usize,
        buffer: Tensor,
    ) -> Self {
        assert_eq!(buffer.extents(), [rows as u64, width as u64]);
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
    pub fn storage_bytes(&self) -> u64 {
        self.buffer.storage_bytes()
    }
    /// Completed FP32 feature rows. Clones retain the same allocation.
    pub(crate) fn buffer(&self) -> &Tensor {
        &self.buffer
    }
}

struct Stem {
    kernel: Kernel<kernels::qwen_vision_stem::Entry>,
    weight: ResidentWeight,
    bias: ResidentWeight,
    table: ResidentWeight,
}

struct Block {
    kernel: Kernel<kernels::qwen_vision_block::Entry>,
    norm1_weight: ResidentWeight,
    norm1_bias: ResidentWeight,
    qkv_weight: ResidentWeight,
    qkv_bias: ResidentWeight,
    projection_weight: ResidentWeight,
    projection_bias: ResidentWeight,
    norm2_weight: ResidentWeight,
    norm2_bias: ResidentWeight,
    up_weight: ResidentWeight,
    up_bias: ResidentWeight,
    down_weight: ResidentWeight,
    down_bias: ResidentWeight,
}

struct Merger {
    kernel: Kernel<kernels::qwen_vision_merger::Entry>,
    norm_weight: ResidentWeight,
    norm_bias: ResidentWeight,
    up_weight: ResidentWeight,
    up_bias: ResidentWeight,
    down_weight: ResidentWeight,
    down_bias: ResidentWeight,
}

pub struct Encoder {
    device: Rc<Device>,
    geometry: Geometry,
    stem: Stem,
    blocks: Vec<Block>,
    merger: Merger,
    output: Kernel<kernels::qwen_vision_feature_output::Entry>,
}

fn resident(
    device: &Device,
    descriptor: &WeightDescriptor,
    import: &mut impl FnMut(&WeightDescriptor, DType) -> Result<ResidentWeight, Error>,
) -> Result<ResidentWeight, Error> {
    let weight = import(descriptor, DType::BF16)?;
    if !weight.belongs_to(device) || weight.descriptor() != descriptor {
        return Err(format!(
            "vision weight {} differs from its role or execution owner",
            descriptor.name
        )
        .into());
    }
    Ok(weight)
}

impl Encoder {
    pub fn load(
        device: Rc<Device>,
        description: &Description,
        preparation: PreparationOptions,
        mut import: impl FnMut(&WeightDescriptor, DType) -> Result<ResidentWeight, Error>,
    ) -> Result<Self, Error> {
        let g = &description.geometry;
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

        let stem_weight = resident(&device, &description.patch.weight, &mut import)?;
        let stem_bias = resident(&device, &description.patch.bias, &mut import)?;
        let table = resident(&device, &description.positions, &mut import)?;
        let stem = Stem {
            kernel: kernels::qwen_vision_stem::for_device_with(
                &device,
                preparation.clone(),
                kernels::qwen_vision_stem::Elements {
                    A: Element::bf16(),
                    W: stem_weight.element(),
                    B: stem_bias.element(),
                },
            )
            .map_err(|error| error.to_string())?,
            weight: stem_weight,
            bias: stem_bias,
            table,
        };

        let mut blocks = Vec::with_capacity(description.blocks.len());
        for source in &description.blocks {
            let norm1_weight = resident(&device, &source.norm1.weight, &mut import)?;
            let norm1_bias = resident(&device, &source.norm1.bias, &mut import)?;
            let qkv_weight = resident(&device, &source.qkv.weight, &mut import)?;
            let qkv_bias = resident(&device, &source.qkv.bias, &mut import)?;
            let projection_weight = resident(&device, &source.projection.weight, &mut import)?;
            let projection_bias = resident(&device, &source.projection.bias, &mut import)?;
            let norm2_weight = resident(&device, &source.norm2.weight, &mut import)?;
            let norm2_bias = resident(&device, &source.norm2.bias, &mut import)?;
            let up_weight = resident(&device, &source.up.weight, &mut import)?;
            let up_bias = resident(&device, &source.up.bias, &mut import)?;
            let down_weight = resident(&device, &source.down.weight, &mut import)?;
            let down_bias = resident(&device, &source.down.bias, &mut import)?;
            let kernel = kernels::qwen_vision_block::for_device_with(
                &device,
                preparation.clone(),
                kernels::qwen_vision_block::Elements {
                    A: Element::bf16(),
                    NW: norm1_weight.element(),
                    NB: norm1_bias.element(),
                    QW: qkv_weight.element(),
                    QB: qkv_bias.element(),
                    PW: projection_weight.element(),
                    PB: projection_bias.element(),
                    UW: up_weight.element(),
                    UB: up_bias.element(),
                    DW: down_weight.element(),
                    DB: down_bias.element(),
                },
            )
            .map_err(|error| error.to_string())?;
            blocks.push(Block {
                kernel,
                norm1_weight,
                norm1_bias,
                qkv_weight,
                qkv_bias,
                projection_weight,
                projection_bias,
                norm2_weight,
                norm2_bias,
                up_weight,
                up_bias,
                down_weight,
                down_bias,
            });
        }

        let norm_weight = resident(&device, &description.merger_norm.weight, &mut import)?;
        let norm_bias = resident(&device, &description.merger_norm.bias, &mut import)?;
        let up_weight = resident(&device, &description.merger_up.weight, &mut import)?;
        let up_bias = resident(&device, &description.merger_up.bias, &mut import)?;
        let down_weight = resident(&device, &description.merger_down.weight, &mut import)?;
        let down_bias = resident(&device, &description.merger_down.bias, &mut import)?;
        let merger = Merger {
            kernel: kernels::qwen_vision_merger::for_device_with(
                &device,
                preparation.clone(),
                kernels::qwen_vision_merger::Elements {
                    A: Element::bf16(),
                    NW: norm_weight.element(),
                    NB: norm_bias.element(),
                    UW: up_weight.element(),
                    UB: up_bias.element(),
                    DW: down_weight.element(),
                    DB: down_bias.element(),
                },
            )
            .map_err(|error| error.to_string())?,
            norm_weight,
            norm_bias,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
        };
        let output = kernels::qwen_vision_feature_output::for_device_with(
            &device,
            preparation,
            kernels::qwen_vision_feature_output::Elements { A: Element::bf16() },
        )
        .map_err(|error| error.to_string())?;

        Ok(Self {
            device,
            geometry: g.clone(),
            stem,
            blocks,
            merger,
            output,
        })
    }

    /// Encodes one image and publishes features only after every stage has
    /// completed successfully.
    pub fn encode(&self, image: &ImagePatches) -> Result<Features, Error> {
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
        let group = g.image.merge * g.image.merge;
        let output_rows = rows / group;
        let pixels = Tensor::from_host(
            &self.device,
            Element::f32(),
            &[
                rows as u64,
                g.image.channels as u64,
                g.image.temporal_patch as u64,
                g.image.patch as u64,
                g.image.patch as u64,
            ],
            image.pixels.data(),
        )?;
        let coordinates_bytes = controls
            .coordinates
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let coordinates = Tensor::from_host(
            &self.device,
            Element::i32(),
            &[rows as u64, 2],
            &coordinates_bytes,
        )?;
        let indices = &controls.indices;
        let indices_bytes = (0..rows)
            .flat_map(|row| (0..4).flat_map(move |i| indices[i][row].to_le_bytes()))
            .collect::<Vec<_>>();
        let indices = Tensor::from_host(
            &self.device,
            Element::i32(),
            &[rows as u64, 4],
            &indices_bytes,
        )?;
        let coefficients = &controls.coefficients;
        let coefficients_bytes = (0..rows)
            .flat_map(|row| (0..4).flat_map(move |i| coefficients[i][row].to_le_bytes()))
            .collect::<Vec<_>>();
        let coefficients = Tensor::from_host(
            &self.device,
            Element::f32(),
            &[rows as u64, 4],
            &coefficients_bytes,
        )?;

        let mut hidden = self
            .stem
            .kernel
            .call(kernels::qwen_vision_stem::Args {
                pixels: &pixels,
                weight: self.stem.weight.tensor(),
                bias: self.stem.bias.tensor(),
                table: self.stem.table.tensor(),
                indices: &indices,
                coefficients: &coefficients,
            })?
            .value;
        for block in &self.blocks {
            let block_input = hidden.reshape(&[
                rows as u64,
                self.geometry.heads as u64,
                4,
                (self.geometry.hidden / self.geometry.heads / 4) as u64,
            ])?;
            hidden = block
                .kernel
                .call(kernels::qwen_vision_block::Args {
                    hidden: &block_input,
                    coordinates: &coordinates,
                    norm1_weight: block.norm1_weight.tensor(),
                    norm1_bias: block.norm1_bias.tensor(),
                    qkv_weight: block.qkv_weight.tensor(),
                    qkv_bias: block.qkv_bias.tensor(),
                    projection_weight: block.projection_weight.tensor(),
                    projection_bias: block.projection_bias.tensor(),
                    norm2_weight: block.norm2_weight.tensor(),
                    norm2_bias: block.norm2_bias.tensor(),
                    up_weight: block.up_weight.tensor(),
                    up_bias: block.up_bias.tensor(),
                    down_weight: block.down_weight.tensor(),
                    down_bias: block.down_bias.tensor(),
                    epsilon: 1e-6,
                })?
                .value;
        }
        let compact = self
            .merger
            .kernel
            .call(kernels::qwen_vision_merger::Args {
                hidden: &hidden,
                norm_weight: self.merger.norm_weight.tensor(),
                norm_bias: self.merger.norm_bias.tensor(),
                up_weight: self.merger.up_weight.tensor(),
                up_bias: self.merger.up_bias.tensor(),
                down_weight: self.merger.down_weight.tensor(),
                down_bias: self.merger.down_bias.tensor(),
                epsilon: 1e-6,
            })?
            .value;
        let buffer = self
            .output
            .call(kernels::qwen_vision_feature_output::Args { source: &compact })?
            .value;
        Ok(Features {
            identity: image.identity.clone(),
            rows: output_rows,
            width: self.geometry.output,
            buffer,
        })
    }
}
