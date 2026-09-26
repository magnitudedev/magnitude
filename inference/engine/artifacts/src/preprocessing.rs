//! Family-neutral host image preprocessing. Model adapters supply the numeric
//! processor contract; this module owns decoding and tensor construction.

use crate::media::{DType, PreparedMedia, PreparedTensor, MAX_PREPARED_BYTES};
use image::{ImageDecoder, ImageReader};
use sha2::{Digest, Sha256};
use std::io::Cursor;

/// Maximum images admitted by one prepared request. Resource planning uses
/// the same bound for output leases retained across a request's lifetime.
pub const MAX_IMAGES_PER_REQUEST: usize = 16;
const ASPECT_LIMIT: usize = 200;

#[derive(Clone, Debug, PartialEq)]
pub struct ImageProcessorConfig {
    pub kind: String,
    pub patch: usize,
    pub merge: usize,
    pub temporal_patch: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImageProcessorConfig {
    pub fn validate(&self) -> Result<(), String> {
        let factor = self
            .patch
            .checked_mul(self.merge)
            .ok_or("image processor factor overflows")?;
        if self.kind.is_empty()
            || factor == 0
            || self.temporal_patch == 0
            || self.min_pixels == 0
            || self.min_pixels > self.max_pixels
            || self.mean.iter().any(|value| !value.is_finite())
            || self
                .std
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("invalid image processor configuration".into());
        }
        Ok(())
    }

    fn identity(&self) -> String {
        let mut digest = Sha256::new();
        for field in [
            self.kind.as_bytes(),
            b"bicubic:a=-0.5:scale-aware-antialias:rgb8",
            b"rescale:1/255",
            &self.patch.to_le_bytes(),
            &self.merge.to_le_bytes(),
            &self.temporal_patch.to_le_bytes(),
            &self.min_pixels.to_le_bytes(),
            &self.max_pixels.to_le_bytes(),
            &MAX_IMAGES_PER_REQUEST.to_le_bytes(),
            &ASPECT_LIMIT.to_le_bytes(),
        ] {
            digest.update((field.len() as u64).to_le_bytes());
            digest.update(field);
        }
        for values in [&self.mean, &self.std] {
            for value in values {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct ImageProcessor {
    config: ImageProcessorConfig,
    identity: String,
}

impl ImageProcessor {
    pub fn new(config: ImageProcessorConfig) -> Result<Self, String> {
        config.validate()?;
        let identity = config.identity();
        Ok(Self { config, identity })
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn smart_resize(&self, height: usize, width: usize) -> Result<(usize, usize), String> {
        if height == 0 || width == 0 {
            return Err("image dimensions must be nonzero".into());
        }
        let short = height.min(width);
        let long = height.max(width);
        if long > short.saturating_mul(ASPECT_LIMIT) {
            return Err("image aspect ratio exceeds processor limit".into());
        }
        let factor = self.config.patch * self.config.merge;
        let round_even = |value: usize| -> usize {
            let quotient = value / factor;
            let remainder = value % factor;
            let rounded = if remainder * 2 < factor {
                quotient
            } else if remainder * 2 > factor || quotient % 2 == 1 {
                quotient + 1
            } else {
                quotient
            };
            rounded.max(1) * factor
        };
        let mut output_height = round_even(height);
        let mut output_width = round_even(width);
        let pixels = output_height
            .checked_mul(output_width)
            .ok_or("resized image extent overflows")?;
        if pixels > self.config.max_pixels {
            let beta = ((height as f64 * width as f64) / self.config.max_pixels as f64).sqrt();
            output_height =
                (((height as f64 / beta) / factor as f64).floor() as usize).max(1) * factor;
            output_width =
                (((width as f64 / beta) / factor as f64).floor() as usize).max(1) * factor;
        } else if pixels < self.config.min_pixels {
            let beta = (self.config.min_pixels as f64 / (height as f64 * width as f64)).sqrt();
            output_height =
                ((height as f64 * beta / factor as f64).ceil() as usize).max(1) * factor;
            output_width = ((width as f64 * beta / factor as f64).ceil() as usize).max(1) * factor;
        }
        output_height
            .checked_mul(output_width)
            .filter(|pixels| *pixels <= self.config.max_pixels)
            .ok_or_else(|| String::from("resized image extent exceeds processor capacity"))?;
        Ok((output_height, output_width))
    }

    pub fn prepare(&self, encoded_images: &[Vec<u8>]) -> Result<PreparedMedia, String> {
        if encoded_images.is_empty() || encoded_images.len() > MAX_IMAGES_PER_REQUEST {
            return Err("image count is outside processor limits".into());
        }
        let width = 3usize
            .checked_mul(self.config.temporal_patch)
            .and_then(|value| value.checked_mul(self.config.patch))
            .and_then(|value| value.checked_mul(self.config.patch))
            .ok_or("image patch width overflows")?;
        let mut pixels = Vec::new();
        let mut grids = Vec::with_capacity(encoded_images.len() * 3 * 8);
        let mut rows = 0usize;
        for encoded in encoded_images {
            let reader = ImageReader::new(Cursor::new(encoded))
                .with_guessed_format()
                .map_err(|error| format!("image format detection failed: {error}"))?;
            let mut decoder = reader
                .into_decoder()
                .map_err(|error| format!("image decode setup failed: {error}"))?;
            let orientation = decoder
                .orientation()
                .map_err(|error| format!("image orientation failed: {error}"))?;
            let mut image = image::DynamicImage::from_decoder(decoder)
                .map_err(|error| format!("image decode failed: {error}"))?;
            image.apply_orientation(orientation);
            let (height, width_pixels) =
                self.smart_resize(image.height() as usize, image.width() as usize)?;
            u32::try_from(height).map_err(|_| "resized image height exceeds decoder domain")?;
            u32::try_from(width_pixels)
                .map_err(|_| "resized image width exceeds decoder domain")?;
            let source = image.to_rgb32f();
            let grid_height = height / self.config.patch;
            let grid_width = width_pixels / self.config.patch;
            rows = rows
                .checked_add(
                    grid_height
                        .checked_mul(grid_width)
                        .ok_or("image grid overflows")?,
                )
                .ok_or("image row count overflows")?;
            rows.checked_mul(width)
                .and_then(|values| values.checked_mul(4))
                .filter(|bytes| *bytes <= MAX_PREPARED_BYTES)
                .ok_or("prepared image pixels exceed capacity")?;
            let resized = resize_bicubic(&source, width_pixels, height)?;
            for value in [1i64, grid_height as i64, grid_width as i64] {
                grids.extend_from_slice(&value.to_le_bytes());
            }
            self.patchify(&resized, grid_height, grid_width, &mut pixels)?;
        }
        let pixel_values =
            PreparedTensor::new("pixel_values".into(), DType::F32, vec![rows, width], pixels)?;
        let image_grid_thw = PreparedTensor::new(
            "image_grid_thw".into(),
            DType::I64,
            vec![encoded_images.len(), 3],
            grids,
        )?;
        PreparedMedia::new(self.identity.clone(), vec![pixel_values, image_grid_thw])
    }

    fn patchify(
        &self,
        image: &image::Rgb32FImage,
        grid_height: usize,
        grid_width: usize,
        output: &mut Vec<u8>,
    ) -> Result<(), String> {
        let patch = self.config.patch;
        let merge = self.config.merge;
        if grid_height % merge != 0 || grid_width % merge != 0 {
            return Err("image grid is not merge aligned".into());
        }
        for block_y in 0..grid_height / merge {
            for block_x in 0..grid_width / merge {
                for merge_y in 0..merge {
                    for merge_x in 0..merge {
                        let patch_y = block_y * merge + merge_y;
                        let patch_x = block_x * merge + merge_x;
                        for channel in 0..3 {
                            for _frame in 0..self.config.temporal_patch {
                                for y in 0..patch {
                                    for x in 0..patch {
                                        let pixel = image.get_pixel(
                                            (patch_x * patch + x) as u32,
                                            (patch_y * patch + y) as u32,
                                        )[channel];
                                        let normalized = (pixel - self.config.mean[channel])
                                            / self.config.std[channel];
                                        output.extend_from_slice(&normalized.to_le_bytes());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct FilterSpan {
    first: usize,
    weights: Vec<f32>,
}

/// Build the same center-addressed, scale-aware cubic filter used by PIL's
/// bicubic resampler. Downscaling widens the two-pixel cubic support before
/// normalizing each destination sample, which is the antialiasing step.
fn cubic(value: f64) -> f64 {
    let value = value.abs();
    if value < 1.0 {
        ((1.5 * value - 2.5) * value) * value + 1.0
    } else if value < 2.0 {
        ((-0.5 * value + 2.5) * value - 4.0) * value + 2.0
    } else {
        0.0
    }
}

fn filter_spans(input: usize, output: usize) -> Result<Vec<FilterSpan>, String> {
    if input == 0 || output == 0 {
        return Err("bicubic resize dimensions must be nonzero".into());
    }
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let mut spans = Vec::with_capacity(output);
    for destination in 0..output {
        let center = (destination as f64 + 0.5) * scale;
        let first = (center - support + 0.5).floor().max(0.0) as usize;
        let end = (center + support + 0.5).floor().min(input as f64) as usize;
        if first >= end {
            return Err("bicubic resize produced an empty filter span".into());
        }
        let mut weights = (first..end)
            .map(|source| cubic((source as f64 + 0.5 - center) / filter_scale))
            .collect::<Vec<_>>();
        let sum = weights.iter().sum::<f64>();
        if !sum.is_finite() || sum == 0.0 {
            return Err("bicubic resize produced invalid filter weights".into());
        }
        let weights = weights
            .drain(..)
            .map(|weight| (weight / sum) as f32)
            .collect();
        spans.push(FilterSpan { first, weights });
    }
    Ok(spans)
}

fn resize_bicubic(
    source: &image::Rgb32FImage,
    output_width: usize,
    output_height: usize,
) -> Result<image::Rgb32FImage, String> {
    let source_width = source.width() as usize;
    let source_height = source.height() as usize;
    if source_width == output_width && source_height == output_height {
        return Ok(source.clone());
    }
    let horizontal = filter_spans(source_width, output_width)?;
    let vertical = filter_spans(source_height, output_height)?;
    let intermediate_len = output_width
        .checked_mul(source_height)
        .and_then(|value| value.checked_mul(3))
        .ok_or("bicubic intermediate extent overflows")?;
    let mut intermediate = vec![0.0f32; intermediate_len];
    for source_y in 0..source_height {
        for (destination_x, span) in horizontal.iter().enumerate() {
            for (offset, weight) in span.weights.iter().enumerate() {
                let pixel = source.get_pixel((span.first + offset) as u32, source_y as u32);
                let base = (source_y * output_width + destination_x) * 3;
                for channel in 0..3 {
                    intermediate[base + channel] += pixel[channel] * weight;
                }
            }
        }
    }
    let mut resized = image::Rgb32FImage::new(output_width as u32, output_height as u32);
    for (destination_y, span) in vertical.iter().enumerate() {
        for destination_x in 0..output_width {
            let mut value = [0.0f32; 3];
            for (offset, weight) in span.weights.iter().enumerate() {
                let base = ((span.first + offset) * output_width + destination_x) * 3;
                for channel in 0..3 {
                    value[channel] += intermediate[base + channel] * weight;
                }
            }
            // PIL keeps an RGB image after resize: overshoot is clipped and
            // each channel is rounded back to 8-bit before rescaling. Matching
            // that quantization is necessary for the 1e-3 normalized-pixel
            // fixture tolerance.
            for channel in &mut value {
                *channel = (channel.clamp(0.0, 1.0) * 255.0).round() / 255.0;
            }
            resized.put_pixel(
                destination_x as u32,
                destination_y as u32,
                image::Rgb(value),
            );
        }
    }
    Ok(resized)
}
