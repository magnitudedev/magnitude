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

/// How an image is resized before it is cut into patches. Sides are
/// multiples of the cell side, `patch · merge` pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageResize {
    /// Sides rounded half to even to cell multiples, then scaled into the
    /// pixel bounds (Qwen3-VL `smart_resize`).
    PixelBounds { min_pixels: usize, max_pixels: usize },
    /// Aspect preserved: the largest cell-multiple sides with at most
    /// `max_patches` patches (Gemma 4 `get_aspect_ratio_preserving_size`).
    PatchBudget { max_patches: usize },
    /// The cell grid closest to the image's aspect with at most `max_cells`
    /// cells (Muse Glimmer `smart_resize`).
    CellBudget { max_cells: usize },
}

/// The separable, scale-aware (antialiasing) resampling filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resampling {
    /// PIL bicubic (a = −0.5, support 2).
    Bicubic,
    /// PIL Lanczos (a = 3, support 3).
    Lanczos,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageProcessorConfig {
    pub resize: ImageResize,
    pub resampling: Resampling,
    pub patch: usize,
    /// Side of one merged cell in patches: patch rows are emitted cell by cell.
    pub merge: usize,
    /// Copies of each patch's pixels in one patch row.
    pub frames: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImageProcessorConfig {
    pub fn validate(&self) -> Result<(), String> {
        let factor = self
            .patch
            .checked_mul(self.merge)
            .ok_or("image processor factor overflows")?;
        if factor == 0
            || self.frames == 0
            || match self.resize {
                ImageResize::PixelBounds {
                    min_pixels,
                    max_pixels,
                } => min_pixels == 0 || min_pixels > max_pixels,
                ImageResize::PatchBudget { max_patches } => max_patches == 0,
                ImageResize::CellBudget { max_cells } => max_cells == 0,
            }
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
        let (resize, bounds): (&[u8], [usize; 2]) = match self.resize {
            ImageResize::PixelBounds {
                min_pixels,
                max_pixels,
            } => (b"pixel-bounds:half-even", [min_pixels, max_pixels]),
            ImageResize::PatchBudget { max_patches } => (b"patch-budget", [max_patches, 0]),
            ImageResize::CellBudget { max_cells } => (b"cell-budget:closest-aspect", [max_cells, 0]),
        };
        let resampling: &[u8] = match self.resampling {
            Resampling::Bicubic => b"bicubic:a=-0.5:scale-aware-antialias:rgb8",
            Resampling::Lanczos => b"lanczos:a=3:scale-aware-antialias:rgb8",
        };
        for field in [
            resize,
            resampling,
            b"rescale:1/255",
            &self.patch.to_le_bytes(),
            &self.merge.to_le_bytes(),
            &self.frames.to_le_bytes(),
            &bounds[0].to_le_bytes(),
            &bounds[1].to_le_bytes(),
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

    /// The `(height, width)` an image of the given size is resized to.
    pub fn resized_size(&self, height: usize, width: usize) -> Result<(usize, usize), String> {
        if height == 0 || width == 0 {
            return Err("image dimensions must be nonzero".into());
        }
        let short = height.min(width);
        let long = height.max(width);
        if long > short.saturating_mul(ASPECT_LIMIT) {
            return Err("image aspect ratio exceeds processor limit".into());
        }
        let cell = self.config.patch * self.config.merge;
        match self.config.resize {
            ImageResize::PixelBounds {
                min_pixels,
                max_pixels,
            } => pixel_bounds_size(height, width, cell, min_pixels, max_pixels),
            ImageResize::PatchBudget { max_patches } => patch_budget_size(
                height,
                width,
                self.config.patch,
                self.config.merge,
                max_patches,
            ),
            ImageResize::CellBudget { max_cells } => cell_budget_size(height, width, cell, max_cells),
        }
    }

    pub fn prepare(&self, encoded_images: &[Vec<u8>]) -> Result<PreparedMedia, String> {
        if encoded_images.is_empty() || encoded_images.len() > MAX_IMAGES_PER_REQUEST {
            return Err("image count is outside processor limits".into());
        }
        let width = 3usize
            .checked_mul(self.config.frames)
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
                self.resized_size(image.height() as usize, image.width() as usize)?;
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
            let resized = resize(&source, width_pixels, height, self.config.resampling)?;
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
                            for _frame in 0..self.config.frames {
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

/// Qwen3-VL `smart_resize`: sides rounded half to even to multiples of
/// `cell`, then scaled into `[min_pixels, max_pixels]`.
fn pixel_bounds_size(
    height: usize,
    width: usize,
    cell: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    let round_even = |value: usize| -> usize {
        let quotient = value / cell;
        let remainder = value % cell;
        let rounded = if remainder * 2 < cell {
            quotient
        } else if remainder * 2 > cell || quotient % 2 == 1 {
            quotient + 1
        } else {
            quotient
        };
        rounded.max(1) * cell
    };
    let mut output_height = round_even(height);
    let mut output_width = round_even(width);
    let pixels = output_height
        .checked_mul(output_width)
        .ok_or("resized image extent overflows")?;
    if pixels > max_pixels {
        let beta = ((height as f64 * width as f64) / max_pixels as f64).sqrt();
        output_height = (((height as f64 / beta) / cell as f64).floor() as usize).max(1) * cell;
        output_width = (((width as f64 / beta) / cell as f64).floor() as usize).max(1) * cell;
    } else if pixels < min_pixels {
        let beta = (min_pixels as f64 / (height as f64 * width as f64)).sqrt();
        output_height = ((height as f64 * beta / cell as f64).ceil() as usize).max(1) * cell;
        output_width = ((width as f64 * beta / cell as f64).ceil() as usize).max(1) * cell;
    }
    output_height
        .checked_mul(output_width)
        .filter(|pixels| *pixels <= max_pixels)
        .ok_or_else(|| String::from("resized image extent exceeds processor capacity"))?;
    Ok((output_height, output_width))
}

/// Gemma 4 `get_aspect_ratio_preserving_size`: the largest sides that are
/// multiples of `patch · merge` with at most `max_patches` patches, the
/// image's aspect preserved (in f64, as the released processor computes it).
fn patch_budget_size(
    height: usize,
    width: usize,
    patch: usize,
    merge: usize,
    max_patches: usize,
) -> Result<(usize, usize), String> {
    let target_pixels = max_patches
        .checked_mul(patch * patch)
        .ok_or("patch budget overflows")?;
    let factor = (target_pixels as f64 / (height as f64 * width as f64)).sqrt();
    let side = merge * patch;
    let mut output_height = (factor * height as f64 / side as f64).floor() as usize * side;
    let mut output_width = (factor * width as f64 / side as f64).floor() as usize * side;
    if output_height == 0 && output_width == 0 {
        return Err("image resizes to no patch".into());
    }
    let longest = (max_patches / (merge * merge)) * side;
    if output_height == 0 {
        output_height = side;
        output_width = ((width as f64 / height as f64).floor() as usize * side).min(longest);
    } else if output_width == 0 {
        output_width = side;
        output_height = ((height as f64 / width as f64).floor() as usize * side).min(longest);
    }
    if output_height * output_width > target_pixels {
        return Err("resized image exceeds the patch budget".into());
    }
    Ok((output_height, output_width))
}

/// Muse Glimmer `smart_resize`: the grid of `cell`-pixel cells whose aspect
/// is closest to the image's among the floor/ceil neighbours of the ideal
/// grid, with at most `max_cells` cells. Ties keep the first candidate in
/// the order CPython iterates the released code's `set` of candidates.
fn cell_budget_size(
    height: usize,
    width: usize,
    cell: usize,
    max_cells: usize,
) -> Result<(usize, usize), String> {
    let mut ideal_height = height as f64 / cell as f64;
    let mut ideal_width = width as f64 / cell as f64;
    let ratio = ideal_width / ideal_height;
    if ideal_height * ideal_width > max_cells as f64 {
        ideal_height = (max_cells as f64 / ratio).powf(0.5);
        ideal_width = ideal_height * ratio;
    }
    let rows = [ideal_height.floor() as u64, ideal_height.ceil() as u64];
    let columns = [ideal_width.floor() as u64, ideal_width.ceil() as u64];
    let product = rows
        .iter()
        .flat_map(|&row| columns.iter().map(move |&column| (row, column)));
    let mut candidates = python_set_order(product)
        .into_iter()
        .filter(|&(row, column)| row >= 1 && column >= 1 && row * column <= max_cells as u64)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        candidates.push((
            (ideal_height.round_ties_even() as u64).max(1),
            (ideal_width.round_ties_even() as u64).max(1),
        ));
    }
    let aspect = height as f64 / width as f64;
    let mut best = candidates[0];
    for &candidate in &candidates[1..] {
        let error = |(row, column): (u64, u64)| (row as f64 / column as f64 - aspect).abs();
        if error(candidate) < error(best) {
            best = candidate;
        }
    }
    let side = |cells: u64| {
        usize::try_from(cells)
            .ok()
            .and_then(|cells| cells.checked_mul(cell))
            .ok_or_else(|| String::from("resized image extent overflows"))
    };
    Ok((side(best.0)?, side(best.1)?))
}

/// CPython's `hash((a, b))` of two small non-negative integers
/// (`tupleobject.c`, xxHash lanes).
fn python_pair_hash(a: u64, b: u64) -> u64 {
    const PRIME_1: u64 = 11_400_714_785_074_694_791;
    const PRIME_2: u64 = 14_029_467_366_897_019_727;
    const PRIME_5: u64 = 2_870_177_450_012_600_261;
    let mut accumulator = PRIME_5;
    for lane in [a, b] {
        accumulator = accumulator.wrapping_add(lane.wrapping_mul(PRIME_2));
        accumulator = accumulator.rotate_left(31);
        accumulator = accumulator.wrapping_mul(PRIME_1);
    }
    accumulator = accumulator.wrapping_add(2 ^ (PRIME_5 ^ 3_527_539));
    if accumulator == u64::MAX {
        1_546_275_796
    } else {
        accumulator
    }
}

/// The iteration order of a CPython `set` built by adding at most four
/// pairs in order (`setobject.c`: an 8-slot table, one probe per step,
/// `i = 5i + 1 + perturb` with `perturb >>= 5`).
fn python_set_order(pairs: impl Iterator<Item = (u64, u64)>) -> Vec<(u64, u64)> {
    let mut table = [None::<(u64, u64)>; 8];
    for pair in pairs {
        let hash = python_pair_hash(pair.0, pair.1);
        let mut slot = (hash & 7) as usize;
        let mut perturb = hash;
        loop {
            match table[slot] {
                None => {
                    table[slot] = Some(pair);
                    break;
                }
                Some(existing) if existing == pair => break,
                Some(_) => {
                    perturb >>= 5;
                    slot = ((slot as u64)
                        .wrapping_mul(5)
                        .wrapping_add(1)
                        .wrapping_add(perturb)
                        & 7) as usize;
                }
            }
        }
    }
    table.into_iter().flatten().collect()
}

#[derive(Clone, Debug)]
struct FilterSpan {
    first: usize,
    weights: Vec<f32>,
}

/// PIL's cubic kernel (a = −0.5).
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

/// PIL's `sinc_filter`.
fn sinc(value: f64) -> f64 {
    if value == 0.0 {
        1.0
    } else {
        let value = value * std::f64::consts::PI;
        value.sin() / value
    }
}

/// PIL's truncated-sinc Lanczos kernel (a = 3).
fn lanczos(value: f64) -> f64 {
    if (-3.0..3.0).contains(&value) {
        sinc(value) * sinc(value / 3.0)
    } else {
        0.0
    }
}

impl Resampling {
    fn support(self) -> f64 {
        match self {
            Self::Bicubic => 2.0,
            Self::Lanczos => 3.0,
        }
    }

    fn kernel(self, value: f64) -> f64 {
        match self {
            Self::Bicubic => cubic(value),
            Self::Lanczos => lanczos(value),
        }
    }
}

/// The same center-addressed, scale-aware filter spans PIL's resampler
/// builds. Downscaling widens the kernel's support before normalizing each
/// destination sample, which is the antialiasing step.
fn filter_spans(
    input: usize,
    output: usize,
    resampling: Resampling,
) -> Result<Vec<FilterSpan>, String> {
    if input == 0 || output == 0 {
        return Err("resize dimensions must be nonzero".into());
    }
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = resampling.support() * filter_scale;
    let mut spans = Vec::with_capacity(output);
    for destination in 0..output {
        let center = (destination as f64 + 0.5) * scale;
        let first = (center - support + 0.5).floor().max(0.0) as usize;
        let end = (center + support + 0.5).floor().min(input as f64) as usize;
        if first >= end {
            return Err("resize produced an empty filter span".into());
        }
        let mut weights = (first..end)
            .map(|source| resampling.kernel((source as f64 + 0.5 - center) / filter_scale))
            .collect::<Vec<_>>();
        let sum = weights.iter().sum::<f64>();
        if !sum.is_finite() || sum == 0.0 {
            return Err("resize produced invalid filter weights".into());
        }
        let weights = weights
            .drain(..)
            .map(|weight| (weight / sum) as f32)
            .collect();
        spans.push(FilterSpan { first, weights });
    }
    Ok(spans)
}

fn resize(
    source: &image::Rgb32FImage,
    output_width: usize,
    output_height: usize,
    resampling: Resampling,
) -> Result<image::Rgb32FImage, String> {
    let source_width = source.width() as usize;
    let source_height = source.height() as usize;
    if source_width == output_width && source_height == output_height {
        return Ok(source.clone());
    }
    let horizontal = filter_spans(source_width, output_width, resampling)?;
    let vertical = filter_spans(source_height, output_height, resampling)?;
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
