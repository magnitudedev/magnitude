//! Muse Glimmer input preparation. Every row, text or media, takes its
//! absolute position; an image expands between its `<|image_start|>` and
//! `<|image_end|>` markers to one `<|patch|>` row per 2 × 2 cell.

use magnitude_artifacts::BoundaryRule;
use magnitude_family_contracts::{
    ImageMarkers, InputPreparationError, ModelDefinition, PositionSampling, SequentialImageInput,
    VisionSpatialControls,
};

/// The text one image renders as: the patch row between the image markers,
/// which input preparation expands to the image's cells.
pub const IMAGE_PLACEHOLDER: &str = "<|image_start|><|patch|><|image_end|>";

pub fn input_adapter(markers: Option<ImageMarkers>) -> SequentialImageInput {
    SequentialImageInput::new("muse-glimmer", markers, BoundaryRule::Causal, spatial_controls)
}

/// One axis of HF's bilinear `grid_sample` of the `side`-row position table
/// at patch centres (`align_corners=False`, zero padding), in F32 as HF
/// computes it: the two taps and their weights.
fn taps(index: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let source = (index as f32 + 0.5) * side as f32 / size as f32 - 0.5;
    let floor = source.floor();
    [0.0f32, 1.0].map(|offset| {
        let tap = floor + offset;
        let weight = (1.0 - (source - floor - offset).abs()).max(0.0);
        let inside = tap >= 0.0 && tap <= (side - 1) as f32;
        (
            tap.clamp(0.0, (side - 1) as f32) as usize,
            if inside { weight } else { 0.0 },
        )
    })
}

/// The tower attends within windows of `window` × `window` patches, so its
/// rows are the patches window by window (windows in raster order, each
/// window's patches in raster order), HF's `get_vision_window_index`. The
/// processor emits the patches cell by cell (2 × 2 cells in raster order);
/// `patch_order` maps each tower row to its processor row. The rotary
/// coordinates are `[column + 1, row + 1]`; the position value is the
/// bilinear sample of the square table at the patch centre.
fn spatial_controls(
    definition: &ModelDefinition,
    grid: [usize; 3],
) -> Result<VisionSpatialControls, InputPreparationError> {
    let vision = definition
        .vision
        .as_ref()
        .ok_or(InputPreparationError::UnsupportedMedia)?;
    let PositionSampling::PixelCenters { side } = vision.stem.positions().sampling else {
        return Err(InputPreparationError::VisionGeometry);
    };
    let (side, window, merge) = (
        side as usize,
        vision.window.ok_or(InputPreparationError::VisionGeometry)? as usize,
        vision.preprocessing.merge as usize,
    );
    let [t, h, w] = grid;
    if t != 1 || side == 0 || window == 0 || merge == 0 || h % merge != 0 || w % merge != 0 {
        return Err(InputPreparationError::VisionGeometry);
    }
    let rows = h * w;
    let mut patch_order = Vec::with_capacity(rows);
    let mut coordinates = Vec::with_capacity(rows);
    let mut indices: [Vec<i32>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    let mut coefficients: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(rows));
    let mut window_ranges = Vec::with_capacity(rows);
    for window_row in 0..h.div_ceil(window) {
        for window_col in 0..w.div_ceil(window) {
            let (rows_in, cols_in) = (
                window_row * window..((window_row + 1) * window).min(h),
                window_col * window..((window_col + 1) * window).min(w),
            );
            let start = patch_order.len();
            let end = start + rows_in.len() * cols_in.len();
            for row in rows_in {
                for col in cols_in.clone() {
                    let cell = (row / merge) * (w / merge) + col / merge;
                    patch_order.push(cell * merge * merge + (row % merge) * merge + col % merge);
                    coordinates.push([col as i32 + 1, row as i32 + 1]);
                    let (vertical, horizontal) = (taps(row, h, side), taps(col, w, side));
                    for (plane, ((y, wy), (x, wx))) in vertical
                        .iter()
                        .flat_map(|y| horizontal.iter().map(move |x| (*y, *x)))
                        .enumerate()
                    {
                        indices[plane].push((y * side + x) as i32);
                        coefficients[plane].push(wy * wx);
                    }
                    window_ranges.push([start as i32, end as i32]);
                }
            }
        }
    }
    VisionSpatialControls::new(patch_order, coordinates, indices, coefficients, window_ranges)
}
