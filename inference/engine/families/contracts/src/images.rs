//! Image input of a family whose rows, text and media alike, take their
//! absolute position on one rotary axis: each image's placeholder (its
//! start, row and end markers) expands to one row per merged cell between
//! the start and end markers, and the family supplies the image's spatial
//! controls.

use crate::{
    InputPreparationError, ModelDefinition, ModelInputAdapter, PreparedModelInput,
    PreparedVisionInput, TokenPlan, VisionSpatialControls,
};
use magnitude_artifacts::{
    media::{DType, PreparedMedia, PreparedTensor},
    BoundaryRule, ImageProcessor, InputLayout, InputSpan, TokenId,
};
use sha2::{Digest, Sha256};
use std::collections::{btree_map::Entry, BTreeMap};

/// The three tokens an image renders as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageMarkers {
    start: TokenId,
    row: TokenId,
    end: TokenId,
}

impl ImageMarkers {
    pub fn new(start: TokenId, row: TokenId, end: TokenId) -> Result<Self, InputPreparationError> {
        if [start, row, end].iter().any(|token| token.0 > i32::MAX as u32)
            || start == row
            || start == end
            || row == end
        {
            return Err(InputPreparationError::TokenDomain);
        }
        Ok(Self { start, row, end })
    }

    fn contains(&self, token: TokenId) -> bool {
        [self.start, self.row, self.end].contains(&token)
    }
}

/// The spatial controls of one image of patch grid `[t, h, w]`.
pub type SpatialControls =
    fn(&ModelDefinition, [usize; 3]) -> Result<VisionSpatialControls, InputPreparationError>;

/// Rows at their absolute positions; images expanded between their markers.
#[derive(Clone, Debug)]
pub struct SequentialImageInput {
    family: &'static str,
    markers: Option<ImageMarkers>,
    /// How a span of image rows may be split across launches: `Causal` when
    /// media rows attend causally, `Indivisible` when they see each other.
    boundaries: BoundaryRule,
    spatial: SpatialControls,
}

impl SequentialImageInput {
    pub fn new(
        family: &'static str,
        markers: Option<ImageMarkers>,
        boundaries: BoundaryRule,
        spatial: SpatialControls,
    ) -> Self {
        Self {
            family,
            markers,
            boundaries,
            spatial,
        }
    }
}

/// One image's patch grid and pixel rows.
struct Image {
    grid: [usize; 3],
    pixels: PreparedTensor,
}

/// The images of one prepared media item, in processor order.
fn images(definition: &ModelDefinition, media: &PreparedMedia) -> Result<(String, Vec<Image>), InputPreparationError> {
    let vision = definition
        .vision
        .as_ref()
        .ok_or(InputPreparationError::UnsupportedMedia)?;
    let processor = ImageProcessor::new(
        vision
            .image_processor_config()
            .map_err(|_| InputPreparationError::VisionGeometry)?,
    )
    .map_err(|_| InputPreparationError::VisionGeometry)?;
    if media.processor() != processor.identity() || media.tensors().len() != 2 {
        return Err(InputPreparationError::UnsupportedMedia);
    }
    let width = vision
        .patch_row_width()
        .ok()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(InputPreparationError::VisionGeometry)?;
    let merge = usize::try_from(vision.preprocessing.merge)
        .ok()
        .filter(|value| *value > 0)
        .ok_or(InputPreparationError::VisionGeometry)?;
    let tensor = |name: &str| {
        media
            .tensors()
            .iter()
            .find(|tensor| tensor.name() == name)
            .ok_or(InputPreparationError::VisionGeometry)
    };
    let (pixels, grids) = (tensor("pixel_values")?, tensor("image_grid_thw")?);
    if pixels.dtype() != DType::F32
        || pixels.shape().len() != 2
        || pixels.shape()[1] != width
        || grids.dtype() != DType::I64
        || grids.shape().len() != 2
        || grids.shape()[1] != 3
        || pixels
            .data()
            .chunks_exact(4)
            .any(|bytes| !f32::from_le_bytes(bytes.try_into().expect("f32 chunk")).is_finite())
    {
        return Err(InputPreparationError::VisionGeometry);
    }
    let mut images = Vec::with_capacity(grids.shape()[0]);
    let mut first = 0usize;
    for row in grids.data().chunks_exact(24) {
        let mut grid = [0usize; 3];
        for (index, bytes) in row.chunks_exact(8).enumerate() {
            grid[index] = usize::try_from(i64::from_le_bytes(bytes.try_into().expect("i64 chunk")))
                .map_err(|_| InputPreparationError::VisionGeometry)?;
        }
        let [t, h, w] = grid;
        if t != 1 || h < merge || w < merge || h % merge != 0 || w % merge != 0 {
            return Err(InputPreparationError::VisionGeometry);
        }
        let rows = h.checked_mul(w).ok_or(InputPreparationError::VisionGeometry)?;
        let bytes = |row: usize| {
            row.checked_mul(width)
                .and_then(|value| value.checked_mul(4))
                .ok_or(InputPreparationError::VisionGeometry)
        };
        let end = first.checked_add(rows).ok_or(InputPreparationError::VisionGeometry)?;
        let values = pixels
            .data()
            .get(bytes(first)?..bytes(end)?)
            .ok_or(InputPreparationError::VisionGeometry)?
            .to_vec();
        images.push(Image {
            grid,
            pixels: PreparedTensor::new("pixel_values".into(), DType::F32, vec![rows, width], values)
                .map_err(|_| InputPreparationError::VisionGeometry)?,
        });
        first = end;
    }
    if first != pixels.shape()[0] {
        return Err(InputPreparationError::VisionGeometry);
    }
    Ok((processor.identity().to_owned(), images))
}

/// The content identity of one image: its processor, grid and pixels.
fn identity(processor: &str, image: &Image) -> String {
    let mut digest = Sha256::new();
    digest.update(processor.as_bytes());
    for value in image.grid {
        digest.update((value as i64).to_le_bytes());
    }
    digest.update(image.pixels.data());
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl ModelInputAdapter for SequentialImageInput {
    fn prepare(
        &self,
        definition: &ModelDefinition,
        token_plan: TokenPlan,
        media: &[PreparedMedia],
    ) -> Result<PreparedModelInput, InputPreparationError> {
        if definition.family.0 != self.family || !token_plan.layout().spans().is_empty() {
            return Err(InputPreparationError::InputAlignment);
        }
        let (tokens, _) = token_plan.into_parts();
        let (processor, images) = match media {
            [] => (String::new(), Vec::new()),
            [prepared] => images(definition, prepared)?,
            _ => return Err(InputPreparationError::UnsupportedMedia),
        };
        let markers = match (self.markers, images.is_empty()) {
            (Some(markers), _) => Some(markers),
            (None, true) => None,
            (None, false) => return Err(InputPreparationError::UnsupportedMedia),
        };
        let cell_rows = definition
            .vision
            .as_ref()
            .map_or(1, |vision| vision.cell_rows() as usize);
        let mut expanded = Vec::with_capacity(tokens.len());
        let mut spans = Vec::with_capacity(images.len());
        let mut vision_inputs = BTreeMap::new();
        let mut images = images.into_iter();
        let mut source = 0usize;
        while source < tokens.len() {
            let token = tokens[source];
            match markers {
                Some(markers) if token == markers.start => {
                    let image = images.next().ok_or(InputPreparationError::InputAlignment)?;
                    if tokens.get(source + 1) != Some(&markers.row)
                        || tokens.get(source + 2) != Some(&markers.end)
                    {
                        return Err(InputPreparationError::InputAlignment);
                    }
                    let [t, h, w] = image.grid;
                    let rows = t * h * w / cell_rows;
                    expanded.push(markers.start);
                    let start = expanded.len();
                    expanded.extend(std::iter::repeat_n(markers.row, rows));
                    expanded.push(markers.end);
                    let identity = identity(&processor, &image);
                    spans.push(InputSpan {
                        start,
                        end: start + rows,
                        identity: identity.clone(),
                        boundaries: self.boundaries,
                        language_history: false,
                    });
                    if let Entry::Vacant(entry) = vision_inputs.entry(identity) {
                        entry.insert(PreparedVisionInput::new(
                            definition,
                            image.grid,
                            image.pixels,
                            (self.spatial)(definition, image.grid)?,
                        )?);
                    }
                    source += 3;
                }
                Some(markers) if markers.contains(token) => {
                    return Err(InputPreparationError::InputAlignment);
                }
                _ => {
                    expanded.push(token);
                    source += 1;
                }
            }
        }
        if images.next().is_some() || expanded.len() > i32::MAX as usize {
            return Err(InputPreparationError::InputAlignment);
        }
        let continuation = expanded.len();
        let coordinates = (0..continuation).map(|position| [position as i32; 3]).collect();
        let layout = InputLayout::new(continuation, spans)
            .map_err(|_| InputPreparationError::InputAlignment)?;
        let plan = TokenPlan::new(expanded, layout)?;
        if vision_inputs.is_empty() {
            return PreparedModelInput::from_text_coordinates(plan, coordinates);
        }
        PreparedModelInput::new(definition, plan, coordinates, continuation, vision_inputs)
    }
}
