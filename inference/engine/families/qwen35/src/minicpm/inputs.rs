//! Image overview/detail placement on the MiniCPM-V shared M-RoPE canvas.
use magnitude_artifacts::{
    BoundaryRule, InputLayout, InputSpan, TokenId,
    media::{DType, PreparedMedia},
};
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, ImageCrop, InputPreparationError, MarkerTokens,
    ModelDefinition, ModelInputAdapter, PositionSampling, PreparedModelInput, PreparedVisionInput,
    TokenPlan, VisionResize, VisionSpatialControls, image_crop_identity, image_crops,
};
use std::collections::{BTreeMap, btree_map::Entry};

#[derive(Clone, Debug)]
struct Markers {
    row: TokenId,
    image_start: TokenId,
    image_end: TokenId,
    slice_start: TokenId,
    slice_end: TokenId,
    id_start: TokenId,
    id_end: TokenId,
    newline: TokenId,
    ids: Vec<Vec<TokenId>>,
}
#[derive(Clone, Debug)]
struct CanvasInput {
    family: String,
    markers: Markers,
}

pub fn adapter(
    definition: &ModelDefinition,
    tokens: &dyn MarkerTokens,
) -> Result<FamilyInputAdapter, FamilyError> {
    let digits = (0..10)
        .map(|i| tokens.marker(&i.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let markers = Markers {
        row: tokens.marker("<|image_pad|>")?,
        image_start: tokens.marker("<image>")?,
        image_end: tokens.marker("</image>")?,
        slice_start: tokens.marker("<slice>")?,
        slice_end: tokens.marker("</slice>")?,
        id_start: tokens.marker("<image_id>")?,
        id_end: tokens.marker("</image_id>")?,
        newline: tokens.marker("\n")?,
        ids: (0..magnitude_artifacts::MAX_IMAGES_PER_REQUEST)
            .map(|i| {
                i.to_string()
                    .bytes()
                    .map(|digit| digits[(digit - b'0') as usize])
                    .collect()
            })
            .collect(),
    };
    Ok(Box::new(CanvasInput {
        family: definition.family.0.clone(),
        markers,
    }))
}

/// Patch rows are emitted in 2×2 cell order. Each fractional coordinate
/// selects one learned position bucket; no interpolation or rotary is used.
pub fn spatial_controls(
    grid: [usize; 3],
    side: usize,
) -> Result<VisionSpatialControls, InputPreparationError> {
    let [t, h, w] = grid;
    if t != 1
        || h == 0
        || w == 0
        || h % 4 != 0
        || w % 4 != 0
        || side == 0
        || side.checked_mul(side).is_none()
    {
        return Err(InputPreparationError::VisionGeometry);
    }
    let rows = h
        .checked_mul(w)
        .filter(|n| *n <= i32::MAX as usize)
        .ok_or(InputPreparationError::VisionGeometry)?;
    let boundaries = (1..side)
        .map(|k| (k as f64 / side as f64) as f32)
        .collect::<Vec<_>>();
    let buckets = |extent: usize| {
        (0..extent)
            .map(|p| {
                let fraction = p as f32 / extent as f32;
                boundaries.partition_point(|boundary| *boundary <= fraction)
            })
            .collect::<Vec<_>>()
    };
    let (vertical, horizontal) = (buckets(h), buckets(w));
    let mut indices = Vec::with_capacity(rows);
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let index = vertical[2 * y + dy] * side + horizontal[2 * x + dx];
                    indices.push(
                        i32::try_from(index).map_err(|_| InputPreparationError::VisionGeometry)?,
                    );
                }
            }
        }
    }
    VisionSpatialControls::new(
        (0..rows).collect(),
        vec![[0, 0]; rows],
        [indices, vec![0; rows], vec![0; rows], vec![0; rows]],
        [
            vec![1.0; rows],
            vec![0.0; rows],
            vec![0.0; rows],
            vec![0.0; rows],
        ],
        Vec::new(),
    )
}

fn coordinate(t: usize, h: usize, w: usize) -> Result<[i32; 3], InputPreparationError> {
    Ok([
        i32::try_from(t).map_err(|_| InputPreparationError::InputAlignment)?,
        i32::try_from(h).map_err(|_| InputPreparationError::InputAlignment)?,
        i32::try_from(w).map_err(|_| InputPreparationError::InputAlignment)?,
    ])
}

struct Placement<'a> {
    definition: &'a ModelDefinition,
    processor: &'a str,
    tokens: Vec<TokenId>,
    coordinates: Vec<[i32; 3]>,
    spans: Vec<InputSpan>,
    images: BTreeMap<String, PreparedVisionInput>,
    side: usize,
    markers: &'a Markers,
}
impl Placement<'_> {
    fn token(&mut self, token: TokenId, coordinates: [i32; 3]) {
        self.tokens.push(token);
        self.coordinates.push(coordinates);
    }
    fn crop(
        &mut self,
        crop: ImageCrop,
        origin: usize,
        offsets: [usize; 2],
        canvas: [usize; 2],
    ) -> Result<(), InputPreparationError> {
        let [_, h, w] = crop.grid;
        let (height, width) = (h / 2, w / 2);
        let start = self.tokens.len();
        let axis = |i: usize, n: usize, extent: usize| {
            if n == 1 {
                0
            } else {
                ((i as f64 * (extent - 1) as f64 / (n - 1) as f64) as f32).round_ties_even()
                    as usize
            }
        };
        for y in 0..height {
            for x in 0..width {
                self.token(
                    self.markers.row,
                    coordinate(
                        origin,
                        origin + offsets[0] + axis(y, height, canvas[0]),
                        origin + offsets[1] + axis(x, width, canvas[1]),
                    )?,
                );
            }
        }
        let identity = image_crop_identity(self.processor, &crop);
        self.spans.push(InputSpan {
            start,
            end: self.tokens.len(),
            identity: identity.clone(),
            boundaries: BoundaryRule::Causal,
            language_history: false,
        });
        if let Entry::Vacant(entry) = self.images.entry(identity) {
            entry.insert(PreparedVisionInput::new(
                self.definition,
                crop.grid,
                crop.pixels,
                spatial_controls(crop.grid, self.side)?,
            )?);
        }
        Ok(())
    }
}

impl ModelInputAdapter for CanvasInput {
    fn prepare(
        &self,
        definition: &ModelDefinition,
        plan: TokenPlan,
        media: &[PreparedMedia],
    ) -> Result<PreparedModelInput, InputPreparationError> {
        if definition.family.0 != self.family || !plan.layout().spans().is_empty() {
            return Err(InputPreparationError::InputAlignment);
        }
        let vision = definition
            .vision
            .as_ref()
            .ok_or(InputPreparationError::UnsupportedMedia)?;
        let VisionResize::Slices { max_slices, .. } = vision.preprocessing.resize else {
            return Err(InputPreparationError::VisionGeometry);
        };
        let PositionSampling::Buckets { side } = vision.stem.positions().sampling else {
            return Err(InputPreparationError::VisionGeometry);
        };
        let (processor, crops, layouts) = match media {
            [] => (String::new(), Vec::new(), Vec::new()),
            [media] => {
                let grid = media
                    .tensors()
                    .iter()
                    .find(|t| t.name() == "image_slice_grid")
                    .ok_or(InputPreparationError::VisionGeometry)?;
                if grid.dtype() != DType::I64
                    || grid.shape().len() != 2
                    || grid.shape()[1] != 2
                    || grid.shape()[0] > self.markers.ids.len()
                {
                    return Err(InputPreparationError::VisionGeometry);
                }
                let layouts = grid
                    .data()
                    .chunks_exact(16)
                    .map(|b| {
                        let y = usize::try_from(i64::from_le_bytes(b[..8].try_into().unwrap()))
                            .map_err(|_| InputPreparationError::VisionGeometry)?;
                        let x = usize::try_from(i64::from_le_bytes(b[8..].try_into().unwrap()))
                            .map_err(|_| InputPreparationError::VisionGeometry)?;
                        if (y == 0) != (x == 0)
                            || y.checked_mul(x).is_none_or(|n| n > max_slices as usize)
                        {
                            return Err(InputPreparationError::VisionGeometry);
                        }
                        Ok([y, x])
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let expected = layouts.iter().try_fold(0usize, |n, [y, x]| {
                    n.checked_add(1 + y * x)
                        .ok_or(InputPreparationError::VisionGeometry)
                })?;
                let grids = media
                    .tensors()
                    .iter()
                    .find(|t| t.name() == "image_grid_thw")
                    .ok_or(InputPreparationError::VisionGeometry)?;
                if grids.shape().first() != Some(&expected) {
                    return Err(InputPreparationError::VisionGeometry);
                }
                let (processor, crops) = image_crops(definition, media)?;
                (processor, crops, layouts)
            }
            _ => return Err(InputPreparationError::UnsupportedMedia),
        };
        let mut crops = crops.into_iter().peekable();
        let mut layouts = layouts.into_iter().enumerate();
        let mut p = Placement {
            definition,
            processor: &processor,
            tokens: Vec::new(),
            coordinates: Vec::new(),
            spans: Vec::new(),
            images: BTreeMap::new(),
            side: side as usize,
            markers: &self.markers,
        };
        let mut position = 0usize;
        for token in plan.tokens() {
            if *token != self.markers.row {
                p.token(*token, coordinate(position, position, position)?);
                position += 1;
                continue;
            }
            let (id, [rows, cols]) = layouts
                .next()
                .ok_or(InputPreparationError::InputAlignment)?;
            for token in std::iter::once(self.markers.id_start)
                .chain(self.markers.ids[id].iter().copied())
                .chain(std::iter::once(self.markers.id_end))
            {
                p.token(token, coordinate(position, position, position)?);
                position += 1;
            }
            let overview = crops.next().ok_or(InputPreparationError::VisionGeometry)?;
            let detail = crops.peek();
            let canvas = if rows == 0 {
                [overview.grid[1] / 2, overview.grid[2] / 2]
            } else {
                let c = detail.ok_or(InputPreparationError::VisionGeometry)?;
                [rows * c.grid[1] / 2, cols * c.grid[2] / 2]
            };
            let origin = position;
            p.token(
                self.markers.image_start,
                coordinate(origin, origin.saturating_sub(1), origin.saturating_sub(1))?,
            );
            p.crop(overview, origin, [0, 0], canvas)?;
            p.token(
                self.markers.image_end,
                coordinate(origin, origin + canvas[0], origin + canvas[1])?,
            );
            let slice_size = if rows == 0 {
                [0, 0]
            } else {
                [canvas[0] / rows, canvas[1] / cols]
            };
            for y in 0..rows {
                for x in 0..cols {
                    let crop = crops.next().ok_or(InputPreparationError::VisionGeometry)?;
                    if [crop.grid[1] / 2, crop.grid[2] / 2] != slice_size {
                        return Err(InputPreparationError::VisionGeometry);
                    }
                    let offsets = [y * slice_size[0], x * slice_size[1]];
                    p.token(
                        self.markers.slice_start,
                        coordinate(origin, origin + offsets[0], origin + offsets[1])?,
                    );
                    p.crop(crop, origin, offsets, slice_size)?;
                    p.token(
                        self.markers.slice_end,
                        coordinate(
                            origin,
                            origin + offsets[0] + slice_size[0] - 1,
                            origin + offsets[1] + slice_size[1] - 1,
                        )?,
                    );
                }
                if y + 1 < rows {
                    p.token(
                        self.markers.newline,
                        coordinate(
                            origin,
                            origin + (y + 1) * slice_size[0] - 1,
                            origin + canvas[1],
                        )?,
                    );
                }
            }
            position = origin + canvas[0].max(canvas[1]) + 1;
        }
        if layouts.next().is_some() || crops.next().is_some() || p.tokens.len() > i32::MAX as usize
        {
            return Err(InputPreparationError::InputAlignment);
        }
        let layout = InputLayout::new(p.tokens.len(), p.spans)
            .map_err(|_| InputPreparationError::InputAlignment)?;
        let plan = TokenPlan::new(p.tokens, layout)?;
        if p.images.is_empty() {
            return PreparedModelInput::from_text_coordinates(plan, p.coordinates);
        }
        PreparedModelInput::new(definition, plan, p.coordinates, position, p.images)
    }
}
